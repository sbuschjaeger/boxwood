// Import of a CatBoost model from its JSON export: numeric borders, one-hot
// values, CTR projections and tables (each table entry's code is computed here
// once), and the trees.

#include "boxwood/ensemble.hpp"

#include <algorithm>
#include <charconv>
#include <map>
#include <numeric>
#include <string>
#include <utility>

#include "json.hpp"

namespace boxwood {

namespace {

/// (num, den) of CatBoost's CTR calc for the statistics of one table entry.
/// Counts are integers, so their sums in double are exact; like CatBoost, they
/// become float32 only in the formula.
std::pair<double, double> counts_to_calc(const std::string& kind, const std::vector<double>& stats,
                                       uint32_t tbi, double counter_den) {
  if (kind == "BinarizedTargetMeanValue" || kind == "FloatTargetMeanValue") {
    return {stats[0], stats[1]};  // (sum of targets, count)
  }
  if (kind == "Counter" || kind == "FeatureFreq") return {stats[0], counter_den};
  const double total = std::accumulate(stats.begin(), stats.end(), 0.0);
  if (kind == "Buckets") return {stats[tbi], total};
  // Borders: the share of observations whose target class is above the target border.
  return {std::accumulate(stats.begin() + tbi + 1, stats.end(), 0.0), total};
}

/// One entry of a CTR table: a key and its statistics fields.
struct CtrEntry {
  uint64_t key;
  std::vector<double> stats;
};

/// The entries of a CTR table, sorted by key. The JSON lists every bucket of
/// CatBoost's hash table; empty buckets have the key 2^64-1 and the statistics
/// of bucket 0, which belong to another key, so they are skipped.
std::vector<CtrEntry> parse_ctr_table(const Json& table) {
  const size_t stride = static_cast<size_t>(table.at("hash_stride").as_double());
  BW_CHECK(stride >= 2, "CTR table hash_stride must be >= 2");
  const Json& raw = table.at("hash_map");
  BW_CHECK(raw.size() % stride == 0, "CTR table hash_map size is not a multiple of hash_stride");
  std::vector<CtrEntry> entries;
  for (size_t i = 0; i < raw.size(); i += stride) {
    const std::string& ks = raw[i].as_string();
    uint64_t key = 0;
    const std::from_chars_result res = std::from_chars(ks.data(), ks.data() + ks.size(), key);
    BW_CHECK(res.ec == std::errc{} && res.ptr == ks.data() + ks.size(), "bad CTR table key: " + ks);
    if (key == ~uint64_t{0}) continue;  // empty bucket
    CtrEntry entry;
    entry.key = key;
    entry.stats.reserve(stride - 1);
    for (size_t j = i + 1; j < i + stride; ++j) entry.stats.push_back(raw[j].as_double());
    entries.push_back(std::move(entry));
  }
  std::sort(entries.begin(), entries.end(),
            [](const CtrEntry& a, const CtrEntry& b) { return a.key < b.key; });
  return entries;
}

/// One element of a CTR projection.
struct ProjElem {
  uint8_t kind = 0;
  uint32_t column = 0;
  float border = 0.0f;
  uint8_t nan_high = 0;
  int32_t value = 0;
};

/// A canonical byte string of a projection's elements, for sharing projections.
std::string projection_key(const std::vector<ProjElem>& elems) {
  std::string key;
  for (const ProjElem& e : elems) {
    key.push_back(static_cast<char>(e.kind));
    key.append(reinterpret_cast<const char*>(&e.column), sizeof(e.column));
    key.append(reinterpret_cast<const char*>(&e.border), sizeof(e.border));
    key.push_back(static_cast<char>(e.nan_high));
    key.append(reinterpret_cast<const char*>(&e.value), sizeof(e.value));
  }
  return key;
}

/// The projection elements of a CTR: the category-hash elements first, in
/// element order, then the other elements in element order (as CatBoost hashes
/// them).
std::vector<ProjElem> projection_elems(const Json& ctr, uint32_t n_numeric, uint32_t n_categorical,
                                      const std::vector<uint8_t>& nan_of) {
  const Json& elements = ctr.at("elements");
  std::vector<ProjElem> elems;
  for (int pass = 0; pass < 2; ++pass) {
    for (const Json& el : elements.array) {
      const std::string& ce = el.at("combination_element").as_string();
      const bool cat_value = ce == "cat_feature_value";
      if ((pass == 0) != cat_value) continue;
      ProjElem e;
      if (cat_value) {
        e.kind = 0;
        e.column = static_cast<uint32_t>(el.at("cat_feature_index").as_double());
        BW_CHECK(e.column < n_categorical, "CTR element references an unknown categorical feature");
      } else if (ce == "float_feature") {
        e.kind = 1;
        e.column = static_cast<uint32_t>(el.at("float_feature_index").as_double());
        BW_CHECK(e.column < n_numeric, "CTR element references an unknown float feature");
        e.border = static_cast<float>(el.at("border").as_double());
        e.nan_high = nan_of[e.column];
      } else {
        BW_CHECK(ce == "cat_feature_exact_value", "unknown combination_element: " + ce);
        e.kind = 2;
        e.column = static_cast<uint32_t>(el.at("cat_feature_index").as_double());
        BW_CHECK(e.column < n_categorical, "CTR element references an unknown categorical feature");
        e.value = static_cast<int32_t>(el.at("value").as_double());
      }
      elems.push_back(e);
    }
  }
  return elems;
}

}  // namespace

ObliviousTreeEnsemble ObliviousTreeEnsemble::from_catboost_json(std::string_view json) {
  const Json root = parse_json(json);
  ObliviousTreeEnsemble e;

  const std::string& loss = root.at("model_info").at("params").at("loss_function").at("type").as_string();
  if (loss == "RMSE") {
    e.task_ = Task::Regression;
  } else if (loss == "Logloss" || loss == "CrossEntropy") {
    e.task_ = Task::Binary;
  } else if (loss == "MultiClass") {
    e.task_ = Task::Multiclass;
  } else {
    BW_CHECK(false, "unsupported loss_function: " + loss);
  }

  const Json& sb = root.at("scale_and_bias");
  BW_CHECK(sb.size() == 2, "scale_and_bias must hold [scale, bias]");
  const double scale = sb[0].as_double();
  const Json& bias_json = sb[1];
  e.K_ = static_cast<uint32_t>(bias_json.size());
  BW_CHECK(e.K_ >= 1, "scale_and_bias[1] must not be empty");
  BW_CHECK(e.task_ == Task::Multiclass || e.K_ == 1, "K must be 1 for regression and binary");
  e.bias_.reserve(e.K_);
  for (uint32_t k = 0; k < e.K_; ++k) e.bias_.push_back(bias_json[k].as_double());

  const Json& fi = root.at("features_info");

  // Numeric features: one numeric split feature per float feature with borders.
  const Json* ff = fi.find("float_features");  // missing when the model has no float features
  e.n_numeric_ = ff != nullptr ? static_cast<uint32_t>(ff->size()) : 0;
  std::vector<uint8_t> nan_of(e.n_numeric_, 0);  // per float feature, for CTR elements
  std::vector<int32_t> split_of(e.n_numeric_, -1);
  for (uint32_t i = 0; i < e.n_numeric_; ++i) {
    const Json& f = (*ff)[i];
    e.numeric_flat_index_.push_back(static_cast<uint32_t>(f.at("flat_feature_index").as_double()));
    nan_of[i] = f.at("nan_value_treatment").as_string() == "AsTrue" ? 1 : 0;
    const Json* b = f.find("borders");
    if (b == nullptr || b->is_null() || b->size() == 0) continue;
    const size_t cnt = b->size();
    BW_CHECK(cnt <= 255, "float feature has more than 255 borders");
    split_of[i] = static_cast<int32_t>(e.numeric_column_.size());
    e.numeric_column_.push_back(i);
    e.border_count_.push_back(static_cast<uint32_t>(cnt));
    e.nan_high_.push_back(nan_of[i]);
    for (size_t j = 0; j < cnt; ++j) e.borders_.push_back(static_cast<float>((*b)[j].as_double()));
  }

  // Categorical features: one one-hot split feature per feature that has values.
  const Json* cats = fi.find("categorical_features");
  const size_t C = cats != nullptr ? cats->size() : 0;
  e.n_categorical_ = static_cast<uint32_t>(C);
  std::vector<int32_t> onehot_of(C, -1);
  for (size_t c = 0; c < C; ++c) {
    const Json& cf = (*cats)[c];
    e.categorical_flat_index_.push_back(
        static_cast<uint32_t>(cf.at("flat_feature_index").as_double()));
    const Json* values = cf.find("values");
    if (values == nullptr || values->is_null() || values->size() == 0) continue;
    BW_CHECK(values->size() <= 255, "one-hot feature has more than 255 values");
    onehot_of[c] = static_cast<int32_t>(e.onehot_column_.size());
    e.onehot_column_.push_back(static_cast<uint32_t>(c));
    e.onehot_count_.push_back(static_cast<uint8_t>(values->size()));
    for (const Json& hv : values->array) {
      e.onehot_values_.push_back(static_cast<int32_t>(hv.as_double()));
    }
  }

  // CTRs: projections, tables and the code of every table entry for this CTR.
  std::vector<std::vector<float>> ctr_borders;  // kept for the split_index map
  const Json* ctrs = fi.find("ctrs");
  if (ctrs != nullptr && ctrs->size() > 0) {
    const Json& ctr_data = root.at("ctr_data");
    std::map<std::string, uint32_t> proj_of;   // equal elements share one projection
    std::map<std::string, uint32_t> table_of;  // equal identifiers share one table
    std::vector<std::vector<CtrEntry>> tables;  // sorted entries, for the codes of later CTRs
    std::vector<double> counter_den_of;         // counter_denominator of each table
    for (const Json& ctr : ctrs->array) {
      const std::vector<ProjElem> elems = projection_elems(ctr, e.n_numeric_, e.n_categorical_, nan_of);
      const std::string pkey = projection_key(elems);
      uint32_t p;
      if (const auto it = proj_of.find(pkey); it != proj_of.end()) {
        p = it->second;
      } else {
        p = static_cast<uint32_t>(e.proj_size_.size());
        proj_of.emplace(pkey, p);
        e.proj_size_.push_back(static_cast<uint32_t>(elems.size()));
        for (const ProjElem& el : elems) {
          e.elem_kind_.push_back(el.kind);
          e.elem_column_.push_back(el.column);
          e.elem_border_.push_back(el.border);
          e.elem_nan_high_.push_back(el.nan_high);
          e.elem_value_.push_back(el.value);
        }
      }

      const std::string& ident = ctr.at("identifier").as_string();
      uint32_t q;
      if (const auto it = table_of.find(ident); it != table_of.end()) {
        q = it->second;
      } else {
        q = static_cast<uint32_t>(e.table_projection_.size());
        table_of.emplace(ident, q);
        const Json& table_json = ctr_data.at(ident);
        tables.push_back(parse_ctr_table(table_json));
        counter_den_of.push_back(table_json.at("counter_denominator").as_double());
        e.table_projection_.push_back(p);
        e.table_size_.push_back(static_cast<uint32_t>(tables.back().size()));
        for (const CtrEntry& entry : tables.back()) e.table_keys_.push_back(entry.key);
      }

      const std::string& kind = ctr.at("ctr_type").as_string();
      const uint32_t tbi = static_cast<uint32_t>(ctr.at("target_border_idx").as_double());
      const float prior_num = static_cast<float>(ctr.at("prior_numerator").as_double());
      const float prior_den = static_cast<float>(ctr.at("prior_denomerator").as_double());
      const float shift = static_cast<float>(ctr.at("shift").as_double());
      const float scale_f = static_cast<float>(ctr.at("scale").as_double());
      const Json* bj = ctr.find("borders");
      std::vector<float> borders;
      if (bj != nullptr && !bj->is_null()) {
        for (const Json& b : bj->array) borders.push_back(static_cast<float>(b.as_double()));
      }
      BW_CHECK(borders.size() <= 255, "CTR has more than 255 borders");

      // The CTR value of an entry in float32; its code is the number of CTR
      // borders below it.
      const auto code_of = [&](double num, double den) -> uint8_t {
        const float value =
            ((static_cast<float>(num) + prior_num) / (static_cast<float>(den) + prior_den) + shift) *
            scale_f;
        return static_cast<uint8_t>(std::lower_bound(borders.begin(), borders.end(), value) -
                                   borders.begin());
      };
      for (const CtrEntry& entry : tables[q]) {
        const auto [num, den] = counts_to_calc(kind, entry.stats, tbi, counter_den_of[q]);
        e.ctr_codes_.push_back(code_of(num, den));
      }
      // A key without a bucket gets calc(0, 0); counters divide by the table's
      // denominator also then (CatBoost's predictor, not the format="cpp" export).
      const bool counter = kind == "Counter" || kind == "FeatureFreq";
      e.ctr_default_code_.push_back(
          code_of(0.0, counter ? counter_den_of[q] : 0.0));
      e.ctr_table_.push_back(q);
      ctr_borders.push_back(std::move(borders));
    }
  }

  e.rebuild_derived();  // the tree loop looks features up through the prefix sums

  // An OnlineCtr split names its CTR only by split_index, a global numbering of
  // all binary splits: one index per border of each float feature, then one per
  // value of each one-hot feature, then one per border of each CTR.
  std::map<uint64_t, std::pair<uint32_t, uint32_t>> ctr_of_split;
  uint64_t sidx = e.borders_.size() + e.onehot_values_.size();
  for (uint32_t r = 0; r < ctr_borders.size(); ++r) {
    for (uint32_t b = 0; b < ctr_borders[r].size(); ++b) ctr_of_split[sidx++] = {r, b};
  }
  const uint32_t F = static_cast<uint32_t>(e.numeric_column_.size());
  const uint32_t G = static_cast<uint32_t>(e.onehot_column_.size());

  const Json& trees = root.at("oblivious_trees");
  for (size_t t = 0; t < trees.size(); ++t) {
    const Json& tree = trees[t];
    const Json* sp = tree.find("splits");
    const size_t d = (sp == nullptr || sp->is_null()) ? 0 : sp->size();
    BW_CHECK(d <= 8, "tree depth exceeds 8");
    for (size_t l = 0; l < d; ++l) {
      const Json& s = (*sp)[l];
      const std::string& st = s.at("split_type").as_string();
      if (st == "FloatFeature") {
        const uint32_t fi_idx = static_cast<uint32_t>(s.at("float_feature_index").as_double());
        BW_CHECK(fi_idx < e.n_numeric_, "split references an unknown float feature");
        const float border = static_cast<float>(s.at("border").as_double());
        const int32_t sf = split_of[fi_idx];
        BW_CHECK(sf >= 0, "split references a float feature without borders");
        const float* b0 = e.borders_.data() + e.border_begin_[static_cast<size_t>(sf)];
        const float* b1 = e.borders_.data() + e.border_begin_[static_cast<size_t>(sf) + 1];
        const float* hit = std::find(b0, b1, border);
        BW_CHECK(hit != b1, "split border not found in the feature borders");
        // CatBoost's split "x > border_j" is "code > j", because code counts the borders below x.
        e.split_feature_.push_back(static_cast<uint32_t>(sf));
        e.split_kind_.push_back(0);  // GT
        e.split_arg_.push_back(static_cast<uint8_t>(hit - b0));
      } else if (st == "OneHotFeature") {
        const uint32_t c = static_cast<uint32_t>(s.at("cat_feature_index").as_double());
        BW_CHECK(c < e.n_categorical_, "split references an unknown categorical feature");
        const int32_t g = onehot_of[c];
        BW_CHECK(g >= 0, "one-hot split on a categorical feature without values");
        const int32_t v = static_cast<int32_t>(s.at("value").as_double());
        const int32_t* b0 = e.onehot_values_.data() + e.onehot_value_begin_[g];
        const int32_t* b1 = e.onehot_values_.data() + e.onehot_value_begin_[g + 1];
        const int32_t* hit = std::find(b0, b1, v);
        BW_CHECK(hit != b1, "one-hot split value not in the feature values");
        e.split_feature_.push_back(F + static_cast<uint32_t>(g));
        e.split_kind_.push_back(1);  // EQ
        e.split_arg_.push_back(static_cast<uint8_t>(hit - b0 + 1));
      } else {
        BW_CHECK(st == "OnlineCtr", "unknown split_type: " + st);
        const uint64_t si = static_cast<uint64_t>(s.at("split_index").as_double());
        const auto it = ctr_of_split.find(si);
        BW_CHECK(it != ctr_of_split.end(), "OnlineCtr split_index does not name a CTR border");
        const uint32_t r = it->second.first;
        const uint32_t b = it->second.second;
        BW_CHECK(ctr_borders[r][b] == static_cast<float>(s.at("border").as_double()),
                 "OnlineCtr border does not match the border of its split_index");
        e.split_feature_.push_back(F + G + r);
        e.split_kind_.push_back(0);  // GT
        e.split_arg_.push_back(static_cast<uint8_t>(b));
      }
    }
    e.depth_.push_back(static_cast<uint8_t>(d));
    const Json& lv = tree.at("leaf_values");
    const size_t want = (size_t{1} << d) * e.K_;
    BW_CHECK(lv.size() == want, "wrong number of leaf values");
    e.leaves_.reserve(e.leaves_.size() + want);
    for (size_t j = 0; j < want; ++j) e.leaves_.push_back(static_cast<float>(scale * lv[j].as_double()));
  }
  e.rebuild_derived();
  return e;
}

}  // namespace boxwood
