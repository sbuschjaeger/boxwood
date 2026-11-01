// Tests for the boxwood C++ core: JSON parser, CityHash, numeric and categorical
// encoder oracles, many-tree sums, serialization round trip and error cases.
// No framework: CHECK prints file:line and counts failures; main returns 1 if
// anything failed.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "backend.hpp"
#include "boxwood/ensemble.hpp"
#include "cityhash.hpp"
#include "json.hpp"
#include "prune.hpp"

static int g_failures = 0;

#define CHECK(cond)                                                        \
  do {                                                                   \
    if (!(cond)) {                                                       \
      ++g_failures;                                                      \
      std::printf("FAIL %s:%d: CHECK(%s)\n", __FILE__, __LINE__, #cond); \
    }                                                                    \
  } while (0)

#define CHECK_THROWS(expr)                                                                \
  do {                                                                                  \
    bool bw_threw = false;                                                                \
    try {                                                                               \
      (void)(expr);                                                                       \
    } catch (const boxwood::Error&) {                                                     \
      bw_threw = true;                                                                    \
    }                                                                                   \
    if (!bw_threw) {                                                                    \
      ++g_failures;                                                                       \
      std::printf("FAIL %s:%d: expected boxwood::Error from %s\n", __FILE__, __LINE__,   \
                  #expr);                                                                 \
    }                                                                                   \
  } while (0)

static bool near(double a, double b, double tol = 1e-12) {
  return std::fabs(a - b) <= tol * std::max(1.0, std::fabs(b));
}

// ---------------------------------------------------------------- JSON helper

static void test_json() {
  const std::string s = R"({
    "a": [1, {"b": true, "c": null}],
    "d": -1.5e-3,
    "e": 0,
    "f": 123456789,
    "g": "A\"B\\C\/D\bE\fF\nG\rH\tIü",
    "h": "\uD83D\uDE00"
  })";
  const boxwood::Json j = boxwood::parse_json(s);
  CHECK(j.type == boxwood::Json::Type::Object);
  CHECK(j.size() == 6);
  CHECK(j.at("a").size() == 2);
  CHECK(j.at("a")[0].as_double() == 1.0);
  CHECK(j.at("a")[1].at("b").bool_value == true);
  CHECK(j.at("a")[1].at("c").is_null());
  CHECK(j.at("d").as_double() == -1.5e-3);
  CHECK(j.at("e").as_double() == 0.0);
  CHECK(j.at("f").as_double() == 123456789.0);
  CHECK(j.at("g").as_string() == "A\"B\\C/D\bE\fF\nG\rH\tIü");  // ü as UTF-8
  CHECK(j.at("h").as_string() == "\xF0\x9F\x98\x80");            // surrogate pair
  CHECK(j.find("zzz") == nullptr);
  CHECK_THROWS(boxwood::parse_json(R"({"a":})"));
  CHECK_THROWS(boxwood::parse_json("[1,"));
  CHECK_THROWS(boxwood::parse_json("\"abc"));
}

// ----------------------------------------------------------- CityHash tests

static void test_cityhash() {
  struct V {
    std::string s;
    uint64_t h;
    uint32_t c;
  };
  std::string long200;
  for (int i = 0; i < 200; ++i) long200 += static_cast<char>('a' + i % 26);
  const std::vector<V> vs = {
      {"", 11160318154034397263ULL, 797982799u},
      {"a", 2603192927274642682ULL, 3489901818u},
      {"ab", 1725057946192985918ULL, 4090249022u},
      {"abc", 4220206313085259313ULL, 978243121u},
      {"abcd", 17823623939509273229ULL, 345588365u},
      {"abcdefgh", 4864636163090995991ULL, 4270775063u},
      {"abcdefghi", 7406163328168016081ULL, 3019552977u},
      {"0123456789abcdef", 692747204941329175ULL, 2645308183u},
      {"0123456789abcdefg", 792568009951096890ULL, 3849924666u},
      {std::string(32, 'x'), 14051161607482389375ULL, 664286079u},
      {std::string(33, 'x'), 9498923847066961975ULL, 1424607287u},
      {std::string(64, 'y'), 14597941181382302230ULL, 1213635094u},
      {std::string(65, 'z'), 8897417500330900799ULL, 1972452671u},
      {std::string(128, 'q'), 399232587716079808ULL, 1569960128u},
      {std::string(129, 'w'), 2575562685817834195ULL, 3907799763u},
      {long200, 10258548877153202531ULL, 95613283u},
      {"Grüße", 17364701079286602353ULL, 632838769u},
      {"nan", 14479776142306688601ULL, 761118297u},
      {"42", 985280342255011280ULL, 1145636304u},
  };
  for (const V& t : vs) {
    CHECK(boxwood::city_hash64(t.s) == t.h);
    CHECK(boxwood::cat_hash(t.s) == t.c);
  }
}

// ------------------------------------------------------- model JSON builders

static std::string join(const std::vector<std::string>& v) {
  std::string out;
  for (size_t i = 0; i < v.size(); ++i) {
    if (i != 0) out += ",";
    out += v[i];
  }
  return out;
}

static std::string make_split(int fi, const std::string& border) {
  return "{\"float_feature_index\":" + std::to_string(fi) + ",\"border\":" + border +
         ",\"split_type\":\"FloatFeature\"}";
}

static std::string make_tree(const std::vector<std::string>& splits, const std::string& leaves) {
  return "{\"splits\":[" + join(splits) + "],\"leaf_values\":[" + leaves + "]}";
}

static std::string make_model(const std::string& loss, const std::string& scale,
                             const std::vector<std::string>& bias,
                             const std::vector<std::string>& feats,
                             const std::vector<std::string>& trees) {
  return "{\"model_info\":{\"params\":{\"loss_function\":{\"type\":\"" + loss + "\"}}},"
         "\"scale_and_bias\":[" +
         scale + ",[" + join(bias) + "]],"
         "\"features_info\":{\"float_features\":[" +
         join(feats) + "]},"
         "\"oblivious_trees\":[" + join(trees) + "]}";
}

// 2 float features (borders {-1, 0, 0.5} AsFalse and {2} AsTrue), 3 trees with
// depths 0, 1, 2, K = 1, scale 2, bias 0.5.
static std::string small_model() {
  const std::string f0 =
      "{\"flat_feature_index\":0,\"borders\":[-1,0,0.5],\"nan_value_treatment\":\"AsFalse\"}";
  const std::string f1 = "{\"flat_feature_index\":1,\"borders\":[2],\"nan_value_treatment\":\"AsTrue\"}";
  const std::vector<std::string> trees = {
      "{\"splits\":null,\"leaf_values\":[0.25]}",
      make_tree({make_split(0, "0")}, "1,2"),
      make_tree({make_split(1, "2"), make_split(0, "0.5")}, "10,20,30,40"),
  };
  return make_model("RMSE", "2", {"0.5"}, {f0, f1}, trees);
}

// The encoder rule from the spec: code = #{borders b : x > b}, NaN = border_count
// if nan_high else 0.
static int code_of(float x, std::initializer_list<float> borders, bool nan_high) {
  if (std::isnan(x)) return nan_high ? static_cast<int>(borders.size()) : 0;
  int c = 0;
  for (float b : borders) {
    if (x > b) ++c;
  }
  return c;
}

// Leaf indices of trees 1 and 2 and the margin of the small model, from the definitions.
struct Expected {
  int leaf1, leaf2;
  double margin;
};

static Expected expected(float x0, float x1) {
  const int c0 = code_of(x0, {-1.f, 0.f, 0.5f}, false);
  const int c1 = code_of(x1, {2.f}, true);
  const int i1 = c0 > 1 ? 1 : 0;                        // tree 1 splits on border 0 (index 1)
  const int i2 = (c1 > 0 ? 1 : 0) | (c0 > 2 ? 2 : 0);  // tree 2 splits on 2 (index 0) and 0.5 (index 2)
  const double l2[4] = {10, 20, 30, 40};
  const double leaf_sum = 0.25 + (i1 == 1 ? 2.0 : 1.0) + l2[i2];
  return {i1, i2, 0.5 + 2.0 * leaf_sum};  // bias + scale * leaves
}

// ------------------------------------------------------- encoder oracle test

static void check_targets(const boxwood::ObliviousTreeEnsemble& e, const std::vector<float>& xs,
                      float fixed, bool vary_first) {
  const size_t n = xs.size();
  std::vector<float> X(2 * n);
  for (size_t r = 0; r < n; ++r) {
    X[2 * r] = vary_first ? xs[r] : fixed;
    X[2 * r + 1] = vary_first ? fixed : xs[r];
  }
  std::vector<double> m(n);
  std::vector<uint8_t> li(3 * n);
  const boxwood::Batch B{std::span<const float>(X), n};
  e.predict_margin(B, m);
  e.leaf_index(B, li);
  for (size_t r = 0; r < n; ++r) {
    const Expected ex = expected(X[2 * r], X[2 * r + 1]);
    CHECK(li[3 * r] == 0);
    CHECK(li[3 * r + 1] == ex.leaf1);
    CHECK(li[3 * r + 2] == ex.leaf2);
    CHECK(near(m[r], ex.margin));
  }
}

static void test_encoder_oracle() {
  const auto e = boxwood::ObliviousTreeEnsemble::from_catboost_json(small_model());
  CHECK(e.info().task == boxwood::Task::Regression);
  CHECK(e.info().n_outputs == 1);
  CHECK(e.info().n_numeric == 2);
  CHECK(e.info().n_trees == 3);
  CHECK(e.info().n_splits == 3);

  const std::vector<float> xs = {
      -std::numeric_limits<float>::infinity(), -2.f, -1.f, -0.99f, 0.f,
      0.25f, 0.5f, 0.51f, 3.f, std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::quiet_NaN()};
  const std::vector<float> ys = {-std::numeric_limits<float>::infinity(), 1.f, 2.f,
                                2.0001f, 3.f, std::numeric_limits<float>::infinity(),
                                std::numeric_limits<float>::quiet_NaN()};
  check_targets(e, xs, 0.f, true);   // vary feature 0
  check_targets(e, ys, 0.f, false);  // vary feature 1
}

// --------------------------------------------------------- many trees test

static void test_many_trees() {
  std::vector<std::string> trees;
  for (int t = 0; t < 150; ++t) {
    trees.push_back(make_tree({make_split(0, "0")},
                            std::to_string(t + 1) + "," + std::to_string(2 * (t + 1))));
  }
  const std::string f = "{\"flat_feature_index\":0,\"borders\":[0],\"nan_value_treatment\":\"AsFalse\"}";
  const auto e = boxwood::ObliviousTreeEnsemble::from_catboost_json(
      make_model("RMSE", "1", {"0"}, {f}, trees));
  CHECK(e.info().n_trees == 150);
  CHECK(e.info().n_splits == 150);
  std::vector<float> X = {1.f, -1.f};
  std::vector<double> m(2);
  const boxwood::Batch B{std::span<const float>(X), 2};
  e.predict_margin(B, m);
  CHECK(near(m[0], 2.0 * 11325.0, 1e-6));  // sum of 2*(t+1)
  CHECK(near(m[1], 11325.0, 1e-6));        // sum of (t+1)
}

// ----------------------------------------------------- serialization tests

// Element size of each serialized array, in layout order (after the header).
static const size_t kElemSize[27] = {4, 4, 4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 1,
                                    4, 4, 4, 8, 4, 1, 1, 1, 4, 1, 1, 4, 8};

// Offset of the data of array `idx` in a serialized model (header: magic 4,
// version 4, task 1, leaf type 1, K 4, n_numeric 4, n_categorical 4 = 22
// bytes; each array is a uint64 count + raw bytes).
static size_t array_data_offset(const std::vector<std::byte>& b, size_t idx) {
  size_t pos = 22;
  for (size_t i = 0; i < idx; ++i) {
    uint64_t cnt = 0;
    std::memcpy(&cnt, b.data() + pos, 8);
    pos += 8 + static_cast<size_t>(cnt) * kElemSize[i];
  }
  return pos + 8;
}

static void test_serialization() {
  const auto e = boxwood::ObliviousTreeEnsemble::from_catboost_json(small_model());
  const std::vector<std::byte> bytes = e.to_bytes();
  const auto e2 = boxwood::ObliviousTreeEnsemble::from_bytes(bytes);
  CHECK(e2.to_bytes() == bytes);

  const std::vector<float> X = {-2.f, 3.f, 0.5f, 0.f, 0.51f, 2.5f};
  const size_t n = X.size() / 2;
  std::vector<double> m1(n), m2(n);
  const boxwood::Batch B{std::span<const float>(X), n};
  e.predict_margin(B, m1);
  e2.predict_margin(B, m2);
  CHECK(m1 == m2);  // bit-exact

  for (size_t len = 0; len < bytes.size(); ++len) {
    CHECK_THROWS(boxwood::ObliviousTreeEnsemble::from_bytes(std::span<const std::byte>(bytes.data(), len)));
  }

  std::vector<std::byte> bad = bytes;
  bad[0] = std::byte{'X'};  // wrong magic
  CHECK_THROWS(boxwood::ObliviousTreeEnsemble::from_bytes(bad));

  bad = bytes;
  bad[4] = std::byte{99};  // wrong version
  CHECK_THROWS(boxwood::ObliviousTreeEnsemble::from_bytes(bad));

  bad = bytes;  // borders count far too large
  {
    const uint64_t huge = 1ULL << 40;
    std::memcpy(bad.data() + array_data_offset(bytes, 5) - 8, &huge, 8);
  }
  CHECK_THROWS(boxwood::ObliviousTreeEnsemble::from_bytes(bad));

  bad = bytes;  // border_count[0] > 255
  {
    const uint32_t v = 300;
    std::memcpy(bad.data() + array_data_offset(bytes, 3), &v, 4);
  }
  CHECK_THROWS(boxwood::ObliviousTreeEnsemble::from_bytes(bad));

  bad = bytes;  // split count does not match the tree depths
  {
    const uint64_t v = 1;
    std::memcpy(bad.data() + array_data_offset(bytes, 22) - 8, &v, 8);
  }
  CHECK_THROWS(boxwood::ObliviousTreeEnsemble::from_bytes(bad));

  bad = bytes;  // split_feature out of range
  {
    const uint32_t v = 99;
    std::memcpy(bad.data() + array_data_offset(bytes, 22), &v, 4);
  }
  CHECK_THROWS(boxwood::ObliviousTreeEnsemble::from_bytes(bad));
}

// ------------------------------------------------------ categorical model

// The CTR projection hash of the reference: acc = M * (acc + M * v).
static const uint64_t kMix = 0x4906BA494954CB65ULL;
static uint64_t mixf(uint64_t acc, uint64_t v) { return kMix * (acc + kMix * v); }
static uint64_t signext(int32_t h) { return static_cast<uint64_t>(static_cast<int64_t>(h)); }

// A model with one numeric feature (f0, borders {0, 1} AsFalse), one float
// feature with AsTrue (f1, border {3}), one one-hot categorical feature with
// values {a, b, c}, one CTR of type Borders on the category (table of 3
// entries plus an empty-bucket entry) and one combination CTR (category hash +
// float border of f1). Split features: f0, f1, one-hot, ctr0, ctr1.
//
// Hand-computed codes (prior 0/1, shift 0, scale 1):
//   ctr0 (Borders, target_border_idx 0, borders [0.5]):
//     "a": stats [1,3] -> 3/4 = 0.6 -> code 1;  "b": [2,1] -> 1/3 -> code 0;
//     "c": [1,1] -> 1/2 -> 0.5 -> code 0;  unknown: 0/1 -> code 0.
//   ctr1 (BinarizedTargetMeanValue, borders [0.25]):
//     ("a", f1>3): [8,16] -> 8/17 -> code 1;  ("b", f1<=3): [4,5] -> 4/6 -> code 1;
//     other keys: 0/1 -> code 0.
static std::string cat_model_json() {
  const int32_t ha = static_cast<int32_t>(boxwood::cat_hash("a"));
  const int32_t hb = static_cast<int32_t>(boxwood::cat_hash("b"));
  const int32_t hc = static_cast<int32_t>(boxwood::cat_hash("c"));
  const uint64_t k0a = mixf(0, signext(ha));  // single-category projection
  const uint64_t k0b = mixf(0, signext(hb));
  const uint64_t k0c = mixf(0, signext(hc));
  const uint64_t ka1 = mixf(mixf(0, signext(ha)), 1);  // ("a", float bit 1)
  const uint64_t kb0 = mixf(mixf(0, signext(hb)), 0);  // ("b", float bit 0)
  return R"({"model_info":{"params":{"loss_function":{"type":"RMSE"}}},
    "scale_and_bias":[2.0,[0.5]],
    "features_info":{
      "float_features":[
        {"flat_feature_index":0,"borders":[0.0,1.0],"nan_value_treatment":"AsFalse"},
        {"flat_feature_index":1,"borders":[3.0],"nan_value_treatment":"AsTrue"}],
      "categorical_features":[
        {"flat_feature_index":1,"values":[)" +
         std::to_string(ha) + "," + std::to_string(hb) + "," + std::to_string(hc) + R"(]}],
      "ctrs":[
        {"ctr_type":"Borders","target_border_idx":0,"prior_numerator":0.0,
         "prior_denomerator":1.0,"shift":0.0,"scale":1.0,"borders":[0.5],"identifier":"t0",
         "elements":[{"combination_element":"cat_feature_value","cat_feature_index":0}]},
        {"ctr_type":"BinarizedTargetMeanValue","target_border_idx":0,"prior_numerator":0.0,
         "prior_denomerator":1.0,"shift":0.0,"scale":1.0,"borders":[0.25],"identifier":"t1",
         "elements":[{"combination_element":"cat_feature_value","cat_feature_index":0},
                    {"combination_element":"float_feature","float_feature_index":1,"border":3.0}]}]},
    "ctr_data":{
      "t0":{"hash_stride":3,"counter_denominator":10,"hash_map":[")" +
      std::to_string(k0a) + R"(",1,3,")" + std::to_string(k0b) + R"(",2,1,")" +
      std::to_string(k0c) + R"(",1,1,"18446744073709551615",1,3]},
      "t1":{"hash_stride":3,"counter_denominator":10,"hash_map":[")" +
      std::to_string(ka1) + R"(",8,16,")" + std::to_string(kb0) + R"(",4,5]}},
    "oblivious_trees":[
      {"splits":[{"cat_feature_index":0,"value":)" + std::to_string(hb) +
                 R"(,"split_type":"OneHotFeature"},
                 {"split_index":6,"border":0.5,"split_type":"OnlineCtr"}],
       "leaf_values":[10,20,30,40]},
      {"splits":[{"split_index":7,"border":0.25,"split_type":"OnlineCtr"},
                 {"float_feature_index":0,"border":1.0,"split_type":"FloatFeature"}],
       "leaf_values":[100,200,300,400]}]})";
}

// Observations with the leaf index of each tree and the margin, all computed by hand
// from the codes above. margin = 0.5 + 2 * (leaf0 + leaf1).
struct CatObs {
  float x0;
  float x1;
  const char* cat;
  int leaf0;
  int leaf1;
  double margin;
};

static const float kNaN = std::numeric_limits<float>::quiet_NaN();

static const CatObs kCatObs[] = {
    {2.0f, 5.0f, "a", 2, 3, 860.5},   // ctr0=1, ctr1=1 (key a1)
    {0.0f, 0.0f, "a", 2, 0, 260.5},   // ctr1 key (a,0) not in table -> default
    {0.6f, 5.0f, "a", 2, 1, 460.5},   // ctr1=1, x0 <= 1
    {2.0f, 0.0f, "b", 1, 3, 840.5},   // one-hot eq, ctr1=1 (key b0)
    {2.0f, 5.0f, "b", 1, 2, 640.5},   // ctr1 key (b,1) not in table
    {0.0f, 0.0f, "c", 0, 0, 220.5},   // ctr0("c")=0
    {2.0f, 0.0f, "z", 0, 2, 620.5},   // unknown category: default codes
    {0.0f, kNaN, "a", 2, 1, 460.5},   // NaN in the float element, AsTrue -> bit 1
};

static void check_cat_obs(const boxwood::ObliviousTreeEnsemble& e) {
  const size_t n = sizeof(kCatObs) / sizeof(CatObs);
  std::vector<float> X(2 * n);
  std::vector<std::string_view> cat(n);
  for (size_t i = 0; i < n; ++i) {
    X[2 * i] = kCatObs[i].x0;
    X[2 * i + 1] = kCatObs[i].x1;
    cat[i] = kCatObs[i].cat;
  }
  const boxwood::Batch B{std::span<const float>(X), n,
                        {std::span<const std::string_view>(cat)}};
  std::vector<uint8_t> li(2 * n);
  std::vector<double> m(n);
  e.leaf_index(B, li);
  e.predict_margin(B, m);
  for (size_t i = 0; i < n; ++i) {
    CHECK(li[2 * i] == kCatObs[i].leaf0);
    CHECK(li[2 * i + 1] == kCatObs[i].leaf1);
    CHECK(near(m[i], kCatObs[i].margin));
  }
  // The same observations dictionary-encoded give the same bits.
  std::vector<std::string_view> dict;
  std::vector<uint32_t> codes;
  for (const std::string_view s : cat) {
    const auto it = std::find(dict.begin(), dict.end(), s);
    codes.push_back(static_cast<uint32_t>(it - dict.begin()));
    if (it == dict.end()) dict.push_back(s);
  }
  boxwood::Batch D{std::span<const float>(X), n, {std::span<const std::string_view>(dict)}};
  D.categorical_codes = {std::span<const uint32_t>(codes)};
  std::vector<uint8_t> li_d(2 * n);
  std::vector<double> m_d(n);
  e.leaf_index(D, li_d);
  e.predict_margin(D, m_d);
  CHECK(li_d == li);
  CHECK(m_d == m);
}

static void test_categorical() {
  const auto e = boxwood::ObliviousTreeEnsemble::from_catboost_json(cat_model_json());
  CHECK(e.info().n_numeric == 2);
  CHECK(e.info().n_categorical == 1);
  CHECK((e.info().numeric_flat_index == std::vector<uint32_t>{0, 1}));
  CHECK((e.info().categorical_flat_index == std::vector<uint32_t>{1}));
  CHECK(e.info().n_trees == 2);
  CHECK(e.info().n_splits == 4);

  const auto s0 = e.splits(0);
  CHECK(s0.size() == 2);
  CHECK(s0[0].source == 1 && s0[0].index == 0 && s0[0].kind == 1 && s0[0].arg == 2);
  CHECK(s0[1].source == 2 && s0[1].index == 0 && s0[1].kind == 0 && s0[1].arg == 0);
  const auto s1 = e.splits(1);
  CHECK(s1.size() == 2);
  CHECK(s1[0].source == 2 && s1[0].index == 1 && s1[0].kind == 0 && s1[0].arg == 0);
  CHECK(s1[1].source == 0 && s1[1].index == 0 && s1[1].kind == 0 && s1[1].arg == 1);

  check_cat_obs(e);
}

static void test_categorical_serialization() {
  const auto e = boxwood::ObliviousTreeEnsemble::from_catboost_json(cat_model_json());
  const std::vector<std::byte> bytes = e.to_bytes();
  const auto e2 = boxwood::ObliviousTreeEnsemble::from_bytes(bytes);
  CHECK(e2.to_bytes() == bytes);
  check_cat_obs(e2);

  for (size_t len = 0; len < bytes.size(); ++len) {
    CHECK_THROWS(boxwood::ObliviousTreeEnsemble::from_bytes(
        std::span<const std::byte>(bytes.data(), len)));
  }

  std::vector<std::byte> bad = bytes;  // onehot_count[0] breaks the values sum
  {
    const uint8_t v = 255;
    std::memcpy(bad.data() + array_data_offset(bytes, 7), &v, 1);
  }
  CHECK_THROWS(boxwood::ObliviousTreeEnsemble::from_bytes(bad));

  bad = bytes;  // elem_kind out of range
  {
    const uint8_t v = 3;
    std::memcpy(bad.data() + array_data_offset(bytes, 10), &v, 1);
  }
  CHECK_THROWS(boxwood::ObliviousTreeEnsemble::from_bytes(bad));

  bad = bytes;  // table keys not ascending (first two keys of table 0 made equal)
  {
    uint64_t k[2];
    std::memcpy(k, bad.data() + array_data_offset(bytes, 17), 16);
    k[0] = k[1];
    std::memcpy(bad.data() + array_data_offset(bytes, 17), k, 16);
  }
  CHECK_THROWS(boxwood::ObliviousTreeEnsemble::from_bytes(bad));

  bad = bytes;  // ctr_codes count does not match the table sizes
  {
    const uint64_t v = 0;
    std::memcpy(bad.data() + array_data_offset(bytes, 20) - 8, &v, 8);
  }
  CHECK_THROWS(boxwood::ObliviousTreeEnsemble::from_bytes(bad));

  bad = bytes;  // split_feature out of range (>= n_features = 5)
  {
    const uint32_t v = 5;
    std::memcpy(bad.data() + array_data_offset(bytes, 22), &v, 4);
  }
  CHECK_THROWS(boxwood::ObliviousTreeEnsemble::from_bytes(bad));
}

// ----------------------------------------------------------- backend tests

// A random numeric model: 4 float features with 1, 3, 17 and 255 borders
// (ascending integers), `n_trees` trees with random depths 0..max_depth and
// random splits, random leaf values, K outputs.
static std::string random_numeric_model(uint32_t K, int n_trees, int max_depth, std::mt19937& rng) {
  const std::vector<int> border_counts = {1, 3, 17, 255};
  std::vector<std::string> feats;
  for (size_t i = 0; i < border_counts.size(); ++i) {
    std::vector<std::string> bs;
    for (int j = 0; j < border_counts[i]; ++j) bs.push_back(std::to_string(j + 1));
    feats.push_back("{\"flat_feature_index\":" + std::to_string(i) + ",\"borders\":[" + join(bs) +
                   "],\"nan_value_treatment\":\"" + (i % 2 == 0 ? "AsFalse" : "AsTrue") + "\"}");
  }
  std::uniform_int_distribution<int> depth_dist(0, max_depth);
  std::uniform_real_distribution<float> leaf_dist(-100.0f, 100.0f);
  std::vector<std::string> trees;
  for (int t = 0; t < n_trees; ++t) {
    const int d = depth_dist(rng);
    std::vector<std::string> splits;
    for (int l = 0; l < d; ++l) {
      const size_t fi = rng() % border_counts.size();
      const int b = static_cast<int>(rng() % border_counts[fi]);
      splits.push_back(make_split(static_cast<int>(fi), std::to_string(b + 1)));
    }
    std::vector<std::string> leaves;
    const size_t nleaf = (size_t{1} << d) * K;
    for (size_t j = 0; j < nleaf; ++j) leaves.push_back(std::to_string(leaf_dist(rng)));
    trees.push_back(make_tree(splits, join(leaves)));
  }
  std::vector<std::string> bias;
  for (uint32_t k = 0; k < K; ++k) bias.push_back("0.5");
  return make_model(K == 1 ? "RMSE" : "MultiClass", "1", bias, feats, trees);
}

// A value for X: mostly inside the border range, with NaN and +/-inf.
static float random_x(std::mt19937& rng) {
  switch (rng() % 8) {
    case 0: return std::numeric_limits<float>::quiet_NaN();
    case 1: return std::numeric_limits<float>::infinity();
    case 2: return -std::numeric_limits<float>::infinity();
    default: return std::uniform_real_distribution<float>(-2.0f, 260.0f)(rng);
  }
}

// The binning rule over all 255 borders of a feature, on every backend: tree t
// splits on border t + 1 of the borders 1..255, so its leaf is 1 exactly when
// the code of x (the number of borders below x) is above t.
static void test_binning_255() {
  std::vector<std::string> bs;
  std::vector<float> borders;
  for (int j = 1; j <= 255; ++j) {
    bs.push_back(std::to_string(j));
    borders.push_back(static_cast<float>(j));
  }
  std::vector<std::string> trees;
  for (int t = 0; t < 255; ++t) trees.push_back(make_tree({make_split(0, std::to_string(t + 1))}, "0,1"));
  for (const bool nan_high : {false, true}) {
    const std::string f = "{\"flat_feature_index\":0,\"borders\":[" + join(bs) +
                          "],\"nan_value_treatment\":\"" + (nan_high ? "AsTrue" : "AsFalse") + "\"}";
    auto e = boxwood::ObliviousTreeEnsemble::from_catboost_json(make_model("RMSE", "1", {"0"}, {f}, trees));
    std::vector<float> X = {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
                            -std::numeric_limits<float>::infinity(), 0.0f, 1.0f, 255.0f, 256.0f};
    for (int j = 0; j < 600; ++j) X.push_back(static_cast<float>(j) * 0.45f - 1.0f);
    const size_t n = X.size();
    for (const std::string& name : boxwood::available_backends()) {
      e.set_backend(name);
      std::vector<uint8_t> leaf(n * 255);
      e.leaf_index(boxwood::Batch{std::span<const float>(X), n}, leaf);
      for (size_t i = 0; i < n; ++i) {
        int code = 0;
        if (std::isnan(X[i])) {
          code = nan_high ? 255 : 0;
        } else {
          for (const float b : borders) code += X[i] > b ? 1 : 0;
        }
        for (int t = 0; t < 255; ++t) CHECK(leaf[i * 255 + t] == (code > t ? 1 : 0));
      }
    }
  }
}

// Every available backend must give bit-identical margins and leaf indices.
// Shallow models (depth at most 5) and deep ones take different lookup paths
// in the SIMD backends.
static void test_backends() {
  const std::vector<std::string> names = boxwood::available_backends();
  CHECK(!names.empty());
  const std::vector<size_t> row_counts = {1, 7, 31, 33, 255, 256, 257, 1000};
  for (const auto [K, max_depth] : {std::pair{1u, 5}, std::pair{1u, 8}, std::pair{3u, 5}, std::pair{3u, 8}}) {
    std::mt19937 rng(1234 + K);
    auto e = boxwood::ObliviousTreeEnsemble::from_catboost_json(
        random_numeric_model(K, 150, max_depth, rng));
    CHECK(e.info().n_trees == 150);
    for (const boxwood::LeafType lt : {boxwood::LeafType::Float32, boxwood::LeafType::Int8}) {
      e.set_leaf_type(lt);
      CHECK(e.info().leaf_type == lt);
      for (const size_t n : row_counts) {
        std::vector<float> X(n * 4);
        for (float& x : X) x = random_x(rng);
        const boxwood::Batch B{std::span<const float>(X), n};
        e.set_backend("scalar");
        std::vector<double> ms(n * K);
        std::vector<uint8_t> ls(n * 150);
        e.predict_margin(B, ms);
        e.leaf_index(B, ls);
        for (const std::string& name : names) {
          e.set_backend(name);
          std::vector<double> mb(n * K);
          std::vector<uint8_t> lb(n * 150);
          e.predict_margin(B, mb);
          e.leaf_index(B, lb);
          CHECK(mb == ms);  // bit-exact
          CHECK(lb == ls);
        }
      }
    }
  }
  // The categorical model of the existing tests, with each backend.
  auto cat = boxwood::ObliviousTreeEnsemble::from_catboost_json(cat_model_json());
  for (const std::string& name : names) {
    cat.set_backend(name);
    check_cat_obs(cat);
  }
  cat.set_backend("");  // best
  check_cat_obs(cat);
  CHECK_THROWS(cat.set_backend("nope"));
}

// Threads split the observations into ranges of blocks: the margins and leaf indices
// are those of one thread, also for observation counts that do not fill the blocks.
static void test_threads() {
  std::mt19937 rng(77);
  auto e = boxwood::ObliviousTreeEnsemble::from_catboost_json(random_numeric_model(3, 40, 8, rng));
  for (const size_t n : {size_t{1}, size_t{1000}, size_t{5000}, size_t{20001}}) {
    std::vector<float> X(n * 4);
    for (float& x : X) x = random_x(rng);
    const boxwood::Batch B{std::span<const float>(X), n};
    e.set_threads(1);
    std::vector<double> m1(n * 3);
    std::vector<uint8_t> l1(n * 40);
    e.predict_margin(B, m1);
    e.leaf_index(B, l1);
    for (const unsigned t : {2u, 3u, 8u}) {
      e.set_threads(t);
      std::vector<double> mt(n * 3);
      std::vector<uint8_t> lt(n * 40);
      e.predict_margin(B, mt);
      e.leaf_index(B, lt);
      CHECK(mt == m1);
      CHECK(lt == l1);
    }
  }
}

// A single-tree model with 8 numeric features (one border 0 each) whose tree
// splits on feature l at level l: the observation with x_l = 1 / -1 sets / clears bit
// l of the leaf index, so a crafted observation hits any of the 256 leaves.
static std::string leaf_per_bit_model(const std::vector<std::string>& leaves) {
  std::vector<std::string> feats, splits;
  for (int i = 0; i < 8; ++i) {
    feats.push_back("{\"flat_feature_index\":" + std::to_string(i) +
                   ",\"borders\":[0],\"nan_value_treatment\":\"AsFalse\"}");
    splits.push_back(make_split(i, "0"));
  }
  return make_model("RMSE", "1", {"0"}, feats, {make_tree(splits, join(leaves))});
}

// The int8 quantization rule: the error of every leaf read through the margin,
// and the round trips through bytes and through Float32.
static void test_int8_quantization() {
  std::mt19937 rng(99);
  std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
  std::vector<std::string> leaves;
  std::vector<float> w(256);
  for (size_t j = 0; j < w.size(); ++j) {
    const std::string s = std::to_string(dist(rng));
    leaves.push_back(s);
    w[j] = std::stof(s);  // the float32 the model stores
  }
  auto e = boxwood::ObliviousTreeEnsemble::from_catboost_json(leaf_per_bit_model(leaves));
  float max_abs = 0.0f;
  for (const float x : w) max_abs = std::max(max_abs, std::fabs(x));
  const float s = max_abs / 127.0f;  // the tree's scale

  e.set_leaf_type(boxwood::LeafType::Int8);
  CHECK(e.info().leaf_type == boxwood::LeafType::Int8);

  // |w - float(q)*s| <= s/2 (+ float32 rounding of the product), per leaf.
  std::vector<float> X(8);
  std::vector<double> m(1);
  for (int j = 0; j < 256; ++j) {
    for (int l = 0; l < 8; ++l) X[l] = (j >> l) & 1 ? 1.0f : -1.0f;
    const boxwood::Batch B{std::span<const float>(X), 1};
    e.predict_margin(B, m);
    CHECK(std::fabs(m[0] - w[j]) <= s / 2.0 + 1e-4);
  }

  // Int8 -> bytes -> Int8 is bit-exact.
  const std::vector<std::byte> b1 = e.to_bytes();
  CHECK(boxwood::ObliviousTreeEnsemble::from_bytes(b1).to_bytes() == b1);

  // Int8 -> Float32 keeps the dequantized values (rounded to float32), and
  // Float32 -> Int8 again gives the same codes and steps (the bytes are equal).
  e.set_leaf_type(boxwood::LeafType::Float32);
  CHECK(e.info().leaf_type == boxwood::LeafType::Float32);
  e.set_leaf_type(boxwood::LeafType::Int8);
  CHECK(e.to_bytes() == b1);

  // All-zero leaves: scale 1, codes 0.
  auto z = boxwood::ObliviousTreeEnsemble::from_catboost_json(
      leaf_per_bit_model(std::vector<std::string>(256, "0")));
  z.set_leaf_type(boxwood::LeafType::Int8);
  const std::vector<float> xz(8, 1.f);
  std::vector<double> mz(1);
  z.predict_margin(boxwood::Batch{std::span<const float>(xz), 1}, mz);
  CHECK(mz[0] == 0.0);
}

// ----------------------------------------------------------- error tests

// A model with one float feature, one one-hot categorical feature and one CTR,
// with the given single-split tree.
static std::string one_cat_model(const std::string& splits, const std::string& leaves) {
  const int32_t ha = static_cast<int32_t>(boxwood::cat_hash("a"));
  const uint64_t k0a = mixf(0, signext(ha));
  return R"({"model_info":{"params":{"loss_function":{"type":"RMSE"}}},
    "scale_and_bias":[1,[0]],
    "features_info":{
      "float_features":[{"flat_feature_index":0,"borders":[0.0],"nan_value_treatment":"AsFalse"}],
      "categorical_features":[{"flat_feature_index":1,"values":[)" + std::to_string(ha) + R"(]}],
      "ctrs":[{"ctr_type":"Borders","target_border_idx":0,"prior_numerator":0.0,
        "prior_denomerator":1.0,"shift":0.0,"scale":1.0,"borders":[0.5],"identifier":"t0",
        "elements":[{"combination_element":"cat_feature_value","cat_feature_index":0}]}]},
    "ctr_data":{"t0":{"hash_stride":3,"counter_denominator":10,"hash_map":[")" +
    std::to_string(k0a) + R"(",1,3]}},
    "oblivious_trees":[{"splits":[)" + splits + R"(],"leaf_values":[)" + leaves + R"(]}]})";
}

static void test_errors() {
  const std::string f = "{\"flat_feature_index\":0,\"borders\":[0],\"nan_value_treatment\":\"AsFalse\"}";

  // depth 9
  std::vector<std::string> splits;
  for (int l = 0; l < 9; ++l) splits.push_back(make_split(0, "0"));
  CHECK_THROWS(boxwood::ObliviousTreeEnsemble::from_catboost_json(
      make_model("RMSE", "1", {"0"}, {f}, {make_tree(splits, "1")})));

  // unknown loss
  CHECK_THROWS(boxwood::ObliviousTreeEnsemble::from_catboost_json(
      make_model("MAE", "1", {"0"}, {}, {})));

  // unknown split type
  const std::string badsplit =
      "{\"float_feature_index\":0,\"border\":0,\"split_type\":\"SomethingElse\"}";
  CHECK_THROWS(boxwood::ObliviousTreeEnsemble::from_catboost_json(
      make_model("RMSE", "1", {"0"}, {f}, {make_tree({badsplit}, "1,2")})));

  // wrong number of leaf values
  CHECK_THROWS(boxwood::ObliviousTreeEnsemble::from_catboost_json(
      make_model("RMSE", "1", {"0"}, {f}, {make_tree({make_split(0, "0")}, "1")})));

  // split border not among the feature's borders
  const std::string f2 =
      "{\"flat_feature_index\":0,\"borders\":[0,1],\"nan_value_treatment\":\"AsFalse\"}";
  CHECK_THROWS(boxwood::ObliviousTreeEnsemble::from_catboost_json(
      make_model("RMSE", "1", {"0"}, {f2}, {make_tree({make_split(0, "0.5")}, "1,2")})));

  // OneHotFeature split with a value that is not in the feature's values
  const int32_t hq = static_cast<int32_t>(boxwood::cat_hash("q"));
  CHECK_THROWS(boxwood::ObliviousTreeEnsemble::from_catboost_json(one_cat_model(
      "{\"cat_feature_index\":0,\"value\":" + std::to_string(hq) +
          ",\"split_type\":\"OneHotFeature\"}",
      "1,2")));

  // OnlineCtr split whose split_index does not name a CTR border
  CHECK_THROWS(boxwood::ObliviousTreeEnsemble::from_catboost_json(one_cat_model(
      "{\"split_index\":99,\"border\":0.5,\"split_type\":\"OnlineCtr\"}", "1,2")));

  // OnlineCtr split whose border does not match the border of its split_index
  // (float borders: 1, one-hot values: 1, so the CTR border is at index 2).
  CHECK_THROWS(boxwood::ObliviousTreeEnsemble::from_catboost_json(one_cat_model(
      "{\"split_index\":2,\"border\":9.0,\"split_type\":\"OnlineCtr\"}", "1,2")));

  // shape checks
  const auto e = boxwood::ObliviousTreeEnsemble::from_catboost_json(small_model());
  const std::vector<float> x3 = {0.f, 0.f, 0.f};  // n = 1 needs 2 floats
  std::vector<double> o1(1), o2(2);
  std::vector<uint8_t> li(2);
  CHECK_THROWS(e.predict_margin(boxwood::Batch{std::span<const float>(x3), 1}, o1));
  CHECK_THROWS(e.predict_margin(boxwood::Batch{std::span<const float>(x3.data(), 2), 1}, o2));
  CHECK_THROWS(e.leaf_index(boxwood::Batch{std::span<const float>(x3.data(), 2), 1}, li));

  // wrong number of categorical columns in a Batch
  const auto cat = boxwood::ObliviousTreeEnsemble::from_catboost_json(cat_model_json());
  const std::vector<float> x2 = {0.f, 0.f};
  std::vector<double> m(1);
  std::vector<uint8_t> one(1);
  CHECK_THROWS(cat.predict_margin(boxwood::Batch{std::span<const float>(x2), 1}, m));
  // a categorical span with the wrong number of values
  const std::vector<std::string_view> two = {"a", "b"};
  boxwood::Batch bad_batch{std::span<const float>(x2), 1,
                          {std::span<const std::string_view>(two)}};
  CHECK_THROWS(cat.predict_margin(bad_batch, m));
  CHECK_THROWS(cat.leaf_index(bad_batch, one));
  // dictionary-encoded: a code past the dictionary, a code span of the wrong
  // length, a code span missing
  const std::vector<uint32_t> code2 = {2};
  const std::vector<uint32_t> codes01 = {0, 1};
  boxwood::Batch dict_batch{std::span<const float>(x2), 1, {std::span<const std::string_view>(two)}};
  dict_batch.categorical_codes = {std::span<const uint32_t>(code2)};
  CHECK_THROWS(cat.predict_margin(dict_batch, m));
  dict_batch.categorical_codes = {std::span<const uint32_t>(codes01)};
  CHECK_THROWS(cat.predict_margin(dict_batch, m));
  dict_batch.categorical_codes = {std::span<const uint32_t>(code2), std::span<const uint32_t>(code2)};
  CHECK_THROWS(cat.predict_margin(dict_batch, m));
}

// ----------------------------------------------------------- link tests

static void test_predict_links() {
  const std::string f = "{\"flat_feature_index\":0,\"borders\":[0],\"nan_value_treatment\":\"AsFalse\"}";
  const auto eb = boxwood::ObliviousTreeEnsemble::from_catboost_json(
      make_model("Logloss", "1", {"0"}, {f}, {make_tree({make_split(0, "0")}, "0,2")}));
  CHECK(eb.info().task == boxwood::Task::Binary);
  std::vector<float> xb = {1.f, -1.f};
  std::vector<double> pb(2);
  eb.predict(boxwood::Batch{std::span<const float>(xb), 2}, pb);
  CHECK(near(pb[0], 1.0 / (1.0 + std::exp(-2.0))));
  CHECK(near(pb[1], 0.5));

  const std::string mc = R"({"model_info":{"params":{"loss_function":{"type":"MultiClass"}}},
    "scale_and_bias":[1,[0,0,0]],"features_info":{"float_features":[]},
    "oblivious_trees":[{"splits":null,"leaf_values":[1,2,3]}]})";
  const auto em = boxwood::ObliviousTreeEnsemble::from_catboost_json(mc);
  CHECK(em.info().task == boxwood::Task::Multiclass);
  CHECK(em.info().n_outputs == 3);
  std::vector<double> pm(3);
  em.predict(boxwood::Batch{{}, 1}, pm);
  const double e1 = std::exp(-2.0), e2 = std::exp(-1.0), e3 = 1.0;  // relative to max 3
  const double s = e1 + e2 + e3;
  CHECK(near(pm[0], e1 / s));
  CHECK(near(pm[1], e2 / s));
  CHECK(near(pm[2], e3 / s));

  // a model with no float features at all
  const std::string empty = R"({"model_info":{"params":{"loss_function":{"type":"Logloss"}}},
    "scale_and_bias":[1,[0.25]],"features_info":{},
    "oblivious_trees":[{"splits":null,"leaf_values":[0.5]},{"splits":[],"leaf_values":[0.25]}]})";
  const auto e0 = boxwood::ObliviousTreeEnsemble::from_catboost_json(empty);
  CHECK(e0.info().n_numeric == 0);
  CHECK(e0.info().n_trees == 2);
  std::vector<double> m(3);
  e0.predict_margin(boxwood::Batch{{}, 3}, m);
  CHECK(m[0] == 1.0 && m[1] == 1.0 && m[2] == 1.0);  // 0.25 + 0.5 + 0.25
}

// ----------------------------------------------------------- splits tests

static void test_splits() {
  const auto e = boxwood::ObliviousTreeEnsemble::from_catboost_json(small_model());
  CHECK(e.splits(0).empty());  // depth-0 tree
  const auto s1 = e.splits(1);
  CHECK(s1.size() == 1);
  CHECK(s1[0].source == 0 && s1[0].index == 0 && s1[0].kind == 0 && s1[0].arg == 1);  // x0 > 0
  const auto s2 = e.splits(2);
  CHECK(s2.size() == 2);
  CHECK(s2[0].source == 0 && s2[0].index == 1 && s2[0].kind == 0 && s2[0].arg == 0);  // x1 > 2
  CHECK(s2[1].source == 0 && s2[1].index == 0 && s2[1].kind == 0 && s2[1].arg == 2);  // x0 > 0.5
  CHECK_THROWS(e.splits(3));

  // index is the column of Batch.numeric, not the index among split features:
  // the middle float feature has no borders and is therefore not a split feature.
  const std::string with_gap = make_model(
      "RMSE", "1", {"0"},
      {R"({"flat_feature_index":0,"borders":[0],"nan_value_treatment":"AsFalse"})",
       R"({"flat_feature_index":1,"borders":[],"nan_value_treatment":"AsFalse"})",
       R"({"flat_feature_index":2,"borders":[5,6],"nan_value_treatment":"AsTrue"})"},
      {make_tree({make_split(2, "6")}, "1,2")});
  const auto g = boxwood::ObliviousTreeEnsemble::from_catboost_json(with_gap);
  const auto gs = g.splits(0);
  CHECK(gs.size() == 1);
  CHECK(gs[0].source == 0 && gs[0].index == 2 && gs[0].kind == 0 && gs[0].arg == 1);  // x2 > 6
}

// ----------------------------------------------------------- fit tests

static void test_fit() {
  auto e_reg = boxwood::ObliviousTreeEnsemble::from_catboost_json(small_model());
  // 4 observations over the 2 features of small_model; targets and weights of size n.
  const std::vector<float> X = {0.f, 0.f, 1.f, 1.f, -1.f, -1.f, 0.5f, 3.f};
  const std::vector<double> y = {0.0, 1.0, 2.0, 3.0};
  const std::vector<double> w = {1.0, 1.0, 1.0, 1.0};
  boxwood::Batch b{std::span<const float>(X), 4, {}, std::span<const double>(y),
                  std::span<const double>(w)};
  boxwood::Batch none{};

  boxwood::FitOptions opt;
  opt.lambda = 1.0;
  opt.stop.max_iter = 20;
  const boxwood::FitReport r = e_reg.fit(b, none, opt);
  // The refit never raises the objective; the anchor is the start.
  CHECK(r.objective <= r.objective_start + 1e-9);
  CHECK(std::isnan(r.holdout_loss));  // no holdout
  CHECK(r.iterations >= 1);

  // Errors: wrong y size, wrong weight size, negative weight, lambda < 0,
  // use_holdout without holdout observations.
  boxwood::Batch bad_y = b;
  const std::vector<double> y_bad = {0.0, 1.0, 2.0};
  bad_y.y = std::span<const double>(y_bad);
  CHECK_THROWS(e_reg.fit(bad_y, none, opt));

  boxwood::Batch bad_w = b;
  const std::vector<double> w_bad = {1.0, 1.0};
  bad_w.weight = std::span<const double>(w_bad);
  CHECK_THROWS(e_reg.fit(bad_w, none, opt));

  boxwood::Batch neg_w = b;
  const std::vector<double> w_neg = {1.0, -1.0, 1.0, 1.0};
  neg_w.weight = std::span<const double>(w_neg);
  CHECK_THROWS(e_reg.fit(neg_w, none, opt));

  boxwood::FitOptions neg_lam = opt;
  neg_lam.lambda = -1.0;
  CHECK_THROWS(e_reg.fit(b, none, neg_lam));

  boxwood::FitOptions need_hold = opt;
  need_hold.stop.use_holdout = true;
  CHECK_THROWS(e_reg.fit(b, none, need_hold));

  // A holdout with observations gives a finite holdout loss.
  boxwood::Batch hold{std::span<const float>(X), 4, {}, std::span<const double>(y),
                     std::span<const double>(w)};
  need_hold.stop.max_iter = 5;
  const boxwood::FitReport rh = e_reg.fit(b, hold, need_hold);
  CHECK(std::isfinite(rh.holdout_loss));
}

// ----------------------------------------------------------- prune tests

static void test_level_basis() {
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(-100.0, 100.0);
  for (uint32_t d = 0; d <= 8; ++d) {
    const size_t n = size_t{1} << d;
    std::vector<double> w(n);
    for (double& x : w) x = dist(rng);
    const std::vector<double> u = boxwood::to_level_basis(w);
    CHECK(u.size() == 2 * n - 1);
    // Round trip at full depth.
    const std::vector<double> back = boxwood::from_level_basis(u, d);
    for (size_t j = 0; j < n; ++j) CHECK(near(back[j], w[j], 1e-9));
    // Cutting to k levels gives the mean of the leaves in each cut group.
    for (uint32_t k = 0; k <= d; ++k) {
      const std::vector<double> cut = boxwood::from_level_basis(u, k);
      const size_t m = size_t{1} << k;
      CHECK(cut.size() == m);
      for (size_t p = 0; p < m; ++p) {
        double sum = 0.0;
        size_t cnt = 0;
        for (size_t j = p; j < n; j += m) {
          sum += w[j];
          ++cnt;
        }
        CHECK(near(cut[p], sum / static_cast<double>(cnt), 1e-9));
      }
    }
  }
}

static void test_choose_levels() {
  // Tree A: g = (0, 10, 11): segments of slope 10 and slope 1, length 1 each.
  // Tree B: g = (0, 1, 9): not concave; its hull is one segment, slope 4.5, length 2.
  const std::vector<std::vector<double>> g = {{0.0, 10.0, 11.0}, {0.0, 1.0, 9.0}};
  const boxwood::LevelCost levels = {{0, 1, 2}, {0, 1, 2}};
  auto eq = [](std::vector<uint32_t> a, std::vector<uint32_t> b) { return a == b; };
  CHECK(eq(boxwood::choose_levels(g, levels, 0), {0u, 0u}));
  CHECK(eq(boxwood::choose_levels(g, levels, 1), {1u, 0u}));  // B's segment does not fit, the greedy stops
  CHECK(eq(boxwood::choose_levels(g, levels, 2), {1u, 0u}));
  CHECK(eq(boxwood::choose_levels(g, levels, 3), {1u, 2u}));
  CHECK(eq(boxwood::choose_levels(g, levels, 4), {2u, 2u}));
  CHECK(eq(boxwood::choose_levels(g, levels, 9), {2u, 2u}));  // budget beyond the total depth
  CHECK(boxwood::choose_levels({}, {}, 5).empty());
  // Byte-like costs: A's second level costs 8, so per cost B's hull (slope 9 / 6 = 1.5) beats
  // it (1 / 8); A's first level (10 / 2) still comes first.
  const boxwood::LevelCost bytes = {{0, 2, 10}, {0, 3, 6}};
  CHECK(eq(boxwood::choose_levels(g, bytes, 2), {1u, 0u}));
  CHECK(eq(boxwood::choose_levels(g, bytes, 8), {1u, 2u}));
  CHECK(eq(boxwood::choose_levels(g, bytes, 15), {1u, 2u}));  // A's 8 bytes do not fit into 7
  CHECK(eq(boxwood::choose_levels(g, bytes, 16), {2u, 2u}));
}

int main() {
  test_json();
  test_cityhash();
  test_encoder_oracle();
  test_splits();
  test_many_trees();
  test_categorical();
  test_binning_255();
  test_backends();
  test_threads();
  test_int8_quantization();
  test_serialization();
  test_categorical_serialization();
  test_errors();
  test_predict_links();
  test_fit();
  test_level_basis();
  test_choose_levels();
  if (g_failures != 0) {
    std::printf("%d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("all tests passed\n");
  return 0;
}
