#include "boxwood/ensemble.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <numeric>
#include <string>
#include <thread>
#include <tuple>

#include "backend.hpp"
#include "cityhash.hpp"
#include "kernels.hpp"

namespace boxwood {

namespace {

/// A partial vector group of at least this many observations is padded to a
/// full group instead of running through the scalar code.
constexpr size_t kMinPadded = 8;
/// Prediction starts a thread only for at least this many blocks.
constexpr size_t kMinBlocksPerThread = 4;

/// begin[i] = sum of size(j) for j < i, for i = 0..n.
template <class Size>
std::vector<uint32_t> prefix_sums(size_t n, Size size) {
  std::vector<uint32_t> begin(n + 1, 0);
  for (size_t i = 0; i < n; ++i) begin[i + 1] = begin[i] + static_cast<uint32_t>(size(i));
  return begin;
}

/// The int8 quantization of the leaf values w (all trees, leaf_begin gives
/// each tree's range), the only place the int8 rule lives. The trees form
/// blocks of kInt8TreeBlock; M = the largest |w| of a block, s = M / 127 (1 if
/// M is 0). Tree t of the block gets the step s / 2^d with the largest d <= 7
/// for which its largest |w| is at most M / 2^d, so a tree with small leaves
/// gets a finer step, and the codes round-half-away-from-zero(w / step)
/// clamped to [-127, 127]. Writes the codes to `q` and the steps to `step`.
void quantize_leaves(std::span<const float> w, std::span<const uint32_t> leaf_begin,
                     std::vector<int8_t>& q, std::vector<float>& step) {
  const size_t T = leaf_begin.size() - 1;
  std::vector<float> tree_max(T, 0.0f);
  for (size_t t = 0; t < T; ++t) {
    for (size_t j = leaf_begin[t]; j < leaf_begin[t + 1]; ++j) {
      tree_max[t] = std::max(tree_max[t], std::fabs(w[j]));
    }
  }
  q.resize(w.size());
  step.resize(T);
  for (size_t t0 = 0; t0 < T; t0 += kInt8TreeBlock) {
    const size_t t1 = std::min<size_t>(t0 + kInt8TreeBlock, T);
    const float M = *std::max_element(tree_max.begin() + t0, tree_max.begin() + t1);
    const float s = M > 0.0f ? M / 127.0f : 1.0f;
    for (size_t t = t0; t < t1; ++t) {
      int d = 0;
      while (d < 7 && tree_max[t] <= std::ldexp(M, -(d + 1))) ++d;
      step[t] = std::ldexp(s, -d);
      for (size_t j = leaf_begin[t]; j < leaf_begin[t + 1]; ++j) {
        const long r = std::lround(w[j] / step[t]);  // rounds half away from zero
        q[j] = static_cast<int8_t>(std::min<long>(std::max<long>(r, -127), 127));
      }
    }
  }
}

/// Checks the categorical columns of X (n_categorical spans of n values, or
/// dictionary-encoded: n_categorical spans of n codes, each less than its
/// dictionary's size) and returns the hash of every dictionary entry (empty
/// when X is not dictionary-encoded), so each distinct value is hashed once.
std::vector<std::vector<int32_t>> dictionary_hashes(const Batch& X, size_t n_categorical) {
  BW_CHECK(X.categorical.size() == n_categorical, "X.categorical size does not match n_categorical");
  std::vector<std::vector<int32_t>> hashes;
  if (X.categorical_codes.empty()) {
    for (const std::span<const std::string_view>& col : X.categorical) {
      BW_CHECK(col.size() == X.n, "a categorical column does not have n values");
    }
    return hashes;
  }
  BW_CHECK(X.categorical_codes.size() == n_categorical,
           "X.categorical_codes size does not match n_categorical");
  for (size_t c = 0; c < n_categorical; ++c) {
    const std::span<const uint32_t> codes = X.categorical_codes[c];
    const size_t size = X.categorical[c].size();
    BW_CHECK(codes.size() == X.n, "a categorical code column does not have n codes");
    BW_CHECK(std::all_of(codes.begin(), codes.end(), [&](uint32_t code) { return code < size; }),
             "a categorical code is not less than its dictionary's size");
    hashes.emplace_back();
    for (const std::string_view value : X.categorical[c]) {
      hashes.back().push_back(static_cast<int32_t>(cat_hash(value)));
    }
  }
  return hashes;
}

}  // namespace

void ObliviousTreeEnsemble::rebuild_derived() {
  border_begin_ = prefix_sums(border_count_.size(), [&](size_t f) { return border_count_[f]; });
  onehot_value_begin_ = prefix_sums(onehot_count_.size(), [&](size_t g) { return onehot_count_[g]; });
  proj_elem_begin_ = prefix_sums(proj_size_.size(), [&](size_t p) { return proj_size_[p]; });
  build_ctr_hash_tables();
  split_begin_ = prefix_sums(depth_.size(), [&](size_t t) { return depth_[t]; });
  build_split_conditions();
  leaf_begin_ = prefix_sums(depth_.size(), [&](size_t t) { return (size_t{1} << depth_[t]) * K_; });

  // The binning of the kernels walks a complete binary search tree of 8
  // levels per feature, stored breadth-first (see scalar::bin_code): the
  // borders padded with +inf to 255 sorted values; node c of level l is the
  // sorted value at (2c + 1) * 2^(7-l) - 1, the middle of its range. Entry
  // 255 only pads the table to 256 floats.
  border_heap_.assign(numeric_column_.size() * 256, std::numeric_limits<float>::infinity());
  for (size_t f = 0; f < numeric_column_.size(); ++f) {
    const size_t b0 = border_begin_[f];
    const size_t nb = border_begin_[f + 1] - b0;
    for (size_t l = 0; l < 8; ++l) {
      const size_t step = size_t{128} >> l;
      for (size_t c = 0; c < (size_t{1} << l); ++c) {
        const size_t pos = (2 * c + 1) * step - 1;
        if (pos < nb) border_heap_[f * 256 + (size_t{1} << l) - 1 + c] = borders_[b0 + pos];
      }
    }
  }

  // The leaf tables of the SIMD backends: for each tree and output, that
  // output's leaf values (or int8 codes) contiguous, padded with zeros to at
  // least 16 entries so a register lookup reads a full vector.
  leaf_out_begin_ = prefix_sums(depth_.size() * K_,
                               [&](size_t i) { return std::max<size_t>(size_t{1} << depth_[i / K_], 16); });
  const bool int8 = leaf_type_ == LeafType::Int8;
  leaves_by_output_.assign(int8 ? 0 : leaf_out_begin_.back(), 0.0f);
  leaves_q_by_output_.assign(int8 ? leaf_out_begin_.back() : 0, 0);
  for (size_t t = 0; t < depth_.size(); ++t) {
    for (size_t k = 0; k < K_; ++k) {
      const size_t out = leaf_out_begin_[t * K_ + k];
      for (size_t j = 0; j < (size_t{1} << depth_[t]); ++j) {
        const size_t idx = leaf_begin_[t] + j * K_ + k;
        if (int8) {
          leaves_q_by_output_[out + j] = leaves_q_[idx];
        } else {
          leaves_by_output_[out + j] = leaves_[idx];
        }
      }
    }
  }

  // Int8: the unit of each block of trees is its largest step / 128, and
  // tree t's multiplier is step[t] / unit (a power of two in 1 .. 128 by the
  // quantization rule, checked when bytes are loaded).
  tree_mult_.clear();
  block_unit_.clear();
  if (int8) {
    for (size_t t0 = 0; t0 < depth_.size(); t0 += kInt8TreeBlock) {
      const size_t t1 = std::min<size_t>(t0 + kInt8TreeBlock, depth_.size());
      const float largest = *std::max_element(leaf_scale_.begin() + t0, leaf_scale_.begin() + t1);
      const double unit = static_cast<double>(largest) / 128.0;
      block_unit_.push_back(unit);
      for (size_t t = t0; t < t1; ++t) tree_mult_.push_back(static_cast<uint8_t>(leaf_scale_[t] / unit));
    }
  }
}

void ObliviousTreeEnsemble::build_split_conditions() {
  struct Cond {
    uint32_t feature;
    uint8_t kind, arg;
    auto operator<=>(const Cond&) const = default;
  };
  std::vector<Cond> conds(split_feature_.size());
  for (size_t s = 0; s < conds.size(); ++s) conds[s] = {split_feature_[s], split_kind_[s], split_arg_[s]};
  std::sort(conds.begin(), conds.end());
  conds.erase(std::unique(conds.begin(), conds.end()), conds.end());
  cond_feature_.resize(conds.size());
  cond_kind_.resize(conds.size());
  cond_arg_.resize(conds.size());
  for (size_t c = 0; c < conds.size(); ++c) {
    cond_feature_[c] = conds[c].feature;
    cond_kind_[c] = conds[c].kind;
    cond_arg_[c] = conds[c].arg;
  }
  split_cond_.resize(split_feature_.size());
  for (size_t s = 0; s < split_cond_.size(); ++s) {
    const Cond key{split_feature_[s], split_kind_[s], split_arg_[s]};
    split_cond_[s] = static_cast<uint32_t>(std::lower_bound(conds.begin(), conds.end(), key) - conds.begin());
  }
}

void ObliviousTreeEnsemble::build_ctr_hash_tables() {
  const size_t Q = table_size_.size();
  const size_t R = ctr_table_.size();
  // Each table has a power of two of more than twice its keys slots: never
  // full, and at most half full, so most keys sit in their first slot (see
  // ctr_slots). The empty key is the smallest value that is not a key (the
  // keys are ascending).
  table_slot_begin_ = prefix_sums(Q, [&](size_t q) { return std::bit_ceil(table_size_[q] * 2 + 1); });
  table_slot_keys_.resize(table_slot_begin_.back());
  table_empty_key_.resize(Q);
  // pos[s] = the position of the key of slot s in its table's ascending keys
  // (where the codes of the CTRs are), kNone at free slots.
  constexpr uint32_t kNone = 0xFFFFFFFFu;
  std::vector<uint32_t> pos(table_slot_begin_.back(), kNone);
  std::vector<uint32_t> key_begin = prefix_sums(Q, [&](size_t q) { return table_size_[q]; });
  for (size_t q = 0; q < Q; ++q) {
    const uint64_t* keys = table_keys_.data() + key_begin[q];
    uint64_t empty = 0;
    for (uint32_t j = 0; j < table_size_[q] && keys[j] == empty; ++j) ++empty;
    table_empty_key_[q] = empty;
    uint64_t* slots = table_slot_keys_.data() + table_slot_begin_[q];
    const uint32_t mask = table_slot_begin_[q + 1] - table_slot_begin_[q] - 1;
    std::fill(slots, slots + mask + 1, empty);
    for (uint32_t j = 0; j < table_size_[q]; ++j) {
      uint32_t s = ctr_slot(keys[j], mask);
      while (slots[s] != empty) s = (s + 1) & mask;
      slots[s] = keys[j];
      pos[table_slot_begin_[q] + s] = j;
    }
  }
  // The hashing order of the tables (see OTEView::table_order): sorted by
  // the elements of their projections, then the shared prefix lengths.
  const auto elem = [&](uint32_t e) {
    return std::tuple(elem_kind_[e], elem_column_[e], elem_border_[e], elem_nan_high_[e], elem_value_[e]);
  };
  const auto elems_of = [&](uint32_t q) {
    const uint32_t p = table_projection_[q];
    return std::pair(proj_elem_begin_[p], proj_elem_begin_[p + 1]);
  };
  table_order_.resize(Q);
  std::iota(table_order_.begin(), table_order_.end(), 0u);
  std::stable_sort(table_order_.begin(), table_order_.end(), [&](uint32_t a, uint32_t b) {
    const auto [a0, a1] = elems_of(a);
    const auto [b0, b1] = elems_of(b);
    for (uint32_t i = 0; a0 + i < a1 && b0 + i < b1; ++i) {
      if (elem(a0 + i) != elem(b0 + i)) return elem(a0 + i) < elem(b0 + i);
    }
    return a1 - a0 < b1 - b0;
  });
  table_shared_.assign(Q, 0);
  for (size_t j = 1; j < Q; ++j) {
    const auto [a0, a1] = elems_of(table_order_[j - 1]);
    const auto [b0, b1] = elems_of(table_order_[j]);
    uint32_t n = 0;
    while (a0 + n < a1 && b0 + n < b1 && elem(a0 + n) == elem(b0 + n)) ++n;
    table_shared_[j] = n;
  }

  // The CTRs grouped by table (a counting sort), and each CTR's code per slot
  // of its table: the stored code of the slot's key, the default code at a
  // free slot.
  std::vector<uint32_t> fill(Q, 0);
  for (const uint32_t q : ctr_table_) ++fill[q];
  table_ctr_begin_ = prefix_sums(Q, [&](size_t q) { return fill[q]; });
  std::copy(table_ctr_begin_.begin(), table_ctr_begin_.end() - 1, fill.begin());
  table_ctrs_.resize(R);
  for (uint32_t r = 0; r < R; ++r) table_ctrs_[fill[ctr_table_[r]]++] = r;
  ctr_slot_code_begin_ = prefix_sums(R, [&](size_t r) {
    const uint32_t q = ctr_table_[r];
    return table_slot_begin_[q + 1] - table_slot_begin_[q];
  });
  ctr_slot_codes_.resize(ctr_slot_code_begin_.back());
  std::vector<uint32_t> code_begin = prefix_sums(R, [&](size_t r) { return table_size_[ctr_table_[r]]; });
  for (uint32_t r = 0; r < R; ++r) {
    const uint32_t q = ctr_table_[r];
    for (uint32_t s = table_slot_begin_[q]; s < table_slot_begin_[q + 1]; ++s) {
      ctr_slot_codes_[ctr_slot_code_begin_[r] + s - table_slot_begin_[q]] =
          pos[s] != kNone ? ctr_codes_[code_begin[r] + pos[s]] : ctr_default_code_[r];
    }
  }
}

OTEView ObliviousTreeEnsemble::view() const {
  OTEView v;
  v.K = K_;
  v.n_trees = static_cast<uint32_t>(depth_.size());
  v.n_numeric_features = static_cast<uint32_t>(numeric_column_.size());
  v.n_features = v.n_numeric_features + static_cast<uint32_t>(onehot_column_.size()) +
                 static_cast<uint32_t>(ctr_table_.size());
  v.n_numeric = n_numeric_;
  v.n_categorical = n_categorical_;
  v.numeric_column = numeric_column_.data();
  v.border_begin = border_begin_.data();
  v.border_heap = border_heap_.data();
  v.nan_high = nan_high_.data();
  v.n_onehot = static_cast<uint32_t>(onehot_column_.size());
  v.onehot_column = onehot_column_.data();
  v.onehot_value_begin = onehot_value_begin_.data();
  v.onehot_values = onehot_values_.data();
  v.n_projections = static_cast<uint32_t>(proj_size_.size());
  v.proj_elem_begin = proj_elem_begin_.data();
  v.elem_kind = elem_kind_.data();
  v.elem_column = elem_column_.data();
  v.elem_border = elem_border_.data();
  v.elem_nan_high = elem_nan_high_.data();
  v.elem_value = elem_value_.data();
  v.n_tables = static_cast<uint32_t>(table_projection_.size());
  v.table_projection = table_projection_.data();
  v.table_slot_begin = table_slot_begin_.data();
  v.table_slot_keys = table_slot_keys_.data();
  v.table_empty_key = table_empty_key_.data();
  v.table_ctr_begin = table_ctr_begin_.data();
  v.table_ctrs = table_ctrs_.data();
  v.table_order = table_order_.data();
  v.table_shared = table_shared_.data();
  v.n_ctr = static_cast<uint32_t>(ctr_table_.size());
  v.ctr_table = ctr_table_.data();
  v.ctr_slot_code_begin = ctr_slot_code_begin_.data();
  v.ctr_slot_codes = ctr_slot_codes_.data();
  v.depth = depth_.data();
  v.split_begin = split_begin_.data();
  v.n_conds = static_cast<uint32_t>(cond_feature_.size());
  v.cond_feature = cond_feature_.data();
  v.cond_kind = cond_kind_.data();
  v.cond_arg = cond_arg_.data();
  v.split_cond = split_cond_.data();
  v.leaf_type = leaf_type_;
  v.tree_mult = tree_mult_.data();
  v.block_unit = block_unit_.data();
  v.leaves_by_output = leaves_by_output_.data();
  v.leaves_q_by_output = leaves_q_by_output_.data();
  v.leaf_out_begin = leaf_out_begin_.data();
  return v;
}

template <class T, class Kernel>
void ObliviousTreeEnsemble::for_each_block(const Batch& X, std::span<T> out, size_t out_per_obs,
                                          Kernel kernel) const {
  BW_CHECK(X.numeric.size() == X.n * n_numeric_, "X.numeric size does not match n * n_numeric");
  const bool dict = !X.categorical_codes.empty();
  const std::vector<std::vector<int32_t>> dict_hash = dictionary_hashes(X, n_categorical_);
  BW_CHECK(out.size() == X.n * out_per_obs, "output size does not match the number of observations");
  const Backend& be = backend_ != nullptr ? *backend_ : best_backend();
  const OTEView v = view();
  // Observations are encoded in blocks of kBlockSize, so the codes stay in cache for the
  // kernel. Each block writes only its own observations of the output, so threads can
  // take contiguous ranges of blocks, each with its own buffers.
  const auto run = [&](size_t block_begin, size_t block_end) {
    std::vector<uint8_t> codes;
    std::vector<int32_t> hashes;
    std::vector<float> x_pad;
    std::vector<T> out_pad;
    for (size_t i0 = block_begin * kBlockSize; i0 < std::min(X.n, block_end * kBlockSize); i0 += kBlockSize) {
      const size_t b = std::min(kBlockSize, X.n - i0);
      const float* Xb = X.numeric.data() + i0 * n_numeric_;
      T* out_b = out.data() + i0 * out_per_obs;
      // The SIMD kernels run observations that do not fill a vector group through the
      // scalar code, which is many times slower per observation. A partial group of
      // at least 8 observations is cheaper as a full group: the block is padded with
      // zero observations, and only the real observations are copied out.
      const size_t partial = b % be.group_size;
      const size_t n_obs = partial >= kMinPadded ? b + be.group_size - partial : b;
      if (n_obs != b) {
        x_pad.assign(n_obs * n_numeric_, 0.0f);
        std::copy(Xb, Xb + b * n_numeric_, x_pad.begin());
        Xb = x_pad.data();
        out_pad.assign(n_obs * out_per_obs, T{});
        std::copy(out_b, out_b + b * out_per_obs, out_pad.begin());
      }
      codes.resize(n_obs * v.n_features);
      be.encode_numeric(v, Xb, n_obs, codes.data());
      hashes.assign(n_obs * n_categorical_, 0);
      for (size_t c = 0; c < n_categorical_; ++c) {
        for (size_t i = 0; i < b; ++i) {
          hashes[i * n_categorical_ + c] = dict ? dict_hash[c][X.categorical_codes[c][i0 + i]]
                                                : static_cast<int32_t>(cat_hash(X.categorical[c][i0 + i]));
        }
      }
      scalar::encode_categorical(v, Xb, hashes.data(), n_obs, be.ctr_lookup, codes.data());
      kernel(be, v, codes.data(), n_obs, n_obs != b ? out_pad.data() : out_b);
      if (n_obs != b) std::copy(out_pad.begin(), out_pad.begin() + b * out_per_obs, out_b);
    }
  };
  // At least 4 blocks per thread, so starting a thread costs little next to
  // its work.
  const size_t blocks = (X.n + kBlockSize - 1) / kBlockSize;
  const size_t n_threads = std::clamp<size_t>(blocks / kMinBlocksPerThread, 1, threads_);
  if (n_threads == 1) {
    run(0, blocks);
    return;
  }
  std::vector<std::jthread> pool;
  for (size_t t = 0; t < n_threads; ++t) {
    pool.emplace_back(run, blocks * t / n_threads, blocks * (t + 1) / n_threads);
  }
}

void ObliviousTreeEnsemble::predict_margin(const Batch& X, std::span<double> out) const {
  for (size_t i = 0; i < out.size(); i += K_) std::copy(bias_.begin(), bias_.end(), out.begin() + i);
  for_each_block(X, out, K_, [](const Backend& be, const OTEView& v, const uint8_t* codes, size_t n_obs,
                                double* margins) { be.accumulate(v, codes, n_obs, margins); });
}

void ObliviousTreeEnsemble::predict(const Batch& X, std::span<double> out) const {
  predict_margin(X, out);
  switch (task_) {
    case Task::Regression:
      break;
    case Task::Binary:
      for (double& m : out) m = 1.0 / (1.0 + std::exp(-m));
      break;
    case Task::Multiclass:
      for (size_t i = 0; i < X.n; ++i) {
        double* m = out.data() + i * K_;
        const double mx = *std::max_element(m, m + K_);
        double sum = 0.0;
        for (uint32_t k = 0; k < K_; ++k) {
          m[k] = std::exp(m[k] - mx);
          sum += m[k];
        }
        for (uint32_t k = 0; k < K_; ++k) m[k] /= sum;
      }
      break;
  }
}

void ObliviousTreeEnsemble::leaf_index(const Batch& X, std::span<uint8_t> out) const {
  const size_t T = depth_.size();
  for_each_block(X, out, T, [](const Backend& be, const OTEView& v, const uint8_t* codes, size_t n_obs,
                               uint8_t* leaves) { be.leaf_index(v, codes, n_obs, leaves); });
}

void ObliviousTreeEnsemble::set_backend(std::string_view name) {
  if (name.empty()) {
    backend_ = nullptr;  // best_backend()
    return;
  }
  const Backend* b = find_backend(name);
  BW_CHECK(b != nullptr, "unknown or unavailable backend: " + std::string(name));
  backend_ = b;
}

void ObliviousTreeEnsemble::set_threads(unsigned n) {
  threads_ = n != 0 ? n : std::max(1u, std::thread::hardware_concurrency());
}

double ObliviousTreeEnsemble::leaf_value(size_t j, size_t tree) const {
  if (leaf_type_ == LeafType::Int8) return static_cast<double>(leaves_q_[j]) * leaf_scale_[tree];
  return leaves_[j];
}

std::vector<float> ObliviousTreeEnsemble::float_leaves() const {
  std::vector<float> w(leaf_begin_.back());
  for (size_t t = 0; t < depth_.size(); ++t) {
    for (size_t j = leaf_begin_[t]; j < leaf_begin_[t + 1]; ++j) w[j] = static_cast<float>(leaf_value(j, t));
  }
  return w;
}

void ObliviousTreeEnsemble::store_leaves(std::vector<float> w) {
  if (leaf_type_ == LeafType::Int8) {
    // The tree ranges of w follow depth_, which can be newer than leaf_begin_.
    const std::vector<uint32_t> begin =
        prefix_sums(depth_.size(), [&](size_t t) { return (size_t{1} << depth_[t]) * K_; });
    quantize_leaves(w, begin, leaves_q_, leaf_scale_);
    leaves_.clear();
  } else {
    leaves_ = std::move(w);
    leaves_q_.clear();
    leaf_scale_.clear();
  }
  rebuild_derived();
}

void ObliviousTreeEnsemble::set_leaf_type(LeafType type) {
  std::vector<float> w = float_leaves();
  leaf_type_ = type;
  store_leaves(std::move(w));
}


Info ObliviousTreeEnsemble::info() const {
  Info i;
  i.task = task_;
  i.n_outputs = K_;
  i.n_numeric = n_numeric_;
  i.n_categorical = n_categorical_;
  i.n_trees = static_cast<uint32_t>(depth_.size());
  i.n_splits = static_cast<uint32_t>(split_feature_.size());
  i.tree_bytes = 0;
  for (const uint8_t d : depth_) i.tree_bytes += tree_bytes(d);
  i.leaf_type = leaf_type_;
  i.numeric_flat_index = numeric_flat_index_;
  i.categorical_flat_index = categorical_flat_index_;
  i.fit_lambda = fit_lambda_;
  i.fit_sigma = fit_sigma_;
  i.fit_optimizer = std::isnan(fit_lambda_)
      ? std::string()
      : optimizer_name(static_cast<OptimizerKind>(fit_optimizer_));
  return i;
}

const char* optimizer_name(OptimizerKind kind) {
  switch (kind) {
    case OptimizerKind::Adam:
      return "adam";
    case OptimizerKind::GradientDescent:
      return "gradient_descent";
    case OptimizerKind::TreeNewton:
      break;
  }
  return "tree_newton";
}

std::vector<double> ObliviousTreeEnsemble::leaves(uint32_t tree) const {
  BW_CHECK(tree < depth_.size(), "tree index out of range");
  std::vector<double> out;
  for (size_t j = leaf_begin_[tree]; j < leaf_begin_[tree + 1]; ++j) out.push_back(leaf_value(j, tree));
  return out;
}

std::vector<double> ObliviousTreeEnsemble::bias() const { return bias_; }

std::vector<ObliviousTreeEnsemble::Split> ObliviousTreeEnsemble::splits(uint32_t tree) const {
  BW_CHECK(tree < depth_.size(), "tree index out of range");
  const uint32_t F = static_cast<uint32_t>(numeric_column_.size());
  const uint32_t G = static_cast<uint32_t>(onehot_column_.size());
  std::vector<Split> out;
  for (uint32_t s = split_begin_[tree]; s < split_begin_[tree + 1]; ++s) {
    // split_feature_ counts split features; report the source and the index of
    // the feature inside that source.
    const uint32_t f = split_feature_[s];
    if (f < F) {
      out.push_back(Split{0, numeric_column_[f], split_kind_[s], split_arg_[s]});
    } else if (f < F + G) {
      out.push_back(Split{1, onehot_column_[f - F], split_kind_[s], split_arg_[s]});
    } else {
      out.push_back(Split{2, f - F - G, split_kind_[s], split_arg_[s]});
    }
  }
  return out;
}

}  // namespace boxwood
