// to_bytes / from_bytes: a small header, then every stored array with its length,
// floats copied raw so a round trip is bit-exact. from_bytes checks every count
// and index before the model is used.

#include "boxwood/ensemble.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <numeric>

#include "kernels.hpp"  // kInt8TreeBlock

namespace boxwood {

static_assert(std::endian::native == std::endian::little, "boxwood stores little-endian bytes");

namespace {

using Magic = std::array<char, 4>;
constexpr Magic kMagic = {'B', 'X', 'W', 'D'};
// Version 2: int8 steps are shared per block of trees (see quantize_leaves).
constexpr uint32_t kVersion = 2;

/// Appends a scalar raw, or an array as its length and its raw elements.
template <class T>
void put(std::vector<std::byte>& out, const T& v) {
  const std::byte* p = reinterpret_cast<const std::byte*>(&v);
  out.insert(out.end(), p, p + sizeof(T));
}

template <class T>
void put(std::vector<std::byte>& out, const std::vector<T>& a) {
  put<uint64_t>(out, static_cast<uint64_t>(a.size()));
  if (!a.empty()) {
    const std::byte* p = reinterpret_cast<const std::byte*>(a.data());
    out.insert(out.end(), p, p + a.size() * sizeof(T));
  }
}

/// Reads the serialized stream: little-endian scalars and length-prefixed arrays.
struct Reader {
  const std::byte* p;
  size_t rem;

  template <class T>
  T scalar() {
    BW_CHECK(rem >= sizeof(T), "truncated bytes");
    T v;
    std::memcpy(&v, p, sizeof(T));
    p += sizeof(T);
    rem -= sizeof(T);
    return v;
  }

  template <class T>
  std::vector<T> array() {
    const uint64_t count = scalar<uint64_t>();
    BW_CHECK(count <= rem / sizeof(T), "array does not fit the remaining bytes");
    std::vector<T> a(static_cast<size_t>(count));
    if (!a.empty()) {
      std::memcpy(a.data(), p, a.size() * sizeof(T));
      p += a.size() * sizeof(T);
      rem -= a.size() * sizeof(T);
    }
    return a;
  }

  /// Reads what put wrote for `v`.
  template <class T>
  void read(T& v) {
    v = scalar<T>();
  }
  template <class T>
  void read(std::vector<T>& a) {
    a = array<T>();
  }
};

}  // namespace

template <class Self, class F>
void ObliviousTreeEnsemble::for_each_stored(Self& e, F f) {
  f(e.numeric_flat_index_);
  f(e.categorical_flat_index_);
  f(e.numeric_column_);
  f(e.border_count_);
  f(e.nan_high_);
  f(e.borders_);
  f(e.onehot_column_);
  f(e.onehot_count_);
  f(e.onehot_values_);
  f(e.proj_size_);
  f(e.elem_kind_);
  f(e.elem_column_);
  f(e.elem_border_);
  f(e.elem_nan_high_);
  f(e.elem_value_);
  f(e.table_projection_);
  f(e.table_size_);
  f(e.table_keys_);
  f(e.ctr_table_);
  f(e.ctr_default_code_);
  f(e.ctr_codes_);
  f(e.depth_);
  f(e.split_feature_);
  f(e.split_kind_);
  f(e.split_arg_);
  if (e.leaf_type_ == LeafType::Int8) {
    f(e.leaves_q_);
    f(e.leaf_scale_);
  } else {
    f(e.leaves_);
  }
  f(e.bias_);
  // Fit state: the lambda, sigma, optimizer and learning rate of the last fit
  // and the Adam state it ended with.
  f(e.fit_lambda_);
  f(e.fit_sigma_);
  f(e.fit_optimizer_);
  f(e.fit_learning_rate_);
  f(e.adam_t_);
  f(e.adam_m_);
  f(e.adam_v_);
}

std::vector<std::byte> ObliviousTreeEnsemble::to_bytes() const {
  std::vector<std::byte> out;
  put(out, kMagic);
  put<uint32_t>(out, kVersion);
  put(out, static_cast<uint8_t>(task_));
  put(out, static_cast<uint8_t>(leaf_type_));
  put<uint32_t>(out, K_);
  put<uint32_t>(out, n_numeric_);
  put<uint32_t>(out, n_categorical_);
  for_each_stored(*this, [&](const auto& member) { put(out, member); });
  return out;
}

namespace {

template <class T>
uint64_t sum_of(const std::vector<T>& counts) {
  return std::accumulate(counts.begin(), counts.end(), uint64_t{0});
}

}  // namespace

void ObliviousTreeEnsemble::check_arrays() const {
  const size_t F = numeric_column_.size();
  const size_t G = onehot_column_.size();
  const size_t P = proj_size_.size();
  const size_t Q = table_projection_.size();
  const size_t R = ctr_table_.size();
  const size_t N = F + G + R;  // all split features

  BW_CHECK(numeric_flat_index_.size() == n_numeric_, "numeric_flat_index does not match n_numeric");
  BW_CHECK(categorical_flat_index_.size() == n_categorical_,
           "categorical_flat_index does not match n_categorical");

  BW_CHECK(border_count_.size() == F && nan_high_.size() == F,
           "numeric encoder arrays must have equal length");
  for (size_t f = 0; f < F; ++f) {
    BW_CHECK(border_count_[f] <= 255, "border_count exceeds 255");
    BW_CHECK(numeric_column_[f] < n_numeric_, "numeric_column out of range");
  }
  BW_CHECK(sum_of(border_count_) == borders_.size(), "border counts do not match the borders array");

  BW_CHECK(onehot_count_.size() == G, "onehot_count does not match the one-hot features");
  for (size_t g = 0; g < G; ++g) BW_CHECK(onehot_column_[g] < n_categorical_, "onehot_column out of range");
  BW_CHECK(sum_of(onehot_count_) == onehot_values_.size(), "one-hot counts do not match the values array");

  BW_CHECK(elem_kind_.size() == elem_column_.size() && elem_border_.size() == elem_kind_.size() &&
               elem_nan_high_.size() == elem_kind_.size() && elem_value_.size() == elem_kind_.size(),
           "projection element arrays must have equal length");
  BW_CHECK(sum_of(proj_size_) == elem_kind_.size(), "projection sizes do not match the element arrays");
  for (size_t e = 0; e < elem_kind_.size(); ++e) {
    BW_CHECK(elem_kind_[e] <= 2, "elem_kind out of range");
    const uint32_t limit = elem_kind_[e] == 1 ? n_numeric_ : n_categorical_;
    BW_CHECK(elem_column_[e] < limit, "elem_column out of range");
  }

  BW_CHECK(table_size_.size() == Q, "table_size does not match the tables");
  for (size_t q = 0; q < Q; ++q) BW_CHECK(table_projection_[q] < P, "table_projection out of range");
  BW_CHECK(sum_of(table_size_) == table_keys_.size(), "table sizes do not match the keys array");
  uint64_t key_off = 0;
  for (size_t q = 0; q < Q; ++q) {
    for (size_t j = 1; j < table_size_[q]; ++j) {
      BW_CHECK(table_keys_[key_off + j - 1] < table_keys_[key_off + j], "table keys not ascending");
    }
    key_off += table_size_[q];
  }

  BW_CHECK(ctr_default_code_.size() == R, "ctr_default_code does not match the CTRs");
  uint64_t code_sum = 0;
  for (size_t r = 0; r < R; ++r) {
    BW_CHECK(ctr_table_[r] < Q, "ctr_table out of range");
    code_sum += table_size_[ctr_table_[r]];
  }
  BW_CHECK(code_sum == ctr_codes_.size(), "CTR codes do not match the table sizes");

  const size_t T = depth_.size();
  for (size_t t = 0; t < T; ++t) BW_CHECK(depth_[t] <= 8, "tree depth exceeds 8");
  const uint64_t split_sum = sum_of(depth_);
  BW_CHECK(split_sum == split_feature_.size() && split_kind_.size() == split_sum &&
               split_arg_.size() == split_sum,
           "split arrays do not match the tree depths");
  for (size_t s = 0; s < split_feature_.size(); ++s) {
    BW_CHECK(split_feature_[s] < N, "split_feature out of range");
    BW_CHECK(split_kind_[s] <= 1, "split_kind out of range");
  }

  uint64_t leaf_sum = 0;
  for (size_t t = 0; t < T; ++t) leaf_sum += (uint64_t{1} << depth_[t]) * K_;
  if (leaf_type_ == LeafType::Int8) {
    BW_CHECK(leaf_sum == leaves_q_.size(), "leaf counts do not match the int8 leaves array");
    BW_CHECK(leaf_scale_.size() == T, "leaf_scale does not match the number of trees");
    // The kernels sum a block's codes as integers: every step of a block must
    // be the block's largest step / 2^d with d in 0 .. 7.
    for (size_t t0 = 0; t0 < T; t0 += kInt8TreeBlock) {
      const size_t t1 = std::min<size_t>(t0 + kInt8TreeBlock, T);
      const float s = *std::max_element(leaf_scale_.begin() + t0, leaf_scale_.begin() + t1);
      BW_CHECK(std::isfinite(s) && s > 0.0f, "an int8 step is not positive and finite");
      for (size_t t = t0; t < t1; ++t) {
        bool ok = false;
        for (int d = 0; d <= 7; ++d) ok = ok || leaf_scale_[t] == std::ldexp(s, -d);
        BW_CHECK(ok, "an int8 step is not its block's largest step / 2^d, d <= 7");
      }
    }
  } else {
    BW_CHECK(leaf_sum == leaves_.size(), "leaf counts do not match the leaves array");
  }
  BW_CHECK(bias_.size() == K_, "bias size does not match K");
  BW_CHECK(fit_optimizer_ <= 2, "unknown optimizer of the last fit");
  BW_CHECK(adam_m_.size() == adam_v_.size(), "adam moments must have equal length");
}

ObliviousTreeEnsemble ObliviousTreeEnsemble::from_bytes(std::span<const std::byte> bytes) {
  Reader r{bytes.data(), bytes.size()};
  ObliviousTreeEnsemble e;

  BW_CHECK(r.scalar<Magic>() == kMagic, "bad magic: not a boxwood model");
  BW_CHECK(r.scalar<uint32_t>() == kVersion, "unsupported version");
  const uint8_t task = r.scalar<uint8_t>();
  BW_CHECK(task <= 2, "bad task value");
  const uint8_t leaf_type = r.scalar<uint8_t>();
  BW_CHECK(leaf_type <= 1, "unsupported leaf type");
  e.leaf_type_ = static_cast<LeafType>(leaf_type);
  e.K_ = r.scalar<uint32_t>();
  e.n_numeric_ = r.scalar<uint32_t>();
  e.n_categorical_ = r.scalar<uint32_t>();
  BW_CHECK(e.K_ >= 1, "K must be >= 1");
  BW_CHECK(task == 2 || e.K_ == 1, "K must be 1 for regression and binary");
  e.task_ = static_cast<Task>(task);

  for_each_stored(e, [&](auto& member) { r.read(member); });
  BW_CHECK(r.rem == 0, "trailing bytes after the model");

  e.check_arrays();
  e.rebuild_derived();
  return e;
}

}  // namespace boxwood
