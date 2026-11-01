// The scalar reference backend: plain C++ for one block of observations.

#include <algorithm>
#include <cmath>
#include <vector>

#include "kernels.hpp"

namespace boxwood {
namespace scalar {
namespace {

/// Code of one value of numeric split feature f: the number of borders b with
/// x > b. The borders, padded with +inf to 255, are stored as a complete binary
/// search tree of 8 levels in breadth-first order (node h has the children
/// 2h + 1 and 2h + 2; level l holds the nodes 2^l - 1 .. 2^(l+1) - 2), so the
/// search has no branches and a SIMD backend can read a whole level from
/// registers. x > +inf is false and NaN compares false, so both walk left
/// where the real borders end (to 0 for NaN). NaN then gets border_count if
/// nan_high.
uint8_t bin_code(const OTEView& v, uint32_t f, float x) {
  const float* t = v.border_heap + static_cast<size_t>(f) * 256;
  uint32_t h = 0;
  for (int l = 0; l < 8; ++l) h = 2 * h + 1 + (x > t[h] ? 1u : 0u);
  uint32_t code = h - 255;
  if (std::isnan(x) && v.nan_high[f] != 0) {
    code = v.border_begin[f + 1] - v.border_begin[f];
  }
  return static_cast<uint8_t>(code);
}

/// Leaf indices of observations [i0, n_obs) of a block in tree t, into idx[0, observations - i0):
/// bit l = split l is true (GT: code > arg, EQ: code == arg), index =
/// sum_l bit_l * 2^l. One pass over the codes per split, which the compiler
/// vectorizes.
void tree_leaves(const OTEView& v, const uint8_t* codes, size_t n_obs, size_t i0, uint32_t t, uint8_t* idx) {
  const size_t n = n_obs - i0;
  std::fill(idx, idx + n, uint8_t{0});
  for (uint32_t l = 0; l < v.depth[t]; ++l) {
    const uint32_t cond = v.split_cond[v.split_begin[t] + l];
    const uint8_t* c = codes + static_cast<size_t>(v.cond_feature[cond]) * n_obs + i0;
    const uint8_t arg = v.cond_arg[cond];
    const uint8_t bit = static_cast<uint8_t>(1u << l);
    if (v.cond_kind[cond] != 0) {
      for (size_t i = 0; i < n; ++i) idx[i] |= c[i] == arg ? bit : uint8_t{0};
    } else {
      for (size_t i = 0; i < n; ++i) idx[i] |= c[i] > arg ? bit : uint8_t{0};
    }
  }
}

}  // namespace

void encode_numeric_tail(const OTEView& v, const float* X, size_t n_obs, size_t i0, uint8_t* codes) {
  for (uint32_t f = 0; f < v.n_numeric_features; ++f) {
    const float* x = X + v.numeric_column[f];
    uint8_t* c = codes + static_cast<size_t>(f) * n_obs;
    for (size_t i = i0; i < n_obs; ++i) c[i] = bin_code(v, f, x[i * v.n_numeric]);
  }
}

void accumulate_tail(const OTEView& v, const uint8_t* codes, size_t n_obs, size_t i0, double* margins) {
  uint8_t idx[kBlockSize];
  if (v.leaf_type == LeafType::Float32) {
    for (uint32_t t = 0; t < v.n_trees; ++t) {
      tree_leaves(v, codes, n_obs, i0, t, idx);
      for (size_t i = i0; i < n_obs; ++i) {
        for (uint32_t k = 0; k < v.K; ++k) {
          const float w = v.leaves_by_output[v.leaf_out_begin[t * v.K + k] + idx[i - i0]];
          margins[i * v.K + k] += static_cast<double>(w);
        }
      }
    }
    return;
  }
  const size_t n = (n_obs - i0) * v.K;
  std::vector<int32_t> sum(n);
  for (uint32_t t0 = 0; t0 < v.n_trees; t0 += kInt8TreeBlock) {
    std::fill(sum.begin(), sum.end(), 0);
    for (uint32_t t = t0; t < std::min(t0 + kInt8TreeBlock, v.n_trees); ++t) {
      tree_leaves(v, codes, n_obs, i0, t, idx);
      for (size_t i = 0; i < n_obs - i0; ++i) {
        for (uint32_t k = 0; k < v.K; ++k) {
          sum[i * v.K + k] += v.leaves_q_by_output[v.leaf_out_begin[t * v.K + k] + idx[i]] * v.tree_mult[t];
        }
      }
    }
    const double unit = v.block_unit[t0 / kInt8TreeBlock];
    for (size_t j = 0; j < n; ++j) margins[i0 * v.K + j] += unit * static_cast<double>(sum[j]);
  }
}

void leaf_index_tail(const OTEView& v, const uint8_t* codes, size_t n_obs, size_t i0, uint8_t* out) {
  uint8_t idx[kBlockSize];
  for (uint32_t t = 0; t < v.n_trees; ++t) {
    tree_leaves(v, codes, n_obs, i0, t, idx);
    for (size_t i = i0; i < n_obs; ++i) out[i * v.n_trees + t] = idx[i - i0];
  }
}

void encode_numeric(const OTEView& v, const float* X, size_t n_obs, uint8_t* codes) {
  encode_numeric_tail(v, X, n_obs, 0, codes);
}

void accumulate(const OTEView& v, const uint8_t* codes, size_t n_obs, double* margins) {
  accumulate_tail(v, codes, n_obs, 0, margins);
}

void leaf_index(const OTEView& v, const uint8_t* codes, size_t n_obs, uint8_t* out) {
  leaf_index_tail(v, codes, n_obs, 0, out);
}

}  // namespace scalar
}  // namespace boxwood
