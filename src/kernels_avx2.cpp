// The AVX2 backend. This is the only file that may use intrinsics; CMake
// compiles it with -mavx2 -mfma (/arch:AVX2 on MSVC). Every function gives
// bit-identical results to the scalar reference: the same search tree for the
// binning, and the leaf values added to the float64 margins by the rule of
// scalar::accumulate_tail. Observations that do not fill a full vector group fall back
// to the scalar code.
//
// Small tables (the levels 0..6 of the border search tree, the float leaf
// values of trees up to depth 5) are looked up with register permutes over
// 8-float pieces. The last border level is gathered. Deeper trees load their
// float leaf values one observation at a time: with 8 or more permutes per 8 observations, and
// with AVX2 gathers, the lookup is slower than scalar loads. Int8 codes are
// looked up with byte shuffles (16 leaves for 32 observations per shuffle) and summed
// as integers.

#include <immintrin.h>


#include "kernels.hpp"
#include "simd.hpp"

namespace boxwood {
namespace avx2 {
namespace {

// No namespace-scope SIMD constants: their initializers would run at load time,
// also on a CPU without AVX2, before the run-time check picks a backend.

/// The result of every split condition for observations [i0, i0 + 32), from the
/// feature-major codes: byte 32c + j of conds is 0xFF if condition c is true
/// for observation i0 + j (GT: code > arg, EQ: code == arg), else 0.
void cond_vecs(const OTEView& v, const uint8_t* codes, size_t n_obs, size_t i0, uint8_t* conds) {
  const __m256i kSign = _mm256_set1_epi8(static_cast<char>(0x80));
  for (uint32_t c = 0; c < v.n_conds; ++c) {
    const __m256i x = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(codes + static_cast<size_t>(v.cond_feature[c]) * n_obs + i0));
    const __m256i arg = _mm256_set1_epi8(static_cast<char>(v.cond_arg[c]));
    // Unsigned byte compare: flip the sign bit of both sides and use the
    // signed cmpgt. EQ needs no flip.
    const __m256i hit = v.cond_kind[c] != 0
                            ? _mm256_cmpeq_epi8(x, arg)
                            : _mm256_cmpgt_epi8(_mm256_xor_si256(x, kSign), _mm256_xor_si256(arg, kSign));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(conds + 32 * static_cast<size_t>(c)), hit);
  }
}

/// 32 byte leaf indices of tree t for the observations of the condition vectors
/// (from cond_vecs): bit l = split l is true.
__m256i leaf_vec(const OTEView& v, const uint8_t* conds, uint32_t t) {
  __m256i idx = _mm256_setzero_si256();
  const uint32_t s0 = v.split_begin[t];
  for (uint32_t l = 0; l < v.depth[t]; ++l) {
    const __m256i hit = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(conds + 32 * static_cast<size_t>(v.split_cond[s0 + l])));
    idx = _mm256_or_si256(idx, _mm256_and_si256(hit, _mm256_set1_epi8(static_cast<char>(1u << l))));
  }
  return idx;
}

/// The 8 bytes [s*8, s*8+8) of `v` widened to 8 int32 lanes.
__m256i widen8(__m256i v, int s) {
  const __m128i half = s < 2 ? _mm256_castsi256_si128(v) : _mm256_extracti128_si256(v, 1);
  const __m128i bytes = (s & 1) != 0 ? _mm_unpackhi_epi64(half, half) : half;
  return _mm256_cvtepu8_epi32(bytes);
}

/// table[idx] in 8 lanes from a table of 8 * P floats: one permute per piece
/// of 8 floats (index bits 0..2), then index bits 3, 4, ... pick the piece,
/// each bit halving the candidates. blendv reads the sign bit, so the bit is
/// shifted there.
template <int P>
__m256 lookup(const float* table, __m256i idx) {
  __m256 r[P];
  for (int p = 0; p < P; ++p) r[p] = _mm256_permutevar8x32_ps(_mm256_loadu_ps(table + 8 * p), idx);
  int bit = 3;
  for (int n = P; n > 1; n /= 2, ++bit) {
    const __m256 sel = _mm256_castsi256_ps(_mm256_slli_epi32(idx, 31 - bit));
    for (int g = 0; g < n / 2; ++g) r[g] = _mm256_blendv_ps(r[2 * g], r[2 * g + 1], sel);
  }
  return r[0];
}

/// The 8 leaf values of output k of tree t (depth at most 5) for the leaf
/// indices `li` (8 int32 lanes), by register permutes over the tree's leaf
/// table for output k.
__m256 tree_leaves8(const OTEView& v, uint32_t t, uint32_t k, __m256i li) {
  const float* table = v.leaves_by_output + v.leaf_out_begin[static_cast<size_t>(t) * v.K + k];
  switch (v.depth[t]) {
    case 4: return lookup<2>(table, li);
    case 5: return lookup<4>(table, li);
    default: return lookup<1>(table, li);  // depth 0..3
  }
}

/// table[idx] in 32 byte lanes from a table of 16 * P int8 codes: pshufb looks
/// up 16 bytes with the low 4 index bits (one shuffle per piece of 16), then
/// index bits 4, 5, ... pick the piece, each bit halving the candidates.
/// blendv reads the sign bit of each byte, so the bit is shifted there; a
/// 16-bit shift by at most 3 moves no other bit into a byte's sign bit.
template <int P>
__m256i lookup_q(const int8_t* table, __m256i idx) {
  const __m256i lo = _mm256_and_si256(idx, _mm256_set1_epi8(15));
  __m256i r[P];
  for (int p = 0; p < P; ++p) {
    const __m128i piece = _mm_loadu_si128(reinterpret_cast<const __m128i*>(table + 16 * p));
    r[p] = _mm256_shuffle_epi8(_mm256_broadcastsi128_si256(piece), lo);
  }
  int bit = 4;
  for (int n = P; n > 1; n /= 2, ++bit) {
    const __m256i sel = _mm256_slli_epi16(idx, 7 - bit);
    for (int g = 0; g < n / 2; ++g) r[g] = _mm256_blendv_epi8(r[2 * g], r[2 * g + 1], sel);
  }
  return r[0];
}

/// The int8 codes of output k of tree t for 32 observations with the leaf bytes idx.
__m256i tree_codes32(const OTEView& v, uint32_t t, uint32_t k, __m256i idx) {
  const int8_t* table = v.leaves_q_by_output + v.leaf_out_begin[static_cast<size_t>(t) * v.K + k];
  switch (v.depth[t]) {
    case 5: return lookup_q<2>(table, idx);
    case 6: return lookup_q<4>(table, idx);
    case 7: return lookup_q<8>(table, idx);
    case 8: return lookup_q<16>(table, idx);
    default: return lookup_q<1>(table, idx);  // depth 0..4
  }
}

/// One level of the border search: h = 2h + 1 + (x > node), node = the value
/// of node h, from the level's table of 8 * P nodes that starts at node
/// `first` (P = 0: gathered from the whole tree).
template <int P>
__m256i search_step(__m256i h, __m256 x, const float* tree, int first) {
  __m256 node;
  if constexpr (P == 0) {
    node = _mm256_i32gather_ps(tree, h, 4);
  } else {
    node = lookup<P>(tree + first, _mm256_sub_epi32(h, _mm256_set1_epi32(first)));
  }
  const __m256i one = _mm256_set1_epi32(1);
  // Ordered compare: NaN > b is false, so NaN walks left. A true compare is
  // all ones (-1), so subtracting it adds 1.
  const __m256i gt = _mm256_castps_si256(_mm256_cmp_ps(x, node, _CMP_GT_OQ));
  return _mm256_sub_epi32(_mm256_add_epi32(_mm256_slli_epi32(h, 1), one), gt);
}

}  // namespace

void encode_numeric(const OTEView& v, const float* X, size_t n_obs, uint8_t* codes) {
  const __m256i seq = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
  const int n = static_cast<int>(v.n_numeric);
  const size_t full = n_obs - n_obs % 8;  // the rest goes through the scalar code
  for (uint32_t f = 0; f < v.n_numeric_features; ++f) {
    // Levels 0..2 of the search tree (7 nodes) share one table, levels 3..6
    // have 8, 16, 32, 64 nodes from node 7, 15, 31, 63; level 7 (128 nodes)
    // is gathered.
    const float* t = v.border_heap + static_cast<size_t>(f) * 256;
    const float* x = X + v.numeric_column[f];
    uint8_t* c = codes + static_cast<size_t>(f) * n_obs;
    for (size_t i = 0; i < full; i += 8) {
      // Gather the 8 values of this feature from the row-major X.
      const __m256i off = _mm256_add_epi32(_mm256_set1_epi32(static_cast<int>(i) * n),
                                           _mm256_mullo_epi32(seq, _mm256_set1_epi32(n)));
      const __m256 val = _mm256_i32gather_ps(x, off, 4);
      __m256i h = _mm256_setzero_si256();
      for (int l = 0; l < 3; ++l) h = search_step<1>(h, val, t, 0);
      h = search_step<1>(h, val, t, 7);
      h = search_step<2>(h, val, t, 15);
      h = search_step<4>(h, val, t, 31);
      h = search_step<8>(h, val, t, 63);
      h = search_step<0>(h, val, t, 0);
      __m256i code = _mm256_sub_epi32(h, _mm256_set1_epi32(255));
      if (v.nan_high[f] != 0) {
        const __m256 is_nan = _mm256_cmp_ps(val, val, _CMP_UNORD_Q);
        const __m256i high = _mm256_set1_epi32(
            static_cast<int>(v.border_begin[f + 1] - v.border_begin[f]));
        code = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(code),
                                                   _mm256_castsi256_ps(high), is_nan));
      }
      // Pack the 8 codes (0..255) into 8 bytes.
      const __m128i lo = _mm256_castsi256_si128(code);
      const __m128i hi = _mm256_extracti128_si256(code, 1);
      const __m128i packed = _mm_packus_epi16(_mm_packus_epi32(lo, hi), _mm_setzero_si128());
      _mm_storel_epi64(reinterpret_cast<__m128i*>(c + i), packed);
    }
  }
  scalar::encode_numeric_tail(v, X, n_obs, full, codes);
}

namespace {

// The float accumulate works on blocks of kTreeBlock trees and 32 observations: the
// leaf bytes of the block first (once for all outputs), then the values of
// output k, added to the margins in tree order. A tree at most
// kMaxPermuteDepth deep looks its values up with register permutes; a deeper
// one loads them one observation at a time (8 or more permutes per 8 observations cost more).
constexpr uint32_t kTreeBlock = 16;
constexpr uint32_t kMaxPermuteDepth = 5;

/// Adds output k of trees [t0, t0 + nb) to the margins m[j * K] of 32 observations,
/// with the leaf bytes idx[b] of tree t0 + b.
void add_block(const OTEView& v, uint32_t t0, uint32_t nb, uint32_t k, const uint8_t (*idx)[32],
               double* m) {
  const size_t K = v.K;
  // acc[i] holds observations 4i .. 4i + 3: eight independent sums, so the adds of
  // one tree do not wait for each other.
  __m256d acc[8];
  for (int i = 0; i < 8; ++i) {
    const double* p = m + 4 * i * K;
    acc[i] = _mm256_set_pd(p[3 * K], p[2 * K], p[K], p[0]);
  }
  for (uint32_t b = 0; b < nb; ++b) {
    const uint32_t t = t0 + b;
    if (v.depth[t] <= kMaxPermuteDepth) {
      const __m256i iv = _mm256_load_si256(reinterpret_cast<const __m256i*>(idx[b]));
      for (int s = 0; s < 4; ++s) {
        const __m256 val = tree_leaves8(v, t, k, widen8(iv, s));
        acc[2 * s] = _mm256_add_pd(acc[2 * s], _mm256_cvtps_pd(_mm256_castps256_ps128(val)));
        acc[2 * s + 1] = _mm256_add_pd(acc[2 * s + 1], _mm256_cvtps_pd(_mm256_extractf128_ps(val, 1)));
      }
    } else {
      const float* tb = v.leaves_by_output + v.leaf_out_begin[static_cast<size_t>(t) * K + k];
      const uint8_t* ib = idx[b];
      for (int i = 0; i < 8; ++i) {
        const __m128 val =
            _mm_setr_ps(tb[ib[4 * i]], tb[ib[4 * i + 1]], tb[ib[4 * i + 2]], tb[ib[4 * i + 3]]);
        acc[i] = _mm256_add_pd(acc[i], _mm256_cvtps_pd(val));
      }
    }
  }
  for (int i = 0; i < 8; ++i) {
    alignas(32) double tmp[4];
    _mm256_store_pd(tmp, acc[i]);
    for (int j = 0; j < 4; ++j) m[(4 * i + j) * K] = tmp[j];
  }
}

void accumulate_float(const OTEView& v, const uint8_t* codes, size_t n_obs, double* margins) {
  Scratch<uint8_t> conds(32 * static_cast<size_t>(v.n_conds));
  size_t i0 = 0;
  for (; i0 + 32 <= n_obs; i0 += 32) {
    cond_vecs(v, codes, n_obs, i0, conds.data());
    for (uint32_t t0 = 0; t0 < v.n_trees; t0 += kTreeBlock) {
      const uint32_t nb = min_u32(kTreeBlock, v.n_trees - t0);
      alignas(32) uint8_t idx[kTreeBlock][32];
      for (uint32_t b = 0; b < nb; ++b) {
        _mm256_store_si256(reinterpret_cast<__m256i*>(idx[b]), leaf_vec(v, conds.data(), t0 + b));
      }
      for (uint32_t k = 0; k < v.K; ++k) add_block(v, t0, nb, k, idx, margins + i0 * v.K + k);
    }
  }
  scalar::accumulate_tail(v, codes, n_obs, i0, margins);
}

/// Adds q[j] * mult[j] of four trees to the int32 sums of 32 observations, where q[j]
/// holds the codes of tree j (byte r = observation r) and mult packs the four tree
/// multipliers into the bytes of one int32. The bytes are transposed so one
/// int32 lane holds the four trees' codes of one observation; maddubs multiplies them
/// by the multipliers and adds pairs (at most 2 * 127 * 128, no saturation),
/// madd adds the two pairs. The unpacks work within 128-bit lanes, so sum[i]
/// holds observations 4i .. 4i + 3 and 16 + 4i .. 16 + 4i + 3.
void add_four(__m256i* sum, const __m256i* q, uint32_t mult) {
  const __m256i a = _mm256_unpacklo_epi8(q[0], q[1]);
  const __m256i b = _mm256_unpackhi_epi8(q[0], q[1]);
  const __m256i c = _mm256_unpacklo_epi8(q[2], q[3]);
  const __m256i d = _mm256_unpackhi_epi8(q[2], q[3]);
  const __m256i n_obs[4] = {_mm256_unpacklo_epi16(a, c), _mm256_unpackhi_epi16(a, c),
                           _mm256_unpacklo_epi16(b, d), _mm256_unpackhi_epi16(b, d)};
  const __m256i m = _mm256_set1_epi32(static_cast<int>(mult));
  const __m256i ones = _mm256_set1_epi16(1);
  for (int i = 0; i < 4; ++i) {
    sum[i] = _mm256_add_epi32(sum[i], _mm256_madd_epi16(_mm256_maddubs_epi16(m, n_obs[i]), ones));
  }
}

void accumulate_int8(const OTEView& v, const uint8_t* codes, size_t n_obs, double* margins) {
  const uint32_t K = v.K;
  // Observation of element e of the float64 sums of sum[i]'s low half (h = 0) and
  // high half (h = 1): observations 4i + e, then 16 + 4i + e. The margins of one
  // 32-observation group are kept in that order: m[32k + 8i + 4h + e].
  const auto obs_of = [](int i, int h, int e) { return 16 * h + 4 * i + e; };
  Scratch<double> m(32 * static_cast<size_t>(K));
  // Copies the margins of the group at i0 into m (in) or back.
  const auto copy_margins = [&](size_t i0, bool in) {
    for (uint32_t k = 0; k < K; ++k) {
      for (int i = 0; i < 4; ++i) {
        for (int h = 0; h < 2; ++h) {
          for (int e = 0; e < 4; ++e) {
            double& mine = m[32 * k + 8 * i + 4 * h + e];
            double& theirs = margins[(i0 + obs_of(i, h, e)) * K + k];
            if (in) {
              mine = theirs;
            } else {
              theirs = mine;
            }
          }
        }
      }
    }
  };
  Scratch<uint8_t> conds(32 * static_cast<size_t>(v.n_conds));
  size_t i0 = 0;
  for (; i0 + 32 <= n_obs; i0 += 32) {
    cond_vecs(v, codes, n_obs, i0, conds.data());
    copy_margins(i0, true);
    for (uint32_t t0 = 0; t0 < v.n_trees; t0 += kInt8TreeBlock) {
      const uint32_t t1 = min_u32(t0 + kInt8TreeBlock, v.n_trees);
      // The leaf bytes of the block's trees, computed once for all outputs;
      // a block's last group may have fewer than four trees: the missing ones
      // add zero codes with multiplier 0.
      __m256i idx[kInt8TreeBlock];
      for (uint32_t t = t0; t < t1; ++t) idx[t - t0] = leaf_vec(v, conds.data(), t);
      const __m256d unit = _mm256_set1_pd(v.block_unit[t0 / kInt8TreeBlock]);
      for (uint32_t k = 0; k < K; ++k) {
        __m256i sum[4] = {};
        for (uint32_t t = t0; t < t1; t += 4) {
          __m256i q[4];
          uint32_t mult = 0;
          for (uint32_t j = 0; j < 4; ++j) {
            if (t + j < t1) {
              q[j] = tree_codes32(v, t + j, k, idx[t + j - t0]);
              mult |= static_cast<uint32_t>(v.tree_mult[t + j]) << (8 * j);
            } else {
              q[j] = _mm256_setzero_si256();
            }
          }
          add_four(sum, q, mult);
        }
        // unit * sum is exact; the add is the one rounding, as in the scalar code.
        for (int i = 0; i < 4; ++i) {
          double* mi = m.data() + 32 * k + 8 * i;
          const __m256d lo = _mm256_cvtepi32_pd(_mm256_castsi256_si128(sum[i]));
          const __m256d hi = _mm256_cvtepi32_pd(_mm256_extracti128_si256(sum[i], 1));
          _mm256_storeu_pd(mi, _mm256_add_pd(_mm256_loadu_pd(mi), _mm256_mul_pd(unit, lo)));
          _mm256_storeu_pd(mi + 4, _mm256_add_pd(_mm256_loadu_pd(mi + 4), _mm256_mul_pd(unit, hi)));
        }
      }
    }
    copy_margins(i0, false);
  }
  scalar::accumulate_tail(v, codes, n_obs, i0, margins);
}

}  // namespace

void accumulate(const OTEView& v, const uint8_t* codes, size_t n_obs, double* margins) {
  if (v.leaf_type == LeafType::Int8) {
    accumulate_int8(v, codes, n_obs, margins);
  } else {
    accumulate_float(v, codes, n_obs, margins);
  }
}

void leaf_index(const OTEView& v, const uint8_t* codes, size_t n_obs, uint8_t* out) {
  const uint32_t T = v.n_trees;
  Scratch<uint8_t> conds(32 * static_cast<size_t>(v.n_conds));
  size_t i0 = 0;
  for (; i0 + 32 <= n_obs; i0 += 32) {
    cond_vecs(v, codes, n_obs, i0, conds.data());
    for (uint32_t t = 0; t < T; ++t) {
      alignas(32) uint8_t idx[32];
      _mm256_store_si256(reinterpret_cast<__m256i*>(idx), leaf_vec(v, conds.data(), t));
      for (int j = 0; j < 32; ++j) out[(i0 + static_cast<size_t>(j)) * T + t] = idx[j];
    }
  }
  scalar::leaf_index_tail(v, codes, n_obs, i0, out);
}

}  // namespace avx2
}  // namespace boxwood
