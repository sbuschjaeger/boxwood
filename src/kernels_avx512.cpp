// The AVX-512 backends. This is the only file that may use AVX-512 intrinsics.
// It is compiled twice: as the avx512 backend with -mavx512f -mavx512bw
// -mavx512vl -mavx2 -mfma (/arch:AVX512 on MSVC), and, included by
// kernels_avx512_vbmi.cpp with BOXWOOD_AVX512_VBMI = 1, as the avx512_vbmi
// backend, which also needs VBMI and VNNI. Every function gives bit-identical
// results to the scalar reference: the same search tree for the binning, and
// the leaf values added to the float64 margins by the rule of
// scalar::accumulate_tail. Observations that do not fill a full group fall back to the
// scalar code.
//
// The speed comes from table lookups in registers: a level of the border search
// tree, or one output's leaf values of a tree (leaves_by_output, padded to 16
// floats), is held in up to 8 pairs of registers, so a lookup is a permute
// instead of a gather. Int8 codes are looked up with byte shuffles (16 leaves
// for 64 observations per shuffle; with VBMI, byte permutes over 64 or 128 leaves) and
// summed as integers.

#include <immintrin.h>


#include "kernels.hpp"
#include "simd.hpp"

#ifndef BOXWOOD_AVX512_VBMI
#define BOXWOOD_AVX512_VBMI 0
#endif

namespace boxwood {
#if BOXWOOD_AVX512_VBMI
namespace avx512_vbmi {
#else
namespace avx512 {
#endif
namespace {

// No namespace-scope SIMD constants: their initializers would run at load time,
// also on a CPU without AVX-512, before the run-time check picks a backend.

/// Number of registers of a table of 32 * G floats; G = 0: at most 16 floats.
constexpr int table_regs(int G) { return G == 0 ? 1 : 2 * G; }

/// Loads a table (padded to at least 16 floats) into registers.
template <int G>
void load_table(__m512* tab, const float* table) {
  for (int i = 0; i < table_regs(G); ++i) tab[i] = _mm512_loadu_ps(table + 16 * i);
}

/// table[idx] in 16 lanes from a table loaded by load_table<G>. A permute reads
/// only the low 4 (G = 0) or 5 bits of a lane's index; for G > 1 bits 5, 6, 7
/// pick one of the G permutes of 32 floats each. Higher bits of a lane are
/// ignored.
template <int G>
__m512 lookup(const __m512* tab, __m512i idx) {
  if constexpr (G == 0) {
    return _mm512_permutexvar_ps(idx, tab[0]);
  } else {
    __m512 r[G];
    for (int g = 0; g < G; ++g) r[g] = _mm512_permutex2var_ps(tab[2 * g], idx, tab[2 * g + 1]);
    // Each index bit halves the candidates: r[2g] and r[2g+1] differ in it.
    int bit = 32;
    for (int n = G; n > 1; n /= 2, bit *= 2) {
      const __mmask16 hi = _mm512_test_epi32_mask(idx, _mm512_set1_epi32(bit));
      for (int g = 0; g < n / 2; ++g) r[g] = _mm512_mask_blend_ps(hi, r[2 * g], r[2 * g + 1]);
    }
    return r[0];
  }
}

/// One level of the border search: h = 2h + 1 + (x > node), node = the value
/// looked up for h.
__m512i search_step(__m512i h, __m512 x, __m512 node) {
  const __m512i one = _mm512_set1_epi32(1);
  const __m512i next = _mm512_add_epi32(_mm512_slli_epi32(h, 1), one);
  // Ordered compare: NaN > b is false, so NaN walks left.
  return _mm512_mask_add_epi32(next, _mm512_cmp_ps_mask(x, node, _CMP_GT_OQ), next, one);
}

/// The result of every split condition for observations [i0, i0 + 64), from the
/// feature-major codes: bit j of masks[c] = condition c is true for observation
/// i0 + j (GT: code > arg, EQ: code == arg).
void cond_masks(const OTEView& v, const uint8_t* codes, size_t n_obs, size_t i0, uint64_t* masks) {
  for (uint32_t c = 0; c < v.n_conds; ++c) {
    const __m512i x = _mm512_loadu_si512(
        reinterpret_cast<const void*>(codes + static_cast<size_t>(v.cond_feature[c]) * n_obs + i0));
    const __m512i arg = _mm512_set1_epi8(static_cast<char>(v.cond_arg[c]));
    masks[c] = _cvtmask64_u64(v.cond_kind[c] != 0 ? _mm512_cmpeq_epi8_mask(x, arg)
                                                  : _mm512_cmpgt_epu8_mask(x, arg));
  }
}

/// 64 byte leaf indices of tree t for the observations of the condition masks (from
/// cond_masks): bit l = split l is true. Each split sets its bit in the 64
/// observations with one masked byte add.
__m512i leaf_bytes64(const OTEView& v, const uint64_t* masks, uint32_t t) {
  __m512i idx = _mm512_setzero_si512();
  const uint32_t s0 = v.split_begin[t];
  for (uint32_t l = 0; l < v.depth[t]; ++l) {
    const __mmask64 hit = _cvtu64_mask64(masks[v.split_cond[s0 + l]]);
    // The bit is not set yet, so adding it is the same as or-ing it in.
    idx = _mm512_mask_add_epi8(idx, hit, idx, _mm512_set1_epi8(static_cast<char>(1u << l)));
  }
  return idx;
}

/// Adds output k of tree t (32 * G leaves; G = 0: at most 16) to the float64
/// sums of 64 observations with the leaf bytes idx64 (from leaf_bytes64). The observations are
/// in lookup order: acc[2q] lane j is observation 4j + q, acc[2q + 1] lane j is observation
/// 4(j + 8) + q (see accumulate_impl).
template <int G>
void add_tree(__m512d* acc, const OTEView& v, __m512i idx64, uint32_t t, uint32_t k) {
  __m512 tab[table_regs(G)];
  load_table<G>(tab, v.leaves_by_output + v.leaf_out_begin[static_cast<size_t>(t) * v.K + k]);
  for (int q = 0; q < 4; ++q) {
    // 32-bit lane j of idx64 holds the leaf bytes of observations 4j .. 4j + 3; the
    // shift brings the byte of observation 4j + q to the bottom, where the lookup reads.
    const __m512 val = lookup<G>(tab, _mm512_srli_epi32(idx64, 8 * q));
    // The upper 8 floats moved into the low 256 bits (extractf32x8 would need
    // AVX512DQ, which this backend does not require).
    const __m256 lo = _mm512_castps512_ps256(val);
    const __m256 hi = _mm512_castps512_ps256(_mm512_shuffle_f32x4(val, val, _MM_SHUFFLE(3, 2, 3, 2)));
    acc[2 * q] = _mm512_add_pd(acc[2 * q], _mm512_cvtps_pd(lo));
    acc[2 * q + 1] = _mm512_add_pd(acc[2 * q + 1], _mm512_cvtps_pd(hi));
  }
}

/// Copies the margins of the 64 observations from i0 into m (in) or back: the kernels
/// keep them in the order their vectors produce, m[64k + p] is output k of
/// observation i0 + obs_of(p).
template <class ObsOf>
void copy_margins(double* margins, uint32_t K, size_t i0, double* m, ObsOf obs_of, bool in) {
  for (uint32_t k = 0; k < K; ++k) {
    for (int p = 0; p < 64; ++p) {
      double& theirs = margins[(i0 + static_cast<size_t>(obs_of(p))) * K + k];
      if (in) {
        m[64 * k + p] = theirs;
      } else {
        theirs = m[64 * k + p];
      }
    }
  }
}

void accumulate_float(const OTEView& v, const uint8_t* codes, size_t n_obs, double* margins) {
  const uint32_t K = v.K;
  // The margins of one 64-observation group in lookup order: m[64k + 16q + j] is
  // output k of observation 4j + q (acc[2q] holds j < 8, acc[2q + 1] holds j >= 8).
  const auto obs_of = [](int p) { return 4 * (p % 16) + p / 16; };
  Scratch<double> m(64 * static_cast<size_t>(K));
  Scratch<uint64_t> masks(v.n_conds);
  size_t i0 = 0;
  for (; i0 + 64 <= n_obs; i0 += 64) {
    cond_masks(v, codes, n_obs, i0, masks.data());
    copy_margins(margins, K, i0, m.data(), obs_of, true);
    // With several outputs, the leaf bytes of up to 64 trees are computed once
    // for all of them; with one output they are computed where they are used.
    for (uint32_t t0 = 0; t0 < v.n_trees; t0 += 64) {
      const uint32_t t1 = min_u32(t0 + 64u, v.n_trees);
      __m512i idx[64];
      if (K > 1) {
        for (uint32_t t = t0; t < t1; ++t) idx[t - t0] = leaf_bytes64(v, masks.data(), t);
      }
      for (uint32_t k = 0; k < K; ++k) {
        __m512d acc[8];
        for (int i = 0; i < 8; ++i) acc[i] = _mm512_loadu_pd(m.data() + 64 * k + 8 * i);
        for (uint32_t t = t0; t < t1; ++t) {
          const __m512i ix = K > 1 ? idx[t - t0] : leaf_bytes64(v, masks.data(), t);
          switch (v.depth[t]) {
            case 5: add_tree<1>(acc, v, ix, t, k); break;
            case 6: add_tree<2>(acc, v, ix, t, k); break;
            case 7: add_tree<4>(acc, v, ix, t, k); break;
            case 8: add_tree<8>(acc, v, ix, t, k); break;
            default: add_tree<0>(acc, v, ix, t, k); break;  // depth 0..4
          }
        }
        for (int i = 0; i < 8; ++i) _mm512_storeu_pd(m.data() + 64 * k + 8 * i, acc[i]);
      }
    }
    copy_margins(margins, K, i0, m.data(), obs_of, false);
  }
  scalar::accumulate_tail(v, codes, n_obs, i0, margins);
}

/// table[idx] in 64 byte lanes from a table of 16 * P int8 codes: pshufb looks
/// up 16 bytes with the low 4 index bits (one shuffle per piece of 16), then
/// index bits 4, 5, ... pick the piece, each bit halving the candidates.
template <int P>
__m512i lookup_q(const int8_t* table, __m512i idx) {
  const __m512i lo = _mm512_and_si512(idx, _mm512_set1_epi8(15));
  __m512i r[P];
  for (int p = 0; p < P; ++p) {
    const __m128i piece = _mm_loadu_si128(reinterpret_cast<const __m128i*>(table + 16 * p));
    r[p] = _mm512_shuffle_epi8(_mm512_broadcast_i32x4(piece), lo);
  }
  int bit = 16;
  for (int n = P; n > 1; n /= 2, bit *= 2) {
    const __mmask64 hi = _mm512_test_epi8_mask(idx, _mm512_set1_epi8(static_cast<char>(bit)));
    for (int g = 0; g < n / 2; ++g) r[g] = _mm512_mask_blend_epi8(hi, r[2 * g], r[2 * g + 1]);
  }
  return r[0];
}

/// The int8 codes of output k of tree t for 64 observations with the leaf bytes idx.
__m512i tree_codes64(const OTEView& v, uint32_t t, uint32_t k, __m512i idx) {
  const int8_t* table = v.leaves_q_by_output + v.leaf_out_begin[static_cast<size_t>(t) * v.K + k];
#if BOXWOOD_AVX512_VBMI
  // A byte permute reads the low 6 bits of each index from one register of 64
  // codes, or the low 7 bits from two; for 256 codes, index bit 7 (the sign
  // bit) picks one of two such lookups.
  const auto part = [&](int p) { return _mm512_loadu_si512(reinterpret_cast<const void*>(table + 64 * p)); };
  switch (v.depth[t]) {
    case 5:
      return _mm512_permutexvar_epi8(
          idx, _mm512_zextsi256_si512(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(table))));
    case 6: return _mm512_permutexvar_epi8(idx, part(0));
    case 7: return _mm512_permutex2var_epi8(part(0), idx, part(1));
    case 8:
      return _mm512_mask_blend_epi8(_mm512_movepi8_mask(idx), _mm512_permutex2var_epi8(part(0), idx, part(1)),
                                    _mm512_permutex2var_epi8(part(2), idx, part(3)));
    default: return lookup_q<1>(table, idx);  // depth 0..4
  }
#else
  switch (v.depth[t]) {
    case 5: return lookup_q<2>(table, idx);
    case 6: return lookup_q<4>(table, idx);
    case 7: return lookup_q<8>(table, idx);
    case 8: return lookup_q<16>(table, idx);
    default: return lookup_q<1>(table, idx);  // depth 0..4
  }
#endif
}

/// Adds q[j] * mult[j] of four trees to the int32 sums of 64 observations, where q[j]
/// holds the codes of tree j (byte r = observation r) and mult packs the four tree
/// multipliers into the bytes of one int32. The bytes are transposed so one
/// int32 lane holds the four trees' codes of one observation; maddubs multiplies them
/// by the multipliers and adds pairs (at most 2 * 127 * 128, no saturation),
/// madd adds the two pairs (VNNI: one dot product of the four bytes). The
/// unpacks work within 128-bit lanes, so lane L of sum[i] holds observations
/// 16L + 4i .. 16L + 4i + 3.
void add_four(__m512i* sum, const __m512i* q, uint32_t mult) {
  const __m512i a = _mm512_unpacklo_epi8(q[0], q[1]);
  const __m512i b = _mm512_unpackhi_epi8(q[0], q[1]);
  const __m512i c = _mm512_unpacklo_epi8(q[2], q[3]);
  const __m512i d = _mm512_unpackhi_epi8(q[2], q[3]);
  const __m512i n_obs[4] = {_mm512_unpacklo_epi16(a, c), _mm512_unpackhi_epi16(a, c),
                           _mm512_unpacklo_epi16(b, d), _mm512_unpackhi_epi16(b, d)};
  const __m512i m = _mm512_set1_epi32(static_cast<int>(mult));
#if BOXWOOD_AVX512_VBMI
  for (int i = 0; i < 4; ++i) sum[i] = _mm512_dpbusd_epi32(sum[i], m, n_obs[i]);
#else
  const __m512i ones = _mm512_set1_epi16(1);
  for (int i = 0; i < 4; ++i) {
    sum[i] = _mm512_add_epi32(sum[i], _mm512_madd_epi16(_mm512_maddubs_epi16(m, n_obs[i]), ones));
  }
#endif
}

void accumulate_int8(const OTEView& v, const uint8_t* codes, size_t n_obs, double* margins) {
  const uint32_t K = v.K;
  // The float64 sums of sum[i] are stored at m[64k + 16i + 8h + e]: h = 0
  // is the low half of sum[i] (128-bit lanes 0 and 1), h = 1 the high half,
  // and lane L of sum[i] holds observations 16L + 4i .. 16L + 4i + 3.
  const auto obs_of = [](int p) {
    const int i = p / 16, h = p / 8 % 2, e = p % 8;
    return 16 * (2 * h + e / 4) + 4 * i + e % 4;
  };
  Scratch<double> m(64 * static_cast<size_t>(K));
  Scratch<uint64_t> masks(v.n_conds);
  size_t i0 = 0;
  for (; i0 + 64 <= n_obs; i0 += 64) {
    cond_masks(v, codes, n_obs, i0, masks.data());
    copy_margins(margins, K, i0, m.data(), obs_of, true);
    for (uint32_t t0 = 0; t0 < v.n_trees; t0 += kInt8TreeBlock) {
      const uint32_t t1 = min_u32(t0 + kInt8TreeBlock, v.n_trees);
      // The leaf bytes of the block's trees, computed once for all outputs;
      // a block's last group may have fewer than four trees: the missing ones
      // add zero codes with multiplier 0.
      __m512i idx[kInt8TreeBlock];
      for (uint32_t t = t0; t < t1; ++t) idx[t - t0] = leaf_bytes64(v, masks.data(), t);
      const __m512d unit = _mm512_set1_pd(v.block_unit[t0 / kInt8TreeBlock]);
      for (uint32_t k = 0; k < K; ++k) {
        __m512i sum[4] = {};
        for (uint32_t t = t0; t < t1; t += 4) {
          __m512i q[4];
          uint32_t mult = 0;
          for (uint32_t j = 0; j < 4; ++j) {
            if (t + j < t1) {
              q[j] = tree_codes64(v, t + j, k, idx[t + j - t0]);
              mult |= static_cast<uint32_t>(v.tree_mult[t + j]) << (8 * j);
            } else {
              q[j] = _mm512_setzero_si512();
            }
          }
          add_four(sum, q, mult);
        }
        // unit * sum is exact; the add is the one rounding, as in the scalar code.
        for (int i = 0; i < 4; ++i) {
          double* mi = m.data() + 64 * k + 16 * i;
          const __m512d lo = _mm512_cvtepi32_pd(_mm512_castsi512_si256(sum[i]));
          const __m512d hi = _mm512_cvtepi32_pd(_mm512_extracti64x4_epi64(sum[i], 1));
          _mm512_storeu_pd(mi, _mm512_add_pd(_mm512_loadu_pd(mi), _mm512_mul_pd(unit, lo)));
          _mm512_storeu_pd(mi + 8, _mm512_add_pd(_mm512_loadu_pd(mi + 8), _mm512_mul_pd(unit, hi)));
        }
      }
    }
    copy_margins(margins, K, i0, m.data(), obs_of, false);
  }
  scalar::accumulate_tail(v, codes, n_obs, i0, margins);
}

}  // namespace

void encode_numeric(const OTEView& v, const float* X, size_t n_obs, uint8_t* codes) {
  const __m512i seq =
      _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
  const int n = static_cast<int>(v.n_numeric);
  const size_t full = n_obs - n_obs % 16;  // the rest goes through the scalar code
  for (uint32_t f = 0; f < v.n_numeric_features; ++f) {
    // The search tree's levels in registers: levels 0..3 (15 nodes) share one
    // table, levels 4, 5, 6, 7 have 16, 32, 64, 128 nodes starting at node
    // 15, 31, 63, 127.
    const float* t = v.border_heap + static_cast<size_t>(f) * 256;
    __m512 top[1], l4[1], l5[2], l6[4], l7[8];
    load_table<0>(top, t);
    load_table<0>(l4, t + 15);
    load_table<1>(l5, t + 31);
    load_table<2>(l6, t + 63);
    load_table<4>(l7, t + 127);
    const float* x = X + v.numeric_column[f];
    uint8_t* c = codes + static_cast<size_t>(f) * n_obs;
    for (size_t i = 0; i < full; i += 16) {
      // Gather the 16 values of this feature from the row-major X.
      const __m512i off = _mm512_mullo_epi32(
          _mm512_add_epi32(_mm512_set1_epi32(static_cast<int>(i)), seq), _mm512_set1_epi32(n));
      const __m512 val = _mm512_i32gather_ps(off, x, 4);
      __m512i h = _mm512_setzero_si512();
      for (int l = 0; l < 4; ++l) h = search_step(h, val, lookup<0>(top, h));
      const auto at = [&](int base) { return _mm512_sub_epi32(h, _mm512_set1_epi32(base)); };
      h = search_step(h, val, lookup<0>(l4, at(15)));
      h = search_step(h, val, lookup<1>(l5, at(31)));
      h = search_step(h, val, lookup<2>(l6, at(63)));
      h = search_step(h, val, lookup<4>(l7, at(127)));
      __m512i code = _mm512_sub_epi32(h, _mm512_set1_epi32(255));
      if (v.nan_high[f] != 0) {
        const __mmask16 is_nan = _mm512_cmp_ps_mask(val, val, _CMP_UNORD_Q);
        const __m512i high = _mm512_set1_epi32(
            static_cast<int>(v.border_begin[f + 1] - v.border_begin[f]));
        code = _mm512_mask_mov_epi32(code, is_nan, high);
      }
      // Pack the 16 codes (0..255) into 16 bytes.
      _mm_storeu_si128(reinterpret_cast<__m128i*>(c + i), _mm512_cvtusepi32_epi8(code));
    }
  }
  scalar::encode_numeric_tail(v, X, n_obs, full, codes);
}

void accumulate(const OTEView& v, const uint8_t* codes, size_t n_obs, double* margins) {
  if (v.leaf_type == LeafType::Int8) {
    accumulate_int8(v, codes, n_obs, margins);
  } else {
    accumulate_float(v, codes, n_obs, margins);
  }
}

void leaf_index(const OTEView& v, const uint8_t* codes, size_t n_obs, uint8_t* out) {
  const uint32_t T = v.n_trees;
  Scratch<uint64_t> masks(v.n_conds);
  size_t i0 = 0;
  for (; i0 + 64 <= n_obs; i0 += 64) {
    cond_masks(v, codes, n_obs, i0, masks.data());
    for (uint32_t t = 0; t < T; ++t) {
      alignas(64) uint8_t idx[64];
      _mm512_store_si512(reinterpret_cast<void*>(idx), leaf_bytes64(v, masks.data(), t));
      for (int j = 0; j < 64; ++j) out[(i0 + static_cast<size_t>(j)) * T + t] = idx[j];
    }
  }
  scalar::leaf_index_tail(v, codes, n_obs, i0, out);
}

#if BOXWOOD_AVX512_VBMI
void ctr_lookup(const OTEView& v, uint32_t q, const uint64_t* hash, size_t n_obs, uint8_t* codes) {
  uint32_t slot[kBlockSize];
  scalar::ctr_slots(v, q, hash, n_obs, slot);
  // A table of at most 256 slots: each CTR's slot codes fit in 4 registers,
  // and the codes of 64 observations are two byte permutes of 128 codes each, picked
  // by bit 7 of the slot (most tables have 128 or 256 slots).
  const uint32_t n_slots = v.table_slot_begin[q + 1] - v.table_slot_begin[q];
  size_t i0 = 0;
  if (n_slots <= 256) {
    const size_t full = n_obs / 64 * 64;
    alignas(64) uint8_t slot8[kBlockSize];
    for (size_t i = 0; i < full; i += 16) {
      _mm_store_si128(reinterpret_cast<__m128i*>(slot8 + i), _mm512_cvtepi32_epi8(_mm512_loadu_si512(slot + i)));
    }
    for (uint32_t c = v.table_ctr_begin[q]; c < v.table_ctr_begin[q + 1]; ++c) {
      const uint32_t r = v.table_ctrs[c];
      const uint8_t* slot_codes = v.ctr_slot_codes + v.ctr_slot_code_begin[r];
      uint8_t* out = codes + static_cast<size_t>(v.n_numeric_features + v.n_onehot + r) * n_obs;
      // The slot codes in 64-byte pieces; a masked load reads only the
      // table's own bytes, the rest of a piece is never looked up.
      __m512i tab[4];
      for (uint32_t g = 0; g < 4; ++g) {
        const uint32_t n = n_slots > 64 * g ? min_u32(64, n_slots - 64 * g) : 0;
        tab[g] = n == 0 ? _mm512_setzero_si512()
                        : _mm512_maskz_loadu_epi8(n == 64 ? ~0ULL : (1ULL << n) - 1, slot_codes + 64 * g);
      }
      for (size_t i = 0; i < full; i += 64) {
        const __m512i idx = _mm512_load_si512(slot8 + i);
        __m512i res = _mm512_permutex2var_epi8(tab[0], idx, tab[1]);
        if (n_slots > 128) {
          res = _mm512_mask_blend_epi8(_mm512_movepi8_mask(idx), res, _mm512_permutex2var_epi8(tab[2], idx, tab[3]));
        }
        _mm512_storeu_si512(out + i, res);
      }
    }
    i0 = full;
  }
  scalar::ctr_codes_tail(v, q, slot, n_obs, i0, codes);
}
#endif

}  // namespace avx512 or avx512_vbmi
}  // namespace boxwood
