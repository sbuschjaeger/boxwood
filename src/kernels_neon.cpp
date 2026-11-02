// The NEON backend for AArch64, where NEON is always present (no run-time
// check, no extra compiler flags). It speeds up the int8 leaves: the split
// conditions and leaf indices of 16 observations per vector, the int8 codes looked up
// with table instructions (tbl reads up to 64 table bytes per instruction,
// tbx extends a lookup past 64) and summed as integers. Float32 leaves and the
// numeric encoding use the scalar code. Every function gives bit-identical
// results to the scalar reference; observations that do not fill a group of 16 go
// through the scalar code.

#include <arm_neon.h>

#include "kernels.hpp"
#include "simd.hpp"

namespace boxwood {
namespace neon {
namespace {

/// The result of every split condition for observations [i0, i0 + 16), from the
/// feature-major codes: byte 16c + j of conds is 0xFF if condition c is true
/// for observation i0 + j (GT: code > arg, EQ: code == arg), else 0.
void cond_vecs(const OTEView& v, const uint8_t* codes, size_t n_obs, size_t i0, uint8_t* conds) {
  for (uint32_t c = 0; c < v.n_conds; ++c) {
    const uint8x16_t x = vld1q_u8(codes + static_cast<size_t>(v.cond_feature[c]) * n_obs + i0);
    const uint8x16_t arg = vdupq_n_u8(v.cond_arg[c]);
    vst1q_u8(conds + 16 * static_cast<size_t>(c), v.cond_kind[c] != 0 ? vceqq_u8(x, arg) : vcgtq_u8(x, arg));
  }
}

/// 16 byte leaf indices of tree t for the observations of the condition vectors
/// (from cond_vecs): bit l = split l is true.
uint8x16_t leaf_vec(const OTEView& v, const uint8_t* conds, uint32_t t) {
  uint8x16_t idx = vdupq_n_u8(0);
  const uint32_t s0 = v.split_begin[t];
  for (uint32_t l = 0; l < v.depth[t]; ++l) {
    const uint8x16_t hit = vld1q_u8(conds + 16 * static_cast<size_t>(v.split_cond[s0 + l]));
    idx = vorrq_u8(idx, vandq_u8(hit, vdupq_n_u8(static_cast<uint8_t>(1u << l))));
  }
  return idx;
}

/// The four 16-byte pieces of table[64p, 64p + 64).
uint8x16x4_t piece64(const uint8_t* table, int p) {
  const uint8_t* t = table + 64 * p;
  return {{vld1q_u8(t), vld1q_u8(t + 16), vld1q_u8(t + 32), vld1q_u8(t + 48)}};
}

/// The int8 codes of output k of tree t for 16 observations with the leaf bytes idx.
/// The table has 2^depth entries, at least 16 (see OTEView), so a lookup
/// reads only the table: tbl gives 0 for an index past its pieces, and each
/// further tbx fills in the next 64 entries (index - 64p).
int8x16_t tree_codes16(const OTEView& v, uint32_t t, uint32_t k, uint8x16_t idx) {
  const uint8_t* table = reinterpret_cast<const uint8_t*>(v.leaves_q_by_output) +
                         v.leaf_out_begin[static_cast<size_t>(t) * v.K + k];
  uint8x16_t r;
  switch (v.depth[t]) {
    case 5: r = vqtbl2q_u8({{vld1q_u8(table), vld1q_u8(table + 16)}}, idx); break;
    case 6: r = vqtbl4q_u8(piece64(table, 0), idx); break;
    case 7:
    case 8:
      r = vqtbl4q_u8(piece64(table, 0), idx);
      for (int p = 1; p < (v.depth[t] == 7 ? 2 : 4); ++p) {
        r = vqtbx4q_u8(r, piece64(table, p), vsubq_u8(idx, vdupq_n_u8(static_cast<uint8_t>(64 * p))));
      }
      break;
    default: r = vqtbl1q_u8(vld1q_u8(table), idx); break;  // depth 0..4
  }
  return vreinterpretq_s8_u8(r);
}

void accumulate_int8(const OTEView& v, const uint8_t* codes, size_t n_obs, double* margins) {
  const uint32_t K = v.K;
  Scratch<uint8_t> conds(16 * static_cast<size_t>(v.n_conds));
  size_t i0 = 0;
  for (; i0 + 16 <= n_obs; i0 += 16) {
    cond_vecs(v, codes, n_obs, i0, conds.data());
    for (uint32_t t0 = 0; t0 < v.n_trees; t0 += kInt8TreeBlock) {
      const uint32_t t1 = min_u32(t0 + kInt8TreeBlock, v.n_trees);
      // The leaf bytes of the block's trees, computed once for all outputs.
      uint8x16_t idx[kInt8TreeBlock];
      for (uint32_t t = t0; t < t1; ++t) idx[t - t0] = leaf_vec(v, conds.data(), t);
      const double unit = v.block_unit[t0 / kInt8TreeBlock];
      for (uint32_t k = 0; k < K; ++k) {
        // q * mult is at most 127 * 128 in magnitude, so it fits int16; the
        // block sum (at most 2^21) is kept in int32, observations 4i .. 4i + 3 in sum[i].
        int32x4_t sum[4] = {vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0), vdupq_n_s32(0)};
        for (uint32_t t = t0; t < t1; ++t) {
          const int8x16_t q = tree_codes16(v, t, k, idx[t - t0]);
          const int16_t mult = v.tree_mult[t];
          const int16x8_t lo = vmulq_n_s16(vmovl_s8(vget_low_s8(q)), mult);
          const int16x8_t hi = vmulq_n_s16(vmovl_s8(vget_high_s8(q)), mult);
          sum[0] = vaddw_s16(sum[0], vget_low_s16(lo));
          sum[1] = vaddw_s16(sum[1], vget_high_s16(lo));
          sum[2] = vaddw_s16(sum[2], vget_low_s16(hi));
          sum[3] = vaddw_s16(sum[3], vget_high_s16(hi));
        }
        int32_t s[16];
        for (int i = 0; i < 4; ++i) vst1q_s32(s + 4 * i, sum[i]);
        // unit * sum is exact; the add is the one rounding, as in the scalar code.
        for (int j = 0; j < 16; ++j) margins[(i0 + j) * K + k] += unit * static_cast<double>(s[j]);
      }
    }
  }
  scalar::accumulate_tail(v, codes, n_obs, i0, margins, scalar::leaves_one);
}

}  // namespace

void accumulate(const OTEView& v, const uint8_t* codes, size_t n_obs, double* margins) {
  if (v.leaf_type == LeafType::Int8) {
    accumulate_int8(v, codes, n_obs, margins);
  } else {
    scalar::accumulate(v, codes, n_obs, margins);
  }
}

void leaf_index(const OTEView& v, const uint8_t* codes, size_t n_obs, uint8_t* out) {
  const uint32_t T = v.n_trees;
  Scratch<uint8_t> conds(16 * static_cast<size_t>(v.n_conds));
  size_t i0 = 0;
  for (; i0 + 16 <= n_obs; i0 += 16) {
    cond_vecs(v, codes, n_obs, i0, conds.data());
    for (uint32_t t = 0; t < T; ++t) {
      uint8_t idx[16];
      vst1q_u8(idx, leaf_vec(v, conds.data(), t));
      for (int j = 0; j < 16; ++j) out[(i0 + static_cast<size_t>(j)) * T + t] = idx[j];
    }
  }
  scalar::leaf_index_tail(v, codes, n_obs, i0, out, scalar::leaves_one);
}

}  // namespace neon
}  // namespace boxwood
