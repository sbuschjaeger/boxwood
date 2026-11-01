// Categorical encoding shared by all backends: one-hot codes, projection hashes
// and CTR table lookups. No checks and no exceptions in this code.

#include "kernels.hpp"

#include <algorithm>
#include <vector>

namespace boxwood {
namespace scalar {
namespace {

constexpr uint64_t kMix = 0x4906BA494954CB65ULL;  // CatBoost's CalcHash multiplier
// CatBoost's step acc = M * (acc + M * v) is M * acc + M^2 * v (mod 2^64): two
// independent multiplies, and for a 0/1 part only one.
constexpr uint64_t kMix2 = kMix * kMix;

/// Mixes element e of a projection into the hashes of the observations of a block:
/// out[i] = M * in[i] + M^2 * v_i. One pass over the observations per element, so the
/// element kind is decided once per element, not per observation.
void mix_element(const OTEView& v, const float* X, const int32_t* cat_hashes, size_t n_obs, uint32_t e,
                 const uint64_t* in, uint64_t* out) {
  const uint32_t col = v.elem_column[e];
  switch (v.elem_kind[e]) {
    case 0:  // category hash, sign-extended to 64 bits
      for (size_t i = 0; i < n_obs; ++i) {
        const int64_t h = cat_hashes[i * v.n_categorical + col];
        out[i] = kMix * in[i] + kMix2 * static_cast<uint64_t>(h);
      }
      break;
    // The bits of the two other kinds are data-dependent and often close to
    // a coin flip, so they are mixed in by a multiply, not a branch.
    case 1: {  // float border with the feature's NaN treatment
      // x > border is false for NaN; !(x <= border) is x > border or NaN.
      const float border = v.elem_border[e];
      const bool nan_high = v.elem_nan_high[e] != 0;
      for (size_t i = 0; i < n_obs; ++i) {
        const float x = X[i * v.n_numeric + col];
        const uint64_t bit = nan_high ? !(x <= border) : x > border;
        out[i] = kMix * in[i] + kMix2 * bit;
      }
      break;
    }
    default: {  // category equals value
      const int32_t value = v.elem_value[e];
      for (size_t i = 0; i < n_obs; ++i) {
        const uint64_t bit = static_cast<uint64_t>(cat_hashes[i * v.n_categorical + col] == value);
        out[i] = kMix * in[i] + kMix2 * bit;
      }
      break;
    }
  }
}

}  // namespace

void ctr_slots(const OTEView& v, uint32_t q, const uint64_t* hash, size_t n_obs, uint32_t* slot) {
  const uint64_t* keys = v.table_slot_keys + v.table_slot_begin[q];
  const uint32_t mask = v.table_slot_begin[q + 1] - v.table_slot_begin[q] - 1;
  const uint64_t empty = v.table_empty_key[q];
  // Most keys sit in their first slot. A probe loop per observation mispredicts its
  // exit whenever the probe count varies, so the probes go in rounds without
  // a branch per observation: each round moves every observation still in `more` one slot on
  // and keeps those whose key is not there yet. The table is never full, so
  // a probe ends at the key or a free slot. The probes of the observations are
  // independent, so their cache misses overlap.
  uint32_t more[kBlockSize];
  uint32_t n_more = 0;
  for (size_t i = 0; i < n_obs; ++i) {
    const uint32_t s = ctr_slot(hash[i], mask);
    slot[i] = s;
    more[n_more] = static_cast<uint32_t>(i);
    n_more += (keys[s] != hash[i] && keys[s] != empty) ? 1u : 0u;
  }
  while (n_more > 0) {
    uint32_t n_next = 0;
    for (uint32_t m = 0; m < n_more; ++m) {
      const uint32_t i = more[m];
      const uint32_t s = (slot[i] + 1) & mask;
      slot[i] = s;
      more[n_next] = i;
      n_next += (keys[s] != hash[i] && keys[s] != empty) ? 1u : 0u;
    }
    n_more = n_next;
  }
}

void ctr_codes_tail(const OTEView& v, uint32_t q, const uint32_t* slot, size_t n_obs, size_t i0, uint8_t* codes) {
  for (uint32_t c = v.table_ctr_begin[q]; c < v.table_ctr_begin[q + 1]; ++c) {
    const uint32_t r = v.table_ctrs[c];
    const uint8_t* slot_codes = v.ctr_slot_codes + v.ctr_slot_code_begin[r];
    uint8_t* out = codes + static_cast<size_t>(v.n_numeric_features + v.n_onehot + r) * n_obs;
    for (size_t i = i0; i < n_obs; ++i) out[i] = slot_codes[slot[i]];
  }
}

void ctr_lookup(const OTEView& v, uint32_t q, const uint64_t* hash, size_t n_obs, uint8_t* codes) {
  uint32_t slot[kBlockSize];
  ctr_slots(v, q, hash, n_obs, slot);
  ctr_codes_tail(v, q, slot, n_obs, 0, codes);
}

void encode_categorical(const OTEView& v, const float* X, const int32_t* cat_hashes, size_t n_obs,
                        CtrLookup lookup, uint8_t* codes) {
  const uint32_t F = v.n_numeric_features;
  for (uint32_t g = 0; g < v.n_onehot; ++g) {
    const int32_t* b = v.onehot_values + v.onehot_value_begin[g];
    const int32_t* e = v.onehot_values + v.onehot_value_begin[g + 1];
    uint8_t* out = codes + static_cast<size_t>(F + g) * n_obs;
    for (size_t i = 0; i < n_obs; ++i) {
      const int32_t* hit = std::find(b, e, cat_hashes[i * v.n_categorical + v.onehot_column[g]]);
      out[i] = hit != e ? static_cast<uint8_t>(hit - b + 1) : 0;
    }
  }
  // Table by table: the projection hashes of all observations, then the lookup of
  // all observations. level[d * n_obs + i] is the hash of observation i over the first d
  // elements of the current projection (level 0: 0, the start of the hash);
  // the tables come in an order where each projection only adds the elements
  // after the prefix it shares with the one before (see table_order).
  uint32_t max_size = 0;
  for (uint32_t p = 0; p < v.n_projections; ++p) {
    max_size = std::max(max_size, v.proj_elem_begin[p + 1] - v.proj_elem_begin[p]);
  }
  std::vector<uint64_t> level((max_size + 1) * n_obs, 0);
  for (uint32_t j = 0; j < v.n_tables; ++j) {
    const uint32_t q = v.table_order[j];
    const uint32_t p = v.table_projection[q];
    const uint32_t first = v.proj_elem_begin[p];
    const uint32_t size = v.proj_elem_begin[p + 1] - first;
    for (uint32_t d = v.table_shared[j]; d < size; ++d) {
      mix_element(v, X, cat_hashes, n_obs, first + d, level.data() + d * n_obs, level.data() + (d + 1) * n_obs);
    }
    lookup(v, q, level.data() + size * n_obs, n_obs, codes);
  }
}

}  // namespace scalar
}  // namespace boxwood
