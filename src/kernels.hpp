#pragma once

#include <cstddef>
#include <cstdint>

#include "boxwood/ensemble.hpp"  // LeafType

namespace boxwood {

/// First slot to probe for `key` in a table of mask + 1 slots. The keys are
/// CatBoost hashes, but their low bits can be weak (the hash multiplies), so
/// they are mixed once more and the high half is used.
inline uint32_t ctr_slot(uint64_t key, uint32_t mask) {
  return static_cast<uint32_t>((key * 0x9E3779B97F4A7C15ULL) >> 32) & mask;
}

/// Raw pointers into the ensemble, built by the ensemble; the kernels know nothing
/// else. All arrays are the stored arrays of the ensemble (see ensemble.hpp).
/// Split features are numbered: first the numeric ones, then the one-hot ones,
/// then the CTR ones.
///
/// The kernels work on a block of at most kBlockSize observations. The codes of a block are
/// feature-major: codes[f * n_obs + i] is the code of split feature f of observation i,
/// so a SIMD backend reads the codes of many observations of one feature with one load.
/// Nothing outside a block sees the codes.
struct OTEView {
  uint32_t K = 0;
  uint32_t n_trees = 0;
  uint32_t n_features = 0;       // all split features
  uint32_t n_numeric_features = 0;  // the numeric ones, the first F codes
  uint32_t n_numeric = 0;        // columns of the input matrix
  uint32_t n_categorical = 0;    // categorical columns of the input
  const uint32_t* numeric_column = nullptr;
  const uint32_t* border_begin = nullptr;  // F + 1, into the stored borders
  const float* border_heap = nullptr;  // F * 256: each feature's search tree (see encode_numeric)
  const uint8_t* nan_high = nullptr;

  // One-hot split features: code = 1 + position of the observation's signed hash in the
  // values, 0 if the value is not there.
  uint32_t n_onehot = 0;
  const uint32_t* onehot_column = nullptr;      // categorical feature index
  const uint32_t* onehot_value_begin = nullptr;  // G + 1, into onehot_values
  const int32_t* onehot_values = nullptr;

  // CTR projections: the elements that build the combination hash, one byte of
  // kind per element, structure of arrays, elements of all projections concatenated.
  uint32_t n_projections = 0;
  const uint32_t* proj_elem_begin = nullptr;  // P + 1, into the elem arrays
  const uint8_t* elem_kind = nullptr;  // 0 = category hash, 1 = float border, 2 = category equals value
  const uint32_t* elem_column = nullptr;  // categorical index (kinds 0, 2), numeric column (kind 1)
  const float* elem_border = nullptr;     // kind 1
  const uint8_t* elem_nan_high = nullptr;  // kind 1: the float feature's NaN treatment
  const int32_t* elem_value = nullptr;     // kind 2

  // CTR tables: each table's keys in an open-addressing hash table of a
  // power-of-two size (linear probing from ctr_slot), so a lookup is one probe
  // in most cases (a binary search over a large table is a chain of cache
  // misses). A free slot holds the table's empty key, which is not a key.
  uint32_t n_tables = 0;
  const uint32_t* table_projection = nullptr;  // projection of each table
  const uint32_t* table_slot_begin = nullptr;  // Q + 1, into table_slot_keys
  const uint64_t* table_slot_keys = nullptr;
  const uint64_t* table_empty_key = nullptr;   // Q
  const uint32_t* table_ctr_begin = nullptr;   // Q + 1, into table_ctrs
  const uint32_t* table_ctrs = nullptr;        // the CTRs, grouped by table
  // The order in which the kernels hash the tables' projections: sorted by
  // their elements, so a projection shares the longest possible prefix of
  // elements with the one before; table_shared[j] is the length of that
  // prefix for table table_order[j] (0 for j = 0).
  const uint32_t* table_order = nullptr;   // Q
  const uint32_t* table_shared = nullptr;  // Q

  // CTR split features: the code of every slot of the CTR's table (the
  // default code at free slots), from the codes computed at import. The code
  // of CTR r at slot s of its table is ctr_slot_codes[ctr_slot_code_begin[r] + s]:
  // the CTRs of a table share its slots, and each has its own codes for them.
  uint32_t n_ctr = 0;
  const uint32_t* ctr_table = nullptr;  // table of each CTR
  const uint32_t* ctr_slot_code_begin = nullptr;  // R + 1, into ctr_slot_codes
  const uint8_t* ctr_slot_codes = nullptr;

  const uint8_t* depth = nullptr;
  const uint32_t* split_begin = nullptr;  // T + 1, into split_cond

  // The splits as the distinct split conditions (feature, kind, arg), sorted,
  // and the condition of every split: many trees split on the same border, so
  // the SIMD backends compare the codes once per condition and observation group, and
  // a tree level only reads its condition's result.
  uint32_t n_conds = 0;
  const uint32_t* cond_feature = nullptr;
  const uint8_t* cond_kind = nullptr;
  const uint8_t* cond_arg = nullptr;
  const uint32_t* split_cond = nullptr;

  // The splits again as 8 byte slots per tree (tree t: slots 8t .. 8t + 7, slot l = level l), for the
  // single-observation kernels (leaves_one). Levels beyond a tree's depth, and the trees that pad the
  // count to a multiple of 8, hold a split that is never true (code > 255). So the outcomes of the 64
  // slots of 8 trees form one 64-bit mask whose byte j is the leaf index of tree j. Null if there are
  // more than 256 split features (a slot's feature is one byte).
  const uint8_t* slot_feature = nullptr;
  const uint8_t* slot_arg = nullptr;
  const uint64_t* slot_eq = nullptr;  // per 8 trees: bit s set = slot s tests code == arg, else code > arg
  LeafType leaf_type = LeafType::Float32;

  // Int8 leaves: the value of code q of tree t is q * unit * mult[t], where
  // unit = block_unit[t / kInt8TreeBlock] and mult[t] is a power of two in
  // 1 .. 128 (the tree's step is unit * mult[t]). So the leaves of a block of
  // trees add up as integers: sum_t q_t * mult[t] is exact in int32, and
  // unit * sum is exact in float64.
  const uint8_t* tree_mult = nullptr;  // T
  const double* block_unit = nullptr;  // one per block of kInt8TreeBlock trees

  // The leaf tables every backend reads: for each tree t and output k, the
  // L = 2^depth[t] leaf values of output k (Float32) or their int8 codes
  // (Int8) contiguous, padded with zeros to at least 16 entries so a SIMD
  // register lookup reads a full vector.
  // leaf_out_begin is the prefix-sum array over the T*K pairs in that order.
  const float* leaves_by_output = nullptr;
  const int8_t* leaves_q_by_output = nullptr;
  const uint32_t* leaf_out_begin = nullptr;
};

/// The most observations of one block: the encoded codes of a block stay in
/// cache while the kernels evaluate all trees, and the kernels keep per-block
/// scratch arrays of this size on the stack.
constexpr size_t kBlockSize = 256;

/// Int8 leaves are summed in blocks of this many consecutive trees (the last
/// block may be shorter); the trees of a block share one unit. At most
/// 64 * 127 * 128 < 2^21 per block, so the sum is exact as int32 and its
/// product with the unit (24 bits) exact as float64.
constexpr uint32_t kInt8TreeBlock = 64;

namespace scalar {

// The scalar reference. It is also the shared code of all backends: the SIMD
// backends call it for the observations of a block that do not fill a vector
// group (the *_tail functions), and every backend uses its categorical
// encoding and, except avx512_vbmi, its CTR lookup. The *_tail functions work on observations
// [i0, n_obs) of a block of `n_obs` observations.

/// The leaf index of observation i of a block of `n_obs` observations in every tree, into idx, which has
/// room for n_trees rounded up to a multiple of 8. Each backend has one; the *_tail functions call it for
/// each observation when there are only a few, where a pass over the observations per split would cost
/// more in loop setup than in compares. This one evaluates every distinct split condition once and lets
/// the trees read their results.
void leaves_one(const OTEView& v, const uint8_t* codes, size_t n_obs, size_t i, uint8_t* idx);

/// A backend's leaves_one.
using LeavesOne = void (*)(const OTEView& v, const uint8_t* codes, size_t n_obs, size_t i, uint8_t* idx);

/// Writes the numeric codes of observations [i0, n_obs) (see encode_numeric).
void encode_numeric_tail(const OTEView& v, const float* X, size_t n_obs, size_t i0, uint8_t* codes);

/// Adds the leaf values of every tree to the float64 margins of observations
/// [i0, n_obs) (margins: observations x K of the block). The numeric rule for every
/// backend, so every backend gives the same bits: Float32 adds each tree's
/// leaf value to the margin in tree order; Int8 adds, block by block,
/// unit * (the exact integer sum of q * mult over the block's trees).
/// `one` is the backend's leaves_one.
void accumulate_tail(const OTEView& v, const uint8_t* codes, size_t n_obs, size_t i0, double* margins,
                     LeavesOne one);

/// Writes the leaf indices of observations [i0, n_obs) in every tree (out: observations x
/// n_trees of the block, row-major); `one` is the backend's leaves_one.
void leaf_index_tail(const OTEView& v, const uint8_t* codes, size_t n_obs, size_t i0, uint8_t* out,
                     LeavesOne one);

/// Byte codes for a block of `n_obs` observations, feature-major: codes[f * n_obs + i].
/// The code of a numeric split feature is the number of its borders b with
/// x > b (float32 compare); NaN gets border_count if nan_high else 0.
/// X is the row-major block of `n_obs` input observations. The borders are searched as
/// a complete binary search tree (border_heap, see kernels_scalar.cpp).
void encode_numeric(const OTEView& v, const float* X, size_t n_obs, uint8_t* codes);

/// The slot of every observation's key in table q: slot[i] for the observations i of a block
/// of `n_obs` observations, where hash[i] is observation i's hash of the table's projection.
/// A key that is not in the table gets a free slot (the default code).
void ctr_slots(const OTEView& v, uint32_t q, const uint64_t* hash, size_t n_obs, uint32_t* slot);

/// Writes the code of every CTR of table q (see OTEView::table_ctrs) for observations
/// [i0, n_obs) of a block, from the observations' slots (ctr_slots).
void ctr_codes_tail(const OTEView& v, uint32_t q, const uint32_t* slot, size_t n_obs, size_t i0, uint8_t* codes);

/// Writes the code of every CTR of table q for all observations of a block; hash[i]
/// is observation i's hash of the table's projection.
void ctr_lookup(const OTEView& v, uint32_t q, const uint64_t* hash, size_t n_obs, uint8_t* codes);

/// A backend's ctr_lookup.
using CtrLookup = void (*)(const OTEView& v, uint32_t q, const uint64_t* hash, size_t n_obs, uint8_t* codes);

/// Writes the one-hot and CTR codes of a block into the codes of the split features
/// [n_numeric_features, n_features). cat_hashes is observations x n_categorical,
/// the signed 32-bit hash of every categorical value; X is the row-major block
/// the float elements of the projections read. The projection hashes are
/// shared by all backends; `lookup` (the backend's ctr_lookup) finds them in
/// the tables.
void encode_categorical(const OTEView& v, const float* X, const int32_t* cat_hashes, size_t n_obs,
                        CtrLookup lookup, uint8_t* codes);

/// Adds the leaf values of every tree to the margins: margins is observations x K,
/// float64, and keeps what it holds (see accumulate_tail for the rule).
void accumulate(const OTEView& v, const uint8_t* codes, size_t n_obs, double* margins);

/// Leaf index of every observation of the block in every tree: out is observations x n_trees,
/// row-major.
void leaf_index(const OTEView& v, const uint8_t* codes, size_t n_obs, uint8_t* out);

}  // namespace scalar

namespace avx2 {

/// AVX2 versions of the scalar block kernels; bit-identical results.
void encode_numeric(const OTEView& v, const float* X, size_t n_obs, uint8_t* codes);
void accumulate(const OTEView& v, const uint8_t* codes, size_t n_obs, double* margins);
void leaf_index(const OTEView& v, const uint8_t* codes, size_t n_obs, uint8_t* out);

/// leaves_one with byte shuffles over 16-byte pieces of the observation's codes; also the
/// avx512 backend's (AVX-512 CPUs have AVX2).
void leaves_one(const OTEView& v, const uint8_t* codes, size_t n_obs, size_t i, uint8_t* idx);

}  // namespace avx2

namespace avx512 {

/// AVX-512 versions of the scalar block kernels (F + BW + VL); bit-identical
/// results.
void encode_numeric(const OTEView& v, const float* X, size_t n_obs, uint8_t* codes);
void accumulate(const OTEView& v, const uint8_t* codes, size_t n_obs, double* margins);
void leaf_index(const OTEView& v, const uint8_t* codes, size_t n_obs, uint8_t* out);

}  // namespace avx512

namespace avx512_vbmi {

/// The AVX-512 kernels with VBMI and VNNI for the int8 path and the CTR codes;
/// bit-identical results.
void encode_numeric(const OTEView& v, const float* X, size_t n_obs, uint8_t* codes);
void accumulate(const OTEView& v, const uint8_t* codes, size_t n_obs, double* margins);
void leaf_index(const OTEView& v, const uint8_t* codes, size_t n_obs, uint8_t* out);
void ctr_lookup(const OTEView& v, uint32_t q, const uint64_t* hash, size_t n_obs, uint8_t* codes);

/// leaves_one with one two-table byte permute per 8 trees (up to 128 split features).
void leaves_one(const OTEView& v, const uint8_t* codes, size_t n_obs, size_t i, uint8_t* idx);

}  // namespace avx512_vbmi

namespace neon {

/// NEON versions (AArch64) of the int8 accumulate and the leaf indices;
/// bit-identical results. Float32 leaves and the encoding use the scalar code.
void accumulate(const OTEView& v, const uint8_t* codes, size_t n_obs, double* margins);
void leaf_index(const OTEView& v, const uint8_t* codes, size_t n_obs, uint8_t* out);

}  // namespace neon

}  // namespace boxwood
