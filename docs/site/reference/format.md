# Serialized format

`to_bytes` writes, and `from_bytes` reads, one little-endian byte string (version 2). Pickle uses it, so this is
also what a pickled `ObliviousTreeEnsemble` holds.

## Layout

A header of scalars, then arrays. Every array is a `uint64` element count followed by the raw elements. Floats are
copied raw, so a round trip is bit-exact.

| field | type | content |
|---|---|---|
| magic | 4 bytes | `BXWD` |
| version | uint32 | 2 |
| task | uint8 | 0 regression, 1 binary, 2 multiclass |
| leaf type | uint8 | 0 float32, 1 int8 |
| K | uint32 | number of outputs |
| n_numeric, n_categorical | uint32 each | input columns |
| `numeric_flat_index`, `categorical_flat_index` | uint32[] | CatBoost column of each input feature |
| `numeric_column`, `border_count`, `nan_high`, `borders` | uint32[], uint32[], uint8[], float[] | numeric split features |
| `onehot_column`, `onehot_count`, `onehot_values` | uint32[], uint32[], int32[] | one-hot split features (category hashes) |
| `proj_size`, `elem_kind`, `elem_column`, `elem_border`, `elem_nan_high`, `elem_value` | | CTR projections and their elements |
| `table_projection`, `table_size`, `table_keys` | uint32[], uint32[], uint64[] | CTR tables, keys ascending per table |
| `ctr_table`, `ctr_default_code`, `ctr_codes` | uint32[], uint8[], uint8[] | CTR split features: the code of every table entry |
| `depth`, `split_feature`, `split_kind`, `split_arg` | uint8[], uint32[], uint8[], uint8[] | the trees |
| leaves | float[] (float32) or int8[] + float[] (int8: codes, one step per tree) | \(2^{d_t} \times K\) values per tree, leaf-major |
| bias | double[] | K values |
| fit state | double, double, uint8, double, int64, double[], double[] | λ, σ, optimizer and learning rate of the last fit; Adam's step count and moments |

Derived tables (the border search trees, the CTR hash tables, the split conditions, the per-output leaf tables)
are rebuilt when the bytes are loaded; they are not stored.

## Checks

`from_bytes` checks the magic and the version, that every array fits the remaining bytes, that all counts agree
(for example, the leaves match the tree depths), that every index is in range, that the CTR keys are ascending,
and, for int8 leaves, that the steps of every block of 64 trees follow the int8 rule. Any violation raises
`boxwood.Error` (`ValueError`). Bytes of another version are rejected.
