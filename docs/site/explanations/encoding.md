# Encoding CatBoost's features

Before the trees run, boxwood turns each observation into **one byte per split feature**. A split feature is the input
of a split: a numeric column with its borders, a one-hot categorical feature, or a CTR feature. After the
encoding, every split of every tree is a byte comparison (`code > arg` or `code == arg`), and the prediction code
knows nothing about categories, strings or CTRs.

The encoding reproduces CatBoost's predictor exactly; the import oracle checks the margins against CatBoost on
all 51 TabArena-Lite tasks.

## Numeric features

CatBoost quantizes each numeric feature with borders \(b_1 < \dots < b_m\) (254 by default; boxwood takes up to
255), and every split asks
"\(x > b_j\)". The code of a value is the number of borders below it,

\[
\operatorname{code}(x) = \#\{\, j : x > b_j \,\},
\]

so "\(x > b_j\)" is the same as "\(\operatorname{code}(x) > j - 1\)". The comparison is in float32, as in
CatBoost. A missing value (NaN) gets code 0 (CatBoost's `nan_mode = "Min"`, every split false) or \(m\)
(`"Max"`, every split true).

boxwood stores each feature's borders as a complete binary search tree of 8 levels in breadth-first order (padded
with \(+\infty\) to 255 values). The code is then found by 8 branch-free steps,
\(h \leftarrow 2h + 1 + [x > \text{node}_h]\), and the SIMD backends hold whole levels of that search tree in
registers.

## Categorical features

A category value is a string (integers are turned into strings, as CatBoost does). It is hashed once with
CatBoost's 32-bit CityHash-based hash. Two kinds of split features use the hash.

**One-hot features.** For categorical features with few values CatBoost splits on "value = v". The code is
1 + the position of the observation's hash in the feature's list of values, 0 for a value not in the list; the split is
`code == arg`.

**CTR features.** For the others CatBoost uses CTRs ("counter" target statistics): a number per category that
summarises the target of the training observations with that category, for example the share of positive labels. A CTR
can also be of a **combination** of several categorical features and binarized numeric features (a
"projection"): CatBoost builds combinations greedily while it grows the trees.

For each observation and each projection, boxwood combines the parts the way CatBoost does,

\[
\text{key} \leftarrow M \cdot (\text{key} + M \cdot v) \pmod{2^{64}}, \qquad M = \texttt{0x4906BA494954CB65},
\]

over the parts \(v\) of the projection (a category hash, or a 0/1 border result). The key is looked up in the
CTR table that CatBoost saved with the model; the table holds the statistics of each key, from which the CTR
value follows with CatBoost's prior and formula, for example for the `Borders` type

\[
\text{ctr} = \Big(\frac{\text{positives} + p_{\text{num}}}{\text{total} + p_{\text{den}}} + \text{shift}\Big)
\cdot \text{scale},
\]

in float32. A key that is not in the table gets the value of empty statistics. The code is the number of the
CTR's borders below the value, as for numeric features.

## Precomputed at import

Everything that does not depend on the observation is computed once, when the model is loaded:

- the code of every CTR table entry, so prediction never computes a CTR value: projection key → table slot →
  code;
- an open-addressing hash table per CTR table (at most half full), so a lookup is usually one probe; a binary
  search over a large table would be a chain of cache misses;
- the codes of all CTRs that share a table, by slot, so one probe serves all of them;
- an order of the tables sorted by the parts of their projections: many projections share their first parts
  (the same two categorical columns, then different borders), and in this order each projection key is
  computed from the key of the prefix it shares with the one before, which halves the hashing on the
  benchmark model below.

At prediction time boxwood hashes each distinct category value once per call (from 256 observations on, the Python
layer passes each column dictionary-encoded; a smaller batch goes as it is, every value its own entry), computes the projection keys column by column over a block of 256 observations, and probes
each table for all observations of the block. The 0/1 parts are mixed in without branches (a border test is often
close to a coin flip). The probes go in rounds, also without a branch per observation: every observation checks its first
slot, then each round moves the observations whose key is elsewhere one slot on (a probe loop per observation would mispredict
its exit for about one observation in eight). With AVX-512 VBMI, the codes of 64 observations are two byte permutes over the
slot codes of a table of up to 256 slots, which most tables are. Gathering the keys with AVX-512 or AVX2
gathers was slower than scalar loads on Zen 5. On the categorical benchmark model (4 categorical columns, 651
CTR features over 528 tables) this is 2.4–3.1× faster than CatBoost on one thread.
