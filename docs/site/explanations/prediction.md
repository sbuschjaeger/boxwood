# How prediction is fast

A prediction has two parts: the **encoding** of each observation into one byte per split feature (see
[Encoding](encoding.md)), and the **accumulation**: for every tree, find the observation's leaf and add its value. For a
model of 1,000 depth-6 trees the accumulation is 6,000 comparisons and 1,000 table lookups per observation; that is where
the time goes.

## Blocks of observations

boxwood processes observations in blocks of 256. The codes of a block are stored feature-major (all 256 codes of a
feature next to each other), so one vector load reads the codes of 32 (AVX2) or 64 (AVX-512) observations of one feature.
A block's codes are a few kB and stay in the L1 cache while all trees run over them.

Threads (`set_threads`) take contiguous ranges of blocks; each block writes only its own observations of the output, so
the threads share nothing but the read-only model.

## Shared split conditions

A split is a condition "code of feature \(f\) > arg" (or "= arg" for one-hot features). In a typical CatBoost
model only a quarter to 40 % of the splits are distinct: many trees split on the same border. boxwood
precomputes the distinct conditions at load time and, for each group of 64 observations, evaluates each of them once:
one byte comparison gives a 64-bit mask (AVX-512) or a 32-byte vector (AVX2). A tree level then only reads its
condition's result and sets its bit in the leaf index:

\[
\operatorname{leaf}_t(x) = \sum_l 2^l \, c_{t,l}(x).
\]

The leaf indices of 64 observations are 64 bytes, one vector register.

This is the part of the QuickScorer idea (evaluate each threshold once and share the result across trees) that
carries over to oblivious trees. QuickScorer itself solves a different problem: finding the exit leaf of
irregular trees without branches. For an oblivious tree the leaf index is already a few byte operations.

## Leaf values from registers

The leaf values of one tree and one output are a small table of \(2^d\) entries. Instead of loading each observation's
value from memory (a gather), boxwood holds the table in registers and uses permute instructions:

- **AVX-512, float32:** `vpermt2ps` looks up 32 floats for 16 observations in one instruction; deeper trees combine
  several permutes with masked blends on the higher index bits.
- **AVX2, float32:** `vpermps` covers 8 floats; for trees up to depth 5 permutes plus blends are fastest, deeper
  trees load their values one observation at a time (AVX2 gathers are slower than that), into eight independent
  float64 sums of four observations each so the additions do not wait for each other.
- **int8 leaves:** `pshufb` looks up 16 bytes for 32 or 64 observations at once. The codes of four trees are then
  interleaved so that one 32-bit lane holds the four codes of one observation, and `pmaddubsw` / `pmaddwd` multiply
  them by the trees' power-of-two factors and add them in one step (see [Numerics](numerics.md)). With
  AVX-512 VBMI, `vpermb` / `vpermt2b` look up 64 or 128 bytes per instruction, and VNNI's `vpdpbusd` does the
  multiply-add in one instruction.

## The float64 sum

Each tree's leaf value (float32) is added to a float64 margin, in tree order. The SIMD backends keep the
float64 sums of a group of observations in registers and do exactly the same additions in the same order as the plain
C++ backend, so all backends give the same bits.

## Small batches

The vector code works on whole groups of 32 or 64 observations. The observations of a partial group at the end of a batch would
go through the scalar code, which is many times slower per observation; boxwood pads a partial group of at least 8
observations to a full one instead. A batch of 100 observations then costs about as much as 128 observations. One observation still goes
through the scalar code, which is faster for a single observation.

## Backends

| backend | needs | used on |
|---|---|---|
| `avx512_vbmi` | AVX-512 F, BW, VL, VBMI, VNNI | Intel Ice Lake and newer, AMD Zen 4 and newer |
| `avx512` | AVX-512 F, BW, VL | Intel Skylake-X and newer servers, Ice Lake and newer clients, AMD Zen 4 and newer |
| `avx2` | AVX2, FMA | x86-64 CPUs since about 2013 |
| `neon` | AArch64 (always present) | 64-bit ARM; int8 leaves only, float32 leaves and the encoding use the scalar code |
| `scalar` | nothing | everything else |

Each backend is one C++ file compiled with its own flags (`avx512_vbmi` is the `avx512` file compiled again
with VBMI and VNNI; only its int8 path and the CTR code lookup differ); the fastest one the CPU supports is picked at run time. The categorical encoding (hashes and CTR table probes) is plain C++ shared by all backends; `avx512_vbmi` looks up the CTR codes with byte permutes.
