# Performance

Measured with [`experiments/bench_predict.py`](../how-to/benchmark.md) on an AMD Ryzen AI Max+ 395 (Zen 5) under
Linux, GCC 15.2, CatBoost 1.2.10. Each model has 1,000 trees and 20 numeric features; the times are `predict_margin`
against CatBoost's `predict(prediction_type="RawFormulaVal")` on the same model, best of 5. The number in
parentheses is the speedup over CatBoost with the same number of threads.

## 50,000 observations, one thread

| model | CatBoost | avx512 float32 | avx512 int8 | avx512_vbmi int8 | avx2 float32 | avx2 int8 | scalar float32 |
|---|---|---|---|---|---|---|---|
| regression, depth 4 | 22.7 ms | 6.6 ms (3.4×) | 4.9 ms (4.7×) | 4.5 ms (5.0×) | 12.1 ms (1.9×) | 8.4 ms (2.7×) | 56.9 ms (0.4×) |
| regression, depth 6 | 26.1 ms | 8.3 ms (3.1×) | 6.1 ms (4.3×) | 5.6 ms (4.7×) | 16.5 ms (1.6×) | 11.1 ms (2.4×) | 62.2 ms (0.4×) |
| regression, depth 8 | 29.8 ms | 13.6 ms (2.2×) | 9.2 ms (3.2×) | 7.3 ms (4.1×) | 20.2 ms (1.5×) | 18.3 ms (1.6×) | 66.2 ms (0.5×) |
| 3 classes, depth 6 | 66.3 ms | 15.7 ms (4.2×) | 9.4 ms (7.0×) | 8.1 ms (8.2×) | 35.5 ms (1.9×) | 17.0 ms (3.9×) | 80.2 ms (0.8×) |
| depth 6 + 4 categorical columns | 178.6 ms | 65.9 ms (2.7×) | 63.5 ms (2.8×) | 57.3 ms (3.1×) | 74.4 ms (2.4×) | 69.3 ms (2.6×) | 119.3 ms (1.5×) |

## 50,000 observations, 8 threads

| model | CatBoost | avx512 float32 | avx512 int8 | avx512_vbmi int8 | avx2 float32 | avx2 int8 |
|---|---|---|---|---|---|---|
| regression, depth 4 | 4.7 ms | 1.3 ms (3.6×) | 1.0 ms (4.5×) | 0.97 ms (4.9×) | 2.3 ms (2.0×) | 1.7 ms (2.8×) |
| regression, depth 6 | 5.5 ms | 1.6 ms (3.4×) | 1.3 ms (4.4×) | 1.2 ms (4.7×) | 3.2 ms (1.7×) | 2.2 ms (2.5×) |
| regression, depth 8 | 6.5 ms | 2.5 ms (2.6×) | 1.8 ms (3.7×) | 1.4 ms (4.5×) | 3.9 ms (1.7×) | 3.4 ms (1.9×) |
| 3 classes, depth 6 | 12.7 ms | 3.2 ms (4.0×) | 1.7 ms (7.6×) | 1.6 ms (8.0×) | 6.7 ms (1.9×) | 2.5 ms (5.1×) |
| depth 6 + 4 categorical columns | 50.1 ms | 18.5 ms (2.7×) | 15.2 ms (3.3×) | 14.2 ms (3.5×) | 20.0 ms (2.5×) | 16.1 ms (3.1×) |

## Small batches, one thread

| model | observations | CatBoost | avx512 float32 | avx512 int8 | avx512_vbmi int8 | avx2 float32 |
|---|---|---|---|---|---|---|
| regression, depth 6 | 1 | 102 µs | 20 µs (5.1×) | 18 µs (5.8×) | 18 µs (5.8×) | 20 µs (5.0×) |
| regression, depth 6 | 100 | 186 µs | 25 µs (7.4×) | 21 µs (9.0×) | 20 µs (9.5×) | 52 µs (3.6×) |
| 3 classes, depth 6 | 100 | 215 µs | 47 µs (4.6×) | 29 µs (7.4×) | 26 µs (8.3×) | 102 µs (2.1×) |
| depth 6 + 4 categorical columns | 100 | 1006 µs | 479 µs (2.1×) | 475 µs (2.1×) | 460 µs (2.2×) | 480 µs (2.1×) |

## Notes

- `avx512_vbmi` differs from `avx512` in the int8 path and in the CTR code lookup; on numeric models its float32
  times are those of `avx512`.
- The scalar backend is the fallback for CPUs without AVX2 (and for float32 leaves on ARM); it is plain C++ and
  slower than CatBoost's own SSE code. The `neon` backend (64-bit ARM, int8 leaves) is tested in qemu; its times
  on real hardware are not measured yet.
- With categorical features most of the time is the CTR encoding (hashing the category combinations and looking
  them up). Only the code lookup of `avx512_vbmi` is vectorised, so the encoding depends little on the leaf type
  or the backend.
- Int8 margins differ from float32 margins by at most half the sum of the int8 steps; see
  [Numerics and int8 leaves](../explanations/numerics.md).
