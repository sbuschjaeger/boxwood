# Fast inference

boxwood picks the fastest prediction code for the CPU at run time. This tutorial shows the knobs: the backend,
threads, int8 leaves, and how to pass the input so nothing is copied.

## A model to time

```python
import time

import numpy as np
from catboost import CatBoostRegressor
import boxwood as bw

rng = np.random.default_rng(6)
X = rng.normal(size=(50000, 20)).astype(np.float32)
y = X[:, 0] * 2 + X[:, 1] ** 2 + rng.normal(scale=0.1, size=len(X))
cb = CatBoostRegressor(iterations=1000, depth=6, verbose=0, random_seed=0, thread_count=8).fit(X, y)
ote = bw.ObliviousTreeEnsemble.from_catboost(cb)

def best_ms(f, repeat=5):
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        f()
        best = min(best, time.perf_counter() - start)
    return 1e3 * best

print("CatBoost", best_ms(lambda: cb.predict(X, prediction_type="RawFormulaVal", thread_count=1)))
print("boxwood ", best_ms(lambda: ote.predict_margin(X)))
```

## Backends

`bw.backends()` lists the backends this CPU can run, fastest first: `"avx512_vbmi"` (AVX-512 with VBMI
and VNNI, faster int8), `"avx512"` (AVX-512 F, BW and VL), `"avx2"`, `"neon"` on 64-bit ARM (int8 leaves), and
`"scalar"` (plain C++, always there). The default is the first one. All backends give **bit-identical** results, so switching is only a
matter of speed:

```python
print(bw.backends())                      # for example ['avx512_vbmi', 'avx512', 'avx2', 'scalar']
base = ote.predict_margin(X)
for name in bw.backends():
    ote.set_backend(name)
    print(name, best_ms(lambda: ote.predict_margin(X)), np.array_equal(ote.predict_margin(X), base))
ote.set_backend("")                       # back to the best one
```

## Threads

Prediction runs on one thread by default. `set_threads(n)` splits the observations into ranges of 256-observation blocks, one
range per thread (`0` = one thread per hardware thread). The result does not depend on the number of threads.

```python
ote.set_threads(8)
print("8 threads", best_ms(lambda: ote.predict_margin(X)))
ote.set_threads(1)
```

Inputs of fewer than 1,024 observations per thread use fewer threads, because starting a thread costs more than it
saves.

## Int8 leaves

`set_leaf_type("int8")` stores every leaf value as one byte plus one step per tree. The steps of a block of 64
trees are powers of two apart, so the leaves of a block add up as exact integers; this is the fastest leaf type
and a quarter of the leaf memory. The margins move by at most half the sum of the steps (see
[Numerics and int8 leaves](../explanations/numerics.md)):

```python
m32 = ote.predict_margin(X)
ote.set_leaf_type("int8")
m8 = ote.predict_margin(X)
print("int8", best_ms(lambda: ote.predict_margin(X)))
print("largest change of a margin", np.max(np.abs(m8 - m32)))   # about 0.03 here (y has std 2.5)
```

The int8 leaves are part of the saved model. `set_leaf_type("float32")` turns them back into floats (the
dequantized values, not the original ones).

## Small batches and single observations

A call has a fixed cost of a few microseconds; after that the observations are processed in groups of 32 (AVX2) or 64
(AVX-512) observations. A batch that does not fill its last group is padded, so 100 observations cost about as much as 128.
For one observation, boxwood is about 5× faster than CatBoost (about 20 µs against 100 µs for 1,000 trees), mostly
because the Python call is thin.

## Pass the input without a copy

A C-contiguous `float32` numpy array of a numeric model goes to the C++ core without a copy. Other inputs
(float64, a DataFrame, a column subset) are converted to one first; here that adds about 0.1 ms to 5.8 ms:

```python
X64 = X.astype(np.float64)
print("float32 input", best_ms(lambda: ote.predict_margin(X)))
print("float64 input", best_ms(lambda: ote.predict_margin(X64)))
```

Categorical columns are passed as strings. From 256 observations on, boxwood converts each distinct value once per
call; a smaller batch (a DataFrame or a numpy object array) is converted as one block, which skips a fixed cost per
column (see [Encoding CatBoost's features](../explanations/encoding.md)).

The measured speedups over CatBoost for many scenarios are in [Performance](../reference/performance.md).
