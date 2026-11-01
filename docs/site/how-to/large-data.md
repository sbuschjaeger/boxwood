# Fit on data larger than memory

`fit` reads the features once, to find every observation's leaf in every tree. After that it works only on these leaf
indices (one byte per observation and tree), the targets and the weights. So the features need not be in memory: they can
be a `numpy.memmap`. For data that arrives over time, or that is too large even for the leaf indices, fit on a
part and pass the rest to `update`, window by window.

## A memory-mapped array

```python
import tempfile
from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor
import boxwood as bw

rng = np.random.default_rng(8)
X = rng.normal(size=(60000, 8)).astype(np.float32)
y = np.sin(2 * X[:, 0]) + X[:, 1] * X[:, 2] + rng.normal(scale=0.5, size=len(X))
path = Path(tempfile.mkdtemp()) / "X.f32"
X[:50000].tofile(path)                                  # in practice: written by your data pipeline
X_train = np.memmap(path, dtype=np.float32, mode="r", shape=(50000, 8))
X_hold, y_hold = X[50000:], y[50000:]
cb = CatBoostRegressor(iterations=300, depth=6, verbose=0, random_seed=0).fit(X[:3000], y[:3000])

ote = bw.ObliviousTreeEnsemble.from_catboost(cb)
report = ote.fit(X_train, y[:50000], X_hold, y_hold, strategy="path")
print(report["lambda"], report["holdout_loss"])
```

An all-numeric float32 array in C order is read in place. Any other array (another dtype, categorical columns, a
DataFrame) is converted to float32 first, which is a copy in memory. The holdout is read the same way.

Memory of a fit: `n × n_trees` bytes of leaf indices plus 8 bytes per observation for the weights and `8 × K` bytes per
observation for the targets, for the training observations and the holdout. 10 million observations and 1,000 trees take 10 GB of leaf
indices.

## A stream: fit, then update

`update(X, y, steps)` adapts the leaf values to one window of observations with the optimizer, λ and scale of the last
`fit`, anchored at the current leaves. Call it once per window as the data comes in:

```python
stream = bw.ObliviousTreeEnsemble.from_catboost(cb)
stream.fit(X[:10000], y[:10000], X_hold, y_hold, strategy="path")
for start in range(10000, 50000, 10000):
    rep = stream.update(X[start:start + 10000], y[start:start + 10000], steps=5)
    print(rep["objective_start"], rep["objective"])
```

Each call lowers the objective of its window (here from about 0.080 to 0.072). A window sees only its own observations:
unlike a `fit`, an update does not choose λ on a holdout and does not fall back to the start. With
`optimizer="adam"` in the `fit`, Adam keeps its moments in the ensemble (and in its bytes) from window to window;
its fixed step size (0.01 by default) can then raise a window's objective once the leaves are close to the
optimum. See [Update a model online](../tutorials/online-update.md).
