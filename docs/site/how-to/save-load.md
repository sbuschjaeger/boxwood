# Save and load a model

An `ObliviousTreeEnsemble` pickles, and so do the estimators; joblib works as well.

```python
import pickle

import numpy as np
from catboost import CatBoostRegressor
import boxwood as bw

rng = np.random.default_rng(9)
X = rng.normal(size=(2000, 5)).astype(np.float32)
y = X[:, 0] + rng.normal(scale=0.1, size=len(X))
ote = bw.ObliviousTreeEnsemble.from_catboost(
    CatBoostRegressor(iterations=100, depth=4, verbose=0).fit(X, y))

data = pickle.dumps(ote)
again = pickle.loads(data)
print(np.array_equal(again.predict_margin(X), ote.predict_margin(X)))   # True
```

Under the hood, pickle uses `to_bytes` / `from_bytes`, which you can also call directly, for example to store a
model in a database or ship it to a C++ program:

```python
blob = ote.to_bytes()
print(len(blob))                                   # about 10 kB for this model
same = bw.ObliviousTreeEnsemble.from_bytes(blob)
print(same.to_bytes() == blob)                     # True
```

The bytes hold the whole model: the encoder (borders, categories, CTR tables), the trees, the leaves (float32 or
int8), the bias and the state of the last fit, so `update` continues exactly where it stopped. Floats are copied
raw, so a round trip is bit-exact. `from_bytes` checks every count and index and raises `ValueError` for
damaged, truncated or foreign bytes. The format is described in [Serialized format](../reference/format.md).

The backend and thread count are not saved; a loaded model uses the best backend and one thread.

boxwood does not write CatBoost model files (there is no exporter yet).
