# boxwood

boxwood takes a trained CatBoost model and makes it a better deployment artifact. It imports the trees
exactly, can refit the leaf values to your data, prune the model to a budget of split evaluations, update it
online, and predicts several times faster than CatBoost's own predictor.

It is a small C++20 library with a Python package. Prediction is bit-exact on every CPU backend.

## Why boxwood

- **Exact import.** boxwood reads CatBoost's JSON export with all of its feature handling: numeric borders,
  one-hot features, and CTR target statistics of categories and their combinations. The imported model gives
  CatBoost's raw margins (`RawFormulaVal`) within 1e-5 on all 51 TabArena-Lite tasks.
- **Fast prediction.** With AVX-512, prediction is 2–4× faster than CatBoost with float32 leaves and 3.5–6× faster
  with int8 leaves; with AVX2 1.3–2× and 1.8–4×. Small batches and single observations are 5–10× faster. See
  [Performance](reference/performance.md).
- **Better leaves for the same trees.** CatBoost fits each leaf once, greedily. boxwood keeps the splits and
  refits all leaf values together, as one convex problem, with a strength chosen on a holdout. See
  [Refitting the leaves](explanations/leaf-fit.md).
- **A budget for the model.** `split_budget` cuts the ensemble to at most B split evaluations per observation: each tree
  keeps its most useful levels. See [Pruning](explanations/pruning.md).
- **Online updates.** `update` moves the leaf values towards a new window of data, anchored at the current model.

## A first look

```python
import numpy as np
from catboost import CatBoostRegressor
import boxwood as bw

rng = np.random.default_rng(0)
X = rng.normal(size=(5000, 10)).astype(np.float32)
y = X[:, 0] * 2 + X[:, 1] ** 2 + rng.normal(scale=0.3, size=len(X))
cb = CatBoostRegressor(iterations=300, depth=6, verbose=0, random_seed=0).fit(X, y)

ote = bw.ObliviousTreeEnsemble.from_catboost(cb)
margin = ote.predict_margin(X)  # float64, shape (n, 1)
print(np.max(np.abs(margin[:, 0] - cb.predict(X, prediction_type="RawFormulaVal"))))
```

## Where to go next

- [Getting started](getting-started.md): build boxwood and run the first model.
- [Tutorials](tutorials/index.md): step-by-step guides through import, refit, pruning, updates and fast inference.
- [How-to guides](how-to/index.md): short recipes for one task each.
- [Explanations](explanations/index.md): why boxwood works the way it does.
- [Reference](reference/python.md): the Python and C++ API, the serialized format and the benchmarks.

A technical report with the methods and their mathematics will follow.
