# Prune to a split budget

The cost of a prediction is the number of split evaluations per observation: the sum of the tree depths. `split_budget`
cuts the ensemble to at most that many, then refits the leaves. Each tree keeps a prefix of its levels (a tree
cut to k levels is again an oblivious tree, with \(2^k\) leaves); a tree cut to 0 levels is removed and its mean
goes into the bias. Which levels to keep is chosen together with the leaf values; see
[Pruning](../explanations/pruning.md).

## A large model on a budget

```python
import time

import numpy as np
from catboost import CatBoostRegressor
import boxwood as bw

rng = np.random.default_rng(3)

def make(n):
    X = rng.normal(size=(n, 8)).astype(np.float32)
    y = np.sin(2 * X[:, 0]) + X[:, 1] * X[:, 2] + 0.5 * X[:, 3] ** 2 + rng.normal(scale=0.5, size=n)
    return X, y

X_train, y_train = make(20000)
X_hold, y_hold = make(5000)
X_test, y_test = make(20000)

cb = CatBoostRegressor(iterations=1000, depth=6, learning_rate=0.05, verbose=0, random_seed=0)
cb.fit(X_train, y_train)

def rmse(pred):
    return float(np.sqrt(np.mean((pred - y_test) ** 2)))

def predict_ms(model):
    best = float("inf")
    for _ in range(5):
        start = time.perf_counter()
        model.predict(X_test)
        best = min(best, time.perf_counter() - start)
    return 1e3 * best

print("CatBoost", rmse(cb.predict(X_test)))     # about 0.532, 6,000 splits
```

Now prune to a quarter of the 6,000 splits:

```python
ote = bw.ObliviousTreeEnsemble.from_catboost(cb)
report = ote.fit(X_train, y_train, X_hold, y_hold, strategy="path", split_budget=1500)
print(report["splits_used"], report["trees_removed"])   # 1500 748
print(ote.info()["n_trees"])                             # 252
print("pruned  ", rmse(ote.predict(X_test)[:, 0]))       # about 0.533
```

The pruned model has the same test error as CatBoost's 1,000 trees with a quarter of the split evaluations.

## The trade-off

Different budgets on the same model (the times are `predict` on 20,000 observations, one thread, AVX-512):

| budget | trees kept | test RMSE | predict |
|---|---|---|---|
| 6,000 (all) | 1,000 | 0.534 | 3.0 ms |
| 3,000 | 500 | 0.532 | 1.7 ms |
| 1,500 | 252 | 0.533 | 1.0 ms |
| 600 | 101 | 0.543 | 0.54 ms |
| 300 | 50 | 0.552 | 0.37 ms |

(CatBoost's model: 0.532.) Here the pruning keeps most trees at their full depth and removes the others. With
deeper trees or a tighter budget it also cuts trees to fewer levels.

!!! note "The budget is a hard limit"
    The pruned model is kept even when its holdout loss is worse than that of the unpruned model. A fit with
    a holdout only falls back to its own start, which is the pruned model before the leaf refit.

Next: [Update a model online](online-update.md).
