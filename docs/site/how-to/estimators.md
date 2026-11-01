# Use the scikit-learn estimators

`BoxwoodClassifier` and `BoxwoodRegressor` run the whole pipeline behind one `fit`: split off a holdout, train
CatBoost with early stopping, import the trees, and refit the leaf values on all training observations.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import boxwood as bw

rng = np.random.default_rng(7)
n = 6000
X = pd.DataFrame(rng.normal(size=(n, 5)), columns=list("abcde"))
X["color"] = rng.choice(["red", "green", "blue"], size=n)
y = ((X["a"] + (X["color"] == "red") * 1.5 + rng.normal(scale=0.7, size=n)) > 0.5).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = bw.BoxwoodClassifier(catboost_params={"iterations": 500, "thread_count": 4})
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))              # accuracy
print(clf.predict_proba(X_test)[:3])          # one column per class
print(clf.fit_report_["lambda"], clf.ensemble_.info()["n_trees"])
```

Categorical columns are found automatically: every non-numeric and every bool column of a DataFrame is
categorical, as for CatBoost's `cat_features`.

## The parameters

| parameter | default | meaning |
|---|---|---|
| `catboost_params` | `None` | extra CatBoost parameters on top of `learning_rate=0.05, depth=6, iterations=10000` and AutoGluon's early-stopping patience (300 rounds up to 10,000 observations, then `max(20, round(300 * 10000 / n))`) |
| `strategy` | `"path"` | how the refit chooses its strength: `"path"`, `"early_stop"`, `"fixed"` |
| `optimizer` | `"tree_newton"` | `"tree_newton"`, `"adam"`, `"gradient_descent"` |
| `split_budget` | `None` | prune to at most this many split evaluations per observation |
| `max_bytes` | `None` | prune to at most this many bytes of `to_bytes()`; not together with `split_budget` |
| `leaf_type` | `"int8"` | the leaf type of the fitted ensemble: `"int8"` or `"float32"` |
| `holdout_fraction` | `0.125` | the share of the training observations held off when `fit` gets no holdout |
| `random_state` | `0` | seed of the holdout split and CatBoost |

Pass your own holdout with `fit(X, y, X_holdout, y_holdout)`; then all of `X` is used for training.
`sample_weight` weights the observations in CatBoost and in the refit; when the holdout is split off `X`, its part
of the weights weighs the holdout too.

## Updating

`clf.update(X_new, y_new)` and `reg.update(X_new, y_new)` call
[`ObliviousTreeEnsemble.update`](../tutorials/online-update.md) on the fitted ensemble.
