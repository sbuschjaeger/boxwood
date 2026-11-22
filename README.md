# boxwood

[![Documentation](https://img.shields.io/badge/docs-online-4c9aff?logo=readthedocs&logoColor=white)](https://sbuschjaeger.github.io/boxwood/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**[Documentation](https://sbuschjaeger.github.io/boxwood/)** |
**[Getting started](https://sbuschjaeger.github.io/boxwood/getting-started/)** |
**[Tutorials](https://sbuschjaeger.github.io/boxwood/tutorials/)** |
**[How-to guides](https://sbuschjaeger.github.io/boxwood/how-to/)** |
**[Explanations](https://sbuschjaeger.github.io/boxwood/explanations/)** |
**[Reference](https://sbuschjaeger.github.io/boxwood/reference/python/)**

boxwood takes a trained CatBoost model and makes it a better deployment artifact. It imports the trees
exactly, can refit the leaf values to your data, prune the model to a budget of split evaluations, update it
online, and predicts several times faster than CatBoost's own predictor.

It is a small C++20 library (standard library only, with AVX2, AVX-512 and NEON backends) with a Python package
built with nanobind.

## Install

boxwood needs a C++20 compiler (GCC, Clang or MSVC) and Python 3.10 or newer:

```bash
pip install .
```

## Quick tour

In this quick tour a trained CatBoost model becomes an exact import, a better model, and half the model.
Each block continues the one before it.

### Import a model and predict

Train a CatBoost model as usual, import it, and check that the raw margins agree:

```python
import numpy as np
from catboost import CatBoostRegressor
import boxwood as bw

rng = np.random.default_rng(0)

def make(n):
    X = rng.normal(size=(n, 10)).astype(np.float32)
    y = X[:, 0] * 2 + X[:, 1] ** 2 + rng.normal(scale=0.3, size=n)
    return X, y

X_train, y_train = make(3000)     # CatBoost trains on this sample
X_all, y_all = make(30000)        # you have more data than that
X_hold, y_hold = make(5000)       # a holdout chooses the strength of the refit
X_test, y_test = make(5000)

cb = CatBoostRegressor(iterations=300, depth=6, verbose=0, random_seed=0, thread_count=1, allow_writing_files=False).fit(X_train, y_train)

def rmse(pred):
    return float(np.sqrt(np.mean((pred - y_test) ** 2)))

print("CatBoost", round(rmse(cb.predict(X_test)), 4))            # about 0.383

ote = bw.ObliviousTreeEnsemble.from_catboost(cb)
margin = ote.predict_margin(X_test)                              # float64, shape (n, 1)
ref = cb.predict(X_test, prediction_type="RawFormulaVal")
print("margin diff", np.max(np.abs(margin[:, 0] - ref)))         # about 1e-7
```

Boxwood imports numeric borders, one-hot features, and CTR target statistics. This example compares
the imported model's raw margins with CatBoost's `RawFormulaVal`.

### Refit the leaf values on your data

CatBoost fits each leaf once, greedily. boxwood keeps the splits and refits all leaf values together, as one
convex problem, on all your data:

```python
report = ote.fit(X_all, y_all, X_hold, y_hold, strategy="path")
print("refit", round(rmse(ote.predict(X_test)[:, 0]), 4), "at lambda", report["lambda"])
```

The output is `refit 0.3595 at lambda 0.01`. The `path` strategy tries a few strengths and keeps the best one
using the holdout. The RMSE shown here is measured on a separate test set.

### Prune to a split budget

`split_budget` cuts the ensemble to at most that many split evaluations per observation - each tree keeps its
most useful levels - and refits what remains:

```python
lean = bw.ObliviousTreeEnsemble.from_catboost(cb)
report = lean.fit(X_all, y_all, X_hold, y_hold, strategy="path", split_budget=900)
print("pruned", round(rmse(lean.predict(X_test)[:, 0]), 4),
      "with", report["splits_used"], "of", ote.info()["n_splits"], "splits")
```

The output is `pruned 0.3585 with 900 of 1800 splits`: half the split evaluations with similar test RMSE in this example.

The model is a plain Python object: it pickles, `to_bytes` / `from_bytes` give a compact binary form, and
`update` moves the leaf values towards a new window of data. The same flow is also available as
scikit-learn estimators, `bw.BoxwoodRegressor` and `bw.BoxwoodClassifier`.

## Documentation

The full documentation - tutorials, how-to guides, explanations, and the Python and C++ reference - is in
[`docs/site`](docs/site). GitHub Pages builds from the default branch once Pages is configured to use
GitHub Actions. To build or preview the documentation locally:

```bash
pip install ".[dev]"
zensical serve             # live preview
zensical build --clean     # HTML in build/site
```

The development build and the tests are described in the
[getting started guide](https://sbuschjaeger.github.io/boxwood/getting-started/).

## License

MIT, see `LICENSE`.
