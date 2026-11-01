# boxwood

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

## Documentation

The documentation (tutorials, how-to guides, explanations, Python and C++ reference) is in `docs/site`. Build it
with `zensical build --clean` (output in `build/site`) or preview it with `zensical serve`. The development build
and the tests are described in `docs/site/getting-started.md`.

## License

MIT, see `LICENSE`.
