# Getting started

## Install

boxwood needs a C++20 compiler (GCC, Clang or MSVC) and Python 3.10 or newer. From the repository:

```bash
pip install .
```

pip builds the C++ extension with CMake (through scikit-build-core, in `build/pip`) and installs `numpy`,
`pandas`, `scikit-learn` and `catboost` with it. Without network access, install `scikit-build-core` and
`nanobind` into the environment first and run `pip install --no-build-isolation .`.

## Build for development

The development build needs CMake 3.21 and Ninja, and a Python environment with `nanobind`, `numpy`, `pandas`,
`catboost` and `scikit-learn`.

```bash
python -m venv .venv
.venv/bin/pip install nanobind numpy pandas catboost scikit-learn
cmake --preset release           # configures build/release with .venv/bin/python
cmake --build build/release -j8
```

The build puts the Python package into `build/release/python/boxwood`. Put that folder on the Python path:

```bash
export PYTHONPATH=$PWD/build/release/python
```

To check the build, run the tests:

```bash
(cd build/release && ctest)          # C++ tests
.venv/bin/python -m pytest tests     # Python tests (CatBoost trains small models)
```

The Python tests use the package in `build/release/python`; to test another copy, such as an unpacked wheel, set
`BOXWOOD_PKG_DIR` to the folder that contains its `boxwood` folder.

The `debug` preset builds the C++ library and tests with AddressSanitizer and UndefinedBehaviorSanitizer.

## The first model

Train a CatBoost model as usual, then import it:

```python
import numpy as np
from catboost import CatBoostRegressor
import boxwood as bw

rng = np.random.default_rng(0)
X = rng.normal(size=(5000, 10)).astype(np.float32)
y = X[:, 0] * 2 + X[:, 1] ** 2 + rng.normal(scale=0.3, size=len(X))
cb = CatBoostRegressor(iterations=300, depth=6, verbose=0, random_seed=0).fit(X, y)

ote = bw.ObliviousTreeEnsemble.from_catboost(cb)
print(ote.info()["n_trees"], ote.info()["n_splits"])  # 300 1800
```

`predict_margin` gives the raw margins (CatBoost's `RawFormulaVal`), `predict` the task output (the value for
regression, probabilities for classification):

```python
margin = ote.predict_margin(X)
print(margin.shape)                                    # (5000, 1)
ref = cb.predict(X, prediction_type="RawFormulaVal")
print(np.max(np.abs(margin[:, 0] - ref)) < 1e-5)       # True
```

The model is a plain Python object: it pickles, and `to_bytes` / `from_bytes` give a compact binary form.

```python
import pickle

again = pickle.loads(pickle.dumps(ote))
print(np.array_equal(again.predict_margin(X), margin))  # True: bit-exact
```

Next: the [tutorials](tutorials/index.md).
