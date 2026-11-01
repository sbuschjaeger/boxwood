# Import a CatBoost model and predict

This tutorial imports CatBoost models with numeric and categorical features, checks that boxwood gives
CatBoost's margins, and looks inside the imported model.

## Numeric features

```python
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
import boxwood as bw

rng = np.random.default_rng(1)
n = 4000
X = rng.normal(size=(n, 6)).astype(np.float32)
y = X[:, 0] - 2 * X[:, 1] * X[:, 2] + rng.normal(scale=0.3, size=n)

reg = CatBoostRegressor(iterations=200, depth=6, verbose=0, random_seed=0).fit(X, y)
ote = bw.ObliviousTreeEnsemble.from_catboost(reg)

margin = ote.predict_margin(X)
print(np.max(np.abs(margin[:, 0] - reg.predict(X, prediction_type="RawFormulaVal"))))
```

The difference is at the level of float rounding: boxwood adds the leaf values in float64, CatBoost in its own
order.

## Categorical features

boxwood takes a pandas DataFrame with the same columns as CatBoost, in the same order. Categorical columns are
strings (or integers, which are turned into strings as CatBoost does); missing values are fine.

```python
df = pd.DataFrame(X, columns=[f"x{i}" for i in range(6)])
df["city"] = rng.choice(["berlin", "paris", "rome", "oslo"], size=n)
df["shop"] = rng.choice([f"s{i}" for i in range(30)], size=n)
effect = df["city"].map({"berlin": 1.0, "paris": -1.0, "rome": 0.5, "oslo": 0.0}).to_numpy()
label = (X[:, 0] + effect + rng.normal(scale=0.5, size=n) > 0).astype(int)

clf = CatBoostClassifier(iterations=200, depth=6, verbose=0, random_seed=0)
clf.fit(df, label, cat_features=["city", "shop"])
ote = bw.ObliviousTreeEnsemble.from_catboost(clf)

proba = ote.predict(df)                    # sigmoid of the margin, shape (n, 1)
ref = clf.predict_proba(df)[:, 1]
print(np.max(np.abs(proba[:, 0] - ref)))
```

CatBoost turns the two categorical columns into many CTR features (target statistics of each category and of
combinations of categories). boxwood imports all of them with the CTR tables; see
[Encoding CatBoost's features](../explanations/encoding.md).

`info()` tells which column is which:

```python
info = ote.info()
print(info["task"], info["n_numeric"], info["n_categorical"])  # binary 6 2
print(info["numeric_flat_index"], info["categorical_flat_index"])  # [0, 1, 2, 3, 4, 5] [6, 7]
```

## Multiclass

For K classes the margins have K columns and `predict` returns the softmax:

```python
y3 = np.digitize(X[:, 0] + X[:, 1], [-0.5, 0.5])
mc = CatBoostClassifier(iterations=200, depth=6, verbose=0, random_seed=0, loss_function="MultiClass")
mc.fit(X, y3)
ote3 = bw.ObliviousTreeEnsemble.from_catboost(mc)
p = ote3.predict(X)
print(p.shape, np.allclose(p.sum(axis=1), 1.0))                 # (4000, 3) True
print(np.max(np.abs(p - mc.predict_proba(X))))
```

## Look inside

Every tree is oblivious: all nodes of one level share the same split. `splits()` lists them per tree as
`(feature kind, feature index, "gt" or "eq", argument)`; `leaf_index` gives the leaf of every observation in every tree.

```python
print(ote.splits()[0][:3])          # the first three levels of tree 0
leaves = ote.leaf_index(df)         # uint8, shape (n, n_trees)
print(leaves.shape, leaves.max())   # (4000, 200) 63 (at most 2^6 - 1 for depth 6)
```

The argument of a numeric split is the index of a border: the split is true when the observation's value is above that
border. CTR splits compare the CTR value's bucket the same way, one-hot splits test for one category.

Next: [Refit the leaf values](refit-leaves.md).
