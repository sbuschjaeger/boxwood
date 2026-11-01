# Refit the leaf values

CatBoost builds its trees greedily: each new tree's leaves are fit once, given the trees before it, and never
change again. boxwood keeps CatBoost's splits and refits **all leaf values together**. With the splits fixed, the
model is linear in the leaf values, so this is a convex problem that boxwood solves exactly in float64. See
[Refitting the leaves](../explanations/leaf-fit.md) for the details.

A common use: train the trees on a sample (CatBoost's tree search is the expensive part), then fit the leaves on
all the data.

## Trees on a sample, leaves on everything

```python
import numpy as np
from catboost import CatBoostRegressor
import boxwood as bw

rng = np.random.default_rng(2)

def make(n):
    X = rng.normal(size=(n, 8)).astype(np.float32)
    y = np.sin(2 * X[:, 0]) + X[:, 1] * X[:, 2] + 0.5 * X[:, 3] ** 2 + rng.normal(scale=0.5, size=n)
    return X, y

X_small, y_small = make(2000)    # CatBoost sees only these observations
X_big, y_big = make(40000)       # the leaves are refit on these
X_hold, y_hold = make(5000)      # holdout: chooses the strength
X_test, y_test = make(20000)

cb = CatBoostRegressor(iterations=500, depth=6, verbose=0, random_seed=0).fit(X_small, y_small)

def rmse(pred):
    return float(np.sqrt(np.mean((pred - y_test) ** 2)))

print("CatBoost", rmse(cb.predict(X_test)))     # about 0.59
```

Import and refit with the `path` strategy:

```python
ote = bw.ObliviousTreeEnsemble.from_catboost(cb)
report = ote.fit(X_big, y_big, X_hold, y_hold, strategy="path")
print("refit   ", rmse(ote.predict(X_test)[:, 0]))   # about 0.54
print(report["lambda"], report["no_gain"])            # 0.01 False
```

The test error drops from about 0.59 to 0.54 with the same 500 trees and the same 3,000 splits.

## What the fit does

The fit minimises

\[
J(w, b) = \frac{1}{\sum_i s_i} \sum_i s_i \,\ell\big(y_i, F(x_i)\big) + \frac{\lambda}{2} \lVert w - a \rVert^2,
\]

where \(w\) are all leaf values, \(b\) the bias, \(s_i\) the sample weights, \(\ell\) the loss of the task
(squared error, logistic, softmax) and \(a\) the **anchor**: the leaf values at the start, here CatBoost's. A
large \(\lambda\) keeps the model close to CatBoost; a small one trusts the new data more. Regression is fit on
\(y / \operatorname{std}(y)\), so \(\lambda\) has the same meaning for every scale of \(y\).

The strategy picks \(\lambda\):

| strategy | what it does | needs a holdout |
|---|---|---|
| `"path"` | \(\lambda = 10^2, 10^1, \dots, 10^{-4}\), each run warm-started from the last; keeps the run with the best holdout loss and stops after two \(\lambda\) without improvement | yes |
| `"early_stop"` | one run with a small \(\lambda\) (default \(10^{-4}\)), keeps the iterate with the best holdout loss | yes |
| `"fixed"` | one run with `lam` (default 1), no holdout | no |

The report shows each step of the path:

```python
for step in report["path"]:
    print(step["lambda"], round(step["holdout_loss"], 4), step["iterations"])
```

```text
100.0 0.0785 5
10.0 0.0774 14
1.0 0.0734 97
0.1 0.0671 92
0.01 0.0646 195
0.001 0.0683 200
0.0001 0.0774 200
```

The holdout loss falls until \(\lambda = 0.01\) and rises after: below that, the leaves start to fit the noise
of the 40,000 observations. (The losses are in the units of the fit: half the squared error of \(y / \operatorname{std}(y)\).)

## Never worse than the start

With a holdout, every strategy compares its best run with the model it started from. If no run has a lower
holdout loss, the model stays exactly as it was and the report says `no_gain = True`. So a fit with a holdout
does not make a model worse on that holdout.

## The other strategies and optimizers

```python
fixed = bw.ObliviousTreeEnsemble.from_catboost(cb)
fixed.fit(X_big, y_big, strategy="fixed", lam=1.0)
print("fixed     ", rmse(fixed.predict(X_test)[:, 0]))     # about 0.57

early = bw.ObliviousTreeEnsemble.from_catboost(cb)
early.fit(X_big, y_big, X_hold, y_hold, strategy="early_stop")
print("early_stop", rmse(early.predict(X_test)[:, 0]))     # about 0.57
```

The default optimizer, `"tree_newton"`, cycles over the trees and takes an exact Newton step per tree: each observation
is in exactly one leaf of a tree, so the curvature of a tree's leaves is diagonal. `"adam"` and
`"gradient_descent"` are the alternatives; which optimizer is the best default is still being measured.

## Classification

Classification works the same way; the loss is the logistic loss (binary) or the softmax cross-entropy
(multiclass), and the margins are fit directly. Binary `y` holds 0/1 labels (or probabilities); multiclass `y`
holds the labels `0 .. K-1` or an `(n, K)` array of probabilities.

The lambda of the chosen run is stored in the model (`info()["fit_lambda"]`) and used by `update`; with
`no_gain`, it is the lambda of the best run.

Next: [Prune to a split budget](prune.md).
