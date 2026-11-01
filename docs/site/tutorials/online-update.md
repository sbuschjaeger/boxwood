# Update a model online

Data that arrives over time often drifts: the relation between features and target changes slowly. Retraining
the trees for every window is expensive. `update` keeps the trees and moves only the leaf values and the bias
towards the newest window.

Each call minimises the fit objective on the window, anchored at the current leaves:

\[
J(w, b) = \frac{1}{n_{\text{window}}} \sum_{i \in \text{window}} \ell\big(y_i, F(x_i)\big)
          + \frac{\lambda}{2} \lVert w - w_{\text{before}} \rVert^2 .
\]

\(\lambda\) is the strength of the last `fit` (`info()["fit_lambda"]`): it says how far one window may move the
model. A large \(\lambda\) changes little per window; a small one follows the newest window closely and forgets
fast.

## A drifting stream

```python
import numpy as np
from catboost import CatBoostRegressor
import boxwood as bw

def make(rng, n, shift):
    """Observations whose target depends more and more on x1 and x2 as `shift` grows."""
    X = rng.normal(size=(n, 6)).astype(np.float32)
    y = np.sin(2 * X[:, 0]) + (1.0 + shift) * X[:, 1] + shift * X[:, 2] + rng.normal(scale=0.3, size=n)
    return X, y

rng = np.random.default_rng(4)
X0, y0 = make(rng, 10000, 0.0)
cb = CatBoostRegressor(iterations=300, depth=6, verbose=0, random_seed=0).fit(X0, y0)

frozen = bw.ObliviousTreeEnsemble.from_catboost(cb)
live = bw.ObliviousTreeEnsemble.from_catboost(cb)
live.fit(X0, y0, strategy="fixed", lam=1.0)   # sets the strength the updates use
```

Now ten windows arrive, each drifting a bit further. Each window is first predicted (that is the error a user
sees), then used to update the live model:

```python
rng = np.random.default_rng(5)
for window in range(10):
    X, y = make(rng, 2000, 0.1 * (window + 1))
    err_frozen = np.sqrt(np.mean((frozen.predict(X)[:, 0] - y) ** 2))
    err_live = np.sqrt(np.mean((live.predict(X)[:, 0] - y) ** 2))
    print(f"window {window}: frozen {err_frozen:.3f}  updated {err_live:.3f}")
    live.update(X, y)
```

```text
window 0: frozen 0.342  updated 0.343
...
window 7: frozen 1.192  updated 0.418
window 8: frozen 1.302  updated 0.392
window 9: frozen 1.446  updated 0.399
```

The frozen model's error grows with the drift; the updated one stays near the noise level. The trees still hold
the right splits (on x1 and x2), so moving the leaf values is enough.

## Choosing the strength

The same stream with other strengths (error in the last three windows):

| `lam` of the first fit | windows 7–9 |
|---|---|
| 10 | 0.53, 0.51, 0.53 |
| 1 | 0.42, 0.39, 0.40 |
| 0.1 | 0.40, 0.39, 0.40 |
| 0.01 | 0.44, 0.44, 0.45 |

Too strong and the model lags behind; too weak and each window's noise moves it. With a holdout, `fit(...,
strategy="path")` chooses the strength for the first data; you can also fit with `strategy="fixed"` and the
strength you want for the stream, as above.

## Steps per update

`update(X, y, steps=1)` runs `steps` iterations of the optimizer of the last fit. With `tree_newton` one
iteration is a cycle over the trees with an exact Newton step for each (for squared error); more steps move
further towards the window's minimum, which matters little here (5 or 20 steps give within 0.01 the same
errors). Adam keeps its moment estimates in the model between windows (and in its pickle).

Next: [Fast inference](fast-inference.md).
