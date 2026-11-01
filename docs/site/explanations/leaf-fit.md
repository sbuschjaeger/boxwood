# Refitting the leaves

## Why refit

Gradient boosting fits tree \(t\) to the residuals of trees \(1 \dots t-1\) and never revisits it. Each leaf
value is the result of one Newton step on the data at that moment, shrunk by the learning rate. Once all trees
exist, the leaf values are rarely the best ones for those trees together: early trees were fit before the later
ones existed, and the learning rate deliberately under-fits each step.

With the splits fixed, the model is linear in the leaf values (see [Oblivious trees](oblivious-trees.md)), so
the best leaf values for given trees are the solution of a convex problem. boxwood solves it.

## The objective

\[
J(w, b) = \frac{1}{\sum_i s_i} \sum_{i=1}^{n} s_i\, \ell\big(y_i, F(x_i)\big) + \frac{\lambda}{2} \lVert w - a \rVert^2,
\qquad F(x) = b + \sum_t w_{t, \operatorname{leaf}_t(x)} .
\]

- \(\ell\) is the loss of the task: \(\tfrac12 (F - y)^2\) for regression (on \(y / \operatorname{std}(y)\)),
  the logistic loss for binary, the softmax cross-entropy for multiclass.
- \(a\) is the **anchor**, the leaf values at the start (CatBoost's, or the previous model's). The penalty pulls
  towards the anchor, not towards zero: \(\lambda \to \infty\) returns the start model, \(\lambda \to 0\) the
  unregularised least-loss leaves. The bias is not penalised.
- \(J\) is strongly convex for \(\lambda > 0\), so it has one minimiser.

## Choosing \(\lambda\) on a holdout

The right strength depends on how much data there is and how noisy it is. The `path` strategy fits a
sequence \(\lambda = 10^2, 10^1, \dots, 10^{-4}\); each fit starts from the previous solution (warm start), so
the whole path costs little more than the last fit. The run with the lowest holdout loss is kept; the path stops
after two values without improvement. `early_stop` runs one small \(\lambda\) and keeps the iterate with the
best holdout loss (early stopping acts as the regulariser). `fixed` uses the given \(\lambda\).

With a holdout every strategy also compares with the start: if no run is strictly better on the holdout, the
model is left untouched (`no_gain`).

## TreeNewton

Let \(g_i, h_i\) be the first and second derivative of the loss at observation \(i\). Every observation is in exactly one leaf
of a tree, so the Hessian of \(J\) restricted to the leaves of one tree is diagonal. For leaf \(j\) of tree \(t\):

\[
G_j = \frac{1}{S}\sum_{i \in j} s_i g_i + \lambda (w_j - a_j), \qquad
H_j = \frac{1}{S}\sum_{i \in j} s_i h_i + \lambda, \qquad
\Delta w_j = -\frac{G_j}{H_j} .
\]

For squared error this step is the exact minimum over the tree's leaves, given the other trees. TreeNewton cycles
over the trees (and the bias), keeping the margins of all observations up to date, so one iteration is a full block
coordinate descent sweep. For the logistic and softmax losses the step is halved while it increases \(J\) (at
most 10 times).

## Adam and gradient descent

`adam` (with the usual \(\beta_1 = 0.9\), \(\beta_2 = 0.999\)) and `gradient_descent` (with backtracking) take
steps along the full gradient, one step per pass over the observations. A pass reads every tree of every observation once, about
half the work of a TreeNewton sweep (which reads each observation twice per tree), but a step moves all leaves at once
without the exact per-tree curvature.

## Exactness

All sums are float64. The features are read once per fit, to compute the leaf index of every observation in every tree;
all passes then read these indices (one byte per observation and tree), so the features can be memory-mapped. The fitted
values are written back as float32 leaves (or quantized to int8).
