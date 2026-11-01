# Oblivious trees

## One split per level

In an ordinary decision tree each node has its own split. In an **oblivious** tree all nodes of one level share
the same split: level \(l\) asks the same question \(c_l(x)\) of every observation, whatever the answers above were. A
tree of depth \(d\) is therefore just \(d\) conditions, and the leaf of an observation is the binary number formed by
their answers:

\[
\operatorname{leaf}(x) = \sum_{l=0}^{d-1} c_l(x)\, 2^l, \qquad c_l(x) \in \{0, 1\}.
\]

The tree has \(2^d\) leaves, one value per leaf (per output, for multiclass). CatBoost grows only oblivious
trees (its `SymmetricTree` policy, the default), up to depth 16; boxwood supports depth 0 to 8, which covers
CatBoost's default of 6 and the usual tuning range.

An ensemble of \(T\) trees adds the leaf values to a bias:

\[
F(x) = b + \sum_{t=1}^{T} w_{t,\operatorname{leaf}_t(x)} .
\]

Regression uses \(F\) directly, binary classification \(\sigma(F)\), multiclass the softmax of the \(K\)
outputs.

## Why this is good for speed

- **No branches.** An observation's path through the tree is not a sequence of decisions but \(d\) independent
  comparisons. A CPU can evaluate them for 32 or 64 observations at once with vector instructions and combine the bits.
- **Shared work.** The conditions are of the form "feature \(f\) is above border \(j\)". Many trees ask the same
  question; boxwood answers each distinct question once per group of observations (see
  [How prediction is fast](prediction.md)).
- **Small, regular tables.** The leaf values of a tree are one array of \(2^d\) entries, indexed by the leaf
  number. Up to 16 or 32 of them fit into a vector register, so a lookup is a register permute instead of a
  memory access.

## Why this is good for refitting

With the splits fixed, \(F\) is **linear** in the leaf values: each observation selects one leaf per tree, and \(F\) is
the sum of the selected values. Any convex loss of \(F\) is then a convex function of the leaf values, and a
penalty \(\frac{\lambda}{2}\lVert w - a\rVert^2\) makes it strongly convex. So the best leaf values for given
trees are a well-posed problem with one solution, which boxwood finds in float64 (see
[Refitting the leaves](leaf-fit.md)).

A tree's leaves partition the observations: every observation is in exactly one leaf of each tree. The curvature of the loss
with respect to the leaves of one tree is therefore diagonal, so a Newton step for one tree is exact and cheap.

## Why this is good for pruning

Cutting an oblivious tree after \(k\) levels gives again an oblivious tree, with \(2^k\) leaves: the leaves
whose first \(k\) answers agree are merged. The cost of a tree is its depth, so the cost of an ensemble is
\(\sum_t d_t\) split evaluations per observation, and it can be reduced in steps of one level. See [Pruning](pruning.md).
