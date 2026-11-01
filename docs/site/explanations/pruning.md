# Pruning

`split_budget = B` reduces the ensemble to at most \(B\) split evaluations per observation, \(\sum_t k_t \le B\), where
tree \(t\) keeps its first \(k_t\) levels. The question is which levels to keep, and the answer depends on the
leaf values, which are refit at the same time.

## The level basis

Cutting a tree after \(k\) levels merges the leaves that agree in their first \(k\) answers. A natural value for
a merged leaf is the mean of the leaves it merges. This suggests writing a tree in a basis where cutting is
just setting coefficients to zero.

Let \(v_l(p)\) be the mean of the leaves \(j\) with \(j \bmod 2^l = p\) (the node \(p\) of level \(l\); \(v_d\) are
the leaves themselves). The **level basis** holds the differences to the parent node,

\[
u_0 = v_0, \qquad u_l(p) = v_l(p) - v_{l-1}(p \bmod 2^{l-1}), \qquad
w_j = \sum_{l=0}^{d} u_l(j \bmod 2^l) .
\]

A leaf value is the sum of the node values on its path. Cutting the tree to \(k\) levels sets \(u_l = 0\) for
\(l > k\), and the leaves of the cut tree are the means of the merged leaves. Cutting to 0 levels leaves \(u_0\),
the tree's mean, which goes into the bias: the tree is removed.

## Iterative hard thresholding

boxwood minimises the fit objective \(J\) (with the original trees, leaf values rebuilt from \(u\)) under the
budget constraint, by projected gradient descent:

\[
u \leftarrow P\big(u - \eta \nabla_u J\big),
\]

with a backtracking line search on \(\eta\). The projection \(P\) keeps, per tree, the levels that matter most
and zeroes the rest. The importance of the first \(k\) levels of tree \(t\) is the energy of their node values,

\[
g_t(k) = \sum_{l=1}^{k} \lVert u_l \rVert^2, \qquad g_t(0) = 0,
\]

and \(P\) chooses the \(k_t\) that maximise \(\sum_t g_t(k_t)\) subject to \(\sum_t k_t \le B\).

## The knapsack over levels

Choosing \(k_t\) is a knapsack problem: every tree offers its levels in order, each at cost 1. boxwood solves it
greedily on the **upper concave hull** of each \(g_t\): the hull splits into segments of decreasing slope (gain
per level), and the segments of all trees are taken by slope, largest first, while they fit into the budget. A
segment of length 2 or more means that one level alone gains little but two levels together gain a lot; the hull
lets the greedy choice see that.

## After the pruning

When the kept levels stop changing and \(J\) has converged, boxwood cuts every tree to its \(k_t\) levels (trees
with \(k_t = 0\) are removed, their mean folded into the bias) and refits the leaf values of the smaller ensemble
with the chosen strategy, as a fit without a budget would.

The budget is a hard limit. Pruning with a holdout never falls back to the unpruned model, even when that would
have a lower holdout loss.
