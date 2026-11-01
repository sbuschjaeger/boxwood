# Numerics and int8 leaves

## Bit-exact backends

Floating-point addition is not associative: \((a + b) + c\) and \(a + (b + c)\) can differ in the last bit.
A SIMD kernel that sums the trees in a different order than the plain C++ code would give slightly different
margins on different CPUs. boxwood fixes the order by a rule that every backend follows:

1. the stored values (leaves, borders, CTR values) are float32, as in CatBoost;
2. prediction adds each tree's leaf value to a **float64** margin, in tree order;
3. fit, pruning and update compute in float64 and round to float32 (or int8) only when they write the leaves
   back.

So `predict` gives the same bits on AVX-512, AVX2 and the scalar backend, with any number of threads. The
tests check this bit-exactly. float64 also keeps the sum of thousands of trees accurate: the import matches
CatBoost within \(10^{-5}\) relative on all 51 test tasks.

## Int8 leaves

`set_leaf_type("int8")` stores each leaf value as a signed byte \(q \in [-127, 127]\) times a per-tree
**step** \(s_t\). Small leaves are not only a quarter of the memory; they can be looked up 16 at a time with byte
shuffles and summed as integers. The difficulty is the sum: trees of a boosted ensemble have very different
leaf sizes, so they need different steps, and values with different steps cannot be added as integers.

boxwood groups the trees into **blocks of 64**. Let \(M\) be the largest absolute leaf value of a block and
\(m_t\) that of tree \(t\). The block's coarsest step is \(s = M / 127\), and tree \(t\) gets

\[
s_t = s \cdot 2^{-d_t}, \qquad d_t = \max\{\, d \le 7 : m_t \le M \cdot 2^{-d} \,\},
\qquad q = \operatorname{round}(w / s_t).
\]

So each tree's step is within a factor 2 of its own best step \(m_t / 127\) (unless it is more than 128 times
smaller than the block's largest tree), and all steps of a block are powers of two apart. With
\(u = s / 128\) and the integer factor \(f_t = 2^{7 - d_t} \in \{1, 2, \dots, 128\}\):

\[
\sum_{t \in \text{block}} q_t\, s_t \;=\; u \cdot \underbrace{\sum_{t \in \text{block}} q_t f_t}_{\text{integer}} .
\]

The integer sum is at most \(64 \cdot 127 \cdot 128 < 2^{21}\) in absolute value, so it is exact in int32, and
the product with \(u\) (24 significant bits) needs at most 45 bits, so it is exact in float64. The prediction
adds one exact value per block to the float64 margin, in block order. Every backend gives the same bits by
construction, and the only rounding is one float64 addition per 64 trees.

## How far int8 moves a prediction

Rounding moves each leaf by at most half its step, so a margin moves by at most

\[
|F_{\text{int8}}(x) - F_{\text{float32}}(x)| \;\le\; \tfrac12 \sum_t s_t .
\]

In practice the errors of different trees partly cancel, and the change is far below the bound: for the
1,000-tree depth-6 model of the [fast inference tutorial](../tutorials/fast-inference.md) the largest change of
a margin over 50,000 observations is 0.03, with a target of standard deviation 2.5.

Int8 is a separate leaf type, not a default: the margins are close to float32 but not equal. `fit`, `update`
and pruning work in float64 also with int8 leaves and quantize the result again.
