# Explanations

These pages explain why boxwood is built the way it is. They need no code; the mathematics is kept short here
and will be worked out in a technical report.

- [Oblivious trees](oblivious-trees.md): what makes CatBoost's trees special, and why that matters for speed and
  for refitting.
- [Encoding CatBoost's features](encoding.md): how a raw observation becomes one byte per split feature, including
  CatBoost's CTR statistics.
- [How prediction is fast](prediction.md): blocks, shared split conditions, register lookups, threads.
- [Numerics and int8 leaves](numerics.md): why every backend gives the same bits, and how int8 leaves stay exact.
- [Refitting the leaves](leaf-fit.md): the convex problem behind `fit`, the anchor and the strength.
- [Pruning](pruning.md): the level basis, iterative hard thresholding and the knapsack over tree levels.
