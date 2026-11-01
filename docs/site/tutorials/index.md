# Tutorials

Each tutorial is a short, complete program that you can run top to bottom. They build on each other but each
one starts from scratch.

1. [Import a CatBoost model and predict](import-and-predict.md): numeric and categorical data, regression and
   classification, and how to look inside the imported model.
2. [Refit the leaf values](refit-leaves.md): keep CatBoost's trees, fit all leaf values together with a strength
   chosen on a holdout.
3. [Prune to a split budget](prune.md): make the model cheaper to evaluate while refitting it.
4. [Update a model online](online-update.md): follow a drifting target window by window.
5. [Fast inference](fast-inference.md): backends, threads, int8 leaves and small batches.

The tutorials use synthetic data so they run in seconds. All code needs `numpy`, `pandas` and `catboost`.
