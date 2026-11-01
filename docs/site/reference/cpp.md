# C++ API

Everything is in namespace `boxwood`, declared in `include/boxwood/ensemble.hpp` and
`include/boxwood/error.hpp`. The doc comments in the headers are the authoritative description; this page
collects them. See [Use boxwood from C++](../how-to/cpp.md) for a complete program.

## `ObliviousTreeEnsemble`

### Creating and saving

`static ObliviousTreeEnsemble from_catboost_json(std::string_view json)`
:   Import a CatBoost JSON model (`save_model(..., format="json")`) with all of CatBoost's feature handling:
    numeric, one-hot and CTR features, including combinations. Throws `Error` if the model is malformed, has a
    tree deeper than 8, an unsupported loss, more than 255 borders per feature or CTR, more than 255 one-hot
    values, or a split that does not match the feature data.

`static ObliviousTreeEnsemble from_bytes(std::span<const std::byte> bytes)`
:   Load a model saved by `to_bytes`. Throws `Error` if the bytes are damaged, truncated, of another version, or
    inconsistent.

`std::vector<std::byte> to_bytes() const`
:   Serialize the whole model ([format](format.md)); a round trip is bit-exact.

### Prediction

`void predict_margin(const Batch& X, std::span<double> out) const`
:   The raw margins (bias + leaf sums) of the observations of `X`, n × K, observation-major.

`void predict(const Batch& X, std::span<double> out) const`
:   The task output: identity (regression), sigmoid (binary) or softmax (multiclass), n × K.

`void leaf_index(const Batch& X, std::span<uint8_t> out) const`
:   The leaf of every observation in every tree, n × n_trees, observation-major.

All three throw `Error` if `X.numeric.size() != X.n * n_numeric`, the categorical columns do not have the right
count and length (dictionary-encoded: a code is not less than its dictionary's size), or `out` has the wrong size.

`void set_backend(std::string_view name)`
:   `"scalar"`, `"avx2"`, `"avx512"`, `"avx512_vbmi"`, `"neon"`; `""` picks the best one for this CPU. Throws `Error` for an unknown or
    unavailable backend. Not serialized.

`void set_threads(unsigned n)`
:   Threads for the three prediction functions (default 1; 0 = one per hardware thread). Results do not depend
    on the number of threads. Not serialized.

`void set_leaf_type(LeafType type)`
:   Store the leaves as `Float32` or `Int8` ([int8 rule](../explanations/numerics.md)). Part of the serialized
    model.

### Fitting

`FitReport fit(const Batch& train, const Batch& holdout, const FitOptions& options)`
:   Refit the leaf values and the bias, keeping the splits (see [Refitting the leaves](../explanations/leaf-fit.md)).
    An empty `holdout` (`n = 0`) means none. The features are read once, to compute the leaf indices
    (n × n_trees bytes, kept during the fit), so the spans can point into memory-mapped files. Throws `Error` if
    `y` or `weight` have the wrong size, a weight is negative or all are 0, `lambda < 0`, `learning_rate < 0`,
    `Path` / `EarlyStop` get no holdout observations, or the features have the wrong shape (as in `predict_margin`).

`UpdateReport update(const Batch& window, int steps)`
:   Run `steps` iterations of the last fit's optimizer on `window`, anchored at the current leaves, with the
    last fit's λ: one call per window of a stream. Throws `Error` if there was no fit before, `steps < 1`, or the window's `y` / `weight` are
    wrong.

### Inspection

`Info info() const`
:   The shape of the ensemble.

`std::vector<Split> splits(uint32_t tree) const`
:   The splits of one tree from the root level down; empty for a depth-0 tree. Throws `Error` if `tree` is out
    of range.

## Data

```cpp
struct Batch {
  std::span<const float> numeric;            // n x n_numeric, observation-major, NaN = missing
  size_t n = 0;
  std::vector<std::span<const std::string_view>> categorical = {};  // per feature: n values, or a dictionary
  std::span<const double> y = {};            // fit / update: n values (multiclass: n x K probabilities)
  std::span<const double> weight = {};       // n values; empty = all 1
  std::vector<std::span<const uint32_t>> categorical_codes = {};    // optional: per feature n dictionary codes
};
```

## Options and reports

```cpp
enum class Task : uint8_t { Regression, Binary, Multiclass };
enum class LeafType : uint8_t { Float32, Int8 };
enum class OptimizerKind : uint8_t { TreeNewton, Adam, GradientDescent };
enum class Strategy : uint8_t { Fixed, Path, EarlyStop };

struct StopRule {
  int max_iter = 200;         // iterations per optimizer run
  double tol = 1e-7;          // relative drop of J that counts as converged
  bool use_holdout = false;   // keep the best iterate by holdout loss
  int patience = 10;          // iterations without improvement before stopping
};

struct FitOptions {
  double lambda = 1.0;
  OptimizerKind optimizer = OptimizerKind::TreeNewton;
  double learning_rate = 0.0;                // 0 = the optimizer's default
  StopRule stop;
  Strategy strategy = Strategy::Fixed;
  std::optional<uint32_t> split_budget;      // prune to at most this many split evaluations per observation
};

struct FitReport {
  double objective_start, objective;         // J at the start and at the result (fit units)
  double holdout_loss;                       // mean holdout loss at the result; NaN without holdout
  int iterations;
  bool converged;
  double lambda;                             // of the result; NaN with no gain
  bool no_gain;                              // no run beat the start on the holdout
  double holdout_loss_start;
  std::vector<PathStep> path;                // one entry per optimizer run: lambda, objective, holdout_loss, iterations
  uint32_t splits_used = 0;                  // sum of the tree depths after the fit
  uint32_t trees_removed = 0;                // by a split budget
};

struct UpdateReport { double objective_start, objective; int steps; };

struct Info {
  Task task;
  uint32_t n_outputs, n_numeric, n_categorical, n_trees, n_splits;
  LeafType leaf_type;
  std::vector<uint32_t> numeric_flat_index, categorical_flat_index;  // CatBoost columns
  double fit_lambda, fit_sigma;              // of the last fit; NaN before the first
  std::string fit_optimizer;                 // "" before the first fit
};

struct ObliviousTreeEnsemble::Split {
  uint8_t source;   // 0 numeric, 1 one-hot, 2 CTR
  uint32_t index;   // numeric column, categorical feature or CTR feature
  uint8_t kind;     // 0: code > arg, 1: code == arg
  uint8_t arg;      // border or value index inside the feature
};
```

## Functions

`std::vector<std::string> available_backends()`
:   The backends available on this CPU, fastest first; the first is what `set_backend("")` picks.

`const char* optimizer_name(OptimizerKind kind)`
:   `"tree_newton"`, `"adam"` or `"gradient_descent"`.

## Errors

```cpp
class Error : public std::runtime_error { using std::runtime_error::runtime_error; };
```

The only exception the library throws, always at the API boundary. Kernels and hot loops do not check and do
not throw.
