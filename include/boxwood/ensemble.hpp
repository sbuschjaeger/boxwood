#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "boxwood/error.hpp"

namespace boxwood {

struct OTEView;     // defined in src/kernels.hpp
struct Backend;     // defined in src/backend.hpp
struct FitData;     // defined in src/fit.hpp
struct AdamState;   // defined in src/fit.hpp

enum class Task : uint8_t { Regression = 0, Binary = 1, Multiclass = 2 };

/// How the leaf values are stored: float32, or int8 codes with one float32
/// step per tree (the value of a leaf is q * step; see set_leaf_type).
enum class LeafType : uint8_t { Float32 = 0, Int8 = 1 };

/// A batch of observations. numeric: n x n_numeric float32, row-major, NaN = missing.
/// categorical: one span of n values per categorical feature, in CatBoost's
/// order of categorical_features. Or dictionary-encoded: categorical_codes
/// holds one span of n codes per categorical feature, categorical[c] is then
/// the dictionary of feature c and observation i has the value
/// categorical[c][categorical_codes[c][i]] (each distinct value is hashed
/// once, not once per observation).
struct Batch {
  std::span<const float> numeric;
  size_t n = 0;
  std::vector<std::span<const std::string_view>> categorical = {};
  std::span<const double> y = {};      // n values; Multiclass: n x K probabilities, row-major
  std::span<const double> weight = {};  // n values; empty = all 1
  std::vector<std::span<const uint32_t>> categorical_codes = {};
};

/// When the leaf refit stops. `tol` is the relative drop in the objective that
/// counts as converged; with `use_holdout` the best iterate by holdout loss is
/// kept and the fit stops after `patience` iterations without improvement.
struct StopRule {
  int max_iter = 200;
  double tol = 1e-3;
  bool use_holdout = false;
  int patience = 10;
};

/// Which leaf-refit optimizer to use.
enum class OptimizerKind : uint8_t { TreeNewton, Adam, GradientDescent };

/// The name of an optimizer kind: "tree_newton", "adam" or "gradient_descent".
const char* optimizer_name(OptimizerKind kind);

/// The names of the prediction backends available on this CPU, fastest first
/// ("avx512_vbmi", "avx512", "avx2" on x86-64, "neon" on AArch64, then "scalar");
/// the first one is the default of every ensemble (see
/// ObliviousTreeEnsemble::set_backend).
std::vector<std::string> available_backends();

/// How `fit` chooses the strength lambda: one run with a fixed lambda, a
/// warm-started path over decreasing lambdas selected on the holdout, or one
/// run stopped early on the holdout.
enum class Strategy : uint8_t { Fixed, Path, EarlyStop };

/// Options for a leaf refit: the penalty strength, the optimizer, the
/// learning rate (0 = the optimizer's default), the stop rule, the
/// strategy that chooses the strength and an optional budget for pruning
/// (split evaluations or bytes, not both).
struct FitOptions {
  double lambda = 1.0;
  OptimizerKind optimizer = OptimizerKind::TreeNewton;
  double learning_rate = 0.0;
  StopRule stop;
  Strategy strategy = Strategy::Fixed;
  std::optional<uint32_t> split_budget;  // prune to at most this many split evaluations per observation
  // Prune to at most this many bytes of to_bytes(), without the state of online
  // updates with Adam (16 bytes per leaf value once `update` ran with Adam).
  std::optional<uint64_t> max_bytes;
  bool verbose = false;  // one line per optimizer run (lambda, objective, holdout loss) on stderr
};

/// One optimizer run of a fit: its lambda, the objective and the holdout loss
/// of the returned parameters, and the iteration count.
struct PathStep {
  double lambda;
  double objective;
  double holdout_loss;
  int iterations;
};

/// What a leaf refit reports. `objective_start` is J at the anchor, `objective`
/// at the returned parameters; `holdout_loss` (mean loss, no penalty) is NaN
/// without a holdout. Regression values are in the units of the fit, y / std(y).
/// `lambda` is the strength of the returned parameters (NaN with no gain),
/// `holdout_loss_start` the holdout loss of the start parameters (NaN without
/// a holdout), and `path` holds one entry per optimizer run. `splits_used` is
/// the sum of the tree depths of the ensemble after the fit and
/// `trees_removed` the number of trees a budget removed.
struct FitReport {
  double objective_start;
  double objective;
  double holdout_loss;
  int iterations;
  bool converged;
  double lambda;
  bool no_gain;
  double holdout_loss_start;
  std::vector<PathStep> path;
  uint32_t splits_used = 0;
  uint32_t trees_removed = 0;
};

/// What one `update` reports: the objective of the window at the leaf values
/// before the call and after the steps, in the units of the last fit.
struct UpdateReport {
  double objective_start;
  double objective;
  int steps;
};

/// Summary of the ensemble shape. The flat-index lists give the CatBoost column
/// (flat_feature_index) of every numeric and every categorical feature, so the
/// Python layer can pick the model's columns out of a frame. `leaf_type` is how
/// the leaf values are stored; `tree_bytes` is the part of to_bytes() the trees
/// take (depths, splits, leaves, int8 steps), the rest is the input encoding
/// (borders, one-hot values, CTR tables), the bias and the header. `fit_lambda` is the strength of the last fit
/// (NaN before the first one), `fit_sigma` the scale that fit used (1 for
/// classification, NaN before the first one) and `fit_optimizer` the name of
/// that fit's optimizer ("" before the first one).
struct Info {
  Task task;
  uint32_t n_outputs;
  uint32_t n_numeric;
  uint32_t n_categorical;
  uint32_t n_trees;
  uint32_t n_splits;
  uint64_t tree_bytes;
  LeafType leaf_type = LeafType::Float32;
  std::vector<uint32_t> numeric_flat_index;
  std::vector<uint32_t> categorical_flat_index;
  double fit_lambda = std::numeric_limits<double>::quiet_NaN();
  double fit_sigma = std::numeric_limits<double>::quiet_NaN();
  std::string fit_optimizer;
};

/// An oblivious tree ensemble imported from CatBoost.
///
/// The codes of an observation are one byte per split feature. Split features are
/// numbered: first the numeric ones, then the one-hot ones, then the CTR ones.
/// The leaf index of an observation in tree t is bit l = (split l of tree t is true),
/// index = sum_l bit_l * 2^l. The margin of an observation is bias + the sum of the leaf
/// values over all trees.
class ObliviousTreeEnsemble {
 public:
  /// Import the CatBoost JSON model `json`, with all of CatBoost's feature
  /// handling: numeric, one-hot and CTR features (including combinations).
  /// Throws Error if the model is malformed, has a tree of depth > 8, an
  /// unsupported loss, more than 255 borders per feature or per CTR, more than
  /// 255 one-hot values, or a split that does not match the feature data.
  static ObliviousTreeEnsemble from_catboost_json(std::string_view json);

  /// Load the model saved by to_bytes from `bytes`. Throws Error if the bytes are
  /// damaged, truncated, of another version, or internally inconsistent.
  static ObliviousTreeEnsemble from_bytes(std::span<const std::byte> bytes);

  /// Serialize to the format read by from_bytes; floats are copied raw, so a
  /// round trip is bit-exact.
  std::vector<std::byte> to_bytes() const;

  /// Write the raw margins of the observations of X (bias + leaf sums) to `out`,
  /// n x K, row-major.
  /// Throws Error if X.numeric.size() != X.n * n_numeric, X.categorical does
  /// not hold n_categorical spans of n values (dictionary-encoded: if
  /// X.categorical_codes does not hold n_categorical spans of n codes, each
  /// less than its dictionary's size), or out.size() != X.n * K.
  void predict_margin(const Batch& X, std::span<double> out) const;

  /// Write the task output of the margins of X -- identity (regression), sigmoid
  /// (binary) or softmax (multiclass) -- to `out`, n x K, row-major.
  /// Throws Error with the same shape rules as predict_margin.
  void predict(const Batch& X, std::span<double> out) const;

  /// Write the leaf index of every observation of X in every tree to `out`,
  /// n x n_trees, row-major.
  /// Throws Error with the same shape rules as predict_margin.
  void leaf_index(const Batch& X, std::span<uint8_t> out) const;

  /// Refit the leaf values and the bias to `train` following `options.strategy`,
  /// keeping the splits. The anchor is the current leaf values; the fit runs in
  /// float64 (regression on y / std(y)) and writes the result back as float32.
  /// `Fixed` runs once with `options.lambda`; `Path` runs the warm-started
  /// lambda path 1e2 .. 1e-4 and keeps the best run on the holdout; `EarlyStop`
  /// runs once with the holdout keeping the best iterate. With a holdout, no run
  /// strictly better than the start keeps the start parameters untouched
  /// ("no gain"). The lambda of the best run is stored for `update` and
  /// reported. With `options.split_budget` the ensemble is first pruned to that
  /// many split evaluations per observation, with `options.max_bytes` to that many
  /// bytes in the current leaf type: a hard limit kept also when the unpruned
  /// ensemble would do better on the holdout.
  /// The features of `train` and `holdout` are read once, to compute the leaf
  /// indices (with a budget twice: for the pruning and for the fit of the cut
  /// ensemble), so their spans may point into memory-mapped files; the fit holds
  /// the leaf indices (n x n_trees bytes) and the targets and weights in float64.
  /// Returns the objective before and after and the iteration count.
  /// Throws Error if y is not n * K values, weight is not n values, a weight is
  /// negative or their sum is 0, lambda < 0, learning_rate < 0, Path /
  /// EarlyStop has no holdout observations, a feature shape is wrong (as in
  /// predict_margin), both budgets are given, or max_bytes is smaller than the
  /// model without trees.
  FitReport fit(const Batch& train, const Batch& holdout, const FitOptions& options);

  /// Adapt the leaf values and the bias to a window of new data, without new
  /// trees: the call for data that arrives over time or is too large for one
  /// fit, one call per window. Runs exactly `steps` iterations of the
  /// optimizer of the last `fit` (with its learning rate) on the objective of
  /// that fit over the window: anchor = the leaf values before the call,
  /// lambda = `fit_lambda` and sigma = `fit_sigma` of the last fit (never
  /// recomputed from the window, so the units stay those of the fit). Adam
  /// continues from the state stored in the ensemble (zeros when there is
  /// none) and stores its new state there. Splits, borders and CTR tables stay
  /// fixed.
  /// Returns the window objective before and after the steps.
  /// Throws Error if no fit happened before, steps < 1, the window's y is not
  /// n * K values, its weight is not n values, or a weight is negative.
  UpdateReport update(const Batch& window, int steps);

  /// The ensemble shape: task, K, n_numeric, n_categorical, number of trees,
  /// number of splits, and the CatBoost column of every feature.
  Info info() const;

  /// The number of outputs K (info().n_outputs, without building the whole Info).
  uint32_t n_outputs() const { return K_; }

  /// The number of trees (info().n_trees, without building the whole Info).
  uint32_t n_trees() const { return static_cast<uint32_t>(depth_.size()); }

  /// One split of a tree. `source` is 0 = numeric (index = numeric column),
  /// 1 = one-hot (index = categorical feature index) and 2 = CTR (index = CTR
  /// feature). `kind` is 0 for "code > arg" and 1 for "code == arg"; `arg` is
  /// the index of the border or value inside that feature.
  struct Split {
    uint8_t source;
    uint32_t index;
    uint8_t kind;
    uint8_t arg;
  };

  /// The splits of tree `tree`, ordered from the root level to the leaf level;
  /// empty for a depth-0 tree.
  /// Throws Error if tree >= the number of trees.
  std::vector<Split> splits(uint32_t tree) const;

  /// The leaf values of tree `tree` as stored (int8 leaves dequantized, exact
  /// in float64): 2^depth x K values, leaf-major (value k of leaf j at j*K + k).
  /// Throws Error if tree >= the number of trees.
  std::vector<double> leaves(uint32_t tree) const;

  /// The bias, K values.
  std::vector<double> bias() const;

  /// Choose the prediction backend by name ("scalar", "avx2", "avx512",
  /// "avx512_vbmi", "neon"); "" selects the best backend available on this CPU. The
  /// choice is not serialized. Throws Error if the name is unknown or the
  /// backend is not available here.
  void set_backend(std::string_view name);

  /// Use `n` threads for predict_margin, predict and leaf_index (1 by
  /// default; 0 = one per hardware thread). Each thread takes a range of
  /// 256-observation blocks; small inputs use fewer threads. The choice is not
  /// serialized.
  void set_threads(unsigned n);

  /// Change how the leaf values are stored. Float32 -> Int8 quantizes the
  /// float leaves in blocks of 64 consecutive trees: with M the largest
  /// absolute leaf value of a block, a tree gets the step (M / 127) / 2^d with
  /// the largest d <= 7 for which its own largest absolute leaf value is at
  /// most M / 2^d, and int8 codes q = round-half-away-from-zero(value / step)
  /// clamped to [-127, 127]; a leaf's value is q * step. Prediction sums the
  /// codes of a block as integers, which is exact, so int8 is faster than
  /// float32 and every backend gives the same bits. Int8 -> Float32 stores the
  /// dequantized values (rounded to float32). The leaf type is part of the
  /// serialized model.
  void set_leaf_type(LeafType type);

 private:
  /// Rebuild the derived, not serialized arrays (prefix sums, split
  /// conditions, CTR hash tables, border search trees, leaf tables) from the
  /// stored arrays. Called after import, after from_bytes and whenever the
  /// leaves change.
  void rebuild_derived();

  /// Build the distinct split conditions and each split's condition (see
  /// OTEView). Part of rebuild_derived.
  void build_split_conditions();

  /// Build the hash tables of the CTR tables and the codes by slot from the
  /// stored keys and codes. Part of rebuild_derived.
  void build_ctr_hash_tables();

  /// Calls f(member) for every stored member after the header (the leaves
  /// of the current leaf type), in the order of the serialized format, so
  /// to_bytes and from_bytes cannot disagree. Self is this class, const or not.
  template <class Self, class F>
  static void for_each_stored(Self& e, F f);

  /// Check that the stored arrays are consistent with the header counts. Called
  /// by from_bytes before rebuild_derived. Throws Error on any mismatch.
  void check_arrays() const;

  /// The raw pointer view the kernels work on.
  OTEView view() const;

  /// The value of leaf value j of tree t: the stored float for Float32,
  /// q * step (exact in float64) for Int8.
  double leaf_value(size_t j, size_t tree) const;
  /// All leaf values as floats, in the layout of the leaves.
  std::vector<float> float_leaves() const;
  /// Stores leaf values `w` (all trees, the layout of the leaves, sized by depth_)
  /// in the current leaf type, quantizing for Int8, and rebuilds the derived
  /// arrays. The one place where leaves are written.
  void store_leaves(std::vector<float> w);

  /// Checks the fit inputs and builds the fixed data of a fit: sigma (the
  /// weighted standard deviation of y of `train`, or the given one), the
  /// anchor, and the training and holdout observations, all in fit units.
  FitData fit_data(const Batch& train, const Batch& holdout, const FitOptions& options,
                   std::optional<double> sigma = std::nullopt) const;
  /// The start parameters of a fit: the anchor, then the bias, in fit units.
  std::vector<double> start_theta(const FitData& d) const;
  /// Writes the fitted parameters back: float32 leaves and the bias, times sigma.
  void write_back(const FitData& d, const std::vector<double>& theta);
  /// The part of `fit` and `update` after fit_data: runs the strategy of
  /// `options` around runs of its optimizer, writes the result back unless
  /// there is no gain, and stores the fit state: lambda, sigma, optimizer,
  /// learning rate and the Adam state `st` (empty for the other optimizers,
  /// which clears the stored one).
  FitReport run_fit(FitData& d, const FitOptions& options, AdamState& st);

  /// Cut every tree to its kept level: tree t keeps its first k_t splits and the
  /// level-basis leaves from_level_basis(u_t, k_t) as float32 (times sigma); a
  /// tree with k_t = 0 is removed and its u_0 (one value per output) is added
  /// to the bias. `u` and `bias` are in fit units, laid out as in PruneResult.
  /// Returns the number of removed trees.
  uint32_t cut_to_levels(const std::vector<double>& u, const std::vector<uint32_t>& levels,
                        const std::vector<double>& bias, double sigma);

  /// The bytes a tree of depth k adds to to_bytes() in the current leaf type:
  /// its depth, k splits, 2^k * K leaves and, for Int8, its step.
  uint64_t tree_bytes(uint32_t k) const;

  /// Checks the shapes of X and of `out` (`out_per_obs` values per observation),
  /// encodes X in observation blocks and calls kernel(backend, view, codes, n_obs,
  /// out_block) per block, where out_block holds the block's observations of `out`.
  template <class T, class Kernel>
  void for_each_block(const Batch& X, std::span<T> out, size_t out_per_obs, Kernel kernel) const;

  Task task_ = Task::Regression;
  uint32_t K_ = 1;
  uint32_t n_numeric_ = 0;     // columns of Batch.numeric
  uint32_t n_categorical_ = 0;  // columns of Batch.categorical

  // CatBoost column (flat_feature_index) of every feature, for the Python layer.
  std::vector<uint32_t> numeric_flat_index_;      // n_numeric
  std::vector<uint32_t> categorical_flat_index_;  // n_categorical

  // Numeric encoder, one entry per numeric split feature.
  std::vector<uint32_t> numeric_column_;  // column of Batch.numeric
  std::vector<uint32_t> border_count_;    // <= 255 per feature
  std::vector<uint8_t> nan_high_;         // 1 = NaN counts as above all borders
  std::vector<float> borders_;            // all features concatenated, ascending

  // One-hot encoder, one entry per categorical feature that has values.
  std::vector<uint32_t> onehot_column_;  // categorical feature index
  std::vector<uint8_t> onehot_count_;    // <= 255 per feature
  std::vector<int32_t> onehot_values_;   // the signed hashes from "values", concatenated

  // CTR projections: the elements that build the combination hash, structure
  // of arrays, the elements of all projections concatenated.
  std::vector<uint32_t> proj_size_;      // elements per projection
  std::vector<uint8_t> elem_kind_;       // 0 = category hash, 1 = float border, 2 = category equals value
  std::vector<uint32_t> elem_column_;    // categorical index (kinds 0, 2), numeric column (kind 1)
  std::vector<float> elem_border_;       // kind 1
  std::vector<uint8_t> elem_nan_high_;   // kind 1: the float feature's nan_value_treatment == "AsTrue"
  std::vector<int32_t> elem_value_;      // kind 2

  // CTR tables: the keys of each table, ascending; the statistics are folded
  // into the codes of the CTRs that use the table.
  std::vector<uint32_t> table_projection_;  // projection of each table
  std::vector<uint32_t> table_size_;        // keys per table
  std::vector<uint64_t> table_keys_;       // all concatenated, ascending within a table

  // CTR split features. The import computes the code of every table entry once,
  // so prediction is: projection hash -> hash table lookup of the key -> code.
  std::vector<uint32_t> ctr_table_;        // table of each CTR
  std::vector<uint8_t> ctr_default_code_;  // code of a key that is not in the table
  std::vector<uint8_t> ctr_codes_;         // table_size[ctr_table[r]] codes per CTR, in key order

  // Trees. The splits of tree t are consecutive; sum depth[t] entries.
  std::vector<uint8_t> depth_;           // 0..8; depth 0 = one leaf
  std::vector<uint32_t> split_feature_;  // index of the split feature
  std::vector<uint8_t> split_kind_;      // 0 = GT "code > arg", 1 = EQ "code == arg"
  std::vector<uint8_t> split_arg_;
  std::vector<float> leaves_;            // tree t: 2^depth[t] x K, leaf-major (j*K + k)
  LeafType leaf_type_ = LeafType::Float32;
  std::vector<int8_t> leaves_q_;       // Int8: the quantized leaves, same layout as leaves_
  std::vector<float> leaf_scale_;      // Int8: the step of each tree (leaf value = q * step)
  std::vector<double> bias_;             // K
  double fit_lambda_ = std::numeric_limits<double>::quiet_NaN();  // lambda of the last fit

  // Fit state for `update`: the sigma, optimizer and learning rate of the last
  // fit, and the Adam state it ended with (empty when the optimizer was not Adam).
  double fit_sigma_ = std::numeric_limits<double>::quiet_NaN();
  uint8_t fit_optimizer_ = 0;  // OptimizerKind of the last fit
  double fit_learning_rate_ = 0.0;
  int64_t adam_t_ = 0;
  std::vector<double> adam_m_;
  std::vector<double> adam_v_;

  // Derived, not serialized: prefix sums.
  std::vector<uint32_t> border_begin_;       // F + 1, into borders_
  std::vector<uint32_t> onehot_value_begin_;  // G + 1, into onehot_values_
  std::vector<uint32_t> proj_elem_begin_;     // P + 1, into the elem arrays
  // Derived, not serialized: each CTR table as an open-addressing hash table
  // (see build_ctr_hash_tables in ensemble.cpp).
  std::vector<uint32_t> table_slot_begin_;    // Q + 1, into table_slot_keys_
  std::vector<uint64_t> table_slot_keys_;     // the keys by slot, empty_key at free slots
  std::vector<uint64_t> table_empty_key_;     // Q: a value that is not a key of the table
  std::vector<uint32_t> table_ctr_begin_;     // Q + 1, into table_ctrs_
  std::vector<uint32_t> table_ctrs_;          // the CTRs, grouped by table
  std::vector<uint32_t> table_order_;         // Q: the hashing order of the tables (see OTEView)
  std::vector<uint32_t> table_shared_;        // Q: elements shared with the table before in that order
  std::vector<uint8_t> ctr_slot_codes_;       // per CTR: the code of each slot of its table
  std::vector<uint32_t> ctr_slot_code_begin_;  // R + 1, into ctr_slot_codes_
  std::vector<uint32_t> split_begin_;         // T + 1, into the split arrays
  std::vector<uint32_t> cond_feature_;        // the distinct split conditions, sorted
  std::vector<uint8_t> cond_kind_;
  std::vector<uint8_t> cond_arg_;
  std::vector<uint32_t> split_cond_;          // the condition of each split
  std::vector<uint8_t> slot_feature_;         // the splits in 8 slots per tree (see OTEView::slot_feature)
  std::vector<uint8_t> slot_arg_;
  std::vector<uint64_t> slot_eq_;
  std::vector<uint32_t> leaf_begin_;          // T + 1, into leaves_; tree t occupies 2^depth[t] * K values

  // Derived, not serialized: each numeric split feature's borders as a
  // breadth-first binary search tree of 256 floats, for the binning of the kernels.
  std::vector<float> border_heap_;  // F * 256

  // Derived, not serialized: the leaf values of each tree and output stored
  // contiguous (leaves[j*K + k] over j), padded with zeros to at least 16
  // floats, so the SIMD backends can look a leaf up with register permutes.
  std::vector<float> leaves_by_output_;      // Float32
  std::vector<int8_t> leaves_q_by_output_;  // Int8
  std::vector<uint32_t> leaf_out_begin_;    // T*K + 1, into either table

  // Derived, not serialized, Int8: each tree's multiplier and each block's
  // unit (see OTEView).
  std::vector<uint8_t> tree_mult_;
  std::vector<double> block_unit_;

  // Derived, not serialized: the chosen backend; null = best_backend().
  const Backend* backend_ = nullptr;
  unsigned threads_ = 1;  // not serialized; see set_threads
};

}  // namespace boxwood
