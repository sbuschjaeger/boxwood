// The entry points of the leaf refit: input checks and the fixed data of a fit,
// the strategies for lambda, pruning to a split budget, and the online update.

#include "boxwood/ensemble.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <optional>
#include <string>
#include <utility>

#include "fit.hpp"
#include "prune.hpp"

namespace boxwood {

FitData ObliviousTreeEnsemble::fit_data(const Batch& train, const Batch& holdout,
                                        const FitOptions& options,
                                        std::optional<double> sigma) const {
  BW_CHECK(options.lambda >= 0.0, "lambda must be >= 0");
  BW_CHECK(options.learning_rate >= 0.0, "learning_rate must be >= 0");
  BW_CHECK(!options.stop.use_holdout || holdout.n > 0, "use_holdout needs a holdout with observations");
  const auto check_targets = [&](const Batch& b) {
    BW_CHECK(b.y.size() == b.n * K_, "y size does not match n * n_outputs");
    BW_CHECK(b.weight.empty() || b.weight.size() == b.n, "weight size does not match n");
    for (const double v : b.weight) BW_CHECK(v >= 0.0, "weights must be >= 0");
  };
  check_targets(train);
  if (holdout.n > 0) check_targets(holdout);

  // Regression without a given sigma: sigma = the weighted standard deviation
  // of y (West's one-pass update, stable for any mean; 1 if the variance is
  // 0). The fit then works on y / sigma.
  double S = 0.0, mean = 0.0, m2 = 0.0;
  if (!sigma && task_ == Task::Regression) {
    for (size_t i = 0; i < train.n; ++i) {
      const double s = train.weight.empty() ? 1.0 : train.weight[i];
      if (s == 0.0) continue;
      S += s;
      const double delta = train.y[i] - mean;
      mean += s / S * delta;
      m2 += s * delta * (train.y[i] - mean);
    }
  }

  const size_t T = depth_.size();
  FitData d;
  d.K = K_;
  d.n_trees = static_cast<uint32_t>(T);
  d.leaf_begin = leaf_begin_.data();
  d.total = leaf_begin_[T];
  d.lambda = options.lambda;
  d.sigma = sigma.value_or(m2 > 0.0 ? std::sqrt(m2 / S) : 1.0);
  d.stop = options.stop;
  d.anchor.resize(d.total);
  for (size_t t = 0; t < T; ++t) {
    for (size_t j = leaf_begin_[t]; j < leaf_begin_[t + 1]; ++j) {
      d.anchor[j] = static_cast<double>(leaf_value(j, t)) / d.sigma;
    }
  }

  // The observations in fit units, with their leaf indices: the only read of the
  // features during a fit.
  const auto fit_observations = [&](const Batch& b) {
    FitObservations r;
    r.n = b.n;
    r.leaf.resize(b.n * T);
    leaf_index(b, std::span<uint8_t>(r.leaf));
    r.y.resize(b.n * K_);
    for (size_t i = 0; i < b.n * K_; ++i) r.y[i] = b.y[i] / d.sigma;
    r.weight = b.weight.empty() ? std::vector<double>(b.n, 1.0) : std::vector<double>(b.weight.begin(), b.weight.end());
    for (const double v : r.weight) r.S += v;
    return r;
  };
  d.train = fit_observations(train);
  BW_CHECK(d.train.S > 0.0, "the sum of the weights must be positive");
  if (holdout.n > 0) {
    d.hold = fit_observations(holdout);
    BW_CHECK(d.hold.S > 0.0, "the sum of the holdout weights must be positive");
  }
  return d;
}

std::vector<double> ObliviousTreeEnsemble::start_theta(const FitData& d) const {
  std::vector<double> theta(d.anchor);
  for (uint32_t k = 0; k < K_; ++k) theta.push_back(bias_[k] / d.sigma);
  return theta;
}

void ObliviousTreeEnsemble::write_back(const FitData& d, const std::vector<double>& theta) {
  std::vector<float> w(d.total);
  for (size_t j = 0; j < d.total; ++j) w[j] = static_cast<float>(theta[j] * d.sigma);
  for (uint32_t k = 0; k < K_; ++k) bias_[k] = theta[d.total + k] * d.sigma;
  store_leaves(std::move(w));
}

namespace {

/// The path stops after this many lambdas in a row without a better holdout loss.
constexpr int kPathPatience = 2;

/// Runs the strategy of `options` around runs of its optimizer, which read
/// lambda and the stop rule from `d` (set here before each run) and Adam's
/// state from `st`. `theta` starts at the anchor (the leaves before the call)
/// and every run warm-starts from the previous one; the anchor stays the same
/// for all lambdas.
/// On return `theta` holds the parameters to write back and `fit_lambda` the λ
/// of the best run (what `update` uses later). With a holdout a run improves
/// only with a strictly lower holdout loss; if no run beats the start, the
/// report says `no_gain` and `theta` stays at the start parameters.
FitReport run_strategy(Task task, FitData& d, const FitOptions& options, std::vector<double>& theta,
                       double& fit_lambda, AdamState& st) {
  const bool has_hold = d.hold.n > 0;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double hold_start = has_hold ? holdout_loss_at(task, d, theta) : nan;
  const std::vector<double> theta_start(theta);

  StopRule stop = options.stop;
  std::vector<double> lambdas;
  switch (options.strategy) {
    case Strategy::Fixed:
      lambdas = {options.lambda};
      break;
    case Strategy::Path:
      BW_CHECK(has_hold, "the path strategy needs a holdout");
      stop.use_holdout = false;  // the path compares the runs itself, on the holdout
      lambdas = {1e2, 1e1, 1.0, 1e-1, 1e-2, 1e-3, 1e-4};
      break;
    case Strategy::EarlyStop:
      BW_CHECK(has_hold, "the early_stop strategy needs a holdout");
      stop.use_holdout = true;
      lambdas = {options.lambda};
      break;
  }

  FitReport rep{};
  rep.holdout_loss_start = hold_start;
  std::vector<double> best_theta;
  FitReport best{};
  double best_hold = std::numeric_limits<double>::infinity();
  int no_improve = 0;
  for (size_t i = 0; i < lambdas.size(); ++i) {
    const double lam = lambdas[i];
    d.lambda = lam;
    d.stop = stop;
    const FitReport r = run_optimizer(task, options.optimizer, options.learning_rate, d, theta, st);
    const double hold = has_hold ? holdout_loss_at(task, d, theta) : nan;
    if (i == 0) rep.objective_start = r.objective_start;  // the first run starts at the anchor
    rep.path.push_back(PathStep{lam, r.objective, hold, r.iterations});
    if (options.verbose) {
      std::fprintf(stderr, "boxwood fit: lambda %.0e, objective %.6g, holdout loss %.6g, %d iterations\n", lam,
                   r.objective, hold, r.iterations);
    }
    if (!has_hold || hold < best_hold) {
      best_hold = hold;
      best_theta = theta;
      best = r;
      best.holdout_loss = hold;
      fit_lambda = lam;
      no_improve = 0;
    } else {
      ++no_improve;
      if (options.strategy == Strategy::Path && no_improve >= kPathPatience) break;
    }
  }

  if (has_hold && !(best_hold < hold_start)) {
    rep.no_gain = true;
    theta = theta_start;
    rep.objective = rep.objective_start;  // J at the start theta: the penalty is 0 there
    rep.holdout_loss = hold_start;
    rep.iterations = 0;
    rep.converged = true;
    rep.lambda = nan;
  } else {
    theta = best_theta;
    rep.objective = best.objective;
    rep.holdout_loss = best_hold;
    rep.iterations = best.iterations;
    rep.converged = best.converged;
    rep.lambda = fit_lambda;
  }
  return rep;
}

}  // namespace

uint64_t ObliviousTreeEnsemble::tree_bytes(uint32_t k) const {
  const bool q = leaf_type_ == LeafType::Int8;
  const uint64_t split = sizeof(split_feature_[0]) + sizeof(split_kind_[0]) + sizeof(split_arg_[0]);
  const uint64_t leaf = q ? sizeof(leaves_q_[0]) : sizeof(leaves_[0]);
  return sizeof(depth_[0]) + k * split + (uint64_t{1} << k) * K_ * leaf + (q ? sizeof(leaf_scale_[0]) : 0);
}

FitReport ObliviousTreeEnsemble::fit(const Batch& train, const Batch& holdout,
                                    const FitOptions& options) {
  BW_CHECK(!(options.split_budget && options.max_bytes), "give split_budget or max_bytes, not both");
  if (!options.split_budget && !options.max_bytes) {
    FitData d = fit_data(train, holdout, options);
    AdamState fresh;
    return run_fit(d, options, fresh);
  }
  // The cost of keeping k levels of tree t: k split evaluations, or the
  // bytes of the cut tree (a tree cut to 0 levels goes into the bias). For
  // bytes, the budget of the trees is max_bytes minus everything else.
  LevelCost cost(depth_.size());
  uint64_t budget = 0;
  for (size_t t = 0; t < depth_.size(); ++t) {
    for (uint32_t k = 0; k <= depth_[t]; ++k) {
      cost[t].push_back(options.split_budget ? k : k == 0 ? 0 : tree_bytes(k));
    }
  }
  if (options.split_budget) {
    budget = *options.split_budget;
  } else {
    const uint64_t rest =
        to_bytes().size() - (adam_m_.size() + adam_v_.size()) * sizeof(double) - info().tree_bytes;
    BW_CHECK(*options.max_bytes >= rest,
             "max_bytes is smaller than the model without trees (" + std::to_string(rest) + " bytes)");
    budget = *options.max_bytes - rest;
  }
  // Prune in the level basis of the current leaves, on the objective of the
  // fit (anchor = these leaves, lambda = options.lambda), cut this ensemble to
  // the kept levels and fit its leaf values as a fit without a budget would
  // (the anchor of that fit is the cut ensemble's leaves).
  uint32_t removed = 0;
  {
    const FitData pd = fit_data(train, holdout, options);
    std::vector<double> bias0(K_);
    for (uint32_t k = 0; k < K_; ++k) bias0[k] = bias_[k] / pd.sigma;
    const PruneResult pr = prune_to_budget(task_, pd, std::move(bias0), cost, budget);
    removed = cut_to_levels(pr.u, pr.levels, pr.bias, pd.sigma);
  }
  FitOptions plain = options;
  plain.split_budget = std::nullopt;
  plain.max_bytes = std::nullopt;
  FitReport r = fit(train, holdout, plain);
  r.trees_removed = removed;
  return r;
}

UpdateReport ObliviousTreeEnsemble::update(const Batch& window, int steps) {
  BW_CHECK(!std::isnan(fit_lambda_), "update needs a previous fit");
  BW_CHECK(steps >= 1, "steps must be >= 1");
  FitOptions opt;
  opt.lambda = fit_lambda_;
  opt.optimizer = static_cast<OptimizerKind>(fit_optimizer_);
  opt.learning_rate = fit_learning_rate_;
  // tol < 0 never counts as converged, so the run does exactly `steps` iterations.
  opt.stop = StopRule{steps, -1.0, false, 1};
  FitData d = fit_data(window, Batch{}, opt, fit_sigma_);
  AdamState st{adam_t_, std::move(adam_m_), std::move(adam_v_)};
  const FitReport r = run_fit(d, opt, st);
  return UpdateReport{r.objective_start, r.objective, steps};
}

FitReport ObliviousTreeEnsemble::run_fit(FitData& d, const FitOptions& options, AdamState& st) {
  std::vector<double> theta = start_theta(d);
  double fit_lambda = options.lambda;
  FitReport r = run_strategy(task_, d, options, theta, fit_lambda, st);
  r.splits_used = static_cast<uint32_t>(split_feature_.size());
  if (!r.no_gain) write_back(d, theta);  // with no gain the leaves and bias stay bit-identical
  fit_lambda_ = fit_lambda;
  fit_sigma_ = d.sigma;
  fit_optimizer_ = static_cast<uint8_t>(options.optimizer);
  fit_learning_rate_ = options.learning_rate;
  adam_t_ = st.t;
  adam_m_ = std::move(st.m);
  adam_v_ = std::move(st.v);
  return r;
}

uint32_t ObliviousTreeEnsemble::cut_to_levels(const std::vector<double>& u,
                                             const std::vector<uint32_t>& levels,
                                             const std::vector<double>& bias, double sigma) {
  std::vector<uint8_t> new_depth;
  std::vector<uint32_t> new_split_feature;
  std::vector<uint8_t> new_split_kind, new_split_arg;
  std::vector<float> new_leaves;
  std::vector<double> new_bias(K_);
  for (uint32_t k = 0; k < K_; ++k) new_bias[k] = bias[k] * sigma;
  uint32_t removed = 0;
  size_t u_off = 0;
  for (size_t t = 0; t < depth_.size(); ++t) {
    const uint32_t keep = levels[t];
    const size_t nodes = (size_t{1} << (depth_[t] + 1)) - 1;
    if (keep == 0) {
      ++removed;
      for (uint32_t k = 0; k < K_; ++k) new_bias[k] += u[u_off + k] * sigma;  // the u_0 of the tree
    } else {
      new_depth.push_back(static_cast<uint8_t>(keep));
      for (uint32_t s = split_begin_[t]; s < split_begin_[t] + keep; ++s) {
        new_split_feature.push_back(split_feature_[s]);
        new_split_kind.push_back(split_kind_[s]);
        new_split_arg.push_back(split_arg_[s]);
      }
      const size_t L = size_t{1} << keep;
      std::vector<std::vector<double>> w(K_);
      for (uint32_t k = 0; k < K_; ++k) {
        std::vector<double> uk(nodes);
        for (size_t i = 0; i < nodes; ++i) uk[i] = u[u_off + i * K_ + k];
        w[k] = from_level_basis(uk, keep);
      }
      for (size_t j = 0; j < L; ++j)
        for (uint32_t k = 0; k < K_; ++k) new_leaves.push_back(static_cast<float>(w[k][j] * sigma));
    }
    u_off += nodes * K_;
  }
  depth_ = std::move(new_depth);
  split_feature_ = std::move(new_split_feature);
  split_kind_ = std::move(new_split_kind);
  split_arg_ = std::move(new_split_arg);
  bias_ = std::move(new_bias);
  store_leaves(std::move(new_leaves));  // reads the new depth_; rebuilds the derived arrays
  return removed;
}

}  // namespace boxwood
