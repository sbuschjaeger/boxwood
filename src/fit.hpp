#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "boxwood/ensemble.hpp"

namespace boxwood {

/// Per-observation loss with its gradient and curvature, one small struct per task.
/// `eval` writes the loss value, g = dloss/dF_k and h = d2loss/dF_k^2 for every
/// output k, reading the margin F and the target y (K values each).
struct Squared {
  static constexpr bool line_search = false;  // the Newton step is exact
  static void eval(const double* F, const double* y, uint32_t K, double& loss, double* g, double* h);
};

struct Logistic {
  static constexpr bool line_search = true;
  static void eval(const double* F, const double* y, uint32_t K, double& loss, double* g, double* h);
};

struct Softmax {
  static constexpr bool line_search = true;
  static void eval(const double* F, const double* y, uint32_t K, double& loss, double* g, double* h);
};

/// The observations of a fit as the optimizers read them: the leaf index of every observation
/// in every tree (n x n_trees), the target in fit units (y / sigma, n x K) and
/// the weights (n values, 1 when none are given) with their sum S. The
/// features are read once, to compute the leaf indices.
struct FitObservations {
  size_t n = 0;
  std::vector<uint8_t> leaf;
  std::vector<double> y;
  std::vector<double> weight;
  double S = 0.0;
};

/// The fixed data of a fit. The parameters are one flat vector theta: the leaf
/// values of all trees in `leaf_begin` order, then the K bias values. Everything
/// is float64 and in the units of the fit (regression: divided by sigma).
struct FitData {
  uint32_t K = 1;
  uint32_t n_trees = 0;
  const uint32_t* leaf_begin = nullptr;  // n_trees + 1
  size_t total = 0;                     // number of leaf values
  double lambda = 0.0;
  double sigma = 1.0;                   // regression scale, 1 otherwise
  std::vector<double> anchor;           // total values: the leaves at the start of the fit
  StopRule stop;
  FitObservations train;
  FitObservations hold;  // n = 0: no holdout
};

/// Adam's optimizer state: the step count and the first and second moments,
/// one value per parameter. Empty moments mean a fresh start.
struct AdamState {
  int64_t t = 0;
  std::vector<double> m;
  std::vector<double> v;
};

// ---------------------------------------------------------------------------
// The pass over the training observations gives the objective, its gradient and its
// diagonal curvature. Shared by the optimizers and by the pruning fit.

/// The raw float64 sums of a pass: the weighted loss, the weight, and the
/// gradient and curvature per parameter -- the leaf values in `leaf_begin`
/// order, then the K bias outputs.
struct PassSums {
  double loss = 0.0;
  double S = 0.0;
  std::vector<double> g;  // total + K
  std::vector<double> h;  // total + K
};

/// F = bias + the leaf value of every tree the observation falls in, read from the flat
/// `theta`. `leaf_x` is the observation's leaf index per tree (n_trees values).
inline void margin_from_theta(const FitData& f, const uint8_t* leaf_x, const double* theta,
                             double* F) {
  for (uint32_t k = 0; k < f.K; ++k) F[k] = theta[f.total + k];
  for (uint32_t t = 0; t < f.n_trees; ++t) {
    const double* lv = theta + f.leaf_begin[t] + static_cast<size_t>(leaf_x[t]) * f.K;
    for (uint32_t k = 0; k < f.K; ++k) F[k] += lv[k];
  }
}

/// The penalty (lambda/2) * ||w - anchor||^2 of the leaf part of `theta`.
inline double penalty(const FitData& f, std::span<const double> theta) {
  double p = 0.0;
  for (size_t j = 0; j < f.total; ++j) {
    const double d = theta[j] - f.anchor[j];
    p += 0.5 * f.lambda * d * d;
  }
  return p;
}

/// One pass over the training observations with parameters `theta`.
template <class Loss>
inline PassSums pass(const FitData& f, std::span<const double> theta) {
  const uint32_t K = f.K, T = f.n_trees;
  const FitObservations& r = f.train;
  PassSums out;
  out.g.assign(f.total + K, 0.0);
  out.h.assign(f.total + K, 0.0);
  std::vector<double> F(K), g(K), h(K);
  for (size_t i = 0; i < r.n; ++i) {
    margin_from_theta(f, &r.leaf[i * T], theta.data(), F.data());
    double l;
    Loss::eval(F.data(), &r.y[i * K], K, l, g.data(), h.data());
    const double s = r.weight[i];
    out.loss += s * l;
    for (uint32_t t = 0; t < T; ++t) {
      const size_t base = f.leaf_begin[t] + static_cast<size_t>(r.leaf[i * T + t]) * K;
      for (uint32_t k = 0; k < K; ++k) {
        out.g[base + k] += s * g[k];
        out.h[base + k] += s * h[k];
      }
    }
    for (uint32_t k = 0; k < K; ++k) {
      out.g[f.total + k] += s * g[k];
      out.h[f.total + k] += s * h[k];
    }
  }
  out.S = r.S;
  return out;
}

/// J = (1/S) * weighted loss + penalty.
inline double objective(const FitData& f, const PassSums& sums, std::span<const double> theta) {
  return sums.loss / sums.S + penalty(f, theta);
}

/// The full-batch gradient and curvature of parameter `j` from the pass sums:
/// the normalized loss derivative plus, for a leaf value, the penalty term.
inline void grad_curv(const FitData& f, const PassSums& s, std::span<const double> theta, size_t j,
                     double& G, double& H) {
  const bool leaf_val = j < f.total;
  G = s.g[j] / s.S + (leaf_val ? f.lambda * (theta[j] - f.anchor[j]) : 0.0);
  H = s.h[j] / s.S + (leaf_val ? f.lambda : 0.0);
}

/// Calls fn(Loss{}) with the loss of the task.
template <class Fn>
auto with_loss(Task task, Fn fn) -> decltype(fn(Squared{})) {
  switch (task) {
    case Task::Regression:
      return fn(Squared{});
    case Task::Binary:
      return fn(Logistic{});
    case Task::Multiclass:
      return fn(Softmax{});
  }
  return decltype(fn(Squared{})){};
}

/// Runs `optimizer` (with `learning_rate`, 0 = its default) on the fit `d`
/// until its stop rule ends it; theta starts at the start parameters and holds
/// the result. Adam starts from `adam` (fresh when its moments are empty) and
/// leaves its new state there; the other optimizers do not touch it.
FitReport run_optimizer(Task task, OptimizerKind optimizer, double learning_rate, const FitData& d,
                        std::vector<double>& theta, AdamState& adam);

/// The holdout's mean weighted loss (no penalty) at `theta`. Needs a holdout
/// (d.hold.n > 0); the caller checks.
double holdout_loss_at(Task task, const FitData& d, std::span<const double> theta);

}  // namespace boxwood
