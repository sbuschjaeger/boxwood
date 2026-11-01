#include "fit.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace boxwood {

void Squared::eval(const double* F, const double* y, uint32_t, double& loss, double* g, double* h) {
  const double d = F[0] - y[0];
  loss = 0.5 * d * d;
  g[0] = d;
  h[0] = 1.0;
}

void Logistic::eval(const double* F, const double* y, uint32_t, double& loss, double* g, double* h) {
  const double f = F[0];
  // One exp for both: softplus(f) = max(f,0) + log1p(e) and sigmoid(f) with
  // e = exp(-|f|), so neither overflows for any f.
  const double e = std::exp(-std::fabs(f));
  const double sp = std::max(f, 0.0) + std::log1p(e);
  const double p = f >= 0.0 ? 1.0 / (1.0 + e) : e / (1.0 + e);
  loss = sp - y[0] * f;
  g[0] = p - y[0];
  h[0] = p * (1.0 - p);
}

void Softmax::eval(const double* F, const double* y, uint32_t K, double& loss, double* g, double* h) {
  double m = F[0];
  for (uint32_t k = 1; k < K; ++k) m = std::max(m, F[k]);
  double sum = 0.0;
  for (uint32_t k = 0; k < K; ++k) sum += std::exp(F[k] - m);
  const double lse = m + std::log(sum);
  loss = lse;
  for (uint32_t k = 0; k < K; ++k) {
    loss -= y[k] * F[k];
    const double p = std::exp(F[k] - lse);
    g[k] = p - y[k];
    h[k] = p * (1.0 - p);
  }
}

namespace {

/// Stop-rule and holdout bookkeeping shared by every optimizer. After one
/// iteration `step` records the objective, keeps the best iterate by holdout
/// loss and reports whether the loop continues.
class FitLoop {
 public:
  FitLoop(const StopRule& stop, std::span<const double> theta, double J_start)
      : stop_(stop), last_J_(J_start), best_J_(J_start) {
    if (stop_.use_holdout) best_theta_.assign(theta.begin(), theta.end());
  }

  /// Records one finished iteration: objective `J` (started at `J_before`) and
  /// holdout loss `hold` (ignored without a holdout). Returns false to stop.
  bool step(std::span<const double> theta, double J_before, double J, double hold) {
    ++iters_;
    last_J_ = J;
    if (stop_.use_holdout) {
      if (hold < best_hold_) {
        best_hold_ = hold;
        best_J_ = J;
        best_theta_.assign(theta.begin(), theta.end());
        no_improve_ = 0;
      } else {
        ++no_improve_;
      }
    }
    // Converged on a small decrease. An increase (Adam can overshoot) is not
    // convergence: the loop keeps going. TreeNewton and backtracking GD never
    // increase J, so for them this is the plain relative-drop test.
    const double drop = J_before - J;
    if (drop >= 0.0 && drop <= stop_.tol * std::max(1.0, std::fabs(J))) {
      converged_ = true;
      return false;
    }
    return iters_ < stop_.max_iter && !(stop_.use_holdout && no_improve_ >= stop_.patience);
  }

  /// With a holdout, put the best iterate back into `theta` so the caller writes
  /// the parameters the report describes.
  void restore_best(std::vector<double>& theta) const {
    if (stop_.use_holdout) theta = best_theta_;
  }

  /// Whether the stop rule still allows another iteration; checked before each
  /// one so `max_iter = 0` runs none.
  bool running() const { return iters_ < stop_.max_iter; }

  FitReport report(double J_start) const {
    FitReport r;
    r.objective_start = J_start;
    r.objective = stop_.use_holdout ? best_J_ : last_J_;
    r.holdout_loss = stop_.use_holdout ? best_hold_ : std::numeric_limits<double>::quiet_NaN();
    r.iterations = iters_;
    r.converged = converged_;
    return r;
  }

 private:
  StopRule stop_;
  double last_J_;
  double best_J_;
  double best_hold_ = std::numeric_limits<double>::infinity();
  std::vector<double> best_theta_;
  int iters_ = 0;
  int no_improve_ = 0;
  bool converged_ = false;
};

/// The holdout's mean weighted loss (no penalty) at `theta`.
template <class Loss>
double holdout_loss(const FitData& f, std::span<const double> theta) {
  const uint32_t K = f.K, T = f.n_trees;
  const FitObservations& r = f.hold;
  std::vector<double> F(K), g(K), h(K);
  double sum = 0.0;
  for (size_t i = 0; i < r.n; ++i) {
    margin_from_theta(f, &r.leaf[i * T], theta.data(), F.data());
    double l;
    Loss::eval(F.data(), &r.y[i * K], K, l, g.data(), h.data());
    sum += r.weight[i] * l;
  }
  return sum / r.S;
}

/// The holdout loss the stop rule needs after an iteration (0 without a holdout).
template <class Loss>
double hold_for_stop(const FitData& f, std::span<const double> theta) {
  return f.stop.use_holdout ? holdout_loss<Loss>(f, theta) : 0.0;
}

/// TreeNewton. The blocks are the trees and, as block T, the bias (one leaf,
/// no penalty). A block's step is Newton's with the exact diagonal curvature
/// (each observation is in exactly one leaf of a tree); Logistic and Softmax halve it
/// while the objective increases (up to 10 times, else no step). The margins
/// of all observations are kept, and so are the loss derivatives at them: the loss at
/// an accepted step is evaluated with its derivatives, which the next block
/// sums. So a block step evaluates the loss once per observation (plus once per halving).
template <class Loss>
FitReport tree_newton(const FitData& f, std::vector<double>& theta) {
  const uint32_t K = f.K, T = f.n_trees;
  const FitObservations& r = f.train;
  const size_t n = r.n;
  const std::vector<uint8_t>& leaf = r.leaf;
  const std::vector<double>& y = r.y;
  const std::vector<double>& s = r.weight;
  const double S = r.S;

  // F: the margins; g, h: the loss derivatives at F. trial_g, trial_h: the
  // derivatives at the margins of the last data_loss call.
  std::vector<double> F(n * K), Fc(K), g(n * K), h(n * K), trial_g(n * K), trial_h(n * K);
  for (size_t i = 0; i < n; ++i) margin_from_theta(f, &leaf[i * T], theta.data(), &F[i * K]);
  // Mean weighted loss with the margins moved by c * step[slot(i) + k].
  const auto data_loss = [&](const auto& slot, const std::vector<double>& step, double c) {
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
      for (uint32_t k = 0; k < K; ++k) Fc[k] = F[i * K + k] + c * step[slot(i) + k];
      double l;
      Loss::eval(Fc.data(), &y[i * K], K, l, &trial_g[i * K], &trial_h[i * K]);
      sum += s[i] * l;
    }
    return sum / S;
  };
  const std::vector<double> no_step(K, 0.0);
  double data = data_loss([](size_t) { return size_t{0}; }, no_step, 0.0);
  g.swap(trial_g);
  h.swap(trial_h);
  double pen = penalty(f, theta);
  const double J_start = data + pen;
  double J = J_start;
  FitLoop loop(f.stop, theta, J_start);
  std::vector<double> G, H, step;  // of one block, reused across blocks

  while (loop.running()) {
    const double J_before = J;
    for (uint32_t b = 0; b <= T; ++b) {
      const bool is_bias = b == T;
      const size_t off = is_bias ? f.total : f.leaf_begin[b];
      const size_t len = is_bias ? K : f.leaf_begin[b + 1] - f.leaf_begin[b];
      const auto slot = [&](size_t i) { return is_bias ? size_t{0} : size_t{leaf[i * T + b]} * K; };
      G.assign(len, 0.0);
      H.assign(len, 0.0);
      for (size_t i = 0; i < n; ++i) {
        for (uint32_t k = 0; k < K; ++k) {
          G[slot(i) + k] += s[i] * g[i * K + k];
          H[slot(i) + k] += s[i] * h[i * K + k];
        }
      }
      const double lam = is_bias ? 0.0 : f.lambda;
      step.resize(len);
      for (size_t j = 0; j < len; ++j) {
        const double Gj = G[j] / S + lam * (is_bias ? 0.0 : theta[off + j] - f.anchor[off + j]);
        const double Hj = H[j] / S + lam;
        step[j] = Hj != 0.0 ? -Gj / Hj : 0.0;
      }
      for (int bt = 0; bt <= (Loss::line_search ? 10 : 0); ++bt) {
        const double c = std::ldexp(1.0, -bt);
        const double data_c = data_loss(slot, step, c);
        double pen_c = pen;
        for (size_t j = 0; is_bias == false && j < len; ++j) {
          const double d0 = theta[off + j] - f.anchor[off + j];
          const double d1 = d0 + c * step[j];
          pen_c += 0.5 * lam * (d1 * d1 - d0 * d0);
        }
        if (!Loss::line_search || data_c + pen_c <= data + pen) {
          for (size_t j = 0; j < len; ++j) theta[off + j] += c * step[j];
          for (size_t i = 0; i < n; ++i) {
            for (uint32_t k = 0; k < K; ++k) F[i * K + k] += c * step[slot(i) + k];
          }
          g.swap(trial_g);
          h.swap(trial_h);
          data = data_c;
          pen = pen_c;
          break;
        }
      }
    }
    J = data + pen;
    if (!loop.step(theta, J_before, J, hold_for_stop<Loss>(f, theta))) break;
  }
  loop.restore_best(theta);
  return loop.report(J_start);
}

// ---------------------------------------------------------------------------
// Gradient descent and Adam: one pass gives the objective and its gradient;
// the optimizers step from the pass sums (see fit.hpp).

/// The iteration loop of gradient descent and Adam: J at the start from one
/// pass, then `iteration(sums, J)` until the stop rule ends. An iteration
/// moves theta, leaves the pass sums at the new theta in `sums` and returns
/// the new J. With a holdout the best iterate is put back into theta.
template <class Loss, class Iteration>
FitReport pass_loop(const FitData& f, std::vector<double>& theta, Iteration iteration) {
  PassSums sums = pass<Loss>(f, theta);
  const double J_start = objective(f, sums, theta);
  double J = J_start;
  FitLoop loop(f.stop, theta, J_start);
  while (loop.running()) {
    const double J_before = J;
    J = iteration(sums, J_before);
    if (!loop.step(theta, J_before, J, hold_for_stop<Loss>(f, theta))) break;
  }
  loop.restore_best(theta);
  return loop.report(J_start);
}

/// The gradient of J from the pass sums `s` (leaf values and bias).
std::vector<double> gradient(const FitData& f, const PassSums& s, std::span<const double> theta) {
  std::vector<double> G(f.total + f.K);
  for (size_t j = 0; j < G.size(); ++j) {
    double H;  // unused: gradient descent and Adam need only the gradient
    grad_curv(f, s, theta, j, G[j], H);
  }
  return G;
}

/// GradientDescent: theta <- theta - eta*G with backtracking (start at eta,
/// halve until J does not increase, up to 30 times; after an accepted step
/// try 2*eta next time). No step that keeps J from increasing ends the fit.
template <class Loss>
FitReport gradient_descent(const FitData& f, std::vector<double>& theta, double lr) {
  double eta = lr > 0.0 ? lr : 1.0;
  return pass_loop<Loss>(f, theta, [&](PassSums& sums, double J) {
    const std::vector<double> G = gradient(f, sums, theta);
    for (int bt = 0; bt <= 30; ++bt, eta *= 0.5) {
      std::vector<double> trial(theta.size());
      for (size_t j = 0; j < trial.size(); ++j) trial[j] = theta[j] - eta * G[j];
      PassSums ts = pass<Loss>(f, trial);
      const double Jt = objective(f, ts, trial);
      if (Jt <= J) {
        theta = std::move(trial);
        sums = std::move(ts);
        eta *= 2.0;
        return Jt;
      }
    }
    return J;  // no step: the drop is 0, so the loop counts it as converged
  });
}

/// Adam (beta1 = 0.9, beta2 = 0.999, eps = 1e-8) with bias correction, one
/// step per pass. The state `st` goes in and comes out, so a later run (the
/// next lambda of a path, or `update`) continues where this one ended.
template <class Loss>
FitReport adam(const FitData& f, std::vector<double>& theta, double lr, AdamState& st) {
  const size_t P = f.total + f.K;
  if (st.m.size() != P || st.v.size() != P) {  // no usable state: start fresh
    st.m.assign(P, 0.0);
    st.v.assign(P, 0.0);
    st.t = 0;
  }
  const double eta = lr > 0.0 ? lr : 0.01;
  const double b1 = 0.9, b2 = 0.999, eps = 1e-8;
  return pass_loop<Loss>(f, theta, [&](PassSums& sums, double) {
    const std::vector<double> G = gradient(f, sums, theta);
    ++st.t;
    const double c1 = 1.0 - std::pow(b1, static_cast<double>(st.t));
    const double c2 = 1.0 - std::pow(b2, static_cast<double>(st.t));
    for (size_t j = 0; j < P; ++j) {
      st.m[j] = b1 * st.m[j] + (1.0 - b1) * G[j];
      st.v[j] = b2 * st.v[j] + (1.0 - b2) * G[j] * G[j];
      theta[j] -= eta * (st.m[j] / c1) / (std::sqrt(st.v[j] / c2) + eps);
    }
    sums = pass<Loss>(f, theta);
    return objective(f, sums, theta);
  });
}

}  // namespace

double holdout_loss_at(Task task, const FitData& d, std::span<const double> theta) {
  return with_loss(task, [&](auto loss) { return holdout_loss<decltype(loss)>(d, theta); });
}

FitReport run_optimizer(Task task, OptimizerKind optimizer, double learning_rate, const FitData& d,
                        std::vector<double>& theta, AdamState& adam_state) {
  return with_loss(task, [&](auto loss) {
    using Loss = decltype(loss);
    switch (optimizer) {
      case OptimizerKind::Adam:
        return adam<Loss>(d, theta, learning_rate, adam_state);
      case OptimizerKind::GradientDescent:
        return gradient_descent<Loss>(d, theta, learning_rate);
      case OptimizerKind::TreeNewton:
        break;
    }
    return tree_newton<Loss>(d, theta);
  });
}

}  // namespace boxwood
