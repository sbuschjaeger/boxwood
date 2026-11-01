#include "prune.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <utility>

namespace boxwood {

std::vector<double> to_level_basis(std::span<const double> w) {
  const size_t n = w.size();  // 2^d
  const uint32_t d = static_cast<uint32_t>(std::bit_width(n) - 1);
  // v_l(p) = mean of the leaves j with j mod 2^l = p; the children of (l, p)
  // are (l+1, p) and (l+1, p + 2^l) because bit l of j is the split-l result.
  std::vector<double> v(2 * n - 1);
  std::copy(w.begin(), w.end(), v.begin() + (n - 1));  // v_d = w
  for (uint32_t l = d; l-- > 0;) {
    const size_t base = (size_t{1} << l) - 1, child = 2 * base + 1, half = size_t{1} << l;
    for (size_t p = 0; p < half; ++p) v[base + p] = 0.5 * (v[child + p] + v[child + half + p]);
  }
  std::vector<double> u(2 * n - 1);
  u[0] = v[0];
  for (uint32_t l = 1; l <= d; ++l) {
    const size_t base = (size_t{1} << l) - 1, pbase = (size_t{1} << (l - 1)) - 1;
    const size_t mask = (size_t{1} << (l - 1)) - 1;
    for (size_t p = 0; p < (size_t{1} << l); ++p) u[base + p] = v[base + p] - v[pbase + (p & mask)];
  }
  return u;
}

std::vector<double> from_level_basis(std::span<const double> u, uint32_t k) {
  const size_t n = size_t{1} << k;
  std::vector<double> w(n, 0.0);
  for (uint32_t l = 0; l <= k; ++l) {
    const size_t base = (size_t{1} << l) - 1, mask = base;
    for (size_t p = 0; p < n; ++p) w[p] += u[base + (p & mask)];
  }
  return w;
}

std::vector<uint32_t> choose_levels(const std::vector<std::vector<double>>& g, const LevelCost& cost,
                                    uint64_t B) {
  struct Seg {
    double slope;
    uint32_t tree;
    uint32_t len;   // levels
    uint64_t cost;
  };
  std::vector<Seg> segs;
  for (size_t t = 0; t < g.size(); ++t) {
    // Upper concave hull of the points (c[k], g[t][k]): keep only left turns.
    const std::vector<uint64_t>& c = cost[t];
    std::vector<size_t> hull;
    for (size_t k = 0; k < g[t].size(); ++k) {
      while (hull.size() >= 2) {
        const size_t a = hull[hull.size() - 2], b = hull.back();
        if ((g[t][b] - g[t][a]) * static_cast<double>(c[k] - c[b]) <=
            (g[t][k] - g[t][b]) * static_cast<double>(c[b] - c[a]))
          hull.pop_back();
        else
          break;
      }
      hull.push_back(k);
    }
    for (size_t i = 1; i < hull.size(); ++i) {
      const size_t a = hull[i - 1], b = hull[i];
      segs.push_back({(g[t][b] - g[t][a]) / static_cast<double>(c[b] - c[a]), static_cast<uint32_t>(t),
                      static_cast<uint32_t>(b - a), c[b] - c[a]});
    }
  }
  std::stable_sort(segs.begin(), segs.end(),
                  [](const Seg& x, const Seg& y) { return x.slope > y.slope; });
  std::vector<uint32_t> k(g.size(), 0);
  uint64_t rem = B;
  for (const Seg& s : segs) {
    if (s.cost > rem) break;
    rem -= s.cost;
    k[s.tree] += s.len;
  }
  return k;
}

namespace {

/// Iterative hard thresholding. The model of this phase is the full ensemble
/// with the leaf values w(u) rebuilt from the level basis, so the leaf indices
/// and the objective of the fit apply unchanged; the projection P keeps the
/// levels choose_levels picked for the current u and zeroes the rest.
template <class Loss>
PruneResult iht(const FitData& f, std::vector<double> bias, const LevelCost& cost, uint64_t budget) {
  const uint32_t K = f.K, T = f.n_trees;
  std::vector<uint32_t> depth(T);
  std::vector<size_t> begin(T + 1, 0);  // tree block starts in u
  for (uint32_t t = 0; t < T; ++t) {
    const size_t leaves = (f.leaf_begin[t + 1] - f.leaf_begin[t]) / K;  // 2^d
    depth[t] = static_cast<uint32_t>(std::bit_width(leaves) - 1);
    begin[t + 1] = begin[t] + (2 * leaves - 1) * K;
  }

  // u_start: the anchor in the level basis, per tree and output.
  std::vector<double> u(begin[T], 0.0);
  for (uint32_t t = 0; t < T; ++t) {
    const size_t L = size_t{1} << depth[t];
    for (uint32_t k = 0; k < K; ++k) {
      std::vector<double> w(L);
      for (size_t j = 0; j < L; ++j) w[j] = f.anchor[f.leaf_begin[t] + j * K + k];
      const std::vector<double> ub = to_level_basis(w);
      for (size_t i = 0; i < ub.size(); ++i) u[begin[t] + i * K + k] = ub[i];
    }
  }

  // P: choose the levels for the current u and zero the levels above them.
  auto project = [&](std::vector<double>& uu) {
    std::vector<std::vector<double>> g(T);
    for (uint32_t t = 0; t < T; ++t) {
      g[t].assign(depth[t] + 1, 0.0);
      for (uint32_t l = 1; l <= depth[t]; ++l) {
        const size_t base = begin[t] + ((size_t{1} << l) - 1) * K;
        const size_t cnt = (size_t{1} << l) * K;
        double s = 0.0;
        for (size_t i = 0; i < cnt; ++i) s += uu[base + i] * uu[base + i];
        g[t][l] = g[t][l - 1] + s;
      }
    }
    const std::vector<uint32_t> lv = choose_levels(g, cost, budget);
    for (uint32_t t = 0; t < T; ++t)
      for (uint32_t l = lv[t] + 1; l <= depth[t]; ++l)
        std::fill(uu.begin() + begin[t] + ((size_t{1} << l) - 1) * K,
                  uu.begin() + begin[t] + ((size_t{1} << (l + 1)) - 1) * K, 0.0);
    return lv;
  };

  // J and its gradient in the level basis and the bias, from one pass.
  auto eval = [&](const std::vector<double>& uu, const std::vector<double>& bb,
                 std::vector<double>& gu, std::vector<double>& gb) {
    std::vector<double> theta(f.total + K);
    for (uint32_t t = 0; t < T; ++t) {
      const size_t nodes = (size_t{2} << depth[t]) - 1;
      for (uint32_t k = 0; k < K; ++k) {
        std::vector<double> ub(nodes);
        for (size_t i = 0; i < nodes; ++i) ub[i] = uu[begin[t] + i * K + k];
        const std::vector<double> w = from_level_basis(ub, depth[t]);
        for (size_t j = 0; j < w.size(); ++j) theta[f.leaf_begin[t] + j * K + k] = w[j];
      }
    }
    for (uint32_t k = 0; k < K; ++k) theta[f.total + k] = bb[k];
    const PassSums sums = pass<Loss>(f, theta);
    // dJ/du_l(p) = the sum of dJ/dw_j over the leaves j with j mod 2^l = p:
    // start at the leaves, add each node's sum into its parent.
    std::fill(gu.begin(), gu.end(), 0.0);
    for (uint32_t t = 0; t < T; ++t) {
      const uint32_t d = depth[t];
      const size_t L = size_t{1} << d, lbase = begin[t] + (L - 1) * K;
      for (size_t j = 0; j < L * K; ++j) {
        double G, H;
        grad_curv(f, sums, theta, f.leaf_begin[t] + j, G, H);
        gu[lbase + j] = G;
      }
      for (uint32_t l = d; l >= 1; --l) {
        const size_t base = begin[t] + ((size_t{1} << l) - 1) * K;
        const size_t pbase = begin[t] + ((size_t{1} << (l - 1)) - 1) * K;
        const size_t half = size_t{1} << (l - 1);
        for (size_t p = 0; p < half; ++p)
          for (uint32_t k = 0; k < K; ++k)
            gu[pbase + p * K + k] += gu[base + p * K + k] + gu[base + (p + half) * K + k];
      }
    }
    for (uint32_t k = 0; k < K; ++k) gb[k] = sums.g[f.total + k] / sums.S;
    return objective(f, sums, theta);
  };

  std::vector<uint32_t> levels = project(u);
  std::vector<double> gu(u.size()), gb(K), trial_u(u.size()), trial_b(K), tgu(u.size()), tgb(K);
  double J = eval(u, bias, gu, gb);
  double eta = 1.0;
  for (int it = 0; it < f.stop.max_iter; ++it) {
    const double J_before = J;
    const std::vector<uint32_t> levels_before = levels;
    for (int bt = 0; bt <= 30; ++bt) {
      for (size_t j = 0; j < u.size(); ++j) trial_u[j] = u[j] - eta * gu[j];
      for (uint32_t k = 0; k < K; ++k) trial_b[k] = bias[k] - eta * gb[k];
      std::vector<uint32_t> trial_levels = project(trial_u);
      const double Jt = eval(trial_u, trial_b, tgu, tgb);
      if (Jt <= J_before) {
        u.swap(trial_u);  // swap, not move: the trial buffers stay sized for the next round
        bias.swap(trial_b);
        levels.swap(trial_levels);
        gu.swap(tgu);
        gb.swap(tgb);
        J = Jt;
        eta *= 2.0;
        break;
      }
      eta *= 0.5;
    }
    const double drop = J_before - J;
    if (levels == levels_before && drop <= f.stop.tol * std::max(1.0, std::fabs(J))) break;
  }
  return PruneResult{std::move(u), std::move(levels), std::move(bias)};
}

}  // namespace

PruneResult prune_to_budget(Task task, const FitData& d, std::vector<double> bias, const LevelCost& cost,
                            uint64_t budget) {
  return with_loss(task, [&](auto loss) { return iht<decltype(loss)>(d, std::move(bias), cost, budget); });
}

}  // namespace boxwood
