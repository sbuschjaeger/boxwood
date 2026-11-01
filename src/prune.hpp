#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "boxwood/ensemble.hpp"
#include "fit.hpp"

namespace boxwood {

/// The level basis of one tree's leaf values for one output. `w` holds 2^d leaf
/// values; the result holds 2^(d+1) - 1 node values in heap order: level l
/// (0 = root) at index 2^l - 1 + p, p < 2^l. A leaf value is the sum of the
/// node values on its path: w_j = sum_l u_l(j mod 2^l).
std::vector<double> to_level_basis(std::span<const double> w);

/// The 2^k leaf values of the tree cut to its first k levels, from the level
/// basis `u`: leaf p is the mean of the leaves j with j mod 2^k = p.
std::vector<double> from_level_basis(std::span<const double> u, uint32_t k);

/// The cost of keeping k = 0..d_t levels of every tree, with cost[t][0] = 0
/// and the cost increasing in k: k itself for a split budget, the bytes of the
/// cut tree for a byte budget.
using LevelCost = std::vector<std::vector<uint64_t>>;

/// The kept levels for a budget. `g[t]` holds g_t(0..d_t) with g_t(0) = 0.
/// The upper concave hull of the points (cost[t][k], g_t(k)) splits into
/// segments of decreasing slope (gain per cost); the segments of all trees are
/// taken greedily by slope (descending, ties: lower tree index first) while
/// their costs fit into `B`, stopping at the first segment that does not fit.
std::vector<uint32_t> choose_levels(const std::vector<std::vector<double>>& g, const LevelCost& cost,
                                    uint64_t B);

/// The result of iterative hard thresholding, all in the units of the fit:
/// the level basis of every tree and output (zeroed above the kept levels),
/// the kept levels per tree and the bias. The u blocks of the trees are
/// concatenated; tree t occupies (2^(d_t+1) - 1) * K values, the node (l, p)
/// of output k at offset (2^l - 1 + p) * K + k inside its block.
struct PruneResult {
  std::vector<double> u;
  std::vector<uint32_t> levels;
  std::vector<double> bias;
};

/// Iterative hard thresholding to a budget (see choose_levels) on the training observations of `d`:
/// start at the anchor of `d` in the level basis, project to the budget, then
/// repeat u' = P(u - eta * J') with the bias stepping in the same move and a
/// backtracking line search on J
/// (eta starts at 1, halves while J increases, up to 30 times; after an
/// accepted step try 2*eta next time). Stops after `d.stop.max_iter`
/// iterations or when the kept levels stop changing and the relative drop of J
/// falls below `d.stop.tol`. `bias` holds the K bias values in fit units.
PruneResult prune_to_budget(Task task, const FitData& d, std::vector<double> bias, const LevelCost& cost,
                            uint64_t budget);

}  // namespace boxwood
