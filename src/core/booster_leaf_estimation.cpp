#include "booster_fit_internal.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace ctboost::booster_detail {
namespace {

constexpr double kMinimumLeafDenominator = 1e-12;

struct LeafExtrema {
  double minimum{std::numeric_limits<double>::infinity()};
  double maximum{-std::numeric_limits<double>::infinity()};
  bool has_leaf{false};
};

LeafExtrema MergeExtrema(const LeafExtrema& left, const LeafExtrema& right) {
  if (!left.has_leaf) {
    return right;
  }
  if (!right.has_leaf) {
    return left;
  }
  return LeafExtrema{
      std::min(left.minimum, right.minimum),
      std::max(left.maximum, right.maximum),
      true,
  };
}

LeafExtrema SubtreeExtrema(const Tree& tree, int node_index) {
  const auto& nodes = tree.nodes();
  if (node_index < 0 || static_cast<std::size_t>(node_index) >= nodes.size()) {
    return LeafExtrema{};
  }
  const Node& node = nodes[static_cast<std::size_t>(node_index)];
  if (node.is_leaf) {
    const double value = static_cast<double>(node.leaf_weight);
    return LeafExtrema{value, value, true};
  }
  return MergeExtrema(SubtreeExtrema(tree, node.left_child),
                      SubtreeExtrema(tree, node.right_child));
}

void ClampSubtree(Tree& tree, int node_index, double lower_bound, double upper_bound) {
  const auto& nodes = tree.nodes();
  if (node_index < 0 || static_cast<std::size_t>(node_index) >= nodes.size()) {
    return;
  }
  const Node& node = nodes[static_cast<std::size_t>(node_index)];
  if (node.is_leaf) {
    tree.SetLeafWeight(
        static_cast<std::size_t>(node_index),
        static_cast<float>(std::clamp(static_cast<double>(node.leaf_weight),
                                      lower_bound,
                                      upper_bound)));
    return;
  }
  ClampSubtree(tree, node.left_child, lower_bound, upper_bound);
  ClampSubtree(tree, node.right_child, lower_bound, upper_bound);
}

LeafExtrema ProjectMonotoneSubtree(Tree& tree,
                                   int node_index,
                                   const std::vector<int>& constraints) {
  const auto& nodes = tree.nodes();
  if (node_index < 0 || static_cast<std::size_t>(node_index) >= nodes.size()) {
    return LeafExtrema{};
  }
  const Node& node = nodes[static_cast<std::size_t>(node_index)];
  if (node.is_leaf) {
    const double value = static_cast<double>(node.leaf_weight);
    return LeafExtrema{value, value, true};
  }

  LeafExtrema left = ProjectMonotoneSubtree(tree, node.left_child, constraints);
  LeafExtrema right = ProjectMonotoneSubtree(tree, node.right_child, constraints);
  const int monotone_sign =
      node.split_feature_id < 0 ||
              static_cast<std::size_t>(node.split_feature_id) >= constraints.size()
          ? 0
          : constraints[static_cast<std::size_t>(node.split_feature_id)];
  if (left.has_leaf && right.has_leaf && monotone_sign > 0 &&
      left.maximum > right.minimum) {
    const double midpoint = 0.5 * (left.maximum + right.minimum);
    ClampSubtree(tree,
                 node.left_child,
                 -std::numeric_limits<double>::infinity(),
                 midpoint);
    ClampSubtree(tree,
                 node.right_child,
                 midpoint,
                 std::numeric_limits<double>::infinity());
    left = SubtreeExtrema(tree, node.left_child);
    right = SubtreeExtrema(tree, node.right_child);
  } else if (left.has_leaf && right.has_leaf && monotone_sign < 0 &&
             left.minimum < right.maximum) {
    const double midpoint = 0.5 * (left.minimum + right.maximum);
    ClampSubtree(tree,
                 node.left_child,
                 midpoint,
                 std::numeric_limits<double>::infinity());
    ClampSubtree(tree,
                 node.right_child,
                 -std::numeric_limits<double>::infinity(),
                 midpoint);
    left = SubtreeExtrema(tree, node.left_child);
    right = SubtreeExtrema(tree, node.right_child);
  }
  return MergeExtrema(left, right);
}

void ProjectMonotoneLeafWeights(Tree& tree, const std::vector<int>* constraints) {
  if (constraints == nullptr || constraints->empty() || tree.nodes().empty()) {
    return;
  }
  (void)ProjectMonotoneSubtree(tree, 0, *constraints);
}

float RefinedLeafWeight(float current_leaf_weight,
                        double gradient_sum,
                        double hessian_sum,
                        double lambda_l2,
                        double max_leaf_weight) {
  const double current = static_cast<double>(current_leaf_weight);
  const double denominator = std::max(0.0, hessian_sum) + lambda_l2;
  if (!(denominator > kMinimumLeafDenominator)) {
    return current_leaf_weight;
  }
  const double regularized_gradient = gradient_sum + lambda_l2 * current;
  const double candidate = current - regularized_gradient / denominator;
  if (!std::isfinite(candidate)) {
    return current_leaf_weight;
  }
  const double bounded = max_leaf_weight > 0.0
                             ? std::clamp(candidate, -max_leaf_weight, max_leaf_weight)
                             : candidate;
  return static_cast<float>(bounded);
}

LeafStatistics AccumulateSingleOutputStatistics(
    const Tree& tree,
    const std::vector<std::size_t>& row_indices,
    const std::vector<LeafRowRange>& leaf_row_ranges,
    const std::vector<float>& gradients,
    const std::vector<float>& hessians,
    const std::vector<float>& weights) {
  const auto& nodes = tree.nodes();
  LeafStatistics statistics;
  statistics.gradient_sums.assign(nodes.size(), 0.0);
  statistics.hessian_sums.assign(nodes.size(), 0.0);
  for (std::size_t node_index = 0; node_index < nodes.size(); ++node_index) {
    if (!nodes[node_index].is_leaf || node_index >= leaf_row_ranges.size()) {
      continue;
    }
    const LeafRowRange& range = leaf_row_ranges[node_index];
    for (std::size_t position = range.begin; position < range.end; ++position) {
      const std::size_t row = row_indices[position];
      const double sample_weight = static_cast<double>(weights[row]);
      statistics.gradient_sums[node_index] += sample_weight * gradients[row];
      statistics.hessian_sums[node_index] += sample_weight * hessians[row];
    }
  }
  return statistics;
}

}  // namespace

void RefineSingleOutputTreeLeaves(const FitLoopContext& context,
                                  Tree& tree,
                                  const std::vector<std::size_t>& row_indices,
                                  const std::vector<LeafRowRange>& leaf_row_ranges,
                                  const std::vector<float>& iteration_weights,
                                  const std::vector<float>& gradient_predictions,
                                  DistributedCoordinator* distributed_coordinator) {
  std::vector<float> refined_predictions;
  for (int step = 1; step < context.leaf_estimation_iterations; ++step) {
    refined_predictions = gradient_predictions;
    UpdatePredictionsFromLeafRanges(tree,
                                    row_indices,
                                    leaf_row_ranges,
                                    1.0,
                                    1,
                                    0,
                                    refined_predictions);
    context.objective->compute_gradients(refined_predictions,
                                         *context.labels,
                                         context.workspace->gradients,
                                         context.workspace->hessians,
                                         context.num_classes,
                                         context.ranking);
    LeafStatistics statistics = AccumulateSingleOutputStatistics(
        tree,
        row_indices,
        leaf_row_ranges,
        context.workspace->gradients,
        context.workspace->hessians,
        iteration_weights);
    const std::string label = "leaf_estimation_" + std::to_string(step);
    statistics =
        AllReduceLeafStatistics(distributed_coordinator, label.c_str(), statistics);

    const auto& nodes = tree.nodes();
    for (std::size_t node_index = 0; node_index < nodes.size(); ++node_index) {
      if (!nodes[node_index].is_leaf) {
        continue;
      }
      tree.SetLeafWeight(
          node_index,
          RefinedLeafWeight(nodes[node_index].leaf_weight,
                            statistics.gradient_sums[node_index],
                            statistics.hessian_sums[node_index],
                            context.lambda_l2,
                            context.max_leaf_weight));
    }
    ProjectMonotoneLeafWeights(tree, context.monotone_constraints);
  }
}

}  // namespace ctboost::booster_detail
