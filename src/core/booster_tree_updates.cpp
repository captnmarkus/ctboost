#include "booster_internal.hpp"

#include <algorithm>

namespace ctboost::booster_detail {
namespace {

constexpr double kMinimumLeafDenominator = 1e-12;

}  // namespace

void UpdatePredictionsFromLeafRanges(const Tree& tree,
                                     const std::vector<std::size_t>& row_indices,
                                     const std::vector<LeafRowRange>& leaf_row_ranges,
                                     double learning_rate,
                                     int prediction_dimension,
                                     int class_index,
                                     std::vector<float>& predictions) {
  const auto& nodes = tree.nodes();
  if (nodes.empty() || row_indices.empty() || leaf_row_ranges.size() < nodes.size()) {
    return;
  }
  if (tree.is_vector_leaf()) {
    for (std::size_t node_index = 0; node_index < nodes.size(); ++node_index) {
      const Node& node = nodes[node_index];
      if (!node.is_leaf) continue;
      const LeafRowRange& range = leaf_row_ranges[node_index];
      for (std::size_t position = range.begin; position < range.end; ++position) {
        const std::size_t offset = row_indices[position] * static_cast<std::size_t>(prediction_dimension);
        for (int output = 0; output < prediction_dimension; ++output) {
          // Preserve the scalar path's separately rounded product. Fusing the
          // multiply/add changes the gradients and can select another split.
          const float update = static_cast<float>(learning_rate) *
                               node.leaf_weights[static_cast<std::size_t>(output)];
          predictions[offset + static_cast<std::size_t>(output)] += update;
        }
      }
    }
    return;
  }
  if (prediction_dimension == 1) {
    for (std::size_t node_index = 0; node_index < nodes.size(); ++node_index) {
      const Node& node = nodes[node_index];
      if (!node.is_leaf) {
        continue;
      }
      const LeafRowRange& range = leaf_row_ranges[node_index];
      if (range.end <= range.begin) {
        continue;
      }
      const float update = static_cast<float>(learning_rate) * node.leaf_weight;
      for (std::size_t position = range.begin; position < range.end; ++position) {
        predictions[row_indices[position]] += update;
      }
    }
    return;
  }
  for (std::size_t node_index = 0; node_index < nodes.size(); ++node_index) {
    const Node& node = nodes[node_index];
    if (!node.is_leaf) {
      continue;
    }
    const LeafRowRange& range = leaf_row_ranges[node_index];
    if (range.end <= range.begin) {
      continue;
    }
    const float update = static_cast<float>(learning_rate) * node.leaf_weight;
    for (std::size_t position = range.begin; position < range.end; ++position) {
      const std::size_t row = row_indices[position];
      const std::size_t offset =
          row * static_cast<std::size_t>(prediction_dimension) + class_index;
      predictions[offset] += update;
    }
  }
}

float ComputeLeafWeightFromSums(double gradient_sum, double hessian_sum, double lambda_l2) {
  const double denominator = hessian_sum + lambda_l2;
  return denominator <= kMinimumLeafDenominator
             ? 0.0F
             : static_cast<float>(-gradient_sum / denominator);
}

void BuildSharedMulticlassTargets(const std::vector<float>& gradients,
                                  const std::vector<float>& hessians,
                                  const std::vector<float>& weights,
                                  std::size_t num_rows,
                                  int prediction_dimension,
                                  std::vector<float>& structure_gradients,
                                  std::vector<float>& structure_hessians,
                                  const DistributedCoordinator* distributed_coordinator) {
  structure_gradients.assign(num_rows, 0.0F);
  structure_hessians.assign(num_rows, 0.0F);
  if (prediction_dimension <= 0) {
    return;
  }

  std::vector<double> gradient_sums(static_cast<std::size_t>(prediction_dimension), 0.0);
  std::vector<double> gradient_square_sums(static_cast<std::size_t>(prediction_dimension), 0.0);
  std::vector<double> weight_sums(static_cast<std::size_t>(prediction_dimension), 0.0);
  for (std::size_t row = 0; row < num_rows; ++row) {
    const double sample_weight = static_cast<double>(weights[row]);
    const std::size_t offset = row * static_cast<std::size_t>(prediction_dimension);
    for (int class_index = 0; class_index < prediction_dimension; ++class_index) {
      const double gradient = gradients[offset + static_cast<std::size_t>(class_index)];
      gradient_sums[static_cast<std::size_t>(class_index)] += sample_weight * gradient;
      gradient_square_sums[static_cast<std::size_t>(class_index)] += sample_weight * gradient * gradient;
      weight_sums[static_cast<std::size_t>(class_index)] += sample_weight;
    }
  }

  if (distributed_coordinator != nullptr && distributed_coordinator->world_size > 1) {
    // Reuse the double-precision statistic reduction for the two moments. The
    // final gradient buffer entry carries their common total sample weight.
    LeafStatistics moments{gradient_sums, gradient_square_sums};
    moments.gradient_sums.push_back(weight_sums.front());
    moments.hessian_sums.push_back(0.0);
    moments = AllReduceLeafStatistics(distributed_coordinator, "multiclass_structure", moments);
    weight_sums.assign(static_cast<std::size_t>(prediction_dimension), moments.gradient_sums.back());
    moments.gradient_sums.pop_back();
    moments.hessian_sums.pop_back();
    gradient_sums = std::move(moments.gradient_sums);
    gradient_square_sums = std::move(moments.hessian_sums);
  }

  int structure_class = 0;
  double best_variance = -1.0;
  for (int class_index = 0; class_index < prediction_dimension; ++class_index) {
    const double total_weight = weight_sums[static_cast<std::size_t>(class_index)];
    if (total_weight <= 0.0) {
      continue;
    }
    const double mean_gradient = gradient_sums[static_cast<std::size_t>(class_index)] / total_weight;
    const double variance = std::max(0.0,
                                     gradient_square_sums[static_cast<std::size_t>(class_index)] /
                                             total_weight -
                                         mean_gradient * mean_gradient);
    if (variance > best_variance) {
      best_variance = variance;
      structure_class = class_index;
    }
  }

  for (std::size_t row = 0; row < num_rows; ++row) {
    const std::size_t offset = row * static_cast<std::size_t>(prediction_dimension);
    const std::size_t target_index = offset + static_cast<std::size_t>(structure_class);
    structure_gradients[row] = gradients[target_index];
    structure_hessians[row] = std::max(0.0F, hessians[target_index]);
  }
}

void UpdatePredictionsFromLeafIndices(const Tree& tree,
                                      const std::vector<int>& leaf_indices,
                                      double learning_rate,
                                      int prediction_dimension,
                                      int class_index,
                                      std::vector<float>& predictions) {
  const auto& nodes = tree.nodes();
  if (nodes.empty() || leaf_indices.empty()) {
    return;
  }
  if (tree.is_vector_leaf()) {
    for (std::size_t row = 0; row < leaf_indices.size(); ++row) {
      const int leaf_index = leaf_indices[row];
      if (leaf_index < 0) continue;
      const Node& node = nodes[static_cast<std::size_t>(leaf_index)];
      const std::size_t offset = row * static_cast<std::size_t>(prediction_dimension);
      for (int output = 0; output < prediction_dimension; ++output) {
        // Keep the same rounding boundary as scalar prediction, including the
        // external-memory, validation, and DART paths that reuse leaf indices.
        const float update = static_cast<float>(learning_rate) *
                             node.leaf_weights[static_cast<std::size_t>(output)];
        predictions[offset + static_cast<std::size_t>(output)] += update;
      }
    }
    return;
  }
  if (prediction_dimension == 1) {
    for (std::size_t row = 0; row < leaf_indices.size(); ++row) {
      const int leaf_index = leaf_indices[row];
      if (leaf_index >= 0) {
        const float update = static_cast<float>(learning_rate) *
                             nodes[static_cast<std::size_t>(leaf_index)].leaf_weight;
        predictions[row] += update;
      }
    }
    return;
  }
  for (std::size_t row = 0; row < leaf_indices.size(); ++row) {
    const int leaf_index = leaf_indices[row];
    if (leaf_index < 0) {
      continue;
    }
    const std::size_t offset = row * static_cast<std::size_t>(prediction_dimension) + class_index;
    const float update = static_cast<float>(learning_rate) *
                         nodes[static_cast<std::size_t>(leaf_index)].leaf_weight;
    predictions[offset] += update;
  }
}

std::vector<Tree> MaterializeMulticlassTreesFromStructure(
    const Tree& structure_tree,
    const std::vector<std::size_t>& row_indices,
    const std::vector<LeafRowRange>& leaf_row_ranges,
    const std::vector<float>& gradients,
    const std::vector<float>& hessians,
    const std::vector<float>& weights,
    int prediction_dimension,
    double lambda_l2,
    double max_leaf_weight,
    bool vector_leaves,
    const DistributedCoordinator* distributed_coordinator) {
  std::vector<Tree> class_trees;
  if (vector_leaves) {
    // Keep the conditional-inference topology once, with one weight per class.
    std::vector<Node> nodes = structure_tree.nodes();
    for (Node& node : nodes) {
      node.leaf_weights.assign(static_cast<std::size_t>(prediction_dimension), node.leaf_weight);
    }
    class_trees.emplace_back();
    class_trees.back().LoadState(std::move(nodes), structure_tree.shared_quantization_schema(),
                                structure_tree.feature_importances());
  } else {
    class_trees.assign(static_cast<std::size_t>(prediction_dimension), structure_tree);
  }
  const auto& structure_nodes = structure_tree.nodes();
  if (structure_nodes.empty()) {
    return class_trees;
  }

  const std::size_t width = static_cast<std::size_t>(prediction_dimension);
  LeafStatistics statistics;
  statistics.gradient_sums.assign(structure_nodes.size() * width, 0.0);
  statistics.hessian_sums.assign(structure_nodes.size() * width, 0.0);
  for (std::size_t node_index = 0; node_index < structure_nodes.size(); ++node_index) {
    if (!structure_nodes[node_index].is_leaf || node_index >= leaf_row_ranges.size()) {
      continue;
    }
    const LeafRowRange& range = leaf_row_ranges[node_index];
    for (std::size_t position = range.begin; position < range.end; ++position) {
      const std::size_t row = row_indices[position];
      const double sample_weight = static_cast<double>(weights[row]);
      const std::size_t offset = row * static_cast<std::size_t>(prediction_dimension);
      for (int class_index = 0; class_index < prediction_dimension; ++class_index) {
        const std::size_t target_index = offset + static_cast<std::size_t>(class_index);
        const std::size_t statistic_index = node_index * width + static_cast<std::size_t>(class_index);
        statistics.gradient_sums[statistic_index] += sample_weight * gradients[target_index];
        statistics.hessian_sums[statistic_index] += sample_weight * hessians[target_index];
      }
    }
  }
  if (distributed_coordinator != nullptr && distributed_coordinator->world_size > 1) {
    statistics = AllReduceLeafStatistics(distributed_coordinator, "multiclass_leaves", statistics);
  }
  for (std::size_t node_index = 0; node_index < structure_nodes.size(); ++node_index) {
    if (!structure_nodes[node_index].is_leaf) {
      continue;
    }
    std::vector<float> leaf_weights(static_cast<std::size_t>(prediction_dimension), 0.0F);
    for (int class_index = 0; class_index < prediction_dimension; ++class_index) {
      float leaf_weight = ComputeLeafWeightFromSums(
          statistics.gradient_sums[node_index * width + static_cast<std::size_t>(class_index)],
          statistics.hessian_sums[node_index * width + static_cast<std::size_t>(class_index)], lambda_l2);
      if (max_leaf_weight > 0.0) {
        leaf_weight = static_cast<float>(std::clamp(
            static_cast<double>(leaf_weight), -max_leaf_weight, max_leaf_weight));
      }
      if (vector_leaves) {
        leaf_weights[static_cast<std::size_t>(class_index)] = leaf_weight;
      } else {
        class_trees[static_cast<std::size_t>(class_index)].SetLeafWeight(node_index, leaf_weight);
      }
    }
    if (vector_leaves) {
      class_trees.front().SetLeafWeights(node_index, std::move(leaf_weights));
    }
  }
  return class_trees;
}

}  // namespace ctboost::booster_detail
