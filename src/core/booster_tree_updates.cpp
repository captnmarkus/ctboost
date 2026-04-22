#include "booster_internal.hpp"

#include <algorithm>

namespace ctboost::booster_detail {

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
  return static_cast<float>(-gradient_sum / (hessian_sum + lambda_l2));
}

void BuildSharedMulticlassTargets(const std::vector<float>& gradients,
                                  const std::vector<float>& hessians,
                                  const std::vector<float>& weights,
                                  std::size_t num_rows,
                                  int prediction_dimension,
                                  std::vector<float>& structure_gradients,
                                  std::vector<float>& structure_hessians) {
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
  if (prediction_dimension == 1) {
    for (std::size_t row = 0; row < leaf_indices.size(); ++row) {
      const int leaf_index = leaf_indices[row];
      if (leaf_index >= 0) {
        predictions[row] += learning_rate * nodes[static_cast<std::size_t>(leaf_index)].leaf_weight;
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
    predictions[offset] += learning_rate * nodes[static_cast<std::size_t>(leaf_index)].leaf_weight;
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
    double lambda_l2) {
  std::vector<Tree> class_trees(static_cast<std::size_t>(prediction_dimension), structure_tree);
  const auto& structure_nodes = structure_tree.nodes();
  if (structure_nodes.empty() || row_indices.empty() || leaf_row_ranges.size() < structure_nodes.size()) {
    return class_trees;
  }

  for (std::size_t node_index = 0; node_index < structure_nodes.size(); ++node_index) {
    const Node& node = structure_nodes[node_index];
    if (!node.is_leaf) {
      continue;
    }
    const LeafRowRange& range = leaf_row_ranges[node_index];
    if (range.end <= range.begin) {
      for (Tree& tree : class_trees) {
        tree.SetLeafWeight(node_index, 0.0F);
      }
      continue;
    }

    std::vector<double> gradient_sums(static_cast<std::size_t>(prediction_dimension), 0.0);
    std::vector<double> hessian_sums(static_cast<std::size_t>(prediction_dimension), 0.0);
    for (std::size_t position = range.begin; position < range.end; ++position) {
      const std::size_t row = row_indices[position];
      const double sample_weight = static_cast<double>(weights[row]);
      const std::size_t offset = row * static_cast<std::size_t>(prediction_dimension);
      for (int class_index = 0; class_index < prediction_dimension; ++class_index) {
        const std::size_t target_index = offset + static_cast<std::size_t>(class_index);
        gradient_sums[static_cast<std::size_t>(class_index)] += sample_weight * gradients[target_index];
        hessian_sums[static_cast<std::size_t>(class_index)] += sample_weight * hessians[target_index];
      }
    }
    for (int class_index = 0; class_index < prediction_dimension; ++class_index) {
      class_trees[static_cast<std::size_t>(class_index)].SetLeafWeight(
          node_index,
          ComputeLeafWeightFromSums(gradient_sums[static_cast<std::size_t>(class_index)],
                                    hessian_sums[static_cast<std::size_t>(class_index)],
                                    lambda_l2));
    }
  }
  return class_trees;
}

}  // namespace ctboost::booster_detail
