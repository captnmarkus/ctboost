#include "tree_internal.hpp"

#include <algorithm>
#include <stdexcept>

namespace ctboost::detail {

SplitChoice SelectBestMulticlassSplit(const HistMatrix& hist,
                                     const NodeHistogramSet& node_stats,
                                     int feature_id,
                                     const TreeBuildOptions& options,
                                     std::size_t row_begin,
                                     std::size_t row_end) {
  SplitChoice best;
  const auto feature = static_cast<std::size_t>(feature_id);
  const auto& stats = node_stats.by_feature[feature];
  const std::size_t bins = stats.gradient_sums.size();
  const std::size_t dimension = options.split_dimension;
  const bool categorical = hist.is_categorical(feature);
  if (bins <= 1U) {
    return best;
  }
  if (categorical && bins > kMaxCategoricalRouteBins) {
    throw std::invalid_argument("categorical split routing supports at most 256 bins");
  }

  // Aggregate only the feature already accepted by the conditional test.
  // Feature-test response vectors and their weight contract remain independent.
  std::vector<double> gradients(bins * dimension, 0.0);
  std::vector<double> hessians(bins * dimension, 0.0);
  std::vector<double> total_g(dimension, 0.0), total_h(dimension, 0.0);
  const auto feature_bins = hist.feature_bins(feature);
  for (std::size_t index = row_begin; index < row_end; ++index) {
    const std::size_t row = (*options.statistic_row_indices)[index];
    const double weight = (*options.statistic_weights)[row];
    const std::size_t offset = feature_bins[row] * dimension;
    for (std::size_t k = 0; k < dimension; ++k) {
      const double g = (*options.split_gradients)[row * dimension + k] * weight;
      const double h = (*options.split_hessians)[row * dimension + k] * weight;
      gradients[offset + k] += g;
      hessians[offset + k] += h;
      total_g[k] += g;
      total_h[k] += h;
    }
  }
  std::vector<std::size_t> order;
  order.reserve(bins);
  for (std::size_t bin = 0; bin < bins; ++bin) {
    if (!categorical || stats.weight_sums[bin] > 0.0) {
      order.push_back(bin);
    }
  }
  if (categorical) {
    // Preserve the scalar structure-coordinate category ordering. Joint scoring
    // searches these prefixes, not every possible category partition.
    const auto scalar_weight = [&](std::size_t bin) {
      const double denominator = stats.hessian_sums[bin] + options.lambda_l2;
      return denominator > 0.0 ? stats.gradient_sums[bin] / denominator : 0.0;
    };
    std::sort(order.begin(), order.end(), [&](std::size_t lhs, std::size_t rhs) {
      const double left = scalar_weight(lhs), right = scalar_weight(rhs);
      return left == right ? lhs < rhs : left < right;
    });
  }
  double parent_gain = 0.0;
  for (std::size_t k = 0; k < dimension; ++k) {
    parent_gain += ComputeGain(total_g[k], total_h[k], options.lambda_l2);
  }
  std::vector<double> left_g(dimension, 0.0), left_h(dimension, 0.0);
  double left_gradient = 0.0, left_hessian = 0.0, left_count = 0.0;
  for (std::size_t cut = 0; cut + 1U < order.size(); ++cut) {
    const std::size_t bin = order[cut];
    left_gradient += stats.gradient_sums[bin];
    left_hessian += stats.hessian_sums[bin];
    left_count += stats.weight_sums[bin];
    for (std::size_t k = 0; k < dimension; ++k) {
      left_g[k] += gradients[bin * dimension + k];
      left_h[k] += hessians[bin * dimension + k];
    }
    const double right_count = node_stats.sample_weight_sum - left_count;
    const double right_gradient = node_stats.total_gradient - left_gradient;
    const double right_hessian = node_stats.total_hessian - left_hessian;
    // Keep the existing structure-coordinate child admissibility rules.
    if (left_count <= 0.0 || right_count <= 0.0 ||
        left_count < options.min_data_in_leaf || right_count < options.min_data_in_leaf ||
        left_hessian < options.min_child_weight || right_hessian < options.min_child_weight) {
      continue;
    }
    double gain = -parent_gain;
    for (std::size_t k = 0; k < dimension; ++k) {
      gain += ComputeGain(left_g[k], left_h[k], options.lambda_l2) +
              ComputeGain(total_g[k] - left_g[k], total_h[k] - left_h[k], options.lambda_l2);
    }
    if (gain <= options.min_split_gain || (best.valid && gain <= best.gain)) {
      continue;
    }
    best.valid = true;
    best.is_categorical = categorical;
    best.split_bin = static_cast<std::uint16_t>(bin);
    best.gain = gain;
    best.left_leaf_weight = ComputeLeafWeight(left_gradient, left_hessian, options.lambda_l2);
    best.right_leaf_weight = ComputeLeafWeight(right_gradient, right_hessian, options.lambda_l2);
    if (categorical) {
      best.left_categories.fill(0);
      for (std::size_t index = 0; index <= cut; ++index) {
        best.left_categories[order[index]] = 1;
      }
    }
  }
  return best;
}

}  // namespace ctboost::detail
