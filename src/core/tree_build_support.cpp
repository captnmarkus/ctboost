#include "tree_internal.hpp"

#include <algorithm>
#include <stdexcept>

namespace ctboost::detail {

double ComputeLeafWeight(double gradient_sum, double hessian_sum, double lambda_l2) {
  return -gradient_sum / (hessian_sum + lambda_l2);
}

double ComputeGain(double gradient_sum, double hessian_sum, double lambda_l2) {
  return (gradient_sum * gradient_sum) / (hessian_sum + lambda_l2);
}

double ComputeGradientVariance(double weighted_gradient_sum,
                               double weighted_gradient_square_sum,
                               double sample_weight_sum) {
  if (sample_weight_sum <= 0.0) {
    return 0.0;
  }

  const double mean_gradient = weighted_gradient_sum / sample_weight_sum;
  const double second_moment = weighted_gradient_square_sum / sample_weight_sum;
  return std::max(0.0, second_moment - mean_gradient * mean_gradient);
}

const QuantizationSchema& RequireQuantizationSchema(const QuantizationSchemaPtr& schema) {
  if (schema == nullptr) {
    throw std::runtime_error("tree quantization schema is not initialized");
  }
  return *schema;
}

double ClampLeafWeight(double leaf_weight, double lower_bound, double upper_bound) {
  return std::clamp(leaf_weight, lower_bound, upper_bound);
}

NodeHistogramSet ComputeNodeHistogramSet(const HistMatrix& hist,
                                         const std::vector<float>& gradients,
                                         const std::vector<float>& hessians,
                                         const std::vector<float>& weights,
                                         const std::vector<std::size_t>& row_indices,
                                         std::size_t row_begin,
                                         std::size_t row_end,
                                         bool use_gpu,
                                         GpuHistogramWorkspace* gpu_workspace) {
  NodeHistogramSet stats;
  stats.by_feature.resize(hist.num_cols);

  if (use_gpu) {
    (void)gpu_workspace;
    throw std::invalid_argument(
        "GPU node histogram materialization is no longer supported in the CPU search path");
  }

  for (std::size_t index = row_begin; index < row_end; ++index) {
    const std::size_t row = row_indices[index];
    const double gradient = gradients[row];
    const double hessian = hessians[row];
    const double sample_weight = weights[row];
    ++stats.sample_count;
    stats.total_gradient += sample_weight * gradient;
    stats.total_hessian += sample_weight * hessian;
    stats.gradient_square_sum += sample_weight * gradient * gradient;
    stats.sample_weight_sum += sample_weight;
  }
  stats.gradient_variance = ComputeGradientVariance(
      stats.total_gradient, stats.gradient_square_sum, stats.sample_weight_sum);

  for (std::size_t feature = 0; feature < hist.num_cols; ++feature) {
    const std::size_t feature_bins_count = hist.num_bins(feature);
    BinStatistics feature_stats;
    feature_stats.gradient_sums.assign(feature_bins_count, 0.0);
    feature_stats.hessian_sums.assign(feature_bins_count, 0.0);
    feature_stats.weight_sums.assign(feature_bins_count, 0.0);

    const auto feature_bins = hist.feature_bins(feature);
    for (std::size_t index = row_begin; index < row_end; ++index) {
      const std::size_t row = row_indices[index];
      const std::size_t bin = feature_bins[row];
      const double sample_weight = static_cast<double>(weights[row]);
      feature_stats.gradient_sums[bin] += sample_weight * gradients[row];
      feature_stats.hessian_sums[bin] += sample_weight * hessians[row];
      feature_stats.weight_sums[bin] += sample_weight;
    }

    stats.by_feature[feature] = std::move(feature_stats);
  }
  return stats;
}

NodeHistogramSet SubtractNodeHistogramSet(const NodeHistogramSet& parent,
                                          const NodeHistogramSet& child) {
  if (parent.by_feature.size() != child.by_feature.size()) {
    throw std::invalid_argument("parent and child histogram sets must have the same feature count");
  }

  NodeHistogramSet derived;
  derived.by_feature.resize(parent.by_feature.size());
  derived.sample_count =
      parent.sample_count >= child.sample_count ? parent.sample_count - child.sample_count : 0U;
  derived.sample_weight_sum = parent.sample_weight_sum - child.sample_weight_sum;
  derived.total_gradient = parent.total_gradient - child.total_gradient;
  derived.total_hessian = parent.total_hessian - child.total_hessian;
  derived.gradient_square_sum = parent.gradient_square_sum - child.gradient_square_sum;

  for (std::size_t feature = 0; feature < parent.by_feature.size(); ++feature) {
    const BinStatistics& parent_stats = parent.by_feature[feature];
    const BinStatistics& child_stats = child.by_feature[feature];
    if (parent_stats.gradient_sums.size() != child_stats.gradient_sums.size()) {
      throw std::invalid_argument(
          "parent and child histogram sets must have matching bin counts");
    }

    BinStatistics feature_stats;
    feature_stats.gradient_sums.resize(parent_stats.gradient_sums.size());
    feature_stats.hessian_sums.resize(parent_stats.hessian_sums.size());
    feature_stats.weight_sums.resize(parent_stats.weight_sums.size());
    for (std::size_t bin = 0; bin < parent_stats.gradient_sums.size(); ++bin) {
      feature_stats.gradient_sums[bin] =
          parent_stats.gradient_sums[bin] - child_stats.gradient_sums[bin];
      feature_stats.hessian_sums[bin] = std::max(
          0.0, parent_stats.hessian_sums[bin] - child_stats.hessian_sums[bin]);
      feature_stats.weight_sums[bin] =
          std::max(0.0, parent_stats.weight_sums[bin] - child_stats.weight_sums[bin]);
    }
    derived.by_feature[feature] = std::move(feature_stats);
  }

  derived.sample_weight_sum = std::max(0.0, derived.sample_weight_sum);
  derived.total_hessian = std::max(0.0, derived.total_hessian);
  derived.gradient_square_sum = std::max(0.0, derived.gradient_square_sum);
  derived.gradient_variance = ComputeGradientVariance(
      derived.total_gradient, derived.gradient_square_sum, derived.sample_weight_sum);
  return derived;
}

}  // namespace ctboost::detail
