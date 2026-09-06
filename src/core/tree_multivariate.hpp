#pragma once

#include "ctboost/multivariate_statistics.hpp"
#include "ctboost/tree.hpp"

namespace ctboost::detail {

// One covariance decomposition per node, reused across all eligible features.
class JointFeatureStatistic {
 public:
  JointFeatureStatistic(const HistMatrix& hist, const TreeBuildOptions& options,
                        std::size_t row_begin, std::size_t row_end)
      : hist_(hist), options_(options), row_begin_(row_begin), row_end_(row_end),
        response_(ComputeMultivariateResponseStatistics(
            *options.multivariate_gradients, *options.statistic_weights,
            *options.statistic_row_indices, row_begin, row_end,
            options.multivariate_dimension)) {}

  LinearStatisticScore Evaluate(std::size_t feature) const {
    const std::size_t dimension = options_.multivariate_dimension;
    MultivariateBinStatistics bins;
    bins.response_dimension = dimension;
    bins.weight_sums.assign(hist_.num_bins(feature), 0.0);
    bins.gradient_sums.assign(bins.weight_sums.size() * dimension, 0.0);
    const FeatureBinView feature_bins = hist_.feature_bins(feature);
    for (std::size_t position = row_begin_; position < row_end_; ++position) {
      const std::size_t row = (*options_.statistic_row_indices)[position];
      const double weight = (*options_.statistic_weights)[row];
      if (weight <= 0.0) continue;
      const std::size_t bin = feature_bins[row];
      bins.weight_sums[bin] += weight;
      for (std::size_t k = 0; k < dimension; ++k) {
        bins.gradient_sums[bin * dimension + k] +=
            weight * (*options_.multivariate_gradients)[row * dimension + k];
      }
    }
    if (options_.feature_test == FeatureTest::Grouped && !hist_.is_categorical(feature)) {
      const std::size_t missing = !hist_.has_missing_values(feature)
          ? kNoMissingStatisticBin
          : (hist_.nan_mode_for_feature(feature) == NanMode::Min
                 ? 0U : bins.weight_sums.size() - 1U);
      bins = GroupOrderedMultivariateBinStatistics(bins, options_.feature_test_bins, missing);
    }
    return EvaluateMultivariateStatisticFromBins(bins, response_);
  }

 private:
  const HistMatrix& hist_;
  const TreeBuildOptions& options_;
  std::size_t row_begin_;
  std::size_t row_end_;
  MultivariateResponseStatistics response_;
};

}  // namespace ctboost::detail
