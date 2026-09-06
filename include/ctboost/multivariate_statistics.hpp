#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "ctboost/statistics.hpp"

namespace ctboost {

inline constexpr std::size_t kMaxMultivariateResponseDimension = 32;
inline constexpr double kMultivariateRelativeRankTolerance = 1e-10;

// Responses are the complete K-dimensional gradient vectors, not a selected
// coordinate. Matrices are row-major K x K. covariance is the weighted
// population covariance; whitening columns for discarded eigenvalues are zero.
// The rank threshold is relative to the largest covariance eigenvalue, so the
// softmax all-ones null direction is removed without choosing a reference class.
struct MultivariateResponseStatistics {
  std::size_t response_dimension{0};
  double weight_sum{0.0};
  std::vector<double> mean;
  std::vector<double> covariance;
  std::vector<double> covariance_pseudoinverse;
  std::vector<double> whitening;
  std::size_t covariance_rank{0};
  bool frequency_weights{true};
};

// gradient_sums[b*K + k] = sum_{rows in b} weight[row] * gradient[row, k].
// Empty bins have zero weight and zero gradient sums. Missing values are an
// ordinary, separate bin, matching CTBoost's existing missing-bin policy.
struct MultivariateBinStatistics {
  std::size_t response_dimension{0};
  std::vector<double> gradient_sums;
  std::vector<double> weight_sums;
};

MultivariateResponseStatistics ComputeMultivariateResponseStatistics(
    const std::vector<float>& gradients,
    const std::vector<float>& weights,
    const std::vector<std::size_t>& row_indices,
    std::size_t row_begin,
    std::size_t row_end,
    std::size_t response_dimension);

MultivariateResponseStatistics ComputeMultivariateResponseStatistics(
    const std::vector<float>& gradients,
    const std::vector<float>& weights,
    std::size_t response_dimension);

MultivariateBinStatistics ComputeMultivariateBinStatistics(
    const std::vector<float>& gradients,
    const std::vector<float>& weights,
    const std::vector<std::uint16_t>& bins,
    std::size_t response_dimension,
    std::size_t num_bins);

MultivariateBinStatistics GroupOrderedMultivariateBinStatistics(
    const MultivariateBinStatistics& stats,
    std::size_t requested_groups,
    std::size_t missing_bin = kNoMissingStatisticBin);

// With total mass W, bin mass w_b, bin-centered gradient sum S_b and response
// covariance C, the statistic is
//   Q = (W-1)/W * sum_b(S_b' C^+ S_b / w_b),
// with df = (number of occupied bins - 1) * rank(C).
// This is the quadratic statistic from the conditional permutation covariance
// for one-hot feature bins and multivariate responses. Integer weights denote
// literal frequency-expanded observations. The returned chi-square tail is an
// asymptotic approximation, not an exact finite-sample permutation p-value.
// Fractional weights use the same frequency-formula continuation, have no
// literal permutation interpretation, and carry no general type-I guarantee;
// frequency_weights reports that distinction. Weight rescaling changes Q.
LinearStatisticScore EvaluateMultivariateStatisticFromBins(
    const MultivariateBinStatistics& bins,
    const MultivariateResponseStatistics& response);

}  // namespace ctboost
