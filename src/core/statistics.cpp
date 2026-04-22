#include "ctboost/statistics.hpp"

#include "statistics_internal.hpp"

#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace ctboost {

LinearStatistic::LinearStatistic(double epsilon) : epsilon_(epsilon) {
  if (epsilon_ <= 0.0) {
    throw std::invalid_argument("epsilon must be positive");
  }
}

BinStatistics LinearStatistic::ComputeBinStatistics(const std::vector<float>& gradients,
                                                    const std::vector<float>& hessians,
                                                    const std::vector<float>& weights,
                                                    const std::vector<std::uint16_t>& bins,
                                                    std::size_t num_bins) const {
  detail::ValidateStatisticSizes(gradients, hessians, weights, bins, num_bins);

  BinStatistics stats;
  stats.gradient_sums.assign(num_bins, 0.0);
  stats.hessian_sums.assign(num_bins, 0.0);
  stats.weight_sums.assign(num_bins, 0.0);

  for (std::size_t i = 0; i < gradients.size(); ++i) {
    const std::size_t bin = static_cast<std::size_t>(bins[i]);
    if (bin >= num_bins) {
      throw std::invalid_argument("bin index is out of range");
    }
    const double sample_weight = static_cast<double>(weights[i]);
    stats.gradient_sums[bin] += sample_weight * gradients[i];
    stats.hessian_sums[bin] += sample_weight * hessians[i];
    stats.weight_sums[bin] += sample_weight;
  }

  return stats;
}

LinearStatisticScore LinearStatistic::EvaluateScoreFromBinStatistics(const BinStatistics& stats,
                                                                     double total_gradient,
                                                                     double sample_weight_sum,
                                                                     double gradient_variance) const {
  LinearStatisticScore result;

  const std::size_t num_bins = stats.gradient_sums.size();
  if (stats.hessian_sums.size() != num_bins || stats.weight_sums.size() != num_bins) {
    throw std::invalid_argument("bin statistics vectors must have the same size");
  }

  std::size_t active_bins = 0;
  for (std::size_t bin = 0; bin < num_bins; ++bin) {
    if (stats.weight_sums[bin] > 0.0) {
      ++active_bins;
    }
  }

  if (sample_weight_sum <= 1.0 || active_bins <= 1) {
    result.degrees_of_freedom = 0;
    result.p_value = 1.0;
    return result;
  }

  result.degrees_of_freedom = active_bins - 1;
  if (gradient_variance <= std::numeric_limits<double>::epsilon()) {
    result.p_value = 1.0;
    return result;
  }

  const double node_count = sample_weight_sum;
  const double gradient_mean = total_gradient / node_count;
  const double diagonal_scale = (node_count / (node_count - 1.0)) * gradient_variance;
  const double outer_scale = gradient_variance / (node_count - 1.0);

  double diff_quadratic = 0.0;
  double weighted_projection = 0.0;
  double diagonal_projection = 0.0;
  std::size_t processed_bins = 0;
  for (std::size_t bin = 0; bin < num_bins && processed_bins < result.degrees_of_freedom; ++bin) {
    const double bin_weight = stats.weight_sums[bin];
    if (bin_weight <= 0.0) {
      continue;
    }

    const double diff = stats.gradient_sums[bin] - bin_weight * gradient_mean;
    const double diagonal = diagonal_scale * bin_weight + epsilon_;
    diff_quadratic += (diff * diff) / diagonal;
    weighted_projection += (bin_weight * diff) / diagonal;
    diagonal_projection += (bin_weight * bin_weight) / diagonal;
    ++processed_bins;
  }

  const double denominator = 1.0 - outer_scale * diagonal_projection;
  if (denominator <= epsilon_) {
    const LinearStatisticResult fallback =
        EvaluateFromBinStatistics(stats, total_gradient, sample_weight_sum, gradient_variance);
    result.degrees_of_freedom = fallback.degrees_of_freedom;
    result.chi_square = fallback.chi_square;
    result.p_value = fallback.p_value;
    return result;
  }

  result.chi_square =
      diff_quadratic + outer_scale * weighted_projection * weighted_projection / denominator;
  result.p_value = ChiSquareSurvival(result.chi_square, result.degrees_of_freedom);
  return result;
}

LinearStatisticResult LinearStatistic::EvaluateFromBinStatistics(const BinStatistics& stats,
                                                                double total_gradient,
                                                                double sample_weight_sum,
                                                                double gradient_variance) const {
  LinearStatisticResult result;

  const std::size_t num_bins = stats.gradient_sums.size();
  if (stats.hessian_sums.size() != num_bins || stats.weight_sums.size() != num_bins) {
    throw std::invalid_argument("bin statistics vectors must have the same size");
  }

  std::vector<std::size_t> active_bins;
  active_bins.reserve(num_bins);
  for (std::size_t bin = 0; bin < num_bins; ++bin) {
    if (stats.weight_sums[bin] > 0.0) {
      active_bins.push_back(bin);
    }
  }

  result.num_bins = active_bins.size();
  if (sample_weight_sum <= 1.0 || active_bins.size() <= 1) {
    result.degrees_of_freedom = 0;
    result.p_value = 1.0;
    return result;
  }

  result.statistic.reserve(active_bins.size());
  result.expectation.reserve(active_bins.size());

  const double node_count = sample_weight_sum;
  const double gradient_mean = total_gradient / node_count;

  for (const std::size_t active_bin : active_bins) {
    result.statistic.push_back(stats.gradient_sums[active_bin]);
    result.expectation.push_back(stats.weight_sums[active_bin] * gradient_mean);
  }

  result.degrees_of_freedom = active_bins.size() - 1;
  if (gradient_variance <= std::numeric_limits<double>::epsilon()) {
    result.covariance.assign(result.degrees_of_freedom * result.degrees_of_freedom, 0.0);
    result.p_value = 1.0;
    return result;
  }

  std::vector<double> reduced_diff(result.degrees_of_freedom, 0.0);
  result.covariance.assign(result.degrees_of_freedom * result.degrees_of_freedom, 0.0);

  const double diagonal_scale = (node_count / (node_count - 1.0)) * gradient_variance;
  const double outer_scale = gradient_variance / (node_count - 1.0);

  for (std::size_t i = 0; i < result.degrees_of_freedom; ++i) {
    const double count_i = stats.weight_sums[active_bins[i]];
    reduced_diff[i] = result.statistic[i] - result.expectation[i];
    for (std::size_t j = 0; j < result.degrees_of_freedom; ++j) {
      const double count_j = stats.weight_sums[active_bins[j]];
      double covariance = -outer_scale * count_i * count_j;
      if (i == j) {
        covariance += diagonal_scale * count_i;
        covariance += epsilon_;
      }
      result.covariance[i * result.degrees_of_freedom + j] = covariance;
    }
  }

  const std::vector<double> solved =
      detail::SolveStatisticLinearSystem(result.covariance, reduced_diff);
  result.chi_square = std::inner_product(
      reduced_diff.begin(), reduced_diff.end(), solved.begin(), 0.0);
  result.p_value = ChiSquareSurvival(result.chi_square, result.degrees_of_freedom);
  return result;
}

LinearStatisticResult LinearStatistic::Evaluate(const std::vector<float>& gradients,
                                                const std::vector<float>& hessians,
                                                const std::vector<float>& weights,
                                                const std::vector<std::uint16_t>& bins,
                                                std::size_t num_bins) const {
  const BinStatistics stats = ComputeBinStatistics(gradients, hessians, weights, bins, num_bins);
  double total_gradient = 0.0;
  double total_weight = 0.0;
  for (std::size_t i = 0; i < gradients.size(); ++i) {
    total_gradient += static_cast<double>(weights[i]) * gradients[i];
    total_weight += static_cast<double>(weights[i]);
  }
  const double gradient_mean = total_weight <= 0.0 ? 0.0 : total_gradient / total_weight;

  double centered_sum_of_squares = 0.0;
  for (std::size_t i = 0; i < gradients.size(); ++i) {
    const double centered = static_cast<double>(gradients[i]) - gradient_mean;
    centered_sum_of_squares += static_cast<double>(weights[i]) * centered * centered;
  }
  const double gradient_variance =
      total_weight <= 0.0 ? 0.0 : centered_sum_of_squares / total_weight;

  return EvaluateFromBinStatistics(stats, total_gradient, total_weight, gradient_variance);
}

double LinearStatistic::epsilon() const noexcept { return epsilon_; }

double ChiSquareSurvival(double statistic, std::size_t degrees_of_freedom) {
  if (degrees_of_freedom == 0 || statistic <= 0.0) {
    return 1.0;
  }
  return detail::RegularizedGammaQ(
      0.5 * static_cast<double>(degrees_of_freedom), 0.5 * statistic);
}

}  // namespace ctboost
