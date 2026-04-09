#include "ctboost/statistics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace ctboost {
namespace {

constexpr int kMaxGammaIterations = 200;
constexpr double kGammaTolerance = 3e-14;
constexpr double kGammaTiny = 1e-300;

void ValidateSizes(const std::vector<float>& gradients,
                   const std::vector<float>& hessians,
                   const std::vector<std::uint16_t>& bins,
                   std::size_t num_bins) {
  if (gradients.size() != hessians.size() || gradients.size() != bins.size()) {
    throw std::invalid_argument("gradients, hessians, and bins must have the same size");
  }
  if (num_bins == 0) {
    throw std::invalid_argument("num_bins must be greater than zero");
  }
}

double RegularizedGammaPSeries(double a, double x) {
  double ap = a;
  double del = 1.0 / a;
  double sum = del;

  for (int n = 1; n <= kMaxGammaIterations; ++n) {
    ap += 1.0;
    del *= x / ap;
    sum += del;
    if (std::fabs(del) <= std::fabs(sum) * kGammaTolerance) {
      break;
    }
  }

  return sum * std::exp(-x + a * std::log(x) - std::lgamma(a));
}

double RegularizedGammaQContinuedFraction(double a, double x) {
  double b = x + 1.0 - a;
  double c = 1.0 / kGammaTiny;
  double d = 1.0 / std::max(b, kGammaTiny);
  double h = d;

  for (int i = 1; i <= kMaxGammaIterations; ++i) {
    const double i_as_double = static_cast<double>(i);
    const double an = -i_as_double * (i_as_double - a);
    b += 2.0;
    d = an * d + b;
    if (std::fabs(d) < kGammaTiny) {
      d = kGammaTiny;
    }
    c = b + an / c;
    if (std::fabs(c) < kGammaTiny) {
      c = kGammaTiny;
    }
    d = 1.0 / d;
    const double del = d * c;
    h *= del;
    if (std::fabs(del - 1.0) <= kGammaTolerance) {
      break;
    }
  }

  return std::exp(-x + a * std::log(x) - std::lgamma(a)) * h;
}

double RegularizedGammaQ(double a, double x) {
  if (a <= 0.0) {
    throw std::invalid_argument("gamma shape parameter must be positive");
  }
  if (x <= 0.0) {
    return 1.0;
  }
  if (x < a + 1.0) {
    return 1.0 - RegularizedGammaPSeries(a, x);
  }
  return RegularizedGammaQContinuedFraction(a, x);
}

std::vector<double> SolveLinearSystem(std::vector<double> matrix,
                                      std::vector<double> rhs) {
  const std::size_t n = rhs.size();
  if (matrix.size() != n * n) {
    throw std::invalid_argument("matrix dimensions do not match the RHS");
  }

  for (std::size_t pivot = 0; pivot < n; ++pivot) {
    std::size_t best_row = pivot;
    double best_value = std::fabs(matrix[pivot * n + pivot]);
    for (std::size_t row = pivot + 1; row < n; ++row) {
      const double candidate = std::fabs(matrix[row * n + pivot]);
      if (candidate > best_value) {
        best_value = candidate;
        best_row = row;
      }
    }

    if (best_value < kGammaTiny) {
      throw std::runtime_error("covariance matrix is singular");
    }

    if (best_row != pivot) {
      for (std::size_t col = 0; col < n; ++col) {
        std::swap(matrix[pivot * n + col], matrix[best_row * n + col]);
      }
      std::swap(rhs[pivot], rhs[best_row]);
    }

    const double diagonal = matrix[pivot * n + pivot];
    for (std::size_t col = pivot; col < n; ++col) {
      matrix[pivot * n + col] /= diagonal;
    }
    rhs[pivot] /= diagonal;

    for (std::size_t row = 0; row < n; ++row) {
      if (row == pivot) {
        continue;
      }
      const double factor = matrix[row * n + pivot];
      if (factor == 0.0) {
        continue;
      }
      for (std::size_t col = pivot; col < n; ++col) {
        matrix[row * n + col] -= factor * matrix[pivot * n + col];
      }
      rhs[row] -= factor * rhs[pivot];
    }
  }

  return rhs;
}

}  // namespace

LinearStatistic::LinearStatistic(double epsilon) : epsilon_(epsilon) {
  if (epsilon_ <= 0.0) {
    throw std::invalid_argument("epsilon must be positive");
  }
}

BinStatistics LinearStatistic::ComputeBinStatistics(const std::vector<float>& gradients,
                                                    const std::vector<float>& hessians,
                                                    const std::vector<std::uint16_t>& bins,
                                                    std::size_t num_bins) const {
  ValidateSizes(gradients, hessians, bins, num_bins);

  BinStatistics stats;
  stats.gradient_sums.assign(num_bins, 0.0);
  stats.hessian_sums.assign(num_bins, 0.0);
  stats.counts.assign(num_bins, 0);

  for (std::size_t i = 0; i < gradients.size(); ++i) {
    const std::size_t bin = static_cast<std::size_t>(bins[i]);
    if (bin >= num_bins) {
      throw std::invalid_argument("bin index is out of range");
    }
    stats.gradient_sums[bin] += gradients[i];
    stats.hessian_sums[bin] += hessians[i];
    stats.counts[bin] += 1;
  }

  return stats;
}

LinearStatisticResult LinearStatistic::Evaluate(const std::vector<float>& gradients,
                                                const std::vector<float>& hessians,
                                                const std::vector<std::uint16_t>& bins,
                                                std::size_t num_bins) const {
  const BinStatistics stats = ComputeBinStatistics(gradients, hessians, bins, num_bins);

  LinearStatisticResult result;

  std::vector<std::size_t> active_bins;
  active_bins.reserve(num_bins);
  for (std::size_t bin = 0; bin < num_bins; ++bin) {
    if (stats.counts[bin] > 0) {
      active_bins.push_back(bin);
    }
  }

  result.num_bins = active_bins.size();
  if (gradients.size() <= 1 || active_bins.size() <= 1) {
    result.degrees_of_freedom = 0;
    result.p_value = 1.0;
    return result;
  }

  result.statistic.reserve(active_bins.size());
  result.expectation.reserve(active_bins.size());

  const double total_gradient = std::accumulate(
      gradients.begin(), gradients.end(), 0.0,
      [](double acc, float value) { return acc + static_cast<double>(value); });
  const double node_count = static_cast<double>(gradients.size());
  const double gradient_mean = total_gradient / node_count;

  double centered_sum_of_squares = 0.0;
  for (const float gradient : gradients) {
    const double centered = static_cast<double>(gradient) - gradient_mean;
    centered_sum_of_squares += centered * centered;
  }
  const double gradient_variance = centered_sum_of_squares / node_count;

  for (const std::size_t active_bin : active_bins) {
    result.statistic.push_back(stats.gradient_sums[active_bin]);
    result.expectation.push_back(
        static_cast<double>(stats.counts[active_bin]) * gradient_mean);
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
    const double count_i = static_cast<double>(stats.counts[active_bins[i]]);
    reduced_diff[i] = result.statistic[i] - result.expectation[i];
    for (std::size_t j = 0; j < result.degrees_of_freedom; ++j) {
      const double count_j = static_cast<double>(stats.counts[active_bins[j]]);
      double covariance = -outer_scale * count_i * count_j;
      if (i == j) {
        covariance += diagonal_scale * count_i;
        covariance += epsilon_;
      }
      result.covariance[i * result.degrees_of_freedom + j] = covariance;
    }
  }

  const std::vector<double> solved = SolveLinearSystem(result.covariance, reduced_diff);
  result.chi_square = std::inner_product(
      reduced_diff.begin(), reduced_diff.end(), solved.begin(), 0.0);
  result.p_value = ChiSquareSurvival(result.chi_square, result.degrees_of_freedom);
  return result;
}

double LinearStatistic::epsilon() const noexcept { return epsilon_; }

double ChiSquareSurvival(double statistic, std::size_t degrees_of_freedom) {
  if (degrees_of_freedom == 0 || statistic <= 0.0) {
    return 1.0;
  }
  return RegularizedGammaQ(0.5 * static_cast<double>(degrees_of_freedom), 0.5 * statistic);
}

}  // namespace ctboost
