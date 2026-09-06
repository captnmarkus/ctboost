#include "ctboost/multivariate_statistics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace ctboost {
namespace {

void ValidateDimension(std::size_t dimension) {
  if (dimension == 0 || dimension > kMaxMultivariateResponseDimension) {
    throw std::invalid_argument("joint response dimension must be in [1, 32]");
  }
}

void ValidateResponseSize(const std::vector<float>& gradients,
                          const std::vector<float>& weights,
                          std::size_t dimension) {
  ValidateDimension(dimension);
  if (gradients.size() % dimension != 0 || gradients.size() / dimension != weights.size()) {
    throw std::invalid_argument("joint gradients must have shape (number of weights, response dimension)");
  }
}

void ValidateBins(const MultivariateBinStatistics& stats) {
  ValidateDimension(stats.response_dimension);
  if (stats.gradient_sums.size() % stats.response_dimension != 0 ||
      stats.gradient_sums.size() / stats.response_dimension != stats.weight_sums.size()) {
    throw std::invalid_argument("joint bin gradient sums must have shape (bins, response dimension)");
  }
  for (std::size_t bin = 0; bin < stats.weight_sums.size(); ++bin) {
    const double weight = stats.weight_sums[bin];
    if (!std::isfinite(weight) || weight < 0.0) {
      throw std::invalid_argument("joint bin weights must be finite and nonnegative");
    }
    for (std::size_t output = 0; output < stats.response_dimension; ++output) {
      const double gradient = stats.gradient_sums[bin * stats.response_dimension + output];
      if (!std::isfinite(gradient) || (weight == 0.0 && gradient != 0.0)) {
        throw std::invalid_argument("joint bin gradients must be finite and empty bins must have zero sums");
      }
    }
  }
}

// Cyclic symmetric Jacobi needs only a bounded K x K workspace. Scaling before
// decomposition avoids an absolute variance cutoff and keeps tiny responses
// equivalent to rescaled responses. No ridge is added to the singular softmax
// covariance: the test's degrees of freedom reflect its numerical rank.
void FactorResponseCovariance(MultivariateResponseStatistics& response) {
  const std::size_t dimension = response.response_dimension;
  const std::size_t matrix_size = dimension * dimension;
  response.covariance_pseudoinverse.assign(matrix_size, 0.0);
  response.whitening.assign(matrix_size, 0.0);
  double scale = 0.0;
  for (std::size_t i = 0; i < dimension; ++i) {
    scale = std::max(scale, response.covariance[i * dimension + i]);
  }
  if (scale == 0.0) {
    return;
  }
  if (!std::isfinite(scale) || scale < 0.0) {
    throw std::invalid_argument("joint response covariance must be finite and positive semidefinite");
  }
  std::vector<double> matrix = response.covariance;
  std::vector<double> eigenvectors(matrix_size, 0.0);
  for (double& value : matrix) {
    value /= scale;
  }
  for (std::size_t i = 0; i < dimension; ++i) {
    eigenvectors[i * dimension + i] = 1.0;
  }
  constexpr double tolerance = 64.0 * std::numeric_limits<double>::epsilon();
  bool converged = false;
  for (std::size_t sweep = 0; sweep < 64; ++sweep) {
    double largest_off_diagonal = 0.0;
    for (std::size_t p = 0; p < dimension; ++p) {
      for (std::size_t q = p + 1; q < dimension; ++q) {
        const double off_diagonal = matrix[p * dimension + q];
        largest_off_diagonal = std::max(largest_off_diagonal, std::abs(off_diagonal));
        if (std::abs(off_diagonal) <= tolerance) {
          continue;
        }
        const double tau = (matrix[q * dimension + q] - matrix[p * dimension + p]) /
                           (2.0 * off_diagonal);
        const double tangent = std::copysign(1.0 / (std::abs(tau) + std::hypot(1.0, tau)), tau);
        const double cosine = 1.0 / std::sqrt(1.0 + tangent * tangent);
        const double sine = tangent * cosine;
        matrix[p * dimension + p] -= tangent * off_diagonal;
        matrix[q * dimension + q] += tangent * off_diagonal;
        matrix[p * dimension + q] = matrix[q * dimension + p] = 0.0;
        for (std::size_t k = 0; k < dimension; ++k) {
          if (k != p && k != q) {
            const double at_p = matrix[k * dimension + p];
            const double at_q = matrix[k * dimension + q];
            matrix[k * dimension + p] = matrix[p * dimension + k] = cosine * at_p - sine * at_q;
            matrix[k * dimension + q] = matrix[q * dimension + k] = sine * at_p + cosine * at_q;
          }
          const double at_p = eigenvectors[k * dimension + p];
          const double at_q = eigenvectors[k * dimension + q];
          eigenvectors[k * dimension + p] = cosine * at_p - sine * at_q;
          eigenvectors[k * dimension + q] = sine * at_p + cosine * at_q;
        }
      }
    }
    if (largest_off_diagonal <= tolerance) {
      converged = true;
      break;
    }
  }
  if (!converged) {
    throw std::runtime_error("joint response covariance eigendecomposition did not converge");
  }
  double maximum_eigenvalue = 0.0;
  for (std::size_t i = 0; i < dimension; ++i) {
    maximum_eigenvalue = std::max(maximum_eigenvalue, matrix[i * dimension + i]);
  }
  const double cutoff = kMultivariateRelativeRankTolerance * maximum_eigenvalue;
  for (std::size_t column = 0; column < dimension; ++column) {
    const double eigenvalue = matrix[column * dimension + column];
    if (!std::isfinite(eigenvalue) || eigenvalue < -cutoff) {
      throw std::invalid_argument("joint response covariance is not positive semidefinite");
    }
    if (eigenvalue <= cutoff) {
      continue;
    }
    ++response.covariance_rank;
    const double inverse_root = 1.0 / std::sqrt(eigenvalue * scale);
    for (std::size_t row = 0; row < dimension; ++row) {
      response.whitening[row * dimension + column] =
          eigenvectors[row * dimension + column] * inverse_root;
    }
  }
  for (std::size_t row = 0; row < dimension; ++row) {
    for (std::size_t col = 0; col <= row; ++col) {
      double value = 0.0;
      for (std::size_t component = 0; component < dimension; ++component) {
        value += response.whitening[row * dimension + component] *
                 response.whitening[col * dimension + component];
      }
      response.covariance_pseudoinverse[row * dimension + col] = value;
      response.covariance_pseudoinverse[col * dimension + row] = value;
    }
  }
}

}  // namespace

MultivariateResponseStatistics ComputeMultivariateResponseStatistics(
    const std::vector<float>& gradients,
    const std::vector<float>& weights,
    const std::vector<std::size_t>& row_indices,
    std::size_t row_begin,
    std::size_t row_end,
    std::size_t response_dimension) {
  ValidateResponseSize(gradients, weights, response_dimension);
  if (row_begin > row_end || row_end > row_indices.size()) {
    throw std::invalid_argument("joint response row range is out of bounds");
  }
  MultivariateResponseStatistics response;
  response.response_dimension = response_dimension;
  response.mean.assign(response_dimension, 0.0);
  response.covariance.assign(response_dimension * response_dimension, 0.0);
  std::vector<double> origin(response_dimension, 0.0);
  bool has_origin = false;
  for (std::size_t position = row_begin; position < row_end; ++position) {
    const std::size_t row = row_indices[position];
    if (row >= weights.size()) {
      throw std::invalid_argument("joint response row index is out of bounds");
    }
    const double weight = weights[row];
    if (!std::isfinite(weight) || weight < 0.0) {
      throw std::invalid_argument("joint response weights must be finite and nonnegative");
    }
    response.frequency_weights = response.frequency_weights && std::floor(weight) == weight;
    response.weight_sum += weight;
    for (std::size_t output = 0; output < response_dimension; ++output) {
      const double gradient = gradients[row * response_dimension + output];
      if (!std::isfinite(gradient)) {
        throw std::invalid_argument("joint gradients must be finite");
      }
      if (weight > 0.0 && !has_origin) {
        origin[output] = gradient;
      }
      response.mean[output] += weight * (gradient - origin[output]);
    }
    has_origin = has_origin || weight > 0.0;
  }
  if (response.weight_sum > 0.0) {
    for (std::size_t output = 0; output < response_dimension; ++output) {
      response.mean[output] = origin[output] + response.mean[output] / response.weight_sum;
    }
    std::vector<double> centered(response_dimension);
    for (std::size_t position = row_begin; position < row_end; ++position) {
      const std::size_t row = row_indices[position];
      const double weight = weights[row];
      if (weight == 0.0) {
        continue;
      }
      for (std::size_t output = 0; output < response_dimension; ++output) {
        centered[output] = gradients[row * response_dimension + output] - response.mean[output];
      }
      for (std::size_t output = 0; output < response_dimension; ++output) {
        for (std::size_t other = 0; other <= output; ++other) {
          response.covariance[output * response_dimension + other] +=
              weight * centered[output] * centered[other];
        }
      }
    }
    for (std::size_t output = 0; output < response_dimension; ++output) {
      for (std::size_t other = 0; other <= output; ++other) {
        const double value = response.covariance[output * response_dimension + other] / response.weight_sum;
        response.covariance[output * response_dimension + other] = value;
        response.covariance[other * response_dimension + output] = value;
      }
    }
  }
  FactorResponseCovariance(response);
  return response;
}

MultivariateResponseStatistics ComputeMultivariateResponseStatistics(
    const std::vector<float>& gradients,
    const std::vector<float>& weights,
    std::size_t response_dimension) {
  std::vector<std::size_t> rows(weights.size());
  std::iota(rows.begin(), rows.end(), std::size_t{0});
  return ComputeMultivariateResponseStatistics(gradients, weights, rows, 0, rows.size(), response_dimension);
}

MultivariateBinStatistics ComputeMultivariateBinStatistics(
    const std::vector<float>& gradients,
    const std::vector<float>& weights,
    const std::vector<std::uint16_t>& bins,
    std::size_t response_dimension,
    std::size_t num_bins) {
  ValidateResponseSize(gradients, weights, response_dimension);
  if (bins.size() != weights.size() || num_bins == 0 || num_bins > 65536U) {
    throw std::invalid_argument("joint bins must match rows and num_bins must be in [1, 65536]");
  }
  MultivariateBinStatistics result;
  result.response_dimension = response_dimension;
  result.gradient_sums.assign(num_bins * response_dimension, 0.0);
  result.weight_sums.assign(num_bins, 0.0);
  for (std::size_t row = 0; row < bins.size(); ++row) {
    const std::size_t bin = bins[row];
    const double weight = weights[row];
    if (bin >= num_bins || !std::isfinite(weight) || weight < 0.0) {
      throw std::invalid_argument("joint bins must be in range and weights finite and nonnegative");
    }
    result.weight_sums[bin] += weight;
    for (std::size_t output = 0; output < response_dimension; ++output) {
      const double gradient = gradients[row * response_dimension + output];
      if (!std::isfinite(gradient)) {
        throw std::invalid_argument("joint gradients must be finite");
      }
      result.gradient_sums[bin * response_dimension + output] += weight * gradient;
    }
  }
  return result;
}

MultivariateBinStatistics GroupOrderedMultivariateBinStatistics(
    const MultivariateBinStatistics& stats,
    std::size_t requested_groups,
    std::size_t missing_bin) {
  ValidateBins(stats);
  if (requested_groups < 2 || requested_groups > 64) {
    throw std::invalid_argument("requested joint statistic groups must be in [2, 64]");
  }
  const bool has_missing = missing_bin != kNoMissingStatisticBin;
  if (has_missing && missing_bin >= stats.weight_sums.size()) {
    throw std::invalid_argument("missing joint statistic bin is out of range");
  }
  MultivariateBinStatistics grouped;
  grouped.response_dimension = stats.response_dimension;
  double nonmissing_weight = 0.0;
  for (std::size_t bin = 0; bin < stats.weight_sums.size(); ++bin) {
    if (!has_missing || bin != missing_bin) {
      nonmissing_weight += stats.weight_sums[bin];
    }
  }
  const auto append_bin = [&](std::size_t bin, bool new_group) {
    if (new_group) {
      grouped.weight_sums.push_back(0.0);
      grouped.gradient_sums.resize(grouped.gradient_sums.size() + stats.response_dimension, 0.0);
    }
    grouped.weight_sums.back() += stats.weight_sums[bin];
    const std::size_t offset = grouped.gradient_sums.size() - stats.response_dimension;
    for (std::size_t output = 0; output < stats.response_dimension; ++output) {
      grouped.gradient_sums[offset + output] += stats.gradient_sums[bin * stats.response_dimension + output];
    }
  };
  double before = 0.0;
  std::size_t previous = kNoMissingStatisticBin;
  for (std::size_t bin = 0; bin < stats.weight_sums.size(); ++bin) {
    const double weight = stats.weight_sums[bin];
    if (weight == 0.0 || (has_missing && bin == missing_bin)) {
      continue;
    }
    const std::size_t group = std::min(
        requested_groups - 1, static_cast<std::size_t>(requested_groups * (before + 0.5 * weight) / nonmissing_weight));
    append_bin(bin, group != previous);
    previous = group;
    before += weight;
  }
  if (has_missing && stats.weight_sums[missing_bin] > 0.0) {
    append_bin(missing_bin, true);
  }
  return grouped;
}

LinearStatisticScore EvaluateMultivariateStatisticFromBins(
    const MultivariateBinStatistics& bins,
    const MultivariateResponseStatistics& response) {
  ValidateBins(bins);
  const std::size_t dimension = response.response_dimension;
  if (dimension != bins.response_dimension || response.mean.size() != dimension ||
      response.whitening.size() != dimension * dimension || response.covariance_rank > dimension ||
      !std::isfinite(response.weight_sum) || response.weight_sum < 0.0) {
    throw std::invalid_argument("invalid joint response statistics");
  }
  for (const double value : response.mean) {
    if (!std::isfinite(value)) throw std::invalid_argument("joint response means must be finite");
  }
  for (const double value : response.whitening) {
    if (!std::isfinite(value)) throw std::invalid_argument("joint response whitening must be finite");
  }
  const double bin_weight_sum = std::accumulate(bins.weight_sums.begin(), bins.weight_sums.end(), 0.0);
  if (!std::isfinite(bin_weight_sum) ||
      std::abs(bin_weight_sum - response.weight_sum) > 1e-9 * std::max(1.0, response.weight_sum)) {
    throw std::invalid_argument("joint bin and response weights must describe the same node");
  }
  LinearStatisticScore result;
  const std::size_t occupied = static_cast<std::size_t>(std::count_if(
      bins.weight_sums.begin(), bins.weight_sums.end(), [](double weight) { return weight > 0.0; }));
  if (response.weight_sum <= 1.0 || occupied <= 1 || response.covariance_rank == 0) {
    return result;
  }
  result.degrees_of_freedom = (occupied - 1) * response.covariance_rank;
  std::vector<double> residual(dimension);
  double statistic = 0.0;
  for (std::size_t bin = 0; bin < bins.weight_sums.size(); ++bin) {
    const double weight = bins.weight_sums[bin];
    if (weight == 0.0) continue;
    for (std::size_t output = 0; output < dimension; ++output) {
      residual[output] = bins.gradient_sums[bin * dimension + output] - weight * response.mean[output];
    }
    // Squared whitened projections avoid cancellation in S' C^+ S when C is
    // singular or poorly conditioned, and never produce a negative statistic.
    for (std::size_t component = 0; component < dimension; ++component) {
      double projection = 0.0;
      for (std::size_t output = 0; output < dimension; ++output) {
        projection += residual[output] * response.whitening[output * dimension + component];
      }
      statistic += (projection / weight) * projection;
    }
  }
  result.chi_square = ((response.weight_sum - 1.0) / response.weight_sum) * statistic;
  if (!std::isfinite(result.chi_square)) {
    throw std::invalid_argument("joint statistic is non-finite");
  }
  result.p_value = ChiSquareSurvival(result.chi_square, result.degrees_of_freedom);
  return result;
}

}  // namespace ctboost
