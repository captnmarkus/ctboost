#include "ctboost/statistics.hpp"

#include "statistics_internal.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ctboost::detail {
namespace {

constexpr int kMaxGammaIterations = 200;
constexpr double kGammaTolerance = 3e-14;
constexpr double kGammaTiny = 1e-300;

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

}  // namespace

void ValidateStatisticSizes(const std::vector<float>& gradients,
                            const std::vector<float>& hessians,
                            const std::vector<float>& weights,
                            const std::vector<std::uint16_t>& bins,
                            std::size_t num_bins) {
  if (gradients.size() != hessians.size() || gradients.size() != weights.size() ||
      gradients.size() != bins.size()) {
    throw std::invalid_argument("gradients, hessians, weights, and bins must have the same size");
  }
  if (num_bins == 0) {
    throw std::invalid_argument("num_bins must be greater than zero");
  }
}

std::vector<double> SolveStatisticLinearSystem(std::vector<double> matrix,
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

}  // namespace ctboost::detail
