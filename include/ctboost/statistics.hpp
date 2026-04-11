#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ctboost {

struct BinStatistics {
  std::vector<double> gradient_sums;
  std::vector<double> hessian_sums;
  std::vector<double> weight_sums;
};

struct LinearStatisticResult {
  std::vector<double> statistic;
  std::vector<double> expectation;
  std::vector<double> covariance;
  std::size_t num_bins{0};
  std::size_t degrees_of_freedom{0};
  double chi_square{0.0};
  double p_value{1.0};
};

class LinearStatistic {
 public:
  explicit LinearStatistic(double epsilon = 1e-7);

  BinStatistics ComputeBinStatistics(const std::vector<float>& gradients,
                                     const std::vector<float>& hessians,
                                     const std::vector<float>& weights,
                                     const std::vector<std::uint16_t>& bins,
                                     std::size_t num_bins) const;

  LinearStatisticResult EvaluateFromBinStatistics(const BinStatistics& stats,
                                                  double total_gradient,
                                                  double sample_weight_sum,
                                                  double gradient_variance) const;

  LinearStatisticResult Evaluate(const std::vector<float>& gradients,
                                 const std::vector<float>& hessians,
                                 const std::vector<float>& weights,
                                 const std::vector<std::uint16_t>& bins,
                                 std::size_t num_bins) const;

  double epsilon() const noexcept;

 private:
  double epsilon_{1e-7};
};

double ChiSquareSurvival(double statistic, std::size_t degrees_of_freedom);

}  // namespace ctboost
