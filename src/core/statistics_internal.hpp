#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ctboost::detail {

void ValidateStatisticSizes(const std::vector<float>& gradients,
                            const std::vector<float>& hessians,
                            const std::vector<float>& weights,
                            const std::vector<std::uint16_t>& bins,
                            std::size_t num_bins);
std::vector<double> SolveStatisticLinearSystem(std::vector<double> matrix,
                                               std::vector<double> rhs);
double RegularizedGammaQ(double a, double x);

}  // namespace ctboost::detail
