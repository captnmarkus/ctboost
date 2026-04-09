#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ctboost {

bool CudaBackendCompiled() noexcept;
std::string CudaRuntimeVersionString();
void BuildHistogramsGpu(const std::vector<std::uint16_t>& bins,
                        std::size_t num_rows,
                        std::size_t num_bins,
                        const std::vector<float>& gradients,
                        const std::vector<float>& hessians,
                        std::vector<float>& out_gradient_sums,
                        std::vector<float>& out_hessian_sums,
                        std::vector<std::uint32_t>& out_counts);

}  // namespace ctboost
