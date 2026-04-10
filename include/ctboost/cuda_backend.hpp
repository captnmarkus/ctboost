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
                        std::size_t num_features,
                        const std::vector<std::uint16_t>& num_bins_per_feature,
                        const std::vector<float>& gradients,
                        const std::vector<float>& hessians,
                        std::vector<float>& out_gradient_sums,
                        std::vector<float>& out_hessian_sums,
                        std::vector<std::uint32_t>& out_counts,
                        std::vector<std::size_t>& out_feature_offsets);

}  // namespace ctboost
