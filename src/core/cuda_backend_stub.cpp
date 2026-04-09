#include "ctboost/cuda_backend.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ctboost {

bool CudaBackendCompiled() noexcept { return false; }

std::string CudaRuntimeVersionString() { return "not compiled"; }

void BuildHistogramsGpu(const std::vector<std::uint16_t>& bins,
                        std::size_t num_rows,
                        std::size_t num_bins,
                        const std::vector<float>& gradients,
                        const std::vector<float>& hessians,
                        std::vector<float>& out_gradient_sums,
                        std::vector<float>& out_hessian_sums,
                        std::vector<std::uint32_t>& out_counts) {
  (void)bins;
  (void)num_rows;
  (void)num_bins;
  (void)gradients;
  (void)hessians;
  (void)out_gradient_sums;
  (void)out_hessian_sums;
  (void)out_counts;
  throw std::runtime_error("CUDA histogram builder requested but CTBoost was compiled without CUDA");
}

}  // namespace ctboost
