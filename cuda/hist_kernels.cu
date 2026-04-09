#include "hist_kernels.cuh"

__global__ void HistSumsKernel(const std::uint16_t* bins,
                               const float* gradients,
                               const float* hessians,
                               float* gradient_sums,
                               float* hessian_sums,
                               std::uint32_t* counts,
                               std::size_t num_rows,
                               std::size_t num_bins) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= num_rows) {
    return;
  }

  const std::size_t bin = static_cast<std::size_t>(bins[index]);
  if (bin >= num_bins) {
    return;
  }

  atomicAdd(&gradient_sums[bin], gradients[index]);
  atomicAdd(&hessian_sums[bin], hessians[index]);
  atomicAdd(&counts[bin], 1U);
}
