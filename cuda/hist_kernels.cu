#include "hist_kernels.cuh"

__global__ void HistMatrixSumsKernel(const std::uint16_t* bins,
                                     const std::size_t* feature_offsets,
                                     const float* gradients,
                                     const float* hessians,
                                     float* gradient_sums,
                                     float* hessian_sums,
                                     std::uint32_t* counts,
                                     std::size_t num_rows,
                                     std::size_t num_features) {
  const std::size_t row = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t feature = static_cast<std::size_t>(blockIdx.y);
  if (row >= num_rows || feature >= num_features) {
    return;
  }

  const std::size_t feature_offset = feature_offsets[feature];
  const std::size_t next_feature_offset = feature_offsets[feature + 1];
  const std::size_t bin =
      static_cast<std::size_t>(bins[feature * num_rows + row]);
  if (feature_offset + bin >= next_feature_offset) {
    return;
  }

  const std::size_t output_index = feature_offset + bin;
  atomicAdd(&gradient_sums[output_index], gradients[row]);
  atomicAdd(&hessian_sums[output_index], hessians[row]);
  atomicAdd(&counts[output_index], 1U);
}
