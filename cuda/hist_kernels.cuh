#pragma once

#include <cstddef>
#include <cstdint>

__global__ void HistMatrixSumsKernel(const std::uint16_t* bins,
                                     const std::size_t* feature_offsets,
                                     const float* gradients,
                                     const float* hessians,
                                     float* gradient_sums,
                                     float* hessian_sums,
                                     std::uint32_t* counts,
                                     std::size_t num_rows,
                                     std::size_t num_features);
