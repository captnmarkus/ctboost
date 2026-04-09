#pragma once

#include <cstddef>
#include <cstdint>

__global__ void HistSumsKernel(const std::uint16_t* bins,
                               const float* gradients,
                               const float* hessians,
                               float* gradient_sums,
                               float* hessian_sums,
                               std::uint32_t* counts,
                               std::size_t num_rows,
                               std::size_t num_bins);
