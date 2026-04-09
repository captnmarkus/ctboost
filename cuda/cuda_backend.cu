#include "ctboost/cuda_backend.hpp"
#include "hist_kernels.cuh"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#define CTBOOST_CUDA_CHECK(expr)                                                       \
  do {                                                                                 \
    const cudaError_t status = (expr);                                                 \
    if (status != cudaSuccess) {                                                       \
      throw std::runtime_error(std::string("CUDA failure: ") + cudaGetErrorString(status)); \
    }                                                                                  \
  } while (false)

namespace ctboost {

bool CudaBackendCompiled() noexcept { return true; }

std::string CudaRuntimeVersionString() {
  int runtime_version = 0;
  const cudaError_t status = cudaRuntimeGetVersion(&runtime_version);
  if (status != cudaSuccess) {
    return std::string("error: ") + cudaGetErrorString(status);
  }

  const int major = runtime_version / 1000;
  const int minor = (runtime_version % 1000) / 10;
  return std::to_string(major) + "." + std::to_string(minor);
}

void BuildHistogramsGpu(const std::vector<std::uint16_t>& bins,
                        std::size_t num_rows,
                        std::size_t num_bins,
                        const std::vector<float>& gradients,
                        const std::vector<float>& hessians,
                        std::vector<float>& out_gradient_sums,
                        std::vector<float>& out_hessian_sums,
                        std::vector<std::uint32_t>& out_counts) {
  if (bins.size() != num_rows || gradients.size() != num_rows || hessians.size() != num_rows) {
    throw std::invalid_argument("GPU histogram inputs must all have num_rows elements");
  }
  if (num_bins == 0) {
    throw std::invalid_argument("num_bins must be greater than zero");
  }

  out_gradient_sums.assign(num_bins, 0.0F);
  out_hessian_sums.assign(num_bins, 0.0F);
  out_counts.assign(num_bins, 0);

  std::uint16_t* d_bins = nullptr;
  float* d_gradients = nullptr;
  float* d_hessians = nullptr;
  float* d_gradient_sums = nullptr;
  float* d_hessian_sums = nullptr;
  std::uint32_t* d_counts = nullptr;

  CTBOOST_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_bins),
                                bins.size() * sizeof(std::uint16_t)));
  CTBOOST_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gradients),
                                gradients.size() * sizeof(float)));
  CTBOOST_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hessians),
                                hessians.size() * sizeof(float)));
  CTBOOST_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gradient_sums),
                                num_bins * sizeof(float)));
  CTBOOST_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hessian_sums),
                                num_bins * sizeof(float)));
  CTBOOST_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_counts),
                                num_bins * sizeof(std::uint32_t)));

  try {
    CTBOOST_CUDA_CHECK(cudaMemcpy(
        d_bins, bins.data(), bins.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice));
    CTBOOST_CUDA_CHECK(cudaMemcpy(d_gradients,
                                  gradients.data(),
                                  gradients.size() * sizeof(float),
                                  cudaMemcpyHostToDevice));
    CTBOOST_CUDA_CHECK(cudaMemcpy(d_hessians,
                                  hessians.data(),
                                  hessians.size() * sizeof(float),
                                  cudaMemcpyHostToDevice));
    CTBOOST_CUDA_CHECK(cudaMemset(d_gradient_sums, 0, num_bins * sizeof(float)));
    CTBOOST_CUDA_CHECK(cudaMemset(d_hessian_sums, 0, num_bins * sizeof(float)));
    CTBOOST_CUDA_CHECK(cudaMemset(d_counts, 0, num_bins * sizeof(std::uint32_t)));

    const int threads = 256;
    const int blocks = static_cast<int>((num_rows + threads - 1) / threads);
    HistSumsKernel<<<blocks, threads>>>(
        d_bins, d_gradients, d_hessians, d_gradient_sums, d_hessian_sums, d_counts, num_rows, num_bins);
    CTBOOST_CUDA_CHECK(cudaGetLastError());
    CTBOOST_CUDA_CHECK(cudaDeviceSynchronize());

    CTBOOST_CUDA_CHECK(cudaMemcpy(out_gradient_sums.data(),
                                  d_gradient_sums,
                                  num_bins * sizeof(float),
                                  cudaMemcpyDeviceToHost));
    CTBOOST_CUDA_CHECK(cudaMemcpy(out_hessian_sums.data(),
                                  d_hessian_sums,
                                  num_bins * sizeof(float),
                                  cudaMemcpyDeviceToHost));
    CTBOOST_CUDA_CHECK(cudaMemcpy(out_counts.data(),
                                  d_counts,
                                  num_bins * sizeof(std::uint32_t),
                                  cudaMemcpyDeviceToHost));
  } catch (...) {
    cudaFree(d_bins);
    cudaFree(d_gradients);
    cudaFree(d_hessians);
    cudaFree(d_gradient_sums);
    cudaFree(d_hessian_sums);
    cudaFree(d_counts);
    throw;
  }

  cudaFree(d_bins);
  cudaFree(d_gradients);
  cudaFree(d_hessians);
  cudaFree(d_gradient_sums);
  cudaFree(d_hessian_sums);
  cudaFree(d_counts);
}

}  // namespace ctboost
