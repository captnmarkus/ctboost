#include "ctboost/cuda_backend.hpp"
#include "hist_kernels.cuh"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/system_error.h>

#define CTBOOST_CUDA_CHECK(expr)                                                            \
  do {                                                                                      \
    const cudaError_t status = (expr);                                                      \
    if (status != cudaSuccess) {                                                            \
      throw std::runtime_error(std::string("CUDA failure: ") + cudaGetErrorString(status)); \
    }                                                                                       \
  } while (false)

namespace ctboost {
namespace {

std::vector<std::size_t> BuildFeatureOffsets(
    const std::vector<std::uint16_t>& num_bins_per_feature) {
  std::vector<std::size_t> feature_offsets(num_bins_per_feature.size() + 1, 0);
  for (std::size_t feature = 0; feature < num_bins_per_feature.size(); ++feature) {
    feature_offsets[feature + 1] =
        feature_offsets[feature] + static_cast<std::size_t>(num_bins_per_feature[feature]);
  }
  return feature_offsets;
}

}  // namespace

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
                        std::size_t num_features,
                        const std::vector<std::uint16_t>& num_bins_per_feature,
                        const std::vector<float>& gradients,
                        const std::vector<float>& hessians,
                        std::vector<float>& out_gradient_sums,
                        std::vector<float>& out_hessian_sums,
                        std::vector<std::uint32_t>& out_counts,
                        std::vector<std::size_t>& out_feature_offsets) {
  if (gradients.size() != num_rows || hessians.size() != num_rows) {
    throw std::invalid_argument("GPU histogram gradients and hessians must match num_rows");
  }
  if (num_bins_per_feature.size() != num_features) {
    throw std::invalid_argument("GPU histogram num_bins_per_feature must match num_features");
  }
  if (bins.size() != num_rows * num_features) {
    throw std::invalid_argument("GPU histogram bins must have num_rows * num_features elements");
  }

  out_feature_offsets = BuildFeatureOffsets(num_bins_per_feature);
  const std::size_t total_bins = out_feature_offsets.back();
  out_gradient_sums.assign(total_bins, 0.0F);
  out_hessian_sums.assign(total_bins, 0.0F);
  out_counts.assign(total_bins, 0);

  if (num_rows == 0 || num_features == 0 || total_bins == 0) {
    return;
  }

  try {
    thrust::device_vector<std::uint16_t> d_bins(bins.begin(), bins.end());
    thrust::device_vector<std::size_t> d_feature_offsets(
        out_feature_offsets.begin(), out_feature_offsets.end());
    thrust::device_vector<float> d_gradients(gradients.begin(), gradients.end());
    thrust::device_vector<float> d_hessians(hessians.begin(), hessians.end());
    thrust::device_vector<float> d_gradient_sums(total_bins, 0.0F);
    thrust::device_vector<float> d_hessian_sums(total_bins, 0.0F);
    thrust::device_vector<std::uint32_t> d_counts(total_bins, 0U);

    const int threads = 256;
    const int row_blocks = static_cast<int>((num_rows + threads - 1) / threads);
    const dim3 blocks(row_blocks, static_cast<unsigned int>(num_features), 1U);
    HistMatrixSumsKernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(d_bins.data()),
        thrust::raw_pointer_cast(d_feature_offsets.data()),
        thrust::raw_pointer_cast(d_gradients.data()),
        thrust::raw_pointer_cast(d_hessians.data()),
        thrust::raw_pointer_cast(d_gradient_sums.data()),
        thrust::raw_pointer_cast(d_hessian_sums.data()),
        thrust::raw_pointer_cast(d_counts.data()),
        num_rows,
        num_features);
    CTBOOST_CUDA_CHECK(cudaGetLastError());
    CTBOOST_CUDA_CHECK(cudaDeviceSynchronize());

    thrust::copy(d_gradient_sums.begin(), d_gradient_sums.end(), out_gradient_sums.begin());
    thrust::copy(d_hessian_sums.begin(), d_hessian_sums.end(), out_hessian_sums.begin());
    thrust::copy(d_counts.begin(), d_counts.end(), out_counts.begin());
  } catch (const thrust::system_error& error) {
    throw std::runtime_error(std::string("CUDA thrust failure: ") + error.what());
  }
}

}  // namespace ctboost
