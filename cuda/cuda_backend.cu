#include "ctboost/cuda_backend.hpp"
#include "ctboost/histogram.hpp"
#include "hist_kernels.cuh"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
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

constexpr int kHistogramThreads = 256;
constexpr std::size_t kHistogramRowTileSize = 1024;
constexpr int kPredictionThreads = 256;

std::vector<std::size_t> BuildFeatureOffsets(
    const std::vector<std::uint16_t>& num_bins_per_feature) {
  std::vector<std::size_t> feature_offsets(num_bins_per_feature.size() + 1, 0);
  for (std::size_t feature = 0; feature < num_bins_per_feature.size(); ++feature) {
    feature_offsets[feature + 1] =
        feature_offsets[feature] + static_cast<std::size_t>(num_bins_per_feature[feature]);
  }
  return feature_offsets;
}

template <typename T>
void CopyHostVectorToDevice(const std::vector<T>& source, thrust::device_vector<T>& destination) {
  destination.resize(source.size());
  if (source.empty()) {
    return;
  }
  thrust::copy(source.begin(), source.end(), destination.begin());
}

}  // namespace

struct GpuHistogramWorkspace {
  std::size_t num_rows{0};
  std::size_t num_features{0};
  std::size_t total_bins{0};
  std::size_t max_feature_bins{0};
  std::size_t max_shared_hist_bins{0};
  std::vector<std::size_t> feature_offsets;
  thrust::device_vector<std::uint16_t> bins;
  thrust::device_vector<float> weights;
  thrust::device_vector<float> gradients;
  thrust::device_vector<float> hessians;
  thrust::device_vector<std::size_t> row_indices;
  thrust::device_vector<float> gradient_sums;
  thrust::device_vector<float> hessian_sums;
  thrust::device_vector<float> weight_sums;
};

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

void DestroyGpuHistogramWorkspace(GpuHistogramWorkspace* workspace) noexcept {
  delete workspace;
}

GpuHistogramWorkspacePtr CreateGpuHistogramWorkspace(const HistMatrix& hist,
                                                     const std::vector<float>& weights) {
  if (weights.size() != hist.num_rows) {
    throw std::invalid_argument("GPU histogram weights must match the histogram row count");
  }
  if (hist.bin_indices.size() != hist.num_rows * hist.num_cols) {
    throw std::invalid_argument("GPU histogram bins must have num_rows * num_cols elements");
  }

  try {
    auto workspace = std::make_unique<GpuHistogramWorkspace>();
    workspace->num_rows = hist.num_rows;
    workspace->num_features = hist.num_cols;
    workspace->feature_offsets = BuildFeatureOffsets(hist.num_bins_per_feature);
    workspace->total_bins = workspace->feature_offsets.empty() ? 0 : workspace->feature_offsets.back();
    for (const std::uint16_t feature_bins : hist.num_bins_per_feature) {
      workspace->max_feature_bins =
          std::max(workspace->max_feature_bins, static_cast<std::size_t>(feature_bins));
    }
    cudaDeviceProp device_properties{};
    CTBOOST_CUDA_CHECK(cudaGetDeviceProperties(&device_properties, 0));
    workspace->max_shared_hist_bins =
        static_cast<std::size_t>(device_properties.sharedMemPerBlock) / (3U * sizeof(float));

    CopyHostVectorToDevice(hist.bin_indices, workspace->bins);
    CopyHostVectorToDevice(weights, workspace->weights);
    workspace->gradients.resize(hist.num_rows, 0.0F);
    workspace->hessians.resize(hist.num_rows, 0.0F);
    workspace->gradient_sums.resize(workspace->total_bins, 0.0F);
    workspace->hessian_sums.resize(workspace->total_bins, 0.0F);
    workspace->weight_sums.resize(workspace->total_bins, 0.0F);

    return GpuHistogramWorkspacePtr(workspace.release(), DestroyGpuHistogramWorkspace);
  } catch (const thrust::system_error& error) {
    throw std::runtime_error(std::string("CUDA thrust failure: ") + error.what());
  }
}

void UploadHistogramTargetsGpu(GpuHistogramWorkspace* workspace,
                               const std::vector<float>& gradients,
                               const std::vector<float>& hessians) {
  if (workspace == nullptr) {
    throw std::invalid_argument("GPU histogram workspace must not be null");
  }
  if (gradients.size() != workspace->num_rows || hessians.size() != workspace->num_rows) {
    throw std::invalid_argument(
        "GPU histogram gradients and hessians must match the histogram row count");
  }

  try {
    CopyHostVectorToDevice(gradients, workspace->gradients);
    CopyHostVectorToDevice(hessians, workspace->hessians);
  } catch (const thrust::system_error& error) {
    throw std::runtime_error(std::string("CUDA thrust failure: ") + error.what());
  }
}

void BuildHistogramsGpu(GpuHistogramWorkspace* workspace,
                        const std::vector<std::size_t>& row_indices,
                        std::vector<float>& out_gradient_sums,
                        std::vector<float>& out_hessian_sums,
                        std::vector<float>& out_weight_sums,
                        std::vector<std::size_t>& out_feature_offsets) {
  if (workspace == nullptr) {
    throw std::invalid_argument("GPU histogram workspace must not be null");
  }

  out_feature_offsets = workspace->feature_offsets;
  out_gradient_sums.assign(workspace->total_bins, 0.0F);
  out_hessian_sums.assign(workspace->total_bins, 0.0F);
  out_weight_sums.assign(workspace->total_bins, 0.0F);

  if (row_indices.empty() || workspace->num_features == 0 || workspace->total_bins == 0) {
    return;
  }

  try {
    CopyHostVectorToDevice(row_indices, workspace->row_indices);
    thrust::fill(workspace->gradient_sums.begin(), workspace->gradient_sums.end(), 0.0F);
    thrust::fill(workspace->hessian_sums.begin(), workspace->hessian_sums.end(), 0.0F);
    thrust::fill(workspace->weight_sums.begin(), workspace->weight_sums.end(), 0.0F);

    const unsigned int row_tiles = static_cast<unsigned int>(
        (row_indices.size() + kHistogramRowTileSize - 1) / kHistogramRowTileSize);
    for (std::size_t feature = 0; feature < workspace->num_features; ++feature) {
      const std::size_t feature_offset = workspace->feature_offsets[feature];
      const std::size_t next_feature_offset = workspace->feature_offsets[feature + 1];
      const std::size_t feature_bin_count = next_feature_offset - feature_offset;
      if (feature_bin_count == 0) {
        continue;
      }

      if (feature_bin_count <= workspace->max_shared_hist_bins) {
        const std::size_t shared_bytes = feature_bin_count * 3U * sizeof(float);
        HistMatrixFeatureSumsKernel<<<row_tiles, kHistogramThreads, shared_bytes>>>(
            thrust::raw_pointer_cast(workspace->bins.data()),
            thrust::raw_pointer_cast(workspace->row_indices.data()),
            thrust::raw_pointer_cast(workspace->gradients.data()),
            thrust::raw_pointer_cast(workspace->hessians.data()),
            thrust::raw_pointer_cast(workspace->weights.data()),
            thrust::raw_pointer_cast(workspace->gradient_sums.data()),
            thrust::raw_pointer_cast(workspace->hessian_sums.data()),
            thrust::raw_pointer_cast(workspace->weight_sums.data()),
            row_indices.size(),
            workspace->num_rows,
            feature,
            feature_offset,
            feature_bin_count);
      } else {
        ChunkedHistMatrixFeatureSumsKernel<<<row_tiles, kHistogramThreads>>>(
            thrust::raw_pointer_cast(workspace->bins.data()),
            thrust::raw_pointer_cast(workspace->row_indices.data()),
            thrust::raw_pointer_cast(workspace->gradients.data()),
            thrust::raw_pointer_cast(workspace->hessians.data()),
            thrust::raw_pointer_cast(workspace->weights.data()),
            thrust::raw_pointer_cast(workspace->gradient_sums.data()),
            thrust::raw_pointer_cast(workspace->hessian_sums.data()),
            thrust::raw_pointer_cast(workspace->weight_sums.data()),
            row_indices.size(),
            workspace->num_rows,
            feature,
            feature_offset,
            feature_bin_count);
      }
    }

    CTBOOST_CUDA_CHECK(cudaGetLastError());
    CTBOOST_CUDA_CHECK(cudaDeviceSynchronize());

    thrust::copy(
        workspace->gradient_sums.begin(), workspace->gradient_sums.end(), out_gradient_sums.begin());
    thrust::copy(
        workspace->hessian_sums.begin(), workspace->hessian_sums.end(), out_hessian_sums.begin());
    thrust::copy(workspace->weight_sums.begin(), workspace->weight_sums.end(), out_weight_sums.begin());
  } catch (const thrust::system_error& error) {
    throw std::runtime_error(std::string("CUDA thrust failure: ") + error.what());
  }
}

void PredictRawGpu(const HistMatrix& hist,
                   const std::vector<GpuTreeNode>& nodes,
                   const std::vector<std::int32_t>& tree_offsets,
                   float learning_rate,
                   int prediction_dimension,
                   std::vector<float>& out_predictions) {
  if (prediction_dimension <= 0) {
    throw std::invalid_argument("prediction_dimension must be positive");
  }
  if (nodes.empty() || tree_offsets.empty() || hist.num_rows == 0) {
    return;
  }
  if (hist.bin_indices.size() != hist.num_rows * hist.num_cols) {
    throw std::invalid_argument("GPU prediction bins must have num_rows * num_cols elements");
  }
  if (out_predictions.size() != hist.num_rows * static_cast<std::size_t>(prediction_dimension)) {
    throw std::invalid_argument("GPU prediction output buffer has an unexpected size");
  }

  try {
    thrust::device_vector<std::uint16_t> device_bins(hist.bin_indices.begin(), hist.bin_indices.end());
    thrust::device_vector<GpuTreeNode> device_nodes(nodes.begin(), nodes.end());
    thrust::device_vector<std::int32_t> device_tree_offsets(tree_offsets.begin(), tree_offsets.end());
    thrust::device_vector<float> device_predictions(out_predictions.size(), 0.0F);

    const int blocks =
        static_cast<int>((hist.num_rows + kPredictionThreads - 1) / kPredictionThreads);
    PredictForestKernel<<<blocks, kPredictionThreads>>>(
        thrust::raw_pointer_cast(device_bins.data()),
        thrust::raw_pointer_cast(device_nodes.data()),
        thrust::raw_pointer_cast(device_tree_offsets.data()),
        thrust::raw_pointer_cast(device_predictions.data()),
        hist.num_rows,
        tree_offsets.size(),
        prediction_dimension,
        learning_rate);
    CTBOOST_CUDA_CHECK(cudaGetLastError());
    CTBOOST_CUDA_CHECK(cudaDeviceSynchronize());

    thrust::copy(device_predictions.begin(), device_predictions.end(), out_predictions.begin());
  } catch (const thrust::system_error& error) {
    throw std::runtime_error(std::string("CUDA thrust failure: ") + error.what());
  }
}

}  // namespace ctboost
