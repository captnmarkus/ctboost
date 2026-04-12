#include "ctboost/cuda_backend.hpp"
#include "ctboost/histogram.hpp"
#include "hist_kernels.cuh"

#include <cuda_runtime_api.h>

#include <algorithm>
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
constexpr std::size_t kHistogramChunkBins = 256;
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

template <typename T>
void CopyHostSliceToDevice(const std::vector<T>& source,
                           std::size_t begin,
                           std::size_t end,
                           thrust::device_vector<T>& destination) {
  if (begin > end || end > source.size()) {
    throw std::invalid_argument("GPU histogram row range is out of bounds");
  }

  destination.resize(end - begin);
  if (begin == end) {
    return;
  }
  thrust::copy(source.begin() + static_cast<std::ptrdiff_t>(begin),
               source.begin() + static_cast<std::ptrdiff_t>(end),
               destination.begin());
}

}  // namespace

struct GpuHistogramWorkspace {
  std::size_t num_rows{0};
  std::size_t num_features{0};
  std::size_t total_bins{0};
  std::size_t max_feature_bins{0};
  std::size_t histogram_chunk_bins{kHistogramChunkBins};
  std::vector<std::size_t> feature_offsets;
  thrust::device_vector<std::uint16_t> bins;
  thrust::device_vector<float> weights;
  thrust::device_vector<float> gradients;
  thrust::device_vector<float> hessians;
  thrust::device_vector<float> multitarget_gradients;
  thrust::device_vector<float> multitarget_hessians;
  bool multitarget_enabled{false};
  std::size_t target_stride{1};
  std::size_t active_target_index{0};
  thrust::device_vector<std::size_t> row_indices;
  thrust::device_vector<float> gradient_sums;
  thrust::device_vector<float> hessian_sums;
  thrust::device_vector<float> weight_sums;
  thrust::device_vector<double> node_statistics;
  thrust::device_vector<std::uint32_t> chunk_feature_indices;
  thrust::device_vector<std::uint32_t> chunk_bin_starts;
  thrust::device_vector<std::uint32_t> chunk_bin_counts;
  thrust::device_vector<std::uint32_t> chunk_output_offsets;
};

namespace {

const float* ResolveGradientPointer(const GpuHistogramWorkspace& workspace) {
  return workspace.multitarget_enabled
             ? thrust::raw_pointer_cast(workspace.multitarget_gradients.data())
             : thrust::raw_pointer_cast(workspace.gradients.data());
}

const float* ResolveHessianPointer(const GpuHistogramWorkspace& workspace) {
  return workspace.multitarget_enabled
             ? thrust::raw_pointer_cast(workspace.multitarget_hessians.data())
             : thrust::raw_pointer_cast(workspace.hessians.data());
}

std::size_t ResolveTargetStride(const GpuHistogramWorkspace& workspace) {
  return workspace.multitarget_enabled ? workspace.target_stride : 1U;
}

std::size_t ResolveTargetOffset(const GpuHistogramWorkspace& workspace) {
  return workspace.multitarget_enabled ? workspace.active_target_index : 0U;
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

    CopyHostVectorToDevice(hist.bin_indices, workspace->bins);
    CopyHostVectorToDevice(weights, workspace->weights);
    workspace->gradients.resize(hist.num_rows, 0.0F);
    workspace->hessians.resize(hist.num_rows, 0.0F);
    workspace->gradient_sums.resize(workspace->total_bins, 0.0F);
    workspace->hessian_sums.resize(workspace->total_bins, 0.0F);
    workspace->weight_sums.resize(workspace->total_bins, 0.0F);
    workspace->node_statistics.resize(4, 0.0);

    std::vector<std::uint32_t> chunk_feature_indices;
    std::vector<std::uint32_t> chunk_bin_starts;
    std::vector<std::uint32_t> chunk_bin_counts;
    std::vector<std::uint32_t> chunk_output_offsets;
    chunk_feature_indices.reserve(hist.num_cols);
    chunk_bin_starts.reserve(hist.num_cols);
    chunk_bin_counts.reserve(hist.num_cols);
    chunk_output_offsets.reserve(hist.num_cols);
    for (std::size_t feature = 0; feature < hist.num_cols; ++feature) {
      const std::size_t feature_bin_count = hist.num_bins(feature);
      for (std::size_t bin_start = 0; bin_start < feature_bin_count;
           bin_start += workspace->histogram_chunk_bins) {
        const std::size_t chunk_bins =
            std::min(workspace->histogram_chunk_bins, feature_bin_count - bin_start);
        chunk_feature_indices.push_back(static_cast<std::uint32_t>(feature));
        chunk_bin_starts.push_back(static_cast<std::uint32_t>(bin_start));
        chunk_bin_counts.push_back(static_cast<std::uint32_t>(chunk_bins));
        chunk_output_offsets.push_back(
            static_cast<std::uint32_t>(workspace->feature_offsets[feature]));
      }
    }
    CopyHostVectorToDevice(chunk_feature_indices, workspace->chunk_feature_indices);
    CopyHostVectorToDevice(chunk_bin_starts, workspace->chunk_bin_starts);
    CopyHostVectorToDevice(chunk_bin_counts, workspace->chunk_bin_counts);
    CopyHostVectorToDevice(chunk_output_offsets, workspace->chunk_output_offsets);

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
    workspace->multitarget_enabled = false;
    workspace->target_stride = 1;
    workspace->active_target_index = 0;
    workspace->multitarget_gradients.clear();
    workspace->multitarget_hessians.clear();
  } catch (const thrust::system_error& error) {
    throw std::runtime_error(std::string("CUDA thrust failure: ") + error.what());
  }
}

void UploadHistogramTargetMatrixGpu(GpuHistogramWorkspace* workspace,
                                    const std::vector<float>& gradients,
                                    const std::vector<float>& hessians,
                                    std::size_t target_stride) {
  if (workspace == nullptr) {
    throw std::invalid_argument("GPU histogram workspace must not be null");
  }
  if (target_stride == 0) {
    throw std::invalid_argument("GPU histogram target_stride must be positive");
  }
  if (gradients.size() != workspace->num_rows * target_stride ||
      hessians.size() != workspace->num_rows * target_stride) {
    throw std::invalid_argument(
        "GPU histogram multitarget buffers must match num_rows * target_stride");
  }

  try {
    CopyHostVectorToDevice(gradients, workspace->multitarget_gradients);
    CopyHostVectorToDevice(hessians, workspace->multitarget_hessians);
    workspace->multitarget_enabled = true;
    workspace->target_stride = target_stride;
    workspace->active_target_index = 0;
  } catch (const thrust::system_error& error) {
    throw std::runtime_error(std::string("CUDA thrust failure: ") + error.what());
  }
}

void SelectHistogramTargetGpuClass(GpuHistogramWorkspace* workspace, std::size_t class_index) {
  if (workspace == nullptr) {
    throw std::invalid_argument("GPU histogram workspace must not be null");
  }
  if (!workspace->multitarget_enabled) {
    if (class_index != 0) {
      throw std::invalid_argument("GPU histogram single-target workspace only supports class_index 0");
    }
    workspace->active_target_index = 0;
    return;
  }
  if (class_index >= workspace->target_stride) {
    throw std::invalid_argument("GPU histogram class_index is out of range for the active target stride");
  }
  workspace->active_target_index = class_index;
}

void BuildHistogramsGpu(GpuHistogramWorkspace* workspace,
                        const std::vector<std::size_t>& row_indices,
                        std::size_t row_begin,
                        std::size_t row_end,
                        std::vector<float>& out_gradient_sums,
                        std::vector<float>& out_hessian_sums,
                        std::vector<float>& out_weight_sums,
                        std::vector<std::size_t>& out_feature_offsets,
                        GpuNodeStatistics* out_node_stats) {
  if (workspace == nullptr) {
    throw std::invalid_argument("GPU histogram workspace must not be null");
  }

  out_feature_offsets = workspace->feature_offsets;
  out_gradient_sums.assign(workspace->total_bins, 0.0F);
  out_hessian_sums.assign(workspace->total_bins, 0.0F);
  out_weight_sums.assign(workspace->total_bins, 0.0F);
  if (out_node_stats != nullptr) {
    *out_node_stats = GpuNodeStatistics{};
  }

  if (row_begin > row_end || row_end > row_indices.size()) {
    throw std::invalid_argument("GPU histogram row range is out of bounds");
  }
  const std::size_t row_count = row_end - row_begin;
  if (row_count == 0 || workspace->num_features == 0 || workspace->total_bins == 0) {
    return;
  }

  try {
    CopyHostSliceToDevice(row_indices, row_begin, row_end, workspace->row_indices);
    thrust::fill(workspace->gradient_sums.begin(), workspace->gradient_sums.end(), 0.0F);
    thrust::fill(workspace->hessian_sums.begin(), workspace->hessian_sums.end(), 0.0F);
    thrust::fill(workspace->weight_sums.begin(), workspace->weight_sums.end(), 0.0F);
    thrust::fill(workspace->node_statistics.begin(), workspace->node_statistics.end(), 0.0);

    const float* gradients = ResolveGradientPointer(*workspace);
    const float* hessians = ResolveHessianPointer(*workspace);
    const std::size_t target_stride = ResolveTargetStride(*workspace);
    const std::size_t target_offset = ResolveTargetOffset(*workspace);

    const int statistics_blocks =
        std::max<int>(1, static_cast<int>((row_count + kHistogramThreads - 1) / kHistogramThreads));
    NodeTargetStatisticsKernel<<<statistics_blocks, kHistogramThreads>>>(
        thrust::raw_pointer_cast(workspace->row_indices.data()),
        gradients,
        hessians,
        thrust::raw_pointer_cast(workspace->weights.data()),
        thrust::raw_pointer_cast(workspace->node_statistics.data()),
        row_count,
        target_stride,
        target_offset);

    const unsigned int row_tiles = static_cast<unsigned int>(
        (row_count + kHistogramRowTileSize - 1) / kHistogramRowTileSize);
    const dim3 grid(row_tiles, static_cast<unsigned int>(workspace->chunk_feature_indices.size()));
    HistMatrixFeatureChunksKernel<<<grid, kHistogramThreads>>>(
        thrust::raw_pointer_cast(workspace->bins.data()),
        thrust::raw_pointer_cast(workspace->row_indices.data()),
        gradients,
        hessians,
        thrust::raw_pointer_cast(workspace->weights.data()),
        thrust::raw_pointer_cast(workspace->chunk_feature_indices.data()),
        thrust::raw_pointer_cast(workspace->chunk_bin_starts.data()),
        thrust::raw_pointer_cast(workspace->chunk_bin_counts.data()),
        thrust::raw_pointer_cast(workspace->chunk_output_offsets.data()),
        thrust::raw_pointer_cast(workspace->gradient_sums.data()),
        thrust::raw_pointer_cast(workspace->hessian_sums.data()),
        thrust::raw_pointer_cast(workspace->weight_sums.data()),
        row_count,
        workspace->num_rows,
        target_stride,
        target_offset);

    CTBOOST_CUDA_CHECK(cudaGetLastError());
    CTBOOST_CUDA_CHECK(cudaDeviceSynchronize());

    thrust::copy(
        workspace->gradient_sums.begin(), workspace->gradient_sums.end(), out_gradient_sums.begin());
    thrust::copy(
        workspace->hessian_sums.begin(), workspace->hessian_sums.end(), out_hessian_sums.begin());
    thrust::copy(workspace->weight_sums.begin(), workspace->weight_sums.end(), out_weight_sums.begin());
    if (out_node_stats != nullptr) {
      std::vector<double> host_node_stats(workspace->node_statistics.size(), 0.0);
      thrust::copy(
          workspace->node_statistics.begin(), workspace->node_statistics.end(), host_node_stats.begin());
      out_node_stats->sample_weight_sum = host_node_stats[0];
      out_node_stats->total_gradient = host_node_stats[1];
      out_node_stats->total_hessian = host_node_stats[2];
      out_node_stats->gradient_square_sum = host_node_stats[3];
    }
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
