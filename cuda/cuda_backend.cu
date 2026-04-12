#include "ctboost/cuda_backend.hpp"
#include "ctboost/histogram.hpp"
#include "hist_kernels.cuh"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/partition.h>
#include <thrust/sequence.h>
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

__host__ __device__ __forceinline__ std::uint16_t ReadWorkspaceBin(const std::uint8_t* bins_u8,
                                                                   const std::uint16_t* bins_u16,
                                                                   std::uint8_t bin_index_bytes,
                                                                   std::size_t index) {
  return bin_index_bytes == 1 ? static_cast<std::uint16_t>(bins_u8[index]) : bins_u16[index];
}

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

struct RowSplitPredicate {
  const std::uint8_t* bins_u8{nullptr};
  const std::uint16_t* bins_u16{nullptr};
  std::uint8_t bin_index_bytes{2};
  std::size_t num_rows{0};
  std::size_t feature_index{0};
  bool is_categorical{false};
  std::uint16_t split_bin{0};
  std::uint8_t left_categories[kGpuCategoricalRouteBins]{};

  __host__ __device__ bool operator()(const std::size_t row) const {
    const std::uint16_t bin = ReadWorkspaceBin(
        bins_u8, bins_u16, bin_index_bytes, feature_index * num_rows + row);
    return is_categorical ? left_categories[bin] != 0 : bin <= split_bin;
  }
};

}  // namespace

struct GpuHistogramWorkspace {
  std::size_t num_rows{0};
  std::size_t num_features{0};
  std::size_t total_bins{0};
  std::size_t max_feature_bins{0};
  std::size_t histogram_chunk_bins{kHistogramChunkBins};
  std::uint8_t bin_index_bytes{2};
  std::vector<std::size_t> feature_offsets;
  thrust::device_vector<std::uint8_t> bins_u8;
  thrust::device_vector<std::uint16_t> bins_u16;
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
  thrust::device_vector<std::uint32_t> feature_offsets_u32;
  thrust::device_vector<std::uint16_t> num_bins_per_feature;
  thrust::device_vector<std::uint8_t> categorical_mask;
  thrust::device_vector<GpuFeatureSearchResult> feature_search_results;
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
  const std::size_t expected_bin_count = hist.num_rows * hist.num_cols;
  if (hist.uses_compact_bin_storage()) {
    if (hist.compact_bin_indices.size() != expected_bin_count) {
      throw std::invalid_argument("GPU histogram compact bins must have num_rows * num_cols elements");
    }
  } else if (hist.bin_indices.size() != expected_bin_count) {
    throw std::invalid_argument("GPU histogram bins must have num_rows * num_cols elements");
  }

  try {
    auto workspace = std::make_unique<GpuHistogramWorkspace>();
    workspace->num_rows = hist.num_rows;
    workspace->num_features = hist.num_cols;
    workspace->bin_index_bytes = hist.bin_storage_bytes();
    workspace->feature_offsets = BuildFeatureOffsets(hist.num_bins_per_feature);
    workspace->total_bins = workspace->feature_offsets.empty() ? 0 : workspace->feature_offsets.back();
    for (const std::uint16_t feature_bins : hist.num_bins_per_feature) {
      workspace->max_feature_bins =
          std::max(workspace->max_feature_bins, static_cast<std::size_t>(feature_bins));
    }

    if (hist.uses_compact_bin_storage()) {
      CopyHostVectorToDevice(hist.compact_bin_indices, workspace->bins_u8);
      workspace->bins_u16.clear();
    } else {
      CopyHostVectorToDevice(hist.bin_indices, workspace->bins_u16);
      workspace->bins_u8.clear();
    }
    CopyHostVectorToDevice(weights, workspace->weights);
    workspace->row_indices.resize(hist.num_rows);
    thrust::sequence(workspace->row_indices.begin(), workspace->row_indices.end(), std::size_t{0});
    workspace->gradients.resize(hist.num_rows, 0.0F);
    workspace->hessians.resize(hist.num_rows, 0.0F);
    workspace->gradient_sums.resize(workspace->total_bins, 0.0F);
    workspace->hessian_sums.resize(workspace->total_bins, 0.0F);
    workspace->weight_sums.resize(workspace->total_bins, 0.0F);
    workspace->node_statistics.resize(4, 0.0);
    workspace->feature_search_results.resize(hist.num_cols);

    std::vector<std::uint32_t> feature_offsets_u32(workspace->feature_offsets.size(), 0U);
    for (std::size_t index = 0; index < workspace->feature_offsets.size(); ++index) {
      feature_offsets_u32[index] = static_cast<std::uint32_t>(workspace->feature_offsets[index]);
    }
    CopyHostVectorToDevice(feature_offsets_u32, workspace->feature_offsets_u32);
    CopyHostVectorToDevice(hist.num_bins_per_feature, workspace->num_bins_per_feature);
    CopyHostVectorToDevice(hist.categorical_mask, workspace->categorical_mask);

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

std::size_t EstimateGpuHistogramWorkspaceBytes(const GpuHistogramWorkspace* workspace) noexcept {
  if (workspace == nullptr) {
    return 0;
  }

  return workspace->bins_u8.size() * sizeof(std::uint8_t) +
         workspace->bins_u16.size() * sizeof(std::uint16_t) +
         workspace->weights.size() * sizeof(float) +
         workspace->gradients.size() * sizeof(float) +
         workspace->hessians.size() * sizeof(float) +
         workspace->multitarget_gradients.size() * sizeof(float) +
         workspace->multitarget_hessians.size() * sizeof(float) +
         workspace->row_indices.size() * sizeof(std::size_t) +
         workspace->gradient_sums.size() * sizeof(float) +
         workspace->hessian_sums.size() * sizeof(float) +
         workspace->weight_sums.size() * sizeof(float) +
         workspace->node_statistics.size() * sizeof(double) +
         workspace->feature_offsets_u32.size() * sizeof(std::uint32_t) +
         workspace->num_bins_per_feature.size() * sizeof(std::uint16_t) +
         workspace->categorical_mask.size() * sizeof(std::uint8_t) +
         workspace->feature_search_results.size() * sizeof(GpuFeatureSearchResult) +
         workspace->chunk_feature_indices.size() * sizeof(std::uint32_t) +
         workspace->chunk_bin_starts.size() * sizeof(std::uint32_t) +
         workspace->chunk_bin_counts.size() * sizeof(std::uint32_t) +
         workspace->chunk_output_offsets.size() * sizeof(std::uint32_t);
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

void ResetHistogramRowIndicesGpu(GpuHistogramWorkspace* workspace) {
  if (workspace == nullptr) {
    throw std::invalid_argument("GPU histogram workspace must not be null");
  }

  try {
    workspace->row_indices.resize(workspace->num_rows);
    thrust::sequence(workspace->row_indices.begin(), workspace->row_indices.end(), std::size_t{0});
  } catch (const thrust::system_error& error) {
    throw std::runtime_error(std::string("CUDA thrust failure: ") + error.what());
  }
}

void DownloadHistogramRowIndicesGpu(const GpuHistogramWorkspace* workspace,
                                    std::vector<std::size_t>& out_row_indices) {
  if (workspace == nullptr) {
    throw std::invalid_argument("GPU histogram workspace must not be null");
  }

  try {
    out_row_indices.resize(workspace->num_rows);
    thrust::copy(
        workspace->row_indices.begin(), workspace->row_indices.end(), out_row_indices.begin());
  } catch (const thrust::system_error& error) {
    throw std::runtime_error(std::string("CUDA thrust failure: ") + error.what());
  }
}

std::size_t PartitionHistogramRowsGpu(
    GpuHistogramWorkspace* workspace,
    std::size_t row_begin,
    std::size_t row_end,
    std::size_t feature_index,
    bool is_categorical,
    std::uint16_t split_bin,
    const std::array<std::uint8_t, kGpuCategoricalRouteBins>& left_categories) {
  if (workspace == nullptr) {
    throw std::invalid_argument("GPU histogram workspace must not be null");
  }
  if (feature_index >= workspace->num_features) {
    throw std::invalid_argument("GPU histogram feature index is out of bounds");
  }
  if (row_begin > row_end || row_end > workspace->num_rows) {
    throw std::invalid_argument("GPU histogram row range is out of bounds");
  }
  if (row_begin == row_end) {
    return row_begin;
  }

  try {
    RowSplitPredicate predicate;
    predicate.bins_u8 = workspace->bins_u8.empty() ? nullptr : thrust::raw_pointer_cast(workspace->bins_u8.data());
    predicate.bins_u16 =
        workspace->bins_u16.empty() ? nullptr : thrust::raw_pointer_cast(workspace->bins_u16.data());
    predicate.bin_index_bytes = workspace->bin_index_bytes;
    predicate.num_rows = workspace->num_rows;
    predicate.feature_index = feature_index;
    predicate.is_categorical = is_categorical;
    predicate.split_bin = split_bin;
    std::copy(left_categories.begin(), left_categories.end(), predicate.left_categories);

    auto begin = workspace->row_indices.begin() + static_cast<std::ptrdiff_t>(row_begin);
    auto end = workspace->row_indices.begin() + static_cast<std::ptrdiff_t>(row_end);
    auto middle = thrust::stable_partition(begin, end, predicate);
    return row_begin + static_cast<std::size_t>(std::distance(begin, middle));
  } catch (const thrust::system_error& error) {
    throw std::runtime_error(std::string("CUDA thrust failure: ") + error.what());
  }
}

void BuildHistogramsGpu(GpuHistogramWorkspace* workspace,
                        std::size_t row_begin,
                        std::size_t row_end,
                        GpuNodeStatistics* out_node_stats) {
  if (workspace == nullptr) {
    throw std::invalid_argument("GPU histogram workspace must not be null");
  }
  if (out_node_stats != nullptr) {
    *out_node_stats = GpuNodeStatistics{};
  }

  if (row_begin > row_end || row_end > workspace->num_rows) {
    throw std::invalid_argument("GPU histogram row range is out of bounds");
  }
  const std::size_t row_count = row_end - row_begin;
  if (row_count == 0 || workspace->num_features == 0 || workspace->total_bins == 0) {
    return;
  }

  try {
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
        thrust::raw_pointer_cast(workspace->row_indices.data()) + row_begin,
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
        workspace->bins_u8.empty() ? nullptr : thrust::raw_pointer_cast(workspace->bins_u8.data()),
        workspace->bins_u16.empty() ? nullptr : thrust::raw_pointer_cast(workspace->bins_u16.data()),
        workspace->bin_index_bytes,
        thrust::raw_pointer_cast(workspace->row_indices.data()) + row_begin,
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

void SearchBestNodeSplitGpu(GpuHistogramWorkspace* workspace,
                            const std::vector<int>* allowed_features,
                            double lambda_l2,
                            int min_data_in_leaf,
                            double min_child_weight,
                            double min_split_gain,
                            GpuNodeSearchResult* out_result) {
  if (workspace == nullptr) {
    throw std::invalid_argument("GPU histogram workspace must not be null");
  }
  if (out_result == nullptr) {
    throw std::invalid_argument("GPU node search result must not be null");
  }

  try {
    if (workspace->num_features == 0) {
      *out_result = GpuNodeSearchResult{};
      return;
    }
    const int blocks = static_cast<int>(
        (workspace->num_features + kHistogramThreads - 1) / kHistogramThreads);
    std::vector<double> host_node_stats(workspace->node_statistics.size(), 0.0);
    thrust::copy(
        workspace->node_statistics.begin(), workspace->node_statistics.end(), host_node_stats.begin());
    out_result->node_statistics.sample_weight_sum = host_node_stats[0];
    out_result->node_statistics.total_gradient = host_node_stats[1];
    out_result->node_statistics.total_hessian = host_node_stats[2];
    out_result->node_statistics.gradient_square_sum = host_node_stats[3];

    const double sample_weight_sum = out_result->node_statistics.sample_weight_sum;
    const double mean_gradient = sample_weight_sum <= 0.0
                                     ? 0.0
                                     : out_result->node_statistics.total_gradient / sample_weight_sum;
    const double gradient_variance =
        sample_weight_sum <= 0.0
            ? 0.0
            : std::max(0.0,
                       out_result->node_statistics.gradient_square_sum / sample_weight_sum -
                           mean_gradient * mean_gradient);

    EvaluateFeatureSearchKernel<<<blocks, kHistogramThreads>>>(
        thrust::raw_pointer_cast(workspace->gradient_sums.data()),
        thrust::raw_pointer_cast(workspace->hessian_sums.data()),
        thrust::raw_pointer_cast(workspace->weight_sums.data()),
        thrust::raw_pointer_cast(workspace->feature_offsets_u32.data()),
        thrust::raw_pointer_cast(workspace->num_bins_per_feature.data()),
        thrust::raw_pointer_cast(workspace->categorical_mask.data()),
        out_result->node_statistics.total_gradient,
        out_result->node_statistics.total_hessian,
        out_result->node_statistics.sample_weight_sum,
        gradient_variance,
        lambda_l2,
        min_data_in_leaf,
        min_child_weight,
        min_split_gain,
        thrust::raw_pointer_cast(workspace->feature_search_results.data()),
        workspace->num_features);
    CTBOOST_CUDA_CHECK(cudaGetLastError());
    CTBOOST_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<GpuFeatureSearchResult> host_results(workspace->num_features);
    thrust::copy(workspace->feature_search_results.begin(),
                 workspace->feature_search_results.end(),
                 host_results.begin());

    out_result->feature_id = -1;
    out_result->p_value = 1.0;
    out_result->chi_square = -std::numeric_limits<double>::infinity();
    out_result->split_valid = false;
    out_result->is_categorical = false;
    out_result->split_bin = 0;
    out_result->gain = 0.0;
    out_result->left_categories.fill(0);

    const auto consider_feature = [&](std::size_t feature_index) {
      const GpuFeatureSearchResult& result = host_results[feature_index];
      if (result.degrees_of_freedom == 0) {
        return;
      }
      if (out_result->feature_id < 0 || result.p_value < out_result->p_value ||
          (std::abs(result.p_value - out_result->p_value) <= 1e-12 &&
           result.chi_square > out_result->chi_square)) {
        out_result->feature_id = static_cast<int>(feature_index);
        out_result->p_value = result.p_value;
        out_result->chi_square = result.chi_square;
        out_result->split_valid = result.split_valid != 0U;
        out_result->is_categorical = result.is_categorical != 0U;
        out_result->split_bin = result.split_bin;
        out_result->gain = result.gain;
        std::copy(result.left_categories,
                  result.left_categories + kGpuCategoricalRouteBins,
                  out_result->left_categories.begin());
      }
    };

    if (allowed_features != nullptr && !allowed_features->empty()) {
      for (int feature_id : *allowed_features) {
        if (feature_id < 0 || static_cast<std::size_t>(feature_id) >= workspace->num_features) {
          continue;
        }
        consider_feature(static_cast<std::size_t>(feature_id));
      }
      return;
    }

    for (std::size_t feature = 0; feature < workspace->num_features; ++feature) {
      consider_feature(feature);
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
  const std::size_t expected_bin_count = hist.num_rows * hist.num_cols;
  if (hist.uses_compact_bin_storage()) {
    if (hist.compact_bin_indices.size() != expected_bin_count) {
      throw std::invalid_argument("GPU prediction compact bins must have num_rows * num_cols elements");
    }
  } else if (hist.bin_indices.size() != expected_bin_count) {
    throw std::invalid_argument("GPU prediction bins must have num_rows * num_cols elements");
  }
  if (out_predictions.size() != hist.num_rows * static_cast<std::size_t>(prediction_dimension)) {
    throw std::invalid_argument("GPU prediction output buffer has an unexpected size");
  }

  try {
    thrust::device_vector<std::uint8_t> device_bins_u8;
    thrust::device_vector<std::uint16_t> device_bins_u16;
    if (hist.uses_compact_bin_storage()) {
      device_bins_u8.assign(hist.compact_bin_indices.begin(), hist.compact_bin_indices.end());
    } else {
      device_bins_u16.assign(hist.bin_indices.begin(), hist.bin_indices.end());
    }
    thrust::device_vector<GpuTreeNode> device_nodes(nodes.begin(), nodes.end());
    thrust::device_vector<std::int32_t> device_tree_offsets(tree_offsets.begin(), tree_offsets.end());
    thrust::device_vector<float> device_predictions(out_predictions.size(), 0.0F);

    const int blocks =
        static_cast<int>((hist.num_rows + kPredictionThreads - 1) / kPredictionThreads);
    PredictForestKernel<<<blocks, kPredictionThreads>>>(
        device_bins_u8.empty() ? nullptr : thrust::raw_pointer_cast(device_bins_u8.data()),
        device_bins_u16.empty() ? nullptr : thrust::raw_pointer_cast(device_bins_u16.data()),
        hist.bin_storage_bytes(),
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
