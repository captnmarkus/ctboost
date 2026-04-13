#include "ctboost/cuda_backend.hpp"
#include "ctboost/histogram.hpp"
#include "hist_kernels.cuh"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cctype>
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

class DeviceGuard {
 public:
  explicit DeviceGuard(int device_id) {
    CTBOOST_CUDA_CHECK(cudaGetDevice(&previous_device_));
    CTBOOST_CUDA_CHECK(cudaSetDevice(device_id));
  }

  ~DeviceGuard() noexcept { cudaSetDevice(previous_device_); }

 private:
  int previous_device_{0};
};

struct DeviceWorkspace {
  int device_id{0};
  std::vector<std::size_t> assigned_features;
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
  thrust::device_vector<GpuBestFeatureResult> best_feature_result;
  thrust::device_vector<std::uint32_t> chunk_feature_indices;
  thrust::device_vector<std::uint32_t> chunk_bin_starts;
  thrust::device_vector<std::uint32_t> chunk_bin_counts;
  thrust::device_vector<std::uint32_t> chunk_output_offsets;

  ~DeviceWorkspace() noexcept { cudaSetDevice(device_id); }
};

struct GpuHistogramWorkspace {
  std::size_t num_rows{0};
  std::size_t num_features{0};
  std::size_t total_bins{0};
  std::size_t max_feature_bins{0};
  std::size_t histogram_chunk_bins{kHistogramChunkBins};
  std::uint8_t bin_index_bytes{2};
  std::vector<std::size_t> feature_offsets;
  std::vector<int> device_ids;
  std::vector<DeviceWorkspace> devices;
};

namespace {

DeviceWorkspace& PrimaryDeviceWorkspace(GpuHistogramWorkspace* workspace) {
  return workspace->devices.front();
}

const DeviceWorkspace& PrimaryDeviceWorkspace(const GpuHistogramWorkspace* workspace) {
  return workspace->devices.front();
}

const float* ResolveGradientPointer(const DeviceWorkspace& workspace) {
  return workspace.multitarget_enabled
             ? thrust::raw_pointer_cast(workspace.multitarget_gradients.data())
             : thrust::raw_pointer_cast(workspace.gradients.data());
}

const float* ResolveHessianPointer(const DeviceWorkspace& workspace) {
  return workspace.multitarget_enabled
             ? thrust::raw_pointer_cast(workspace.multitarget_hessians.data())
             : thrust::raw_pointer_cast(workspace.hessians.data());
}

std::size_t ResolveTargetStride(const DeviceWorkspace& workspace) {
  return workspace.multitarget_enabled ? workspace.target_stride : 1U;
}

std::size_t ResolveTargetOffset(const DeviceWorkspace& workspace) {
  return workspace.multitarget_enabled ? workspace.active_target_index : 0U;
}

std::vector<int> ParseDeviceList(const std::string& devices) {
  std::vector<int> parsed_devices;
  std::string token;
  const auto flush_token = [&]() {
    if (token.empty()) {
      return;
    }
    const int device_id = std::stoi(token);
    if (device_id < 0) {
      throw std::invalid_argument("devices must contain only non-negative CUDA device ids");
    }
    if (std::find(parsed_devices.begin(), parsed_devices.end(), device_id) == parsed_devices.end()) {
      parsed_devices.push_back(device_id);
    }
    token.clear();
  };

  for (const char ch : devices) {
    if (std::isdigit(static_cast<unsigned char>(ch)) != 0) {
      token.push_back(ch);
      continue;
    }
    if (ch == ',' || ch == ';' || std::isspace(static_cast<unsigned char>(ch)) != 0) {
      flush_token();
      continue;
    }
    throw std::invalid_argument(
        "devices must be a comma-separated list of non-negative CUDA device ids");
  }
  flush_token();
  if (parsed_devices.empty()) {
    parsed_devices.push_back(0);
  }
  return parsed_devices;
}

std::vector<int> ResolveRequestedDevices(const std::string& devices) {
  int device_count = 0;
  CTBOOST_CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count <= 0) {
    throw std::runtime_error("no CUDA devices are available for GPU training");
  }

  std::vector<int> device_ids = ParseDeviceList(devices);
  for (const int device_id : device_ids) {
    if (device_id < 0 || device_id >= device_count) {
      throw std::invalid_argument("devices contains a CUDA device id that is not available");
    }
  }
  return device_ids;
}

std::vector<std::vector<std::size_t>> AssignFeaturesToDevices(const HistMatrix& hist,
                                                              std::size_t num_devices) {
  std::vector<std::vector<std::size_t>> assignments(num_devices);
  std::vector<std::size_t> device_loads(num_devices, 0U);
  for (std::size_t feature = 0; feature < hist.num_cols; ++feature) {
    const std::size_t device_index = static_cast<std::size_t>(
        std::distance(device_loads.begin(),
                      std::min_element(device_loads.begin(), device_loads.end())));
    assignments[device_index].push_back(feature);
    device_loads[device_index] += std::max<std::size_t>(1U, hist.num_bins(feature));
  }
  return assignments;
}

std::size_t EstimateDeviceWorkspaceBytes(const DeviceWorkspace& workspace) noexcept {
  return workspace.bins_u8.size() * sizeof(std::uint8_t) +
         workspace.bins_u16.size() * sizeof(std::uint16_t) +
         workspace.weights.size() * sizeof(float) +
         workspace.gradients.size() * sizeof(float) +
         workspace.hessians.size() * sizeof(float) +
         workspace.multitarget_gradients.size() * sizeof(float) +
         workspace.multitarget_hessians.size() * sizeof(float) +
         workspace.row_indices.size() * sizeof(std::size_t) +
         workspace.gradient_sums.size() * sizeof(float) +
         workspace.hessian_sums.size() * sizeof(float) +
         workspace.weight_sums.size() * sizeof(float) +
         workspace.node_statistics.size() * sizeof(double) +
         workspace.feature_offsets_u32.size() * sizeof(std::uint32_t) +
         workspace.num_bins_per_feature.size() * sizeof(std::uint16_t) +
         workspace.categorical_mask.size() * sizeof(std::uint8_t) +
         workspace.feature_search_results.size() * sizeof(GpuFeatureSearchResult) +
         workspace.best_feature_result.size() * sizeof(GpuBestFeatureResult) +
         workspace.chunk_feature_indices.size() * sizeof(std::uint32_t) +
         workspace.chunk_bin_starts.size() * sizeof(std::uint32_t) +
         workspace.chunk_bin_counts.size() * sizeof(std::uint32_t) +
         workspace.chunk_output_offsets.size() * sizeof(std::uint32_t);
}

std::array<double, 4> DownloadNodeStatistics(const DeviceWorkspace& workspace) {
  std::vector<double> host_node_stats(workspace.node_statistics.size(), 0.0);
  thrust::copy(workspace.node_statistics.begin(), workspace.node_statistics.end(), host_node_stats.begin());
  return {
      host_node_stats.size() > 0 ? host_node_stats[0] : 0.0,
      host_node_stats.size() > 1 ? host_node_stats[1] : 0.0,
      host_node_stats.size() > 2 ? host_node_stats[2] : 0.0,
      host_node_stats.size() > 3 ? host_node_stats[3] : 0.0,
  };
}

void UploadNodeStatistics(DeviceWorkspace& workspace, const std::array<double, 4>& node_statistics) {
  std::vector<double> host_node_stats(node_statistics.begin(), node_statistics.end());
  thrust::copy(host_node_stats.begin(), host_node_stats.end(), workspace.node_statistics.begin());
}

std::vector<std::uint32_t> CandidateFeaturesForDevice(const DeviceWorkspace& workspace,
                                                      const std::vector<int>* allowed_features,
                                                      std::size_t num_features) {
  std::vector<std::uint32_t> host_allowed_features;
  if (allowed_features != nullptr && !allowed_features->empty()) {
    host_allowed_features.reserve(allowed_features->size());
    for (const int feature_id : *allowed_features) {
      if (feature_id < 0 || static_cast<std::size_t>(feature_id) >= num_features) {
        continue;
      }
      if (std::find(workspace.assigned_features.begin(),
                    workspace.assigned_features.end(),
                    static_cast<std::size_t>(feature_id)) == workspace.assigned_features.end()) {
        continue;
      }
      host_allowed_features.push_back(static_cast<std::uint32_t>(feature_id));
    }
    return host_allowed_features;
  }

  host_allowed_features.reserve(workspace.assigned_features.size());
  for (const std::size_t feature_id : workspace.assigned_features) {
    host_allowed_features.push_back(static_cast<std::uint32_t>(feature_id));
  }
  return host_allowed_features;
}

bool IsBetterHostFeatureResult(const GpuBestFeatureResult& candidate,
                               const GpuBestFeatureResult& best) {
  if (candidate.feature_id < 0 || candidate.search_result.degrees_of_freedom == 0U) {
    return false;
  }
  if (best.feature_id < 0 || best.search_result.degrees_of_freedom == 0U) {
    return true;
  }
  if (candidate.search_result.p_value < best.search_result.p_value) {
    return true;
  }
  if (std::fabs(candidate.search_result.p_value - best.search_result.p_value) <= 1e-12 &&
      candidate.search_result.chi_square > best.search_result.chi_square) {
    return true;
  }
  return std::fabs(candidate.search_result.p_value - best.search_result.p_value) <= 1e-12 &&
         std::fabs(candidate.search_result.chi_square - best.search_result.chi_square) <= 1e-12 &&
         candidate.feature_id < best.feature_id;
}

void InitializeDeviceWorkspace(DeviceWorkspace& device_workspace,
                               const HistMatrix& hist,
                               const std::vector<float>& weights,
                               const std::vector<std::uint32_t>& feature_offsets_u32,
                               const std::vector<std::size_t>& feature_offsets,
                               std::size_t histogram_chunk_bins) {
  if (hist.uses_compact_bin_storage()) {
    CopyHostVectorToDevice(hist.compact_bin_indices, device_workspace.bins_u8);
    device_workspace.bins_u16.clear();
  } else {
    CopyHostVectorToDevice(hist.bin_indices, device_workspace.bins_u16);
    device_workspace.bins_u8.clear();
  }
  CopyHostVectorToDevice(weights, device_workspace.weights);
  device_workspace.row_indices.resize(hist.num_rows);
  thrust::sequence(device_workspace.row_indices.begin(),
                   device_workspace.row_indices.end(),
                   std::size_t{0});
  device_workspace.gradients.resize(hist.num_rows, 0.0F);
  device_workspace.hessians.resize(hist.num_rows, 0.0F);
  device_workspace.gradient_sums.resize(feature_offsets.back(), 0.0F);
  device_workspace.hessian_sums.resize(feature_offsets.back(), 0.0F);
  device_workspace.weight_sums.resize(feature_offsets.back(), 0.0F);
  device_workspace.node_statistics.resize(4, 0.0);
  device_workspace.feature_search_results.resize(hist.num_cols);
  device_workspace.best_feature_result.resize(1);
  CopyHostVectorToDevice(feature_offsets_u32, device_workspace.feature_offsets_u32);
  CopyHostVectorToDevice(hist.num_bins_per_feature, device_workspace.num_bins_per_feature);
  CopyHostVectorToDevice(hist.categorical_mask, device_workspace.categorical_mask);

  std::vector<std::uint32_t> chunk_feature_indices;
  std::vector<std::uint32_t> chunk_bin_starts;
  std::vector<std::uint32_t> chunk_bin_counts;
  std::vector<std::uint32_t> chunk_output_offsets;
  for (const std::size_t feature : device_workspace.assigned_features) {
    const std::size_t feature_bin_count = hist.num_bins(feature);
    for (std::size_t bin_start = 0; bin_start < feature_bin_count; bin_start += histogram_chunk_bins) {
      const std::size_t chunk_bins =
          std::min(histogram_chunk_bins, feature_bin_count - bin_start);
      chunk_feature_indices.push_back(static_cast<std::uint32_t>(feature));
      chunk_bin_starts.push_back(static_cast<std::uint32_t>(bin_start));
      chunk_bin_counts.push_back(static_cast<std::uint32_t>(chunk_bins));
      chunk_output_offsets.push_back(static_cast<std::uint32_t>(feature_offsets[feature]));
    }
  }
  CopyHostVectorToDevice(chunk_feature_indices, device_workspace.chunk_feature_indices);
  CopyHostVectorToDevice(chunk_bin_starts, device_workspace.chunk_bin_starts);
  CopyHostVectorToDevice(chunk_bin_counts, device_workspace.chunk_bin_counts);
  CopyHostVectorToDevice(chunk_output_offsets, device_workspace.chunk_output_offsets);
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
                                                     const std::vector<float>& weights,
                                                     const std::string& devices) {
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
    workspace->total_bins =
        workspace->feature_offsets.empty() ? 0 : workspace->feature_offsets.back();
    for (const std::uint16_t feature_bins : hist.num_bins_per_feature) {
      workspace->max_feature_bins =
          std::max(workspace->max_feature_bins, static_cast<std::size_t>(feature_bins));
    }

    std::vector<std::uint32_t> feature_offsets_u32(workspace->feature_offsets.size(), 0U);
    for (std::size_t index = 0; index < workspace->feature_offsets.size(); ++index) {
      feature_offsets_u32[index] = static_cast<std::uint32_t>(workspace->feature_offsets[index]);
    }

    std::vector<int> requested_devices = ResolveRequestedDevices(devices);
    std::vector<std::vector<std::size_t>> feature_assignments =
        AssignFeaturesToDevices(hist, requested_devices.size());
    for (std::size_t device_index = 0; device_index < requested_devices.size(); ++device_index) {
      if (feature_assignments[device_index].empty() && hist.num_cols > 0) {
        continue;
      }
      workspace->device_ids.push_back(requested_devices[device_index]);
      workspace->devices.emplace_back();
      DeviceWorkspace& device_workspace = workspace->devices.back();
      device_workspace.device_id = requested_devices[device_index];
      device_workspace.assigned_features = std::move(feature_assignments[device_index]);
      DeviceGuard device_guard(device_workspace.device_id);
      InitializeDeviceWorkspace(device_workspace,
                                hist,
                                weights,
                                feature_offsets_u32,
                                workspace->feature_offsets,
                                workspace->histogram_chunk_bins);
    }
    if (workspace->devices.empty()) {
      throw std::runtime_error(
          "GPU histogram workspace could not assign features to any CUDA device");
    }

    return GpuHistogramWorkspacePtr(workspace.release(), DestroyGpuHistogramWorkspace);
  } catch (const thrust::system_error& error) {
    throw std::runtime_error(std::string("CUDA thrust failure: ") + error.what());
  }
}

std::size_t EstimateGpuHistogramWorkspaceBytes(const GpuHistogramWorkspace* workspace) noexcept {
  if (workspace == nullptr) {
    return 0;
  }
  std::size_t total_bytes = 0;
  for (const DeviceWorkspace& device_workspace : workspace->devices) {
    total_bytes += EstimateDeviceWorkspaceBytes(device_workspace);
  }
  return total_bytes;
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
    for (DeviceWorkspace& device_workspace : workspace->devices) {
      DeviceGuard device_guard(device_workspace.device_id);
      CopyHostVectorToDevice(gradients, device_workspace.gradients);
      CopyHostVectorToDevice(hessians, device_workspace.hessians);
      device_workspace.multitarget_enabled = false;
      device_workspace.target_stride = 1;
      device_workspace.active_target_index = 0;
      device_workspace.multitarget_gradients.clear();
      device_workspace.multitarget_hessians.clear();
    }
  } catch (const thrust::system_error& error) {
    throw std::runtime_error(std::string("CUDA thrust failure: ") + error.what());
  }
}

void UploadHistogramWeightsGpu(GpuHistogramWorkspace* workspace,
                               const std::vector<float>& weights) {
  if (workspace == nullptr) {
    throw std::invalid_argument("GPU histogram workspace must not be null");
  }
  if (weights.size() != workspace->num_rows) {
    throw std::invalid_argument("GPU histogram weights must match the histogram row count");
  }

  try {
    for (DeviceWorkspace& device_workspace : workspace->devices) {
      DeviceGuard device_guard(device_workspace.device_id);
      CopyHostVectorToDevice(weights, device_workspace.weights);
    }
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
    for (DeviceWorkspace& device_workspace : workspace->devices) {
      DeviceGuard device_guard(device_workspace.device_id);
      CopyHostVectorToDevice(gradients, device_workspace.multitarget_gradients);
      CopyHostVectorToDevice(hessians, device_workspace.multitarget_hessians);
      device_workspace.multitarget_enabled = true;
      device_workspace.target_stride = target_stride;
      device_workspace.active_target_index = 0;
    }
  } catch (const thrust::system_error& error) {
    throw std::runtime_error(std::string("CUDA thrust failure: ") + error.what());
  }
}

void SelectHistogramTargetGpuClass(GpuHistogramWorkspace* workspace, std::size_t class_index) {
  if (workspace == nullptr) {
    throw std::invalid_argument("GPU histogram workspace must not be null");
  }
  if (!PrimaryDeviceWorkspace(workspace).multitarget_enabled) {
    if (class_index != 0) {
      throw std::invalid_argument("GPU histogram single-target workspace only supports class_index 0");
    }
    for (DeviceWorkspace& device_workspace : workspace->devices) {
      device_workspace.active_target_index = 0;
    }
    return;
  }
  if (class_index >= PrimaryDeviceWorkspace(workspace).target_stride) {
    throw std::invalid_argument("GPU histogram class_index is out of range for the active target stride");
  }
  for (DeviceWorkspace& device_workspace : workspace->devices) {
    device_workspace.active_target_index = class_index;
  }
}

void ResetHistogramRowIndicesGpu(GpuHistogramWorkspace* workspace) {
  if (workspace == nullptr) {
    throw std::invalid_argument("GPU histogram workspace must not be null");
  }

  try {
    for (DeviceWorkspace& device_workspace : workspace->devices) {
      DeviceGuard device_guard(device_workspace.device_id);
      device_workspace.row_indices.resize(workspace->num_rows);
      thrust::sequence(device_workspace.row_indices.begin(),
                       device_workspace.row_indices.end(),
                       std::size_t{0});
    }
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
    const DeviceWorkspace& primary_workspace = PrimaryDeviceWorkspace(workspace);
    DeviceGuard device_guard(primary_workspace.device_id);
    out_row_indices.resize(workspace->num_rows);
    thrust::copy(primary_workspace.row_indices.begin(),
                 primary_workspace.row_indices.end(),
                 out_row_indices.begin());
  } catch (const thrust::system_error& error) {
    throw std::runtime_error(std::string("CUDA thrust failure: ") + error.what());
  }
}

void DownloadHistogramSnapshotGpu(const GpuHistogramWorkspace* workspace,
                                  GpuHistogramSnapshot* out_snapshot) {
  if (workspace == nullptr) {
    throw std::invalid_argument("GPU histogram workspace must not be null");
  }
  if (out_snapshot == nullptr) {
    throw std::invalid_argument("GPU histogram snapshot output must not be null");
  }

  try {
    out_snapshot->gradient_sums.assign(workspace->total_bins, 0.0F);
    out_snapshot->hessian_sums.assign(workspace->total_bins, 0.0F);
    out_snapshot->weight_sums.assign(workspace->total_bins, 0.0F);
    for (std::size_t device_index = 0; device_index < workspace->devices.size(); ++device_index) {
      const DeviceWorkspace& device_workspace = workspace->devices[device_index];
      DeviceGuard device_guard(device_workspace.device_id);
      std::vector<float> device_gradient_sums(device_workspace.gradient_sums.size(), 0.0F);
      std::vector<float> device_hessian_sums(device_workspace.hessian_sums.size(), 0.0F);
      std::vector<float> device_weight_sums(device_workspace.weight_sums.size(), 0.0F);
      thrust::copy(device_workspace.gradient_sums.begin(),
                   device_workspace.gradient_sums.end(),
                   device_gradient_sums.begin());
      thrust::copy(device_workspace.hessian_sums.begin(),
                   device_workspace.hessian_sums.end(),
                   device_hessian_sums.begin());
      thrust::copy(device_workspace.weight_sums.begin(),
                   device_workspace.weight_sums.end(),
                   device_weight_sums.begin());
      for (std::size_t index = 0; index < workspace->total_bins; ++index) {
        out_snapshot->gradient_sums[index] += device_gradient_sums[index];
        out_snapshot->hessian_sums[index] += device_hessian_sums[index];
        out_snapshot->weight_sums[index] += device_weight_sums[index];
      }
      if (device_index == 0) {
        const std::array<double, 4> host_node_stats = DownloadNodeStatistics(device_workspace);
        out_snapshot->node_statistics.sample_weight_sum = host_node_stats[0];
        out_snapshot->node_statistics.total_gradient = host_node_stats[1];
        out_snapshot->node_statistics.total_hessian = host_node_stats[2];
        out_snapshot->node_statistics.gradient_square_sum = host_node_stats[3];
      }
    }
  } catch (const thrust::system_error& error) {
    throw std::runtime_error(std::string("CUDA thrust failure: ") + error.what());
  }
}

void UploadHistogramSnapshotGpu(GpuHistogramWorkspace* workspace,
                                const GpuHistogramSnapshot& snapshot) {
  if (workspace == nullptr) {
    throw std::invalid_argument("GPU histogram workspace must not be null");
  }
  if (snapshot.gradient_sums.size() != workspace->total_bins ||
      snapshot.hessian_sums.size() != workspace->total_bins ||
      snapshot.weight_sums.size() != workspace->total_bins) {
    throw std::invalid_argument("GPU histogram snapshot buffer sizes do not match the workspace");
  }

  try {
    const std::array<double, 4> host_node_stats{
        snapshot.node_statistics.sample_weight_sum,
        snapshot.node_statistics.total_gradient,
        snapshot.node_statistics.total_hessian,
        snapshot.node_statistics.gradient_square_sum,
    };
    for (DeviceWorkspace& device_workspace : workspace->devices) {
      DeviceGuard device_guard(device_workspace.device_id);
      thrust::copy(snapshot.gradient_sums.begin(),
                   snapshot.gradient_sums.end(),
                   device_workspace.gradient_sums.begin());
      thrust::copy(snapshot.hessian_sums.begin(),
                   snapshot.hessian_sums.end(),
                   device_workspace.hessian_sums.begin());
      thrust::copy(snapshot.weight_sums.begin(),
                   snapshot.weight_sums.end(),
                   device_workspace.weight_sums.begin());
      UploadNodeStatistics(device_workspace, host_node_stats);
    }
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
    std::size_t left_end = row_begin;
    bool left_end_initialized = false;
    for (DeviceWorkspace& device_workspace : workspace->devices) {
      DeviceGuard device_guard(device_workspace.device_id);
      RowSplitPredicate predicate;
      predicate.bins_u8 =
          device_workspace.bins_u8.empty() ? nullptr : thrust::raw_pointer_cast(device_workspace.bins_u8.data());
      predicate.bins_u16 =
          device_workspace.bins_u16.empty() ? nullptr : thrust::raw_pointer_cast(device_workspace.bins_u16.data());
      predicate.bin_index_bytes = workspace->bin_index_bytes;
      predicate.num_rows = workspace->num_rows;
      predicate.feature_index = feature_index;
      predicate.is_categorical = is_categorical;
      predicate.split_bin = split_bin;
      std::copy(left_categories.begin(), left_categories.end(), predicate.left_categories);

      auto begin = device_workspace.row_indices.begin() + static_cast<std::ptrdiff_t>(row_begin);
      auto end = device_workspace.row_indices.begin() + static_cast<std::ptrdiff_t>(row_end);
      auto middle = thrust::stable_partition(begin, end, predicate);
      const std::size_t device_left_end =
          row_begin + static_cast<std::size_t>(std::distance(begin, middle));
      if (!left_end_initialized) {
        left_end = device_left_end;
        left_end_initialized = true;
      } else if (device_left_end != left_end) {
        throw std::runtime_error("multi-GPU row partition produced inconsistent child boundaries");
      }
    }
    return left_end;
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
    for (std::size_t device_index = 0; device_index < workspace->devices.size(); ++device_index) {
      DeviceWorkspace& device_workspace = workspace->devices[device_index];
      DeviceGuard device_guard(device_workspace.device_id);
      thrust::fill(device_workspace.gradient_sums.begin(), device_workspace.gradient_sums.end(), 0.0F);
      thrust::fill(device_workspace.hessian_sums.begin(), device_workspace.hessian_sums.end(), 0.0F);
      thrust::fill(device_workspace.weight_sums.begin(), device_workspace.weight_sums.end(), 0.0F);
      thrust::fill(device_workspace.node_statistics.begin(), device_workspace.node_statistics.end(), 0.0);

      const float* gradients = ResolveGradientPointer(device_workspace);
      const float* hessians = ResolveHessianPointer(device_workspace);
      const std::size_t target_stride = ResolveTargetStride(device_workspace);
      const std::size_t target_offset = ResolveTargetOffset(device_workspace);

      const int statistics_blocks =
          std::max<int>(1, static_cast<int>((row_count + kHistogramThreads - 1) / kHistogramThreads));
      NodeTargetStatisticsKernel<<<statistics_blocks, kHistogramThreads>>>(
          thrust::raw_pointer_cast(device_workspace.row_indices.data()) + row_begin,
          gradients,
          hessians,
          thrust::raw_pointer_cast(device_workspace.weights.data()),
          thrust::raw_pointer_cast(device_workspace.node_statistics.data()),
          row_count,
          target_stride,
          target_offset);

      if (!device_workspace.chunk_feature_indices.empty()) {
        const unsigned int row_tiles = static_cast<unsigned int>(
            (row_count + kHistogramRowTileSize - 1) / kHistogramRowTileSize);
        const dim3 grid(row_tiles,
                        static_cast<unsigned int>(device_workspace.chunk_feature_indices.size()));
        HistMatrixFeatureChunksKernel<<<grid, kHistogramThreads>>>(
            device_workspace.bins_u8.empty() ? nullptr : thrust::raw_pointer_cast(device_workspace.bins_u8.data()),
            device_workspace.bins_u16.empty() ? nullptr : thrust::raw_pointer_cast(device_workspace.bins_u16.data()),
            workspace->bin_index_bytes,
            thrust::raw_pointer_cast(device_workspace.row_indices.data()) + row_begin,
            gradients,
            hessians,
            thrust::raw_pointer_cast(device_workspace.weights.data()),
            thrust::raw_pointer_cast(device_workspace.chunk_feature_indices.data()),
            thrust::raw_pointer_cast(device_workspace.chunk_bin_starts.data()),
            thrust::raw_pointer_cast(device_workspace.chunk_bin_counts.data()),
            thrust::raw_pointer_cast(device_workspace.chunk_output_offsets.data()),
            thrust::raw_pointer_cast(device_workspace.gradient_sums.data()),
            thrust::raw_pointer_cast(device_workspace.hessian_sums.data()),
            thrust::raw_pointer_cast(device_workspace.weight_sums.data()),
            row_count,
            workspace->num_rows,
            target_stride,
            target_offset);
      }

      CTBOOST_CUDA_CHECK(cudaGetLastError());
      CTBOOST_CUDA_CHECK(cudaDeviceSynchronize());
      if (out_node_stats != nullptr && device_index == 0) {
        const std::array<double, 4> host_node_stats = DownloadNodeStatistics(device_workspace);
        out_node_stats->sample_weight_sum = host_node_stats[0];
        out_node_stats->total_gradient = host_node_stats[1];
        out_node_stats->total_hessian = host_node_stats[2];
        out_node_stats->gradient_square_sum = host_node_stats[3];
      }
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
    const DeviceWorkspace& primary_workspace = PrimaryDeviceWorkspace(workspace);
    std::array<double, 4> host_node_stats{};
    {
      DeviceGuard device_guard(primary_workspace.device_id);
      host_node_stats = DownloadNodeStatistics(primary_workspace);
    }
    const double sample_weight_sum = host_node_stats[0];
    const double mean_gradient = sample_weight_sum <= 0.0
                                     ? 0.0
                                     : host_node_stats[1] / sample_weight_sum;
    const double gradient_variance =
        sample_weight_sum <= 0.0
            ? 0.0
            : std::max(0.0, host_node_stats[3] / sample_weight_sum - mean_gradient * mean_gradient);

    GpuBestFeatureResult host_best_result{};
    host_best_result.feature_id = -1;
    host_best_result.search_result.p_value = 1.0;
    host_best_result.search_result.chi_square = -INFINITY;

    for (DeviceWorkspace& device_workspace : workspace->devices) {
      DeviceGuard device_guard(device_workspace.device_id);
      const int blocks = static_cast<int>(
          (workspace->num_features + kHistogramThreads - 1) / kHistogramThreads);
      EvaluateFeatureSearchKernel<<<blocks, kHistogramThreads>>>(
          thrust::raw_pointer_cast(device_workspace.gradient_sums.data()),
          thrust::raw_pointer_cast(device_workspace.hessian_sums.data()),
          thrust::raw_pointer_cast(device_workspace.weight_sums.data()),
          thrust::raw_pointer_cast(device_workspace.feature_offsets_u32.data()),
          thrust::raw_pointer_cast(device_workspace.num_bins_per_feature.data()),
          thrust::raw_pointer_cast(device_workspace.categorical_mask.data()),
          host_node_stats[1],
          host_node_stats[2],
          host_node_stats[0],
          gradient_variance,
          lambda_l2,
          min_data_in_leaf,
          min_child_weight,
          min_split_gain,
          thrust::raw_pointer_cast(device_workspace.feature_search_results.data()),
          workspace->num_features);
      CTBOOST_CUDA_CHECK(cudaGetLastError());

      const std::vector<std::uint32_t> host_allowed_features =
          CandidateFeaturesForDevice(device_workspace, allowed_features, workspace->num_features);
      if (host_allowed_features.empty()) {
        continue;
      }
      thrust::device_vector<std::uint32_t> device_allowed_features(
          host_allowed_features.begin(), host_allowed_features.end());
      SelectBestFeatureKernel<<<1, kHistogramThreads>>>(
          thrust::raw_pointer_cast(device_workspace.feature_search_results.data()),
          thrust::raw_pointer_cast(device_allowed_features.data()),
          device_allowed_features.size(),
          thrust::raw_pointer_cast(device_workspace.best_feature_result.data()));
      CTBOOST_CUDA_CHECK(cudaGetLastError());
      CTBOOST_CUDA_CHECK(cudaDeviceSynchronize());

      GpuBestFeatureResult device_best_result{};
      device_best_result.feature_id = -1;
      CTBOOST_CUDA_CHECK(cudaMemcpy(&device_best_result,
                                    thrust::raw_pointer_cast(device_workspace.best_feature_result.data()),
                                    sizeof(GpuBestFeatureResult),
                                    cudaMemcpyDeviceToHost));
      if (IsBetterHostFeatureResult(device_best_result, host_best_result)) {
        host_best_result = device_best_result;
      }
    }

    out_result->feature_id = host_best_result.feature_id;
    out_result->p_value = host_best_result.search_result.p_value;
    out_result->chi_square = host_best_result.search_result.chi_square;
    out_result->split_valid = host_best_result.search_result.split_valid != 0U;
    out_result->is_categorical = host_best_result.search_result.is_categorical != 0U;
    out_result->split_bin = host_best_result.search_result.split_bin;
    out_result->gain = host_best_result.search_result.gain;
    out_result->node_statistics.sample_weight_sum = host_node_stats[0];
    out_result->node_statistics.total_gradient = host_node_stats[1];
    out_result->node_statistics.total_hessian = host_node_stats[2];
    out_result->node_statistics.gradient_square_sum = host_node_stats[3];
    out_result->left_categories.fill(0);
    std::copy(host_best_result.search_result.left_categories,
              host_best_result.search_result.left_categories + kGpuCategoricalRouteBins,
              out_result->left_categories.begin());
  } catch (const thrust::system_error& error) {
    throw std::runtime_error(std::string("CUDA thrust failure: ") + error.what());
  }
}

void PredictRawGpu(const HistMatrix& hist,
                   const std::vector<GpuTreeNode>& nodes,
                   const std::vector<std::int32_t>& tree_offsets,
                   float learning_rate,
                   int prediction_dimension,
                   std::vector<float>& out_predictions,
                   const std::string& devices) {
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
    const int prediction_device = ResolveRequestedDevices(devices).front();
    DeviceGuard device_guard(prediction_device);
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
