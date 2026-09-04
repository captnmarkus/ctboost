#include "ctboost/cuda_backend.hpp"
#include "ctboost/histogram.hpp"
#include "hist_kernels.cuh"

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#if defined(_WIN32)
// CUDA 12.8's libcu++ expands the UCRT INFINITY macro through an overflowing
// float expression while Thrust headers are parsed by NVCC. Provide the same
// float infinity through libcu++ while those headers are parsed.
#include <cuda/std/limits>
#pragma push_macro("INFINITY")
#undef INFINITY
#define INFINITY (::cuda::std::numeric_limits<float>::infinity())
#endif

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/partition.h>
#include <thrust/sequence.h>
#include <thrust/system_error.h>

#if defined(_WIN32)
#pragma pop_macro("INFINITY")
#endif

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
constexpr unsigned int kCuMemGetAddressRangeAbiVersion = 3020U;

PFN_cuMemGetAddressRange_v3020 ResolveCuMemGetAddressRange() {
  struct EntryPoint {
    PFN_cuMemGetAddressRange_v3020 function{nullptr};
    cudaError_t runtime_status{cudaSuccess};
    cudaDriverEntryPointQueryResult driver_status{cudaDriverEntryPointSymbolNotFound};
  };

  static const EntryPoint entry_point = [] {
    EntryPoint result;
    void* function = nullptr;
    result.runtime_status = cudaGetDriverEntryPointByVersion(
        "cuMemGetAddressRange",
        &function,
        kCuMemGetAddressRangeAbiVersion,
        cudaEnableDefault,
        &result.driver_status);
    if (result.runtime_status == cudaSuccess &&
        result.driver_status == cudaDriverEntryPointSuccess && function != nullptr) {
      result.function = reinterpret_cast<PFN_cuMemGetAddressRange_v3020>(function);
    }
    return result;
  }();

  if (entry_point.runtime_status != cudaSuccess) {
    throw std::runtime_error(
        std::string("CUDA driver entry-point lookup failed: ") +
        cudaGetErrorString(entry_point.runtime_status));
  }
  if (entry_point.driver_status != cudaDriverEntryPointSuccess ||
      entry_point.function == nullptr) {
    throw std::runtime_error(
        "CUDA driver does not expose an ABI-compatible cuMemGetAddressRange entry point");
  }
  return entry_point.function;
}

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

void CopyHistogramBinsToDevice(const HistMatrix& hist,
                               thrust::device_vector<std::uint8_t>& bins_u8,
                               thrust::device_vector<std::uint16_t>& bins_u16) {
  const std::size_t expected_bin_count = hist.num_rows * hist.num_cols;
  if (hist.uses_compact_bin_storage()) {
    bins_u8.resize(expected_bin_count);
    bins_u16.clear();
    if (!hist.uses_external_bin_storage()) {
      thrust::copy(hist.compact_bin_indices.begin(), hist.compact_bin_indices.end(), bins_u8.begin());
      return;
    }
    for (std::size_t feature = 0; feature < hist.num_cols; ++feature) {
      const auto feature_view = hist.feature_bins(feature);
      thrust::copy(feature_view.data_u8,
                   feature_view.data_u8 + static_cast<std::ptrdiff_t>(hist.num_rows),
                   bins_u8.begin() + static_cast<std::ptrdiff_t>(feature * hist.num_rows));
    }
    return;
  }

  bins_u16.resize(expected_bin_count);
  bins_u8.clear();
  if (!hist.uses_external_bin_storage()) {
    thrust::copy(hist.bin_indices.begin(), hist.bin_indices.end(), bins_u16.begin());
    return;
  }
  for (std::size_t feature = 0; feature < hist.num_cols; ++feature) {
    const auto feature_view = hist.feature_bins(feature);
    thrust::copy(feature_view.data_u16,
                 feature_view.data_u16 + static_cast<std::ptrdiff_t>(hist.num_rows),
                 bins_u16.begin() + static_cast<std::ptrdiff_t>(feature * hist.num_rows));
  }
}

template <typename BinType>
__global__ void CopyValidateCudaQuantizedBinsKernel(
    const std::uint8_t* source,
    std::size_t row_stride_bytes,
    std::size_t col_stride_bytes,
    BinType* destination,
    const std::uint16_t* num_bins_per_feature,
    std::size_t num_rows,
    std::size_t num_cols,
    unsigned long long* first_invalid_index) {
  const std::size_t total_values = num_rows * num_cols;
  for (std::size_t destination_index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       destination_index < total_values;
       destination_index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const std::size_t feature = destination_index / num_rows;
    const std::size_t row = destination_index - feature * num_rows;
    const auto* source_value = reinterpret_cast<const BinType*>(
        source + row * row_stride_bytes + feature * col_stride_bytes);
    const BinType bin = *source_value;
    destination[destination_index] = bin;
    if (static_cast<std::uint64_t>(bin) >=
        static_cast<std::uint64_t>(num_bins_per_feature[feature])) {
      atomicMin(first_invalid_index,
                static_cast<unsigned long long>(destination_index));
    }
  }
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

  ~DeviceGuard() noexcept { (void)cudaSetDevice(previous_device_); }

 private:
  int previous_device_{0};
};

class CurrentDeviceRestorer {
 public:
  CurrentDeviceRestorer() noexcept : valid_(cudaGetDevice(&device_) == cudaSuccess) {}
  CurrentDeviceRestorer(const CurrentDeviceRestorer&) = delete;
  CurrentDeviceRestorer& operator=(const CurrentDeviceRestorer&) = delete;
  CurrentDeviceRestorer(CurrentDeviceRestorer&&) = delete;
  CurrentDeviceRestorer& operator=(CurrentDeviceRestorer&&) = delete;

  ~CurrentDeviceRestorer() noexcept {
    if (valid_) {
      (void)cudaSetDevice(device_);
    }
  }

 private:
  int device_{0};
  bool valid_{false};
};

struct DeviceWorkspace {
  DeviceWorkspace() = default;
  DeviceWorkspace(const DeviceWorkspace&) = delete;
  DeviceWorkspace& operator=(const DeviceWorkspace&) = delete;
  DeviceWorkspace(DeviceWorkspace&&) = delete;
  DeviceWorkspace& operator=(DeviceWorkspace&&) = delete;

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
  thrust::device_vector<double> feature_weights;
  thrust::device_vector<double> first_feature_use_penalties;
  thrust::device_vector<std::uint8_t> model_feature_used_mask;
  thrust::device_vector<int> monotone_constraints;
  thrust::device_vector<GpuFeatureSearchResult> feature_search_results;
  thrust::device_vector<GpuBestFeatureResult> best_feature_result;
  thrust::device_vector<std::uint32_t> chunk_feature_indices;
  thrust::device_vector<std::uint32_t> chunk_bin_starts;
  thrust::device_vector<std::uint32_t> chunk_bin_counts;
  thrust::device_vector<std::uint32_t> chunk_output_offsets;

  ~DeviceWorkspace() noexcept { (void)cudaSetDevice(device_id); }
};

static_assert(!std::is_copy_constructible_v<DeviceWorkspace>);
static_assert(!std::is_copy_assignable_v<DeviceWorkspace>);
static_assert(!std::is_move_constructible_v<DeviceWorkspace>);
static_assert(!std::is_move_assignable_v<DeviceWorkspace>);

struct GpuHistogramWorkspace {
  std::size_t num_rows{0};
  std::size_t num_features{0};
  std::size_t total_bins{0};
  std::size_t max_feature_bins{0};
  std::size_t histogram_chunk_bins{kHistogramChunkBins};
  std::uint8_t bin_index_bytes{2};
  std::vector<std::size_t> feature_offsets;
  std::vector<int> device_ids;
  std::deque<DeviceWorkspace> devices;
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
         workspace.feature_weights.size() * sizeof(double) +
         workspace.first_feature_use_penalties.size() * sizeof(double) +
         workspace.model_feature_used_mask.size() * sizeof(std::uint8_t) +
         workspace.monotone_constraints.size() * sizeof(int) +
         workspace.feature_search_results.size() * sizeof(GpuFeatureSearchResult) +
         workspace.best_feature_result.size() * sizeof(GpuBestFeatureResult) +
         workspace.chunk_feature_indices.size() * sizeof(std::uint32_t) +
         workspace.chunk_bin_starts.size() * sizeof(std::uint32_t) +
         workspace.chunk_bin_counts.size() * sizeof(std::uint32_t) +
         workspace.chunk_output_offsets.size() * sizeof(std::uint32_t);
}

std::array<double, 5> DownloadNodeStatistics(const DeviceWorkspace& workspace) {
  std::vector<double> host_node_stats(workspace.node_statistics.size(), 0.0);
  thrust::copy(workspace.node_statistics.begin(), workspace.node_statistics.end(), host_node_stats.begin());
  return {
      host_node_stats.size() > 0 ? host_node_stats[0] : 0.0,
      host_node_stats.size() > 1 ? host_node_stats[1] : 0.0,
      host_node_stats.size() > 2 ? host_node_stats[2] : 0.0,
      host_node_stats.size() > 3 ? host_node_stats[3] : 0.0,
      host_node_stats.size() > 4 ? host_node_stats[4] : 0.0,
  };
}

void UploadNodeStatistics(DeviceWorkspace& workspace, const std::array<double, 5>& node_statistics) {
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

bool IsBetterAdjustedHostFeatureResult(const GpuBestFeatureResult& candidate,
                                       const GpuBestFeatureResult& best) {
  if (candidate.feature_id < 0 || candidate.search_result.degrees_of_freedom == 0U ||
      candidate.search_result.split_valid == 0U || candidate.search_result.gain <= 0.0) {
    return false;
  }
  if (best.feature_id < 0 || best.search_result.degrees_of_freedom == 0U ||
      best.search_result.split_valid == 0U || best.search_result.gain <= 0.0) {
    return true;
  }
  if (candidate.adjusted_gain > best.adjusted_gain) {
    return true;
  }
  if (std::fabs(candidate.adjusted_gain - best.adjusted_gain) <= 1e-12 &&
      candidate.search_result.p_value < best.search_result.p_value) {
    return true;
  }
  if (std::fabs(candidate.adjusted_gain - best.adjusted_gain) <= 1e-12 &&
      std::fabs(candidate.search_result.p_value - best.search_result.p_value) <= 1e-12 &&
      candidate.search_result.chi_square > best.search_result.chi_square) {
    return true;
  }
  return std::fabs(candidate.adjusted_gain - best.adjusted_gain) <= 1e-12 &&
         std::fabs(candidate.search_result.p_value - best.search_result.p_value) <= 1e-12 &&
         std::fabs(candidate.search_result.chi_square - best.search_result.chi_square) <= 1e-12 &&
         candidate.feature_id < best.feature_id;
}

int ResolveCudaQuantizedPointerDevice(const CudaQuantizedMatrixView& view) {
  if (view.element_bytes != 1U && view.element_bytes != 2U) {
    throw std::invalid_argument("CUDA quantized data must use uint8 or uint16 elements");
  }
  if (view.data % view.element_bytes != 0U || view.row_stride_bytes < 0 ||
      view.col_stride_bytes < 0 ||
      view.row_stride_bytes % static_cast<std::int64_t>(view.element_bytes) != 0 ||
      view.col_stride_bytes % static_cast<std::int64_t>(view.element_bytes) != 0) {
    throw std::invalid_argument(
        "CUDA quantized data pointer and strides must be non-negative and dtype-aligned");
  }
  if (view.num_rows == 0U || view.num_cols == 0U) {
    return -1;
  }
  if (view.num_cols > std::numeric_limits<std::size_t>::max() / view.element_bytes ||
      view.num_rows > std::numeric_limits<std::size_t>::max() / view.element_bytes) {
    throw std::invalid_argument("CUDA quantized contiguous strides overflow size_t");
  }
  const std::size_t c_row_stride = view.num_cols * view.element_bytes;
  const std::size_t f_col_stride = view.num_rows * view.element_bytes;
  const bool c_contiguous =
      view.col_stride_bytes == static_cast<std::int64_t>(view.element_bytes) &&
      (view.num_rows <= 1U ||
       view.row_stride_bytes == static_cast<std::int64_t>(c_row_stride));
  const bool f_contiguous =
      view.row_stride_bytes == static_cast<std::int64_t>(view.element_bytes) &&
      (view.num_cols <= 1U ||
       view.col_stride_bytes == static_cast<std::int64_t>(f_col_stride));
  if (!c_contiguous && !f_contiguous) {
    throw std::invalid_argument(
        "CUDA quantized data must be C-contiguous or Fortran-contiguous");
  }
  cudaPointerAttributes attributes{};
  CTBOOST_CUDA_CHECK(
      cudaPointerGetAttributes(&attributes, reinterpret_cast<const void*>(view.data)));
  if (attributes.type != cudaMemoryTypeDevice) {
    throw std::invalid_argument(
        "CUDA quantized data pointer must refer to CUDA device memory");
  }

  const auto checked_product = [](std::size_t count,
                                  std::size_t stride) -> std::size_t {
    if (count != 0U && stride > std::numeric_limits<std::size_t>::max() / count) {
      throw std::invalid_argument("CUDA quantized data device span overflows size_t");
    }
    return count * stride;
  };
  const std::size_t row_offset = checked_product(
      view.num_rows - 1U, static_cast<std::size_t>(view.row_stride_bytes));
  const std::size_t col_offset = checked_product(
      view.num_cols - 1U, static_cast<std::size_t>(view.col_stride_bytes));
  if (row_offset > std::numeric_limits<std::size_t>::max() - col_offset ||
      row_offset + col_offset >
          std::numeric_limits<std::size_t>::max() - view.element_bytes) {
    throw std::invalid_argument("CUDA quantized data device span overflows size_t");
  }
  const std::size_t required_span = row_offset + col_offset + view.element_bytes;

  DeviceGuard source_device_guard(attributes.device);
  CUdeviceptr allocation_base = 0U;
  std::size_t allocation_size = 0U;
  const CUresult range_status = ResolveCuMemGetAddressRange()(
      &allocation_base,
      &allocation_size,
      static_cast<CUdeviceptr>(view.data));
  if (range_status != CUDA_SUCCESS) {
    throw std::invalid_argument(
        "CUDA quantized data pointer must belong to a resolvable CUDA device allocation");
  }
  const CUdeviceptr data_pointer = static_cast<CUdeviceptr>(view.data);
  if (data_pointer < allocation_base) {
    throw std::invalid_argument(
        "CUDA quantized data span is outside its CUDA device allocation");
  }
  const CUdeviceptr allocation_offset_value = data_pointer - allocation_base;
  if (allocation_offset_value >
      static_cast<CUdeviceptr>(std::numeric_limits<std::size_t>::max())) {
    throw std::invalid_argument(
        "CUDA quantized data span is outside its CUDA device allocation");
  }
  const std::size_t allocation_offset =
      static_cast<std::size_t>(allocation_offset_value);
  if (allocation_offset > allocation_size ||
      required_span > allocation_size - allocation_offset) {
    throw std::invalid_argument(
        "CUDA quantized data span is outside its CUDA device allocation");
  }
  return attributes.device;
}

void SynchronizeCudaQuantizedProducer(const CudaQuantizedMatrixView& view) {
  if (view.producer_stream == 2U) {
    throw std::invalid_argument(
        "CUDA Array Interface per-thread default stream marker 2 is not supported by "
        "deferred CUDA quantized Pool consumption");
  }
  if (view.producer_stream == 0U || view.num_rows == 0U || view.num_cols == 0U) {
    return;
  }
  cudaStream_t producer_stream = nullptr;
  if (view.producer_stream == 1U) {
    producer_stream = cudaStreamLegacy;
  } else {
    producer_stream = reinterpret_cast<cudaStream_t>(view.producer_stream);
  }
  CTBOOST_CUDA_CHECK(cudaStreamSynchronize(producer_stream));
}

template <typename BinType>
void CopyCudaQuantizedBinsToDevice(
    const CudaQuantizedMatrixView& view,
    const thrust::device_vector<std::uint16_t>& num_bins_per_feature,
    thrust::device_vector<BinType>& destination) {
  const std::size_t total_values = view.num_rows * view.num_cols;
  destination.resize(total_values);
  if (total_values == 0U) {
    return;
  }

  thrust::device_vector<unsigned long long> first_invalid_index(
      1U, std::numeric_limits<unsigned long long>::max());
  constexpr int kCopyThreads = 256;
  const std::size_t required_blocks =
      (total_values + static_cast<std::size_t>(kCopyThreads) - 1U) /
      static_cast<std::size_t>(kCopyThreads);
  const int blocks = static_cast<int>(std::min<std::size_t>(required_blocks, 65535U));
  CopyValidateCudaQuantizedBinsKernel<<<blocks, kCopyThreads>>>(
      reinterpret_cast<const std::uint8_t*>(view.data),
      static_cast<std::size_t>(view.row_stride_bytes),
      static_cast<std::size_t>(view.col_stride_bytes),
      thrust::raw_pointer_cast(destination.data()),
      thrust::raw_pointer_cast(num_bins_per_feature.data()),
      view.num_rows,
      view.num_cols,
      thrust::raw_pointer_cast(first_invalid_index.data()));
  CTBOOST_CUDA_CHECK(cudaGetLastError());
  CTBOOST_CUDA_CHECK(cudaDeviceSynchronize());

  unsigned long long invalid_index = std::numeric_limits<unsigned long long>::max();
  CTBOOST_CUDA_CHECK(cudaMemcpy(&invalid_index,
                                thrust::raw_pointer_cast(first_invalid_index.data()),
                                sizeof(invalid_index),
                                cudaMemcpyDeviceToHost));
  if (invalid_index != std::numeric_limits<unsigned long long>::max()) {
    const std::size_t feature = static_cast<std::size_t>(invalid_index) / view.num_rows;
    const std::size_t row = static_cast<std::size_t>(invalid_index) % view.num_rows;
    throw std::invalid_argument(
        "CUDA quantized bin is outside its schema range at row " +
        std::to_string(row) + ", feature " + std::to_string(feature));
  }
}

void CopyCudaQuantizedBinsToDevice(const CudaQuantizedMatrixView& view,
                                   DeviceWorkspace& device_workspace) {
  SynchronizeCudaQuantizedProducer(view);
  if (view.element_bytes == 1U) {
    device_workspace.bins_u16.clear();
    CopyCudaQuantizedBinsToDevice(
        view, device_workspace.num_bins_per_feature, device_workspace.bins_u8);
    return;
  }
  if (view.element_bytes == 2U) {
    device_workspace.bins_u8.clear();
    CopyCudaQuantizedBinsToDevice(
        view, device_workspace.num_bins_per_feature, device_workspace.bins_u16);
    return;
  }
  throw std::invalid_argument("CUDA quantized bin element width must be one or two bytes");
}

void InitializeDeviceWorkspace(DeviceWorkspace& device_workspace,
                               const HistMatrix& hist,
                               const std::vector<float>& weights,
                               const std::vector<std::uint32_t>& feature_offsets_u32,
                               const std::vector<std::size_t>& feature_offsets,
                               std::size_t histogram_chunk_bins,
                               const CudaQuantizedMatrixView* cuda_quantized) {
  CopyHostVectorToDevice(hist.num_bins_per_feature, device_workspace.num_bins_per_feature);
  if (cuda_quantized == nullptr) {
    CopyHistogramBinsToDevice(hist, device_workspace.bins_u8, device_workspace.bins_u16);
  } else {
    CopyCudaQuantizedBinsToDevice(*cuda_quantized, device_workspace);
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
  device_workspace.node_statistics.resize(5, 0.0);
  device_workspace.feature_search_results.resize(hist.num_cols);
  device_workspace.best_feature_result.resize(1);
  CopyHostVectorToDevice(feature_offsets_u32, device_workspace.feature_offsets_u32);
  CopyHostVectorToDevice(hist.categorical_mask, device_workspace.categorical_mask);
  device_workspace.feature_weights.assign(hist.num_cols, 1.0);
  device_workspace.first_feature_use_penalties.assign(hist.num_cols, 0.0);
  device_workspace.model_feature_used_mask.assign(hist.num_cols, 0U);
  device_workspace.monotone_constraints.assign(hist.num_cols, 0);

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
  if (workspace == nullptr) {
    return;
  }
  CurrentDeviceRestorer restore_caller_device;
  delete workspace;
}

namespace {

GpuHistogramWorkspacePtr CreateGpuHistogramWorkspaceImpl(
    const HistMatrix& hist,
    const std::vector<float>& weights,
    const std::string& devices,
    const CudaQuantizedMatrixView* cuda_quantized) {
  if (weights.size() != hist.num_rows) {
    throw std::invalid_argument("GPU histogram weights must match the histogram row count");
  }
  if (cuda_quantized == nullptr) {
    const std::size_t expected_bin_count = hist.num_rows * hist.num_cols;
    if (hist.uses_compact_bin_storage() && !hist.uses_external_bin_storage()) {
      if (hist.compact_bin_indices.size() != expected_bin_count) {
        throw std::invalid_argument(
            "GPU histogram compact bins must have num_rows * num_cols elements");
      }
    } else if (!hist.uses_compact_bin_storage() && !hist.uses_external_bin_storage() &&
               hist.bin_indices.size() != expected_bin_count) {
      throw std::invalid_argument("GPU histogram bins must have num_rows * num_cols elements");
    }
  } else {
    if (cuda_quantized->num_rows != hist.num_rows ||
        cuda_quantized->num_cols != hist.num_cols) {
      throw std::invalid_argument(
          "CUDA quantized matrix shape must match the histogram shape");
    }
    if (cuda_quantized->element_bytes != hist.bin_storage_bytes()) {
      throw std::invalid_argument(
          "CUDA quantized matrix dtype must match the histogram bin storage width");
    }
    if (ParseDeviceList(devices).size() != 1U) {
      throw std::invalid_argument(
          "CUDA quantized training currently supports exactly one CUDA device");
    }
  }

  CurrentDeviceRestorer restore_caller_device;
  try {
    GpuHistogramWorkspacePtr workspace(new GpuHistogramWorkspace,
                                       DestroyGpuHistogramWorkspace);
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
    if (cuda_quantized != nullptr) {
      const int pointer_device = ResolveCudaQuantizedPointerDevice(*cuda_quantized);
      if (pointer_device >= 0 && pointer_device != requested_devices.front()) {
        throw std::invalid_argument(
            "CUDA quantized data and its training workspace must use the same CUDA device");
      }
    }
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
                                workspace->histogram_chunk_bins,
                                cuda_quantized);
    }
    if (workspace->devices.empty()) {
      throw std::runtime_error(
          "GPU histogram workspace could not assign features to any CUDA device");
    }

    return workspace;
  } catch (const thrust::system_error& error) {
    throw std::runtime_error(std::string("CUDA thrust failure: ") + error.what());
  }
}

}  // namespace

GpuHistogramWorkspacePtr CreateGpuHistogramWorkspace(const HistMatrix& hist,
                                                     const std::vector<float>& weights,
                                                     const std::string& devices) {
  return CreateGpuHistogramWorkspaceImpl(hist, weights, devices, nullptr);
}

GpuHistogramWorkspacePtr CreateGpuHistogramWorkspaceFromCudaQuantized(
    const HistMatrix& hist,
    const CudaQuantizedMatrixView& cuda_quantized,
    const std::vector<float>& weights,
    const std::string& devices) {
  return CreateGpuHistogramWorkspaceImpl(hist, weights, devices, &cuda_quantized);
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
        const std::array<double, 5> host_node_stats = DownloadNodeStatistics(device_workspace);
        out_snapshot->node_statistics.sample_weight_sum = host_node_stats[0];
        out_snapshot->node_statistics.total_gradient = host_node_stats[1];
        out_snapshot->node_statistics.total_hessian = host_node_stats[2];
        out_snapshot->node_statistics.gradient_square_sum = host_node_stats[3];
        out_snapshot->node_statistics.sample_count =
            static_cast<std::uint64_t>(std::llround(host_node_stats[4]));
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
    const std::array<double, 5> host_node_stats{
        snapshot.node_statistics.sample_weight_sum,
        snapshot.node_statistics.total_gradient,
        snapshot.node_statistics.total_hessian,
        snapshot.node_statistics.gradient_square_sum,
        static_cast<double>(snapshot.node_statistics.sample_count),
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

void UploadFeatureControlsGpu(GpuHistogramWorkspace* workspace,
                              const std::vector<double>* feature_weights,
                              const std::vector<double>* first_feature_use_penalties,
                              const std::vector<std::uint8_t>* model_feature_used_mask,
                              const std::vector<int>* monotone_constraints) {
  if (workspace == nullptr) {
    throw std::invalid_argument("GPU histogram workspace must not be null");
  }

  const auto resolve_vector = [&](std::size_t expected_size,
                                  const auto* values,
                                  const auto& default_value) {
    using ValueType = typename std::decay_t<decltype(default_value)>::value_type;
    std::vector<ValueType> resolved(expected_size, ValueType{});
    if (values == nullptr || values->empty()) {
      resolved = default_value;
      return resolved;
    }
    if (values->size() != expected_size) {
      throw std::invalid_argument(
          "GPU feature control vectors must match the histogram feature count");
    }
    resolved.assign(values->begin(), values->end());
    return resolved;
  };

  try {
    const std::vector<double> host_feature_weights = resolve_vector(
        workspace->num_features,
        feature_weights,
        std::vector<double>(workspace->num_features, 1.0));
    const std::vector<double> host_first_feature_use_penalties = resolve_vector(
        workspace->num_features,
        first_feature_use_penalties,
        std::vector<double>(workspace->num_features, 0.0));
    const std::vector<std::uint8_t> host_model_feature_used_mask = resolve_vector(
        workspace->num_features,
        model_feature_used_mask,
        std::vector<std::uint8_t>(workspace->num_features, 0U));
    const std::vector<int> host_monotone_constraints = resolve_vector(
        workspace->num_features,
        monotone_constraints,
        std::vector<int>(workspace->num_features, 0));

    for (DeviceWorkspace& device_workspace : workspace->devices) {
      DeviceGuard device_guard(device_workspace.device_id);
      thrust::copy(host_feature_weights.begin(),
                   host_feature_weights.end(),
                   device_workspace.feature_weights.begin());
      thrust::copy(host_first_feature_use_penalties.begin(),
                   host_first_feature_use_penalties.end(),
                   device_workspace.first_feature_use_penalties.begin());
      thrust::copy(host_model_feature_used_mask.begin(),
                   host_model_feature_used_mask.end(),
                   device_workspace.model_feature_used_mask.begin());
      thrust::copy(host_monotone_constraints.begin(),
                   host_monotone_constraints.end(),
                   device_workspace.monotone_constraints.begin());
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
  const bool no_histogram_work =
      row_count == 0 || workspace->num_features == 0 || workspace->total_bins == 0;

  try {
    for (std::size_t device_index = 0; device_index < workspace->devices.size(); ++device_index) {
      DeviceWorkspace& device_workspace = workspace->devices[device_index];
      DeviceGuard device_guard(device_workspace.device_id);
      thrust::fill(device_workspace.gradient_sums.begin(), device_workspace.gradient_sums.end(), 0.0F);
      thrust::fill(device_workspace.hessian_sums.begin(), device_workspace.hessian_sums.end(), 0.0F);
      thrust::fill(device_workspace.weight_sums.begin(), device_workspace.weight_sums.end(), 0.0F);
      thrust::fill(device_workspace.node_statistics.begin(), device_workspace.node_statistics.end(), 0.0);

      // Empty local children are valid in row-sharded distributed training.
      // Their collective contribution must be zero, never a stale parent or
      // sibling histogram left in the reusable device buffers.
      if (no_histogram_work) {
        continue;
      }

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
        const std::array<double, 5> host_node_stats = DownloadNodeStatistics(device_workspace);
        out_node_stats->sample_weight_sum = host_node_stats[0];
        out_node_stats->total_gradient = host_node_stats[1];
        out_node_stats->total_hessian = host_node_stats[2];
        out_node_stats->gradient_square_sum = host_node_stats[3];
        out_node_stats->sample_count =
            static_cast<std::uint64_t>(std::llround(host_node_stats[4]));
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
                            double alpha,
                            int depth,
                            std::size_t row_begin,
                            std::size_t row_end,
                            double leaf_lower_bound,
                            double leaf_upper_bound,
                            std::uint64_t random_seed,
                            double random_strength,
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
    std::array<double, 5> host_node_stats{};
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
    host_best_result.search_result.chi_square = -std::numeric_limits<double>::infinity();
    GpuBestFeatureResult host_best_adjusted_result{};
    host_best_adjusted_result.feature_id = -1;
    host_best_adjusted_result.adjusted_gain = 0.0;
    host_best_adjusted_result.search_result.p_value = 1.0;
    host_best_adjusted_result.search_result.chi_square =
        -std::numeric_limits<double>::infinity();

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
          thrust::raw_pointer_cast(device_workspace.monotone_constraints.data()),
          host_node_stats[1],
          host_node_stats[2],
          host_node_stats[0],
          gradient_variance,
          lambda_l2,
          min_data_in_leaf,
          min_child_weight,
          min_split_gain,
          leaf_lower_bound,
          leaf_upper_bound,
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

      GpuBestFeatureResult device_best_result{};
      device_best_result.feature_id = -1;
      CTBOOST_CUDA_CHECK(cudaMemcpy(&device_best_result,
                                    thrust::raw_pointer_cast(device_workspace.best_feature_result.data()),
                                    sizeof(GpuBestFeatureResult),
                                    cudaMemcpyDeviceToHost));
      if (IsBetterHostFeatureResult(device_best_result, host_best_result)) {
        host_best_result = device_best_result;
      }

      SelectBestAdjustedFeatureKernel<<<1, kHistogramThreads>>>(
          thrust::raw_pointer_cast(device_workspace.feature_search_results.data()),
          thrust::raw_pointer_cast(device_allowed_features.data()),
          device_allowed_features.size(),
          alpha,
          depth,
          row_begin,
          row_end,
          random_seed,
          random_strength,
          thrust::raw_pointer_cast(device_workspace.feature_weights.data()),
          thrust::raw_pointer_cast(device_workspace.first_feature_use_penalties.data()),
          thrust::raw_pointer_cast(device_workspace.model_feature_used_mask.data()),
          thrust::raw_pointer_cast(device_workspace.best_feature_result.data()));
      CTBOOST_CUDA_CHECK(cudaGetLastError());
      CTBOOST_CUDA_CHECK(cudaDeviceSynchronize());

      GpuBestFeatureResult device_best_adjusted_result{};
      device_best_adjusted_result.feature_id = -1;
      CTBOOST_CUDA_CHECK(cudaMemcpy(&device_best_adjusted_result,
                                    thrust::raw_pointer_cast(device_workspace.best_feature_result.data()),
                                    sizeof(GpuBestFeatureResult),
                                    cudaMemcpyDeviceToHost));
      if (IsBetterAdjustedHostFeatureResult(device_best_adjusted_result,
                                            host_best_adjusted_result)) {
        host_best_adjusted_result = device_best_adjusted_result;
      }
    }

    const GpuBestFeatureResult& resolved_result =
        host_best_adjusted_result.feature_id >= 0 ? host_best_adjusted_result : host_best_result;
    out_result->feature_id = resolved_result.feature_id;
    out_result->p_value = resolved_result.search_result.p_value;
    out_result->chi_square = resolved_result.search_result.chi_square;
    out_result->split_valid = resolved_result.search_result.split_valid != 0U;
    out_result->is_categorical = resolved_result.search_result.is_categorical != 0U;
    out_result->split_bin = resolved_result.search_result.split_bin;
    out_result->gain = resolved_result.search_result.gain;
    out_result->adjusted_gain =
        resolved_result.feature_id >= 0 && resolved_result.search_result.split_valid != 0U &&
                resolved_result.search_result.gain > 0.0
            ? (host_best_adjusted_result.feature_id >= 0 ? resolved_result.adjusted_gain
                                                         : resolved_result.search_result.gain)
            : -std::numeric_limits<double>::infinity();
    out_result->left_leaf_weight = resolved_result.search_result.left_leaf_weight;
    out_result->right_leaf_weight = resolved_result.search_result.right_leaf_weight;
    out_result->node_statistics.sample_weight_sum = host_node_stats[0];
    out_result->node_statistics.total_gradient = host_node_stats[1];
    out_result->node_statistics.total_hessian = host_node_stats[2];
    out_result->node_statistics.gradient_square_sum = host_node_stats[3];
    out_result->node_statistics.sample_count =
        static_cast<std::uint64_t>(std::llround(host_node_stats[4]));
    out_result->left_categories.fill(0);
    std::copy(resolved_result.search_result.left_categories,
              resolved_result.search_result.left_categories + kGpuCategoricalRouteBins,
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
  if (hist.uses_compact_bin_storage() && !hist.uses_external_bin_storage()) {
    if (hist.compact_bin_indices.size() != expected_bin_count) {
      throw std::invalid_argument("GPU prediction compact bins must have num_rows * num_cols elements");
    }
  } else if (!hist.uses_compact_bin_storage() && !hist.uses_external_bin_storage() &&
             hist.bin_indices.size() != expected_bin_count) {
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
    CopyHistogramBinsToDevice(hist, device_bins_u8, device_bins_u16);
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
