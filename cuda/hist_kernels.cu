#include "hist_kernels.cuh"

namespace {

constexpr std::size_t kSharedHistogramBins = 256;
constexpr std::size_t kHistogramRowTileSize = 1024;

}  // namespace

__global__ void HistMatrixFeatureSumsKernel(const std::uint16_t* bins,
                                            const std::size_t* row_indices,
                                            const float* gradients,
                                            const float* hessians,
                                            const float* weights,
                                            float* gradient_sums,
                                            float* hessian_sums,
                                            float* weight_sums,
                                            std::size_t node_row_count,
                                            std::size_t total_rows,
                                            std::size_t feature_index,
                                            std::size_t feature_offset,
                                            std::size_t feature_bin_count) {
  extern __shared__ float shared_storage[];
  float* shared_gradient_sums = shared_storage;
  float* shared_hessian_sums = shared_gradient_sums + feature_bin_count;
  float* shared_weight_sums = shared_hessian_sums + feature_bin_count;

  if (feature_bin_count == 0) {
    return;
  }

  for (std::size_t bin = threadIdx.x; bin < feature_bin_count; bin += blockDim.x) {
    shared_gradient_sums[bin] = 0.0F;
    shared_hessian_sums[bin] = 0.0F;
    shared_weight_sums[bin] = 0.0F;
  }
  __syncthreads();

  const std::size_t tile_index = static_cast<std::size_t>(blockIdx.x);
  const std::size_t tile_start = tile_index * kHistogramRowTileSize;
  const std::size_t tile_end = tile_start + kHistogramRowTileSize < node_row_count
                                   ? tile_start + kHistogramRowTileSize
                                   : node_row_count;
  for (std::size_t row_index = tile_start + threadIdx.x;
       row_index < tile_end;
       row_index += blockDim.x) {
    const std::size_t row = row_indices[row_index];
    const std::size_t bin = static_cast<std::size_t>(bins[feature_index * total_rows + row]);
    if (bin >= feature_bin_count) {
      continue;
    }

    const float sample_weight = weights[row];
    atomicAdd(&shared_gradient_sums[bin], sample_weight * gradients[row]);
    atomicAdd(&shared_hessian_sums[bin], sample_weight * hessians[row]);
    atomicAdd(&shared_weight_sums[bin], sample_weight);
  }
  __syncthreads();

  for (std::size_t bin = threadIdx.x; bin < feature_bin_count; bin += blockDim.x) {
    const std::size_t output_index = feature_offset + bin;
    atomicAdd(&gradient_sums[output_index], shared_gradient_sums[bin]);
    atomicAdd(&hessian_sums[output_index], shared_hessian_sums[bin]);
    atomicAdd(&weight_sums[output_index], shared_weight_sums[bin]);
  }
}

__global__ void ChunkedHistMatrixFeatureSumsKernel(const std::uint16_t* bins,
                                                   const std::size_t* row_indices,
                                                   const float* gradients,
                                                   const float* hessians,
                                                   const float* weights,
                                                   float* gradient_sums,
                                                   float* hessian_sums,
                                                   float* weight_sums,
                                                   std::size_t node_row_count,
                                                   std::size_t total_rows,
                                                   std::size_t feature_index,
                                                   std::size_t feature_offset,
                                                   std::size_t feature_bin_count) {
  __shared__ float shared_gradient_sums[kSharedHistogramBins];
  __shared__ float shared_hessian_sums[kSharedHistogramBins];
  __shared__ float shared_weight_sums[kSharedHistogramBins];

  if (feature_bin_count == 0) {
    return;
  }

  const std::size_t tile_index = static_cast<std::size_t>(blockIdx.x);
  const std::size_t tile_start = tile_index * kHistogramRowTileSize;
  const std::size_t tile_end = tile_start + kHistogramRowTileSize < node_row_count
                                   ? tile_start + kHistogramRowTileSize
                                   : node_row_count;
  for (std::size_t chunk_start = 0; chunk_start < feature_bin_count;
       chunk_start += kSharedHistogramBins) {
    const std::size_t chunk_bins = feature_bin_count - chunk_start < kSharedHistogramBins
                                       ? feature_bin_count - chunk_start
                                       : kSharedHistogramBins;
    for (std::size_t bin = threadIdx.x; bin < chunk_bins; bin += blockDim.x) {
      shared_gradient_sums[bin] = 0.0F;
      shared_hessian_sums[bin] = 0.0F;
      shared_weight_sums[bin] = 0.0F;
    }
    __syncthreads();

    for (std::size_t row_index = tile_start + threadIdx.x;
         row_index < tile_end;
         row_index += blockDim.x) {
      const std::size_t row = row_indices[row_index];
      const std::size_t bin = static_cast<std::size_t>(bins[feature_index * total_rows + row]);
      if (bin < chunk_start || bin >= chunk_start + chunk_bins) {
        continue;
      }

      const std::size_t chunk_bin = bin - chunk_start;
      const float sample_weight = weights[row];
      atomicAdd(&shared_gradient_sums[chunk_bin], sample_weight * gradients[row]);
      atomicAdd(&shared_hessian_sums[chunk_bin], sample_weight * hessians[row]);
      atomicAdd(&shared_weight_sums[chunk_bin], sample_weight);
    }
    __syncthreads();

    for (std::size_t bin = threadIdx.x; bin < chunk_bins; bin += blockDim.x) {
      const std::size_t output_index = feature_offset + chunk_start + bin;
      atomicAdd(&gradient_sums[output_index], shared_gradient_sums[bin]);
      atomicAdd(&hessian_sums[output_index], shared_hessian_sums[bin]);
      atomicAdd(&weight_sums[output_index], shared_weight_sums[bin]);
    }
    __syncthreads();
  }
}

__global__ void PredictForestKernel(const std::uint16_t* bins,
                                    const ctboost::GpuTreeNode* nodes,
                                    const std::int32_t* tree_offsets,
                                    float* predictions,
                                    std::size_t num_rows,
                                    std::size_t num_trees,
                                    int prediction_dimension,
                                    float learning_rate) {
  const std::size_t row = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= num_rows) {
    return;
  }

  float* row_predictions = predictions + row * static_cast<std::size_t>(prediction_dimension);
  for (std::size_t tree_index = 0; tree_index < num_trees; ++tree_index) {
    std::int32_t node_index = tree_offsets[tree_index];
    while (node_index >= 0 && nodes[node_index].is_leaf == 0U) {
      const ctboost::GpuTreeNode& node = nodes[node_index];
      const std::size_t bin = static_cast<std::size_t>(
          bins[static_cast<std::size_t>(node.split_feature_id) * num_rows + row]);
      node_index = node.is_categorical_split != 0U
                       ? (node.left_categories[bin] != 0U ? node.left_child : node.right_child)
                       : (bin <= static_cast<std::size_t>(node.split_bin_index) ? node.left_child
                                                                                : node.right_child);
    }

    if (node_index < 0) {
      continue;
    }

    const float leaf_value = learning_rate * nodes[node_index].leaf_weight;
    if (prediction_dimension == 1) {
      row_predictions[0] += leaf_value;
    } else {
      row_predictions[tree_index % static_cast<std::size_t>(prediction_dimension)] += leaf_value;
    }
  }
}
