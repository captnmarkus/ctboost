#include "hist_kernels.cuh"

namespace {

constexpr std::size_t kHistogramChunkBins = 256;
constexpr std::size_t kHistogramRowTileSize = 1024;

}  // namespace

__global__ void HistMatrixFeatureChunksKernel(const std::uint16_t* bins,
                                              const std::size_t* row_indices,
                                              const float* gradients,
                                              const float* hessians,
                                              const float* weights,
                                              const std::uint32_t* chunk_feature_indices,
                                              const std::uint32_t* chunk_bin_starts,
                                              const std::uint32_t* chunk_bin_counts,
                                              const std::uint32_t* chunk_output_offsets,
                                              float* gradient_sums,
                                              float* hessian_sums,
                                              float* weight_sums,
                                              std::size_t node_row_count,
                                              std::size_t total_rows,
                                              std::size_t target_stride,
                                              std::size_t target_offset) {
  __shared__ float shared_gradient_sums[kHistogramChunkBins];
  __shared__ float shared_hessian_sums[kHistogramChunkBins];
  __shared__ float shared_weight_sums[kHistogramChunkBins];

  const std::size_t chunk_index = static_cast<std::size_t>(blockIdx.y);
  const std::size_t chunk_bin_start = static_cast<std::size_t>(chunk_bin_starts[chunk_index]);
  const std::size_t chunk_bin_count = static_cast<std::size_t>(chunk_bin_counts[chunk_index]);
  if (chunk_bin_count == 0) {
    return;
  }

  for (std::size_t bin = threadIdx.x; bin < chunk_bin_count; bin += blockDim.x) {
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
  const std::size_t feature_index = static_cast<std::size_t>(chunk_feature_indices[chunk_index]);
  for (std::size_t row_index = tile_start + threadIdx.x;
       row_index < tile_end;
       row_index += blockDim.x) {
    const std::size_t row = row_indices[row_index];
    const std::size_t bin = static_cast<std::size_t>(bins[feature_index * total_rows + row]);
    if (bin < chunk_bin_start || bin >= chunk_bin_start + chunk_bin_count) {
      continue;
    }

    const std::size_t chunk_bin = bin - chunk_bin_start;
    const float sample_weight = weights[row];
    const std::size_t target_index = row * target_stride + target_offset;
    atomicAdd(&shared_gradient_sums[chunk_bin], sample_weight * gradients[target_index]);
    atomicAdd(&shared_hessian_sums[chunk_bin], sample_weight * hessians[target_index]);
    atomicAdd(&shared_weight_sums[chunk_bin], sample_weight);
  }
  __syncthreads();

  const std::size_t feature_offset = static_cast<std::size_t>(chunk_output_offsets[chunk_index]);
  for (std::size_t bin = threadIdx.x; bin < chunk_bin_count; bin += blockDim.x) {
    const std::size_t output_index = feature_offset + chunk_bin_start + bin;
    atomicAdd(&gradient_sums[output_index], shared_gradient_sums[bin]);
    atomicAdd(&hessian_sums[output_index], shared_hessian_sums[bin]);
    atomicAdd(&weight_sums[output_index], shared_weight_sums[bin]);
  }
}

__global__ void NodeTargetStatisticsKernel(const std::size_t* row_indices,
                                           const float* gradients,
                                           const float* hessians,
                                           const float* weights,
                                           double* node_statistics,
                                           std::size_t node_row_count,
                                           std::size_t target_stride,
                                           std::size_t target_offset) {
  __shared__ double shared_gradient[256];
  __shared__ double shared_hessian[256];
  __shared__ double shared_weight[256];
  __shared__ double shared_gradient_square[256];

  double thread_gradient = 0.0;
  double thread_hessian = 0.0;
  double thread_weight = 0.0;
  double thread_gradient_square = 0.0;
  for (std::size_t row_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       row_index < node_row_count;
       row_index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const std::size_t row = row_indices[row_index];
    const double sample_weight = static_cast<double>(weights[row]);
    const std::size_t target_index = row * target_stride + target_offset;
    const double gradient = static_cast<double>(gradients[target_index]);
    const double hessian = static_cast<double>(hessians[target_index]);
    thread_gradient += sample_weight * gradient;
    thread_hessian += sample_weight * hessian;
    thread_weight += sample_weight;
    thread_gradient_square += sample_weight * gradient * gradient;
  }

  shared_gradient[threadIdx.x] = thread_gradient;
  shared_hessian[threadIdx.x] = thread_hessian;
  shared_weight[threadIdx.x] = thread_weight;
  shared_gradient_square[threadIdx.x] = thread_gradient_square;
  __syncthreads();

  for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      shared_gradient[threadIdx.x] += shared_gradient[threadIdx.x + offset];
      shared_hessian[threadIdx.x] += shared_hessian[threadIdx.x + offset];
      shared_weight[threadIdx.x] += shared_weight[threadIdx.x + offset];
      shared_gradient_square[threadIdx.x] += shared_gradient_square[threadIdx.x + offset];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    atomicAdd(&node_statistics[0], shared_weight[0]);
    atomicAdd(&node_statistics[1], shared_gradient[0]);
    atomicAdd(&node_statistics[2], shared_hessian[0]);
    atomicAdd(&node_statistics[3], shared_gradient_square[0]);
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
