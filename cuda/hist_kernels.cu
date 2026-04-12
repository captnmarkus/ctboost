#include "hist_kernels.cuh"

namespace {

constexpr std::size_t kHistogramChunkBins = 256;
constexpr std::size_t kHistogramRowTileSize = 1024;
constexpr int kMaxGammaIterations = 200;
constexpr double kGammaTolerance = 3e-14;
constexpr double kGammaTiny = 1e-300;
constexpr double kStatisticEpsilon = 1e-7;

__device__ __forceinline__ std::uint16_t ReadBin(const std::uint8_t* bins_u8,
                                                 const std::uint16_t* bins_u16,
                                                 std::uint8_t bin_index_bytes,
                                                 std::size_t index) {
  return bin_index_bytes == 1 ? static_cast<std::uint16_t>(bins_u8[index]) : bins_u16[index];
}

__device__ double RegularizedGammaPSeries(double a, double x) {
  double ap = a;
  double del = 1.0 / a;
  double sum = del;

  for (int n = 1; n <= kMaxGammaIterations; ++n) {
    ap += 1.0;
    del *= x / ap;
    sum += del;
    if (fabs(del) <= fabs(sum) * kGammaTolerance) {
      break;
    }
  }

  return sum * exp(-x + a * log(x) - lgamma(a));
}

__device__ double RegularizedGammaQContinuedFraction(double a, double x) {
  double b = x + 1.0 - a;
  double c = 1.0 / kGammaTiny;
  double d = 1.0 / ((fabs(b) < kGammaTiny) ? kGammaTiny : b);
  double h = d;

  for (int i = 1; i <= kMaxGammaIterations; ++i) {
    const double i_as_double = static_cast<double>(i);
    const double an = -i_as_double * (i_as_double - a);
    b += 2.0;
    d = an * d + b;
    if (fabs(d) < kGammaTiny) {
      d = kGammaTiny;
    }
    c = b + an / c;
    if (fabs(c) < kGammaTiny) {
      c = kGammaTiny;
    }
    d = 1.0 / d;
    const double del = d * c;
    h *= del;
    if (fabs(del - 1.0) <= kGammaTolerance) {
      break;
    }
  }

  return exp(-x + a * log(x) - lgamma(a)) * h;
}

__device__ double RegularizedGammaQ(double a, double x) {
  if (a <= 0.0) {
    return 1.0;
  }
  if (x <= 0.0) {
    return 1.0;
  }
  if (x < a + 1.0) {
    return 1.0 - RegularizedGammaPSeries(a, x);
  }
  return RegularizedGammaQContinuedFraction(a, x);
}

__device__ double ChiSquareSurvivalDevice(double statistic, std::size_t degrees_of_freedom) {
  if (degrees_of_freedom == 0 || statistic <= 0.0) {
    return 1.0;
  }
  return RegularizedGammaQ(0.5 * static_cast<double>(degrees_of_freedom), 0.5 * statistic);
}

__device__ double ComputeGainDevice(double gradient_sum, double hessian_sum, double lambda_l2) {
  return (gradient_sum * gradient_sum) / (hessian_sum + lambda_l2);
}

}  // namespace

__global__ void HistMatrixFeatureChunksKernel(const std::uint8_t* bins_u8,
                                              const std::uint16_t* bins_u16,
                                              std::uint8_t bin_index_bytes,
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
    const std::size_t bin = static_cast<std::size_t>(ReadBin(
        bins_u8, bins_u16, bin_index_bytes, feature_index * total_rows + row));
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

__global__ void EvaluateFeatureSearchKernel(
    const float* gradient_sums,
    const float* hessian_sums,
    const float* weight_sums,
    const std::uint32_t* feature_offsets,
    const std::uint16_t* num_bins_per_feature,
    const std::uint8_t* categorical_mask,
    double total_gradient,
    double total_hessian,
    double sample_weight_sum,
    double gradient_variance,
    double lambda_l2,
    int min_data_in_leaf,
    double min_child_weight,
    double min_split_gain,
    ctboost::GpuFeatureSearchResult* out_results,
    std::size_t num_features) {
  const std::size_t feature = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (feature >= num_features) {
    return;
  }

  ctboost::GpuFeatureSearchResult result{};
  result.p_value = 1.0;
  const std::size_t begin = static_cast<std::size_t>(feature_offsets[feature]);
  const std::size_t num_bins = static_cast<std::size_t>(num_bins_per_feature[feature]);
  if (num_bins <= 1) {
    out_results[feature] = result;
    return;
  }

  std::size_t active_bins = 0;
  for (std::size_t bin = 0; bin < num_bins; ++bin) {
    if (static_cast<double>(weight_sums[begin + bin]) > 0.0) {
      ++active_bins;
    }
  }

  if (sample_weight_sum > 1.0 && active_bins > 1 &&
      gradient_variance > kStatisticEpsilon) {
    result.degrees_of_freedom = static_cast<std::uint32_t>(active_bins - 1);
    const double node_count = sample_weight_sum;
    const double gradient_mean = total_gradient / node_count;
    const double diagonal_scale = (node_count / (node_count - 1.0)) * gradient_variance;
    const double outer_scale = gradient_variance / (node_count - 1.0);

    double diff_quadratic = 0.0;
    double weighted_projection = 0.0;
    double diagonal_projection = 0.0;
    std::size_t processed_bins = 0;
    for (std::size_t bin = 0; bin < num_bins &&
                               processed_bins < static_cast<std::size_t>(result.degrees_of_freedom);
         ++bin) {
      const double bin_weight = static_cast<double>(weight_sums[begin + bin]);
      if (bin_weight <= 0.0) {
        continue;
      }

      const double diff = static_cast<double>(gradient_sums[begin + bin]) - bin_weight * gradient_mean;
      const double diagonal = diagonal_scale * bin_weight + kStatisticEpsilon;
      diff_quadratic += (diff * diff) / diagonal;
      weighted_projection += (bin_weight * diff) / diagonal;
      diagonal_projection += (bin_weight * bin_weight) / diagonal;
      ++processed_bins;
    }

    const double denominator = 1.0 - outer_scale * diagonal_projection;
    if (denominator > kStatisticEpsilon) {
      result.chi_square =
          diff_quadratic + outer_scale * weighted_projection * weighted_projection / denominator;
      result.p_value = ChiSquareSurvivalDevice(result.chi_square, result.degrees_of_freedom);
    }
  }

  const double parent_gain = ComputeGainDevice(total_gradient, total_hessian, lambda_l2);
  const bool is_categorical = categorical_mask[feature] != 0U;
  result.is_categorical = is_categorical ? 1U : 0U;

  if (!is_categorical) {
    double left_gradient = 0.0;
    double left_hessian = 0.0;
    double left_count = 0.0;
    for (std::size_t split_bin = 0; split_bin + 1 < num_bins; ++split_bin) {
      left_gradient += static_cast<double>(gradient_sums[begin + split_bin]);
      left_hessian += static_cast<double>(hessian_sums[begin + split_bin]);
      left_count += static_cast<double>(weight_sums[begin + split_bin]);

      const double right_count = sample_weight_sum - left_count;
      if (left_count <= 0.0 || right_count <= 0.0 ||
          left_count < static_cast<double>(min_data_in_leaf) ||
          right_count < static_cast<double>(min_data_in_leaf)) {
        continue;
      }

      const double right_gradient = total_gradient - left_gradient;
      const double right_hessian = total_hessian - left_hessian;
      if (left_hessian < min_child_weight || right_hessian < min_child_weight) {
        continue;
      }

      const double gain = ComputeGainDevice(left_gradient, left_hessian, lambda_l2) +
                          ComputeGainDevice(right_gradient, right_hessian, lambda_l2) - parent_gain;
      if (gain <= min_split_gain) {
        continue;
      }

      if (result.split_valid == 0U || gain > result.gain) {
        result.split_valid = 1U;
        result.split_bin = static_cast<std::uint16_t>(split_bin);
        result.gain = gain;
      }
    }
    out_results[feature] = result;
    return;
  }

  if (num_bins > ctboost::kGpuCategoricalRouteBins) {
    out_results[feature] = result;
    return;
  }

  std::uint16_t active_bin_ids[ctboost::kGpuCategoricalRouteBins];
  double active_gradients[ctboost::kGpuCategoricalRouteBins];
  double active_hessians[ctboost::kGpuCategoricalRouteBins];
  double active_counts[ctboost::kGpuCategoricalRouteBins];
  double active_weights[ctboost::kGpuCategoricalRouteBins];
  std::size_t active_count = 0;
  for (std::size_t bin = 0; bin < num_bins; ++bin) {
    const double bin_count = static_cast<double>(weight_sums[begin + bin]);
    if (bin_count <= 0.0) {
      continue;
    }
    const double bin_gradient = static_cast<double>(gradient_sums[begin + bin]);
    const double bin_hessian = static_cast<double>(hessian_sums[begin + bin]);
    const double denominator = bin_hessian + lambda_l2;
    active_bin_ids[active_count] = static_cast<std::uint16_t>(bin);
    active_gradients[active_count] = bin_gradient;
    active_hessians[active_count] = bin_hessian;
    active_counts[active_count] = bin_count;
    active_weights[active_count] = denominator > 0.0 ? bin_gradient / denominator : 0.0;
    ++active_count;
  }

  for (std::size_t i = 1; i < active_count; ++i) {
    const std::uint16_t bin_id = active_bin_ids[i];
    const double gradient_value = active_gradients[i];
    const double hessian_value = active_hessians[i];
    const double count_value = active_counts[i];
    const double weight_value = active_weights[i];
    std::size_t j = i;
    while (j > 0 &&
           (active_weights[j - 1] > weight_value ||
            (active_weights[j - 1] == weight_value && active_bin_ids[j - 1] > bin_id))) {
      active_bin_ids[j] = active_bin_ids[j - 1];
      active_gradients[j] = active_gradients[j - 1];
      active_hessians[j] = active_hessians[j - 1];
      active_counts[j] = active_counts[j - 1];
      active_weights[j] = active_weights[j - 1];
      --j;
    }
    active_bin_ids[j] = bin_id;
    active_gradients[j] = gradient_value;
    active_hessians[j] = hessian_value;
    active_counts[j] = count_value;
    active_weights[j] = weight_value;
  }

  double left_gradient = 0.0;
  double left_hessian = 0.0;
  double left_count = 0.0;
  for (std::size_t split_index = 0; split_index + 1 < active_count; ++split_index) {
    left_gradient += active_gradients[split_index];
    left_hessian += active_hessians[split_index];
    left_count += active_counts[split_index];

    const double right_count = sample_weight_sum - left_count;
    if (left_count <= 0.0 || right_count <= 0.0 ||
        left_count < static_cast<double>(min_data_in_leaf) ||
        right_count < static_cast<double>(min_data_in_leaf)) {
      continue;
    }

    const double right_gradient = total_gradient - left_gradient;
    const double right_hessian = total_hessian - left_hessian;
    if (left_hessian < min_child_weight || right_hessian < min_child_weight) {
      continue;
    }

    const double gain = ComputeGainDevice(left_gradient, left_hessian, lambda_l2) +
                        ComputeGainDevice(right_gradient, right_hessian, lambda_l2) - parent_gain;
    if (gain <= min_split_gain) {
      continue;
    }

    if (result.split_valid == 0U || gain > result.gain) {
      result.split_valid = 1U;
      result.gain = gain;
      for (std::size_t i = 0; i < ctboost::kGpuCategoricalRouteBins; ++i) {
        result.left_categories[i] = 0U;
      }
      for (std::size_t left_index = 0; left_index <= split_index; ++left_index) {
        result.left_categories[active_bin_ids[left_index]] = 1U;
      }
    }
  }

  out_results[feature] = result;
}

__global__ void PredictForestKernel(const std::uint8_t* bins_u8,
                                    const std::uint16_t* bins_u16,
                                    std::uint8_t bin_index_bytes,
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
      const std::size_t bin = static_cast<std::size_t>(ReadBin(
          bins_u8,
          bins_u16,
          bin_index_bytes,
          static_cast<std::size_t>(node.split_feature_id) * num_rows + row));
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
