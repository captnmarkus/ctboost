#pragma once

#include <cstddef>
#include <cstdint>

#include "ctboost/cuda_backend.hpp"

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
                                              std::size_t target_offset);

__global__ void NodeTargetStatisticsKernel(const std::size_t* row_indices,
                                           const float* gradients,
                                           const float* hessians,
                                           const float* weights,
                                           double* node_statistics,
                                           std::size_t node_row_count,
                                           std::size_t target_stride,
                                           std::size_t target_offset);

__global__ void PredictForestKernel(const std::uint16_t* bins,
                                    const ctboost::GpuTreeNode* nodes,
                                    const std::int32_t* tree_offsets,
                                    float* predictions,
                                    std::size_t num_rows,
                                    std::size_t num_trees,
                                    int prediction_dimension,
                                    float learning_rate);
