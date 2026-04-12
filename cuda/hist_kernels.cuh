#pragma once

#include <cstddef>
#include <cstdint>

#include "ctboost/cuda_backend.hpp"

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
                                            std::size_t feature_bin_count);

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
                                                   std::size_t feature_bin_count);

__global__ void PredictForestKernel(const std::uint16_t* bins,
                                    const ctboost::GpuTreeNode* nodes,
                                    const std::int32_t* tree_offsets,
                                    float* predictions,
                                    std::size_t num_rows,
                                    std::size_t num_trees,
                                    int prediction_dimension,
                                    float learning_rate);
