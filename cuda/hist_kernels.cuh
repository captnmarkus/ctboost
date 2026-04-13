#pragma once

#include <cstddef>
#include <cstdint>

#include "ctboost/cuda_backend.hpp"

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
                                              std::size_t target_offset);

__global__ void NodeTargetStatisticsKernel(const std::size_t* row_indices,
                                           const float* gradients,
                                           const float* hessians,
                                           const float* weights,
                                           double* node_statistics,
                                           std::size_t node_row_count,
                                           std::size_t target_stride,
                                           std::size_t target_offset);

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
    std::size_t num_features);

__global__ void SelectBestFeatureKernel(
    const ctboost::GpuFeatureSearchResult* feature_results,
    const std::uint32_t* candidate_feature_indices,
    std::size_t num_candidates,
    ctboost::GpuBestFeatureResult* out_best_result);

__global__ void PredictForestKernel(const std::uint8_t* bins_u8,
                                    const std::uint16_t* bins_u16,
                                    std::uint8_t bin_index_bytes,
                                    const ctboost::GpuTreeNode* nodes,
                                    const std::int32_t* tree_offsets,
                                    float* predictions,
                                    std::size_t num_rows,
                                    std::size_t num_trees,
                                    int prediction_dimension,
                                    float learning_rate);
