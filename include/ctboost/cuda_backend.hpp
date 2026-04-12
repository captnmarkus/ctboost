#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ctboost {

struct HistMatrix;

inline constexpr std::size_t kGpuCategoricalRouteBins = 256;

struct GpuHistogramWorkspace;
void DestroyGpuHistogramWorkspace(GpuHistogramWorkspace* workspace) noexcept;
using GpuHistogramWorkspacePtr =
    std::unique_ptr<GpuHistogramWorkspace, void (*)(GpuHistogramWorkspace*)>;

struct GpuNodeStatistics {
  double sample_weight_sum{0.0};
  double total_gradient{0.0};
  double total_hessian{0.0};
  double gradient_square_sum{0.0};
};

bool CudaBackendCompiled() noexcept;
std::string CudaRuntimeVersionString();

GpuHistogramWorkspacePtr CreateGpuHistogramWorkspace(const HistMatrix& hist,
                                                     const std::vector<float>& weights);
void UploadHistogramTargetsGpu(GpuHistogramWorkspace* workspace,
                               const std::vector<float>& gradients,
                               const std::vector<float>& hessians);
void UploadHistogramTargetMatrixGpu(GpuHistogramWorkspace* workspace,
                                    const std::vector<float>& gradients,
                                    const std::vector<float>& hessians,
                                    std::size_t target_stride);
void SelectHistogramTargetGpuClass(GpuHistogramWorkspace* workspace, std::size_t class_index);
void BuildHistogramsGpu(GpuHistogramWorkspace* workspace,
                        const std::vector<std::size_t>& row_indices,
                        std::size_t row_begin,
                        std::size_t row_end,
                        std::vector<float>& out_gradient_sums,
                        std::vector<float>& out_hessian_sums,
                        std::vector<float>& out_weight_sums,
                        std::vector<std::size_t>& out_feature_offsets,
                        GpuNodeStatistics* out_node_stats);

struct GpuTreeNode {
  std::uint8_t is_leaf{1};
  std::uint8_t is_categorical_split{0};
  std::uint16_t split_bin_index{0};
  std::int32_t split_feature_id{-1};
  std::int32_t left_child{-1};
  std::int32_t right_child{-1};
  float leaf_weight{0.0F};
  std::uint8_t left_categories[kGpuCategoricalRouteBins]{};
};

void PredictRawGpu(const HistMatrix& hist,
                   const std::vector<GpuTreeNode>& nodes,
                   const std::vector<std::int32_t>& tree_offsets,
                   float learning_rate,
                   int prediction_dimension,
                   std::vector<float>& out_predictions);

}  // namespace ctboost
