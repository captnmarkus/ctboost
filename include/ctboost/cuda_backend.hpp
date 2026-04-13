#pragma once

#include <array>
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

struct GpuHistogramSnapshot {
  std::vector<float> gradient_sums;
  std::vector<float> hessian_sums;
  std::vector<float> weight_sums;
  GpuNodeStatistics node_statistics;
};

struct GpuFeatureSearchResult {
  std::uint32_t degrees_of_freedom{0};
  std::uint8_t split_valid{0};
  std::uint8_t is_categorical{0};
  std::uint16_t split_bin{0};
  double chi_square{0.0};
  double p_value{1.0};
  double gain{0.0};
  std::uint8_t left_categories[kGpuCategoricalRouteBins]{};
};

struct GpuBestFeatureResult {
  std::int32_t feature_id{-1};
  GpuFeatureSearchResult search_result;
};

struct GpuNodeSearchResult {
  int feature_id{-1};
  double p_value{1.0};
  double chi_square{0.0};
  bool split_valid{false};
  bool is_categorical{false};
  std::uint16_t split_bin{0};
  double gain{0.0};
  std::array<std::uint8_t, kGpuCategoricalRouteBins> left_categories{};
  GpuNodeStatistics node_statistics;
};

bool CudaBackendCompiled() noexcept;
std::string CudaRuntimeVersionString();

GpuHistogramWorkspacePtr CreateGpuHistogramWorkspace(const HistMatrix& hist,
                                                     const std::vector<float>& weights,
                                                     const std::string& devices = "0");
std::size_t EstimateGpuHistogramWorkspaceBytes(const GpuHistogramWorkspace* workspace) noexcept;
void UploadHistogramTargetsGpu(GpuHistogramWorkspace* workspace,
                               const std::vector<float>& gradients,
                               const std::vector<float>& hessians);
void UploadHistogramWeightsGpu(GpuHistogramWorkspace* workspace,
                               const std::vector<float>& weights);
void UploadHistogramTargetMatrixGpu(GpuHistogramWorkspace* workspace,
                                    const std::vector<float>& gradients,
                                    const std::vector<float>& hessians,
                                    std::size_t target_stride);
void SelectHistogramTargetGpuClass(GpuHistogramWorkspace* workspace, std::size_t class_index);
void ResetHistogramRowIndicesGpu(GpuHistogramWorkspace* workspace);
void DownloadHistogramRowIndicesGpu(const GpuHistogramWorkspace* workspace,
                                    std::vector<std::size_t>& out_row_indices);
void DownloadHistogramSnapshotGpu(const GpuHistogramWorkspace* workspace,
                                  GpuHistogramSnapshot* out_snapshot);
void UploadHistogramSnapshotGpu(GpuHistogramWorkspace* workspace,
                                const GpuHistogramSnapshot& snapshot);
std::size_t PartitionHistogramRowsGpu(
    GpuHistogramWorkspace* workspace,
    std::size_t row_begin,
    std::size_t row_end,
    std::size_t feature_index,
    bool is_categorical,
    std::uint16_t split_bin,
    const std::array<std::uint8_t, kGpuCategoricalRouteBins>& left_categories);
void BuildHistogramsGpu(GpuHistogramWorkspace* workspace,
                        std::size_t row_begin,
                        std::size_t row_end,
                        GpuNodeStatistics* out_node_stats);
void SearchBestNodeSplitGpu(GpuHistogramWorkspace* workspace,
                            const std::vector<int>* allowed_features,
                            double lambda_l2,
                            int min_data_in_leaf,
                            double min_child_weight,
                            double min_split_gain,
                            GpuNodeSearchResult* out_result);

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
                   std::vector<float>& out_predictions,
                   const std::string& devices = "0");

}  // namespace ctboost
