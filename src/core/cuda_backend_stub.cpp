#include "ctboost/cuda_backend.hpp"
#include "ctboost/histogram.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ctboost {

struct GpuHistogramWorkspace {};

bool CudaBackendCompiled() noexcept { return false; }

std::string CudaRuntimeVersionString() { return "not compiled"; }

void DestroyGpuHistogramWorkspace(GpuHistogramWorkspace* workspace) noexcept {
  delete workspace;
}

GpuHistogramWorkspacePtr CreateGpuHistogramWorkspace(const HistMatrix& hist,
                                                     const std::vector<float>& weights) {
  (void)hist;
  (void)weights;
  throw std::runtime_error("CUDA histogram builder requested but CTBoost was compiled without CUDA");
}

std::size_t EstimateGpuHistogramWorkspaceBytes(const GpuHistogramWorkspace* workspace) noexcept {
  (void)workspace;
  return 0;
}

void UploadHistogramTargetsGpu(GpuHistogramWorkspace* workspace,
                               const std::vector<float>& gradients,
                               const std::vector<float>& hessians) {
  (void)workspace;
  (void)gradients;
  (void)hessians;
  throw std::runtime_error("CUDA histogram builder requested but CTBoost was compiled without CUDA");
}

void UploadHistogramTargetMatrixGpu(GpuHistogramWorkspace* workspace,
                                    const std::vector<float>& gradients,
                                    const std::vector<float>& hessians,
                                    std::size_t target_stride) {
  (void)workspace;
  (void)gradients;
  (void)hessians;
  (void)target_stride;
  throw std::runtime_error("CUDA histogram builder requested but CTBoost was compiled without CUDA");
}

void SelectHistogramTargetGpuClass(GpuHistogramWorkspace* workspace, std::size_t class_index) {
  (void)workspace;
  (void)class_index;
  throw std::runtime_error("CUDA histogram builder requested but CTBoost was compiled without CUDA");
}

void ResetHistogramRowIndicesGpu(GpuHistogramWorkspace* workspace) {
  (void)workspace;
  throw std::runtime_error("CUDA histogram builder requested but CTBoost was compiled without CUDA");
}

void DownloadHistogramRowIndicesGpu(const GpuHistogramWorkspace* workspace,
                                    std::vector<std::size_t>& out_row_indices) {
  (void)workspace;
  (void)out_row_indices;
  throw std::runtime_error("CUDA histogram builder requested but CTBoost was compiled without CUDA");
}

std::size_t PartitionHistogramRowsGpu(
    GpuHistogramWorkspace* workspace,
    std::size_t row_begin,
    std::size_t row_end,
    std::size_t feature_index,
    bool is_categorical,
    std::uint16_t split_bin,
    const std::array<std::uint8_t, kGpuCategoricalRouteBins>& left_categories) {
  (void)workspace;
  (void)row_begin;
  (void)row_end;
  (void)feature_index;
  (void)is_categorical;
  (void)split_bin;
  (void)left_categories;
  throw std::runtime_error("CUDA histogram builder requested but CTBoost was compiled without CUDA");
}

void BuildHistogramsGpu(GpuHistogramWorkspace* workspace,
                        std::size_t row_begin,
                        std::size_t row_end,
                        GpuNodeStatistics* out_node_stats) {
  (void)workspace;
  (void)row_begin;
  (void)row_end;
  (void)out_node_stats;
  throw std::runtime_error("CUDA histogram builder requested but CTBoost was compiled without CUDA");
}

void SearchBestNodeSplitGpu(GpuHistogramWorkspace* workspace,
                            const std::vector<int>* allowed_features,
                            double lambda_l2,
                            int min_data_in_leaf,
                            double min_child_weight,
                            double min_split_gain,
                            GpuNodeSearchResult* out_result) {
  (void)workspace;
  (void)allowed_features;
  (void)lambda_l2;
  (void)min_data_in_leaf;
  (void)min_child_weight;
  (void)min_split_gain;
  (void)out_result;
  throw std::runtime_error("CUDA histogram builder requested but CTBoost was compiled without CUDA");
}

void PredictRawGpu(const HistMatrix& hist,
                   const std::vector<GpuTreeNode>& nodes,
                   const std::vector<std::int32_t>& tree_offsets,
                   float learning_rate,
                   int prediction_dimension,
                   std::vector<float>& out_predictions) {
  (void)hist;
  (void)nodes;
  (void)tree_offsets;
  (void)learning_rate;
  (void)prediction_dimension;
  (void)out_predictions;
  throw std::runtime_error("CUDA prediction requested but CTBoost was compiled without CUDA");
}

}  // namespace ctboost
