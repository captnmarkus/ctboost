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

void BuildHistogramsGpu(GpuHistogramWorkspace* workspace,
                        const std::vector<std::size_t>& row_indices,
                        std::size_t row_begin,
                        std::size_t row_end,
                        std::vector<float>& out_gradient_sums,
                        std::vector<float>& out_hessian_sums,
                        std::vector<float>& out_weight_sums,
                        std::vector<std::size_t>& out_feature_offsets,
                        GpuNodeStatistics* out_node_stats) {
  (void)workspace;
  (void)row_indices;
  (void)row_begin;
  (void)row_end;
  (void)out_gradient_sums;
  (void)out_hessian_sums;
  (void)out_weight_sums;
  (void)out_feature_offsets;
  (void)out_node_stats;
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
