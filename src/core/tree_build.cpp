#include "tree_internal.hpp"

#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>

#include "ctboost/cuda_backend.hpp"
#include "ctboost/profiler.hpp"

namespace ctboost {

void Tree::Build(const HistMatrix& hist,
                 const std::vector<float>& gradients,
                 const std::vector<float>& hessians,
                 const std::vector<float>& weights,
                 const TreeBuildOptions& options,
                 GpuHistogramWorkspace* gpu_workspace,
                 const TrainingProfiler* profiler,
                 std::vector<std::size_t>* row_indices_out,
                 std::vector<LeafRowRange>* leaf_row_ranges_out,
                 const QuantizationSchemaPtr& quantization_schema) {
  if (weights.size() != hist.num_rows) {
    throw std::invalid_argument("weight size must match the histogram row count");
  }
  if (options.use_gpu) {
    if (gpu_workspace == nullptr) {
      throw std::invalid_argument("GPU histogram workspace must be provided when task_type='GPU'");
    }
    if ((!gradients.empty() && gradients.size() != hist.num_rows) ||
        (!hessians.empty() && hessians.size() != hist.num_rows)) {
      throw std::invalid_argument(
          "non-empty gradient and hessian buffers must match the histogram row count");
    }
  } else if (gradients.size() != hist.num_rows || hessians.size() != hist.num_rows) {
    throw std::invalid_argument(
        "gradient, hessian, and weight sizes must match the histogram row count");
  }

  nodes_.clear();
  if (quantization_schema != nullptr) {
    if (quantization_schema->num_cols() != hist.num_cols) {
      throw std::invalid_argument(
          "tree quantization schema feature count must match the histogram feature count");
    }
    quantization_schema_ = quantization_schema;
  } else {
    quantization_schema_ = std::make_shared<QuantizationSchema>(MakeQuantizationSchema(hist));
  }
  feature_importances_.assign(hist.num_cols, 0.0);

  std::vector<std::size_t> row_indices;
  const std::size_t initial_row_count = hist.num_rows;
  if (!options.use_gpu) {
    row_indices.resize(hist.num_rows);
    std::iota(row_indices.begin(), row_indices.end(), 0);
  } else {
    ResetHistogramRowIndicesGpu(gpu_workspace);
  }
  if (leaf_row_ranges_out != nullptr) {
    leaf_row_ranges_out->clear();
  }

  int leaf_count = 1;
  const LinearStatistic statistic_engine;
  const double root_leaf_lower_bound =
      options.max_leaf_weight > 0.0 ? -options.max_leaf_weight : -std::numeric_limits<double>::infinity();
  const double root_leaf_upper_bound =
      options.max_leaf_weight > 0.0 ? options.max_leaf_weight : std::numeric_limits<double>::infinity();
  BuildNode(hist,
            gradients,
            hessians,
            weights,
            row_indices,
            0,
            initial_row_count,
            0,
            options,
            gpu_workspace,
            nullptr,
            false,
            nullptr,
            0.0,
            options.allowed_features,
            nullptr,
            root_leaf_lower_bound,
            root_leaf_upper_bound,
            profiler,
            statistic_engine,
            leaf_row_ranges_out,
            &leaf_count);
  if (row_indices_out != nullptr) {
    if (options.use_gpu) {
      DownloadHistogramRowIndicesGpu(gpu_workspace, *row_indices_out);
    } else {
      *row_indices_out = std::move(row_indices);
    }
  }
}

}  // namespace ctboost
