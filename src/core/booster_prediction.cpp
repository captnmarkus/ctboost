#include "booster_internal.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ctboost::booster_detail {
namespace {

bool CanUseCompactBins(const QuantizationSchema& quantization_schema) {
  for (const std::uint16_t feature_bins_count : quantization_schema.num_bins_per_feature) {
    if (feature_bins_count >
        static_cast<std::uint16_t>(std::numeric_limits<std::uint8_t>::max()) + 1U) {
      return false;
    }
  }
  return true;
}

}  // namespace

HistMatrix BuildPredictionHist(const Pool& pool, const Tree& reference_tree) {
  return BuildPredictionHist(pool, RequireQuantizationSchema(reference_tree.shared_quantization_schema()));
}

HistMatrix BuildPredictionHist(const Pool& pool, const QuantizationSchema& quantization_schema) {
  if (pool.num_cols() != quantization_schema.num_cols()) {
    throw std::invalid_argument(
        "prediction pool must have the same number of columns as the fitted model");
  }

  HistMatrix hist;
  hist.num_rows = pool.num_rows();
  hist.num_cols = pool.num_cols();
  ApplyQuantizationSchema(quantization_schema, hist);
  if (CanUseCompactBins(quantization_schema)) {
    hist.compact_bin_indices.resize(hist.num_rows * hist.num_cols, 0);
    hist.bin_index_bytes = 1;
  } else {
    hist.bin_indices.resize(hist.num_rows * hist.num_cols, 0);
    hist.bin_index_bytes = 2;
  }

  for (std::size_t feature = 0; feature < hist.num_cols; ++feature) {
    const float* const contiguous_column = pool.feature_column_ptr(feature);
    for (std::size_t row = 0; row < hist.num_rows; ++row) {
      const float value =
          contiguous_column != nullptr ? contiguous_column[row] : pool.feature_value(row, feature);
      hist.set_bin_index(feature, row, quantization_schema.bin_value(feature, value));
    }
  }
  return hist;
}

std::vector<GpuTreeNode> FlattenTreesForGpu(const std::vector<Tree>& trees,
                                            std::size_t tree_limit,
                                            const std::vector<double>& tree_learning_rates,
                                            double default_learning_rate,
                                            int prediction_dimension,
                                            std::vector<std::int32_t>& tree_offsets) {
  std::size_t total_nodes = 0;
  for (std::size_t tree_index = 0; tree_index < tree_limit; ++tree_index) {
    total_nodes += trees[tree_index].nodes().size();
  }

  std::vector<GpuTreeNode> flattened_nodes;
  flattened_nodes.reserve(total_nodes);
  tree_offsets.clear();
  tree_offsets.reserve(tree_limit);
  for (std::size_t tree_index = 0; tree_index < tree_limit; ++tree_index) {
    const auto& tree_nodes = trees[tree_index].nodes();
    const float tree_learning_rate = static_cast<float>(ResolveIterationLearningRate(
        tree_learning_rates, tree_index, prediction_dimension, default_learning_rate));
    const std::int32_t tree_offset = static_cast<std::int32_t>(flattened_nodes.size());
    tree_offsets.push_back(tree_offset);
    for (const Node& node : tree_nodes) {
      GpuTreeNode gpu_node;
      gpu_node.is_leaf = node.is_leaf ? 1U : 0U;
      gpu_node.is_categorical_split = node.is_categorical_split ? 1U : 0U;
      gpu_node.split_bin_index = node.split_bin_index;
      gpu_node.split_feature_id = static_cast<std::int32_t>(node.split_feature_id);
      gpu_node.left_child = node.left_child < 0 ? -1 : tree_offset + static_cast<std::int32_t>(node.left_child);
      gpu_node.right_child = node.right_child < 0 ? -1 : tree_offset + static_cast<std::int32_t>(node.right_child);
      gpu_node.leaf_weight = node.leaf_weight * tree_learning_rate;
      std::copy(node.left_categories.begin(), node.left_categories.end(), gpu_node.left_categories);
      flattened_nodes.push_back(std::move(gpu_node));
    }
  }
  return flattened_nodes;
}

void UpdatePredictions(const Tree& tree,
                       const HistMatrix& hist,
                       double learning_rate,
                       int prediction_dimension,
                       int class_index,
                       std::vector<float>& predictions) {
  if (prediction_dimension == 1) {
    for (std::size_t row = 0; row < hist.num_rows; ++row) {
      predictions[row] += learning_rate * tree.PredictBinnedRow(hist, row);
    }
    return;
  }
  for (std::size_t row = 0; row < hist.num_rows; ++row) {
    const std::size_t offset = row * static_cast<std::size_t>(prediction_dimension) + class_index;
    predictions[offset] += learning_rate * tree.PredictBinnedRow(hist, row);
  }
}

void AccumulateIterationPredictions(const std::vector<Tree>& trees,
                                    std::size_t iteration_index,
                                    const HistMatrix& hist,
                                    const std::vector<double>& tree_learning_rates,
                                    double default_learning_rate,
                                    int prediction_dimension,
                                    std::vector<float>& predictions) {
  const std::size_t tree_begin = iteration_index * static_cast<std::size_t>(prediction_dimension);
  const std::size_t tree_end = tree_begin + static_cast<std::size_t>(prediction_dimension);
  for (std::size_t tree_index = tree_begin; tree_index < tree_end; ++tree_index) {
    const double tree_learning_rate = ResolveIterationLearningRate(
        tree_learning_rates, tree_index, prediction_dimension, default_learning_rate);
    const int class_index =
        prediction_dimension == 1 ? 0 : static_cast<int>(tree_index % static_cast<std::size_t>(prediction_dimension));
    UpdatePredictions(trees[tree_index],
                      hist,
                      tree_learning_rate,
                      prediction_dimension,
                      class_index,
                      predictions);
  }
}

std::vector<float> PredictFromHist(const std::vector<Tree>& trees,
                                   const HistMatrix& hist,
                                   std::size_t tree_limit,
                                   const std::vector<double>& tree_learning_rates,
                                   double default_learning_rate,
                                   bool use_gpu,
                                   int prediction_dimension,
                                   const std::string& devices) {
  std::vector<float> predictions(hist.num_rows * static_cast<std::size_t>(prediction_dimension), 0.0F);
  if (tree_limit == 0 || hist.num_rows == 0) {
    return predictions;
  }
  if (use_gpu && CudaBackendCompiled()) {
    std::vector<std::int32_t> tree_offsets;
    const std::vector<GpuTreeNode> flattened_nodes = FlattenTreesForGpu(
        trees, tree_limit, tree_learning_rates, default_learning_rate, prediction_dimension, tree_offsets);
    PredictRawGpu(hist, flattened_nodes, tree_offsets, 1.0F, prediction_dimension, predictions, devices);
    return predictions;
  }
  for (std::size_t tree_index = 0; tree_index < tree_limit; ++tree_index) {
    const double tree_learning_rate = ResolveIterationLearningRate(
        tree_learning_rates, tree_index, prediction_dimension, default_learning_rate);
    const int class_index =
        prediction_dimension == 1 ? 0 : static_cast<int>(tree_index % static_cast<std::size_t>(prediction_dimension));
    UpdatePredictions(trees[tree_index],
                      hist,
                      tree_learning_rate,
                      prediction_dimension,
                      class_index,
                      predictions);
  }
  return predictions;
}

void UpdatePredictions(const Tree& tree,
                       const Pool& pool,
                       double learning_rate,
                       int prediction_dimension,
                       int class_index,
                       std::vector<float>& predictions) {
  if (prediction_dimension == 1) {
    for (std::size_t row = 0; row < pool.num_rows(); ++row) {
      predictions[row] += learning_rate * tree.PredictRow(pool, row);
    }
    return;
  }
  for (std::size_t row = 0; row < pool.num_rows(); ++row) {
    const std::size_t offset = row * static_cast<std::size_t>(prediction_dimension) + class_index;
    predictions[offset] += learning_rate * tree.PredictRow(pool, row);
  }
}

std::vector<int> PredictLeafIndicesFromHist(const Tree& tree, const HistMatrix& hist) {
  std::vector<int> leaf_indices(hist.num_rows, -1);
  for (std::size_t row = 0; row < hist.num_rows; ++row) {
    leaf_indices[row] = tree.PredictBinnedLeafIndex(hist, row);
  }
  return leaf_indices;
}

}  // namespace ctboost::booster_detail
