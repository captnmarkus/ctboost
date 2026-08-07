#include "booster_internal.hpp"

#include <algorithm>
#include <cmath>
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

template <typename BinType>
void MaterializePredictionFeature(const Pool& pool,
                                  const QuantizationSchema& quantization_schema,
                                  std::size_t feature,
                                  BinType* destination) {
  // Equivalent to BinValueFromSchema, with schema validation and storage selection
  // hoisted out of the per-row loop.
  const std::size_t bins_for_feature = quantization_schema.num_bins(feature);
  if (bins_for_feature == 0 || pool.num_rows() == 0) {
    return;
  }

  const std::size_t cut_begin = quantization_schema.cut_offsets[feature];
  const std::size_t cut_end = quantization_schema.cut_offsets[feature + 1U];
  const auto begin = quantization_schema.cut_values.begin() +
                     static_cast<std::ptrdiff_t>(cut_begin);
  const auto end = quantization_schema.cut_values.begin() +
                   static_cast<std::ptrdiff_t>(cut_end);
  const bool is_categorical = quantization_schema.is_categorical(feature);
  const bool has_missing_values = quantization_schema.has_missing_values(feature);
  const NanMode nan_mode = quantization_schema.nan_mode_for_feature(feature);
  const std::size_t non_missing_bins =
      bins_for_feature - (has_missing_values ? 1U : 0U);
  const std::uint16_t missing_bin =
      has_missing_values
          ? (nan_mode == NanMode::Max
                 ? static_cast<std::uint16_t>(bins_for_feature - 1U)
                 : 0U)
          : (nan_mode == NanMode::Max && bins_for_feature > 0
                 ? static_cast<std::uint16_t>(bins_for_feature - 1U)
                 : 0U);
  const std::uint16_t non_missing_offset =
      has_missing_values && nan_mode == NanMode::Min ? 1U : 0U;
  const float* const contiguous_column = pool.feature_column_ptr(feature);

  for (std::size_t row = 0; row < pool.num_rows(); ++row) {
    const float value =
        contiguous_column != nullptr ? contiguous_column[row] : pool.feature_value(row, feature);
    std::uint16_t bin = 0;
    if (std::isnan(value)) {
      if (nan_mode == NanMode::Forbidden) {
        throw std::invalid_argument("NaN values are not allowed when nan_mode='Forbidden'");
      }
      bin = missing_bin;
    } else if (non_missing_bins == 0) {
      bin = missing_bin;
    } else if (is_categorical) {
      const auto it = std::lower_bound(begin, end, value);
      const std::size_t insertion = static_cast<std::size_t>(std::distance(begin, it));
      const std::size_t clamped_insertion = std::min(insertion, non_missing_bins - 1U);
      bin = static_cast<std::uint16_t>(
          non_missing_offset +
          (it != end && *it == value
               ? static_cast<std::size_t>(std::distance(begin, it))
               : clamped_insertion));
    } else {
      const auto it = std::upper_bound(begin, end, value);
      bin = static_cast<std::uint16_t>(
          non_missing_offset + static_cast<std::size_t>(std::distance(begin, it)));
    }
    destination[row] = static_cast<BinType>(bin);
  }
}

template <typename BinType>
bool HasCompleteContiguousBinStorage(const HistMatrix& hist,
                                     const std::vector<BinType>& storage) noexcept {
  if (hist.uses_external_bin_storage()) {
    return false;
  }
  if (hist.num_rows == 0) {
    return true;
  }
  return hist.num_cols <= storage.size() / hist.num_rows;
}

template <typename BinType>
int PredictContiguousLeafIndex(const std::vector<Node>& nodes,
                               const BinType* bin_indices,
                               std::size_t num_rows,
                               std::size_t row) noexcept {
  int node_index = 0;
  while (!nodes[static_cast<std::size_t>(node_index)].is_leaf) {
    const Node& node = nodes[static_cast<std::size_t>(node_index)];
    const std::size_t offset =
        static_cast<std::size_t>(node.split_feature_id) * num_rows + row;
    const std::uint16_t bin = static_cast<std::uint16_t>(bin_indices[offset]);
    node_index = node.is_categorical_split
                     ? (node.left_categories[bin] != 0 ? node.left_child : node.right_child)
                     : (bin <= node.split_bin_index ? node.left_child : node.right_child);
  }
  return node_index;
}

template <typename BinType>
void UpdatePredictionsFromContiguousBins(const Tree& tree,
                                         const BinType* bin_indices,
                                         std::size_t num_rows,
                                         double learning_rate,
                                         int prediction_dimension,
                                         int class_index,
                                         std::vector<float>& predictions) {
  const std::vector<Node>& nodes = tree.nodes();
  if (nodes.empty()) {
    return;
  }
  if (prediction_dimension == 1) {
    for (std::size_t row = 0; row < num_rows; ++row) {
      const int leaf_index = PredictContiguousLeafIndex(nodes, bin_indices, num_rows, row);
      predictions[row] += learning_rate * nodes[static_cast<std::size_t>(leaf_index)].leaf_weight;
    }
    return;
  }
  for (std::size_t row = 0; row < num_rows; ++row) {
    const int leaf_index = PredictContiguousLeafIndex(nodes, bin_indices, num_rows, row);
    const std::size_t offset =
        row * static_cast<std::size_t>(prediction_dimension) +
        static_cast<std::size_t>(class_index);
    predictions[offset] +=
        learning_rate * nodes[static_cast<std::size_t>(leaf_index)].leaf_weight;
  }
}

template <typename BinType>
void PredictLeafIndicesFromContiguousBins(const Tree& tree,
                                          const BinType* bin_indices,
                                          std::size_t num_rows,
                                          std::vector<int>& leaf_indices) {
  const std::vector<Node>& nodes = tree.nodes();
  if (nodes.empty()) {
    return;
  }
  for (std::size_t row = 0; row < num_rows; ++row) {
    leaf_indices[row] = PredictContiguousLeafIndex(nodes, bin_indices, num_rows, row);
  }
}

}  // namespace

HistMatrix BuildPredictionHist(const Pool& pool, const Tree& reference_tree) {
  return BuildPredictionHist(pool, RequireQuantizationSchema(reference_tree.shared_quantization_schema()));
}

HistMatrix BuildPredictionHist(const Pool& pool, const QuantizationSchema& quantization_schema) {
  return BuildPredictionHist(pool, quantization_schema, nullptr);
}

HistMatrix BuildPredictionHist(const Pool& pool,
                               const QuantizationSchema& quantization_schema,
                               const std::vector<std::uint8_t>* active_features) {
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
  if (hist.num_rows == 0) {
    return hist;
  }

  for (std::size_t feature = 0; feature < hist.num_cols; ++feature) {
    if (active_features != nullptr && feature < active_features->size() &&
        (*active_features)[feature] == 0U) {
      continue;
    }
    const std::size_t offset = feature * hist.num_rows;
    if (hist.bin_storage_bytes() == 1) {
      MaterializePredictionFeature(pool,
                                   quantization_schema,
                                   feature,
                                   hist.compact_bin_indices.data() + offset);
    } else {
      MaterializePredictionFeature(
          pool, quantization_schema, feature, hist.bin_indices.data() + offset);
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
  if (hist.bin_storage_bytes() == 1 &&
      HasCompleteContiguousBinStorage(hist, hist.compact_bin_indices)) {
    UpdatePredictionsFromContiguousBins(tree,
                                        hist.compact_bin_indices.data(),
                                        hist.num_rows,
                                        learning_rate,
                                        prediction_dimension,
                                        class_index,
                                        predictions);
    return;
  }
  if (hist.bin_storage_bytes() == 2 &&
      HasCompleteContiguousBinStorage(hist, hist.bin_indices)) {
    UpdatePredictionsFromContiguousBins(tree,
                                        hist.bin_indices.data(),
                                        hist.num_rows,
                                        learning_rate,
                                        prediction_dimension,
                                        class_index,
                                        predictions);
    return;
  }
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
  if (hist.bin_storage_bytes() == 1 &&
      HasCompleteContiguousBinStorage(hist, hist.compact_bin_indices)) {
    PredictLeafIndicesFromContiguousBins(
        tree, hist.compact_bin_indices.data(), hist.num_rows, leaf_indices);
    return leaf_indices;
  }
  if (hist.bin_storage_bytes() == 2 &&
      HasCompleteContiguousBinStorage(hist, hist.bin_indices)) {
    PredictLeafIndicesFromContiguousBins(
        tree, hist.bin_indices.data(), hist.num_rows, leaf_indices);
    return leaf_indices;
  }
  for (std::size_t row = 0; row < hist.num_rows; ++row) {
    leaf_indices[row] = tree.PredictBinnedLeafIndex(hist, row);
  }
  return leaf_indices;
}

}  // namespace ctboost::booster_detail
