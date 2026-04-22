#include "tree_internal.hpp"

#include <memory>
#include <stdexcept>

namespace ctboost {

float Tree::PredictRow(const Pool& pool, std::size_t row) const {
  const int leaf_index = PredictLeafIndex(pool, row);
  return leaf_index < 0 ? 0.0F : nodes_[leaf_index].leaf_weight;
}

int Tree::PredictLeafIndex(const Pool& pool, std::size_t row) const {
  if (nodes_.empty()) {
    return -1;
  }

  int node_index = 0;
  while (!nodes_[node_index].is_leaf) {
    const Node& node = nodes_[node_index];
    const std::uint16_t bin = BinValue(
        static_cast<std::size_t>(node.split_feature_id),
        pool.feature_value(row, static_cast<std::size_t>(node.split_feature_id)));
    node_index = node.is_categorical_split
                     ? (node.left_categories[bin] != 0 ? node.left_child : node.right_child)
                     : (bin <= node.split_bin_index ? node.left_child : node.right_child);
  }
  return node_index;
}

void Tree::AccumulateContributions(
    const Pool& pool, std::size_t row, float scale, std::vector<float>& row_contributions) const {
  if (row_contributions.empty()) {
    return;
  }
  if (nodes_.empty()) {
    row_contributions.back() += scale * 0.0F;
    return;
  }

  std::vector<int> path_features;
  int node_index = 0;
  while (!nodes_[node_index].is_leaf) {
    const Node& node = nodes_[node_index];
    path_features.push_back(node.split_feature_id);
    const std::uint16_t bin = BinValue(
        static_cast<std::size_t>(node.split_feature_id),
        pool.feature_value(row, static_cast<std::size_t>(node.split_feature_id)));
    node_index = node.is_categorical_split
                     ? (node.left_categories[bin] != 0 ? node.left_child : node.right_child)
                     : (bin <= node.split_bin_index ? node.left_child : node.right_child);
  }

  const float leaf_value = scale * nodes_[node_index].leaf_weight;
  if (path_features.empty()) {
    row_contributions.back() += leaf_value;
    return;
  }

  const float share = leaf_value / static_cast<float>(path_features.size());
  for (const int feature_index : path_features) {
    row_contributions[static_cast<std::size_t>(feature_index)] += share;
  }
}

float Tree::PredictBinnedRow(const HistMatrix& hist, std::size_t row) const {
  const int leaf_index = PredictBinnedLeafIndex(hist, row);
  return leaf_index < 0 ? 0.0F : nodes_[leaf_index].leaf_weight;
}

int Tree::PredictBinnedLeafIndex(const HistMatrix& hist, std::size_t row) const {
  if (nodes_.empty()) {
    return -1;
  }

  int node_index = 0;
  while (!nodes_[node_index].is_leaf) {
    const Node& node = nodes_[node_index];
    const auto feature_bins = hist.feature_bins(static_cast<std::size_t>(node.split_feature_id));
    const std::uint16_t bin = feature_bins[row];
    node_index = node.is_categorical_split
                     ? (node.left_categories[bin] != 0 ? node.left_child : node.right_child)
                     : (bin <= node.split_bin_index ? node.left_child : node.right_child);
  }
  return node_index;
}

std::vector<float> Tree::Predict(const Pool& pool) const {
  std::vector<float> predictions(pool.num_rows(), 0.0F);
  for (std::size_t row = 0; row < pool.num_rows(); ++row) {
    predictions[row] = PredictRow(pool, row);
  }
  return predictions;
}

void Tree::SetLeafWeight(std::size_t node_index, float leaf_weight) {
  if (node_index >= nodes_.size()) {
    throw std::out_of_range("node index is out of bounds");
  }
  if (!nodes_[node_index].is_leaf) {
    throw std::invalid_argument("leaf weight can only be set on leaf nodes");
  }
  nodes_[node_index].leaf_weight = leaf_weight;
}

void Tree::SetQuantizationSchema(const QuantizationSchemaPtr& quantization_schema) {
  quantization_schema_ = quantization_schema;
}

const QuantizationSchemaPtr& Tree::shared_quantization_schema() const noexcept {
  return quantization_schema_;
}

void Tree::LoadState(std::vector<Node> nodes,
                     const QuantizationSchemaPtr& quantization_schema,
                     std::vector<double> feature_importances) {
  nodes_ = std::move(nodes);
  quantization_schema_ = quantization_schema;
  feature_importances_ = std::move(feature_importances);
}

void Tree::LoadState(std::vector<Node> nodes,
                     std::vector<std::uint16_t> num_bins_per_feature,
                     std::vector<std::size_t> cut_offsets,
                     std::vector<float> cut_values,
                     std::vector<std::uint8_t> categorical_mask,
                     std::vector<std::uint8_t> missing_value_mask,
                     std::uint8_t nan_mode,
                     std::vector<double> feature_importances) {
  auto quantization_schema = std::make_shared<QuantizationSchema>();
  quantization_schema->num_bins_per_feature = std::move(num_bins_per_feature);
  quantization_schema->cut_offsets = std::move(cut_offsets);
  quantization_schema->cut_values = std::move(cut_values);
  quantization_schema->categorical_mask = std::move(categorical_mask);
  quantization_schema->missing_value_mask = std::move(missing_value_mask);
  quantization_schema->nan_mode = nan_mode;
  LoadState(std::move(nodes), quantization_schema, std::move(feature_importances));
}

const std::vector<Node>& Tree::nodes() const noexcept { return nodes_; }

const std::vector<std::uint16_t>& Tree::num_bins_per_feature() const {
  return detail::RequireQuantizationSchema(quantization_schema_).num_bins_per_feature;
}

const std::vector<std::size_t>& Tree::cut_offsets() const {
  return detail::RequireQuantizationSchema(quantization_schema_).cut_offsets;
}

const std::vector<float>& Tree::cut_values() const {
  return detail::RequireQuantizationSchema(quantization_schema_).cut_values;
}

const std::vector<std::uint8_t>& Tree::categorical_mask() const {
  return detail::RequireQuantizationSchema(quantization_schema_).categorical_mask;
}

const std::vector<std::uint8_t>& Tree::missing_value_mask() const {
  return detail::RequireQuantizationSchema(quantization_schema_).missing_value_mask;
}

std::uint8_t Tree::nan_mode() const {
  return detail::RequireQuantizationSchema(quantization_schema_).nan_mode;
}

const std::vector<double>& Tree::feature_importances() const noexcept {
  return feature_importances_;
}

std::uint16_t Tree::BinValue(std::size_t feature_index, float value) const {
  return detail::RequireQuantizationSchema(quantization_schema_).bin_value(feature_index, value);
}

}  // namespace ctboost
