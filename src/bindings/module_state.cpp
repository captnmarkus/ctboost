#include "module_internal.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>

namespace ctboost::bindings {

py::dict QuantizationSchemaToStateDict(const ctboost::QuantizationSchema& quantization_schema) {
  py::dict state;
  state["num_bins_per_feature"] = quantization_schema.num_bins_per_feature;
  state["cut_offsets"] = quantization_schema.cut_offsets;
  state["cut_values"] = quantization_schema.cut_values;
  state["categorical_mask"] = quantization_schema.categorical_mask;
  state["missing_value_mask"] = quantization_schema.missing_value_mask;
  state["nan_mode"] = quantization_schema.nan_mode;
  state["nan_modes"] = quantization_schema.nan_modes;
  return state;
}

ctboost::QuantizationSchemaPtr QuantizationSchemaFromStateDict(const py::handle& handle) {
  const py::dict state = handle.cast<py::dict>();
  auto quantization_schema = std::make_shared<ctboost::QuantizationSchema>();
  quantization_schema->num_bins_per_feature =
      py::cast<std::vector<std::uint16_t>>(state["num_bins_per_feature"]);
  quantization_schema->cut_offsets = py::cast<std::vector<std::size_t>>(state["cut_offsets"]);
  quantization_schema->cut_values = py::cast<std::vector<float>>(state["cut_values"]);
  quantization_schema->categorical_mask =
      py::cast<std::vector<std::uint8_t>>(state["categorical_mask"]);
  quantization_schema->missing_value_mask =
      py::cast<std::vector<std::uint8_t>>(state["missing_value_mask"]);
  quantization_schema->nan_mode = py::cast<std::uint8_t>(state["nan_mode"]);
  if (state.contains("nan_modes")) {
    quantization_schema->nan_modes = py::cast<std::vector<std::uint8_t>>(state["nan_modes"]);
  } else {
    quantization_schema->nan_modes.assign(
        quantization_schema->num_bins_per_feature.size(), quantization_schema->nan_mode);
  }
  return quantization_schema;
}

ctboost::QuantizationSchemaPtr QuantizationSchemaFromTreeStateDict(const py::dict& state) {
  py::dict quantization_schema_state;
  quantization_schema_state["num_bins_per_feature"] = state["num_bins_per_feature"];
  quantization_schema_state["cut_offsets"] = state["cut_offsets"];
  quantization_schema_state["cut_values"] = state["cut_values"];
  quantization_schema_state["categorical_mask"] = state["categorical_mask"];
  quantization_schema_state["missing_value_mask"] = state["missing_value_mask"];
  quantization_schema_state["nan_mode"] = state["nan_mode"];
  if (state.contains("nan_modes")) {
    quantization_schema_state["nan_modes"] = state["nan_modes"];
  }
  return QuantizationSchemaFromStateDict(quantization_schema_state);
}

py::tuple NodeToState(const ctboost::Node& node) {
  return py::make_tuple(node.is_leaf,
                        node.is_categorical_split,
                        node.split_feature_id,
                        node.split_bin_index,
                        node.left_child,
                        node.right_child,
                        node.leaf_weight,
                        std::vector<std::uint8_t>(node.left_categories.begin(),
                                                  node.left_categories.end()));
}

ctboost::Node NodeFromState(const py::handle& handle) {
  const py::tuple state = handle.cast<py::tuple>();
  if (state.size() != 8) {
    throw std::runtime_error("invalid serialized tree node state");
  }

  ctboost::Node node;
  node.is_leaf = state[0].cast<bool>();
  node.is_categorical_split = state[1].cast<bool>();
  node.split_feature_id = state[2].cast<int>();
  node.split_bin_index = state[3].cast<std::uint16_t>();
  node.left_child = state[4].cast<int>();
  node.right_child = state[5].cast<int>();
  node.leaf_weight = state[6].cast<float>();

  const auto categories = state[7].cast<std::vector<std::uint8_t>>();
  if (categories.size() != ctboost::kMaxCategoricalRouteBins) {
    throw std::runtime_error("invalid serialized categorical routing table");
  }
  std::copy(categories.begin(), categories.end(), node.left_categories.begin());
  return node;
}

py::dict NodeToStateDict(const ctboost::Node& node) {
  py::dict state;
  state["is_leaf"] = node.is_leaf;
  state["is_categorical_split"] = node.is_categorical_split;
  state["split_feature_id"] = node.split_feature_id;
  state["split_bin_index"] = node.split_bin_index;
  state["left_child"] = node.left_child;
  state["right_child"] = node.right_child;
  state["leaf_weight"] = node.leaf_weight;
  state["left_categories"] =
      std::vector<std::uint8_t>(node.left_categories.begin(), node.left_categories.end());
  return state;
}

ctboost::Node NodeFromStateDict(const py::handle& handle) {
  const py::dict state = handle.cast<py::dict>();
  ctboost::Node node;
  node.is_leaf = py::cast<bool>(state["is_leaf"]);
  node.is_categorical_split = py::cast<bool>(state["is_categorical_split"]);
  node.split_feature_id = py::cast<int>(state["split_feature_id"]);
  node.split_bin_index = py::cast<std::uint16_t>(state["split_bin_index"]);
  node.left_child = py::cast<int>(state["left_child"]);
  node.right_child = py::cast<int>(state["right_child"]);
  node.leaf_weight = py::cast<float>(state["leaf_weight"]);

  const auto categories = py::cast<std::vector<std::uint8_t>>(state["left_categories"]);
  if (categories.size() != ctboost::kMaxCategoricalRouteBins) {
    throw std::runtime_error("invalid serialized categorical routing table");
  }
  std::copy(categories.begin(), categories.end(), node.left_categories.begin());
  return node;
}

py::tuple TreeToState(const ctboost::Tree& tree) {
  py::list node_states;
  for (const ctboost::Node& node : tree.nodes()) {
    node_states.append(NodeToState(node));
  }

  return py::make_tuple(node_states,
                        tree.num_bins_per_feature(),
                        tree.cut_offsets(),
                        tree.cut_values(),
                        tree.categorical_mask(),
                        tree.missing_value_mask(),
                        tree.nan_mode(),
                        tree.feature_importances());
}

ctboost::Tree TreeFromState(const py::handle& handle) {
  const py::tuple state = handle.cast<py::tuple>();
  if (state.size() != 8) {
    throw std::runtime_error("invalid serialized tree state");
  }

  std::vector<ctboost::Node> nodes;
  for (const py::handle node_handle : state[0].cast<py::list>()) {
    nodes.push_back(NodeFromState(node_handle));
  }

  ctboost::Tree tree;
  tree.LoadState(std::move(nodes),
                 state[1].cast<std::vector<std::uint16_t>>(),
                 state[2].cast<std::vector<std::size_t>>(),
                 state[3].cast<std::vector<float>>(),
                 state[4].cast<std::vector<std::uint8_t>>(),
                 state[5].cast<std::vector<std::uint8_t>>(),
                 state[6].cast<std::uint8_t>(),
                 state[7].cast<std::vector<double>>());
  return tree;
}

py::dict TreeToStateDict(const ctboost::Tree& tree) {
  py::list node_states;
  for (const ctboost::Node& node : tree.nodes()) {
    node_states.append(NodeToStateDict(node));
  }

  py::dict state;
  state["nodes"] = node_states;
  state["feature_importances"] = tree.feature_importances();
  return state;
}

ctboost::Tree TreeFromStateDict(const py::handle& handle,
                                const ctboost::QuantizationSchemaPtr& shared_quantization_schema) {
  const py::dict state = handle.cast<py::dict>();
  std::vector<ctboost::Node> nodes;
  for (const py::handle node_handle : py::cast<py::list>(state["nodes"])) {
    nodes.push_back(NodeFromStateDict(node_handle));
  }

  ctboost::QuantizationSchemaPtr quantization_schema = shared_quantization_schema;
  if (quantization_schema == nullptr && state.contains("num_bins_per_feature")) {
    quantization_schema = QuantizationSchemaFromTreeStateDict(state);
  }
  if (quantization_schema == nullptr) {
    throw std::runtime_error("serialized tree state is missing quantization schema");
  }

  ctboost::Tree tree;
  tree.LoadState(
      std::move(nodes), quantization_schema, py::cast<std::vector<double>>(state["feature_importances"]));
  return tree;
}

}  // namespace ctboost::bindings
