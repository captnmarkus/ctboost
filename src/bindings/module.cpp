#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ctboost/build_info.hpp"
#include "ctboost/booster.hpp"
#include "ctboost/cuda_backend.hpp"
#include "ctboost/data.hpp"
#include "ctboost/histogram.hpp"
#include "ctboost/objective.hpp"
#include "ctboost/statistics.hpp"

namespace py = pybind11;

namespace {

py::array_t<float> VectorToArray(const std::vector<float>& values) {
  py::array_t<float> result(values.size());
  if (!values.empty()) {
    std::memcpy(result.mutable_data(), values.data(), values.size() * sizeof(float));
  }
  return result;
}

py::array_t<std::int32_t> IntVectorToArray(const std::vector<std::int32_t>& values) {
  py::array_t<std::int32_t> result(values.size());
  if (!values.empty()) {
    std::memcpy(result.mutable_data(), values.data(), values.size() * sizeof(std::int32_t));
  }
  return result;
}

std::vector<float> ArrayToVector(py::array_t<float, py::array::forcecast> values,
                                 const char* name) {
  const py::buffer_info info = values.request();
  if (info.ndim != 1) {
    throw std::invalid_argument(std::string(name) + " must be a 1D NumPy array");
  }
  if (info.strides[0] % static_cast<py::ssize_t>(sizeof(float)) != 0) {
    throw std::invalid_argument(std::string(name) + " must have float-compatible strides");
  }

  const auto* ptr = static_cast<const float*>(info.ptr);
  const py::ssize_t stride = info.strides[0] / static_cast<py::ssize_t>(sizeof(float));

  std::vector<float> out(static_cast<std::size_t>(info.shape[0]));
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = *(ptr + static_cast<py::ssize_t>(i) * stride);
  }
  return out;
}

std::vector<std::uint16_t> ArrayToBinVector(py::array_t<std::int64_t, py::array::forcecast> values,
                                            const char* name) {
  const py::buffer_info info = values.request();
  if (info.ndim != 1) {
    throw std::invalid_argument(std::string(name) + " must be a 1D NumPy array");
  }
  if (info.strides[0] % static_cast<py::ssize_t>(sizeof(std::int64_t)) != 0) {
    throw std::invalid_argument(std::string(name) + " must have integer-compatible strides");
  }

  const auto* ptr = static_cast<const std::int64_t*>(info.ptr);
  const py::ssize_t stride = info.strides[0] / static_cast<py::ssize_t>(sizeof(std::int64_t));

  std::vector<std::uint16_t> out(static_cast<std::size_t>(info.shape[0]));
  for (std::size_t i = 0; i < out.size(); ++i) {
    const std::int64_t value = *(ptr + static_cast<py::ssize_t>(i) * stride);
    if (value < 0 || value > static_cast<std::int64_t>(std::numeric_limits<std::uint16_t>::max())) {
      throw std::invalid_argument(std::string(name) + " must contain values in [0, 65535]");
    }
    out[i] = static_cast<std::uint16_t>(value);
  }
  return out;
}

py::dict QuantizationSchemaToStateDict(const ctboost::QuantizationSchema& quantization_schema) {
  py::dict state;
  state["num_bins_per_feature"] = quantization_schema.num_bins_per_feature;
  state["cut_offsets"] = quantization_schema.cut_offsets;
  state["cut_values"] = quantization_schema.cut_values;
  state["categorical_mask"] = quantization_schema.categorical_mask;
  state["missing_value_mask"] = quantization_schema.missing_value_mask;
  state["nan_mode"] = quantization_schema.nan_mode;
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
                                const ctboost::QuantizationSchemaPtr& shared_quantization_schema =
                                    nullptr) {
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

py::dict BoosterToStateDict(const ctboost::GradientBooster& booster) {
  py::list tree_states;
  for (const ctboost::Tree& tree : booster.trees()) {
    tree_states.append(TreeToStateDict(tree));
  }

  py::dict state;
  state["objective_name"] = booster.objective_name();
  state["iterations"] = booster.iterations();
  state["learning_rate"] = booster.learning_rate();
  state["max_depth"] = booster.max_depth();
  state["alpha"] = booster.alpha();
  state["lambda_l2"] = booster.lambda_l2();
  state["colsample_bytree"] = booster.colsample_bytree();
  state["max_leaves"] = booster.max_leaves();
  state["min_data_in_leaf"] = booster.min_data_in_leaf();
  state["min_child_weight"] = booster.min_child_weight();
  state["gamma"] = booster.gamma();
  state["num_classes"] = booster.num_classes();
  state["max_bins"] = booster.max_bins();
  state["nan_mode"] = booster.nan_mode_name();
  state["eval_metric_name"] = booster.eval_metric_name();
  state["quantile_alpha"] = booster.quantile_alpha();
  state["huber_delta"] = booster.huber_delta();
  state["devices"] = booster.devices();
  state["random_seed"] = booster.random_seed();
  state["rng_state"] = booster.rng_state();
  state["task_type"] = booster.use_gpu() ? "GPU" : "CPU";
  state["verbose"] = booster.verbose();
  if (const auto* quantization_schema = booster.quantization_schema(); quantization_schema != nullptr) {
    state["quantization_schema"] = QuantizationSchemaToStateDict(*quantization_schema);
  }
  state["trees"] = tree_states;
  state["loss_history"] = booster.loss_history();
  state["eval_loss_history"] = booster.eval_loss_history();
  state["best_iteration"] = booster.best_iteration();
  state["best_score"] = booster.best_score();
  state["feature_importances"] = booster.get_feature_importances();
  return state;
}

ctboost::GradientBooster BoosterFromStateDict(const py::dict& state) {
  const py::list tree_state_list = py::cast<py::list>(state["trees"]);
  ctboost::QuantizationSchemaPtr quantization_schema;
  if (state.contains("quantization_schema")) {
    quantization_schema = QuantizationSchemaFromStateDict(state["quantization_schema"]);
  } else if (!tree_state_list.empty()) {
    const py::dict first_tree_state = tree_state_list[0].cast<py::dict>();
    if (first_tree_state.contains("num_bins_per_feature")) {
      quantization_schema = QuantizationSchemaFromTreeStateDict(first_tree_state);
    }
  }

  std::vector<ctboost::Tree> trees;
  for (const py::handle tree_handle : tree_state_list) {
    trees.push_back(TreeFromStateDict(tree_handle, quantization_schema));
  }

  const std::string task_type = py::cast<std::string>(state["task_type"]);
  const bool requested_gpu = task_type == "GPU" || task_type == "gpu";
  const bool use_gpu = requested_gpu && ctboost::CudaBackendCompiled();

  ctboost::GradientBooster booster(py::cast<std::string>(state["objective_name"]),
                                   py::cast<int>(state["iterations"]),
                                   py::cast<double>(state["learning_rate"]),
                                   py::cast<int>(state["max_depth"]),
                                   py::cast<double>(state["alpha"]),
                                   py::cast<double>(state["lambda_l2"]),
                                   state.contains("colsample_bytree")
                                       ? py::cast<double>(state["colsample_bytree"])
                                       : 1.0,
                                   state.contains("max_leaves")
                                       ? py::cast<int>(state["max_leaves"])
                                       : 0,
                                   state.contains("min_data_in_leaf")
                                       ? py::cast<int>(state["min_data_in_leaf"])
                                       : 0,
                                   state.contains("min_child_weight")
                                       ? py::cast<double>(state["min_child_weight"])
                                       : 0.0,
                                   state.contains("gamma")
                                       ? py::cast<double>(state["gamma"])
                                       : 0.0,
                                   py::cast<int>(state["num_classes"]),
                                   py::cast<std::size_t>(state["max_bins"]),
                                   py::cast<std::string>(state["nan_mode"]),
                                   py::cast<std::string>(state["eval_metric_name"]),
                                   py::cast<double>(state["quantile_alpha"]),
                                   py::cast<double>(state["huber_delta"]),
                                   use_gpu ? "GPU" : "CPU",
                                   py::cast<std::string>(state["devices"]),
                                   state.contains("random_seed")
                                       ? py::cast<std::uint64_t>(state["random_seed"])
                                       : 0U,
                                   state.contains("verbose") ? py::cast<bool>(state["verbose"]) : false);

  booster.LoadState(std::move(trees),
                    quantization_schema,
                    py::cast<std::vector<double>>(state["loss_history"]),
                    py::cast<std::vector<double>>(state["eval_loss_history"]),
                    std::vector<double>{},
                    py::cast<int>(state["best_iteration"]),
                    py::cast<double>(state["best_score"]),
                    use_gpu,
                    state.contains("rng_state") ? py::cast<std::uint64_t>(state["rng_state"]) : 0U);
  return booster;
}

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "Native backend scaffolding for CTBoost";

  py::class_<ctboost::Pool>(m, "Pool")
      .def(py::init([](py::array_t<float, py::array::forcecast> data,
                       py::array_t<float, py::array::forcecast> label,
                       std::vector<int> cat_features,
                       py::object weight,
                       py::object group_id) {
             py::array_t<float, py::array::forcecast> resolved_weight;
             if (weight.is_none()) {
               const py::buffer_info label_info = label.request();
               py::array_t<float> ones(static_cast<py::ssize_t>(label_info.shape[0]));
               auto mutable_weights = ones.mutable_unchecked<1>();
               for (py::ssize_t index = 0; index < label_info.shape[0]; ++index) {
                 mutable_weights(index) = 1.0F;
               }
               resolved_weight = std::move(ones);
             } else {
               resolved_weight = weight.cast<py::array_t<float, py::array::forcecast>>();
             }
             py::array_t<std::int64_t, py::array::forcecast> resolved_group_id;
             if (!group_id.is_none()) {
               resolved_group_id = group_id.cast<py::array_t<std::int64_t, py::array::forcecast>>();
             }
             return ctboost::Pool(
                 data,
                 label,
                 std::move(cat_features),
                 resolved_weight,
                 resolved_group_id);
           }),
           py::arg("data"),
           py::arg("label"),
           py::arg("cat_features") = std::vector<int>{},
           py::arg("weight") = py::none(),
           py::arg("group_id") = py::none())
      .def("num_rows", &ctboost::Pool::num_rows)
      .def("num_cols", &ctboost::Pool::num_cols)
      .def("feature_data", [](const ctboost::Pool& pool) {
        return VectorToArray(pool.feature_data());
      })
      .def("label", [](const ctboost::Pool& pool) {
        return VectorToArray(pool.labels());
      })
      .def("weight", [](const ctboost::Pool& pool) {
        return VectorToArray(pool.weights());
      })
      .def("group_id", [](const ctboost::Pool& pool) -> py::object {
        if (!pool.has_group_ids()) {
          return py::none();
        }
        py::array_t<std::int64_t> result(pool.group_ids().size());
        if (!pool.group_ids().empty()) {
          std::memcpy(
              result.mutable_data(), pool.group_ids().data(), pool.group_ids().size() * sizeof(std::int64_t));
        }
        return result;
      })
      .def("cat_features", [](const ctboost::Pool& pool) {
        return pool.cat_features();
      })
      .def("set_feature_storage_releasable", &ctboost::Pool::SetFeatureStorageReleasable);

  py::class_<ctboost::GradientBooster>(m, "GradientBooster")
      .def(py::init<std::string,
                    int,
                    double,
                    int,
                    double,
                    double,
                    double,
                    int,
                    int,
                    double,
                    double,
                    int,
                    std::size_t,
                    std::string,
                    std::string,
                    double,
                    double,
                    std::string,
                    std::string,
                    std::uint64_t,
                    bool>(),
           py::arg("objective") = "RMSE",
           py::arg("iterations") = 100,
           py::arg("learning_rate") = 0.1,
           py::arg("max_depth") = 6,
           py::arg("alpha") = 0.05,
           py::arg("lambda_l2") = 1.0,
           py::arg("colsample_bytree") = 1.0,
           py::arg("max_leaves") = 0,
           py::arg("min_data_in_leaf") = 0,
           py::arg("min_child_weight") = 0.0,
           py::arg("gamma") = 0.0,
           py::arg("num_classes") = 1,
           py::arg("max_bins") = 256,
           py::arg("nan_mode") = "Min",
           py::arg("eval_metric") = "",
           py::arg("quantile_alpha") = 0.5,
           py::arg("huber_delta") = 1.0,
           py::arg("task_type") = "CPU",
           py::arg("devices") = "0",
           py::arg("random_seed") = 0,
           py::arg("verbose") = false)
      .def("fit",
           [](ctboost::GradientBooster& booster,
              py::object pool_obj,
              py::object eval_pool,
              int early_stopping_rounds,
              bool continue_training)
               -> ctboost::GradientBooster& {
             auto& pool = pool_obj.cast<ctboost::Pool&>();
             if (eval_pool.is_none()) {
               booster.Fit(pool, nullptr, early_stopping_rounds, continue_training);
             } else {
               auto& eval_pool_ref = eval_pool.cast<ctboost::Pool&>();
               booster.Fit(pool, &eval_pool_ref, early_stopping_rounds, continue_training);
             }
             return booster;
           },
           py::arg("pool"),
           py::arg("eval_pool") = py::none(),
           py::arg("early_stopping_rounds") = 0,
           py::arg("continue_training") = false,
           py::return_value_policy::reference_internal)
      .def("predict",
           [](const ctboost::GradientBooster& booster,
              const ctboost::Pool& pool,
              int num_iteration) {
             return VectorToArray(booster.Predict(pool, num_iteration));
           },
           py::arg("pool"),
           py::arg("num_iteration") = -1)
      .def("predict_leaf_indices",
           [](const ctboost::GradientBooster& booster,
              const ctboost::Pool& pool,
              int num_iteration) {
             return IntVectorToArray(booster.PredictLeafIndices(pool, num_iteration));
           },
           py::arg("pool"),
           py::arg("num_iteration") = -1)
      .def("predict_contributions",
           [](const ctboost::GradientBooster& booster,
              const ctboost::Pool& pool,
              int num_iteration) {
             return VectorToArray(booster.PredictContributions(pool, num_iteration));
           },
           py::arg("pool"),
           py::arg("num_iteration") = -1)
      .def("loss_history", [](const ctboost::GradientBooster& booster) {
        return booster.loss_history();
      })
      .def("eval_loss_history", [](const ctboost::GradientBooster& booster) {
        return booster.eval_loss_history();
      })
      .def("num_trees", &ctboost::GradientBooster::num_trees)
      .def("num_iterations_trained", &ctboost::GradientBooster::num_iterations_trained)
      .def("best_iteration", &ctboost::GradientBooster::best_iteration)
      .def("num_classes", &ctboost::GradientBooster::num_classes)
      .def("prediction_dimension", &ctboost::GradientBooster::prediction_dimension)
      .def("objective_name", &ctboost::GradientBooster::objective_name)
      .def("eval_metric_name", &ctboost::GradientBooster::eval_metric_name)
      .def("iterations", &ctboost::GradientBooster::iterations)
      .def("learning_rate", &ctboost::GradientBooster::learning_rate)
      .def("max_depth", &ctboost::GradientBooster::max_depth)
      .def("alpha", &ctboost::GradientBooster::alpha)
      .def("lambda_l2", &ctboost::GradientBooster::lambda_l2)
      .def("colsample_bytree", &ctboost::GradientBooster::colsample_bytree)
      .def("max_leaves", &ctboost::GradientBooster::max_leaves)
      .def("min_data_in_leaf", &ctboost::GradientBooster::min_data_in_leaf)
      .def("min_child_weight", &ctboost::GradientBooster::min_child_weight)
      .def("gamma", &ctboost::GradientBooster::gamma)
      .def("max_bins", &ctboost::GradientBooster::max_bins)
      .def("nan_mode_name", &ctboost::GradientBooster::nan_mode_name)
      .def("quantile_alpha", &ctboost::GradientBooster::quantile_alpha)
      .def("huber_delta", &ctboost::GradientBooster::huber_delta)
      .def("use_gpu", &ctboost::GradientBooster::use_gpu)
      .def("devices", &ctboost::GradientBooster::devices)
      .def("random_seed", &ctboost::GradientBooster::random_seed)
      .def("rng_state", &ctboost::GradientBooster::rng_state)
      .def("verbose", &ctboost::GradientBooster::verbose)
      .def("export_state", [](const ctboost::GradientBooster& booster) {
        return BoosterToStateDict(booster);
      })
      .def("load_state",
           [](ctboost::GradientBooster& booster, const py::dict& state)
               -> ctboost::GradientBooster& {
             const bool requested_gpu =
                 py::cast<std::string>(state["task_type"]) == "GPU" ||
                 py::cast<std::string>(state["task_type"]) == "gpu";
             const bool use_gpu = requested_gpu && ctboost::CudaBackendCompiled();

             std::vector<ctboost::Tree> trees;
             const py::list tree_state_list = py::cast<py::list>(state["trees"]);
             ctboost::QuantizationSchemaPtr quantization_schema;
             if (state.contains("quantization_schema")) {
               quantization_schema = QuantizationSchemaFromStateDict(state["quantization_schema"]);
             } else if (!tree_state_list.empty()) {
               const py::dict first_tree_state = tree_state_list[0].cast<py::dict>();
               if (first_tree_state.contains("num_bins_per_feature")) {
                 quantization_schema = QuantizationSchemaFromTreeStateDict(first_tree_state);
               }
             }
             for (const py::handle tree_handle : tree_state_list) {
               trees.push_back(TreeFromStateDict(tree_handle, quantization_schema));
             }
             booster.LoadState(std::move(trees),
                               quantization_schema,
                               py::cast<std::vector<double>>(state["loss_history"]),
                               py::cast<std::vector<double>>(state["eval_loss_history"]),
                               std::vector<double>{},
                               py::cast<int>(state["best_iteration"]),
                               py::cast<double>(state["best_score"]),
                               use_gpu,
                               state.contains("rng_state")
                                   ? py::cast<std::uint64_t>(state["rng_state"])
                                   : 0U);
             return booster;
           },
           py::arg("state"),
           py::return_value_policy::reference_internal)
      .def("feature_importances", [](const ctboost::GradientBooster& booster) {
        return VectorToArray(booster.get_feature_importances());
      })
      .def_static("from_state", [](const py::dict& state) {
        return BoosterFromStateDict(state);
      })
      .def(py::pickle(
          [](const ctboost::GradientBooster& booster) -> py::object {
            return BoosterToStateDict(booster);
          },
          [](const py::object& state_object) {
            if (py::isinstance<py::dict>(state_object)) {
              return BoosterFromStateDict(state_object.cast<py::dict>());
            }

            const py::tuple state = state_object.cast<py::tuple>();
            if (state.size() != 19) {
              throw std::runtime_error("invalid serialized GradientBooster state");
            }

            py::dict upgraded_state;
            upgraded_state["objective_name"] = state[0];
            upgraded_state["iterations"] = state[1];
            upgraded_state["learning_rate"] = state[2];
            upgraded_state["max_depth"] = state[3];
            upgraded_state["alpha"] = state[4];
            upgraded_state["lambda_l2"] = state[5];
            upgraded_state["num_classes"] = state[6];
            upgraded_state["max_bins"] = state[7];
            upgraded_state["nan_mode"] = state[8];
            upgraded_state["eval_metric_name"] = state[9];
            upgraded_state["quantile_alpha"] = state[10];
            upgraded_state["huber_delta"] = state[11];
            upgraded_state["devices"] = state[12];
            upgraded_state["task_type"] =
                state[13].cast<bool>() ? py::str("GPU") : py::str("CPU");
            upgraded_state["verbose"] = py::bool_(false);

            py::list tree_states;
            py::dict quantization_schema_state;
            bool quantization_schema_initialized = false;
            for (const py::handle tree_handle : state[14].cast<py::list>()) {
              const ctboost::Tree tree = TreeFromState(tree_handle);
              if (!quantization_schema_initialized) {
                quantization_schema_state =
                    QuantizationSchemaToStateDict(*tree.shared_quantization_schema());
                quantization_schema_initialized = true;
              }
              tree_states.append(TreeToStateDict(tree));
            }
            if (quantization_schema_initialized) {
              upgraded_state["quantization_schema"] = quantization_schema_state;
            }
            upgraded_state["trees"] = tree_states;
            upgraded_state["loss_history"] = state[15];
            upgraded_state["eval_loss_history"] = state[16];
            upgraded_state["best_iteration"] = state[17];

            const auto eval_loss_history = state[16].cast<std::vector<double>>();
            const int best_iteration = state[17].cast<int>();
            double best_score = 0.0;
            if (best_iteration >= 0 &&
                static_cast<std::size_t>(best_iteration) < eval_loss_history.size()) {
              best_score = eval_loss_history[static_cast<std::size_t>(best_iteration)];
            }
            upgraded_state["best_score"] = best_score;
            upgraded_state["feature_importances"] = state[18];
            return BoosterFromStateDict(upgraded_state);
          }));

  m.def("build_info", []() {
    const ctboost::BuildInfo info = ctboost::GetBuildInfo();
    py::dict result;
    result["version"] = info.version;
    result["cuda_enabled"] = info.cuda_enabled;
    result["cuda_runtime"] = info.cuda_runtime;
    result["compiler"] = info.compiler;
    result["cxx_standard"] = info.cxx_standard;
    return result;
  });

  m.def("_booster_from_state", [](const py::dict& state) {
    return BoosterFromStateDict(state);
  });

  m.def("_debug_compute_objective",
        [](const std::string& objective_name,
           py::array_t<float, py::array::forcecast> preds,
           py::array_t<float, py::array::forcecast> labels,
           int num_classes) {
          const auto pred_values = ArrayToVector(preds, "preds");
          const auto label_values = ArrayToVector(labels, "labels");
          std::unique_ptr<ctboost::ObjectiveFunction> objective =
              ctboost::CreateObjectiveFunction(objective_name, ctboost::ObjectiveConfig{});

          std::vector<float> gradients;
          std::vector<float> hessians;
          objective->compute_gradients(
              pred_values, label_values, gradients, hessians, num_classes, nullptr);
          return py::make_tuple(VectorToArray(gradients), VectorToArray(hessians));
        },
        py::arg("objective_name"),
        py::arg("preds"),
        py::arg("labels"),
        py::arg("num_classes") = 1);

  m.def("_debug_compute_pvalue",
        [](py::array_t<float, py::array::forcecast> gradients,
           py::array_t<std::int64_t, py::array::forcecast> binned_feature) {
          const auto gradient_values = ArrayToVector(gradients, "gradients");
          const auto bin_values = ArrayToBinVector(binned_feature, "binned_feature");
          const std::uint16_t max_bin =
              bin_values.empty() ? 0 : *std::max_element(bin_values.begin(), bin_values.end());
          std::vector<float> hessians(gradient_values.size(), 1.0F);
          std::vector<float> weights(gradient_values.size(), 1.0F);
          const ctboost::LinearStatistic statistic;
          const ctboost::LinearStatisticResult result =
              statistic.Evaluate(gradient_values,
                                 hessians,
                                 weights,
                                 bin_values,
                                 static_cast<std::size_t>(max_bin) + 1);

          py::dict out;
          out["p_value"] = result.p_value;
          out["chi_square"] = result.chi_square;
          out["degrees_of_freedom"] = result.degrees_of_freedom;
          out["statistic"] = result.statistic;
          out["expectation"] = result.expectation;
          out["covariance"] = result.covariance;
          return out;
        },
        py::arg("gradients"),
        py::arg("binned_feature"));

  m.def("_debug_build_histogram",
        [](const ctboost::Pool& pool, std::size_t max_bins, const std::string& nan_mode) {
          const auto started = std::chrono::steady_clock::now();
          const ctboost::HistBuilder builder(max_bins, nan_mode);
          const ctboost::HistMatrix hist = builder.Build(pool, nullptr);
          const double elapsed_ms =
              std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
                  .count();

          py::dict out;
          out["elapsed_ms"] = elapsed_ms;
          out["num_rows"] = hist.num_rows;
          out["num_cols"] = hist.num_cols;
          out["num_bins_per_feature"] = hist.num_bins_per_feature;
          out["cut_offsets"] = hist.cut_offsets;
          out["cut_values"] = hist.cut_values;
          out["cut_values_count"] = hist.cut_values.size();
          return out;
        },
        py::arg("pool"),
        py::arg("max_bins") = 256,
        py::arg("nan_mode") = "Min");

}
