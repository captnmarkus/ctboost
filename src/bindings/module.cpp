#include <algorithm>
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
                        tree.feature_importances());
}

ctboost::Tree TreeFromState(const py::handle& handle) {
  const py::tuple state = handle.cast<py::tuple>();
  if (state.size() != 6) {
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
                 state[5].cast<std::vector<double>>());
  return tree;
}

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "Native backend scaffolding for CTBoost";

  py::class_<ctboost::Pool>(m, "Pool")
      .def(py::init<py::array_t<float, py::array::forcecast>,
                    py::array_t<float, py::array::forcecast>,
                    std::vector<int>>(),
           py::arg("data"),
           py::arg("label"),
           py::arg("cat_features") = std::vector<int>{})
      .def("num_rows", &ctboost::Pool::num_rows)
      .def("num_cols", &ctboost::Pool::num_cols)
      .def("feature_data", [](const ctboost::Pool& pool) {
        return VectorToArray(pool.feature_data());
      })
      .def("label", [](const ctboost::Pool& pool) {
        return VectorToArray(pool.labels());
      })
      .def("cat_features", [](const ctboost::Pool& pool) {
        return pool.cat_features();
      });

  py::class_<ctboost::GradientBooster>(m, "GradientBooster")
      .def(py::init<std::string,
                    int,
                    double,
                    int,
                    double,
                    double,
                    int,
                    std::size_t,
                    std::string,
                    std::string>(),
           py::arg("objective") = "RMSE",
           py::arg("iterations") = 100,
           py::arg("learning_rate") = 0.1,
           py::arg("max_depth") = 6,
           py::arg("alpha") = 0.05,
           py::arg("lambda_l2") = 1.0,
           py::arg("num_classes") = 1,
           py::arg("max_bins") = 256,
           py::arg("task_type") = "CPU",
           py::arg("devices") = "0")
      .def("fit",
           [](ctboost::GradientBooster& booster,
              const ctboost::Pool& pool,
              py::object eval_pool,
              int early_stopping_rounds)
               -> ctboost::GradientBooster& {
             if (eval_pool.is_none()) {
               booster.Fit(pool, nullptr, early_stopping_rounds);
             } else {
               const auto& eval_pool_ref = eval_pool.cast<const ctboost::Pool&>();
               booster.Fit(pool, &eval_pool_ref, early_stopping_rounds);
             }
             return booster;
           },
           py::arg("pool"),
           py::arg("eval_pool") = py::none(),
           py::arg("early_stopping_rounds") = 0,
           py::return_value_policy::reference_internal)
      .def("predict",
           [](const ctboost::GradientBooster& booster,
              const ctboost::Pool& pool,
              int num_iteration) {
             return VectorToArray(booster.Predict(pool, num_iteration));
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
      .def("feature_importances", [](const ctboost::GradientBooster& booster) {
        return VectorToArray(booster.get_feature_importances());
      })
      .def(py::pickle(
          [](const ctboost::GradientBooster& booster) {
            py::list tree_states;
            for (const ctboost::Tree& tree : booster.trees()) {
              tree_states.append(TreeToState(tree));
            }

            return py::make_tuple(booster.objective_name(),
                                  booster.iterations(),
                                  booster.learning_rate(),
                                  booster.max_depth(),
                                  booster.alpha(),
                                  booster.lambda_l2(),
                                  booster.num_classes(),
                                  booster.max_bins(),
                                  booster.devices(),
                                  booster.use_gpu(),
                                  tree_states,
                                  booster.loss_history(),
                                  booster.eval_loss_history(),
                                  booster.best_iteration(),
                                  booster.get_feature_importances());
          },
          [](const py::tuple& state) {
            if (state.size() != 15) {
              throw std::runtime_error("invalid serialized GradientBooster state");
            }

            std::vector<ctboost::Tree> trees;
            for (const py::handle tree_handle : state[10].cast<py::list>()) {
              trees.push_back(TreeFromState(tree_handle));
            }

            std::vector<double> feature_importance_sums;
            for (const float value : state[14].cast<std::vector<float>>()) {
              feature_importance_sums.push_back(static_cast<double>(value));
            }

            const bool requested_gpu = state[9].cast<bool>();
            const bool use_gpu = requested_gpu && ctboost::CudaBackendCompiled();
            ctboost::GradientBooster booster(state[0].cast<std::string>(),
                                             state[1].cast<int>(),
                                             state[2].cast<double>(),
                                             state[3].cast<int>(),
                                             state[4].cast<double>(),
                                             state[5].cast<double>(),
                                             state[6].cast<int>(),
                                             state[7].cast<std::size_t>(),
                                             use_gpu ? "GPU" : "CPU",
                                             state[8].cast<std::string>());

            const auto eval_loss_history = state[12].cast<std::vector<double>>();
            const int best_iteration = state[13].cast<int>();
            double best_loss = 0.0;
            if (best_iteration >= 0 &&
                static_cast<std::size_t>(best_iteration) < eval_loss_history.size()) {
              best_loss = eval_loss_history[static_cast<std::size_t>(best_iteration)];
            }

            booster.LoadState(std::move(trees),
                              state[11].cast<std::vector<double>>(),
                              eval_loss_history,
                              std::move(feature_importance_sums),
                              best_iteration,
                              best_loss,
                              use_gpu);
            return booster;
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

  m.def("_debug_compute_objective",
        [](const std::string& objective_name,
           py::array_t<float, py::array::forcecast> preds,
           py::array_t<float, py::array::forcecast> labels,
           int num_classes) {
          const auto pred_values = ArrayToVector(preds, "preds");
          const auto label_values = ArrayToVector(labels, "labels");
          std::unique_ptr<ctboost::ObjectiveFunction> objective =
              ctboost::CreateObjectiveFunction(objective_name);

          std::vector<float> gradients;
          std::vector<float> hessians;
          objective->compute_gradients(
              pred_values, label_values, gradients, hessians, num_classes);
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
          const ctboost::LinearStatistic statistic;
          const ctboost::LinearStatisticResult result =
              statistic.Evaluate(gradient_values,
                                 hessians,
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
}
