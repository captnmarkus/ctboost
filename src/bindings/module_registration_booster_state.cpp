#include "module_internal.hpp"

#include <stdexcept>

namespace ctboost::bindings {

void BindGradientBoosterStateMethods(py::class_<ctboost::GradientBooster>& booster_class) {
  booster_class
      .def("set_iterations", &ctboost::GradientBooster::SetIterations, py::arg("iterations"))
      .def("set_learning_rate",
           &ctboost::GradientBooster::SetLearningRate,
           py::arg("learning_rate"))
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
                               state.contains("tree_learning_rates")
                                   ? py::cast<std::vector<double>>(state["tree_learning_rates"])
                                   : std::vector<double>{},
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
      .def("load_quantization_schema",
           [](ctboost::GradientBooster& booster, const py::dict& state)
               -> ctboost::GradientBooster& {
             booster.LoadQuantizationSchema(QuantizationSchemaFromStateDict(state));
             return booster;
           },
           py::arg("state"),
           py::return_value_policy::reference_internal)
      .def("feature_importances", [](const ctboost::GradientBooster& booster) {
        return VectorToArray(booster.get_feature_importances());
      })
      .def("quantization_schema_state", [](const ctboost::GradientBooster& booster) -> py::object {
        if (const auto* quantization_schema = booster.quantization_schema();
            quantization_schema != nullptr) {
          return QuantizationSchemaToStateDict(*quantization_schema);
        }
        return py::none();
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
}

}  // namespace ctboost::bindings
