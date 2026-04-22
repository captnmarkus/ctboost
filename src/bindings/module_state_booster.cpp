#include "module_internal.hpp"

namespace ctboost::bindings {

py::dict BoosterToStateDict(const ctboost::GradientBooster& booster) {
  py::list tree_states;
  for (const ctboost::Tree& tree : booster.trees()) {
    tree_states.append(TreeToStateDict(tree));
  }

  py::dict state;
  state["objective_name"] = booster.objective_name();
  state["iterations"] = booster.iterations();
  state["learning_rate"] = booster.learning_rate();
  state["tree_learning_rates"] = booster.tree_learning_rates();
  state["max_depth"] = booster.max_depth();
  state["alpha"] = booster.alpha();
  state["lambda_l2"] = booster.lambda_l2();
  state["subsample"] = booster.subsample();
  state["bootstrap_type"] = booster.bootstrap_type();
  state["bagging_temperature"] = booster.bagging_temperature();
  state["boosting_type"] = booster.boosting_type();
  state["drop_rate"] = booster.drop_rate();
  state["skip_drop"] = booster.skip_drop();
  state["max_drop"] = booster.max_drop();
  state["monotone_constraints"] = booster.monotone_constraints();
  state["interaction_constraints"] = booster.interaction_constraints();
  state["colsample_bytree"] = booster.colsample_bytree();
  state["feature_weights"] = booster.feature_weights();
  state["first_feature_use_penalties"] = booster.first_feature_use_penalties();
  state["random_strength"] = booster.random_strength();
  state["grow_policy"] = booster.grow_policy();
  state["max_leaves"] = booster.max_leaves();
  state["min_samples_split"] = booster.min_samples_split();
  state["min_data_in_leaf"] = booster.min_data_in_leaf();
  state["min_child_weight"] = booster.min_child_weight();
  state["gamma"] = booster.gamma();
  state["max_leaf_weight"] = booster.max_leaf_weight();
  state["num_classes"] = booster.num_classes();
  state["max_bins"] = booster.max_bins();
  state["nan_mode"] = booster.nan_mode_name();
  state["max_bin_by_feature"] = booster.max_bin_by_feature();
  state["border_selection_method"] = booster.border_selection_method();
  state["nan_mode_by_feature"] = booster.nan_mode_by_feature();
  state["feature_borders"] = booster.feature_borders();
  state["external_memory"] = booster.external_memory();
  state["external_memory_dir"] = booster.external_memory_dir();
  state["eval_metric_name"] = booster.eval_metric_name();
  state["quantile_alpha"] = booster.quantile_alpha();
  state["huber_delta"] = booster.huber_delta();
  state["tweedie_variance_power"] = booster.tweedie_variance_power();
  state["devices"] = booster.devices();
  state["distributed_world_size"] = booster.distributed_world_size();
  state["distributed_rank"] = booster.distributed_rank();
  state["distributed_root"] = booster.distributed_root();
  state["distributed_run_id"] = booster.distributed_run_id();
  state["distributed_timeout"] = booster.distributed_timeout();
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
                                   state.contains("subsample")
                                       ? py::cast<double>(state["subsample"])
                                       : 1.0,
                                   state.contains("bootstrap_type")
                                       ? py::cast<std::string>(state["bootstrap_type"])
                                       : std::string("No"),
                                   state.contains("bagging_temperature")
                                       ? py::cast<double>(state["bagging_temperature"])
                                       : 0.0,
                                   state.contains("boosting_type")
                                       ? py::cast<std::string>(state["boosting_type"])
                                       : std::string("GradientBoosting"),
                                   state.contains("drop_rate")
                                       ? py::cast<double>(state["drop_rate"])
                                       : 0.1,
                                   state.contains("skip_drop")
                                       ? py::cast<double>(state["skip_drop"])
                                       : 0.5,
                                   state.contains("max_drop")
                                       ? py::cast<int>(state["max_drop"])
                                       : 0,
                                   state.contains("monotone_constraints")
                                       ? py::cast<std::vector<int>>(state["monotone_constraints"])
                                       : std::vector<int>{},
                                   state.contains("interaction_constraints")
                                       ? py::cast<std::vector<std::vector<int>>>(
                                             state["interaction_constraints"])
                                       : std::vector<std::vector<int>>{},
                                   state.contains("colsample_bytree")
                                       ? py::cast<double>(state["colsample_bytree"])
                                       : 1.0,
                                   state.contains("feature_weights")
                                       ? py::cast<std::vector<double>>(state["feature_weights"])
                                       : std::vector<double>{},
                                   state.contains("first_feature_use_penalties")
                                       ? py::cast<std::vector<double>>(
                                             state["first_feature_use_penalties"])
                                       : std::vector<double>{},
                                   state.contains("random_strength")
                                       ? py::cast<double>(state["random_strength"])
                                       : 0.0,
                                   state.contains("grow_policy")
                                       ? py::cast<std::string>(state["grow_policy"])
                                       : std::string("DepthWise"),
                                   state.contains("max_leaves")
                                       ? py::cast<int>(state["max_leaves"])
                                       : 0,
                                   state.contains("min_samples_split")
                                       ? py::cast<int>(state["min_samples_split"])
                                       : 2,
                                   state.contains("min_data_in_leaf")
                                       ? py::cast<int>(state["min_data_in_leaf"])
                                       : 0,
                                   state.contains("min_child_weight")
                                       ? py::cast<double>(state["min_child_weight"])
                                       : 0.0,
                                   state.contains("gamma")
                                       ? py::cast<double>(state["gamma"])
                                       : 0.0,
                                   state.contains("max_leaf_weight")
                                       ? py::cast<double>(state["max_leaf_weight"])
                                       : 0.0,
                                   py::cast<int>(state["num_classes"]),
                                   py::cast<std::size_t>(state["max_bins"]),
                                   py::cast<std::string>(state["nan_mode"]),
                                   state.contains("max_bin_by_feature")
                                       ? py::cast<std::vector<std::uint16_t>>(
                                             state["max_bin_by_feature"])
                                       : std::vector<std::uint16_t>{},
                                   state.contains("border_selection_method")
                                       ? py::cast<std::string>(state["border_selection_method"])
                                       : std::string("Quantile"),
                                   state.contains("nan_mode_by_feature")
                                       ? py::cast<std::vector<std::string>>(
                                             state["nan_mode_by_feature"])
                                       : std::vector<std::string>{},
                                   state.contains("feature_borders")
                                       ? py::cast<std::vector<std::vector<float>>>(
                                             state["feature_borders"])
                                       : std::vector<std::vector<float>>{},
                                   state.contains("external_memory")
                                       ? py::cast<bool>(state["external_memory"])
                                       : false,
                                   state.contains("external_memory_dir")
                                       ? py::cast<std::string>(state["external_memory_dir"])
                                       : std::string(),
                                   py::cast<std::string>(state["eval_metric_name"]),
                                   py::cast<double>(state["quantile_alpha"]),
                                   py::cast<double>(state["huber_delta"]),
                                   state.contains("tweedie_variance_power")
                                       ? py::cast<double>(state["tweedie_variance_power"])
                                       : 1.5,
                                   use_gpu ? "GPU" : "CPU",
                                   py::cast<std::string>(state["devices"]),
                                   state.contains("distributed_world_size")
                                       ? py::cast<int>(state["distributed_world_size"])
                                       : 1,
                                   state.contains("distributed_rank")
                                       ? py::cast<int>(state["distributed_rank"])
                                       : 0,
                                   state.contains("distributed_root")
                                       ? py::cast<std::string>(state["distributed_root"])
                                       : std::string(),
                                   state.contains("distributed_run_id")
                                       ? py::cast<std::string>(state["distributed_run_id"])
                                       : std::string("default"),
                                   state.contains("distributed_timeout")
                                       ? py::cast<double>(state["distributed_timeout"])
                                       : 600.0,
                                   state.contains("random_seed")
                                       ? py::cast<std::uint64_t>(state["random_seed"])
                                       : 0U,
                                   state.contains("verbose") ? py::cast<bool>(state["verbose"]) : false);

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
                    state.contains("rng_state") ? py::cast<std::uint64_t>(state["rng_state"]) : 0U);
  return booster;
}

}  // namespace ctboost::bindings
