#include "module_internal.hpp"

namespace ctboost::bindings {

void BindGradientBoosterAccessors(py::class_<ctboost::GradientBooster>& booster_class) {
  booster_class
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
      .def("tree_learning_rates", &ctboost::GradientBooster::tree_learning_rates)
      .def("max_depth", &ctboost::GradientBooster::max_depth)
      .def("alpha", &ctboost::GradientBooster::alpha)
      .def("lambda_l2", &ctboost::GradientBooster::lambda_l2)
      .def("subsample", &ctboost::GradientBooster::subsample)
      .def("bootstrap_type", &ctboost::GradientBooster::bootstrap_type)
      .def("bagging_temperature", &ctboost::GradientBooster::bagging_temperature)
      .def("boosting_type", &ctboost::GradientBooster::boosting_type)
      .def("drop_rate", &ctboost::GradientBooster::drop_rate)
      .def("skip_drop", &ctboost::GradientBooster::skip_drop)
      .def("max_drop", &ctboost::GradientBooster::max_drop)
      .def("monotone_constraints", &ctboost::GradientBooster::monotone_constraints)
      .def("interaction_constraints", &ctboost::GradientBooster::interaction_constraints)
      .def("colsample_bytree", &ctboost::GradientBooster::colsample_bytree)
      .def("feature_weights", &ctboost::GradientBooster::feature_weights)
      .def("first_feature_use_penalties", &ctboost::GradientBooster::first_feature_use_penalties)
      .def("random_strength", &ctboost::GradientBooster::random_strength)
      .def("grow_policy", &ctboost::GradientBooster::grow_policy)
      .def("max_leaves", &ctboost::GradientBooster::max_leaves)
      .def("min_samples_split", &ctboost::GradientBooster::min_samples_split)
      .def("min_data_in_leaf", &ctboost::GradientBooster::min_data_in_leaf)
      .def("min_child_weight", &ctboost::GradientBooster::min_child_weight)
      .def("gamma", &ctboost::GradientBooster::gamma)
      .def("max_leaf_weight", &ctboost::GradientBooster::max_leaf_weight)
      .def("max_bins", &ctboost::GradientBooster::max_bins)
      .def("nan_mode_name", &ctboost::GradientBooster::nan_mode_name)
      .def("max_bin_by_feature", &ctboost::GradientBooster::max_bin_by_feature)
      .def("border_selection_method", &ctboost::GradientBooster::border_selection_method)
      .def("nan_mode_by_feature", &ctboost::GradientBooster::nan_mode_by_feature)
      .def("feature_borders", &ctboost::GradientBooster::feature_borders)
      .def("external_memory", &ctboost::GradientBooster::external_memory)
      .def("external_memory_dir", &ctboost::GradientBooster::external_memory_dir)
      .def("quantile_alpha", &ctboost::GradientBooster::quantile_alpha)
      .def("huber_delta", &ctboost::GradientBooster::huber_delta)
      .def("tweedie_variance_power", &ctboost::GradientBooster::tweedie_variance_power)
      .def("use_gpu", &ctboost::GradientBooster::use_gpu)
      .def("devices", &ctboost::GradientBooster::devices)
      .def("distributed_world_size", &ctboost::GradientBooster::distributed_world_size)
      .def("distributed_rank", &ctboost::GradientBooster::distributed_rank)
      .def("distributed_root", &ctboost::GradientBooster::distributed_root)
      .def("distributed_run_id", &ctboost::GradientBooster::distributed_run_id)
      .def("distributed_timeout", &ctboost::GradientBooster::distributed_timeout)
      .def("random_seed", &ctboost::GradientBooster::random_seed)
      .def("rng_state", &ctboost::GradientBooster::rng_state)
      .def("verbose", &ctboost::GradientBooster::verbose);
}

}  // namespace ctboost::bindings
