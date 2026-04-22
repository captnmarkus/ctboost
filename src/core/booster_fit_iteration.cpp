#include "booster_fit_internal.hpp"

namespace ctboost::booster_detail {

TreeBuildOptions MakeTreeBuildOptions(const FitLoopContext& context,
                                      const std::vector<int>* allowed_features,
                                      DistributedCoordinator* distributed_coordinator) {
  const bool has_feature_weights =
      context.feature_weights != nullptr && !context.feature_weights->empty();
  const bool has_first_use_penalties =
      context.first_feature_use_penalties != nullptr &&
      !context.first_feature_use_penalties->empty();
  const bool has_monotone_constraints =
      context.monotone_constraints != nullptr && !context.monotone_constraints->empty();
  return TreeBuildOptions{
      context.alpha,
      context.max_depth,
      context.lambda_l2,
      context.use_gpu,
      ParseGrowPolicy(*context.grow_policy),
      context.max_leaves,
      context.min_samples_split,
      context.min_data_in_leaf,
      context.min_child_weight,
      context.gamma,
      context.max_leaf_weight,
      context.random_strength,
      *context.rng_state,
      allowed_features,
      has_feature_weights ? context.feature_weights : nullptr,
      has_first_use_penalties ? context.first_feature_use_penalties : nullptr,
      has_first_use_penalties ? context.model_feature_used_mask : nullptr,
      has_monotone_constraints ? context.monotone_constraints : nullptr,
      context.interaction_constraint_set,
      distributed_coordinator,
  };
}

void PrepareGpuTrainingControls(const FitLoopContext& context,
                                const std::vector<float>& gradients,
                                const std::vector<float>& hessians,
                                const std::vector<float>& iteration_weights) {
  UploadHistogramTargetsGpu(context.workspace->gpu_hist_workspace.get(), gradients, hessians);
  UploadHistogramWeightsGpu(context.workspace->gpu_hist_workspace.get(), iteration_weights);
  UploadFeatureControlsGpu(context.workspace->gpu_hist_workspace.get(),
                           context.feature_weights != nullptr && !context.feature_weights->empty()
                               ? context.feature_weights
                               : nullptr,
                           context.first_feature_use_penalties != nullptr &&
                                   !context.first_feature_use_penalties->empty()
                               ? context.first_feature_use_penalties
                               : nullptr,
                           context.first_feature_use_penalties != nullptr &&
                                   !context.first_feature_use_penalties->empty()
                               ? context.model_feature_used_mask
                               : nullptr,
                           context.monotone_constraints != nullptr &&
                                   !context.monotone_constraints->empty()
                               ? context.monotone_constraints
                               : nullptr);
}

void ApplyDroppedTreeAdjustments(const FitLoopContext& context,
                                 const DartPredictionState& dart_state,
                                 double dropped_tree_scale) {
  if (dart_state.dropped_iterations.empty()) {
    return;
  }
  for (const std::size_t dropped_iteration : dart_state.dropped_iterations) {
    if (context.prediction_dimension == 1) {
      ScaleTreeLeafWeights((*context.trees)[dropped_iteration], dropped_tree_scale);
      continue;
    }
    for (int class_index = 0; class_index < context.prediction_dimension; ++class_index) {
      const std::size_t tree_index =
          dropped_iteration * static_cast<std::size_t>(context.prediction_dimension) +
          static_cast<std::size_t>(class_index);
      ScaleTreeLeafWeights((*context.trees)[tree_index], dropped_tree_scale);
    }
  }

  const float adjustment_scale = static_cast<float>(dropped_tree_scale - 1.0);
  for (std::size_t index = 0; index < context.workspace->predictions.size(); ++index) {
    context.workspace->predictions[index] +=
        adjustment_scale * dart_state.dropped_train_predictions[index];
  }
  if (context.eval_pool == nullptr) {
    return;
  }
  for (std::size_t index = 0; index < context.workspace->eval_predictions.size(); ++index) {
    context.workspace->eval_predictions[index] +=
        adjustment_scale * dart_state.dropped_eval_predictions[index];
  }
}

DartPredictionState PrepareDartPredictionState(const FitLoopContext& context,
                                               const FitLoopState& state) {
  DartPredictionState dart_state;
  if (context.boosting_type != BoostingType::kDart || context.trees->empty()) {
    return dart_state;
  }
  dart_state.dropped_iterations = SampleDroppedTreeGroups(
      static_cast<std::size_t>(state.completed_iterations),
      context.drop_rate,
      context.skip_drop,
      context.max_drop,
      *context.rng_state);
  if (dart_state.dropped_iterations.empty()) {
    return dart_state;
  }

  dart_state.gradient_predictions = context.workspace->predictions;
  dart_state.dropped_train_predictions.assign(dart_state.gradient_predictions.size(), 0.0F);
  if (context.eval_pool != nullptr) {
    dart_state.dropped_eval_predictions.assign(context.workspace->eval_predictions.size(), 0.0F);
  }
  for (const std::size_t dropped_iteration : dart_state.dropped_iterations) {
    AccumulateIterationPredictions(*context.trees,
                                   dropped_iteration,
                                   context.workspace->train_hist,
                                   *context.tree_learning_rates,
                                   context.learning_rate,
                                   context.prediction_dimension,
                                   dart_state.dropped_train_predictions);
    if (context.eval_pool != nullptr) {
      AccumulateIterationPredictions(*context.trees,
                                     dropped_iteration,
                                     context.workspace->eval_hist,
                                     *context.tree_learning_rates,
                                     context.learning_rate,
                                     context.prediction_dimension,
                                     dart_state.dropped_eval_predictions);
    }
  }
  for (std::size_t index = 0; index < dart_state.gradient_predictions.size(); ++index) {
    dart_state.gradient_predictions[index] -= dart_state.dropped_train_predictions[index];
  }
  return dart_state;
}

}  // namespace ctboost::booster_detail
