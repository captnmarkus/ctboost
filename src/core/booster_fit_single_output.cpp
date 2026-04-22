#include "booster_fit_internal.hpp"

#include <chrono>

namespace ctboost::booster_detail {

void RunSingleOutputIteration(const FitLoopContext& context,
                              const FitLoopState& state,
                              DistributedCoordinator* distributed_coordinator,
                              const std::vector<float>& iteration_weights,
                              const DartPredictionState& dart_state,
                              double dropped_tree_scale,
                              double new_tree_scale,
                              int total_iteration,
                              IterationTiming* timing) {
  if (context.use_gpu) {
    PrepareGpuTrainingControls(context, context.workspace->gradients, context.workspace->hessians, iteration_weights);
  }

  Tree tree;
  const std::vector<int> allowed_features = SampleFeatureSubset(
      context.pool->num_cols(), context.colsample_bytree, context.feature_weights, *context.rng_state);
  const TreeBuildOptions build_options = MakeTreeBuildOptions(
      context, allowed_features.empty() ? nullptr : &allowed_features, distributed_coordinator);
  std::vector<std::size_t> training_row_indices;
  std::vector<LeafRowRange> training_leaf_ranges;
  const auto tree_start = std::chrono::steady_clock::now();
  tree.Build(context.workspace->train_hist,
             context.workspace->gradients,
             context.workspace->hessians,
             iteration_weights,
             build_options,
             context.workspace->gpu_hist_workspace.get(),
             context.profiler,
             &training_row_indices,
             &training_leaf_ranges,
             *context.quantization_schema);
  const double single_tree_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tree_start).count();
  timing->tree_ms += single_tree_ms;
  context.profiler->LogTreeBuild(
      total_iteration + 1, state.target_total_iterations, 0, context.prediction_dimension, single_tree_ms);

  const auto prediction_start = std::chrono::steady_clock::now();
  ApplyDroppedTreeAdjustments(context, dart_state, dropped_tree_scale);
  if (new_tree_scale != 1.0) {
    ScaleTreeLeafWeights(tree, new_tree_scale);
  }
  UpdatePredictionsFromLeafRanges(tree,
                                  training_row_indices,
                                  training_leaf_ranges,
                                  context.learning_rate * new_tree_scale,
                                  context.prediction_dimension,
                                  0,
                                  context.workspace->predictions);
  if (context.eval_pool != nullptr) {
    UpdatePredictions(tree,
                      context.workspace->eval_hist,
                      context.learning_rate * new_tree_scale,
                      context.prediction_dimension,
                      0,
                      context.workspace->eval_predictions);
  }
  timing->prediction_ms +=
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - prediction_start).count();
  AccumulateFeatureImportances(tree, *context.feature_importance_sums);
  MarkUsedFeatures(tree, *context.model_feature_used_mask);
  context.trees->push_back(std::move(tree));
  context.tree_learning_rates->push_back(context.learning_rate);
}

}  // namespace ctboost::booster_detail
