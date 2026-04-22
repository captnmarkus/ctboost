#include "booster_fit_internal.hpp"

#include <chrono>

namespace ctboost::booster_detail {

void RunMulticlassIteration(const FitLoopContext& context,
                            const FitLoopState& state,
                            DistributedCoordinator* distributed_coordinator,
                            const std::vector<float>& iteration_weights,
                            const DartPredictionState& dart_state,
                            double dropped_tree_scale,
                            double new_tree_scale,
                            int total_iteration,
                            IterationTiming* timing) {
  std::vector<float> structure_gradients;
  std::vector<float> structure_hessians;
  BuildSharedMulticlassTargets(context.workspace->gradients,
                               context.workspace->hessians,
                               iteration_weights,
                               context.pool->num_rows(),
                               context.prediction_dimension,
                               structure_gradients,
                               structure_hessians);
  if (context.use_gpu) {
    PrepareGpuTrainingControls(context, structure_gradients, structure_hessians, iteration_weights);
  }

  Tree structure_tree;
  const std::vector<int> allowed_features = SampleFeatureSubset(
      context.pool->num_cols(), context.colsample_bytree, context.feature_weights, *context.rng_state);
  const TreeBuildOptions build_options = MakeTreeBuildOptions(
      context, allowed_features.empty() ? nullptr : &allowed_features, distributed_coordinator);
  std::vector<std::size_t> training_row_indices;
  std::vector<LeafRowRange> training_leaf_ranges;
  const auto tree_start = std::chrono::steady_clock::now();
  structure_tree.Build(context.workspace->train_hist,
                       structure_gradients,
                       structure_hessians,
                       iteration_weights,
                       build_options,
                       context.workspace->gpu_hist_workspace.get(),
                       context.profiler,
                       &training_row_indices,
                       &training_leaf_ranges,
                       *context.quantization_schema);
  const double shared_tree_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tree_start).count();
  timing->tree_ms += shared_tree_ms;
  context.profiler->LogTreeBuild(total_iteration + 1,
                                 state.target_total_iterations,
                                 -1,
                                 context.prediction_dimension,
                                 shared_tree_ms);

  const auto leaf_fit_start = std::chrono::steady_clock::now();
  std::vector<Tree> class_trees = MaterializeMulticlassTreesFromStructure(structure_tree,
                                                                          training_row_indices,
                                                                          training_leaf_ranges,
                                                                          context.workspace->gradients,
                                                                          context.workspace->hessians,
                                                                          iteration_weights,
                                                                          context.prediction_dimension,
                                                                          context.lambda_l2);
  timing->tree_ms +=
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - leaf_fit_start).count();

  std::vector<int> eval_leaf_indices;
  if (context.eval_pool != nullptr) {
    eval_leaf_indices = PredictLeafIndicesFromHist(structure_tree, context.workspace->eval_hist);
  }

  const auto prediction_start = std::chrono::steady_clock::now();
  ApplyDroppedTreeAdjustments(context, dart_state, dropped_tree_scale);
  for (int class_index = 0; class_index < context.prediction_dimension; ++class_index) {
    Tree& tree = class_trees[static_cast<std::size_t>(class_index)];
    if (new_tree_scale != 1.0) {
      ScaleTreeLeafWeights(tree, new_tree_scale);
    }
    UpdatePredictionsFromLeafRanges(tree,
                                    training_row_indices,
                                    training_leaf_ranges,
                                    context.learning_rate * new_tree_scale,
                                    context.prediction_dimension,
                                    class_index,
                                    context.workspace->predictions);
    if (context.eval_pool != nullptr) {
      UpdatePredictionsFromLeafIndices(tree,
                                       eval_leaf_indices,
                                       context.learning_rate * new_tree_scale,
                                       context.prediction_dimension,
                                       class_index,
                                       context.workspace->eval_predictions);
    }
    AccumulateFeatureImportances(tree, *context.feature_importance_sums);
    MarkUsedFeatures(tree, *context.model_feature_used_mask);
    context.trees->push_back(std::move(tree));
  }
  context.tree_learning_rates->push_back(context.learning_rate);
  timing->prediction_ms +=
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - prediction_start).count();
}

}  // namespace ctboost::booster_detail
