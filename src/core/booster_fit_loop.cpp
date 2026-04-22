#include "booster_fit_internal.hpp"

#include <chrono>

namespace ctboost::booster_detail {

void RunTrainingLoop(const FitLoopContext& context, FitLoopState& state) {
  for (int iteration = 0; iteration < context.iterations; ++iteration) {
    const auto iteration_start = std::chrono::steady_clock::now();
    const int total_iteration = state.initial_completed_iterations + iteration;
    DistributedCoordinator distributed_coordinator;
    DistributedCoordinator* distributed_ptr = nullptr;
    if (context.distributed_world_size > 1) {
      distributed_coordinator.world_size = context.distributed_world_size;
      distributed_coordinator.rank = context.distributed_rank;
      distributed_coordinator.root = *context.distributed_root;
      distributed_coordinator.run_id = *context.distributed_run_id;
      distributed_coordinator.timeout_seconds = context.distributed_timeout;
      distributed_coordinator.tree_index = static_cast<std::size_t>(total_iteration);
      distributed_coordinator.operation_counter = 0;
      distributed_ptr = &distributed_coordinator;
    }

    BootstrapType iteration_bootstrap_type = context.configured_bootstrap_type;
    if (context.boosting_type == BoostingType::kRandomForest &&
        iteration_bootstrap_type == BootstrapType::kNone && context.subsample >= 1.0) {
      iteration_bootstrap_type = BootstrapType::kPoisson;
    } else if (iteration_bootstrap_type == BootstrapType::kNone && context.subsample < 1.0) {
      iteration_bootstrap_type = BootstrapType::kBernoulli;
    }
    const std::vector<float> iteration_weights = SampleRowWeights(
        *context.weights,
        context.subsample,
        iteration_bootstrap_type,
        context.bagging_temperature,
        *context.rng_state);
    const DartPredictionState dart_state = PrepareDartPredictionState(context, state);
    const std::vector<float>& gradient_predictions =
        context.boosting_type == BoostingType::kRandomForest ? *context.fixed_gradient_predictions
        : (context.boosting_type == BoostingType::kDart &&
           !dart_state.gradient_predictions.empty())
            ? dart_state.gradient_predictions
            : context.workspace->predictions;

    const auto gradient_start = std::chrono::steady_clock::now();
    context.objective->compute_gradients(gradient_predictions,
                                         *context.labels,
                                         context.workspace->gradients,
                                         context.workspace->hessians,
                                         context.num_classes,
                                         context.ranking);
    IterationTiming timing;
    timing.gradient_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - gradient_start).count();
    const double dropped_tree_scale =
        dart_state.dropped_iterations.empty()
            ? 1.0
            : static_cast<double>(dart_state.dropped_iterations.size()) /
                  static_cast<double>(dart_state.dropped_iterations.size() + 1U);
    const double new_tree_scale =
        dart_state.dropped_iterations.empty()
            ? 1.0
            : 1.0 / static_cast<double>(dart_state.dropped_iterations.size() + 1U);

    if (context.prediction_dimension == 1) {
      RunSingleOutputIteration(context,
                               state,
                               distributed_ptr,
                               iteration_weights,
                               dart_state,
                               dropped_tree_scale,
                               new_tree_scale,
                               total_iteration,
                               &timing);
    } else {
      RunMulticlassIteration(context,
                             state,
                             distributed_ptr,
                             iteration_weights,
                             dart_state,
                             dropped_tree_scale,
                             new_tree_scale,
                             total_iteration,
                             &timing);
    }

    const auto metric_start = std::chrono::steady_clock::now();
    MetricSummary metrics = EvaluateIterationMetrics(context, state, distributed_ptr, total_iteration);
    metrics.metric_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - metric_start).count();
    const double iteration_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - iteration_start).count();
    context.profiler->LogIteration(total_iteration + 1,
                                   state.target_total_iterations,
                                   timing.gradient_ms,
                                   timing.tree_ms,
                                   timing.prediction_ms,
                                   metrics.metric_ms,
                                   metrics.eval_ms,
                                   iteration_ms);
    if (metrics.early_stopped) {
      state.early_stopped = true;
      break;
    }
  }
}

}  // namespace ctboost::booster_detail
