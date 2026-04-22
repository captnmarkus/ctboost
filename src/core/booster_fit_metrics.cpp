#include "booster_fit_internal.hpp"

namespace ctboost::booster_detail {

MetricSummary EvaluateIterationMetrics(const FitLoopContext& context,
                                      FitLoopState& state,
                                      const DistributedCoordinator* distributed_coordinator,
                                      int total_iteration) {
  MetricSummary summary;
  const bool distributed_tcp =
      distributed_coordinator != nullptr && DistributedRootUsesTcp(distributed_coordinator->root);

  if (distributed_tcp) {
    DistributedMetricInputs local_train_inputs;
    local_train_inputs.predictions = context.workspace->predictions;
    local_train_inputs.labels = *context.labels;
    local_train_inputs.weights = *context.weights;
    local_train_inputs.has_group_ids = context.ranking != nullptr && context.ranking->group_ids != nullptr;
    if (local_train_inputs.has_group_ids) {
      local_train_inputs.group_ids = *context.ranking->group_ids;
    }
    const DistributedMetricInputs gathered_train_inputs =
        AllGatherDistributedMetricInputs(distributed_coordinator, "train_metric", local_train_inputs);
    const RankingMetadataView gathered_train_ranking{
        gathered_train_inputs.has_group_ids ? &gathered_train_inputs.group_ids : nullptr,
        nullptr,
        nullptr,
        nullptr,
    };
    summary.train_loss = context.objective_metric->Evaluate(gathered_train_inputs.predictions,
                                                            gathered_train_inputs.labels,
                                                            gathered_train_inputs.weights,
                                                            context.num_classes,
                                                            gathered_train_inputs.has_group_ids
                                                                ? &gathered_train_ranking
                                                                : nullptr);
  } else {
    summary.train_loss = context.objective_metric->Evaluate(
        context.workspace->predictions, *context.labels, *context.weights, context.num_classes, context.ranking);
  }
  state.completed_iterations = total_iteration + 1;
  if (context.eval_pool == nullptr) {
    if (distributed_tcp) {
      DistributedMetricControl root_control;
      if (distributed_coordinator->rank == 0) {
        root_control.train_loss = summary.train_loss;
      }
      const DistributedMetricControl synced_control = BroadcastDistributedMetricControl(
          distributed_coordinator, "metric", distributed_coordinator->rank == 0 ? &root_control : nullptr);
      summary.train_loss = synced_control.train_loss;
    }
    context.loss_history->push_back(summary.train_loss);
    return summary;
  }

  if (distributed_tcp) {
    DistributedMetricInputs local_eval_inputs;
    local_eval_inputs.predictions = context.workspace->eval_predictions;
    local_eval_inputs.labels = *context.eval_labels;
    local_eval_inputs.weights = *context.eval_weights;
    local_eval_inputs.has_group_ids =
        context.eval_ranking != nullptr && context.eval_ranking->group_ids != nullptr;
    if (local_eval_inputs.has_group_ids) {
      local_eval_inputs.group_ids = *context.eval_ranking->group_ids;
    }
    const DistributedMetricInputs gathered_eval_inputs =
        AllGatherDistributedMetricInputs(distributed_coordinator, "eval_metric", local_eval_inputs);
    const RankingMetadataView gathered_eval_ranking{
        gathered_eval_inputs.has_group_ids ? &gathered_eval_inputs.group_ids : nullptr,
        nullptr,
        nullptr,
        nullptr,
    };
    summary.eval_score = context.eval_metric->Evaluate(gathered_eval_inputs.predictions,
                                                       gathered_eval_inputs.labels,
                                                       gathered_eval_inputs.weights,
                                                       context.num_classes,
                                                       gathered_eval_inputs.has_group_ids
                                                           ? &gathered_eval_ranking
                                                           : nullptr);
  } else {
    summary.eval_score = context.eval_metric->Evaluate(context.workspace->eval_predictions,
                                                       *context.eval_labels,
                                                       *context.eval_weights,
                                                       context.num_classes,
                                                       context.eval_ranking);
  }

  if (distributed_tcp) {
    DistributedMetricControl root_control;
    if (distributed_coordinator->rank == 0) {
      root_control.train_loss = summary.train_loss;
      root_control.eval_score = summary.eval_score;
      root_control.has_eval = 1U;
      const bool improved = *context.best_iteration < 0 ||
                            (context.maximize_eval_metric ? summary.eval_score > *context.best_score
                                                          : summary.eval_score < *context.best_score);
      if (improved) {
        *context.best_iteration = total_iteration;
        *context.best_score = summary.eval_score;
      }
      root_control.best_iteration = *context.best_iteration;
      root_control.best_score = *context.best_score;
      root_control.should_stop =
          !improved && context.early_stopping_rounds > 0 &&
                  total_iteration - *context.best_iteration >= context.early_stopping_rounds
              ? 1U
              : 0U;
    }
    const DistributedMetricControl synced_control = BroadcastDistributedMetricControl(
        distributed_coordinator, "metric", distributed_coordinator->rank == 0 ? &root_control : nullptr);
    summary.train_loss = synced_control.train_loss;
    if (synced_control.has_eval != 0U) {
      summary.eval_score = synced_control.eval_score;
      *context.best_iteration = synced_control.best_iteration;
      *context.best_score = synced_control.best_score;
    }
    summary.early_stopped = synced_control.should_stop != 0U;
  } else {
    const bool improved = *context.best_iteration < 0 ||
                          (context.maximize_eval_metric ? summary.eval_score > *context.best_score
                                                        : summary.eval_score < *context.best_score);
    if (improved) {
      *context.best_iteration = total_iteration;
      *context.best_score = summary.eval_score;
    } else if (context.early_stopping_rounds > 0 &&
               total_iteration - *context.best_iteration >= context.early_stopping_rounds) {
      summary.early_stopped = true;
    }
  }
  context.loss_history->push_back(summary.train_loss);
  context.eval_loss_history->push_back(summary.eval_score);
  return summary;
}

}  // namespace ctboost::booster_detail
