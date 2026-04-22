#pragma once

#include "booster_internal.hpp"

namespace ctboost::booster_detail {

struct FitWorkspace {
  HistMatrix train_hist;
  HistMatrix eval_hist;
  GpuHistogramWorkspacePtr gpu_hist_workspace{nullptr, DestroyGpuHistogramWorkspace};
  std::vector<float> predictions;
  std::vector<float> eval_predictions;
  std::vector<float> gradients;
  std::vector<float> hessians;

  std::size_t train_hist_bytes() const noexcept { return train_hist.storage_bytes(); }
  std::size_t eval_hist_bytes() const noexcept { return eval_hist.storage_bytes(); }
  std::size_t gpu_workspace_bytes() const noexcept {
    return EstimateGpuHistogramWorkspaceBytes(gpu_hist_workspace.get());
  }

  void ReleaseFitScratch() noexcept {
    predictions.clear();
    predictions.shrink_to_fit();
    eval_predictions.clear();
    eval_predictions.shrink_to_fit();
    gradients.clear();
    gradients.shrink_to_fit();
    hessians.clear();
    hessians.shrink_to_fit();
    eval_hist.ReleaseStorage();
    train_hist.ReleaseStorage();
    gpu_hist_workspace.reset();
  }
};

struct DartPredictionState {
  std::vector<float> gradient_predictions;
  std::vector<float> dropped_train_predictions;
  std::vector<float> dropped_eval_predictions;
  std::vector<std::size_t> dropped_iterations;
};

struct IterationTiming {
  double gradient_ms{0.0};
  double tree_ms{0.0};
  double prediction_ms{0.0};
};

struct MetricSummary {
  double train_loss{0.0};
  double eval_score{0.0};
  double metric_ms{0.0};
  double eval_ms{0.0};
  bool early_stopped{false};
};

struct FitLoopState {
  int initial_completed_iterations{0};
  int completed_iterations{0};
  int target_total_iterations{0};
  bool early_stopped{false};
};

struct FitLoopContext {
  const Pool* pool{nullptr};
  const Pool* eval_pool{nullptr};
  const std::vector<float>* labels{nullptr};
  const std::vector<float>* weights{nullptr};
  const RankingMetadataView* ranking{nullptr};
  const std::vector<float>* eval_labels{nullptr};
  const std::vector<float>* eval_weights{nullptr};
  const RankingMetadataView* eval_ranking{nullptr};
  const InteractionConstraintSet* interaction_constraint_set{nullptr};
  const TrainingProfiler* profiler{nullptr};
  const QuantizationSchemaPtr* quantization_schema{nullptr};
  FitWorkspace* workspace{nullptr};
  ObjectiveFunction* objective{nullptr};
  MetricFunction* objective_metric{nullptr};
  MetricFunction* eval_metric{nullptr};
  std::vector<Tree>* trees{nullptr};
  std::vector<double>* tree_learning_rates{nullptr};
  std::vector<double>* loss_history{nullptr};
  std::vector<double>* eval_loss_history{nullptr};
  std::vector<double>* feature_importance_sums{nullptr};
  std::vector<std::uint8_t>* model_feature_used_mask{nullptr};
  std::vector<float>* fixed_gradient_predictions{nullptr};
  int* best_iteration{nullptr};
  double* best_score{nullptr};
  std::uint64_t* rng_state{nullptr};
  int prediction_dimension{1};
  int num_classes{1};
  int iterations{0};
  double learning_rate{0.0};
  bool use_gpu{false};
  const std::string* devices{nullptr};
  int distributed_world_size{1};
  int distributed_rank{0};
  const std::string* distributed_root{nullptr};
  const std::string* distributed_run_id{nullptr};
  double distributed_timeout{600.0};
  double subsample{1.0};
  double bagging_temperature{0.0};
  double drop_rate{0.0};
  double skip_drop{0.0};
  int max_drop{0};
  double colsample_bytree{1.0};
  bool maximize_eval_metric{false};
  int early_stopping_rounds{0};
  BootstrapType configured_bootstrap_type{BootstrapType::kNone};
  BoostingType boosting_type{BoostingType::kGradientBoosting};
  double alpha{0.05};
  int max_depth{6};
  double lambda_l2{1.0};
  const std::string* grow_policy{nullptr};
  int max_leaves{0};
  int min_samples_split{2};
  int min_data_in_leaf{0};
  double min_child_weight{0.0};
  double gamma{0.0};
  double max_leaf_weight{0.0};
  double random_strength{0.0};
  const std::vector<int>* monotone_constraints{nullptr};
  const std::vector<double>* feature_weights{nullptr};
  const std::vector<double>* first_feature_use_penalties{nullptr};
};

void LogFitMemorySnapshot(const TrainingProfiler& profiler,
                          const char* stage,
                          const Pool& train_pool,
                          const Pool* eval_pool,
                          const FitWorkspace& workspace);
void ValidateFitInputs(const Pool& pool,
                       const Pool* eval_pool,
                       int early_stopping_rounds,
                       const std::vector<int>& monotone_constraints,
                       const std::vector<std::vector<int>>& interaction_constraints,
                       const std::vector<double>& feature_weights,
                       const std::vector<double>& first_feature_use_penalties,
                       int prediction_dimension);
TreeBuildOptions MakeTreeBuildOptions(const FitLoopContext& context,
                                      const std::vector<int>* allowed_features,
                                      DistributedCoordinator* distributed_coordinator);
void PrepareGpuTrainingControls(const FitLoopContext& context,
                                const std::vector<float>& gradients,
                                const std::vector<float>& hessians,
                                const std::vector<float>& iteration_weights);
DartPredictionState PrepareDartPredictionState(const FitLoopContext& context,
                                               const FitLoopState& state);
void ApplyDroppedTreeAdjustments(const FitLoopContext& context,
                                 const DartPredictionState& dart_state,
                                 double dropped_tree_scale);
void RunSingleOutputIteration(const FitLoopContext& context,
                              const FitLoopState& state,
                              DistributedCoordinator* distributed_coordinator,
                              const std::vector<float>& iteration_weights,
                              const DartPredictionState& dart_state,
                              double dropped_tree_scale,
                              double new_tree_scale,
                              int total_iteration,
                              IterationTiming* timing);
void RunMulticlassIteration(const FitLoopContext& context,
                            const FitLoopState& state,
                            DistributedCoordinator* distributed_coordinator,
                            const std::vector<float>& iteration_weights,
                            const DartPredictionState& dart_state,
                            double dropped_tree_scale,
                            double new_tree_scale,
                            int total_iteration,
                            IterationTiming* timing);
MetricSummary EvaluateIterationMetrics(const FitLoopContext& context,
                                      FitLoopState& state,
                                      const DistributedCoordinator* distributed_coordinator,
                                      int total_iteration);
void RunTrainingLoop(const FitLoopContext& context, FitLoopState& state);

}  // namespace ctboost::booster_detail
