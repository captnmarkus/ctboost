#include "booster_fit_internal.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace ctboost {

void GradientBooster::Fit(Pool& pool,
                          Pool* eval_pool,
                          int early_stopping_rounds,
                          bool continue_training) {
  const auto fit_start = std::chrono::steady_clock::now();
  const TrainingProfiler profiler(verbose_);
  profiler.LogFitStart(pool.num_rows(), pool.num_cols(), iterations_, use_gpu_, prediction_dimension_);
  if (early_stopping_rounds < 0) {
    early_stopping_rounds = 0;
  }
  booster_detail::ValidateFitInputs(pool,
                                    eval_pool,
                                    early_stopping_rounds,
                                    monotone_constraints_,
                                    interaction_constraints_,
                                    feature_weights_,
                                    first_feature_use_penalties_,
                                    prediction_dimension_);

  const InteractionConstraintSet interaction_constraint_set =
      booster_detail::BuildInteractionConstraintSet(interaction_constraints_, pool.num_cols());
  const InteractionConstraintSet* interaction_constraint_ptr =
      interaction_constraint_set.groups.empty() ? nullptr : &interaction_constraint_set;
  const bool has_existing_state = continue_training && !trees_.empty();
  const QuantizationSchemaPtr imported_quantization_schema =
      has_existing_state ? QuantizationSchemaPtr{} : quantization_schema_;
  if (!continue_training) {
    trees_.clear();
    tree_learning_rates_.clear();
    quantization_schema_ = imported_quantization_schema;
    loss_history_.clear();
    eval_loss_history_.clear();
    best_iteration_ = -1;
    best_score_ = 0.0;
  } else if (!feature_importance_sums_.empty() && feature_importance_sums_.size() != pool.num_cols()) {
    throw std::invalid_argument(
        "warm-start training requires the same number of features as the initial model");
  }

  booster_detail::FitWorkspace workspace;
  booster_detail::LogFitMemorySnapshot(profiler, "pre_quantize", pool, eval_pool, workspace);
  const auto hist_build_start = std::chrono::steady_clock::now();
  if (has_existing_state || quantization_schema_ != nullptr) {
    workspace.train_hist =
        booster_detail::BuildPredictionHist(pool, booster_detail::RequireQuantizationSchema(quantization_schema_));
    if (external_memory_) {
      workspace.train_hist.SpillBinStorage(external_memory_dir_);
    }
  } else {
    workspace.train_hist = hist_builder_.Build(pool, &profiler);
    quantization_schema_ = std::make_shared<QuantizationSchema>(MakeQuantizationSchema(workspace.train_hist));
  }
  const double hist_build_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - hist_build_start).count();
  profiler.LogFitStage("train_quantize", hist_build_ms);
  booster_detail::LogFitMemorySnapshot(profiler, "post_quantize", pool, eval_pool, workspace);

  const auto& labels = pool.labels();
  const auto& weights = pool.weights();
  if (use_gpu_) {
    workspace.gpu_hist_workspace = CreateGpuHistogramWorkspace(workspace.train_hist, weights, devices_);
  }
  booster_detail::LogFitMemorySnapshot(profiler, "post_workspace", pool, eval_pool, workspace);
  const RankingMetadataView ranking = pool.ranking_metadata();
  const RankingMetadataView* ranking_ptr =
      ranking.group_ids == nullptr && ranking.subgroup_ids == nullptr && ranking.group_weights == nullptr &&
              ranking.pairs == nullptr
          ? nullptr
          : &ranking;
  const double total_weight =
      std::accumulate(weights.begin(), weights.end(), 0.0, [](double acc, float value) {
        return acc + static_cast<double>(value);
      });
  if (total_weight <= 0.0) {
    throw std::invalid_argument("training pool must have a positive total sample weight");
  }
  workspace.predictions = has_existing_state
                              ? booster_detail::PredictFromHist(trees_,
                                                                workspace.train_hist,
                                                                trees_.size(),
                                                                tree_learning_rates_,
                                                                learning_rate_,
                                                                use_gpu_,
                                                                prediction_dimension_,
                                                                devices_)
                              : std::vector<float>(pool.num_rows() * static_cast<std::size_t>(prediction_dimension_), 0.0F);
  booster_detail::AddPoolBaselineToPredictions(pool, prediction_dimension_, workspace.predictions);
  if (use_gpu_) {
    workspace.train_hist.ReleaseBinStorage();
  }

  const std::vector<float>* eval_labels = nullptr;
  const std::vector<float>* eval_weights = nullptr;
  RankingMetadataView eval_ranking;
  const RankingMetadataView* eval_ranking_ptr = nullptr;
  if (eval_pool != nullptr) {
    eval_labels = &eval_pool->labels();
    eval_weights = &eval_pool->weights();
    eval_ranking = eval_pool->ranking_metadata();
    eval_ranking_ptr = eval_ranking.group_ids == nullptr && eval_ranking.subgroup_ids == nullptr &&
                               eval_ranking.group_weights == nullptr && eval_ranking.pairs == nullptr
                           ? nullptr
                           : &eval_ranking;
    const double eval_total_weight =
        std::accumulate(eval_weights->begin(), eval_weights->end(), 0.0, [](double acc, float value) {
          return acc + static_cast<double>(value);
        });
    if (eval_total_weight <= 0.0) {
      throw std::invalid_argument("eval_pool must have a positive total sample weight");
    }
    workspace.eval_hist =
        booster_detail::BuildPredictionHist(*eval_pool, booster_detail::RequireQuantizationSchema(quantization_schema_));
    if (external_memory_) {
      workspace.eval_hist.SpillBinStorage(external_memory_dir_);
    }
    workspace.eval_predictions = has_existing_state
                                     ? booster_detail::PredictFromHist(trees_,
                                                                       workspace.eval_hist,
                                                                       trees_.size(),
                                                                       tree_learning_rates_,
                                                                       learning_rate_,
                                                                       use_gpu_,
                                                                       prediction_dimension_,
                                                                       devices_)
                                     : std::vector<float>(eval_pool->num_rows() * static_cast<std::size_t>(prediction_dimension_), 0.0F);
    booster_detail::AddPoolBaselineToPredictions(*eval_pool, prediction_dimension_, workspace.eval_predictions);
    if (!continue_training || eval_loss_history_.empty()) {
      best_score_ = maximize_eval_metric_ ? -std::numeric_limits<double>::infinity()
                                          : std::numeric_limits<double>::infinity();
      if (!continue_training) {
        best_iteration_ = -1;
      }
    }
  }

  if (distributed_world_size_ > 1 &&
      ((ranking_ptr != nullptr &&
        (ranking.subgroup_ids != nullptr || ranking.group_weights != nullptr || ranking.pairs != nullptr)) ||
       (eval_ranking_ptr != nullptr &&
        (eval_ranking.subgroup_ids != nullptr || eval_ranking.group_weights != nullptr ||
         eval_ranking.pairs != nullptr)))) {
    throw std::invalid_argument(
        "distributed training does not yet support subgroup_id, group_weight, or pairs metadata");
  }
  if (!continue_training || feature_importance_sums_.empty()) {
    feature_importance_sums_.assign(pool.num_cols(), 0.0);
  }
  std::vector<std::uint8_t> model_feature_used_mask(pool.num_cols(), 0U);
  if (has_existing_state) {
    for (const Tree& tree : trees_) {
      booster_detail::MarkUsedFeatures(tree, model_feature_used_mask);
    }
  }
  std::vector<float> fixed_gradient_predictions;
  if (booster_detail::ParseBoostingType(boosting_type_) == booster_detail::BoostingType::kRandomForest) {
    fixed_gradient_predictions = workspace.predictions;
  }
  pool.ReleaseFeatureStorage();
  if (eval_pool != nullptr) {
    eval_pool->ReleaseFeatureStorage();
  }
  booster_detail::LogFitMemorySnapshot(profiler, "post_release_dense", pool, eval_pool, workspace);

  booster_detail::FitLoopState state{
      static_cast<int>(num_iterations_trained()),
      static_cast<int>(num_iterations_trained()),
      static_cast<int>(num_iterations_trained()) + iterations_,
      false,
  };
  booster_detail::FitLoopContext context;
  context.pool = &pool;
  context.eval_pool = eval_pool;
  context.labels = &labels;
  context.weights = &weights;
  context.ranking = ranking_ptr;
  context.eval_labels = eval_labels;
  context.eval_weights = eval_weights;
  context.eval_ranking = eval_ranking_ptr;
  context.interaction_constraint_set = interaction_constraint_ptr;
  context.profiler = &profiler;
  context.quantization_schema = &quantization_schema_;
  context.workspace = &workspace;
  context.objective = objective_.get();
  context.objective_metric = objective_metric_.get();
  context.eval_metric = eval_metric_.get();
  context.trees = &trees_;
  context.tree_learning_rates = &tree_learning_rates_;
  context.loss_history = &loss_history_;
  context.eval_loss_history = &eval_loss_history_;
  context.feature_importance_sums = &feature_importance_sums_;
  context.model_feature_used_mask = &model_feature_used_mask;
  context.fixed_gradient_predictions = &fixed_gradient_predictions;
  context.best_iteration = &best_iteration_;
  context.best_score = &best_score_;
  context.rng_state = &rng_state_;
  context.prediction_dimension = prediction_dimension_;
  context.num_classes = num_classes_;
  context.iterations = iterations_;
  context.learning_rate = learning_rate_;
  context.use_gpu = use_gpu_;
  context.devices = &devices_;
  context.distributed_world_size = distributed_world_size_;
  context.distributed_rank = distributed_rank_;
  context.distributed_root = &distributed_root_;
  context.distributed_run_id = &distributed_run_id_;
  context.distributed_timeout = distributed_timeout_;
  context.subsample = subsample_;
  context.bagging_temperature = bagging_temperature_;
  context.drop_rate = drop_rate_;
  context.skip_drop = skip_drop_;
  context.max_drop = max_drop_;
  context.colsample_bytree = colsample_bytree_;
  context.maximize_eval_metric = maximize_eval_metric_;
  context.early_stopping_rounds = early_stopping_rounds;
  context.configured_bootstrap_type = booster_detail::ParseBootstrapType(bootstrap_type_);
  context.boosting_type = booster_detail::ParseBoostingType(boosting_type_);
  context.alpha = alpha_;
  context.max_depth = max_depth_;
  context.lambda_l2 = lambda_l2_;
  context.grow_policy = &grow_policy_;
  context.max_leaves = max_leaves_;
  context.min_samples_split = min_samples_split_;
  context.min_data_in_leaf = min_data_in_leaf_;
  context.min_child_weight = min_child_weight_;
  context.gamma = gamma_;
  context.max_leaf_weight = max_leaf_weight_;
  context.random_strength = random_strength_;
  context.monotone_constraints = &monotone_constraints_;
  context.feature_weights = &feature_weights_;
  context.first_feature_use_penalties = &first_feature_use_penalties_;
  booster_detail::RunTrainingLoop(context, state);

  const auto finish_fit = [&](double total_fit_ms) {
    workspace.ReleaseFitScratch();
    booster_detail::LogFitMemorySnapshot(profiler, "post_cleanup", pool, eval_pool, workspace);
    profiler.LogFitSummary(hist_build_ms, total_fit_ms);
  };
  if (eval_pool == nullptr) {
    best_iteration_ = state.completed_iterations > 0 ? state.completed_iterations - 1 : -1;
  } else if (state.early_stopped && best_iteration_ >= 0) {
    const std::size_t retained_iterations = static_cast<std::size_t>(best_iteration_ + 1);
    trees_.resize(retained_iterations * static_cast<std::size_t>(prediction_dimension_));
    if (tree_learning_rates_.size() > retained_iterations) {
      tree_learning_rates_.resize(retained_iterations);
    }
    loss_history_.resize(retained_iterations);
    eval_loss_history_.resize(retained_iterations);
    booster_detail::RecomputeFeatureImportances(trees_, pool.num_cols(), feature_importance_sums_);
  }
  const double total_fit_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - fit_start).count();
  finish_fit(total_fit_ms);
}

}  // namespace ctboost
