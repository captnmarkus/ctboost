#include "booster_internal.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ctboost {

GradientBooster::GradientBooster(std::string objective,
                                 int iterations,
                                 double learning_rate,
                                 int max_depth,
                                 double alpha,
                                 double lambda_l2,
                                 double subsample,
                                 std::string bootstrap_type,
                                 double bagging_temperature,
                                 std::string boosting_type,
                                 double drop_rate,
                                 double skip_drop,
                                 int max_drop,
                                 std::vector<int> monotone_constraints,
                                 std::vector<std::vector<int>> interaction_constraints,
                                 double colsample_bytree,
                                 std::vector<double> feature_weights,
                                 std::vector<double> first_feature_use_penalties,
                                 double random_strength,
                                 std::string grow_policy,
                                 int max_leaves,
                                 int min_samples_split,
                                 int min_data_in_leaf,
                                 double min_child_weight,
                                 double gamma,
                                 double max_leaf_weight,
                                 int num_classes,
                                 std::size_t max_bins,
                                 std::string nan_mode,
                                 std::vector<std::uint16_t> max_bin_by_feature,
                                 std::string border_selection_method,
                                 std::vector<std::string> nan_mode_by_feature,
                                 std::vector<std::vector<float>> feature_borders,
                                 bool external_memory,
                                 std::string external_memory_dir,
                                 std::string eval_metric,
                                 double quantile_alpha,
                                 double huber_delta,
                                 double tweedie_variance_power,
                                 std::string task_type,
                                 std::string devices,
                                 int distributed_world_size,
                                 int distributed_rank,
                                 std::string distributed_root,
                                 std::string distributed_run_id,
                                 double distributed_timeout,
                                 std::uint64_t random_seed,
                                 bool verbose)
    : objective_name_(std::move(objective)),
      eval_metric_name_(std::move(eval_metric)),
      objective_config_{huber_delta, quantile_alpha, tweedie_variance_power},
      objective_(CreateObjectiveFunction(objective_name_, objective_config_)),
      objective_metric_(CreateMetricFunctionForObjective(objective_name_, objective_config_)),
      iterations_(iterations),
      learning_rate_(learning_rate),
      max_depth_(max_depth),
      alpha_(alpha),
      lambda_l2_(lambda_l2),
      subsample_(subsample),
      bootstrap_type_(booster_detail::CanonicalBootstrapType(std::move(bootstrap_type))),
      bagging_temperature_(bagging_temperature),
      boosting_type_(booster_detail::CanonicalBoostingType(std::move(boosting_type))),
      drop_rate_(drop_rate),
      skip_drop_(skip_drop),
      max_drop_(max_drop),
      monotone_constraints_(std::move(monotone_constraints)),
      interaction_constraints_(std::move(interaction_constraints)),
      colsample_bytree_(colsample_bytree),
      feature_weights_(std::move(feature_weights)),
      first_feature_use_penalties_(std::move(first_feature_use_penalties)),
      random_strength_(random_strength),
      grow_policy_(booster_detail::CanonicalGrowPolicy(std::move(grow_policy))),
      max_leaves_(max_leaves),
      min_samples_split_(min_samples_split),
      min_data_in_leaf_(min_data_in_leaf),
      min_child_weight_(min_child_weight),
      gamma_(gamma),
      max_leaf_weight_(max_leaf_weight),
      num_classes_(num_classes),
      max_bins_(max_bins),
      external_memory_(external_memory),
      external_memory_dir_(std::move(external_memory_dir)),
      devices_(std::move(devices)),
      distributed_world_size_(distributed_world_size),
      distributed_rank_(distributed_rank),
      distributed_root_(std::move(distributed_root)),
      distributed_run_id_(std::move(distributed_run_id)),
      distributed_timeout_(distributed_timeout),
      random_seed_(random_seed),
      rng_state_(booster_detail::NormalizeRngState(random_seed)),
      verbose_(TrainingProfiler::ResolveEnabled(verbose)),
      hist_builder_(max_bins_,
                    std::move(nan_mode),
                    std::move(max_bin_by_feature),
                    std::move(border_selection_method),
                    std::move(nan_mode_by_feature),
                    std::move(feature_borders),
                    external_memory_,
                    external_memory_dir_) {
  if (eval_metric_name_.empty()) {
    eval_metric_name_ = objective_name_;
  }
  eval_metric_ = CreateMetricFunction(eval_metric_name_, objective_config_);
  maximize_eval_metric_ = eval_metric_->HigherIsBetter();

  if (iterations_ <= 0) {
    throw std::invalid_argument("iterations must be positive");
  }
  if (learning_rate_ <= 0.0) {
    throw std::invalid_argument("learning_rate must be positive");
  }
  if (max_depth_ < 0 || lambda_l2_ < 0.0 || bagging_temperature_ < 0.0 || gamma_ < 0.0 ||
      max_leaf_weight_ < 0.0 || random_strength_ < 0.0) {
    throw std::invalid_argument("depth and regularization parameters must be non-negative");
  }
  if (subsample_ <= 0.0 || subsample_ > 1.0 || drop_rate_ < 0.0 || drop_rate_ > 1.0 ||
      skip_drop_ < 0.0 || skip_drop_ > 1.0 || colsample_bytree_ <= 0.0 ||
      colsample_bytree_ > 1.0) {
    throw std::invalid_argument("sampling parameters are out of range");
  }
  if (max_drop_ < 0 || max_leaves_ < 0 || min_samples_split_ < 2 || min_data_in_leaf_ < 0 ||
      min_child_weight_ < 0.0 || distributed_world_size_ <= 0 ||
      distributed_rank_ < 0 || distributed_rank_ >= distributed_world_size_ ||
      distributed_timeout_ <= 0.0 || num_classes_ <= 0) {
    throw std::invalid_argument("booster configuration is invalid");
  }
  (void)booster_detail::ParseBootstrapType(bootstrap_type_);
  (void)booster_detail::ParseBoostingType(boosting_type_);
  (void)booster_detail::ParseGrowPolicy(grow_policy_);
  for (const double value : feature_weights_) {
    if (value < 0.0) {
      throw std::invalid_argument("feature_weights entries must be non-negative");
    }
  }
  for (const double value : first_feature_use_penalties_) {
    if (value < 0.0) {
      throw std::invalid_argument("first_feature_use_penalties entries must be non-negative");
    }
  }

  const std::string normalized_objective = booster_detail::NormalizeToken(objective_name_);
  if (booster_detail::IsMulticlassObjective(normalized_objective)) {
    if (num_classes_ <= 2) {
      throw std::invalid_argument("multiclass objective requires num_classes greater than two");
    }
    prediction_dimension_ = num_classes_;
  } else if (booster_detail::IsRankingObjective(normalized_objective) ||
             booster_detail::IsRegressionObjective(normalized_objective)) {
    if (num_classes_ != 1) {
      throw std::invalid_argument("single-output objectives require num_classes equal to one");
    }
    prediction_dimension_ = 1;
  } else if (booster_detail::IsBinaryObjective(normalized_objective)) {
    if (num_classes_ != 1 && num_classes_ != 2) {
      throw std::invalid_argument("binary objectives require num_classes equal to one or two");
    }
    prediction_dimension_ = 1;
  }

  const std::string normalized_task_type = booster_detail::NormalizeTaskType(std::move(task_type));
  if (normalized_task_type == "cpu") {
    use_gpu_ = false;
  } else if (normalized_task_type == "gpu") {
    if (!CudaBackendCompiled()) {
      throw std::runtime_error(
          "task_type='GPU' was requested but CTBoost was compiled without CUDA support");
    }
    use_gpu_ = true;
  } else {
    throw std::invalid_argument("task_type must be either 'CPU' or 'GPU'");
  }
  if (use_gpu_ && distributed_world_size_ > 1 &&
      !DistributedRootUsesTcp(distributed_root_)) {
    throw std::invalid_argument(
        "distributed GPU training requires distributed_root to use a tcp://host:port coordinator");
  }
}

}  // namespace ctboost
