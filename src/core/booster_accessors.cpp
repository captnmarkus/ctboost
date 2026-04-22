#include "booster_internal.hpp"

namespace ctboost {

const std::vector<double>& GradientBooster::loss_history() const noexcept { return loss_history_; }
const std::vector<double>& GradientBooster::eval_loss_history() const noexcept { return eval_loss_history_; }
std::size_t GradientBooster::num_trees() const noexcept { return trees_.size(); }
std::size_t GradientBooster::num_iterations_trained() const noexcept {
  return prediction_dimension_ <= 0 ? 0 : trees_.size() / static_cast<std::size_t>(prediction_dimension_);
}
int GradientBooster::num_classes() const noexcept { return num_classes_; }
int GradientBooster::prediction_dimension() const noexcept { return prediction_dimension_; }
int GradientBooster::best_iteration() const noexcept { return best_iteration_; }
double GradientBooster::best_score() const noexcept { return best_score_; }
const std::string& GradientBooster::objective_name() const noexcept { return objective_name_; }
int GradientBooster::iterations() const noexcept { return iterations_; }
double GradientBooster::learning_rate() const noexcept { return learning_rate_; }
const std::vector<double>& GradientBooster::tree_learning_rates() const noexcept { return tree_learning_rates_; }
int GradientBooster::max_depth() const noexcept { return max_depth_; }
double GradientBooster::alpha() const noexcept { return alpha_; }
double GradientBooster::lambda_l2() const noexcept { return lambda_l2_; }
double GradientBooster::subsample() const noexcept { return subsample_; }
const std::string& GradientBooster::bootstrap_type() const noexcept { return bootstrap_type_; }
double GradientBooster::bagging_temperature() const noexcept { return bagging_temperature_; }
const std::string& GradientBooster::boosting_type() const noexcept { return boosting_type_; }
double GradientBooster::drop_rate() const noexcept { return drop_rate_; }
double GradientBooster::skip_drop() const noexcept { return skip_drop_; }
int GradientBooster::max_drop() const noexcept { return max_drop_; }
const std::vector<int>& GradientBooster::monotone_constraints() const noexcept { return monotone_constraints_; }
const std::vector<std::vector<int>>& GradientBooster::interaction_constraints() const noexcept { return interaction_constraints_; }
double GradientBooster::colsample_bytree() const noexcept { return colsample_bytree_; }
const std::vector<double>& GradientBooster::feature_weights() const noexcept { return feature_weights_; }
const std::vector<double>& GradientBooster::first_feature_use_penalties() const noexcept { return first_feature_use_penalties_; }
double GradientBooster::random_strength() const noexcept { return random_strength_; }
const std::string& GradientBooster::grow_policy() const noexcept { return grow_policy_; }
int GradientBooster::max_leaves() const noexcept { return max_leaves_; }
int GradientBooster::min_samples_split() const noexcept { return min_samples_split_; }
int GradientBooster::min_data_in_leaf() const noexcept { return min_data_in_leaf_; }
double GradientBooster::min_child_weight() const noexcept { return min_child_weight_; }
double GradientBooster::gamma() const noexcept { return gamma_; }
double GradientBooster::max_leaf_weight() const noexcept { return max_leaf_weight_; }
std::size_t GradientBooster::max_bins() const noexcept { return max_bins_; }
const std::string& GradientBooster::nan_mode_name() const noexcept { return hist_builder_.nan_mode_name(); }
const std::vector<std::uint16_t>& GradientBooster::max_bin_by_feature() const noexcept { return hist_builder_.max_bins_by_feature(); }
const std::string& GradientBooster::border_selection_method() const noexcept { return hist_builder_.border_selection_method_name(); }
const std::vector<std::string>& GradientBooster::nan_mode_by_feature() const noexcept { return hist_builder_.nan_mode_by_feature_names(); }
const std::vector<std::vector<float>>& GradientBooster::feature_borders() const noexcept { return hist_builder_.feature_borders(); }
bool GradientBooster::external_memory() const noexcept { return external_memory_; }
const std::string& GradientBooster::external_memory_dir() const noexcept { return external_memory_dir_; }
const std::string& GradientBooster::eval_metric_name() const noexcept { return eval_metric_name_; }
double GradientBooster::quantile_alpha() const noexcept { return objective_config_.quantile_alpha; }
double GradientBooster::huber_delta() const noexcept { return objective_config_.huber_delta; }
double GradientBooster::tweedie_variance_power() const noexcept { return objective_config_.tweedie_variance_power; }
bool GradientBooster::use_gpu() const noexcept { return use_gpu_; }
const std::string& GradientBooster::devices() const noexcept { return devices_; }
int GradientBooster::distributed_world_size() const noexcept { return distributed_world_size_; }
int GradientBooster::distributed_rank() const noexcept { return distributed_rank_; }
const std::string& GradientBooster::distributed_root() const noexcept { return distributed_root_; }
const std::string& GradientBooster::distributed_run_id() const noexcept { return distributed_run_id_; }
double GradientBooster::distributed_timeout() const noexcept { return distributed_timeout_; }
std::uint64_t GradientBooster::random_seed() const noexcept { return random_seed_; }
std::uint64_t GradientBooster::rng_state() const noexcept { return rng_state_; }
bool GradientBooster::verbose() const noexcept { return verbose_; }
const QuantizationSchema* GradientBooster::quantization_schema() const noexcept { return quantization_schema_.get(); }
const std::vector<Tree>& GradientBooster::trees() const noexcept { return trees_; }

}  // namespace ctboost
