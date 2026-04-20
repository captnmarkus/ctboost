#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "ctboost/data.hpp"
#include "ctboost/histogram.hpp"
#include "ctboost/metric.hpp"
#include "ctboost/objective.hpp"
#include "ctboost/tree.hpp"

namespace ctboost {

class GradientBooster {
 public:
  GradientBooster(std::string objective = "RMSE",
                  int iterations = 100,
                  double learning_rate = 0.1,
                  int max_depth = 6,
                  double alpha = 0.05,
                  double lambda_l2 = 1.0,
                  double subsample = 1.0,
                  std::string bootstrap_type = "No",
                  double bagging_temperature = 0.0,
                  std::string boosting_type = "GradientBoosting",
                  double drop_rate = 0.1,
                  double skip_drop = 0.5,
                  int max_drop = 0,
                  std::vector<int> monotone_constraints = {},
                  std::vector<std::vector<int>> interaction_constraints = {},
                  double colsample_bytree = 1.0,
                  std::vector<double> feature_weights = {},
                  std::vector<double> first_feature_use_penalties = {},
                  double random_strength = 0.0,
                  std::string grow_policy = "DepthWise",
                  int max_leaves = 0,
                  int min_samples_split = 2,
                  int min_data_in_leaf = 0,
                  double min_child_weight = 0.0,
                  double gamma = 0.0,
                  double max_leaf_weight = 0.0,
                  int num_classes = 1,
                  std::size_t max_bins = 256,
                  std::string nan_mode = "Min",
                  std::vector<std::uint16_t> max_bin_by_feature = {},
                  std::string border_selection_method = "Quantile",
                  std::vector<std::string> nan_mode_by_feature = {},
                  std::vector<std::vector<float>> feature_borders = {},
                  bool external_memory = false,
                  std::string external_memory_dir = "",
                  std::string eval_metric = "",
                  double quantile_alpha = 0.5,
                  double huber_delta = 1.0,
                  double tweedie_variance_power = 1.5,
                  std::string task_type = "CPU",
                  std::string devices = "0",
                  int distributed_world_size = 1,
                  int distributed_rank = 0,
                  std::string distributed_root = "",
                  std::string distributed_run_id = "default",
                  double distributed_timeout = 600.0,
                  std::uint64_t random_seed = 0,
                  bool verbose = false);

  void Fit(Pool& pool,
           Pool* eval_pool = nullptr,
           int early_stopping_rounds = 0,
           bool continue_training = false);
  void SetIterations(int iterations);
  void SetLearningRate(double learning_rate);
  std::vector<float> Predict(const Pool& pool, int num_iteration = -1) const;
  std::vector<std::int32_t> PredictLeafIndices(const Pool& pool, int num_iteration = -1) const;
  std::vector<float> PredictContributions(const Pool& pool, int num_iteration = -1) const;
  void LoadState(std::vector<Tree> trees,
                 QuantizationSchemaPtr quantization_schema,
                 std::vector<double> loss_history,
                 std::vector<double> eval_loss_history,
                 std::vector<double> tree_learning_rates,
                 std::vector<double> feature_importance_sums,
                 int best_iteration,
                 double best_score,
                 bool use_gpu,
                 std::uint64_t rng_state = 0);
  void LoadQuantizationSchema(QuantizationSchemaPtr quantization_schema);

  const std::vector<double>& loss_history() const noexcept;
  const std::vector<double>& eval_loss_history() const noexcept;
  std::size_t num_trees() const noexcept;
  std::size_t num_iterations_trained() const noexcept;
  int num_classes() const noexcept;
  int prediction_dimension() const noexcept;
  int best_iteration() const noexcept;
  double best_score() const noexcept;
  const std::string& objective_name() const noexcept;
  int iterations() const noexcept;
  double learning_rate() const noexcept;
  const std::vector<double>& tree_learning_rates() const noexcept;
  int max_depth() const noexcept;
  double alpha() const noexcept;
  double lambda_l2() const noexcept;
  double subsample() const noexcept;
  const std::string& bootstrap_type() const noexcept;
  double bagging_temperature() const noexcept;
  const std::string& boosting_type() const noexcept;
  double drop_rate() const noexcept;
  double skip_drop() const noexcept;
  int max_drop() const noexcept;
  const std::vector<int>& monotone_constraints() const noexcept;
  const std::vector<std::vector<int>>& interaction_constraints() const noexcept;
  double colsample_bytree() const noexcept;
  const std::vector<double>& feature_weights() const noexcept;
  const std::vector<double>& first_feature_use_penalties() const noexcept;
  double random_strength() const noexcept;
  const std::string& grow_policy() const noexcept;
  int max_leaves() const noexcept;
  int min_samples_split() const noexcept;
  int min_data_in_leaf() const noexcept;
  double min_child_weight() const noexcept;
  double gamma() const noexcept;
  double max_leaf_weight() const noexcept;
  std::size_t max_bins() const noexcept;
  const std::string& nan_mode_name() const noexcept;
  const std::vector<std::uint16_t>& max_bin_by_feature() const noexcept;
  const std::string& border_selection_method() const noexcept;
  const std::vector<std::string>& nan_mode_by_feature() const noexcept;
  const std::vector<std::vector<float>>& feature_borders() const noexcept;
  bool external_memory() const noexcept;
  const std::string& external_memory_dir() const noexcept;
  const std::string& eval_metric_name() const noexcept;
  double quantile_alpha() const noexcept;
  double huber_delta() const noexcept;
  double tweedie_variance_power() const noexcept;
  bool use_gpu() const noexcept;
  const std::string& devices() const noexcept;
  int distributed_world_size() const noexcept;
  int distributed_rank() const noexcept;
  const std::string& distributed_root() const noexcept;
  const std::string& distributed_run_id() const noexcept;
  double distributed_timeout() const noexcept;
  std::uint64_t random_seed() const noexcept;
  std::uint64_t rng_state() const noexcept;
  bool verbose() const noexcept;
  const QuantizationSchema* quantization_schema() const noexcept;
  const std::vector<Tree>& trees() const noexcept;
  std::vector<float> get_feature_importances() const;

 private:
  std::string objective_name_;
  std::string eval_metric_name_;
  ObjectiveConfig objective_config_;
  std::unique_ptr<ObjectiveFunction> objective_;
  std::unique_ptr<MetricFunction> objective_metric_;
  std::unique_ptr<MetricFunction> eval_metric_;
  int iterations_{100};
  double learning_rate_{0.1};
  int max_depth_{6};
  double alpha_{0.05};
  double lambda_l2_{1.0};
  double subsample_{1.0};
  std::string bootstrap_type_{"No"};
  double bagging_temperature_{0.0};
  std::string boosting_type_{"GradientBoosting"};
  double drop_rate_{0.1};
  double skip_drop_{0.5};
  int max_drop_{0};
  std::vector<int> monotone_constraints_;
  std::vector<std::vector<int>> interaction_constraints_;
  double colsample_bytree_{1.0};
  std::vector<double> feature_weights_;
  std::vector<double> first_feature_use_penalties_;
  double random_strength_{0.0};
  std::string grow_policy_{"DepthWise"};
  int max_leaves_{0};
  int min_samples_split_{2};
  int min_data_in_leaf_{0};
  double min_child_weight_{0.0};
  double gamma_{0.0};
  double max_leaf_weight_{0.0};
  int num_classes_{1};
  int prediction_dimension_{1};
  std::size_t max_bins_{256};
  bool external_memory_{false};
  std::string external_memory_dir_;
  bool use_gpu_{false};
  std::string devices_{"0"};
  int distributed_world_size_{1};
  int distributed_rank_{0};
  std::string distributed_root_;
  std::string distributed_run_id_{"default"};
  double distributed_timeout_{600.0};
  std::uint64_t random_seed_{0};
  std::uint64_t rng_state_{0};
  bool verbose_{false};
  HistBuilder hist_builder_;
  QuantizationSchemaPtr quantization_schema_;
  std::vector<Tree> trees_;
  std::vector<double> tree_learning_rates_;
  std::vector<double> loss_history_;
  std::vector<double> eval_loss_history_;
  std::vector<double> feature_importance_sums_;
  int best_iteration_{-1};
  double best_score_{0.0};
  bool maximize_eval_metric_{false};
};

}  // namespace ctboost
