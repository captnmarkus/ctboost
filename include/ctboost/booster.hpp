#pragma once

#include <cstddef>
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
                  int num_classes = 1,
                  std::size_t max_bins = 256,
                  std::string nan_mode = "Min",
                  std::string eval_metric = "",
                  double quantile_alpha = 0.5,
                  double huber_delta = 1.0,
                  std::string task_type = "CPU",
                  std::string devices = "0",
                  bool verbose = false);

  void Fit(const Pool& pool,
           const Pool* eval_pool = nullptr,
           int early_stopping_rounds = 0,
           bool continue_training = false);
  std::vector<float> Predict(const Pool& pool, int num_iteration = -1) const;
  std::vector<std::int32_t> PredictLeafIndices(const Pool& pool, int num_iteration = -1) const;
  std::vector<float> PredictContributions(const Pool& pool, int num_iteration = -1) const;
  void LoadState(std::vector<Tree> trees,
                 std::vector<double> loss_history,
                 std::vector<double> eval_loss_history,
                 std::vector<double> feature_importance_sums,
                 int best_iteration,
                 double best_score,
                 bool use_gpu);

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
  int max_depth() const noexcept;
  double alpha() const noexcept;
  double lambda_l2() const noexcept;
  std::size_t max_bins() const noexcept;
  const std::string& nan_mode_name() const noexcept;
  const std::string& eval_metric_name() const noexcept;
  double quantile_alpha() const noexcept;
  double huber_delta() const noexcept;
  bool use_gpu() const noexcept;
  const std::string& devices() const noexcept;
  bool verbose() const noexcept;
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
  int num_classes_{1};
  int prediction_dimension_{1};
  std::size_t max_bins_{256};
  bool use_gpu_{false};
  std::string devices_{"0"};
  bool verbose_{false};
  HistBuilder hist_builder_;
  std::vector<Tree> trees_;
  std::vector<double> loss_history_;
  std::vector<double> eval_loss_history_;
  std::vector<double> feature_importance_sums_;
  int best_iteration_{-1};
  double best_score_{0.0};
  bool maximize_eval_metric_{false};
};

}  // namespace ctboost
