#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "ctboost/data.hpp"
#include "ctboost/histogram.hpp"
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
                  std::size_t max_bins = 256);

  void Fit(const Pool& pool);
  std::vector<float> Predict(const Pool& pool) const;

  const std::vector<double>& loss_history() const noexcept;
  std::size_t num_trees() const noexcept;

 private:
  std::string objective_name_;
  std::unique_ptr<ObjectiveFunction> objective_;
  int iterations_{100};
  double learning_rate_{0.1};
  int max_depth_{6};
  double alpha_{0.05};
  double lambda_l2_{1.0};
  std::size_t max_bins_{256};
  HistBuilder hist_builder_;
  std::vector<Tree> trees_;
  std::vector<double> loss_history_;
};

}  // namespace ctboost
