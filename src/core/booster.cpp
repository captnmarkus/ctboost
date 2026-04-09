#include "ctboost/booster.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ctboost {
namespace {

double ComputeLoss(const std::string& objective_name,
                   const std::vector<float>& predictions,
                   const std::vector<float>& labels) {
  if (predictions.size() != labels.size()) {
    throw std::invalid_argument("predictions and labels must have the same size");
  }

  std::string normalized = objective_name;
  std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });

  if (normalized == "rmse" || normalized == "squarederror" || normalized == "squared_error") {
    double sum_squared_error = 0.0;
    for (std::size_t i = 0; i < predictions.size(); ++i) {
      const double residual = static_cast<double>(predictions[i]) - labels[i];
      sum_squared_error += residual * residual;
    }
    return predictions.empty() ? 0.0 : sum_squared_error / predictions.size();
  }

  if (normalized == "logloss" || normalized == "binary_logloss" ||
      normalized == "binary:logistic") {
    constexpr double kEpsilon = 1e-12;
    double loss = 0.0;
    for (std::size_t i = 0; i < predictions.size(); ++i) {
      const double margin = predictions[i];
      const double probability =
          margin >= 0.0 ? 1.0 / (1.0 + std::exp(-margin))
                        : std::exp(margin) / (1.0 + std::exp(margin));
      const double clipped = std::clamp(probability, kEpsilon, 1.0 - kEpsilon);
      loss += -labels[i] * std::log(clipped) - (1.0 - labels[i]) * std::log(1.0 - clipped);
    }
    return predictions.empty() ? 0.0 : loss / predictions.size();
  }

  throw std::invalid_argument("unsupported objective for loss computation: " + objective_name);
}

}  // namespace

GradientBooster::GradientBooster(std::string objective,
                                 int iterations,
                                 double learning_rate,
                                 int max_depth,
                                 double alpha,
                                 double lambda_l2,
                                 std::size_t max_bins)
    : objective_name_(std::move(objective)),
      objective_(CreateObjectiveFunction(objective_name_)),
      iterations_(iterations),
      learning_rate_(learning_rate),
      max_depth_(max_depth),
      alpha_(alpha),
      lambda_l2_(lambda_l2),
      max_bins_(max_bins),
      hist_builder_(max_bins_) {
  if (iterations_ <= 0) {
    throw std::invalid_argument("iterations must be positive");
  }
  if (learning_rate_ <= 0.0) {
    throw std::invalid_argument("learning_rate must be positive");
  }
  if (max_depth_ < 0) {
    throw std::invalid_argument("max_depth must be non-negative");
  }
  if (lambda_l2_ < 0.0) {
    throw std::invalid_argument("lambda_l2 must be non-negative");
  }
}

void GradientBooster::Fit(const Pool& pool) {
  trees_.clear();
  loss_history_.clear();

  const HistMatrix hist = hist_builder_.Build(pool);
  std::vector<float> predictions(pool.num_rows(), 0.0F);
  const auto& labels = pool.labels();

  for (int iteration = 0; iteration < iterations_; ++iteration) {
    std::vector<float> gradients;
    std::vector<float> hessians;
    objective_->compute_gradients(predictions, labels, gradients, hessians);

    Tree tree;
    tree.Build(hist, gradients, hessians, alpha_, max_depth_, lambda_l2_);

    for (std::size_t row = 0; row < pool.num_rows(); ++row) {
      predictions[row] += learning_rate_ * tree.PredictBinnedRow(hist, row);
    }

    loss_history_.push_back(ComputeLoss(objective_name_, predictions, labels));
    trees_.push_back(std::move(tree));
  }
}

std::vector<float> GradientBooster::Predict(const Pool& pool) const {
  std::vector<float> predictions(pool.num_rows(), 0.0F);
  for (const Tree& tree : trees_) {
    for (std::size_t row = 0; row < pool.num_rows(); ++row) {
      predictions[row] += learning_rate_ * tree.PredictRow(pool, row);
    }
  }
  return predictions;
}

const std::vector<double>& GradientBooster::loss_history() const noexcept {
  return loss_history_;
}

std::size_t GradientBooster::num_trees() const noexcept { return trees_.size(); }

}  // namespace ctboost
