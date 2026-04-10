#include "ctboost/booster.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ctboost/cuda_backend.hpp"

namespace ctboost {
namespace {

std::string NormalizeToken(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return value;
}

bool IsSquaredErrorObjective(const std::string& normalized_objective) {
  return normalized_objective == "rmse" || normalized_objective == "squarederror" ||
         normalized_objective == "squared_error";
}

bool IsBinaryObjective(const std::string& normalized_objective) {
  return normalized_objective == "logloss" || normalized_objective == "binary_logloss" ||
         normalized_objective == "binary:logistic";
}

bool IsMulticlassObjective(const std::string& normalized_objective) {
  return normalized_objective == "multiclass" || normalized_objective == "softmax" ||
         normalized_objective == "softmaxloss";
}

bool SameCategoricalFeatures(const Pool& lhs, const Pool& rhs) {
  const auto& lhs_features = lhs.cat_features();
  const auto& rhs_features = rhs.cat_features();
  return lhs_features.size() == rhs_features.size() &&
         std::is_permutation(lhs_features.begin(), lhs_features.end(), rhs_features.begin());
}

int LabelToClassIndex(float label, int num_classes) {
  const float rounded = std::round(label);
  if (std::fabs(label - rounded) > 1e-6F) {
    throw std::invalid_argument("multiclass labels must be integer encoded");
  }

  const int class_index = static_cast<int>(rounded);
  if (class_index < 0 || class_index >= num_classes) {
    throw std::invalid_argument("multiclass label is out of range");
  }

  return class_index;
}

double ComputeLoss(const std::string& objective_name,
                   const std::vector<float>& predictions,
                   const std::vector<float>& labels,
                   int num_classes) {
  const std::string normalized = NormalizeToken(objective_name);

  if (IsSquaredErrorObjective(normalized)) {
    if (predictions.size() != labels.size()) {
      throw std::invalid_argument("predictions and labels must have the same size");
    }
    double sum_squared_error = 0.0;
    for (std::size_t i = 0; i < predictions.size(); ++i) {
      const double residual = static_cast<double>(predictions[i]) - labels[i];
      sum_squared_error += residual * residual;
    }
    return predictions.empty() ? 0.0 : sum_squared_error / predictions.size();
  }

  if (IsBinaryObjective(normalized)) {
    if (predictions.size() != labels.size()) {
      throw std::invalid_argument("predictions and labels must have the same size");
    }
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

  if (IsMulticlassObjective(normalized)) {
    if (num_classes <= 2) {
      throw std::invalid_argument("multiclass loss requires num_classes greater than two");
    }
    if (predictions.size() != labels.size() * static_cast<std::size_t>(num_classes)) {
      throw std::invalid_argument(
          "multiclass predictions must have num_rows * num_classes elements");
    }

    constexpr double kEpsilon = 1e-12;
    double loss = 0.0;
    for (std::size_t row = 0; row < labels.size(); ++row) {
      const std::size_t row_offset = row * static_cast<std::size_t>(num_classes);
      double max_logit = static_cast<double>(predictions[row_offset]);
      for (int class_index = 1; class_index < num_classes; ++class_index) {
        max_logit = std::max(max_logit, static_cast<double>(predictions[row_offset + class_index]));
      }

      double exp_sum = 0.0;
      for (int class_index = 0; class_index < num_classes; ++class_index) {
        exp_sum += std::exp(static_cast<double>(predictions[row_offset + class_index]) - max_logit);
      }

      const int target_class = LabelToClassIndex(labels[row], num_classes);
      const double target_probability =
          std::exp(static_cast<double>(predictions[row_offset + target_class]) - max_logit) / exp_sum;
      loss -= std::log(std::clamp(target_probability, kEpsilon, 1.0));
    }
    return labels.empty() ? 0.0 : loss / labels.size();
  }

  throw std::invalid_argument("unsupported objective for loss computation: " + objective_name);
}

std::string NormalizeTaskType(std::string task_type) {
  return NormalizeToken(std::move(task_type));
}

void UpdatePredictions(const Tree& tree,
                       const HistMatrix& hist,
                       double learning_rate,
                       int prediction_dimension,
                       int class_index,
                       std::vector<float>& predictions) {
  if (prediction_dimension == 1) {
    for (std::size_t row = 0; row < hist.num_rows; ++row) {
      predictions[row] += learning_rate * tree.PredictBinnedRow(hist, row);
    }
    return;
  }

  for (std::size_t row = 0; row < hist.num_rows; ++row) {
    const std::size_t offset = row * static_cast<std::size_t>(prediction_dimension) + class_index;
    predictions[offset] += learning_rate * tree.PredictBinnedRow(hist, row);
  }
}

void UpdatePredictions(const Tree& tree,
                       const Pool& pool,
                       double learning_rate,
                       int prediction_dimension,
                       int class_index,
                       std::vector<float>& predictions) {
  if (prediction_dimension == 1) {
    for (std::size_t row = 0; row < pool.num_rows(); ++row) {
      predictions[row] += learning_rate * tree.PredictRow(pool, row);
    }
    return;
  }

  for (std::size_t row = 0; row < pool.num_rows(); ++row) {
    const std::size_t offset = row * static_cast<std::size_t>(prediction_dimension) + class_index;
    predictions[offset] += learning_rate * tree.PredictRow(pool, row);
  }
}

void AccumulateFeatureImportances(const Tree& tree, std::vector<double>& feature_importance_sums) {
  const auto& tree_feature_importances = tree.feature_importances();
  for (std::size_t feature = 0; feature < tree_feature_importances.size(); ++feature) {
    feature_importance_sums[feature] += tree_feature_importances[feature];
  }
}

void RecomputeFeatureImportances(const std::vector<Tree>& trees,
                                 std::size_t num_features,
                                 std::vector<double>& feature_importance_sums) {
  feature_importance_sums.assign(num_features, 0.0);
  for (const Tree& tree : trees) {
    AccumulateFeatureImportances(tree, feature_importance_sums);
  }
}

}  // namespace

GradientBooster::GradientBooster(std::string objective,
                                 int iterations,
                                 double learning_rate,
                                 int max_depth,
                                 double alpha,
                                 double lambda_l2,
                                 int num_classes,
                                 std::size_t max_bins,
                                 std::string task_type,
                                 std::string devices)
    : objective_name_(std::move(objective)),
      objective_(CreateObjectiveFunction(objective_name_)),
      iterations_(iterations),
      learning_rate_(learning_rate),
      max_depth_(max_depth),
      alpha_(alpha),
      lambda_l2_(lambda_l2),
      num_classes_(num_classes),
      max_bins_(max_bins),
      devices_(std::move(devices)),
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
  if (num_classes_ <= 0) {
    throw std::invalid_argument("num_classes must be positive");
  }

  const std::string normalized_objective = NormalizeToken(objective_name_);
  if (IsMulticlassObjective(normalized_objective)) {
    if (num_classes_ <= 2) {
      throw std::invalid_argument("multiclass objective requires num_classes greater than two");
    }
    prediction_dimension_ = num_classes_;
  } else if (IsSquaredErrorObjective(normalized_objective)) {
    if (num_classes_ != 1) {
      throw std::invalid_argument("regression objectives require num_classes equal to one");
    }
    prediction_dimension_ = 1;
  } else if (IsBinaryObjective(normalized_objective)) {
    if (num_classes_ != 1 && num_classes_ != 2) {
      throw std::invalid_argument("binary objectives require num_classes equal to one or two");
    }
    prediction_dimension_ = 1;
  }

  const std::string normalized_task_type = NormalizeTaskType(std::move(task_type));
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
}

void GradientBooster::Fit(const Pool& pool,
                          const Pool* eval_pool,
                          int early_stopping_rounds) {
  if (early_stopping_rounds < 0) {
    early_stopping_rounds = 0;
  }
  if (early_stopping_rounds > 0 && eval_pool == nullptr) {
    throw std::invalid_argument("early_stopping_rounds requires eval_pool");
  }
  if (eval_pool != nullptr) {
    if (eval_pool->num_cols() != pool.num_cols()) {
      throw std::invalid_argument(
          "eval_pool must have the same number of columns as the training pool");
    }
    if (!SameCategoricalFeatures(pool, *eval_pool)) {
      throw std::invalid_argument(
          "eval_pool categorical feature indices must match the training pool");
    }
  }

  trees_.clear();
  loss_history_.clear();
  best_iteration_ = -1;
  best_loss_ = 0.0;

  const HistMatrix hist = hist_builder_.Build(pool);
  const auto& labels = pool.labels();
  std::vector<float> predictions(
      pool.num_rows() * static_cast<std::size_t>(prediction_dimension_), 0.0F);
  feature_importance_sums_.assign(pool.num_cols(), 0.0);
  int completed_iterations = 0;
  bool early_stopped = false;

  const std::vector<float>* eval_labels = nullptr;
  std::vector<float> eval_predictions;
  if (eval_pool != nullptr) {
    eval_labels = &eval_pool->labels();
    eval_predictions.assign(
        eval_pool->num_rows() * static_cast<std::size_t>(prediction_dimension_), 0.0F);
    best_loss_ = std::numeric_limits<double>::infinity();
  }

  for (int iteration = 0; iteration < iterations_; ++iteration) {
    std::vector<float> gradients;
    std::vector<float> hessians;
    objective_->compute_gradients(predictions, labels, gradients, hessians, num_classes_);

    if (prediction_dimension_ == 1) {
      Tree tree;
      tree.Build(hist, gradients, hessians, alpha_, max_depth_, lambda_l2_, use_gpu_);

      UpdatePredictions(tree, hist, learning_rate_, prediction_dimension_, 0, predictions);
      if (eval_pool != nullptr) {
        UpdatePredictions(tree, *eval_pool, learning_rate_, prediction_dimension_, 0, eval_predictions);
      }
      AccumulateFeatureImportances(tree, feature_importance_sums_);

      trees_.push_back(std::move(tree));
    } else {
      std::vector<float> class_gradients(pool.num_rows(), 0.0F);
      std::vector<float> class_hessians(pool.num_rows(), 0.0F);

      for (int class_index = 0; class_index < prediction_dimension_; ++class_index) {
        for (std::size_t row = 0; row < pool.num_rows(); ++row) {
          const std::size_t offset =
              row * static_cast<std::size_t>(prediction_dimension_) + class_index;
          class_gradients[row] = gradients[offset];
          class_hessians[row] = hessians[offset];
        }

        Tree tree;
        tree.Build(hist,
                   class_gradients,
                   class_hessians,
                   alpha_,
                   max_depth_,
                   lambda_l2_,
                   use_gpu_);

        UpdatePredictions(
            tree, hist, learning_rate_, prediction_dimension_, class_index, predictions);
        if (eval_pool != nullptr) {
          UpdatePredictions(
              tree,
              *eval_pool,
              learning_rate_,
              prediction_dimension_,
              class_index,
              eval_predictions);
        }
        AccumulateFeatureImportances(tree, feature_importance_sums_);

        trees_.push_back(std::move(tree));
      }
    }

    loss_history_.push_back(ComputeLoss(objective_name_, predictions, labels, num_classes_));
    completed_iterations = iteration + 1;
    if (eval_pool == nullptr) {
      continue;
    }

    const double eval_loss =
        ComputeLoss(objective_name_, eval_predictions, *eval_labels, num_classes_);
    if (best_iteration_ < 0 || eval_loss < best_loss_) {
      best_iteration_ = iteration;
      best_loss_ = eval_loss;
      continue;
    }

    if (early_stopping_rounds > 0 &&
        iteration - best_iteration_ >= early_stopping_rounds) {
      early_stopped = true;
      break;
    }
  }

  if (eval_pool == nullptr) {
    best_iteration_ = completed_iterations > 0 ? completed_iterations - 1 : -1;
    return;
  }

  if (!early_stopped || best_iteration_ < 0) {
    return;
  }

  const std::size_t retained_iterations =
      static_cast<std::size_t>(best_iteration_ + 1);
  trees_.resize(retained_iterations * static_cast<std::size_t>(prediction_dimension_));
  loss_history_.resize(retained_iterations);
  RecomputeFeatureImportances(trees_, pool.num_cols(), feature_importance_sums_);
}

std::vector<float> GradientBooster::Predict(const Pool& pool) const {
  std::vector<float> predictions(
      pool.num_rows() * static_cast<std::size_t>(prediction_dimension_), 0.0F);
  if (prediction_dimension_ == 1) {
    for (const Tree& tree : trees_) {
      for (std::size_t row = 0; row < pool.num_rows(); ++row) {
        predictions[row] += learning_rate_ * tree.PredictRow(pool, row);
      }
    }
    return predictions;
  }

  for (std::size_t tree_index = 0; tree_index < trees_.size(); ++tree_index) {
    const int class_index = static_cast<int>(
        tree_index % static_cast<std::size_t>(prediction_dimension_));
    for (std::size_t row = 0; row < pool.num_rows(); ++row) {
      predictions[row * static_cast<std::size_t>(prediction_dimension_) + class_index] +=
          learning_rate_ * trees_[tree_index].PredictRow(pool, row);
    }
  }
  return predictions;
}

const std::vector<double>& GradientBooster::loss_history() const noexcept {
  return loss_history_;
}

std::size_t GradientBooster::num_trees() const noexcept { return trees_.size(); }

int GradientBooster::num_classes() const noexcept { return num_classes_; }

int GradientBooster::prediction_dimension() const noexcept { return prediction_dimension_; }

int GradientBooster::best_iteration() const noexcept { return best_iteration_; }

std::vector<float> GradientBooster::get_feature_importances() const {
  std::vector<float> importances(feature_importance_sums_.size(), 0.0F);
  const double total_importance = std::accumulate(
      feature_importance_sums_.begin(), feature_importance_sums_.end(), 0.0);
  if (total_importance <= 0.0) {
    return importances;
  }

  const double scale = 100.0 / total_importance;
  for (std::size_t feature = 0; feature < feature_importance_sums_.size(); ++feature) {
    importances[feature] = static_cast<float>(feature_importance_sums_[feature] * scale);
  }
  return importances;
}

}  // namespace ctboost
