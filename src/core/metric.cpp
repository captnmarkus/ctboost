#include "ctboost/metric.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ctboost {
namespace {

std::string NormalizeName(std::string_view name) {
  std::string normalized;
  normalized.reserve(name.size());
  for (const char ch : name) {
    normalized.push_back(
        static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
  }
  return normalized;
}

void ValidatePredictionLabelWeightSizes(const std::vector<float>& preds,
                                        const std::vector<float>& labels,
                                        const std::vector<float>& weights) {
  if (labels.size() != weights.size()) {
    throw std::invalid_argument("labels and weights must have the same size");
  }
  if (preds.size() != labels.size()) {
    throw std::invalid_argument("predictions and labels must have the same size");
  }
}

void ValidateMulticlassSizes(const std::vector<float>& preds,
                             const std::vector<float>& labels,
                             const std::vector<float>& weights,
                             int num_classes) {
  if (labels.size() != weights.size()) {
    throw std::invalid_argument("labels and weights must have the same size");
  }
  if (num_classes <= 2) {
    throw std::invalid_argument("multiclass metrics require num_classes greater than two");
  }
  if (preds.size() != labels.size() * static_cast<std::size_t>(num_classes)) {
    throw std::invalid_argument(
        "multiclass predictions must have num_rows * num_classes elements");
  }
}

double WeightSum(const std::vector<float>& weights) {
  return std::accumulate(
      weights.begin(), weights.end(), 0.0,
      [](double acc, float value) { return acc + static_cast<double>(value); });
}

float Sigmoid(float margin) {
  if (margin >= 0.0F) {
    const float exp_term = std::exp(-margin);
    return 1.0F / (1.0F + exp_term);
  }
  const float exp_term = std::exp(margin);
  return exp_term / (1.0F + exp_term);
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

class RMSEMetric final : public MetricFunction {
 public:
  double Evaluate(const std::vector<float>& preds,
                  const std::vector<float>& labels,
                  const std::vector<float>& weights,
                  int) const override {
    ValidatePredictionLabelWeightSizes(preds, labels, weights);
    double squared_error_sum = 0.0;
    double weight_sum = 0.0;
    for (std::size_t i = 0; i < preds.size(); ++i) {
      const double residual = static_cast<double>(preds[i]) - labels[i];
      const double sample_weight = weights[i];
      squared_error_sum += sample_weight * residual * residual;
      weight_sum += sample_weight;
    }
    return weight_sum <= 0.0 ? 0.0 : std::sqrt(squared_error_sum / weight_sum);
  }

  bool HigherIsBetter() const noexcept override { return false; }
};

class MAEMetric final : public MetricFunction {
 public:
  double Evaluate(const std::vector<float>& preds,
                  const std::vector<float>& labels,
                  const std::vector<float>& weights,
                  int) const override {
    ValidatePredictionLabelWeightSizes(preds, labels, weights);
    double absolute_error_sum = 0.0;
    double weight_sum = 0.0;
    for (std::size_t i = 0; i < preds.size(); ++i) {
      const double sample_weight = weights[i];
      absolute_error_sum += sample_weight * std::fabs(static_cast<double>(preds[i]) - labels[i]);
      weight_sum += sample_weight;
    }
    return weight_sum <= 0.0 ? 0.0 : absolute_error_sum / weight_sum;
  }

  bool HigherIsBetter() const noexcept override { return false; }
};

class HuberMetric final : public MetricFunction {
 public:
  explicit HuberMetric(double delta) : delta_(delta) {
    if (delta_ <= 0.0) {
      throw std::invalid_argument("huber_delta must be positive");
    }
  }

  double Evaluate(const std::vector<float>& preds,
                  const std::vector<float>& labels,
                  const std::vector<float>& weights,
                  int) const override {
    ValidatePredictionLabelWeightSizes(preds, labels, weights);
    double loss_sum = 0.0;
    double weight_sum = 0.0;
    for (std::size_t i = 0; i < preds.size(); ++i) {
      const double residual = static_cast<double>(preds[i]) - labels[i];
      const double absolute_residual = std::fabs(residual);
      const double sample_weight = weights[i];
      if (absolute_residual <= delta_) {
        loss_sum += sample_weight * 0.5 * residual * residual;
      } else {
        loss_sum += sample_weight * delta_ * (absolute_residual - 0.5 * delta_);
      }
      weight_sum += sample_weight;
    }
    return weight_sum <= 0.0 ? 0.0 : loss_sum / weight_sum;
  }

  bool HigherIsBetter() const noexcept override { return false; }

 private:
  double delta_{1.0};
};

class QuantileMetric final : public MetricFunction {
 public:
  explicit QuantileMetric(double alpha) : alpha_(alpha) {
    if (!(alpha_ > 0.0 && alpha_ < 1.0)) {
      throw std::invalid_argument("quantile_alpha must be in the open interval (0, 1)");
    }
  }

  double Evaluate(const std::vector<float>& preds,
                  const std::vector<float>& labels,
                  const std::vector<float>& weights,
                  int) const override {
    ValidatePredictionLabelWeightSizes(preds, labels, weights);
    double loss_sum = 0.0;
    double weight_sum = 0.0;
    for (std::size_t i = 0; i < preds.size(); ++i) {
      const double residual = static_cast<double>(labels[i]) - preds[i];
      const double sample_weight = weights[i];
      loss_sum += sample_weight *
                  (residual >= 0.0 ? alpha_ * residual : (1.0 - alpha_) * (-residual));
      weight_sum += sample_weight;
    }
    return weight_sum <= 0.0 ? 0.0 : loss_sum / weight_sum;
  }

  bool HigherIsBetter() const noexcept override { return false; }

 private:
  double alpha_{0.5};
};

class BinaryLoglossMetric final : public MetricFunction {
 public:
  double Evaluate(const std::vector<float>& preds,
                  const std::vector<float>& labels,
                  const std::vector<float>& weights,
                  int) const override {
    ValidatePredictionLabelWeightSizes(preds, labels, weights);
    constexpr double kEpsilon = 1e-12;
    double loss_sum = 0.0;
    double weight_sum = 0.0;
    for (std::size_t i = 0; i < preds.size(); ++i) {
      const double probability = Sigmoid(preds[i]);
      const double clipped = std::clamp(probability, kEpsilon, 1.0 - kEpsilon);
      const double sample_weight = weights[i];
      loss_sum += sample_weight *
                  (-labels[i] * std::log(clipped) - (1.0 - labels[i]) * std::log(1.0 - clipped));
      weight_sum += sample_weight;
    }
    return weight_sum <= 0.0 ? 0.0 : loss_sum / weight_sum;
  }

  bool HigherIsBetter() const noexcept override { return false; }
};

class MulticlassLoglossMetric final : public MetricFunction {
 public:
  double Evaluate(const std::vector<float>& preds,
                  const std::vector<float>& labels,
                  const std::vector<float>& weights,
                  int num_classes) const override {
    ValidateMulticlassSizes(preds, labels, weights, num_classes);
    constexpr double kEpsilon = 1e-12;
    double loss_sum = 0.0;
    double weight_sum = 0.0;
    for (std::size_t row = 0; row < labels.size(); ++row) {
      const std::size_t row_offset = row * static_cast<std::size_t>(num_classes);
      double max_logit = static_cast<double>(preds[row_offset]);
      for (int class_index = 1; class_index < num_classes; ++class_index) {
        max_logit = std::max(max_logit, static_cast<double>(preds[row_offset + class_index]));
      }

      double exp_sum = 0.0;
      for (int class_index = 0; class_index < num_classes; ++class_index) {
        exp_sum += std::exp(static_cast<double>(preds[row_offset + class_index]) - max_logit);
      }

      const int target_class = LabelToClassIndex(labels[row], num_classes);
      const double target_probability =
          std::exp(static_cast<double>(preds[row_offset + target_class]) - max_logit) / exp_sum;
      const double sample_weight = weights[row];
      loss_sum -= sample_weight * std::log(std::clamp(target_probability, kEpsilon, 1.0));
      weight_sum += sample_weight;
    }
    return weight_sum <= 0.0 ? 0.0 : loss_sum / weight_sum;
  }

  bool HigherIsBetter() const noexcept override { return false; }
};

class AccuracyMetric final : public MetricFunction {
 public:
  double Evaluate(const std::vector<float>& preds,
                  const std::vector<float>& labels,
                  const std::vector<float>& weights,
                  int num_classes) const override {
    if (num_classes > 2) {
      ValidateMulticlassSizes(preds, labels, weights, num_classes);
      double correct_weight = 0.0;
      const double weight_sum = WeightSum(weights);
      for (std::size_t row = 0; row < labels.size(); ++row) {
        const std::size_t row_offset = row * static_cast<std::size_t>(num_classes);
        int best_class = 0;
        float best_score = preds[row_offset];
        for (int class_index = 1; class_index < num_classes; ++class_index) {
          const float score = preds[row_offset + class_index];
          if (score > best_score) {
            best_score = score;
            best_class = class_index;
          }
        }
        if (best_class == LabelToClassIndex(labels[row], num_classes)) {
          correct_weight += weights[row];
        }
      }
      return weight_sum <= 0.0 ? 0.0 : correct_weight / weight_sum;
    }

    ValidatePredictionLabelWeightSizes(preds, labels, weights);
    double correct_weight = 0.0;
    const double weight_sum = WeightSum(weights);
    for (std::size_t i = 0; i < preds.size(); ++i) {
      const int predicted = preds[i] >= 0.0F ? 1 : 0;
      const int label = static_cast<int>(std::round(labels[i]));
      if (predicted == label) {
        correct_weight += weights[i];
      }
    }
    return weight_sum <= 0.0 ? 0.0 : correct_weight / weight_sum;
  }

  bool HigherIsBetter() const noexcept override { return true; }
};

class PrecisionMetric final : public MetricFunction {
 public:
  double Evaluate(const std::vector<float>& preds,
                  const std::vector<float>& labels,
                  const std::vector<float>& weights,
                  int) const override {
    ValidatePredictionLabelWeightSizes(preds, labels, weights);
    double true_positive = 0.0;
    double false_positive = 0.0;
    for (std::size_t i = 0; i < preds.size(); ++i) {
      const bool predicted_positive = preds[i] >= 0.0F;
      const bool label_positive = std::round(labels[i]) == 1.0F;
      if (predicted_positive && label_positive) {
        true_positive += weights[i];
      } else if (predicted_positive) {
        false_positive += weights[i];
      }
    }
    const double denominator = true_positive + false_positive;
    return denominator <= 0.0 ? 0.0 : true_positive / denominator;
  }

  bool HigherIsBetter() const noexcept override { return true; }
};

class RecallMetric final : public MetricFunction {
 public:
  double Evaluate(const std::vector<float>& preds,
                  const std::vector<float>& labels,
                  const std::vector<float>& weights,
                  int) const override {
    ValidatePredictionLabelWeightSizes(preds, labels, weights);
    double true_positive = 0.0;
    double false_negative = 0.0;
    for (std::size_t i = 0; i < preds.size(); ++i) {
      const bool predicted_positive = preds[i] >= 0.0F;
      const bool label_positive = std::round(labels[i]) == 1.0F;
      if (label_positive && predicted_positive) {
        true_positive += weights[i];
      } else if (label_positive) {
        false_negative += weights[i];
      }
    }
    const double denominator = true_positive + false_negative;
    return denominator <= 0.0 ? 0.0 : true_positive / denominator;
  }

  bool HigherIsBetter() const noexcept override { return true; }
};

class F1Metric final : public MetricFunction {
 public:
  double Evaluate(const std::vector<float>& preds,
                  const std::vector<float>& labels,
                  const std::vector<float>& weights,
                  int num_classes) const override {
    const double precision = precision_.Evaluate(preds, labels, weights, num_classes);
    const double recall = recall_.Evaluate(preds, labels, weights, num_classes);
    const double denominator = precision + recall;
    return denominator <= 0.0 ? 0.0 : 2.0 * precision * recall / denominator;
  }

  bool HigherIsBetter() const noexcept override { return true; }

 private:
  PrecisionMetric precision_;
  RecallMetric recall_;
};

class AUCMetric final : public MetricFunction {
 public:
  double Evaluate(const std::vector<float>& preds,
                  const std::vector<float>& labels,
                  const std::vector<float>& weights,
                  int) const override {
    ValidatePredictionLabelWeightSizes(preds, labels, weights);
    std::vector<std::size_t> order(preds.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](std::size_t lhs, std::size_t rhs) {
      if (preds[lhs] == preds[rhs]) {
        return lhs < rhs;
      }
      return preds[lhs] < preds[rhs];
    });

    double total_positive_weight = 0.0;
    double total_negative_weight = 0.0;
    for (std::size_t i = 0; i < labels.size(); ++i) {
      if (std::round(labels[i]) == 1.0F) {
        total_positive_weight += weights[i];
      } else {
        total_negative_weight += weights[i];
      }
    }
    if (total_positive_weight <= 0.0 || total_negative_weight <= 0.0) {
      return 0.0;
    }

    double cumulative_negative_weight = 0.0;
    double auc_numerator = 0.0;
    std::size_t begin = 0;
    while (begin < order.size()) {
      std::size_t end = begin + 1;
      while (end < order.size() && preds[order[begin]] == preds[order[end]]) {
        ++end;
      }

      double positive_weight = 0.0;
      double negative_weight = 0.0;
      for (std::size_t index = begin; index < end; ++index) {
        const std::size_t row = order[index];
        if (std::round(labels[row]) == 1.0F) {
          positive_weight += weights[row];
        } else {
          negative_weight += weights[row];
        }
      }

      auc_numerator += positive_weight * cumulative_negative_weight;
      auc_numerator += 0.5 * positive_weight * negative_weight;
      cumulative_negative_weight += negative_weight;
      begin = end;
    }

    return auc_numerator / (total_positive_weight * total_negative_weight);
  }

  bool HigherIsBetter() const noexcept override { return true; }
};

}  // namespace

std::unique_ptr<MetricFunction> CreateMetricFunction(std::string_view name,
                                                     const ObjectiveConfig& config) {
  const std::string normalized = NormalizeName(name);

  if (normalized == "rmse" || normalized == "squarederror" ||
      normalized == "squared_error") {
    return std::make_unique<RMSEMetric>();
  }
  if (normalized == "mae" || normalized == "l1" || normalized == "absoluteerror" ||
      normalized == "absolute_error") {
    return std::make_unique<MAEMetric>();
  }
  if (normalized == "huber" || normalized == "huberloss") {
    return std::make_unique<HuberMetric>(config.huber_delta);
  }
  if (normalized == "quantile" || normalized == "quantileloss") {
    return std::make_unique<QuantileMetric>(config.quantile_alpha);
  }
  if (normalized == "logloss" || normalized == "binary_logloss" ||
      normalized == "binary:logistic") {
    return std::make_unique<BinaryLoglossMetric>();
  }
  if (normalized == "multiclass" || normalized == "softmax" ||
      normalized == "softmaxloss" || normalized == "multiclassloss") {
    return std::make_unique<MulticlassLoglossMetric>();
  }
  if (normalized == "accuracy") {
    return std::make_unique<AccuracyMetric>();
  }
  if (normalized == "precision") {
    return std::make_unique<PrecisionMetric>();
  }
  if (normalized == "recall") {
    return std::make_unique<RecallMetric>();
  }
  if (normalized == "f1") {
    return std::make_unique<F1Metric>();
  }
  if (normalized == "auc" || normalized == "roc_auc") {
    return std::make_unique<AUCMetric>();
  }

  throw std::invalid_argument("unknown metric function: " + std::string(name));
}

std::unique_ptr<MetricFunction> CreateMetricFunctionForObjective(
    std::string_view objective_name,
    const ObjectiveConfig& config) {
  return CreateMetricFunction(objective_name, config);
}

}  // namespace ctboost
