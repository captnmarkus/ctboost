#include "metric_internal.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

class BinaryLoglossMetric final : public ctboost::MetricFunction {
 public:
  double Evaluate(const std::vector<float>& preds,
                  const std::vector<float>& labels,
                  const std::vector<float>& weights,
                  int,
                  const ctboost::RankingMetadataView*) const override {
    ctboost::detail::ValidatePredictionLabelWeightSizes(preds, labels, weights);
    constexpr double kEpsilon = 1e-12;
    double loss_sum = 0.0;
    double weight_sum = 0.0;
    for (std::size_t i = 0; i < preds.size(); ++i) {
      const double probability = ctboost::detail::MetricSigmoid(preds[i]);
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

class MulticlassLoglossMetric final : public ctboost::MetricFunction {
 public:
  double Evaluate(const std::vector<float>& preds,
                  const std::vector<float>& labels,
                  const std::vector<float>& weights,
                  int num_classes,
                  const ctboost::RankingMetadataView*) const override {
    ctboost::detail::ValidateMulticlassMetricSizes(preds, labels, weights, num_classes);
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

      const int target_class = ctboost::detail::MetricLabelToClassIndex(labels[row], num_classes);
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

class AccuracyMetric final : public ctboost::MetricFunction {
 public:
  double Evaluate(const std::vector<float>& preds,
                  const std::vector<float>& labels,
                  const std::vector<float>& weights,
                  int num_classes,
                  const ctboost::RankingMetadataView*) const override {
    if (num_classes > 2) {
      ctboost::detail::ValidateMulticlassMetricSizes(preds, labels, weights, num_classes);
      double correct_weight = 0.0;
      const double weight_sum = ctboost::detail::WeightSum(weights);
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
        if (best_class == ctboost::detail::MetricLabelToClassIndex(labels[row], num_classes)) {
          correct_weight += weights[row];
        }
      }
      return weight_sum <= 0.0 ? 0.0 : correct_weight / weight_sum;
    }

    ctboost::detail::ValidatePredictionLabelWeightSizes(preds, labels, weights);
    double correct_weight = 0.0;
    const double weight_sum = ctboost::detail::WeightSum(weights);
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

class BalancedAccuracyMetric final : public ctboost::MetricFunction {
 public:
  double Evaluate(const std::vector<float>& preds,
                  const std::vector<float>& labels,
                  const std::vector<float>& weights,
                  int num_classes,
                  const ctboost::RankingMetadataView*) const override {
    if (num_classes > 2) {
      ctboost::detail::ValidateMulticlassMetricSizes(preds, labels, weights, num_classes);
      std::vector<double> class_weight_sum(static_cast<std::size_t>(num_classes), 0.0);
      std::vector<double> class_correct_weight(static_cast<std::size_t>(num_classes), 0.0);
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
        const int target_class = ctboost::detail::MetricLabelToClassIndex(labels[row], num_classes);
        const double sample_weight = static_cast<double>(weights[row]);
        class_weight_sum[static_cast<std::size_t>(target_class)] += sample_weight;
        if (best_class == target_class) {
          class_correct_weight[static_cast<std::size_t>(target_class)] += sample_weight;
        }
      }

      double recall_sum = 0.0;
      int present_class_count = 0;
      for (int class_index = 0; class_index < num_classes; ++class_index) {
        const double total_weight = class_weight_sum[static_cast<std::size_t>(class_index)];
        if (total_weight <= 0.0) {
          continue;
        }
        recall_sum +=
            class_correct_weight[static_cast<std::size_t>(class_index)] / total_weight;
        ++present_class_count;
      }
      return present_class_count == 0 ? 0.0 : recall_sum / present_class_count;
    }

    ctboost::detail::ValidatePredictionLabelWeightSizes(preds, labels, weights);
    double class_weight_sum[2] = {0.0, 0.0};
    double class_correct_weight[2] = {0.0, 0.0};
    for (std::size_t index = 0; index < preds.size(); ++index) {
      const int label = static_cast<int>(std::round(labels[index]));
      if (label < 0 || label > 1) {
        throw std::invalid_argument(
            "balanced accuracy expects binary labels encoded as 0/1");
      }
      const int predicted = preds[index] >= 0.0F ? 1 : 0;
      const double sample_weight = static_cast<double>(weights[index]);
      class_weight_sum[label] += sample_weight;
      if (predicted == label) {
        class_correct_weight[label] += sample_weight;
      }
    }

    double recall_sum = 0.0;
    int present_class_count = 0;
    for (int class_index = 0; class_index < 2; ++class_index) {
      if (class_weight_sum[class_index] <= 0.0) {
        continue;
      }
      recall_sum += class_correct_weight[class_index] / class_weight_sum[class_index];
      ++present_class_count;
    }
    return present_class_count == 0 ? 0.0 : recall_sum / present_class_count;
  }

  bool HigherIsBetter() const noexcept override { return true; }
};

}  // namespace

namespace ctboost::detail {

std::unique_ptr<MetricFunction> CreateClassificationScoreMetric(std::string_view normalized) {
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
  if (normalized == "balancedaccuracy" || normalized == "balanced_accuracy" ||
      normalized == "balanced-accuracy") {
    return std::make_unique<BalancedAccuracyMetric>();
  }
  return nullptr;
}

}  // namespace ctboost::detail
