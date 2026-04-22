#include "metric_internal.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace {

class PrecisionMetric final : public ctboost::MetricFunction {
 public:
  double Evaluate(const std::vector<float>& preds,
                  const std::vector<float>& labels,
                  const std::vector<float>& weights,
                  int,
                  const ctboost::RankingMetadataView*) const override {
    ctboost::detail::ValidatePredictionLabelWeightSizes(preds, labels, weights);
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

class RecallMetric final : public ctboost::MetricFunction {
 public:
  double Evaluate(const std::vector<float>& preds,
                  const std::vector<float>& labels,
                  const std::vector<float>& weights,
                  int,
                  const ctboost::RankingMetadataView*) const override {
    ctboost::detail::ValidatePredictionLabelWeightSizes(preds, labels, weights);
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

class F1Metric final : public ctboost::MetricFunction {
 public:
  double Evaluate(const std::vector<float>& preds,
                  const std::vector<float>& labels,
                  const std::vector<float>& weights,
                  int num_classes,
                  const ctboost::RankingMetadataView* ranking) const override {
    const double precision = precision_.Evaluate(preds, labels, weights, num_classes, ranking);
    const double recall = recall_.Evaluate(preds, labels, weights, num_classes, ranking);
    const double denominator = precision + recall;
    return denominator <= 0.0 ? 0.0 : 2.0 * precision * recall / denominator;
  }

  bool HigherIsBetter() const noexcept override { return true; }

 private:
  PrecisionMetric precision_;
  RecallMetric recall_;
};

class AUCMetric final : public ctboost::MetricFunction {
 public:
  double Evaluate(const std::vector<float>& preds,
                  const std::vector<float>& labels,
                  const std::vector<float>& weights,
                  int,
                  const ctboost::RankingMetadataView*) const override {
    ctboost::detail::ValidatePredictionLabelWeightSizes(preds, labels, weights);
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

namespace ctboost::detail {

std::unique_ptr<MetricFunction> CreateClassificationMetric(std::string_view normalized,
                                                           const ObjectiveConfig&) {
  if (auto metric = CreateClassificationScoreMetric(normalized)) {
    return metric;
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
  return nullptr;
}

}  // namespace ctboost::detail
