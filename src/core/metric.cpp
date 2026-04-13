#include "ctboost/metric.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
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

void ValidateNonNegativeLabels(const std::vector<float>& labels, const char* metric_name) {
  for (const float label : labels) {
    if (!(label >= 0.0F) || !std::isfinite(label)) {
      throw std::invalid_argument(std::string(metric_name) +
                                  " requires finite non-negative labels");
    }
  }
}

struct SurvivalLabel {
  double time{0.0};
  bool event{false};
};

std::vector<SurvivalLabel> ParseSignedSurvivalLabels(const std::vector<float>& labels,
                                                     const char* metric_name) {
  std::vector<SurvivalLabel> parsed(labels.size());
  bool saw_event = false;
  for (std::size_t index = 0; index < labels.size(); ++index) {
    const double signed_time = static_cast<double>(labels[index]);
    const double absolute_time = std::fabs(signed_time);
    if (!std::isfinite(signed_time) || absolute_time <= 0.0) {
      throw std::invalid_argument(std::string(metric_name) +
                                  " expects signed survival times with non-zero magnitude");
    }
    parsed[index] = SurvivalLabel{absolute_time, signed_time > 0.0};
    saw_event = saw_event || parsed[index].event;
  }
  if (!saw_event) {
    throw std::invalid_argument(std::string(metric_name) + " requires at least one observed event");
  }
  return parsed;
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

const std::vector<std::int64_t>& ValidateRankingMetricInputs(
    const std::vector<float>& preds,
    const std::vector<float>& labels,
    const std::vector<float>& weights,
    int num_classes,
    const std::vector<std::int64_t>* group_ids) {
  if (num_classes != 1) {
    throw std::invalid_argument("ranking metrics expect num_classes equal to one");
  }
  ValidatePredictionLabelWeightSizes(preds, labels, weights);
  if (group_ids == nullptr || group_ids->empty()) {
    throw std::invalid_argument("ranking metrics require group_id values");
  }
  if (group_ids->size() != labels.size()) {
    throw std::invalid_argument("group_id size must match the number of labels");
  }
  return *group_ids;
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
                  int,
                  const std::vector<std::int64_t>*) const override {
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
                  int,
                  const std::vector<std::int64_t>*) const override {
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
                  int,
                  const std::vector<std::int64_t>*) const override {
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
                  int,
                  const std::vector<std::int64_t>*) const override {
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

class PoissonMetric final : public MetricFunction {
 public:
  double Evaluate(const std::vector<float>& preds,
                  const std::vector<float>& labels,
                  const std::vector<float>& weights,
                  int,
                  const std::vector<std::int64_t>*) const override {
    ValidatePredictionLabelWeightSizes(preds, labels, weights);
    ValidateNonNegativeLabels(labels, "poisson metric");
    double loss_sum = 0.0;
    double weight_sum = 0.0;
    for (std::size_t i = 0; i < preds.size(); ++i) {
      const double label = static_cast<double>(labels[i]);
      const double mean = std::exp(static_cast<double>(preds[i]));
      const double sample_weight = static_cast<double>(weights[i]);
      loss_sum += sample_weight * (mean - label * static_cast<double>(preds[i]) + std::lgamma(label + 1.0));
      weight_sum += sample_weight;
    }
    return weight_sum <= 0.0 ? 0.0 : loss_sum / weight_sum;
  }

  bool HigherIsBetter() const noexcept override { return false; }
};

class TweedieMetric final : public MetricFunction {
 public:
  explicit TweedieMetric(double variance_power) : variance_power_(variance_power) {
    if (!(variance_power_ > 1.0 && variance_power_ < 2.0)) {
      throw std::invalid_argument("tweedie_variance_power must be in the open interval (1, 2)");
    }
  }

  double Evaluate(const std::vector<float>& preds,
                  const std::vector<float>& labels,
                  const std::vector<float>& weights,
                  int,
                  const std::vector<std::int64_t>*) const override {
    ValidatePredictionLabelWeightSizes(preds, labels, weights);
    ValidateNonNegativeLabels(labels, "tweedie metric");
    const double one_minus_power = 1.0 - variance_power_;
    const double two_minus_power = 2.0 - variance_power_;
    double loss_sum = 0.0;
    double weight_sum = 0.0;
    for (std::size_t i = 0; i < preds.size(); ++i) {
      const double label = static_cast<double>(labels[i]);
      const double prediction = static_cast<double>(preds[i]);
      const double sample_weight = static_cast<double>(weights[i]);
      const double part_one =
          label <= 0.0 ? 0.0 : label * std::exp(one_minus_power * prediction) / one_minus_power;
      const double part_two = std::exp(two_minus_power * prediction) / two_minus_power;
      loss_sum += sample_weight * (-part_one + part_two);
      weight_sum += sample_weight;
    }
    return weight_sum <= 0.0 ? 0.0 : loss_sum / weight_sum;
  }

  bool HigherIsBetter() const noexcept override { return false; }

 private:
  double variance_power_{1.5};
};

class CoxMetric final : public MetricFunction {
 public:
  double Evaluate(const std::vector<float>& preds,
                  const std::vector<float>& labels,
                  const std::vector<float>& weights,
                  int,
                  const std::vector<std::int64_t>*) const override {
    ValidatePredictionLabelWeightSizes(preds, labels, weights);
    const std::vector<SurvivalLabel> survival = ParseSignedSurvivalLabels(labels, "cox metric");

    std::vector<std::size_t> order(preds.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](std::size_t lhs, std::size_t rhs) {
      if (survival[lhs].time == survival[rhs].time) {
        return survival[lhs].event && !survival[rhs].event;
      }
      return survival[lhs].time > survival[rhs].time;
    });

    std::vector<double> exp_preds(preds.size(), 0.0);
    for (std::size_t index = 0; index < preds.size(); ++index) {
      exp_preds[index] = std::exp(static_cast<double>(preds[index]));
    }

    double loss_sum = 0.0;
    double weight_sum = 0.0;
    double risk_sum = 0.0;
    std::size_t position = 0;
    while (position < order.size()) {
      const double group_time = survival[order[position]].time;
      std::size_t group_end = position;
      double event_weight = 0.0;
      while (group_end < order.size() && survival[order[group_end]].time == group_time) {
        risk_sum += exp_preds[order[group_end]];
        if (survival[order[group_end]].event) {
          event_weight += static_cast<double>(weights[order[group_end]]);
        }
        ++group_end;
      }
      if (event_weight > 0.0) {
        const double log_risk = std::log(std::max(risk_sum, 1e-12));
        for (std::size_t group_position = position; group_position < group_end; ++group_position) {
          const std::size_t row = order[group_position];
          if (!survival[row].event) {
            continue;
          }
          const double sample_weight = static_cast<double>(weights[row]);
          loss_sum += sample_weight * (log_risk - static_cast<double>(preds[row]));
          weight_sum += sample_weight;
        }
      }
      position = group_end;
    }
    return weight_sum <= 0.0 ? 0.0 : loss_sum / weight_sum;
  }

  bool HigherIsBetter() const noexcept override { return false; }
};

class SurvivalExponentialMetric final : public MetricFunction {
 public:
  double Evaluate(const std::vector<float>& preds,
                  const std::vector<float>& labels,
                  const std::vector<float>& weights,
                  int,
                  const std::vector<std::int64_t>*) const override {
    ValidatePredictionLabelWeightSizes(preds, labels, weights);
    const std::vector<SurvivalLabel> survival =
        ParseSignedSurvivalLabels(labels, "survival exponential metric");

    double loss_sum = 0.0;
    double weight_sum = 0.0;
    for (std::size_t index = 0; index < preds.size(); ++index) {
      const double hazard = std::exp(static_cast<double>(preds[index]));
      const double event = survival[index].event ? 1.0 : 0.0;
      const double sample_weight = static_cast<double>(weights[index]);
      loss_sum += sample_weight * (hazard * survival[index].time - event * static_cast<double>(preds[index]));
      weight_sum += sample_weight;
    }
    return weight_sum <= 0.0 ? 0.0 : loss_sum / weight_sum;
  }

  bool HigherIsBetter() const noexcept override { return false; }
};

class BinaryLoglossMetric final : public MetricFunction {
 public:
  double Evaluate(const std::vector<float>& preds,
                  const std::vector<float>& labels,
                  const std::vector<float>& weights,
                  int,
                  const std::vector<std::int64_t>*) const override {
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
                  int num_classes,
                  const std::vector<std::int64_t>*) const override {
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
                  int num_classes,
                  const std::vector<std::int64_t>*) const override {
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
                  int,
                  const std::vector<std::int64_t>*) const override {
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
                  int,
                  const std::vector<std::int64_t>*) const override {
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
                  int num_classes,
                  const std::vector<std::int64_t>* group_ids) const override {
    const double precision = precision_.Evaluate(preds, labels, weights, num_classes, group_ids);
    const double recall = recall_.Evaluate(preds, labels, weights, num_classes, group_ids);
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
                  int,
                  const std::vector<std::int64_t>*) const override {
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

class PairLogitMetric final : public MetricFunction {
 public:
  double Evaluate(const std::vector<float>& preds,
                  const std::vector<float>& labels,
                  const std::vector<float>& weights,
                  int num_classes,
                  const std::vector<std::int64_t>* group_ids) const override {
    const auto& resolved_group_ids =
        ValidateRankingMetricInputs(preds, labels, weights, num_classes, group_ids);
    std::unordered_map<std::int64_t, std::vector<std::size_t>> group_rows;
    group_rows.reserve(resolved_group_ids.size());
    for (std::size_t row = 0; row < resolved_group_ids.size(); ++row) {
      group_rows[resolved_group_ids[row]].push_back(row);
    }

    double loss_sum = 0.0;
    double weight_sum = 0.0;
    for (const auto& entry : group_rows) {
      const auto& rows = entry.second;
      for (std::size_t left = 0; left < rows.size(); ++left) {
        for (std::size_t right = left + 1; right < rows.size(); ++right) {
          const std::size_t i = rows[left];
          const std::size_t j = rows[right];
          if (labels[i] == labels[j]) {
            continue;
          }

          const std::size_t positive = labels[i] > labels[j] ? i : j;
          const std::size_t negative = labels[i] > labels[j] ? j : i;
          const double pair_weight = static_cast<double>(weights[positive]) * weights[negative];
          const double margin = static_cast<double>(preds[positive]) - preds[negative];
          loss_sum += pair_weight * std::log1p(std::exp(-margin));
          weight_sum += pair_weight;
        }
      }
    }

    return weight_sum <= 0.0 ? 0.0 : loss_sum / weight_sum;
  }

  bool HigherIsBetter() const noexcept override { return false; }
};

class NDCGMetric final : public MetricFunction {
 public:
  double Evaluate(const std::vector<float>& preds,
                  const std::vector<float>& labels,
                  const std::vector<float>& weights,
                  int num_classes,
                  const std::vector<std::int64_t>* group_ids) const override {
    const auto& resolved_group_ids =
        ValidateRankingMetricInputs(preds, labels, weights, num_classes, group_ids);
    std::unordered_map<std::int64_t, std::vector<std::size_t>> group_rows;
    group_rows.reserve(resolved_group_ids.size());
    for (std::size_t row = 0; row < resolved_group_ids.size(); ++row) {
      group_rows[resolved_group_ids[row]].push_back(row);
    }

    double ndcg_sum = 0.0;
    double group_weight_sum = 0.0;
    for (const auto& entry : group_rows) {
      const auto& rows = entry.second;
      if (rows.size() <= 1) {
        continue;
      }

      auto gain = [&](std::size_t row_index) {
        return static_cast<double>(weights[row_index]) *
               (std::pow(2.0, static_cast<double>(labels[row_index])) - 1.0);
      };

      std::vector<std::size_t> prediction_order = rows;
      std::sort(prediction_order.begin(), prediction_order.end(), [&](std::size_t lhs, std::size_t rhs) {
        if (preds[lhs] == preds[rhs]) {
          return lhs < rhs;
        }
        return preds[lhs] > preds[rhs];
      });

      std::vector<std::size_t> ideal_order = rows;
      std::sort(ideal_order.begin(), ideal_order.end(), [&](std::size_t lhs, std::size_t rhs) {
        if (labels[lhs] == labels[rhs]) {
          return lhs < rhs;
        }
        return labels[lhs] > labels[rhs];
      });

      double dcg = 0.0;
      double ideal_dcg = 0.0;
      double group_weight = 0.0;
      for (std::size_t rank = 0; rank < rows.size(); ++rank) {
        const double discount = 1.0 / std::log2(static_cast<double>(rank) + 2.0);
        dcg += gain(prediction_order[rank]) * discount;
        ideal_dcg += gain(ideal_order[rank]) * discount;
        group_weight += weights[rows[rank]];
      }
      if (ideal_dcg <= 0.0 || group_weight <= 0.0) {
        continue;
      }

      ndcg_sum += group_weight * (dcg / ideal_dcg);
      group_weight_sum += group_weight;
    }

    return group_weight_sum <= 0.0 ? 0.0 : ndcg_sum / group_weight_sum;
  }

  bool HigherIsBetter() const noexcept override { return true; }
};

class MAPMetric final : public MetricFunction {
 public:
  double Evaluate(const std::vector<float>& preds,
                  const std::vector<float>& labels,
                  const std::vector<float>& weights,
                  int num_classes,
                  const std::vector<std::int64_t>* group_ids) const override {
    const auto& resolved_group_ids =
        ValidateRankingMetricInputs(preds, labels, weights, num_classes, group_ids);
    std::unordered_map<std::int64_t, std::vector<std::size_t>> group_rows;
    group_rows.reserve(resolved_group_ids.size());
    for (std::size_t row = 0; row < resolved_group_ids.size(); ++row) {
      group_rows[resolved_group_ids[row]].push_back(row);
    }

    double map_sum = 0.0;
    double group_weight_sum = 0.0;
    for (const auto& entry : group_rows) {
      std::vector<std::size_t> rows = entry.second;
      std::sort(rows.begin(), rows.end(), [&](std::size_t lhs, std::size_t rhs) {
        if (preds[lhs] == preds[rhs]) {
          return lhs < rhs;
        }
        return preds[lhs] > preds[rhs];
      });

      double precision_sum = 0.0;
      double hit_count = 0.0;
      double relevant_weight = 0.0;
      double group_weight = 0.0;
      for (std::size_t rank = 0; rank < rows.size(); ++rank) {
        const std::size_t row = rows[rank];
        const double sample_weight = static_cast<double>(weights[row]);
        group_weight += sample_weight;
        if (labels[row] > 0.0F) {
          hit_count += 1.0;
          precision_sum += hit_count / static_cast<double>(rank + 1);
          relevant_weight += sample_weight;
        }
      }
      if (relevant_weight <= 0.0 || group_weight <= 0.0) {
        continue;
      }
      map_sum += group_weight * (precision_sum / hit_count);
      group_weight_sum += group_weight;
    }
    return group_weight_sum <= 0.0 ? 0.0 : map_sum / group_weight_sum;
  }

  bool HigherIsBetter() const noexcept override { return true; }
};

class MRRMetric final : public MetricFunction {
 public:
  double Evaluate(const std::vector<float>& preds,
                  const std::vector<float>& labels,
                  const std::vector<float>& weights,
                  int num_classes,
                  const std::vector<std::int64_t>* group_ids) const override {
    const auto& resolved_group_ids =
        ValidateRankingMetricInputs(preds, labels, weights, num_classes, group_ids);
    std::unordered_map<std::int64_t, std::vector<std::size_t>> group_rows;
    group_rows.reserve(resolved_group_ids.size());
    for (std::size_t row = 0; row < resolved_group_ids.size(); ++row) {
      group_rows[resolved_group_ids[row]].push_back(row);
    }

    double reciprocal_rank_sum = 0.0;
    double group_weight_sum = 0.0;
    for (const auto& entry : group_rows) {
      std::vector<std::size_t> rows = entry.second;
      std::sort(rows.begin(), rows.end(), [&](std::size_t lhs, std::size_t rhs) {
        if (preds[lhs] == preds[rhs]) {
          return lhs < rhs;
        }
        return preds[lhs] > preds[rhs];
      });

      double group_weight = 0.0;
      for (const std::size_t row : rows) {
        group_weight += static_cast<double>(weights[row]);
      }
      if (group_weight <= 0.0) {
        continue;
      }

      for (std::size_t rank = 0; rank < rows.size(); ++rank) {
        if (labels[rows[rank]] > 0.0F) {
          reciprocal_rank_sum += group_weight / static_cast<double>(rank + 1);
          group_weight_sum += group_weight;
          break;
        }
      }
    }

    return group_weight_sum <= 0.0 ? 0.0 : reciprocal_rank_sum / group_weight_sum;
  }

  bool HigherIsBetter() const noexcept override { return true; }
};

class ConcordanceIndexMetric final : public MetricFunction {
 public:
  double Evaluate(const std::vector<float>& preds,
                  const std::vector<float>& labels,
                  const std::vector<float>& weights,
                  int,
                  const std::vector<std::int64_t>*) const override {
    ValidatePredictionLabelWeightSizes(preds, labels, weights);
    const std::vector<SurvivalLabel> survival = ParseSignedSurvivalLabels(labels, "cindex");

    double concordant = 0.0;
    double comparable = 0.0;
    for (std::size_t i = 0; i < preds.size(); ++i) {
      if (!survival[i].event) {
        continue;
      }
      for (std::size_t j = 0; j < preds.size(); ++j) {
        if (i == j) {
          continue;
        }
        if (survival[j].time <= survival[i].time) {
          continue;
        }
        const double pair_weight =
            static_cast<double>(weights[i]) * static_cast<double>(weights[j]);
        if (pair_weight <= 0.0) {
          continue;
        }
        comparable += pair_weight;
        if (preds[i] > preds[j]) {
          concordant += pair_weight;
        } else if (preds[i] == preds[j]) {
          concordant += 0.5 * pair_weight;
        }
      }
    }
    return comparable <= 0.0 ? 0.0 : concordant / comparable;
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
  if (normalized == "poisson" || normalized == "poissonregression") {
    return std::make_unique<PoissonMetric>();
  }
  if (normalized == "tweedie" || normalized == "tweedieloss" ||
      normalized == "reg:tweedie") {
    return std::make_unique<TweedieMetric>(config.tweedie_variance_power);
  }
  if (normalized == "cox" || normalized == "coxph" || normalized == "survival:cox") {
    return std::make_unique<CoxMetric>();
  }
  if (normalized == "survivalexponential" || normalized == "survival_exp" ||
      normalized == "survival:exponential") {
    return std::make_unique<SurvivalExponentialMetric>();
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
  if (normalized == "pairlogit" || normalized == "pairwise" ||
      normalized == "ranknet") {
    return std::make_unique<PairLogitMetric>();
  }
  if (normalized == "ndcg") {
    return std::make_unique<NDCGMetric>();
  }
  if (normalized == "map" || normalized == "map@all") {
    return std::make_unique<MAPMetric>();
  }
  if (normalized == "mrr" || normalized == "mrr@all") {
    return std::make_unique<MRRMetric>();
  }
  if (normalized == "cindex" || normalized == "concordance_index") {
    return std::make_unique<ConcordanceIndexMetric>();
  }

  throw std::invalid_argument("unknown metric function: " + std::string(name));
}

std::unique_ptr<MetricFunction> CreateMetricFunctionForObjective(
    std::string_view objective_name,
    const ObjectiveConfig& config) {
  return CreateMetricFunction(objective_name, config);
}

}  // namespace ctboost
