#include "metric_internal.hpp"

#include <cctype>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace ctboost::detail {

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

void ValidateNonNegativeMetricLabels(const std::vector<float>& labels, const char* metric_name) {
  for (const float label : labels) {
    if (!(label >= 0.0F) || !std::isfinite(label)) {
      throw std::invalid_argument(std::string(metric_name) +
                                  " requires finite non-negative labels");
    }
  }
}

std::vector<MetricSurvivalLabel> ParseSignedMetricSurvivalLabels(
    const std::vector<float>& labels,
    const char* metric_name) {
  std::vector<MetricSurvivalLabel> parsed(labels.size());
  bool saw_event = false;
  for (std::size_t index = 0; index < labels.size(); ++index) {
    const double signed_time = static_cast<double>(labels[index]);
    const double absolute_time = std::fabs(signed_time);
    if (!std::isfinite(signed_time) || absolute_time <= 0.0) {
      throw std::invalid_argument(std::string(metric_name) +
                                  " expects signed survival times with non-zero magnitude");
    }
    parsed[index] = MetricSurvivalLabel{absolute_time, signed_time > 0.0};
    saw_event = saw_event || parsed[index].event;
  }
  if (!saw_event) {
    throw std::invalid_argument(std::string(metric_name) + " requires at least one observed event");
  }
  return parsed;
}

void ValidateMulticlassMetricSizes(const std::vector<float>& preds,
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
    const RankingMetadataView* ranking) {
  if (num_classes != 1) {
    throw std::invalid_argument("ranking metrics expect num_classes equal to one");
  }
  ValidatePredictionLabelWeightSizes(preds, labels, weights);
  if (ranking == nullptr || ranking->group_ids == nullptr || ranking->group_ids->empty()) {
    throw std::invalid_argument("ranking metrics require group_id values");
  }
  if (ranking->group_ids->size() != labels.size()) {
    throw std::invalid_argument("group_id size must match the number of labels");
  }
  return *ranking->group_ids;
}

double ResolveMetricGroupWeight(const RankingMetadataView* ranking, std::size_t row) {
  if (ranking == nullptr || ranking->group_weights == nullptr || ranking->group_weights->empty()) {
    return 1.0;
  }
  return static_cast<double>((*ranking->group_weights)[row]);
}

bool SameMetricSubgroup(const RankingMetadataView* ranking, std::size_t left, std::size_t right) {
  if (ranking == nullptr || ranking->subgroup_ids == nullptr || ranking->subgroup_ids->empty()) {
    return false;
  }
  return (*ranking->subgroup_ids)[left] == (*ranking->subgroup_ids)[right];
}

double WeightSum(const std::vector<float>& weights) {
  return std::accumulate(
      weights.begin(), weights.end(), 0.0,
      [](double acc, float value) { return acc + static_cast<double>(value); });
}

float MetricSigmoid(float margin) {
  if (margin >= 0.0F) {
    const float exp_term = std::exp(-margin);
    return 1.0F / (1.0F + exp_term);
  }
  const float exp_term = std::exp(margin);
  return exp_term / (1.0F + exp_term);
}

int MetricLabelToClassIndex(float label, int num_classes) {
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

std::string NormalizeMetricName(std::string_view name) {
  std::string normalized;
  normalized.reserve(name.size());
  for (const char ch : name) {
    normalized.push_back(
        static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
  }
  return normalized;
}

}  // namespace ctboost::detail
