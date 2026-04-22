#pragma once

#include "ctboost/metric.hpp"

#include <memory>
#include <string>
#include <string_view>

namespace ctboost::detail {

void ValidatePredictionLabelWeightSizes(const std::vector<float>& preds,
                                        const std::vector<float>& labels,
                                        const std::vector<float>& weights);
void ValidateNonNegativeMetricLabels(const std::vector<float>& labels, const char* metric_name);

struct MetricSurvivalLabel {
  double time{0.0};
  bool event{false};
};

std::vector<MetricSurvivalLabel> ParseSignedMetricSurvivalLabels(
    const std::vector<float>& labels,
    const char* metric_name);
void ValidateMulticlassMetricSizes(const std::vector<float>& preds,
                                   const std::vector<float>& labels,
                                   const std::vector<float>& weights,
                                   int num_classes);
const std::vector<std::int64_t>& ValidateRankingMetricInputs(
    const std::vector<float>& preds,
    const std::vector<float>& labels,
    const std::vector<float>& weights,
    int num_classes,
    const RankingMetadataView* ranking);
double ResolveMetricGroupWeight(const RankingMetadataView* ranking, std::size_t row);
bool SameMetricSubgroup(const RankingMetadataView* ranking, std::size_t left, std::size_t right);
double WeightSum(const std::vector<float>& weights);
float MetricSigmoid(float margin);
int MetricLabelToClassIndex(float label, int num_classes);
std::string NormalizeMetricName(std::string_view name);

std::unique_ptr<MetricFunction> CreateRegressionMetric(std::string_view normalized,
                                                       const ObjectiveConfig& config);
std::unique_ptr<MetricFunction> CreateSurvivalMetric(std::string_view normalized,
                                                     const ObjectiveConfig& config);
std::unique_ptr<MetricFunction> CreateClassificationScoreMetric(std::string_view normalized);
std::unique_ptr<MetricFunction> CreateClassificationMetric(std::string_view normalized,
                                                           const ObjectiveConfig& config);
std::unique_ptr<MetricFunction> CreateRankingMetric(std::string_view normalized,
                                                    const ObjectiveConfig& config);

}  // namespace ctboost::detail
