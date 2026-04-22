#include "ctboost/metric.hpp"

#include "metric_internal.hpp"

#include <stdexcept>
#include <string>

namespace ctboost {

std::unique_ptr<MetricFunction> CreateMetricFunction(std::string_view name,
                                                     const ObjectiveConfig& config) {
  const std::string normalized = detail::NormalizeMetricName(name);

  if (auto metric = detail::CreateRegressionMetric(normalized, config)) {
    return metric;
  }
  if (auto metric = detail::CreateSurvivalMetric(normalized, config)) {
    return metric;
  }
  if (auto metric = detail::CreateClassificationMetric(normalized, config)) {
    return metric;
  }
  if (auto metric = detail::CreateRankingMetric(normalized, config)) {
    return metric;
  }
  throw std::invalid_argument("unknown metric function: " + std::string(name));
}

std::unique_ptr<MetricFunction> CreateMetricFunctionForObjective(
    std::string_view objective_name,
    const ObjectiveConfig& config) {
  return CreateMetricFunction(objective_name, config);
}

}  // namespace ctboost
