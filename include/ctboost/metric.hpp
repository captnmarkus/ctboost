#pragma once

#include <memory>
#include <string_view>
#include <vector>

#include "ctboost/objective.hpp"

namespace ctboost {

class MetricFunction {
 public:
  virtual ~MetricFunction() = default;

  virtual double Evaluate(const std::vector<float>& preds,
                          const std::vector<float>& labels,
                          const std::vector<float>& weights,
                          int num_classes = 1) const = 0;
  virtual bool HigherIsBetter() const noexcept = 0;
};

std::unique_ptr<MetricFunction> CreateMetricFunction(std::string_view name,
                                                     const ObjectiveConfig& config = {});
std::unique_ptr<MetricFunction> CreateMetricFunctionForObjective(
    std::string_view objective_name,
    const ObjectiveConfig& config = {});

}  // namespace ctboost
