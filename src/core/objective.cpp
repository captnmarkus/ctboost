#include "ctboost/objective.hpp"

#include "objective_internal.hpp"

#include <stdexcept>
#include <string>

namespace ctboost {

std::unique_ptr<ObjectiveFunction> CreateObjectiveFunction(std::string_view name,
                                                           const ObjectiveConfig& config) {
  const std::string normalized = detail::NormalizeObjectiveName(name);

  if (auto objective = detail::CreateRegressionObjective(normalized, config)) {
    return objective;
  }
  if (auto objective = detail::CreateClassificationObjective(normalized, config)) {
    return objective;
  }
  if (auto objective = detail::CreateSurvivalObjective(normalized, config)) {
    return objective;
  }
  if (auto objective = detail::CreateRankingObjective(normalized, config)) {
    return objective;
  }
  throw std::invalid_argument("unknown objective function: " + std::string(name));
}

}  // namespace ctboost
