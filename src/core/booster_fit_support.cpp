#include "booster_fit_internal.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace ctboost::booster_detail {

void LogFitMemorySnapshot(const TrainingProfiler& profiler,
                          const char* stage,
                          const Pool& train_pool,
                          const Pool* eval_pool,
                          const FitWorkspace& workspace) {
  profiler.LogFitMemory(stage,
                        train_pool.dense_feature_bytes(),
                        eval_pool == nullptr ? 0U : eval_pool->dense_feature_bytes(),
                        workspace.train_hist_bytes(),
                        workspace.eval_hist_bytes(),
                        workspace.gpu_workspace_bytes());
}

void ValidateFitInputs(const Pool& pool,
                       const Pool* eval_pool,
                       int early_stopping_rounds,
                       const std::vector<int>& monotone_constraints,
                       const std::vector<std::vector<int>>& interaction_constraints,
                       const std::vector<double>& feature_weights,
                       const std::vector<double>& first_feature_use_penalties,
                       int prediction_dimension) {
  if (early_stopping_rounds > 0 && eval_pool == nullptr) {
    throw std::invalid_argument("early_stopping_rounds requires eval_pool");
  }
  if (eval_pool != nullptr) {
    if (eval_pool->num_cols() != pool.num_cols()) {
      throw std::invalid_argument("eval_pool must have the same number of columns as the training pool");
    }
    if (!SameCategoricalFeatures(pool, *eval_pool)) {
      throw std::invalid_argument("eval_pool categorical feature indices must match the training pool");
    }
  }
  if (!monotone_constraints.empty()) {
    if (monotone_constraints.size() != pool.num_cols()) {
      throw std::invalid_argument("monotone_constraints must have one entry per feature when provided");
    }
    for (const int feature_id : pool.cat_features()) {
      if (feature_id >= 0 && static_cast<std::size_t>(feature_id) < monotone_constraints.size() &&
          monotone_constraints[static_cast<std::size_t>(feature_id)] != 0) {
        throw std::invalid_argument("monotone_constraints can only be applied to numeric features");
      }
    }
    if (prediction_dimension != 1) {
      throw std::invalid_argument("monotone_constraints are only supported for single-output objectives");
    }
  }
  for (const auto& group : interaction_constraints) {
    for (const int feature_id : group) {
      if (feature_id < 0 || static_cast<std::size_t>(feature_id) >= pool.num_cols()) {
        throw std::invalid_argument("interaction_constraints feature index is out of bounds");
      }
    }
  }
  if (!feature_weights.empty() && feature_weights.size() != pool.num_cols()) {
    throw std::invalid_argument("feature_weights must have one entry per feature when provided");
  }
  if (!feature_weights.empty() &&
      std::none_of(feature_weights.begin(), feature_weights.end(), [](double value) {
        return value > 0.0;
      })) {
    throw std::invalid_argument("feature_weights must leave at least one feature with positive weight");
  }
  if (!first_feature_use_penalties.empty() && first_feature_use_penalties.size() != pool.num_cols()) {
    throw std::invalid_argument("first_feature_use_penalties must have one entry per feature when provided");
  }
}

void ValidateUniformQueryWeights(const Pool& pool, const char* consumer_name) {
  if (!pool.has_group_ids()) {
    return;
  }
  const auto& group_ids = pool.group_ids();
  const auto& weights = pool.weights();
  std::unordered_map<std::int64_t, float> weight_by_group;
  weight_by_group.reserve(group_ids.size());
  for (std::size_t row = 0; row < group_ids.size(); ++row) {
    const auto [entry, inserted] = weight_by_group.emplace(group_ids[row], weights[row]);
    if (inserted) {
      continue;
    }
    const double reference = static_cast<double>(entry->second);
    const double current = static_cast<double>(weights[row]);
    const double tolerance = 1e-7 + 1e-6 * std::max(std::fabs(reference), std::fabs(current));
    if (std::fabs(reference - current) > tolerance) {
      throw std::invalid_argument(
          std::string(consumer_name) +
          " requires sample weights to be uniform within each group_id; "
          "use one query weight per group (or group_weight) instead");
    }
  }
}

}  // namespace ctboost::booster_detail
