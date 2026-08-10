#include "booster_internal.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace ctboost {
namespace {

double PositiveTotalWeight(const std::vector<float>& weights) {
  const double total = std::accumulate(
      weights.begin(), weights.end(), 0.0, [](double sum, float weight) {
        if (!std::isfinite(weight) || weight < 0.0F) {
          throw std::invalid_argument("sample weights must be finite and non-negative");
        }
        return sum + static_cast<double>(weight);
      });
  if (!(total > 0.0)) {
    throw std::invalid_argument("training pool must have a positive total sample weight");
  }
  return total;
}

double WeightedMean(const std::vector<float>& labels,
                    const std::vector<float>& weights,
                    bool require_positive,
                    bool require_non_negative) {
  const double total_weight = PositiveTotalWeight(weights);
  double weighted_sum = 0.0;
  for (std::size_t index = 0; index < labels.size(); ++index) {
    const double label = static_cast<double>(labels[index]);
    if (!std::isfinite(label) || (require_positive && !(label > 0.0)) ||
        (require_non_negative && !(label >= 0.0))) {
      throw std::invalid_argument("objective labels are outside the supported range");
    }
    weighted_sum += static_cast<double>(weights[index]) * label;
  }
  return weighted_sum / total_weight;
}

double WeightedQuantile(const std::vector<float>& labels,
                        const std::vector<float>& weights,
                        double alpha) {
  const double target = alpha * PositiveTotalWeight(weights);
  std::vector<std::pair<double, double>> values;
  values.reserve(labels.size());
  for (std::size_t index = 0; index < labels.size(); ++index) {
    const double label = static_cast<double>(labels[index]);
    if (!std::isfinite(label)) {
      throw std::invalid_argument("objective labels must be finite");
    }
    if (weights[index] > 0.0F) {
      values.emplace_back(label, static_cast<double>(weights[index]));
    }
  }
  std::stable_sort(values.begin(), values.end(), [](const auto& left, const auto& right) {
    return left.first < right.first;
  });
  double cumulative = 0.0;
  for (const auto& value : values) {
    cumulative += value.second;
    if (cumulative >= target) {
      return value.first;
    }
  }
  return values.back().first;
}

double WeightedHuberLocation(const std::vector<float>& labels,
                             const std::vector<float>& weights,
                             double delta) {
  (void)PositiveTotalWeight(weights);
  double lower = std::numeric_limits<double>::infinity();
  double upper = -std::numeric_limits<double>::infinity();
  for (const float value : labels) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument("objective labels must be finite");
    }
    lower = std::min(lower, static_cast<double>(value));
    upper = std::max(upper, static_cast<double>(value));
  }
  for (int iteration = 0; iteration < 100; ++iteration) {
    const double midpoint = lower + (upper - lower) * 0.5;
    double gradient = 0.0;
    for (std::size_t index = 0; index < labels.size(); ++index) {
      const double residual = midpoint - static_cast<double>(labels[index]);
      gradient += static_cast<double>(weights[index]) *
                  std::max(-delta, std::min(delta, residual));
    }
    if (gradient > 0.0) {
      upper = midpoint;
    } else {
      lower = midpoint;
    }
  }
  return lower + (upper - lower) * 0.5;
}

std::vector<double> ResolveAverageBaseScore(const std::string& objective,
                                            const ObjectiveConfig& config,
                                            const std::vector<float>& labels,
                                            const std::vector<float>& weights,
                                            int prediction_dimension) {
  if (labels.size() != weights.size() || labels.empty()) {
    throw std::invalid_argument("labels and weights must be non-empty and have matching sizes");
  }
  if (booster_detail::IsSquaredErrorObjective(objective)) {
    return {WeightedMean(labels, weights, false, false)};
  }
  if (booster_detail::IsAbsoluteErrorObjective(objective)) {
    return {WeightedQuantile(labels, weights, 0.5)};
  }
  if (booster_detail::IsHuberObjective(objective)) {
    return {WeightedHuberLocation(labels, weights, config.huber_delta)};
  }
  if (booster_detail::IsQuantileObjective(objective)) {
    return {WeightedQuantile(labels, weights, config.quantile_alpha)};
  }
  if (booster_detail::IsPoissonObjective(objective) ||
      booster_detail::IsTweedieObjective(objective) ||
      booster_detail::IsGammaObjective(objective)) {
    const bool positive = booster_detail::IsGammaObjective(objective);
    const double mean = WeightedMean(labels, weights, positive, !positive);
    // Poisson and Tweedie permit an all-zero target. Their exact constant
    // optimum is a negative-infinite log mean, so use a finite response-scale
    // floor instead of falling back to margin zero (mean one).
    constexpr double kMeanFloor = 1e-12;
    return {std::log(std::max(mean, kMeanFloor))};
  }
  if (booster_detail::IsBinaryObjective(objective)) {
    const double prevalence = WeightedMean(labels, weights, false, true);
    if (prevalence > 1.0) {
      throw std::invalid_argument("binary labels must be in [0, 1]");
    }
    constexpr double kProbabilityFloor = 1e-6;
    const double probability =
        std::max(kProbabilityFloor, std::min(1.0 - kProbabilityFloor, prevalence));
    return {std::log(probability / (1.0 - probability))};
  }
  if (booster_detail::IsMulticlassObjective(objective)) {
    const double total_weight = PositiveTotalWeight(weights);
    std::vector<double> probabilities(static_cast<std::size_t>(prediction_dimension), 0.0);
    for (std::size_t row = 0; row < labels.size(); ++row) {
      const int class_index =
          booster_detail::LabelToClassIndex(labels[row], prediction_dimension);
      probabilities[static_cast<std::size_t>(class_index)] +=
          static_cast<double>(weights[row]) / total_weight;
    }
    constexpr double kProbabilityFloor = 1e-12;
    double normalization = 0.0;
    for (double& probability : probabilities) {
      probability = std::max(probability, kProbabilityFloor);
      normalization += probability;
    }
    std::vector<double> logits(probabilities.size(), 0.0);
    double mean_logit = 0.0;
    for (std::size_t index = 0; index < probabilities.size(); ++index) {
      logits[index] = std::log(probabilities[index] / normalization);
      mean_logit += logits[index] / static_cast<double>(probabilities.size());
    }
    for (double& logit : logits) {
      logit -= mean_logit;
    }
    return logits;
  }
  return std::vector<double>(static_cast<std::size_t>(prediction_dimension), 0.0);
}

}  // namespace

void GradientBooster::InitializeBaseScore(const Pool& pool,
                                          bool allow_average_initialization) {
  if (!configured_base_score_.empty()) {
    if (configured_base_score_.size() != 1U &&
        configured_base_score_.size() != static_cast<std::size_t>(prediction_dimension_)) {
      throw std::invalid_argument(
          "base_score must contain one raw margin or one margin per prediction dimension");
    }
    base_score_.assign(static_cast<std::size_t>(prediction_dimension_),
                       configured_base_score_.front());
    if (configured_base_score_.size() == static_cast<std::size_t>(prediction_dimension_)) {
      base_score_ = configured_base_score_;
    }
    return;
  }
  base_score_.assign(static_cast<std::size_t>(prediction_dimension_), 0.0);
  const std::string normalized_objective = booster_detail::NormalizeToken(objective_name_);
  if (!boost_from_average_ || !allow_average_initialization || pool.has_baseline() ||
      booster_detail::IsRankingObjective(normalized_objective) ||
      booster_detail::IsSurvivalObjective(normalized_objective)) {
    return;
  }

  const std::vector<float>* labels = &pool.labels();
  const std::vector<float>* weights = &pool.weights();
  booster_detail::DistributedMetricInputs gathered;
  if (distributed_world_size_ > 1) {
    DistributedCoordinator coordinator;
    coordinator.world_size = distributed_world_size_;
    coordinator.rank = distributed_rank_;
    coordinator.root = distributed_root_;
    coordinator.run_id = distributed_run_id_;
    coordinator.timeout_seconds = distributed_timeout_;
    coordinator.tree_index = num_iterations_trained();
    booster_detail::DistributedMetricInputs local;
    local.labels = *labels;
    local.weights = *weights;
    gathered = booster_detail::AllGatherDistributedMetricInputs(
        &coordinator, "base_score_inputs", local);
    labels = &gathered.labels;
    weights = &gathered.weights;
  }
  base_score_ = ResolveAverageBaseScore(normalized_objective,
                                        objective_config_,
                                        *labels,
                                        *weights,
                                        prediction_dimension_);
}

}  // namespace ctboost
