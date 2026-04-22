#include "booster_internal.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ctboost::booster_detail {

std::string NormalizeToken(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return value;
}

bool IsSquaredErrorObjective(const std::string& normalized_objective) {
  return normalized_objective == "rmse" || normalized_objective == "squarederror" ||
         normalized_objective == "squared_error";
}

bool IsAbsoluteErrorObjective(const std::string& normalized_objective) {
  return normalized_objective == "mae" || normalized_objective == "l1" ||
         normalized_objective == "absoluteerror" || normalized_objective == "absolute_error";
}

bool IsHuberObjective(const std::string& normalized_objective) {
  return normalized_objective == "huber" || normalized_objective == "huberloss";
}

bool IsQuantileObjective(const std::string& normalized_objective) {
  return normalized_objective == "quantile" || normalized_objective == "quantileloss";
}

bool IsPoissonObjective(const std::string& normalized_objective) {
  return normalized_objective == "poisson" || normalized_objective == "poissonregression";
}

bool IsTweedieObjective(const std::string& normalized_objective) {
  return normalized_objective == "tweedie" || normalized_objective == "tweedieloss" ||
         normalized_objective == "reg:tweedie";
}

bool IsSurvivalObjective(const std::string& normalized_objective) {
  return normalized_objective == "cox" || normalized_objective == "coxph" ||
         normalized_objective == "survival:cox" || normalized_objective == "survivalexponential" ||
         normalized_objective == "survival_exp" ||
         normalized_objective == "survival:exponential";
}

bool IsBinaryObjective(const std::string& normalized_objective) {
  return normalized_objective == "logloss" || normalized_objective == "binary_logloss" ||
         normalized_objective == "binary:logistic";
}

bool IsMulticlassObjective(const std::string& normalized_objective) {
  return normalized_objective == "multiclass" || normalized_objective == "softmax" ||
         normalized_objective == "softmaxloss";
}

bool IsRankingObjective(const std::string& normalized_objective) {
  return normalized_objective == "pairlogit" || normalized_objective == "pairwise" ||
         normalized_objective == "ranknet";
}

bool IsRegressionObjective(const std::string& normalized_objective) {
  return IsSquaredErrorObjective(normalized_objective) ||
         IsAbsoluteErrorObjective(normalized_objective) ||
         IsHuberObjective(normalized_objective) || IsQuantileObjective(normalized_objective) ||
         IsPoissonObjective(normalized_objective) || IsTweedieObjective(normalized_objective) ||
         IsSurvivalObjective(normalized_objective);
}

std::uint64_t NormalizeRngState(std::uint64_t seed) {
  return seed == 0 ? 0x9E3779B97F4A7C15ULL : seed;
}

std::uint64_t NextRandom(std::uint64_t& state) {
  state += 0x9E3779B97F4A7C15ULL;
  std::uint64_t z = state;
  z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31U);
}

std::size_t UniformIndex(std::uint64_t& state, std::size_t limit) {
  if (limit <= 1) {
    return 0;
  }
  const std::uint64_t bound = static_cast<std::uint64_t>(limit);
  const std::uint64_t threshold = static_cast<std::uint64_t>(-bound) % bound;
  while (true) {
    const std::uint64_t value = NextRandom(state);
    if (value >= threshold) {
      return static_cast<std::size_t>(value % bound);
    }
  }
}

bool SameCategoricalFeatures(const Pool& lhs, const Pool& rhs) {
  const auto& lhs_features = lhs.cat_features();
  const auto& rhs_features = rhs.cat_features();
  return lhs_features.size() == rhs_features.size() &&
         std::is_permutation(lhs_features.begin(), lhs_features.end(), rhs_features.begin());
}

void AddPoolBaselineToPredictions(const Pool& pool,
                                  int prediction_dimension,
                                  std::vector<float>& predictions) {
  if (!pool.has_baseline()) {
    return;
  }
  if (pool.baseline_dimension() != prediction_dimension) {
    throw std::invalid_argument("baseline dimension must match the model prediction dimension");
  }
  if (pool.baseline().size() != predictions.size()) {
    throw std::invalid_argument("baseline size must match the flattened prediction size");
  }
  for (std::size_t index = 0; index < predictions.size(); ++index) {
    predictions[index] += pool.baseline()[index];
  }
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

std::string NormalizeTaskType(std::string task_type) {
  return NormalizeToken(std::move(task_type));
}

std::string CanonicalBootstrapType(std::string bootstrap_type) {
  const std::string normalized = NormalizeToken(std::move(bootstrap_type));
  if (normalized.empty() || normalized == "no" || normalized == "none") {
    return "No";
  }
  if (normalized == "bernoulli") {
    return "Bernoulli";
  }
  if (normalized == "poisson") {
    return "Poisson";
  }
  if (normalized == "bayesian") {
    return "Bayesian";
  }
  throw std::invalid_argument("bootstrap_type must be one of: No, Bernoulli, Poisson, Bayesian");
}

BootstrapType ParseBootstrapType(const std::string& bootstrap_type) {
  const std::string normalized = NormalizeToken(bootstrap_type);
  if (normalized == "no" || normalized == "none") {
    return BootstrapType::kNone;
  }
  if (normalized == "bernoulli") {
    return BootstrapType::kBernoulli;
  }
  if (normalized == "poisson") {
    return BootstrapType::kPoisson;
  }
  if (normalized == "bayesian") {
    return BootstrapType::kBayesian;
  }
  throw std::invalid_argument("bootstrap_type must be one of: No, Bernoulli, Poisson, Bayesian");
}

std::string CanonicalBoostingType(std::string boosting_type) {
  const std::string normalized = NormalizeToken(std::move(boosting_type));
  if (normalized.empty() || normalized == "plain" || normalized == "gradientboosting" ||
      normalized == "gbdt") {
    return "GradientBoosting";
  }
  if (normalized == "randomforest" || normalized == "rf") {
    return "RandomForest";
  }
  if (normalized == "dart") {
    return "DART";
  }
  throw std::invalid_argument("boosting_type must be one of: GradientBoosting, RandomForest, DART");
}

BoostingType ParseBoostingType(const std::string& boosting_type) {
  const std::string normalized = NormalizeToken(boosting_type);
  if (normalized == "gradientboosting" || normalized == "plain" || normalized == "gbdt") {
    return BoostingType::kGradientBoosting;
  }
  if (normalized == "randomforest" || normalized == "rf") {
    return BoostingType::kRandomForest;
  }
  if (normalized == "dart") {
    return BoostingType::kDart;
  }
  throw std::invalid_argument("boosting_type must be one of: GradientBoosting, RandomForest, DART");
}

std::string CanonicalGrowPolicy(std::string grow_policy) {
  const std::string normalized = NormalizeToken(std::move(grow_policy));
  if (normalized.empty() || normalized == "depthwise" || normalized == "symmetric") {
    return "DepthWise";
  }
  if (normalized == "leafwise" || normalized == "lossguide" || normalized == "bestfirst") {
    return "LeafWise";
  }
  throw std::invalid_argument("grow_policy must be one of: DepthWise, LeafWise");
}

GrowPolicy ParseGrowPolicy(const std::string& grow_policy) {
  const std::string normalized = NormalizeToken(grow_policy);
  if (normalized == "depthwise" || normalized == "symmetric") {
    return GrowPolicy::DepthWise;
  }
  if (normalized == "leafwise" || normalized == "lossguide" || normalized == "bestfirst") {
    return GrowPolicy::LeafWise;
  }
  throw std::invalid_argument("grow_policy must be one of: DepthWise, LeafWise");
}

const QuantizationSchema& RequireQuantizationSchema(
    const QuantizationSchemaPtr& quantization_schema) {
  if (quantization_schema == nullptr) {
    throw std::runtime_error("booster quantization schema is not initialized");
  }
  return *quantization_schema;
}

}  // namespace ctboost::booster_detail
