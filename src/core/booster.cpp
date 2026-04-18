#include "ctboost/booster.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ctboost/cuda_backend.hpp"
#include "ctboost/distributed_client.hpp"
#include "ctboost/profiler.hpp"

namespace ctboost {
namespace {

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
         normalized_objective == "survival:cox" ||
         normalized_objective == "survivalexponential" ||
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

bool IsRegressionObjective(const std::string& normalized_objective) {
  return IsSquaredErrorObjective(normalized_objective) ||
         IsAbsoluteErrorObjective(normalized_objective) ||
         IsHuberObjective(normalized_objective) ||
         IsQuantileObjective(normalized_objective) ||
         IsPoissonObjective(normalized_objective) ||
         IsTweedieObjective(normalized_objective) ||
         IsSurvivalObjective(normalized_objective);
}

enum class BootstrapType {
  kNone,
  kBernoulli,
  kPoisson,
  kBayesian,
};

enum class BoostingType {
  kGradientBoosting,
  kRandomForest,
  kDart,
};

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

const QuantizationSchema& RequireQuantizationSchema(const QuantizationSchemaPtr& quantization_schema) {
  if (quantization_schema == nullptr) {
    throw std::runtime_error("booster quantization schema is not initialized");
  }
  return *quantization_schema;
}

struct DistributedMetricControl {
  double train_loss{0.0};
  double eval_score{0.0};
  double best_score{0.0};
  int best_iteration{-1};
  std::uint8_t has_eval{0};
  std::uint8_t should_stop{0};
};

std::vector<std::uint8_t> SerializeDistributedMetricControl(
    const DistributedMetricControl& control) {
  std::vector<std::uint8_t> buffer(sizeof(DistributedMetricControl), 0U);
  std::memcpy(buffer.data(), &control, sizeof(DistributedMetricControl));
  return buffer;
}

DistributedMetricControl DeserializeDistributedMetricControl(
    const std::vector<std::uint8_t>& buffer) {
  if (buffer.size() != sizeof(DistributedMetricControl)) {
    throw std::runtime_error("distributed metric control payload size mismatch");
  }
  DistributedMetricControl control{};
  std::memcpy(&control, buffer.data(), sizeof(DistributedMetricControl));
  return control;
}

DistributedMetricControl BroadcastDistributedMetricControl(
    const DistributedCoordinator* coordinator,
    const char* label,
    const DistributedMetricControl* root_control) {
  if (coordinator == nullptr || coordinator->world_size <= 1) {
    return root_control == nullptr ? DistributedMetricControl{} : *root_control;
  }
  if (!DistributedRootUsesTcp(coordinator->root)) {
    return root_control == nullptr ? DistributedMetricControl{} : *root_control;
  }
  std::vector<std::uint8_t> payload;
  if (coordinator->rank == 0 && root_control != nullptr) {
    payload = SerializeDistributedMetricControl(*root_control);
  }
  const std::string key =
      coordinator->run_id + "/" + std::to_string(coordinator->tree_index) + "/" + label;
  return DeserializeDistributedMetricControl(DistributedTcpRequest(coordinator->root,
                                                                   coordinator->timeout_seconds,
                                                                   "broadcast",
                                                                   key,
                                                                   coordinator->rank,
                                                                   coordinator->world_size,
                                                                   payload));
}

template <typename T>
void AppendBinary(std::vector<std::uint8_t>& buffer, const T& value) {
  const auto* bytes = reinterpret_cast<const std::uint8_t*>(&value);
  buffer.insert(buffer.end(), bytes, bytes + sizeof(T));
}

template <typename T>
T ReadBinary(const std::vector<std::uint8_t>& buffer, std::size_t& offset) {
  if (offset + sizeof(T) > buffer.size()) {
    throw std::runtime_error("distributed metric payload is truncated");
  }
  T value{};
  std::memcpy(&value, buffer.data() + offset, sizeof(T));
  offset += sizeof(T);
  return value;
}

struct DistributedMetricInputs {
  std::vector<float> predictions;
  std::vector<float> labels;
  std::vector<float> weights;
  std::vector<std::int64_t> group_ids;
  bool has_group_ids{false};
};

std::vector<std::uint8_t> SerializeDistributedMetricInputs(
    const DistributedMetricInputs& inputs) {
  if (inputs.labels.size() != inputs.weights.size()) {
    throw std::invalid_argument("distributed metric labels and weights must have matching sizes");
  }
  if (inputs.labels.empty() ? !inputs.predictions.empty()
                            : inputs.predictions.size() % inputs.labels.size() != 0U) {
    throw std::invalid_argument(
        "distributed metric prediction size must be a multiple of the label count");
  }
  if (inputs.has_group_ids && inputs.group_ids.size() != inputs.labels.size()) {
    throw std::invalid_argument(
        "distributed metric group_ids must match the label count when provided");
  }
  std::vector<std::uint8_t> buffer;
  const std::uint64_t num_rows = static_cast<std::uint64_t>(inputs.labels.size());
  const std::uint64_t prediction_size = static_cast<std::uint64_t>(inputs.predictions.size());
  const std::uint8_t has_group_ids = inputs.has_group_ids ? 1U : 0U;
  AppendBinary(buffer, num_rows);
  AppendBinary(buffer, prediction_size);
  AppendBinary(buffer, has_group_ids);
  if (!inputs.predictions.empty()) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(inputs.predictions.data());
    buffer.insert(buffer.end(), bytes, bytes + inputs.predictions.size() * sizeof(float));
  }
  if (!inputs.labels.empty()) {
    const auto* label_bytes = reinterpret_cast<const std::uint8_t*>(inputs.labels.data());
    buffer.insert(buffer.end(), label_bytes, label_bytes + inputs.labels.size() * sizeof(float));
    const auto* weight_bytes = reinterpret_cast<const std::uint8_t*>(inputs.weights.data());
    buffer.insert(buffer.end(), weight_bytes, weight_bytes + inputs.weights.size() * sizeof(float));
  }
  if (has_group_ids != 0U && !inputs.group_ids.empty()) {
    const auto* group_bytes = reinterpret_cast<const std::uint8_t*>(inputs.group_ids.data());
    buffer.insert(buffer.end(),
                  group_bytes,
                  group_bytes + inputs.group_ids.size() * sizeof(std::int64_t));
  }
  return buffer;
}

DistributedMetricInputs DeserializeDistributedMetricInputs(
    const std::vector<std::uint8_t>& buffer) {
  std::size_t offset = 0;
  const std::uint64_t num_rows = ReadBinary<std::uint64_t>(buffer, offset);
  const std::uint64_t prediction_size = ReadBinary<std::uint64_t>(buffer, offset);
  const std::uint8_t has_group_ids = ReadBinary<std::uint8_t>(buffer, offset);
  DistributedMetricInputs inputs;
  inputs.has_group_ids = has_group_ids != 0U;
  inputs.predictions.resize(static_cast<std::size_t>(prediction_size));
  inputs.labels.resize(static_cast<std::size_t>(num_rows));
  inputs.weights.resize(static_cast<std::size_t>(num_rows));
  const auto read_float_array = [&](std::vector<float>& values) {
    const std::size_t byte_count = values.size() * sizeof(float);
    if (offset + byte_count > buffer.size()) {
      throw std::runtime_error("distributed metric payload is truncated");
    }
    if (byte_count != 0U) {
      std::memcpy(values.data(), buffer.data() + offset, byte_count);
    }
    offset += byte_count;
  };
  read_float_array(inputs.predictions);
  read_float_array(inputs.labels);
  read_float_array(inputs.weights);
  if (inputs.has_group_ids) {
    inputs.group_ids.resize(static_cast<std::size_t>(num_rows));
    const std::size_t byte_count = inputs.group_ids.size() * sizeof(std::int64_t);
    if (offset + byte_count > buffer.size()) {
      throw std::runtime_error("distributed metric group_id payload is truncated");
    }
    if (byte_count != 0U) {
      std::memcpy(inputs.group_ids.data(), buffer.data() + offset, byte_count);
    }
    offset += byte_count;
  }
  return inputs;
}

std::vector<std::vector<std::uint8_t>> DeserializeGatheredPayloads(
    const std::vector<std::uint8_t>& buffer) {
  std::size_t offset = 0;
  const std::uint64_t payload_count = ReadBinary<std::uint64_t>(buffer, offset);
  std::vector<std::vector<std::uint8_t>> payloads;
  payloads.reserve(static_cast<std::size_t>(payload_count));
  for (std::size_t index = 0; index < static_cast<std::size_t>(payload_count); ++index) {
    const std::uint64_t payload_size = ReadBinary<std::uint64_t>(buffer, offset);
    if (offset + payload_size > buffer.size()) {
      throw std::runtime_error("distributed allgather payload is truncated");
    }
    payloads.emplace_back(buffer.begin() + static_cast<std::ptrdiff_t>(offset),
                          buffer.begin() + static_cast<std::ptrdiff_t>(offset + payload_size));
    offset += static_cast<std::size_t>(payload_size);
  }
  return payloads;
}

DistributedMetricInputs MergeDistributedMetricInputs(
    const std::vector<std::vector<std::uint8_t>>& payloads) {
  DistributedMetricInputs merged;
  bool initialized = false;
  for (const auto& payload : payloads) {
    DistributedMetricInputs shard = DeserializeDistributedMetricInputs(payload);
    if (!initialized) {
      merged.has_group_ids = shard.has_group_ids;
      initialized = true;
    } else if (merged.has_group_ids != shard.has_group_ids) {
      throw std::invalid_argument(
          "distributed metric shards must either all include group_ids or all omit them");
    }
    merged.predictions.insert(
        merged.predictions.end(), shard.predictions.begin(), shard.predictions.end());
    merged.labels.insert(merged.labels.end(), shard.labels.begin(), shard.labels.end());
    merged.weights.insert(merged.weights.end(), shard.weights.begin(), shard.weights.end());
    if (merged.has_group_ids) {
      merged.group_ids.insert(merged.group_ids.end(), shard.group_ids.begin(), shard.group_ids.end());
    }
  }
  return merged;
}

DistributedMetricInputs AllGatherDistributedMetricInputs(
    const DistributedCoordinator* coordinator,
    const char* label,
    const DistributedMetricInputs& local_inputs) {
  if (coordinator == nullptr || coordinator->world_size <= 1 ||
      !DistributedRootUsesTcp(coordinator->root)) {
    return local_inputs;
  }
  const std::string key =
      coordinator->run_id + "/" + std::to_string(coordinator->tree_index) + "/" + label;
  const std::vector<std::uint8_t> response = DistributedTcpRequest(
      coordinator->root,
      coordinator->timeout_seconds,
      "allgather",
      key,
      coordinator->rank,
      coordinator->world_size,
      SerializeDistributedMetricInputs(local_inputs));
  return MergeDistributedMetricInputs(DeserializeGatheredPayloads(response));
}

double UniformUnit(std::uint64_t& state) {
  constexpr double kScale = 1.0 / static_cast<double>(1ULL << 53U);
  return static_cast<double>(NextRandom(state) >> 11U) * kScale;
}

std::uint32_t SamplePoisson(double lambda, std::uint64_t& state) {
  if (lambda <= 0.0) {
    return 0U;
  }
  const double threshold = std::exp(-lambda);
  std::uint32_t count = 0U;
  double product = 1.0;
  do {
    ++count;
    product *= UniformUnit(state);
  } while (product > threshold);
  return count - 1U;
}

float SampleBayesianBootstrapWeight(float base_weight,
                                    double bagging_temperature,
                                    std::uint64_t& state) {
  if (base_weight <= 0.0F) {
    return 0.0F;
  }
  if (bagging_temperature <= 0.0) {
    return base_weight;
  }

  const double uniform = std::max(UniformUnit(state), std::numeric_limits<double>::min());
  const double bootstrap_draw = -std::log(uniform);
  const double weight_scale = std::pow(bootstrap_draw, bagging_temperature);
  return static_cast<float>(static_cast<double>(base_weight) * weight_scale);
}

std::vector<float> SampleRowWeights(const std::vector<float>& base_weights,
                                    double subsample,
                                    BootstrapType bootstrap_type,
                                    double bagging_temperature,
                                    std::uint64_t& rng_state) {
  if (base_weights.empty()) {
    return {};
  }

  if (bootstrap_type == BootstrapType::kNone && subsample >= 1.0) {
    return base_weights;
  }

  std::vector<float> sampled_weights(base_weights.size(), 0.0F);
  double total_weight = 0.0;
  for (std::size_t row = 0; row < base_weights.size(); ++row) {
    const float base_weight = base_weights[row];
    if (base_weight <= 0.0F) {
      continue;
    }

    float row_weight = 0.0F;
    if (bootstrap_type == BootstrapType::kPoisson) {
      row_weight = base_weight * static_cast<float>(SamplePoisson(subsample, rng_state));
    } else if (bootstrap_type == BootstrapType::kBayesian) {
      row_weight = SampleBayesianBootstrapWeight(base_weight, bagging_temperature, rng_state);
    } else {
      const double include_probability = subsample >= 1.0 ? 1.0 : subsample;
      row_weight = UniformUnit(rng_state) < include_probability ? base_weight : 0.0F;
    }
    sampled_weights[row] = row_weight;
    total_weight += static_cast<double>(row_weight);
  }

  if (total_weight > 0.0) {
    return sampled_weights;
  }

  std::vector<std::size_t> positive_rows;
  positive_rows.reserve(base_weights.size());
  for (std::size_t row = 0; row < base_weights.size(); ++row) {
    if (base_weights[row] > 0.0F) {
      positive_rows.push_back(row);
    }
  }
  if (!positive_rows.empty()) {
    const std::size_t selected =
        positive_rows[UniformIndex(rng_state, positive_rows.size())];
    sampled_weights[selected] = base_weights[selected];
  }
  return sampled_weights;
}

void ScaleTreeLeafWeights(Tree& tree, double scale) {
  if (scale == 1.0) {
    return;
  }
  const auto& nodes = tree.nodes();
  for (std::size_t node_index = 0; node_index < nodes.size(); ++node_index) {
    if (!nodes[node_index].is_leaf) {
      continue;
    }
    tree.SetLeafWeight(node_index, static_cast<float>(static_cast<double>(nodes[node_index].leaf_weight) * scale));
  }
}

std::vector<std::size_t> SampleDroppedTreeGroups(std::size_t completed_iterations,
                                                 double drop_rate,
                                                 double skip_drop,
                                                 int max_drop,
                                                 std::uint64_t& rng_state) {
  if (completed_iterations == 0 || drop_rate <= 0.0) {
    return {};
  }
  if (skip_drop > 0.0 && UniformUnit(rng_state) < skip_drop) {
    return {};
  }

  std::vector<std::size_t> dropped_groups;
  dropped_groups.reserve(completed_iterations);
  for (std::size_t iteration = 0; iteration < completed_iterations; ++iteration) {
    if (UniformUnit(rng_state) < drop_rate) {
      dropped_groups.push_back(iteration);
    }
  }

  if (dropped_groups.empty()) {
    dropped_groups.push_back(UniformIndex(rng_state, completed_iterations));
  }
  if (max_drop > 0 && dropped_groups.size() > static_cast<std::size_t>(max_drop)) {
    std::shuffle(dropped_groups.begin(), dropped_groups.end(), std::mt19937_64(NextRandom(rng_state)));
    dropped_groups.resize(static_cast<std::size_t>(max_drop));
    std::sort(dropped_groups.begin(), dropped_groups.end());
  }
  return dropped_groups;
}

InteractionConstraintSet BuildInteractionConstraintSet(
    const std::vector<std::vector<int>>& raw_constraints,
    std::size_t num_features) {
  InteractionConstraintSet constraints;
  constraints.groups.reserve(raw_constraints.size());
  constraints.feature_to_groups.resize(num_features);
  constraints.constrained_feature_mask.assign(num_features, 0U);

  for (std::size_t group_index = 0; group_index < raw_constraints.size(); ++group_index) {
    std::vector<int> group = raw_constraints[group_index];
    group.erase(std::remove_if(group.begin(),
                               group.end(),
                               [num_features](int feature_id) {
                                 return feature_id < 0 ||
                                        static_cast<std::size_t>(feature_id) >= num_features;
                               }),
                group.end());
    std::sort(group.begin(), group.end());
    group.erase(std::unique(group.begin(), group.end()), group.end());
    if (group.empty()) {
      continue;
    }
    const int stored_group_index = static_cast<int>(constraints.groups.size());
    constraints.groups.push_back(group);
    for (const int feature_id : group) {
      constraints.feature_to_groups[static_cast<std::size_t>(feature_id)].push_back(stored_group_index);
      constraints.constrained_feature_mask[static_cast<std::size_t>(feature_id)] = 1U;
    }
  }

  for (auto& feature_groups : constraints.feature_to_groups) {
    std::sort(feature_groups.begin(), feature_groups.end());
    feature_groups.erase(std::unique(feature_groups.begin(), feature_groups.end()), feature_groups.end());
  }
  return constraints;
}

HistMatrix BuildPredictionHist(const Pool& pool, const QuantizationSchema& quantization_schema);

HistMatrix BuildPredictionHist(const Pool& pool, const Tree& reference_tree) {
  return BuildPredictionHist(pool, RequireQuantizationSchema(reference_tree.shared_quantization_schema()));
}

bool CanUseCompactBins(const QuantizationSchema& quantization_schema) {
  for (const std::uint16_t feature_bins_count : quantization_schema.num_bins_per_feature) {
    if (feature_bins_count >
        static_cast<std::uint16_t>(std::numeric_limits<std::uint8_t>::max()) + 1U) {
      return false;
    }
  }
  return true;
}

HistMatrix BuildPredictionHist(const Pool& pool, const QuantizationSchema& quantization_schema) {
  if (pool.num_cols() != quantization_schema.num_cols()) {
    throw std::invalid_argument(
        "prediction pool must have the same number of columns as the fitted model");
  }

  HistMatrix hist;
  hist.num_rows = pool.num_rows();
  hist.num_cols = pool.num_cols();
  ApplyQuantizationSchema(quantization_schema, hist);
  if (CanUseCompactBins(quantization_schema)) {
    hist.compact_bin_indices.resize(hist.num_rows * hist.num_cols, 0);
    hist.bin_index_bytes = 1;
  } else {
    hist.bin_indices.resize(hist.num_rows * hist.num_cols, 0);
    hist.bin_index_bytes = 2;
  }

  for (std::size_t feature = 0; feature < hist.num_cols; ++feature) {
    const float* const contiguous_column = pool.feature_column_ptr(feature);
    for (std::size_t row = 0; row < hist.num_rows; ++row) {
      const float value =
          contiguous_column != nullptr ? contiguous_column[row] : pool.feature_value(row, feature);
      hist.set_bin_index(feature, row, quantization_schema.bin_value(feature, value));
    }
  }

  return hist;
}

std::vector<GpuTreeNode> FlattenTreesForGpu(const std::vector<Tree>& trees,
                                            std::size_t tree_limit,
                                            std::vector<std::int32_t>& tree_offsets) {
  std::size_t total_nodes = 0;
  for (std::size_t tree_index = 0; tree_index < tree_limit; ++tree_index) {
    total_nodes += trees[tree_index].nodes().size();
  }

  std::vector<GpuTreeNode> flattened_nodes;
  flattened_nodes.reserve(total_nodes);
  tree_offsets.clear();
  tree_offsets.reserve(tree_limit);

  for (std::size_t tree_index = 0; tree_index < tree_limit; ++tree_index) {
    const auto& tree_nodes = trees[tree_index].nodes();
    const std::int32_t tree_offset = static_cast<std::int32_t>(flattened_nodes.size());
    tree_offsets.push_back(tree_offset);
    for (const Node& node : tree_nodes) {
      GpuTreeNode gpu_node;
      gpu_node.is_leaf = node.is_leaf ? 1U : 0U;
      gpu_node.is_categorical_split = node.is_categorical_split ? 1U : 0U;
      gpu_node.split_bin_index = node.split_bin_index;
      gpu_node.split_feature_id = static_cast<std::int32_t>(node.split_feature_id);
      gpu_node.left_child =
          node.left_child < 0 ? -1 : tree_offset + static_cast<std::int32_t>(node.left_child);
      gpu_node.right_child =
          node.right_child < 0 ? -1 : tree_offset + static_cast<std::int32_t>(node.right_child);
      gpu_node.leaf_weight = node.leaf_weight;
      std::copy(node.left_categories.begin(), node.left_categories.end(), gpu_node.left_categories);
      flattened_nodes.push_back(std::move(gpu_node));
    }
  }

  return flattened_nodes;
}

void UpdatePredictions(const Tree& tree,
                       const HistMatrix& hist,
                       double learning_rate,
                       int prediction_dimension,
                       int class_index,
                       std::vector<float>& predictions) {
  if (prediction_dimension == 1) {
    for (std::size_t row = 0; row < hist.num_rows; ++row) {
      predictions[row] += learning_rate * tree.PredictBinnedRow(hist, row);
    }
    return;
  }

  for (std::size_t row = 0; row < hist.num_rows; ++row) {
    const std::size_t offset = row * static_cast<std::size_t>(prediction_dimension) + class_index;
    predictions[offset] += learning_rate * tree.PredictBinnedRow(hist, row);
  }
}

void AccumulateIterationPredictions(const std::vector<Tree>& trees,
                                    std::size_t iteration_index,
                                    const HistMatrix& hist,
                                    double learning_rate,
                                    int prediction_dimension,
                                    std::vector<float>& predictions) {
  const std::size_t tree_begin =
      iteration_index * static_cast<std::size_t>(prediction_dimension);
  const std::size_t tree_end = tree_begin + static_cast<std::size_t>(prediction_dimension);
  for (std::size_t tree_index = tree_begin; tree_index < tree_end; ++tree_index) {
    const int class_index =
        prediction_dimension == 1 ? 0
                                  : static_cast<int>(
                                        tree_index % static_cast<std::size_t>(prediction_dimension));
    UpdatePredictions(
        trees[tree_index], hist, learning_rate, prediction_dimension, class_index, predictions);
  }
}

std::vector<float> PredictFromHist(const std::vector<Tree>& trees,
                                   const HistMatrix& hist,
                                   std::size_t tree_limit,
                                   double learning_rate,
                                   bool use_gpu,
                                   int prediction_dimension,
                                   const std::string& devices = "0") {
  std::vector<float> predictions(
      hist.num_rows * static_cast<std::size_t>(prediction_dimension), 0.0F);
  if (tree_limit == 0 || hist.num_rows == 0) {
    return predictions;
  }

  if (use_gpu && CudaBackendCompiled()) {
    std::vector<std::int32_t> tree_offsets;
    const std::vector<GpuTreeNode> flattened_nodes =
        FlattenTreesForGpu(trees, tree_limit, tree_offsets);
    PredictRawGpu(
        hist,
        flattened_nodes,
        tree_offsets,
        static_cast<float>(learning_rate),
        prediction_dimension,
        predictions,
        devices);
    return predictions;
  }

  for (std::size_t tree_index = 0; tree_index < tree_limit; ++tree_index) {
    const int class_index =
        prediction_dimension == 1
            ? 0
            : static_cast<int>(tree_index % static_cast<std::size_t>(prediction_dimension));
    UpdatePredictions(
        trees[tree_index], hist, learning_rate, prediction_dimension, class_index, predictions);
  }
  return predictions;
}

void UpdatePredictionsFromLeafRanges(const Tree& tree,
                                     const std::vector<std::size_t>& row_indices,
                                     const std::vector<LeafRowRange>& leaf_row_ranges,
                                     double learning_rate,
                                     int prediction_dimension,
                                     int class_index,
                                     std::vector<float>& predictions) {
  const auto& nodes = tree.nodes();
  if (nodes.empty() || row_indices.empty() || leaf_row_ranges.size() < nodes.size()) {
    return;
  }

  if (prediction_dimension == 1) {
    for (std::size_t node_index = 0; node_index < nodes.size(); ++node_index) {
      const Node& node = nodes[node_index];
      if (!node.is_leaf) {
        continue;
      }
      const LeafRowRange& range = leaf_row_ranges[node_index];
      if (range.end <= range.begin) {
        continue;
      }
      const float update = static_cast<float>(learning_rate) * node.leaf_weight;
      for (std::size_t position = range.begin; position < range.end; ++position) {
        predictions[row_indices[position]] += update;
      }
    }
    return;
  }

  for (std::size_t node_index = 0; node_index < nodes.size(); ++node_index) {
    const Node& node = nodes[node_index];
    if (!node.is_leaf) {
      continue;
    }
    const LeafRowRange& range = leaf_row_ranges[node_index];
    if (range.end <= range.begin) {
      continue;
    }
    const float update = static_cast<float>(learning_rate) * node.leaf_weight;
    for (std::size_t position = range.begin; position < range.end; ++position) {
      const std::size_t row = row_indices[position];
      const std::size_t offset =
          row * static_cast<std::size_t>(prediction_dimension) + class_index;
      predictions[offset] += update;
    }
  }
}

float ComputeLeafWeightFromSums(double gradient_sum, double hessian_sum, double lambda_l2) {
  return static_cast<float>(-gradient_sum / (hessian_sum + lambda_l2));
}

void BuildSharedMulticlassTargets(const std::vector<float>& gradients,
                                  const std::vector<float>& hessians,
                                  const std::vector<float>& weights,
                                  std::size_t num_rows,
                                  int prediction_dimension,
                                  std::vector<float>& structure_gradients,
                                  std::vector<float>& structure_hessians) {
  structure_gradients.assign(num_rows, 0.0F);
  structure_hessians.assign(num_rows, 0.0F);
  if (prediction_dimension <= 0) {
    return;
  }

  std::vector<double> gradient_sums(static_cast<std::size_t>(prediction_dimension), 0.0);
  std::vector<double> gradient_square_sums(static_cast<std::size_t>(prediction_dimension), 0.0);
  std::vector<double> weight_sums(static_cast<std::size_t>(prediction_dimension), 0.0);
  for (std::size_t row = 0; row < num_rows; ++row) {
    const double sample_weight = static_cast<double>(weights[row]);
    const std::size_t offset = row * static_cast<std::size_t>(prediction_dimension);
    for (int class_index = 0; class_index < prediction_dimension; ++class_index) {
      const double gradient = gradients[offset + static_cast<std::size_t>(class_index)];
      gradient_sums[static_cast<std::size_t>(class_index)] += sample_weight * gradient;
      gradient_square_sums[static_cast<std::size_t>(class_index)] +=
          sample_weight * gradient * gradient;
      weight_sums[static_cast<std::size_t>(class_index)] += sample_weight;
    }
  }

  int structure_class = 0;
  double best_variance = -1.0;
  for (int class_index = 0; class_index < prediction_dimension; ++class_index) {
    const double total_weight = weight_sums[static_cast<std::size_t>(class_index)];
    if (total_weight <= 0.0) {
      continue;
    }
    const double mean_gradient =
        gradient_sums[static_cast<std::size_t>(class_index)] / total_weight;
    const double variance =
        std::max(0.0,
                 gradient_square_sums[static_cast<std::size_t>(class_index)] / total_weight -
                     mean_gradient * mean_gradient);
    if (variance > best_variance) {
      best_variance = variance;
      structure_class = class_index;
    }
  }

  for (std::size_t row = 0; row < num_rows; ++row) {
    const std::size_t offset = row * static_cast<std::size_t>(prediction_dimension);
    const std::size_t target_index = offset + static_cast<std::size_t>(structure_class);
    structure_gradients[row] = gradients[target_index];
    structure_hessians[row] = std::max(0.0F, hessians[target_index]);
  }
}

std::vector<int> PredictLeafIndicesFromHist(const Tree& tree, const HistMatrix& hist) {
  std::vector<int> leaf_indices(hist.num_rows, -1);
  for (std::size_t row = 0; row < hist.num_rows; ++row) {
    leaf_indices[row] = tree.PredictBinnedLeafIndex(hist, row);
  }
  return leaf_indices;
}

void UpdatePredictionsFromLeafIndices(const Tree& tree,
                                      const std::vector<int>& leaf_indices,
                                      double learning_rate,
                                      int prediction_dimension,
                                      int class_index,
                                      std::vector<float>& predictions) {
  const auto& nodes = tree.nodes();
  if (nodes.empty() || leaf_indices.empty()) {
    return;
  }

  if (prediction_dimension == 1) {
    for (std::size_t row = 0; row < leaf_indices.size(); ++row) {
      const int leaf_index = leaf_indices[row];
      if (leaf_index < 0) {
        continue;
      }
      predictions[row] += learning_rate * nodes[static_cast<std::size_t>(leaf_index)].leaf_weight;
    }
    return;
  }

  for (std::size_t row = 0; row < leaf_indices.size(); ++row) {
    const int leaf_index = leaf_indices[row];
    if (leaf_index < 0) {
      continue;
    }
    const std::size_t offset = row * static_cast<std::size_t>(prediction_dimension) + class_index;
    predictions[offset] +=
        learning_rate * nodes[static_cast<std::size_t>(leaf_index)].leaf_weight;
  }
}

std::vector<Tree> MaterializeMulticlassTreesFromStructure(
    const Tree& structure_tree,
    const std::vector<std::size_t>& row_indices,
    const std::vector<LeafRowRange>& leaf_row_ranges,
    const std::vector<float>& gradients,
    const std::vector<float>& hessians,
    const std::vector<float>& weights,
    int prediction_dimension,
    double lambda_l2) {
  std::vector<Tree> class_trees(
      static_cast<std::size_t>(prediction_dimension), structure_tree);
  const auto& structure_nodes = structure_tree.nodes();
  if (structure_nodes.empty() || row_indices.empty() ||
      leaf_row_ranges.size() < structure_nodes.size()) {
    return class_trees;
  }

  for (std::size_t node_index = 0; node_index < structure_nodes.size(); ++node_index) {
    const Node& node = structure_nodes[node_index];
    if (!node.is_leaf) {
      continue;
    }

    const LeafRowRange& range = leaf_row_ranges[node_index];
    if (range.end <= range.begin) {
      for (Tree& tree : class_trees) {
        tree.SetLeafWeight(node_index, 0.0F);
      }
      continue;
    }

    std::vector<double> gradient_sums(static_cast<std::size_t>(prediction_dimension), 0.0);
    std::vector<double> hessian_sums(static_cast<std::size_t>(prediction_dimension), 0.0);
    for (std::size_t position = range.begin; position < range.end; ++position) {
      const std::size_t row = row_indices[position];
      const double sample_weight = static_cast<double>(weights[row]);
      const std::size_t offset = row * static_cast<std::size_t>(prediction_dimension);
      for (int class_index = 0; class_index < prediction_dimension; ++class_index) {
        const std::size_t target_index = offset + static_cast<std::size_t>(class_index);
        gradient_sums[static_cast<std::size_t>(class_index)] +=
            sample_weight * static_cast<double>(gradients[target_index]);
        hessian_sums[static_cast<std::size_t>(class_index)] +=
            sample_weight * static_cast<double>(hessians[target_index]);
      }
    }

    for (int class_index = 0; class_index < prediction_dimension; ++class_index) {
      class_trees[static_cast<std::size_t>(class_index)].SetLeafWeight(
          node_index,
          ComputeLeafWeightFromSums(gradient_sums[static_cast<std::size_t>(class_index)],
                                    hessian_sums[static_cast<std::size_t>(class_index)],
                                    lambda_l2));
    }
  }

  return class_trees;
}

void UpdatePredictions(const Tree& tree,
                       const Pool& pool,
                       double learning_rate,
                       int prediction_dimension,
                       int class_index,
                       std::vector<float>& predictions) {
  if (prediction_dimension == 1) {
    for (std::size_t row = 0; row < pool.num_rows(); ++row) {
      predictions[row] += learning_rate * tree.PredictRow(pool, row);
    }
    return;
  }

  for (std::size_t row = 0; row < pool.num_rows(); ++row) {
    const std::size_t offset = row * static_cast<std::size_t>(prediction_dimension) + class_index;
    predictions[offset] += learning_rate * tree.PredictRow(pool, row);
  }
}

std::vector<int> SampleFeatureSubset(std::size_t num_features,
                                     double colsample_bytree,
                                     const std::vector<double>* feature_weights,
                                     std::uint64_t& rng_state) {
  std::vector<int> eligible_features;
  eligible_features.reserve(num_features);
  if (feature_weights != nullptr && !feature_weights->empty()) {
    for (std::size_t feature = 0; feature < num_features; ++feature) {
      const double feature_weight =
          feature < feature_weights->size() ? (*feature_weights)[feature] : 1.0;
      if (feature_weight > 0.0) {
        eligible_features.push_back(static_cast<int>(feature));
      }
    }
  } else {
    eligible_features.resize(num_features);
    std::iota(eligible_features.begin(), eligible_features.end(), 0);
  }

  if (eligible_features.size() <= 1) {
    return eligible_features.size() == num_features ? std::vector<int>{} : eligible_features;
  }

  const std::size_t eligible_count = eligible_features.size();
  const std::size_t subset_size = std::max<std::size_t>(
      1, static_cast<std::size_t>(std::ceil(colsample_bytree * static_cast<double>(eligible_count))));
  if (subset_size >= eligible_count) {
    return eligible_count == num_features ? std::vector<int>{} : eligible_features;
  }

  if (feature_weights == nullptr || feature_weights->empty()) {
    for (std::size_t index = 0; index < subset_size; ++index) {
      const std::size_t swap_index =
          index + UniformIndex(rng_state, eligible_count - index);
      std::swap(eligible_features[index], eligible_features[swap_index]);
    }
    eligible_features.resize(subset_size);
    std::sort(eligible_features.begin(), eligible_features.end());
    return eligible_features;
  }

  std::vector<std::pair<double, int>> keyed_features;
  keyed_features.reserve(eligible_count);
  for (const int feature_id : eligible_features) {
    const double feature_weight = std::max(
        std::numeric_limits<double>::min(),
        (*feature_weights)[static_cast<std::size_t>(feature_id)]);
    const double uniform = std::max(UniformUnit(rng_state), std::numeric_limits<double>::min());
    const double key = std::log(uniform) / feature_weight;
    keyed_features.emplace_back(key, feature_id);
  }
  std::partial_sort(keyed_features.begin(),
                    keyed_features.begin() + static_cast<std::ptrdiff_t>(subset_size),
                    keyed_features.end(),
                    [](const auto& lhs, const auto& rhs) { return lhs.first > rhs.first; });

  std::vector<int> selected_features;
  selected_features.reserve(subset_size);
  for (std::size_t index = 0; index < subset_size; ++index) {
    selected_features.push_back(keyed_features[index].second);
  }
  std::sort(selected_features.begin(), selected_features.end());
  return selected_features;
}

void AccumulateFeatureImportances(const Tree& tree, std::vector<double>& feature_importance_sums) {
  const auto& tree_feature_importances = tree.feature_importances();
  for (std::size_t feature = 0; feature < tree_feature_importances.size(); ++feature) {
    feature_importance_sums[feature] += tree_feature_importances[feature];
  }
}

void RecomputeFeatureImportances(const std::vector<Tree>& trees,
                                 std::size_t num_features,
                                 std::vector<double>& feature_importance_sums) {
  feature_importance_sums.assign(num_features, 0.0);
  for (const Tree& tree : trees) {
    AccumulateFeatureImportances(tree, feature_importance_sums);
  }
}

void MarkUsedFeatures(const Tree& tree, std::vector<std::uint8_t>& feature_used_mask) {
  for (const auto& node : tree.nodes()) {
    if (node.is_leaf || node.split_feature_id < 0) {
      continue;
    }
    const std::size_t feature_index = static_cast<std::size_t>(node.split_feature_id);
    if (feature_index < feature_used_mask.size()) {
      feature_used_mask[feature_index] = 1U;
    }
  }
}

struct FitWorkspace {
  HistMatrix train_hist;
  HistMatrix eval_hist;
  GpuHistogramWorkspacePtr gpu_hist_workspace{nullptr, DestroyGpuHistogramWorkspace};
  std::vector<float> predictions;
  std::vector<float> eval_predictions;
  std::vector<float> gradients;
  std::vector<float> hessians;

  std::size_t train_hist_bytes() const noexcept { return train_hist.storage_bytes(); }
  std::size_t eval_hist_bytes() const noexcept { return eval_hist.storage_bytes(); }
  std::size_t gpu_workspace_bytes() const noexcept {
    return EstimateGpuHistogramWorkspaceBytes(gpu_hist_workspace.get());
  }

  void ReleaseFitScratch() noexcept {
    predictions.clear();
    predictions.shrink_to_fit();
    eval_predictions.clear();
    eval_predictions.shrink_to_fit();
    gradients.clear();
    gradients.shrink_to_fit();
    hessians.clear();
    hessians.shrink_to_fit();
    eval_hist.ReleaseStorage();
    train_hist.ReleaseStorage();
    gpu_hist_workspace.reset();
  }
};

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

}  // namespace

GradientBooster::GradientBooster(std::string objective,
                                 int iterations,
                                 double learning_rate,
                                 int max_depth,
                                 double alpha,
                                 double lambda_l2,
                                 double subsample,
                                 std::string bootstrap_type,
                                 double bagging_temperature,
                                 std::string boosting_type,
                                 double drop_rate,
                                 double skip_drop,
                                 int max_drop,
                                 std::vector<int> monotone_constraints,
                                 std::vector<std::vector<int>> interaction_constraints,
                                 double colsample_bytree,
                                 std::vector<double> feature_weights,
                                 std::vector<double> first_feature_use_penalties,
                                 double random_strength,
                                 std::string grow_policy,
                                 int max_leaves,
                                 int min_samples_split,
                                 int min_data_in_leaf,
                                 double min_child_weight,
                                 double gamma,
                                 double max_leaf_weight,
                                 int num_classes,
                                 std::size_t max_bins,
                                 std::string nan_mode,
                                 std::vector<std::uint16_t> max_bin_by_feature,
                                 std::string border_selection_method,
                                 std::vector<std::string> nan_mode_by_feature,
                                 std::vector<std::vector<float>> feature_borders,
                                 bool external_memory,
                                 std::string external_memory_dir,
                                 std::string eval_metric,
                                 double quantile_alpha,
                                 double huber_delta,
                                 double tweedie_variance_power,
                                 std::string task_type,
                                 std::string devices,
                                 int distributed_world_size,
                                 int distributed_rank,
                                 std::string distributed_root,
                                 std::string distributed_run_id,
                                 double distributed_timeout,
                                 std::uint64_t random_seed,
                                 bool verbose)
    : objective_name_(std::move(objective)),
      eval_metric_name_(std::move(eval_metric)),
      objective_config_{huber_delta, quantile_alpha, tweedie_variance_power},
      objective_(CreateObjectiveFunction(objective_name_, objective_config_)),
      objective_metric_(CreateMetricFunctionForObjective(objective_name_, objective_config_)),
      iterations_(iterations),
      learning_rate_(learning_rate),
      max_depth_(max_depth),
      alpha_(alpha),
      lambda_l2_(lambda_l2),
      subsample_(subsample),
      bootstrap_type_(CanonicalBootstrapType(std::move(bootstrap_type))),
      bagging_temperature_(bagging_temperature),
      boosting_type_(CanonicalBoostingType(std::move(boosting_type))),
      drop_rate_(drop_rate),
      skip_drop_(skip_drop),
      max_drop_(max_drop),
      monotone_constraints_(std::move(monotone_constraints)),
      interaction_constraints_(std::move(interaction_constraints)),
      colsample_bytree_(colsample_bytree),
      feature_weights_(std::move(feature_weights)),
      first_feature_use_penalties_(std::move(first_feature_use_penalties)),
      random_strength_(random_strength),
      grow_policy_(CanonicalGrowPolicy(std::move(grow_policy))),
      max_leaves_(max_leaves),
      min_samples_split_(min_samples_split),
      min_data_in_leaf_(min_data_in_leaf),
      min_child_weight_(min_child_weight),
      gamma_(gamma),
      max_leaf_weight_(max_leaf_weight),
      num_classes_(num_classes),
      max_bins_(max_bins),
      external_memory_(external_memory),
      external_memory_dir_(std::move(external_memory_dir)),
      devices_(std::move(devices)),
      distributed_world_size_(distributed_world_size),
      distributed_rank_(distributed_rank),
      distributed_root_(std::move(distributed_root)),
      distributed_run_id_(std::move(distributed_run_id)),
      distributed_timeout_(distributed_timeout),
      random_seed_(random_seed),
      rng_state_(NormalizeRngState(random_seed)),
      verbose_(TrainingProfiler::ResolveEnabled(verbose)),
      hist_builder_(max_bins_,
                    std::move(nan_mode),
                    std::move(max_bin_by_feature),
                    std::move(border_selection_method),
                    std::move(nan_mode_by_feature),
                    std::move(feature_borders),
                    external_memory_,
                    external_memory_dir_) {
  if (eval_metric_name_.empty()) {
    eval_metric_name_ = objective_name_;
  }
  eval_metric_ = CreateMetricFunction(eval_metric_name_, objective_config_);
  maximize_eval_metric_ = eval_metric_->HigherIsBetter();

  if (iterations_ <= 0) {
    throw std::invalid_argument("iterations must be positive");
  }
  if (learning_rate_ <= 0.0) {
    throw std::invalid_argument("learning_rate must be positive");
  }
  if (max_depth_ < 0) {
    throw std::invalid_argument("max_depth must be non-negative");
  }
  if (lambda_l2_ < 0.0) {
    throw std::invalid_argument("lambda_l2 must be non-negative");
  }
  if (subsample_ <= 0.0 || subsample_ > 1.0) {
    throw std::invalid_argument("subsample must be in (0, 1]");
  }
  if (bagging_temperature_ < 0.0) {
    throw std::invalid_argument("bagging_temperature must be non-negative");
  }
  if (drop_rate_ < 0.0 || drop_rate_ > 1.0) {
    throw std::invalid_argument("drop_rate must be in [0, 1]");
  }
  if (skip_drop_ < 0.0 || skip_drop_ > 1.0) {
    throw std::invalid_argument("skip_drop must be in [0, 1]");
  }
  if (max_drop_ < 0) {
    throw std::invalid_argument("max_drop must be non-negative");
  }
  if (colsample_bytree_ <= 0.0 || colsample_bytree_ > 1.0) {
    throw std::invalid_argument("colsample_bytree must be in (0, 1]");
  }
  if (max_leaves_ < 0) {
    throw std::invalid_argument("max_leaves must be non-negative");
  }
  if (min_samples_split_ < 2) {
    throw std::invalid_argument("min_samples_split must be at least 2");
  }
  if (min_data_in_leaf_ < 0) {
    throw std::invalid_argument("min_data_in_leaf must be non-negative");
  }
  if (min_child_weight_ < 0.0) {
    throw std::invalid_argument("min_child_weight must be non-negative");
  }
  if (gamma_ < 0.0) {
    throw std::invalid_argument("gamma must be non-negative");
  }
  if (max_leaf_weight_ < 0.0) {
    throw std::invalid_argument("max_leaf_weight must be non-negative");
  }
  if (random_strength_ < 0.0) {
    throw std::invalid_argument("random_strength must be non-negative");
  }
  if (distributed_world_size_ <= 0) {
    throw std::invalid_argument("distributed_world_size must be positive");
  }
  if (distributed_rank_ < 0 || distributed_rank_ >= distributed_world_size_) {
    throw std::invalid_argument("distributed_rank must be in [0, distributed_world_size)");
  }
  if (distributed_timeout_ <= 0.0) {
    throw std::invalid_argument("distributed_timeout must be positive");
  }
  (void)ParseBootstrapType(bootstrap_type_);
  (void)ParseBoostingType(boosting_type_);
  (void)ParseGrowPolicy(grow_policy_);
  if (num_classes_ <= 0) {
    throw std::invalid_argument("num_classes must be positive");
  }
  for (const double value : feature_weights_) {
    if (value < 0.0) {
      throw std::invalid_argument("feature_weights entries must be non-negative");
    }
  }
  for (const double value : first_feature_use_penalties_) {
    if (value < 0.0) {
      throw std::invalid_argument("first_feature_use_penalties entries must be non-negative");
    }
  }

  const std::string normalized_objective = NormalizeToken(objective_name_);
  if (IsMulticlassObjective(normalized_objective)) {
    if (num_classes_ <= 2) {
      throw std::invalid_argument("multiclass objective requires num_classes greater than two");
    }
    prediction_dimension_ = num_classes_;
  } else if (IsRankingObjective(normalized_objective)) {
    if (num_classes_ != 1) {
      throw std::invalid_argument("ranking objectives require num_classes equal to one");
    }
    prediction_dimension_ = 1;
  } else if (IsRegressionObjective(normalized_objective)) {
    if (num_classes_ != 1) {
      throw std::invalid_argument("regression objectives require num_classes equal to one");
    }
    prediction_dimension_ = 1;
  } else if (IsBinaryObjective(normalized_objective)) {
    if (num_classes_ != 1 && num_classes_ != 2) {
      throw std::invalid_argument("binary objectives require num_classes equal to one or two");
    }
    prediction_dimension_ = 1;
  }

  const std::string normalized_task_type = NormalizeTaskType(std::move(task_type));
  if (normalized_task_type == "cpu") {
    use_gpu_ = false;
  } else if (normalized_task_type == "gpu") {
    if (!CudaBackendCompiled()) {
      throw std::runtime_error(
          "task_type='GPU' was requested but CTBoost was compiled without CUDA support");
    }
    use_gpu_ = true;
  } else {
    throw std::invalid_argument("task_type must be either 'CPU' or 'GPU'");
  }
  if (use_gpu_ && distributed_world_size_ > 1 &&
      !DistributedRootUsesTcp(distributed_root_)) {
    throw std::invalid_argument(
        "distributed GPU training requires distributed_root to use a tcp://host:port coordinator");
  }
}

void GradientBooster::Fit(Pool& pool,
                          Pool* eval_pool,
                          int early_stopping_rounds,
                          bool continue_training) {
  const auto fit_start = std::chrono::steady_clock::now();
  const TrainingProfiler profiler(verbose_);
  profiler.LogFitStart(
      pool.num_rows(), pool.num_cols(), iterations_, use_gpu_, prediction_dimension_);
  if (early_stopping_rounds < 0) {
    early_stopping_rounds = 0;
  }
  if (early_stopping_rounds > 0 && eval_pool == nullptr) {
    throw std::invalid_argument("early_stopping_rounds requires eval_pool");
  }
  if (eval_pool != nullptr) {
    if (eval_pool->num_cols() != pool.num_cols()) {
      throw std::invalid_argument(
          "eval_pool must have the same number of columns as the training pool");
    }
    if (!SameCategoricalFeatures(pool, *eval_pool)) {
      throw std::invalid_argument(
          "eval_pool categorical feature indices must match the training pool");
    }
  }
  if (!monotone_constraints_.empty()) {
    if (monotone_constraints_.size() != pool.num_cols()) {
      throw std::invalid_argument(
          "monotone_constraints must have one entry per feature when provided");
    }
    for (const int feature_id : pool.cat_features()) {
      if (feature_id >= 0 && static_cast<std::size_t>(feature_id) < monotone_constraints_.size() &&
          monotone_constraints_[static_cast<std::size_t>(feature_id)] != 0) {
        throw std::invalid_argument(
            "monotone_constraints can only be applied to numeric features");
      }
    }
    if (prediction_dimension_ != 1) {
      throw std::invalid_argument(
          "monotone_constraints are only supported for single-output objectives");
    }
  }
  if (!interaction_constraints_.empty()) {
    for (const auto& group : interaction_constraints_) {
      if (group.empty()) {
        continue;
      }
      for (const int feature_id : group) {
        if (feature_id < 0 || static_cast<std::size_t>(feature_id) >= pool.num_cols()) {
          throw std::invalid_argument(
              "interaction_constraints feature index is out of bounds");
        }
      }
    }
  }
  if (!feature_weights_.empty() && feature_weights_.size() != pool.num_cols()) {
    throw std::invalid_argument("feature_weights must have one entry per feature when provided");
  }
  if (!first_feature_use_penalties_.empty() &&
      first_feature_use_penalties_.size() != pool.num_cols()) {
    throw std::invalid_argument(
        "first_feature_use_penalties must have one entry per feature when provided");
  }
  if (!feature_weights_.empty() &&
      std::none_of(feature_weights_.begin(), feature_weights_.end(), [](double value) {
        return value > 0.0;
      })) {
    throw std::invalid_argument("feature_weights must leave at least one feature with positive weight");
  }

  const InteractionConstraintSet interaction_constraint_set =
      BuildInteractionConstraintSet(interaction_constraints_, pool.num_cols());
  const InteractionConstraintSet* interaction_constraint_ptr =
      interaction_constraint_set.groups.empty() ? nullptr : &interaction_constraint_set;

  const bool has_existing_state = continue_training && !trees_.empty();
  const QuantizationSchemaPtr imported_quantization_schema =
      has_existing_state ? QuantizationSchemaPtr{} : quantization_schema_;
  if (!continue_training) {
    trees_.clear();
    quantization_schema_ = imported_quantization_schema;
    loss_history_.clear();
    eval_loss_history_.clear();
    best_iteration_ = -1;
    best_score_ = 0.0;
  } else if (!feature_importance_sums_.empty() &&
             feature_importance_sums_.size() != pool.num_cols()) {
    throw std::invalid_argument(
        "warm-start training requires the same number of features as the initial model");
  }

  FitWorkspace workspace;
  LogFitMemorySnapshot(profiler, "pre_quantize", pool, eval_pool, workspace);

  const auto hist_build_start = std::chrono::steady_clock::now();
  if (has_existing_state || quantization_schema_ != nullptr) {
    workspace.train_hist = BuildPredictionHist(pool, RequireQuantizationSchema(quantization_schema_));
    if (external_memory_) {
      workspace.train_hist.SpillBinStorage(external_memory_dir_);
    }
  } else {
    workspace.train_hist = hist_builder_.Build(pool, &profiler);
    quantization_schema_ =
        std::make_shared<QuantizationSchema>(MakeQuantizationSchema(workspace.train_hist));
  }
  const double hist_build_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - hist_build_start)
          .count();
  profiler.LogFitStage("train_quantize", hist_build_ms);
  LogFitMemorySnapshot(profiler, "post_quantize", pool, eval_pool, workspace);
  const auto& labels = pool.labels();
  const auto& weights = pool.weights();
  if (use_gpu_) {
    workspace.gpu_hist_workspace = CreateGpuHistogramWorkspace(workspace.train_hist, weights, devices_);
  }
  LogFitMemorySnapshot(profiler, "post_workspace", pool, eval_pool, workspace);
  const RankingMetadataView ranking = pool.ranking_metadata();
  const RankingMetadataView* ranking_ptr =
      ranking.group_ids == nullptr && ranking.subgroup_ids == nullptr &&
              ranking.group_weights == nullptr && ranking.pairs == nullptr
          ? nullptr
          : &ranking;
  const double total_weight = std::accumulate(
      weights.begin(), weights.end(), 0.0,
      [](double acc, float value) { return acc + static_cast<double>(value); });
  if (total_weight <= 0.0) {
    throw std::invalid_argument("training pool must have a positive total sample weight");
  }
  workspace.predictions = has_existing_state
                              ? PredictFromHist(trees_,
                                                workspace.train_hist,
                                                trees_.size(),
                                                learning_rate_,
                                                use_gpu_,
                                                prediction_dimension_,
                                                devices_)
                              : std::vector<float>(
                                    pool.num_rows() *
                                        static_cast<std::size_t>(prediction_dimension_),
                                    0.0F);
  AddPoolBaselineToPredictions(pool, prediction_dimension_, workspace.predictions);
  if (use_gpu_) {
    workspace.train_hist.ReleaseBinStorage();
    LogFitMemorySnapshot(profiler, "post_gpu_train_bin_release", pool, eval_pool, workspace);
  }
  if (!continue_training || feature_importance_sums_.empty()) {
    feature_importance_sums_.assign(pool.num_cols(), 0.0);
  }
  std::vector<std::uint8_t> model_feature_used_mask(pool.num_cols(), 0U);
  if (has_existing_state) {
    for (const Tree& tree : trees_) {
      MarkUsedFeatures(tree, model_feature_used_mask);
    }
  }
  const int initial_completed_iterations = static_cast<int>(num_iterations_trained());
  int completed_iterations = initial_completed_iterations;
  const int target_total_iterations = initial_completed_iterations + iterations_;
  bool early_stopped = false;
  const BoostingType boosting_type = ParseBoostingType(boosting_type_);
  const BootstrapType configured_bootstrap_type = ParseBootstrapType(bootstrap_type_);
  std::vector<float> fixed_gradient_predictions;
  if (boosting_type == BoostingType::kRandomForest) {
    fixed_gradient_predictions = workspace.predictions;
  }
  const bool use_dart = boosting_type == BoostingType::kDart;

  const std::vector<float>* eval_labels = nullptr;
  const std::vector<float>* eval_weights = nullptr;
  RankingMetadataView eval_ranking;
  const RankingMetadataView* eval_ranking_ptr = nullptr;
  if (eval_pool != nullptr) {
    eval_labels = &eval_pool->labels();
    eval_weights = &eval_pool->weights();
    eval_ranking = eval_pool->ranking_metadata();
    eval_ranking_ptr =
        eval_ranking.group_ids == nullptr && eval_ranking.subgroup_ids == nullptr &&
                eval_ranking.group_weights == nullptr && eval_ranking.pairs == nullptr
            ? nullptr
            : &eval_ranking;
    const double eval_total_weight = std::accumulate(
        eval_weights->begin(), eval_weights->end(), 0.0,
        [](double acc, float value) { return acc + static_cast<double>(value); });
    if (eval_total_weight <= 0.0) {
      throw std::invalid_argument("eval_pool must have a positive total sample weight");
    }
    const auto eval_hist_build_start = std::chrono::steady_clock::now();
    workspace.eval_hist = BuildPredictionHist(*eval_pool, RequireQuantizationSchema(quantization_schema_));
    if (external_memory_) {
      workspace.eval_hist.SpillBinStorage(external_memory_dir_);
    }
    const double eval_hist_build_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - eval_hist_build_start)
            .count();
    profiler.LogFitStage("eval_quantize", eval_hist_build_ms);
    workspace.eval_predictions = has_existing_state
                                     ? PredictFromHist(trees_,
                                                       workspace.eval_hist,
                                                       trees_.size(),
                                                       learning_rate_,
                                                       use_gpu_,
                                                       prediction_dimension_,
                                                       devices_)
                                     : std::vector<float>(
                                           eval_pool->num_rows() *
                                               static_cast<std::size_t>(prediction_dimension_),
                                           0.0F);
    AddPoolBaselineToPredictions(*eval_pool, prediction_dimension_, workspace.eval_predictions);
    LogFitMemorySnapshot(profiler, "post_eval_quantize", pool, eval_pool, workspace);
    if (!continue_training || eval_loss_history_.empty()) {
      best_score_ = maximize_eval_metric_ ? -std::numeric_limits<double>::infinity()
                                          : std::numeric_limits<double>::infinity();
      if (!continue_training) {
        best_iteration_ = -1;
      }
    }
  }
  if (distributed_world_size_ > 1 &&
      ((ranking_ptr != nullptr &&
        (ranking.subgroup_ids != nullptr || ranking.group_weights != nullptr || ranking.pairs != nullptr)) ||
       (eval_ranking_ptr != nullptr &&
        (eval_ranking.subgroup_ids != nullptr || eval_ranking.group_weights != nullptr ||
         eval_ranking.pairs != nullptr)))) {
    throw std::invalid_argument(
        "distributed training does not yet support subgroup_id, group_weight, or pairs metadata");
  }

  pool.ReleaseFeatureStorage();
  if (eval_pool != nullptr) {
    eval_pool->ReleaseFeatureStorage();
  }
  LogFitMemorySnapshot(profiler, "post_release_dense", pool, eval_pool, workspace);

  for (int iteration = 0; iteration < iterations_; ++iteration) {
    const auto iteration_start = std::chrono::steady_clock::now();
    const int total_iteration = initial_completed_iterations + iteration;
    DistributedCoordinator distributed_coordinator;
    if (distributed_world_size_ > 1) {
      distributed_coordinator.world_size = distributed_world_size_;
      distributed_coordinator.rank = distributed_rank_;
      distributed_coordinator.root = distributed_root_;
      distributed_coordinator.run_id = distributed_run_id_;
      distributed_coordinator.timeout_seconds = distributed_timeout_;
      distributed_coordinator.tree_index = static_cast<std::size_t>(total_iteration);
      distributed_coordinator.operation_counter = 0;
    }
    BootstrapType iteration_bootstrap_type = configured_bootstrap_type;
    if (boosting_type == BoostingType::kRandomForest &&
        iteration_bootstrap_type == BootstrapType::kNone && subsample_ >= 1.0) {
      iteration_bootstrap_type = BootstrapType::kPoisson;
    } else if (iteration_bootstrap_type == BootstrapType::kNone && subsample_ < 1.0) {
      iteration_bootstrap_type = BootstrapType::kBernoulli;
    }
    const std::vector<float> iteration_weights =
        SampleRowWeights(
            weights, subsample_, iteration_bootstrap_type, bagging_temperature_, rng_state_);
    std::vector<float> dart_gradient_predictions;
    std::vector<float> dropped_train_predictions;
    std::vector<float> dropped_eval_predictions;
    std::vector<std::size_t> dropped_iterations;
    if (use_dart && !trees_.empty()) {
      dropped_iterations = SampleDroppedTreeGroups(
          num_iterations_trained(), drop_rate_, skip_drop_, max_drop_, rng_state_);
      if (!dropped_iterations.empty()) {
        dart_gradient_predictions = workspace.predictions;
        dropped_train_predictions.assign(dart_gradient_predictions.size(), 0.0F);
        if (eval_pool != nullptr) {
          dropped_eval_predictions.assign(workspace.eval_predictions.size(), 0.0F);
        }
        for (const std::size_t dropped_iteration : dropped_iterations) {
          AccumulateIterationPredictions(
              trees_,
              dropped_iteration,
              workspace.train_hist,
              learning_rate_,
              prediction_dimension_,
              dropped_train_predictions);
          if (eval_pool != nullptr) {
            AccumulateIterationPredictions(
                trees_,
                dropped_iteration,
                workspace.eval_hist,
                learning_rate_,
                prediction_dimension_,
                dropped_eval_predictions);
          }
        }
        for (std::size_t index = 0; index < dart_gradient_predictions.size(); ++index) {
          dart_gradient_predictions[index] -= dropped_train_predictions[index];
        }
      }
    }
    const auto gradient_start = std::chrono::steady_clock::now();
    const std::vector<float>& gradient_predictions =
        boosting_type == BoostingType::kRandomForest ? fixed_gradient_predictions
        : (use_dart && !dart_gradient_predictions.empty()) ? dart_gradient_predictions
                                                           : workspace.predictions;
    objective_->compute_gradients(gradient_predictions,
                                  labels,
                                  workspace.gradients,
                                  workspace.hessians,
                                  num_classes_,
                                  ranking_ptr);
    const double gradient_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - gradient_start)
            .count();
    double tree_ms = 0.0;
    double prediction_ms = 0.0;
    const double dropped_tree_scale =
        dropped_iterations.empty() ? 1.0
                                   : static_cast<double>(dropped_iterations.size()) /
                                         static_cast<double>(dropped_iterations.size() + 1U);
    const double new_tree_scale =
        dropped_iterations.empty() ? 1.0
                                   : 1.0 / static_cast<double>(dropped_iterations.size() + 1U);

    if (prediction_dimension_ == 1) {
      if (use_gpu_) {
        UploadHistogramTargetsGpu(
            workspace.gpu_hist_workspace.get(), workspace.gradients, workspace.hessians);
        UploadHistogramWeightsGpu(workspace.gpu_hist_workspace.get(), iteration_weights);
        UploadFeatureControlsGpu(workspace.gpu_hist_workspace.get(),
                                 feature_weights_.empty() ? nullptr : &feature_weights_,
                                 first_feature_use_penalties_.empty()
                                     ? nullptr
                                     : &first_feature_use_penalties_,
                                 first_feature_use_penalties_.empty()
                                     ? nullptr
                                     : &model_feature_used_mask,
                                 monotone_constraints_.empty() ? nullptr : &monotone_constraints_);
      }
      Tree tree;
      const std::vector<int> allowed_features =
          SampleFeatureSubset(pool.num_cols(), colsample_bytree_, &feature_weights_, rng_state_);
      const TreeBuildOptions build_options{
          alpha_,
          max_depth_,
          lambda_l2_,
          use_gpu_,
          ParseGrowPolicy(grow_policy_),
          max_leaves_,
          min_samples_split_,
          min_data_in_leaf_,
          min_child_weight_,
          gamma_,
          max_leaf_weight_,
          random_strength_,
          rng_state_,
          allowed_features.empty() ? nullptr : &allowed_features,
          feature_weights_.empty() ? nullptr : &feature_weights_,
          first_feature_use_penalties_.empty() ? nullptr : &first_feature_use_penalties_,
          first_feature_use_penalties_.empty() ? nullptr : &model_feature_used_mask,
          monotone_constraints_.empty() ? nullptr : &monotone_constraints_,
          interaction_constraint_ptr,
          distributed_world_size_ > 1 ? &distributed_coordinator : nullptr,
      };
      std::vector<std::size_t> training_row_indices;
      std::vector<LeafRowRange> training_leaf_ranges;
      const auto tree_start = std::chrono::steady_clock::now();
      tree.Build(
          workspace.train_hist,
          workspace.gradients,
          workspace.hessians,
          iteration_weights,
          build_options,
          workspace.gpu_hist_workspace.get(),
          &profiler,
          &training_row_indices,
          &training_leaf_ranges,
          quantization_schema_);
      const double single_tree_ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tree_start)
              .count();
      tree_ms += single_tree_ms;
      profiler.LogTreeBuild(
          total_iteration + 1, target_total_iterations, 0, prediction_dimension_, single_tree_ms);

      const auto prediction_start = std::chrono::steady_clock::now();
      if (!dropped_iterations.empty()) {
        for (const std::size_t dropped_iteration : dropped_iterations) {
          ScaleTreeLeafWeights(trees_[dropped_iteration], dropped_tree_scale);
        }
        const float adjustment_scale = static_cast<float>(dropped_tree_scale - 1.0);
        for (std::size_t index = 0; index < workspace.predictions.size(); ++index) {
          workspace.predictions[index] += adjustment_scale * dropped_train_predictions[index];
        }
        if (eval_pool != nullptr) {
          for (std::size_t index = 0; index < workspace.eval_predictions.size(); ++index) {
            workspace.eval_predictions[index] +=
                adjustment_scale * dropped_eval_predictions[index];
          }
        }
      }
      if (new_tree_scale != 1.0) {
        ScaleTreeLeafWeights(tree, new_tree_scale);
      }
      UpdatePredictionsFromLeafRanges(
          tree,
          training_row_indices,
          training_leaf_ranges,
          learning_rate_ * new_tree_scale,
          prediction_dimension_,
          0,
          workspace.predictions);
      if (eval_pool != nullptr) {
        UpdatePredictions(
            tree,
            workspace.eval_hist,
            learning_rate_ * new_tree_scale,
            prediction_dimension_,
            0,
            workspace.eval_predictions);
      }
      prediction_ms +=
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - prediction_start)
              .count();
      AccumulateFeatureImportances(tree, feature_importance_sums_);
      MarkUsedFeatures(tree, model_feature_used_mask);

      trees_.push_back(std::move(tree));
    } else {
      std::vector<float> structure_gradients;
      std::vector<float> structure_hessians;
      BuildSharedMulticlassTargets(workspace.gradients,
                                   workspace.hessians,
                                   iteration_weights,
                                   pool.num_rows(),
                                   prediction_dimension_,
                                   structure_gradients,
                                   structure_hessians);
      if (use_gpu_) {
        UploadHistogramTargetsGpu(
            workspace.gpu_hist_workspace.get(), structure_gradients, structure_hessians);
        UploadHistogramWeightsGpu(workspace.gpu_hist_workspace.get(), iteration_weights);
        UploadFeatureControlsGpu(workspace.gpu_hist_workspace.get(),
                                 feature_weights_.empty() ? nullptr : &feature_weights_,
                                 first_feature_use_penalties_.empty()
                                     ? nullptr
                                     : &first_feature_use_penalties_,
                                 first_feature_use_penalties_.empty()
                                     ? nullptr
                                     : &model_feature_used_mask,
                                 monotone_constraints_.empty() ? nullptr : &monotone_constraints_);
      }

      Tree structure_tree;
      const std::vector<int> allowed_features =
          SampleFeatureSubset(pool.num_cols(), colsample_bytree_, &feature_weights_, rng_state_);
      const TreeBuildOptions build_options{
          alpha_,
          max_depth_,
          lambda_l2_,
          use_gpu_,
          ParseGrowPolicy(grow_policy_),
          max_leaves_,
          min_samples_split_,
          min_data_in_leaf_,
          min_child_weight_,
          gamma_,
          max_leaf_weight_,
          random_strength_,
          rng_state_,
          allowed_features.empty() ? nullptr : &allowed_features,
          feature_weights_.empty() ? nullptr : &feature_weights_,
          first_feature_use_penalties_.empty() ? nullptr : &first_feature_use_penalties_,
          first_feature_use_penalties_.empty() ? nullptr : &model_feature_used_mask,
          monotone_constraints_.empty() ? nullptr : &monotone_constraints_,
          interaction_constraint_ptr,
          distributed_world_size_ > 1 ? &distributed_coordinator : nullptr,
      };
      std::vector<std::size_t> training_row_indices;
      std::vector<LeafRowRange> training_leaf_ranges;
      const auto tree_start = std::chrono::steady_clock::now();
      structure_tree.Build(workspace.train_hist,
                           structure_gradients,
                           structure_hessians,
                           iteration_weights,
                           build_options,
                           workspace.gpu_hist_workspace.get(),
                           &profiler,
                           &training_row_indices,
                           &training_leaf_ranges,
                           quantization_schema_);
      const double shared_tree_ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tree_start)
              .count();
      tree_ms += shared_tree_ms;
      profiler.LogTreeBuild(
          total_iteration + 1, target_total_iterations, -1, prediction_dimension_, shared_tree_ms);

      const auto leaf_fit_start = std::chrono::steady_clock::now();
      std::vector<Tree> class_trees = MaterializeMulticlassTreesFromStructure(
          structure_tree,
          training_row_indices,
          training_leaf_ranges,
          workspace.gradients,
          workspace.hessians,
          iteration_weights,
          prediction_dimension_,
          lambda_l2_);
      tree_ms +=
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - leaf_fit_start)
              .count();

      std::vector<int> eval_leaf_indices;
      if (eval_pool != nullptr) {
        eval_leaf_indices = PredictLeafIndicesFromHist(structure_tree, workspace.eval_hist);
      }

      const auto prediction_start = std::chrono::steady_clock::now();
      if (!dropped_iterations.empty()) {
        for (const std::size_t dropped_iteration : dropped_iterations) {
          for (int class_index = 0; class_index < prediction_dimension_; ++class_index) {
            const std::size_t tree_index =
                dropped_iteration * static_cast<std::size_t>(prediction_dimension_) +
                static_cast<std::size_t>(class_index);
            ScaleTreeLeafWeights(trees_[tree_index], dropped_tree_scale);
          }
        }
        const float adjustment_scale = static_cast<float>(dropped_tree_scale - 1.0);
        for (std::size_t index = 0; index < workspace.predictions.size(); ++index) {
          workspace.predictions[index] += adjustment_scale * dropped_train_predictions[index];
        }
        if (eval_pool != nullptr) {
          for (std::size_t index = 0; index < workspace.eval_predictions.size(); ++index) {
            workspace.eval_predictions[index] +=
                adjustment_scale * dropped_eval_predictions[index];
          }
        }
      }
      for (int class_index = 0; class_index < prediction_dimension_; ++class_index) {
        Tree& tree = class_trees[static_cast<std::size_t>(class_index)];
        if (new_tree_scale != 1.0) {
          ScaleTreeLeafWeights(tree, new_tree_scale);
        }
        UpdatePredictionsFromLeafRanges(tree,
                                        training_row_indices,
                                        training_leaf_ranges,
                                        learning_rate_ * new_tree_scale,
                                        prediction_dimension_,
                                        class_index,
                                        workspace.predictions);
        if (eval_pool != nullptr) {
          UpdatePredictionsFromLeafIndices(
              tree,
              eval_leaf_indices,
              learning_rate_ * new_tree_scale,
              prediction_dimension_,
              class_index,
              workspace.eval_predictions);
        }
        AccumulateFeatureImportances(tree, feature_importance_sums_);
        MarkUsedFeatures(tree, model_feature_used_mask);
        trees_.push_back(std::move(tree));
      }
      prediction_ms +=
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - prediction_start)
              .count();
    }

    const bool distributed_tcp =
        distributed_world_size_ > 1 && DistributedRootUsesTcp(distributed_root_);
    const auto metric_start = std::chrono::steady_clock::now();
    double local_train_loss = 0.0;
    if (distributed_tcp) {
      DistributedMetricInputs local_train_inputs;
      local_train_inputs.predictions = workspace.predictions;
      local_train_inputs.labels = labels;
      local_train_inputs.weights = weights;
      local_train_inputs.has_group_ids = ranking_ptr != nullptr && ranking_ptr->group_ids != nullptr;
      if (local_train_inputs.has_group_ids) {
        local_train_inputs.group_ids = *ranking_ptr->group_ids;
      }
      const DistributedMetricInputs gathered_train_inputs = AllGatherDistributedMetricInputs(
          &distributed_coordinator, "train_metric", local_train_inputs);
      const RankingMetadataView gathered_train_ranking{
          gathered_train_inputs.has_group_ids ? &gathered_train_inputs.group_ids : nullptr,
          nullptr,
          nullptr,
          nullptr,
      };
      local_train_loss = objective_metric_->Evaluate(
          gathered_train_inputs.predictions,
          gathered_train_inputs.labels,
          gathered_train_inputs.weights,
          num_classes_,
          gathered_train_inputs.has_group_ids ? &gathered_train_ranking : nullptr);
    } else {
      local_train_loss =
          objective_metric_->Evaluate(
              workspace.predictions, labels, weights, num_classes_, ranking_ptr);
    }
    const double metric_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - metric_start)
            .count();
    completed_iterations = total_iteration + 1;
    if (eval_pool == nullptr) {
      if (distributed_tcp) {
        DistributedMetricControl root_control;
        if (distributed_rank_ == 0) {
          root_control.train_loss = local_train_loss;
        }
        const DistributedMetricControl synced_control = BroadcastDistributedMetricControl(
            &distributed_coordinator, "metric", distributed_rank_ == 0 ? &root_control : nullptr);
        loss_history_.push_back(synced_control.train_loss);
      } else {
        loss_history_.push_back(local_train_loss);
      }
      const double iteration_ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - iteration_start)
              .count();
      profiler.LogIteration(total_iteration + 1,
                            target_total_iterations,
                            gradient_ms,
                            tree_ms,
                            prediction_ms,
                            metric_ms,
                            0.0,
                            iteration_ms);
      continue;
    }

    const auto eval_start = std::chrono::steady_clock::now();
    double local_eval_score = 0.0;
    if (distributed_tcp) {
      DistributedMetricInputs local_eval_inputs;
      local_eval_inputs.predictions = workspace.eval_predictions;
      local_eval_inputs.labels = *eval_labels;
      local_eval_inputs.weights = *eval_weights;
      local_eval_inputs.has_group_ids = eval_ranking_ptr != nullptr && eval_ranking_ptr->group_ids != nullptr;
      if (local_eval_inputs.has_group_ids) {
        local_eval_inputs.group_ids = *eval_ranking_ptr->group_ids;
      }
      const DistributedMetricInputs gathered_eval_inputs = AllGatherDistributedMetricInputs(
          &distributed_coordinator, "eval_metric", local_eval_inputs);
      const RankingMetadataView gathered_eval_ranking{
          gathered_eval_inputs.has_group_ids ? &gathered_eval_inputs.group_ids : nullptr,
          nullptr,
          nullptr,
          nullptr,
      };
      local_eval_score = eval_metric_->Evaluate(
          gathered_eval_inputs.predictions,
          gathered_eval_inputs.labels,
          gathered_eval_inputs.weights,
          num_classes_,
          gathered_eval_inputs.has_group_ids ? &gathered_eval_ranking : nullptr);
    } else {
      local_eval_score =
          eval_metric_->Evaluate(
              workspace.eval_predictions, *eval_labels, *eval_weights, num_classes_, eval_ranking_ptr);
    }
    const double eval_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - eval_start)
            .count();
    double iteration_train_loss = local_train_loss;
    double iteration_eval_score = local_eval_score;
    if (distributed_tcp) {
      DistributedMetricControl root_control;
      if (distributed_rank_ == 0) {
        root_control.train_loss = local_train_loss;
        root_control.eval_score = local_eval_score;
        root_control.has_eval = 1U;
        const bool improved =
            best_iteration_ < 0 ||
            (maximize_eval_metric_ ? local_eval_score > best_score_ : local_eval_score < best_score_);
        if (improved) {
          best_iteration_ = total_iteration;
          best_score_ = local_eval_score;
        }
        root_control.best_iteration = best_iteration_;
        root_control.best_score = best_score_;
        root_control.should_stop =
            !improved && early_stopping_rounds > 0 &&
                    total_iteration - best_iteration_ >= early_stopping_rounds
                ? 1U
                : 0U;
      }
      const DistributedMetricControl synced_control = BroadcastDistributedMetricControl(
          &distributed_coordinator, "metric", distributed_rank_ == 0 ? &root_control : nullptr);
      iteration_train_loss = synced_control.train_loss;
      if (synced_control.has_eval != 0U) {
        iteration_eval_score = synced_control.eval_score;
        best_iteration_ = synced_control.best_iteration;
        best_score_ = synced_control.best_score;
      }
      early_stopped = synced_control.should_stop != 0U;
    } else {
      const bool improved =
          best_iteration_ < 0 ||
          (maximize_eval_metric_ ? local_eval_score > best_score_ : local_eval_score < best_score_);
      if (improved) {
        best_iteration_ = total_iteration;
        best_score_ = local_eval_score;
      } else if (early_stopping_rounds > 0 &&
                 total_iteration - best_iteration_ >= early_stopping_rounds) {
        early_stopped = true;
      }
    }
    loss_history_.push_back(iteration_train_loss);
    eval_loss_history_.push_back(iteration_eval_score);
    const double iteration_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - iteration_start)
            .count();
    profiler.LogIteration(total_iteration + 1,
                          target_total_iterations,
                          gradient_ms,
                          tree_ms,
                          prediction_ms,
                          metric_ms,
                          eval_ms,
                          iteration_ms);
    if (!early_stopped &&
        (best_iteration_ == total_iteration ||
         (best_iteration_ < 0 && distributed_tcp == false))) {
      continue;
    }

    if (early_stopped) {
      break;
    }
  }

  auto finish_fit = [&](double total_fit_ms) {
    const auto cleanup_start = std::chrono::steady_clock::now();
    workspace.ReleaseFitScratch();
    const double cleanup_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cleanup_start)
            .count();
    profiler.LogFitStage("cleanup", cleanup_ms);
    LogFitMemorySnapshot(profiler, "post_cleanup", pool, eval_pool, workspace);
    profiler.LogFitSummary(hist_build_ms, total_fit_ms);
  };

  if (eval_pool == nullptr) {
    best_iteration_ = completed_iterations > 0 ? completed_iterations - 1 : -1;
    const double total_fit_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - fit_start)
            .count();
    finish_fit(total_fit_ms);
    return;
  }

  if (!early_stopped || best_iteration_ < 0) {
    const double total_fit_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - fit_start)
            .count();
    finish_fit(total_fit_ms);
    return;
  }

  const std::size_t retained_iterations =
      static_cast<std::size_t>(best_iteration_ + 1);
  trees_.resize(retained_iterations * static_cast<std::size_t>(prediction_dimension_));
  loss_history_.resize(retained_iterations);
  eval_loss_history_.resize(retained_iterations);
  RecomputeFeatureImportances(trees_, pool.num_cols(), feature_importance_sums_);
  const double total_fit_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - fit_start)
          .count();
  finish_fit(total_fit_ms);
}

void GradientBooster::SetIterations(int iterations) {
  if (iterations <= 0) {
    throw std::invalid_argument("iterations must be positive");
  }
  iterations_ = iterations;
}

std::vector<float> GradientBooster::Predict(const Pool& pool, int num_iteration) const {
  std::size_t tree_limit = trees_.size();
  if (num_iteration >= 0) {
    tree_limit = std::min(
        trees_.size(),
        static_cast<std::size_t>(num_iteration) * static_cast<std::size_t>(prediction_dimension_));
  }

  std::vector<float> predictions(
      pool.num_rows() * static_cast<std::size_t>(prediction_dimension_), 0.0F);
  if (tree_limit == 0 || pool.num_rows() == 0) {
    AddPoolBaselineToPredictions(pool, prediction_dimension_, predictions);
    return predictions;
  }

  if (use_gpu_ && CudaBackendCompiled()) {
    const HistMatrix hist = BuildPredictionHist(pool, RequireQuantizationSchema(quantization_schema_));
    std::vector<std::int32_t> tree_offsets;
    const std::vector<GpuTreeNode> flattened_nodes = FlattenTreesForGpu(trees_, tree_limit, tree_offsets);
    PredictRawGpu(
        hist,
        flattened_nodes,
        tree_offsets,
        static_cast<float>(learning_rate_),
        prediction_dimension_,
        predictions,
        devices_);
    AddPoolBaselineToPredictions(pool, prediction_dimension_, predictions);
    return predictions;
  }

  if (prediction_dimension_ == 1) {
    for (std::size_t tree_index = 0; tree_index < tree_limit; ++tree_index) {
      const Tree& tree = trees_[tree_index];
      for (std::size_t row = 0; row < pool.num_rows(); ++row) {
        predictions[row] += learning_rate_ * tree.PredictRow(pool, row);
      }
    }
    AddPoolBaselineToPredictions(pool, prediction_dimension_, predictions);
    return predictions;
  }

  for (std::size_t tree_index = 0; tree_index < tree_limit; ++tree_index) {
    const int class_index = static_cast<int>(
        tree_index % static_cast<std::size_t>(prediction_dimension_));
    for (std::size_t row = 0; row < pool.num_rows(); ++row) {
      predictions[row * static_cast<std::size_t>(prediction_dimension_) + class_index] +=
          learning_rate_ * trees_[tree_index].PredictRow(pool, row);
    }
  }
  AddPoolBaselineToPredictions(pool, prediction_dimension_, predictions);
  return predictions;
}

std::vector<std::int32_t> GradientBooster::PredictLeafIndices(const Pool& pool,
                                                              int num_iteration) const {
  std::size_t tree_limit = trees_.size();
  if (num_iteration >= 0) {
    tree_limit = std::min(
        trees_.size(),
        static_cast<std::size_t>(num_iteration) * static_cast<std::size_t>(prediction_dimension_));
  }

  std::vector<std::int32_t> leaf_indices(pool.num_rows() * tree_limit, -1);
  for (std::size_t tree_index = 0; tree_index < tree_limit; ++tree_index) {
    for (std::size_t row = 0; row < pool.num_rows(); ++row) {
      leaf_indices[row * tree_limit + tree_index] =
          trees_[tree_index].PredictLeafIndex(pool, row);
    }
  }
  return leaf_indices;
}

std::vector<float> GradientBooster::PredictContributions(const Pool& pool, int num_iteration) const {
  std::size_t tree_limit = trees_.size();
  if (num_iteration >= 0) {
    tree_limit = std::min(
        trees_.size(),
        static_cast<std::size_t>(num_iteration) * static_cast<std::size_t>(prediction_dimension_));
  }

  const std::size_t row_width =
      static_cast<std::size_t>(prediction_dimension_) * (pool.num_cols() + 1);
  std::vector<float> contributions(pool.num_rows() * row_width, 0.0F);
  for (std::size_t tree_index = 0; tree_index < tree_limit; ++tree_index) {
    const std::size_t class_index = prediction_dimension_ == 1
                                        ? 0
                                        : tree_index % static_cast<std::size_t>(prediction_dimension_);
    for (std::size_t row = 0; row < pool.num_rows(); ++row) {
      std::vector<float> row_buffer(pool.num_cols() + 1, 0.0F);
      trees_[tree_index].AccumulateContributions(pool, row, static_cast<float>(learning_rate_), row_buffer);
      const std::size_t row_offset = row * row_width + class_index * (pool.num_cols() + 1);
      for (std::size_t feature = 0; feature < row_buffer.size(); ++feature) {
        contributions[row_offset + feature] += row_buffer[feature];
      }
    }
  }
  return contributions;
}

void GradientBooster::LoadState(std::vector<Tree> trees,
                                QuantizationSchemaPtr quantization_schema,
                                std::vector<double> loss_history,
                                std::vector<double> eval_loss_history,
                                std::vector<double> feature_importance_sums,
                                int best_iteration,
                                double best_score,
                                bool use_gpu,
                                std::uint64_t rng_state) {
  trees_ = std::move(trees);
  if (quantization_schema == nullptr && !trees_.empty()) {
    auto mutable_quantization_schema = std::make_shared<QuantizationSchema>();
    mutable_quantization_schema->num_bins_per_feature = trees_.front().num_bins_per_feature();
    mutable_quantization_schema->cut_offsets = trees_.front().cut_offsets();
    mutable_quantization_schema->cut_values = trees_.front().cut_values();
    mutable_quantization_schema->categorical_mask = trees_.front().categorical_mask();
    mutable_quantization_schema->missing_value_mask = trees_.front().missing_value_mask();
    mutable_quantization_schema->nan_mode = trees_.front().nan_mode();
    quantization_schema = mutable_quantization_schema;
  }
  quantization_schema_ = quantization_schema;
  if (quantization_schema_ != nullptr) {
    for (Tree& tree : trees_) {
      tree.SetQuantizationSchema(quantization_schema_);
    }
  }
  loss_history_ = std::move(loss_history);
  eval_loss_history_ = std::move(eval_loss_history);
  if (feature_importance_sums.empty()) {
    const std::size_t num_features =
        quantization_schema_ == nullptr ? 0 : quantization_schema_->num_cols();
    RecomputeFeatureImportances(trees_, num_features, feature_importance_sums_);
  } else {
    feature_importance_sums_ = std::move(feature_importance_sums);
  }
  best_iteration_ = best_iteration;
  best_score_ = best_score;
  use_gpu_ = use_gpu;
  if (rng_state != 0) {
    rng_state_ = NormalizeRngState(rng_state);
  }
}

void GradientBooster::LoadQuantizationSchema(QuantizationSchemaPtr quantization_schema) {
  quantization_schema_ = std::move(quantization_schema);
  if (quantization_schema_ != nullptr) {
    for (Tree& tree : trees_) {
      tree.SetQuantizationSchema(quantization_schema_);
    }
  }
}

const std::vector<double>& GradientBooster::loss_history() const noexcept {
  return loss_history_;
}

const std::vector<double>& GradientBooster::eval_loss_history() const noexcept {
  return eval_loss_history_;
}

std::size_t GradientBooster::num_trees() const noexcept { return trees_.size(); }

std::size_t GradientBooster::num_iterations_trained() const noexcept {
  if (prediction_dimension_ <= 0) {
    return 0;
  }
  return trees_.size() / static_cast<std::size_t>(prediction_dimension_);
}

int GradientBooster::num_classes() const noexcept { return num_classes_; }

int GradientBooster::prediction_dimension() const noexcept { return prediction_dimension_; }

int GradientBooster::best_iteration() const noexcept { return best_iteration_; }

double GradientBooster::best_score() const noexcept { return best_score_; }

const std::string& GradientBooster::objective_name() const noexcept {
  return objective_name_;
}

int GradientBooster::iterations() const noexcept { return iterations_; }

double GradientBooster::learning_rate() const noexcept { return learning_rate_; }

int GradientBooster::max_depth() const noexcept { return max_depth_; }

double GradientBooster::alpha() const noexcept { return alpha_; }

double GradientBooster::lambda_l2() const noexcept { return lambda_l2_; }

double GradientBooster::subsample() const noexcept { return subsample_; }

const std::string& GradientBooster::bootstrap_type() const noexcept { return bootstrap_type_; }

double GradientBooster::bagging_temperature() const noexcept { return bagging_temperature_; }

const std::string& GradientBooster::boosting_type() const noexcept { return boosting_type_; }

double GradientBooster::drop_rate() const noexcept { return drop_rate_; }

double GradientBooster::skip_drop() const noexcept { return skip_drop_; }

int GradientBooster::max_drop() const noexcept { return max_drop_; }

const std::vector<int>& GradientBooster::monotone_constraints() const noexcept {
  return monotone_constraints_;
}

const std::vector<std::vector<int>>& GradientBooster::interaction_constraints() const noexcept {
  return interaction_constraints_;
}

double GradientBooster::colsample_bytree() const noexcept { return colsample_bytree_; }

const std::vector<double>& GradientBooster::feature_weights() const noexcept {
  return feature_weights_;
}

const std::vector<double>& GradientBooster::first_feature_use_penalties() const noexcept {
  return first_feature_use_penalties_;
}

double GradientBooster::random_strength() const noexcept { return random_strength_; }

const std::string& GradientBooster::grow_policy() const noexcept { return grow_policy_; }

int GradientBooster::max_leaves() const noexcept { return max_leaves_; }

int GradientBooster::min_samples_split() const noexcept { return min_samples_split_; }

int GradientBooster::min_data_in_leaf() const noexcept { return min_data_in_leaf_; }

double GradientBooster::min_child_weight() const noexcept { return min_child_weight_; }

double GradientBooster::gamma() const noexcept { return gamma_; }

double GradientBooster::max_leaf_weight() const noexcept { return max_leaf_weight_; }

std::size_t GradientBooster::max_bins() const noexcept { return max_bins_; }

const std::string& GradientBooster::nan_mode_name() const noexcept {
  return hist_builder_.nan_mode_name();
}

const std::vector<std::uint16_t>& GradientBooster::max_bin_by_feature() const noexcept {
  return hist_builder_.max_bins_by_feature();
}

const std::string& GradientBooster::border_selection_method() const noexcept {
  return hist_builder_.border_selection_method_name();
}

const std::vector<std::string>& GradientBooster::nan_mode_by_feature() const noexcept {
  return hist_builder_.nan_mode_by_feature_names();
}

const std::vector<std::vector<float>>& GradientBooster::feature_borders() const noexcept {
  return hist_builder_.feature_borders();
}

bool GradientBooster::external_memory() const noexcept { return external_memory_; }

const std::string& GradientBooster::external_memory_dir() const noexcept {
  return external_memory_dir_;
}

const std::string& GradientBooster::eval_metric_name() const noexcept {
  return eval_metric_name_;
}

double GradientBooster::quantile_alpha() const noexcept {
  return objective_config_.quantile_alpha;
}

double GradientBooster::huber_delta() const noexcept {
  return objective_config_.huber_delta;
}

double GradientBooster::tweedie_variance_power() const noexcept {
  return objective_config_.tweedie_variance_power;
}

bool GradientBooster::use_gpu() const noexcept { return use_gpu_; }

const std::string& GradientBooster::devices() const noexcept { return devices_; }

int GradientBooster::distributed_world_size() const noexcept { return distributed_world_size_; }

int GradientBooster::distributed_rank() const noexcept { return distributed_rank_; }

const std::string& GradientBooster::distributed_root() const noexcept {
  return distributed_root_;
}

const std::string& GradientBooster::distributed_run_id() const noexcept {
  return distributed_run_id_;
}

double GradientBooster::distributed_timeout() const noexcept { return distributed_timeout_; }

std::uint64_t GradientBooster::random_seed() const noexcept { return random_seed_; }

std::uint64_t GradientBooster::rng_state() const noexcept { return rng_state_; }

bool GradientBooster::verbose() const noexcept { return verbose_; }

const QuantizationSchema* GradientBooster::quantization_schema() const noexcept {
  return quantization_schema_.get();
}

const std::vector<Tree>& GradientBooster::trees() const noexcept { return trees_; }

std::vector<float> GradientBooster::get_feature_importances() const {
  std::vector<float> importances(feature_importance_sums_.size(), 0.0F);
  const double total_importance = std::accumulate(
      feature_importance_sums_.begin(), feature_importance_sums_.end(), 0.0);
  if (total_importance <= 0.0) {
    return importances;
  }

  const double scale = 100.0 / total_importance;
  for (std::size_t feature = 0; feature < feature_importance_sums_.size(); ++feature) {
    importances[feature] = static_cast<float>(feature_importance_sums_[feature] * scale);
  }
  return importances;
}

}  // namespace ctboost
