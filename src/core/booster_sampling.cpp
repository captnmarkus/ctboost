#include "booster_internal.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>

namespace ctboost::booster_detail {

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
    const std::size_t selected = positive_rows[UniformIndex(rng_state, positive_rows.size())];
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
    tree.SetLeafWeight(node_index,
                       static_cast<float>(static_cast<double>(nodes[node_index].leaf_weight) * scale));
  }
}

double ResolveIterationLearningRate(const std::vector<double>& tree_learning_rates,
                                    std::size_t tree_index,
                                    int prediction_dimension,
                                    double default_learning_rate) {
  if (prediction_dimension <= 0) {
    return default_learning_rate;
  }
  const std::size_t iteration_index = tree_index / static_cast<std::size_t>(prediction_dimension);
  if (iteration_index >= tree_learning_rates.size()) {
    return default_learning_rate;
  }
  return tree_learning_rates[iteration_index];
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
    std::shuffle(
        dropped_groups.begin(), dropped_groups.end(), std::mt19937_64(NextRandom(rng_state)));
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

std::vector<int> SampleFeatureSubset(std::size_t num_features,
                                     double colsample_bytree,
                                     const std::vector<double>* feature_weights,
                                     std::uint64_t& rng_state) {
  std::vector<int> eligible_features;
  eligible_features.reserve(num_features);
  if (feature_weights != nullptr && !feature_weights->empty()) {
    for (std::size_t feature = 0; feature < num_features; ++feature) {
      const double feature_weight = feature < feature_weights->size() ? (*feature_weights)[feature] : 1.0;
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
      const std::size_t swap_index = index + UniformIndex(rng_state, eligible_count - index);
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

}  // namespace ctboost::booster_detail
