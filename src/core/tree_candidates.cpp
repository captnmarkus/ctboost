#include "tree_internal.hpp"

#include <algorithm>
#include <cmath>

namespace ctboost::detail {
namespace {

std::vector<FeatureChoice> RankFeaturesByStatistic(const NodeHistogramSet& node_stats,
                                                   const LinearStatistic& statistic_engine,
                                                   const std::vector<int>* allowed_features) {
  std::vector<FeatureChoice> ranked_features;

  const auto evaluate_feature = [&](std::size_t feature) {
    const BinStatistics& feature_stats = node_stats.by_feature[feature];
    if (feature_stats.weight_sums.size() <= 1) {
      return;
    }

    const auto result = statistic_engine.EvaluateScoreFromBinStatistics(
        feature_stats,
        node_stats.total_gradient,
        node_stats.sample_weight_sum,
        node_stats.gradient_variance);
    if (result.degrees_of_freedom == 0) {
      return;
    }

    ranked_features.push_back(
        FeatureChoice{
            static_cast<int>(feature),
            result.p_value,
            result.chi_square,
        });
  };

  if (allowed_features != nullptr && !allowed_features->empty()) {
    for (int feature_id : *allowed_features) {
      if (feature_id < 0 ||
          static_cast<std::size_t>(feature_id) >= node_stats.by_feature.size()) {
        continue;
      }
      evaluate_feature(static_cast<std::size_t>(feature_id));
    }
  } else {
    for (std::size_t feature = 0; feature < node_stats.by_feature.size(); ++feature) {
      evaluate_feature(feature);
    }
  }

  std::sort(ranked_features.begin(),
            ranked_features.end(),
            [](const FeatureChoice& lhs, const FeatureChoice& rhs) {
              if (std::abs(lhs.p_value - rhs.p_value) > 1e-12) {
                return lhs.p_value < rhs.p_value;
              }
              return lhs.chi_square > rhs.chi_square;
            });
  return ranked_features;
}

bool FeatureAllowedByInteraction(int feature_id,
                                 const InteractionConstraintSet& constraints,
                                 const std::vector<int>* active_groups) {
  if (feature_id < 0 ||
      static_cast<std::size_t>(feature_id) >= constraints.feature_to_groups.size()) {
    return false;
  }
  if (constraints.constrained_feature_mask[static_cast<std::size_t>(feature_id)] == 0U) {
    return true;
  }
  if (active_groups == nullptr || active_groups->empty()) {
    return true;
  }
  const auto& feature_groups = constraints.feature_to_groups[static_cast<std::size_t>(feature_id)];
  for (const int group_id : feature_groups) {
    if (std::find(active_groups->begin(), active_groups->end(), group_id) != active_groups->end()) {
      return true;
    }
  }
  return false;
}

FeatureChoice SelectBestFeature(const NodeHistogramSet& node_stats,
                                const LinearStatistic& statistic_engine,
                                const std::vector<int>* allowed_features) {
  FeatureChoice best;

  const auto evaluate_feature = [&](std::size_t feature) {
    const BinStatistics& feature_stats = node_stats.by_feature[feature];
    if (feature_stats.weight_sums.size() <= 1) {
      return;
    }

    const auto result = statistic_engine.EvaluateScoreFromBinStatistics(
        feature_stats,
        node_stats.total_gradient,
        node_stats.sample_weight_sum,
        node_stats.gradient_variance);
    if (result.degrees_of_freedom == 0) {
      return;
    }

    if (best.feature_id < 0 || result.p_value < best.p_value ||
        (std::abs(result.p_value - best.p_value) <= 1e-12 &&
         result.chi_square > best.chi_square)) {
      best.feature_id = static_cast<int>(feature);
      best.p_value = result.p_value;
      best.chi_square = result.chi_square;
    }
  };

  if (allowed_features != nullptr && !allowed_features->empty()) {
    for (int feature_id : *allowed_features) {
      if (feature_id < 0 ||
          static_cast<std::size_t>(feature_id) >= node_stats.by_feature.size()) {
        continue;
      }
      evaluate_feature(static_cast<std::size_t>(feature_id));
    }
    return best;
  }

  for (std::size_t feature = 0; feature < node_stats.by_feature.size(); ++feature) {
    evaluate_feature(feature);
  }
  return best;
}

}  // namespace

std::vector<int> FilterAllowedFeaturesForInteraction(
    std::size_t num_features,
    const std::vector<int>* parent_allowed_features,
    const InteractionConstraintSet& constraints,
    const std::vector<int>* active_groups) {
  std::vector<int> filtered_features;
  if (parent_allowed_features != nullptr && !parent_allowed_features->empty()) {
    filtered_features.reserve(parent_allowed_features->size());
    for (const int feature_id : *parent_allowed_features) {
      if (FeatureAllowedByInteraction(feature_id, constraints, active_groups)) {
        filtered_features.push_back(feature_id);
      }
    }
    return filtered_features;
  }

  filtered_features.reserve(num_features);
  for (std::size_t feature = 0; feature < num_features; ++feature) {
    const int feature_id = static_cast<int>(feature);
    if (FeatureAllowedByInteraction(feature_id, constraints, active_groups)) {
      filtered_features.push_back(feature_id);
    }
  }
  return filtered_features;
}

std::vector<int> IntersectSortedVectors(const std::vector<int>& lhs, const std::vector<int>& rhs) {
  std::vector<int> intersection;
  std::set_intersection(lhs.begin(),
                        lhs.end(),
                        rhs.begin(),
                        rhs.end(),
                        std::back_inserter(intersection));
  return intersection;
}

CandidateSelectionResult SelectBestCandidateSplit(const HistMatrix& hist,
                                                  const NodeHistogramSet& node_stats,
                                                  const TreeBuildOptions& options,
                                                  const LinearStatistic& statistic_engine,
                                                  const std::vector<int>* node_allowed_features,
                                                  double leaf_lower_bound,
                                                  double leaf_upper_bound,
                                                  int depth,
                                                  std::size_t row_begin,
                                                  std::size_t row_end) {
  CandidateSelectionResult best;
  const bool constrained_search =
      (options.monotone_constraints != nullptr && !options.monotone_constraints->empty()) ||
      options.interaction_constraints != nullptr;
  const bool use_ranked_search = constrained_search || options.random_strength > 0.0 ||
                                 options.feature_weights != nullptr ||
                                 options.first_feature_use_penalties != nullptr;

  if (!use_ranked_search) {
    best.feature_choice = SelectBestFeature(node_stats, statistic_engine, node_allowed_features);
    if (best.feature_choice.feature_id < 0 || best.feature_choice.p_value > options.alpha) {
      return best;
    }

    best.split_choice = SelectBestSplit(
        node_stats.by_feature[static_cast<std::size_t>(best.feature_choice.feature_id)],
        node_stats.total_gradient,
        node_stats.total_hessian,
        node_stats.sample_weight_sum,
        options.lambda_l2,
        options.min_data_in_leaf,
        options.min_child_weight,
        options.min_split_gain,
        hist.is_categorical(static_cast<std::size_t>(best.feature_choice.feature_id)),
        0,
        leaf_lower_bound,
        leaf_upper_bound);
    best.adjusted_gain = best.split_choice.gain;
    return best;
  }

  const std::vector<FeatureChoice> ranked_features =
      RankFeaturesByStatistic(node_stats, statistic_engine, node_allowed_features);
  if (!ranked_features.empty()) {
    best.feature_choice = ranked_features.front();
  }

  for (const FeatureChoice& candidate : ranked_features) {
    if (candidate.p_value > options.alpha) {
      break;
    }

    const int monotone_sign =
        options.monotone_constraints == nullptr ||
                static_cast<std::size_t>(candidate.feature_id) >= options.monotone_constraints->size()
            ? 0
            : (*options.monotone_constraints)[static_cast<std::size_t>(candidate.feature_id)];
    const SplitChoice candidate_split = SelectBestSplit(
        node_stats.by_feature[static_cast<std::size_t>(candidate.feature_id)],
        node_stats.total_gradient,
        node_stats.total_hessian,
        node_stats.sample_weight_sum,
        options.lambda_l2,
        options.min_data_in_leaf,
        options.min_child_weight,
        options.min_split_gain,
        hist.is_categorical(static_cast<std::size_t>(candidate.feature_id)),
        monotone_sign,
        leaf_lower_bound,
        leaf_upper_bound);
    if (!candidate_split.valid || candidate_split.gain <= 0.0) {
      continue;
    }

    const std::size_t adjusted_row_begin = options.distributed == nullptr ? row_begin : 0U;
    const std::size_t adjusted_row_end = options.distributed == nullptr
                                             ? row_end
                                             : static_cast<std::size_t>(node_stats.sample_count);
    const double adjusted_gain =
        AdjustedCandidateGain(options,
                              candidate.feature_id,
                              candidate_split.gain,
                              depth,
                              adjusted_row_begin,
                              adjusted_row_end);
    if (!best.split_choice.valid || adjusted_gain > best.adjusted_gain) {
      best.feature_choice = candidate;
      best.split_choice = candidate_split;
      best.adjusted_gain = adjusted_gain;
    }
  }

  return best;
}

}  // namespace ctboost::detail
