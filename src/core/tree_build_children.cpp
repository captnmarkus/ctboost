#include "tree_internal.hpp"

#include <algorithm>
#include <chrono>

namespace ctboost::detail {

ChildLeafBounds ComputeChildLeafBounds(const TreeBuildOptions& options,
                                       int feature_id,
                                       double left_leaf_weight,
                                       double right_leaf_weight,
                                       double leaf_lower_bound,
                                       double leaf_upper_bound) {
  ChildLeafBounds bounds{
      leaf_lower_bound,
      leaf_upper_bound,
      leaf_lower_bound,
      leaf_upper_bound,
  };
  const int monotone_sign =
      options.monotone_constraints == nullptr || feature_id < 0 ||
              static_cast<std::size_t>(feature_id) >= options.monotone_constraints->size()
          ? 0
          : (*options.monotone_constraints)[static_cast<std::size_t>(feature_id)];
  if (monotone_sign == 0) {
    return bounds;
  }

  const double midpoint = 0.5 * (left_leaf_weight + right_leaf_weight);
  if (monotone_sign > 0) {
    bounds.left_upper_bound = std::min(bounds.left_upper_bound, midpoint);
    bounds.right_lower_bound = std::max(bounds.right_lower_bound, midpoint);
  } else {
    bounds.left_lower_bound = std::max(bounds.left_lower_bound, midpoint);
    bounds.right_upper_bound = std::min(bounds.right_upper_bound, midpoint);
  }
  return bounds;
}

ChildInteractionState ResolveChildInteractionState(
    const HistMatrix& hist,
    const TreeBuildOptions& options,
    int feature_id,
    const std::vector<int>* node_allowed_features,
    const std::vector<int>* active_interaction_groups) {
  ChildInteractionState state;
  state.active_groups = active_interaction_groups;
  state.allowed_features = node_allowed_features;
  if (options.interaction_constraints == nullptr || feature_id < 0) {
    return state;
  }

  const auto& constraints = *options.interaction_constraints;
  if (static_cast<std::size_t>(feature_id) >= constraints.feature_to_groups.size()) {
    return state;
  }

  const auto& feature_groups =
      constraints.feature_to_groups[static_cast<std::size_t>(feature_id)];
  if (feature_groups.empty()) {
    return state;
  }

  if (active_interaction_groups == nullptr || active_interaction_groups->empty()) {
    state.active_groups_storage = feature_groups;
  } else {
    state.active_groups_storage =
        IntersectSortedVectors(*active_interaction_groups, feature_groups);
  }
  state.active_groups =
      state.active_groups_storage.empty() ? nullptr : &state.active_groups_storage;
  state.allowed_features_storage = FilterAllowedFeaturesForInteraction(
      hist.num_cols,
      node_allowed_features,
      constraints,
      state.active_groups);
  state.allowed_features = &state.allowed_features_storage;
  return state;
}

CpuChildHistogramState BuildCpuChildHistogramState(
    const HistMatrix& hist,
    const std::vector<float>& gradients,
    const std::vector<float>& hessians,
    const std::vector<float>& weights,
    const std::vector<std::size_t>& row_indices,
    std::size_t row_begin,
    std::size_t left_end,
    std::size_t row_end,
    bool build_left_direct,
    const TreeBuildOptions& options,
    const NodeHistogramSet& node_stats) {
  CpuChildHistogramState state;
  const std::size_t direct_begin = build_left_direct ? row_begin : left_end;
  const std::size_t direct_end = build_left_direct ? left_end : row_end;
  const auto direct_child_start = std::chrono::steady_clock::now();
  NodeHistogramSet local_direct_child_stats = ComputeNodeHistogramSet(
      hist,
      gradients,
      hessians,
      weights,
      row_indices,
      direct_begin,
      direct_end,
      false,
      nullptr);
  NodeHistogramSet direct_child_stats =
      AllReduceNodeHistogramSet(options.distributed, local_direct_child_stats);
  const double direct_child_histogram_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - direct_child_start)
          .count();
  const auto subtraction_start = std::chrono::steady_clock::now();
  NodeHistogramSet sibling_child_stats = SubtractNodeHistogramSet(node_stats, direct_child_stats);
  const double sibling_child_histogram_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - subtraction_start)
          .count();

  if (build_left_direct) {
    state.left_stats = std::move(direct_child_stats);
    state.right_stats = std::move(sibling_child_stats);
    state.left_histogram_ms = direct_child_histogram_ms;
    state.right_histogram_ms = sibling_child_histogram_ms;
  } else {
    state.left_stats = std::move(sibling_child_stats);
    state.right_stats = std::move(direct_child_stats);
    state.left_histogram_ms = sibling_child_histogram_ms;
    state.right_histogram_ms = direct_child_histogram_ms;
  }
  return state;
}

bool ChooseCpuFirstChild(const HistMatrix& hist,
                         const CpuChildHistogramState& child_histograms,
                         const TreeBuildOptions& options,
                         const LinearStatistic& statistic_engine,
                         const std::vector<int>* child_allowed_features,
                         const ChildLeafBounds& child_bounds,
                         int depth,
                         std::size_t row_begin,
                         std::size_t left_end,
                         std::size_t row_end) {
  if (options.grow_policy != GrowPolicy::LeafWise || options.max_leaves <= 0) {
    return true;
  }

  const CandidateSelectionResult left_selection = SelectBestCandidateSplit(
      hist,
      child_histograms.left_stats,
      options,
      statistic_engine,
      child_allowed_features,
      child_bounds.left_lower_bound,
      child_bounds.left_upper_bound,
      depth + 1,
      row_begin,
      left_end);
  const CandidateSelectionResult right_selection = SelectBestCandidateSplit(
      hist,
      child_histograms.right_stats,
      options,
      statistic_engine,
      child_allowed_features,
      child_bounds.right_lower_bound,
      child_bounds.right_upper_bound,
      depth + 1,
      left_end,
      row_end);
  return right_selection.adjusted_gain <= left_selection.adjusted_gain;
}

}  // namespace ctboost::detail
