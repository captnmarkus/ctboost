#include "tree_internal.hpp"

#include <algorithm>
#include <chrono>
#include <limits>

#include "ctboost/profiler.hpp"

namespace ctboost {

int Tree::BuildNodeCpu(const HistMatrix& hist,
                       const std::vector<float>& gradients,
                       const std::vector<float>& hessians,
                       const std::vector<float>& weights,
                       std::vector<std::size_t>& row_indices,
                       std::size_t row_begin,
                       std::size_t row_end,
                       int depth,
                       const TreeBuildOptions& options,
                       GpuHistogramWorkspace* gpu_workspace,
                       const GpuHistogramSnapshot* precomputed_gpu_histogram,
                       bool precomputed_gpu_histogram_resident,
                       const NodeHistogramSet* precomputed_node_stats,
                       double precomputed_histogram_ms,
                       const std::vector<int>* node_allowed_features,
                       const std::vector<int>* active_interaction_groups,
                       double leaf_lower_bound,
                       double leaf_upper_bound,
                       const TrainingProfiler* profiler,
                       const LinearStatistic& statistic_engine,
                       std::vector<LeafRowRange>* leaf_row_ranges_out,
                       int* leaf_count) {
  const std::size_t row_count = row_end - row_begin;
  (void)gpu_workspace;
  (void)precomputed_gpu_histogram;
  (void)precomputed_gpu_histogram_resident;

  NodeHistogramSet node_stats;
  double histogram_ms = precomputed_histogram_ms;
  if (precomputed_node_stats != nullptr) {
    node_stats = *precomputed_node_stats;
  } else {
    const auto histogram_start = std::chrono::steady_clock::now();
    NodeHistogramSet local_node_stats = detail::ComputeNodeHistogramSet(
        hist, gradients, hessians, weights, row_indices, row_begin, row_end, false, nullptr);
    node_stats = detail::AllReduceNodeHistogramSet(options.distributed, local_node_stats);
    histogram_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - histogram_start)
            .count();
  }
  const std::size_t effective_row_count =
      options.distributed == nullptr ? row_count : static_cast<std::size_t>(node_stats.sample_count);
  if (profiler != nullptr && profiler->enabled()) {
    profiler->LogNodeHistogram(depth, effective_row_count, false, histogram_ms);
  }

  Node node;
  node.leaf_weight = static_cast<float>(detail::ClampLeafWeight(
      detail::ComputeLeafWeight(node_stats.total_gradient, node_stats.total_hessian, options.lambda_l2),
      leaf_lower_bound,
      leaf_upper_bound));

  const int node_index = static_cast<int>(nodes_.size());
  nodes_.push_back(node);
  if (leaf_row_ranges_out != nullptr && leaf_row_ranges_out->size() < nodes_.size()) {
    leaf_row_ranges_out->resize(nodes_.size());
  }
  const auto return_leaf = [&]() {
    if (leaf_row_ranges_out != nullptr) {
      (*leaf_row_ranges_out)[node_index] = LeafRowRange{row_begin, row_end};
    }
    return node_index;
  };

  if (effective_row_count < static_cast<std::size_t>(options.min_samples_split) ||
      depth >= options.max_depth ||
      (options.max_leaves > 0 && leaf_count != nullptr && *leaf_count >= options.max_leaves)) {
    return return_leaf();
  }

  const auto feature_start = std::chrono::steady_clock::now();
  const detail::CandidateSelectionResult selection = detail::SelectBestCandidateSplit(hist,
                                                                                      node_stats,
                                                                                      options,
                                                                                      statistic_engine,
                                                                                      node_allowed_features,
                                                                                      leaf_lower_bound,
                                                                                      leaf_upper_bound,
                                                                                      depth,
                                                                                      row_begin,
                                                                                      row_end);
  const detail::FeatureChoice& feature_choice = selection.feature_choice;
  const double feature_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - feature_start)
          .count();
  if (feature_choice.feature_id < 0 || feature_choice.p_value > options.alpha) {
    if (profiler != nullptr && profiler->enabled()) {
      profiler->LogNodeSearch(depth,
                              effective_row_count,
                              feature_choice.feature_id,
                              feature_choice.p_value,
                              feature_choice.chi_square,
                              false,
                              false,
                              0.0,
                              0,
                              0,
                              feature_ms,
                              0.0,
                              0.0);
    }
    return return_leaf();
  }

  const auto split_start = std::chrono::steady_clock::now();
  const detail::SplitChoice& split_choice = selection.split_choice;
  const double split_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - split_start)
          .count();
  if (!split_choice.valid || split_choice.gain <= 0.0) {
    if (profiler != nullptr && profiler->enabled()) {
      profiler->LogNodeSearch(depth,
                              effective_row_count,
                              feature_choice.feature_id,
                              feature_choice.p_value,
                              feature_choice.chi_square,
                              false,
                              split_choice.is_categorical,
                              split_choice.gain,
                              0,
                              0,
                              feature_ms,
                              split_ms,
                              0.0);
    }
    return return_leaf();
  }

  const auto feature_bins = hist.feature_bins(static_cast<std::size_t>(feature_choice.feature_id));
  const auto left_begin = row_indices.begin() + static_cast<std::ptrdiff_t>(row_begin);
  const auto right_end = row_indices.begin() + static_cast<std::ptrdiff_t>(row_end);
  const auto partition_start = std::chrono::steady_clock::now();
  const auto split_middle = std::partition(left_begin, right_end, [&](std::size_t row) {
    const std::uint16_t bin = feature_bins[row];
    return split_choice.is_categorical ? split_choice.left_categories[bin] != 0
                                       : bin <= split_choice.split_bin;
  });
  const double partition_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - partition_start)
          .count();
  const std::size_t left_end = row_begin + static_cast<std::size_t>(std::distance(left_begin, split_middle));
  const std::size_t left_count = left_end - row_begin;
  const std::size_t right_count = row_end - left_end;
  if (profiler != nullptr && profiler->enabled()) {
    profiler->LogNodeSearch(depth,
                            effective_row_count,
                            feature_choice.feature_id,
                            feature_choice.p_value,
                            feature_choice.chi_square,
                            true,
                            split_choice.is_categorical,
                            split_choice.gain,
                            left_count,
                            right_count,
                            feature_ms,
                            split_ms,
                            partition_ms);
  }
  if (options.distributed == nullptr && (left_end == row_begin || left_end == row_end)) {
    return return_leaf();
  }

  const detail::ChildLeafBounds child_bounds = detail::ComputeChildLeafBounds(
      options,
      feature_choice.feature_id,
      split_choice.left_leaf_weight,
      split_choice.right_leaf_weight,
      leaf_lower_bound,
      leaf_upper_bound);
  const bool build_left_direct = options.distributed != nullptr || left_count <= right_count;
  const detail::CpuChildHistogramState child_histograms = detail::BuildCpuChildHistogramState(
      hist,
      gradients,
      hessians,
      weights,
      row_indices,
      row_begin,
      left_end,
      row_end,
      build_left_direct,
      options,
      node_stats);

  nodes_[node_index].is_leaf = false;
  nodes_[node_index].split_feature_id = feature_choice.feature_id;
  nodes_[node_index].split_bin_index = split_choice.split_bin;
  nodes_[node_index].is_categorical_split = split_choice.is_categorical;
  if (split_choice.is_categorical) {
    nodes_[node_index].left_categories = split_choice.left_categories;
  }
  feature_importances_[static_cast<std::size_t>(feature_choice.feature_id)] += split_choice.gain;
  if (leaf_count != nullptr) {
    ++(*leaf_count);
  }

  const detail::ChildInteractionState child_interaction = detail::ResolveChildInteractionState(
      hist,
      options,
      feature_choice.feature_id,
      node_allowed_features,
      active_interaction_groups);
  const bool build_left_first = detail::ChooseCpuFirstChild(hist,
                                                            child_histograms,
                                                            options,
                                                            statistic_engine,
                                                            child_interaction.allowed_features,
                                                            child_bounds,
                                                            depth,
                                                            row_begin,
                                                            left_end,
                                                            row_end);
  const auto build_child = [&](std::size_t child_begin,
                               std::size_t child_end,
                               const NodeHistogramSet& child_stats,
                               double child_histogram_ms,
                               double child_lower_bound,
                               double child_upper_bound) {
    return BuildNode(hist,
                     gradients,
                     hessians,
                     weights,
                     row_indices,
                     child_begin,
                     child_end,
                     depth + 1,
                     options,
                     nullptr,
                     nullptr,
                     false,
                     &child_stats,
                     child_histogram_ms,
                     child_interaction.allowed_features,
                     child_interaction.active_groups,
                     child_lower_bound,
                     child_upper_bound,
                     profiler,
                     statistic_engine,
                     leaf_row_ranges_out,
                     leaf_count);
  };

  if (build_left_first) {
    nodes_[node_index].left_child = build_child(row_begin,
                                                left_end,
                                                child_histograms.left_stats,
                                                child_histograms.left_histogram_ms,
                                                child_bounds.left_lower_bound,
                                                child_bounds.left_upper_bound);
    nodes_[node_index].right_child = build_child(left_end,
                                                 row_end,
                                                 child_histograms.right_stats,
                                                 child_histograms.right_histogram_ms,
                                                 child_bounds.right_lower_bound,
                                                 child_bounds.right_upper_bound);
  } else {
    nodes_[node_index].right_child = build_child(left_end,
                                                 row_end,
                                                 child_histograms.right_stats,
                                                 child_histograms.right_histogram_ms,
                                                 child_bounds.right_lower_bound,
                                                 child_bounds.right_upper_bound);
    nodes_[node_index].left_child = build_child(row_begin,
                                                left_end,
                                                child_histograms.left_stats,
                                                child_histograms.left_histogram_ms,
                                                child_bounds.left_lower_bound,
                                                child_bounds.left_upper_bound);
  }
  return node_index;
}

}  // namespace ctboost
