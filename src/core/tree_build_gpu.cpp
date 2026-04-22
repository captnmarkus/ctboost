#include "tree_internal.hpp"

#include <algorithm>
#include <chrono>
#include <limits>

#include "ctboost/cuda_backend.hpp"
#include "ctboost/profiler.hpp"

namespace ctboost {

int Tree::BuildNodeGpu(const HistMatrix& hist,
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
  (void)precomputed_node_stats;
  (void)statistic_engine;

  GpuNodeStatistics gpu_node_stats;
  GpuHistogramSnapshot parent_snapshot;
  double histogram_ms = precomputed_histogram_ms;
  if (precomputed_gpu_histogram != nullptr) {
    gpu_node_stats = precomputed_gpu_histogram->node_statistics;
    parent_snapshot = *precomputed_gpu_histogram;
    if (!precomputed_gpu_histogram_resident) {
      UploadHistogramSnapshotGpu(gpu_workspace, parent_snapshot);
    }
  } else {
    const auto histogram_start = std::chrono::steady_clock::now();
    GpuHistogramSnapshot local_snapshot;
    BuildHistogramsGpu(gpu_workspace, row_begin, row_end, &local_snapshot.node_statistics);
    DownloadHistogramSnapshotGpu(gpu_workspace, &local_snapshot);
    if (options.distributed != nullptr) {
      parent_snapshot = detail::AllReduceGpuHistogramSnapshot(options.distributed, local_snapshot);
      UploadHistogramSnapshotGpu(gpu_workspace, parent_snapshot);
    } else {
      parent_snapshot = std::move(local_snapshot);
    }
    gpu_node_stats = parent_snapshot.node_statistics;
    histogram_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - histogram_start)
            .count();
  }
  const std::size_t effective_row_count =
      options.distributed == nullptr ? row_count
                                     : static_cast<std::size_t>(gpu_node_stats.sample_count);
  if (profiler != nullptr && profiler->enabled()) {
    profiler->LogNodeHistogram(depth, effective_row_count, true, histogram_ms);
  }

  Node node;
  node.leaf_weight = static_cast<float>(detail::ClampLeafWeight(
      detail::ComputeLeafWeight(gpu_node_stats.total_gradient, gpu_node_stats.total_hessian, options.lambda_l2),
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

  const auto search_start = std::chrono::steady_clock::now();
  GpuNodeSearchResult node_search;
  const std::size_t adjusted_row_begin = options.distributed == nullptr ? row_begin : 0U;
  const std::size_t adjusted_row_end =
      options.distributed == nullptr ? row_end : effective_row_count;
  SearchBestNodeSplitGpu(gpu_workspace,
                         node_allowed_features,
                         options.lambda_l2,
                         options.min_data_in_leaf,
                         options.min_child_weight,
                         options.min_split_gain,
                         options.alpha,
                         depth,
                         adjusted_row_begin,
                         adjusted_row_end,
                         leaf_lower_bound,
                         leaf_upper_bound,
                         options.random_seed,
                         options.random_strength,
                         &node_search);
  node_search.node_statistics = gpu_node_stats;
  const double feature_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - search_start)
          .count();
  const double split_ms = 0.0;
  if (node_search.feature_id < 0 || node_search.p_value > options.alpha ||
      !node_search.split_valid || node_search.gain <= 0.0) {
    if (profiler != nullptr && profiler->enabled()) {
      profiler->LogNodeSearch(depth,
                              effective_row_count,
                              node_search.feature_id,
                              node_search.p_value,
                              node_search.chi_square,
                              false,
                              node_search.is_categorical,
                              node_search.gain,
                              0,
                              0,
                              feature_ms,
                              split_ms,
                              0.0);
    }
    return return_leaf();
  }

  const auto partition_start = std::chrono::steady_clock::now();
  const std::size_t left_end = PartitionHistogramRowsGpu(gpu_workspace,
                                                         row_begin,
                                                         row_end,
                                                         static_cast<std::size_t>(node_search.feature_id),
                                                         node_search.is_categorical,
                                                         node_search.split_bin,
                                                         node_search.left_categories);
  const double partition_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - partition_start)
          .count();
  const std::size_t left_count = left_end - row_begin;
  const std::size_t right_count = row_end - left_end;
  if (profiler != nullptr && profiler->enabled()) {
    profiler->LogNodeSearch(depth,
                            effective_row_count,
                            node_search.feature_id,
                            node_search.p_value,
                            node_search.chi_square,
                            true,
                            node_search.is_categorical,
                            node_search.gain,
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
      node_search.feature_id,
      node_search.left_leaf_weight,
      node_search.right_leaf_weight,
      leaf_lower_bound,
      leaf_upper_bound);
  const bool build_left_direct = options.distributed != nullptr || left_count <= right_count;
  detail::GpuChildHistogramState child_histograms = detail::BuildGpuChildHistogramState(
      options,
      gpu_workspace,
      row_begin,
      left_end,
      row_end,
      parent_snapshot,
      build_left_direct);

  nodes_[node_index].is_leaf = false;
  nodes_[node_index].split_feature_id = node_search.feature_id;
  nodes_[node_index].split_bin_index = node_search.split_bin;
  nodes_[node_index].is_categorical_split = node_search.is_categorical;
  if (node_search.is_categorical) {
    nodes_[node_index].left_categories = node_search.left_categories;
  }
  feature_importances_[static_cast<std::size_t>(node_search.feature_id)] += node_search.gain;
  if (leaf_count != nullptr) {
    ++(*leaf_count);
  }

  const detail::ChildInteractionState child_interaction = detail::ResolveChildInteractionState(
      hist,
      options,
      node_search.feature_id,
      node_allowed_features,
      active_interaction_groups);
  const bool build_left_first = detail::ChooseGpuFirstChild(options,
                                                            gpu_workspace,
                                                            child_interaction.allowed_features,
                                                            child_bounds,
                                                            depth,
                                                            row_begin,
                                                            left_end,
                                                            row_end,
                                                            &child_histograms);
  const auto build_child = [&](std::size_t child_begin,
                               std::size_t child_end,
                               const GpuHistogramSnapshot& child_snapshot,
                               bool child_snapshot_resident,
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
                     gpu_workspace,
                     &child_snapshot,
                     child_snapshot_resident,
                     nullptr,
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
                                                child_histograms.left_snapshot,
                                                child_histograms.left_snapshot_resident,
                                                child_histograms.left_histogram_ms,
                                                child_bounds.left_lower_bound,
                                                child_bounds.left_upper_bound);
    nodes_[node_index].right_child = build_child(left_end,
                                                 row_end,
                                                 child_histograms.right_snapshot,
                                                 child_histograms.right_snapshot_resident,
                                                 child_histograms.right_histogram_ms,
                                                 child_bounds.right_lower_bound,
                                                 child_bounds.right_upper_bound);
  } else {
    nodes_[node_index].right_child = build_child(left_end,
                                                 row_end,
                                                 child_histograms.right_snapshot,
                                                 child_histograms.right_snapshot_resident,
                                                 child_histograms.right_histogram_ms,
                                                 child_bounds.right_lower_bound,
                                                 child_bounds.right_upper_bound);
    nodes_[node_index].left_child = build_child(row_begin,
                                                left_end,
                                                child_histograms.left_snapshot,
                                                child_histograms.left_snapshot_resident,
                                                child_histograms.left_histogram_ms,
                                                child_bounds.left_lower_bound,
                                                child_bounds.left_upper_bound);
  }
  return node_index;
}

}  // namespace ctboost
