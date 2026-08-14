#include "tree_internal.hpp"

#include <algorithm>
#include <chrono>

namespace ctboost::detail {

GpuChildHistogramState BuildGpuChildHistogramState(const TreeBuildOptions& options,
                                                   GpuHistogramWorkspace* gpu_workspace,
                                                   std::size_t row_begin,
                                                   std::size_t left_end,
                                                   std::size_t row_end,
                                                   const GpuHistogramSnapshot& parent_snapshot,
                                                   bool build_left_direct) {
  GpuChildHistogramState state;
  const std::size_t direct_begin = build_left_direct ? row_begin : left_end;
  const std::size_t direct_end = build_left_direct ? left_end : row_end;
  const auto direct_child_start = std::chrono::steady_clock::now();
  GpuHistogramSnapshot local_direct_child_snapshot;
  BuildHistogramsGpu(gpu_workspace,
                     direct_begin,
                     direct_end,
                     &local_direct_child_snapshot.node_statistics);
  DownloadHistogramSnapshotGpu(gpu_workspace, &local_direct_child_snapshot);
  GpuHistogramSnapshot direct_child_snapshot =
      options.distributed != nullptr
          ? AllReduceGpuHistogramSnapshot(options.distributed, local_direct_child_snapshot)
          : std::move(local_direct_child_snapshot);
  const double direct_child_histogram_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - direct_child_start)
          .count();
  const auto subtraction_start = std::chrono::steady_clock::now();
  GpuHistogramSnapshot sibling_child_snapshot =
      SubtractGpuHistogramSnapshot(parent_snapshot, direct_child_snapshot);
  const double sibling_child_histogram_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - subtraction_start)
          .count();

  if (build_left_direct) {
    state.left_snapshot = std::move(direct_child_snapshot);
    state.right_snapshot = std::move(sibling_child_snapshot);
    state.left_histogram_ms = direct_child_histogram_ms;
    state.right_histogram_ms = sibling_child_histogram_ms;
    state.left_snapshot_resident = options.distributed == nullptr;
    state.right_snapshot_resident = false;
  } else {
    state.left_snapshot = std::move(sibling_child_snapshot);
    state.right_snapshot = std::move(direct_child_snapshot);
    state.left_histogram_ms = sibling_child_histogram_ms;
    state.right_histogram_ms = direct_child_histogram_ms;
    state.left_snapshot_resident = false;
    state.right_snapshot_resident = options.distributed == nullptr;
  }
  return state;
}

bool ChooseGpuFirstChild(const HistMatrix& hist,
                         const TreeBuildOptions& options,
                         const LinearStatistic& statistic_engine,
                         const std::vector<int>* child_allowed_features,
                         const ChildLeafBounds& child_bounds,
                         int depth,
                         std::size_t row_begin,
                         std::size_t left_end,
                         std::size_t row_end,
                         GpuChildHistogramState* child_histograms) {
  if (options.grow_policy != GrowPolicy::LeafWise || options.max_leaves <= 0) {
    return true;
  }

  const NodeHistogramSet left_stats =
      MaterializeGpuHistogramSnapshot(hist, child_histograms->left_snapshot);
  const CandidateSelectionResult left_selection = SelectBestCandidateSplit(
      hist,
      left_stats,
      options,
      statistic_engine,
      child_allowed_features,
      child_bounds.left_lower_bound,
      child_bounds.left_upper_bound,
      depth + 1,
      row_begin,
      left_end);

  const NodeHistogramSet right_stats =
      MaterializeGpuHistogramSnapshot(hist, child_histograms->right_snapshot);
  const CandidateSelectionResult right_selection = SelectBestCandidateSplit(
      hist,
      right_stats,
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
