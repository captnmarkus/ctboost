#include "ctboost/tree.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include "ctboost/cuda_backend.hpp"
#include "ctboost/profiler.hpp"

namespace ctboost {

struct NodeHistogramSet {
  std::vector<BinStatistics> by_feature;
  double sample_weight_sum{0.0};
  double total_gradient{0.0};
  double total_hessian{0.0};
  double gradient_square_sum{0.0};
  double gradient_variance{0.0};
};

namespace {

struct FeatureChoice {
  int feature_id{-1};
  double p_value{1.0};
  double chi_square{-std::numeric_limits<double>::infinity()};
};

struct SplitChoice {
  bool valid{false};
  bool is_categorical{false};
  std::uint16_t split_bin{0};
  std::array<std::uint8_t, kMaxCategoricalRouteBins> left_categories{};
  double gain{-std::numeric_limits<double>::infinity()};
};

float ComputeLeafWeight(double gradient_sum, double hessian_sum, double lambda_l2) {
  return static_cast<float>(-gradient_sum / (hessian_sum + lambda_l2));
}

double ComputeGain(double gradient_sum, double hessian_sum, double lambda_l2) {
  return (gradient_sum * gradient_sum) / (hessian_sum + lambda_l2);
}

double ComputeGradientVariance(double weighted_gradient_sum,
                               double weighted_gradient_square_sum,
                               double sample_weight_sum) {
  if (sample_weight_sum <= 0.0) {
    return 0.0;
  }

  const double mean_gradient = weighted_gradient_sum / sample_weight_sum;
  const double second_moment = weighted_gradient_square_sum / sample_weight_sum;
  return std::max(0.0, second_moment - mean_gradient * mean_gradient);
}

const QuantizationSchema& RequireQuantizationSchema(const QuantizationSchemaPtr& schema) {
  if (schema == nullptr) {
    throw std::runtime_error("tree quantization schema is not initialized");
  }
  return *schema;
}

GpuHistogramSnapshot SubtractGpuHistogramSnapshot(const GpuHistogramSnapshot& parent,
                                                 const GpuHistogramSnapshot& child) {
  if (parent.gradient_sums.size() != child.gradient_sums.size() ||
      parent.hessian_sums.size() != child.hessian_sums.size() ||
      parent.weight_sums.size() != child.weight_sums.size()) {
    throw std::invalid_argument(
        "GPU parent and child histogram snapshots must have matching buffer sizes");
  }

  GpuHistogramSnapshot derived;
  derived.gradient_sums.resize(parent.gradient_sums.size(), 0.0F);
  derived.hessian_sums.resize(parent.hessian_sums.size(), 0.0F);
  derived.weight_sums.resize(parent.weight_sums.size(), 0.0F);
  for (std::size_t index = 0; index < parent.gradient_sums.size(); ++index) {
    derived.gradient_sums[index] = parent.gradient_sums[index] - child.gradient_sums[index];
    derived.hessian_sums[index] =
        std::max(0.0F, parent.hessian_sums[index] - child.hessian_sums[index]);
    derived.weight_sums[index] =
        std::max(0.0F, parent.weight_sums[index] - child.weight_sums[index]);
  }

  derived.node_statistics.sample_weight_sum = std::max(
      0.0, parent.node_statistics.sample_weight_sum - child.node_statistics.sample_weight_sum);
  derived.node_statistics.total_gradient =
      parent.node_statistics.total_gradient - child.node_statistics.total_gradient;
  derived.node_statistics.total_hessian =
      std::max(0.0, parent.node_statistics.total_hessian - child.node_statistics.total_hessian);
  derived.node_statistics.gradient_square_sum = std::max(
      0.0,
      parent.node_statistics.gradient_square_sum - child.node_statistics.gradient_square_sum);
  return derived;
}

NodeHistogramSet ComputeNodeHistogramSet(const HistMatrix& hist,
                                         const std::vector<float>& gradients,
                                         const std::vector<float>& hessians,
                                         const std::vector<float>& weights,
                                         const std::vector<std::size_t>& row_indices,
                                         std::size_t row_begin,
                                         std::size_t row_end,
                                         bool use_gpu,
                                         GpuHistogramWorkspace* gpu_workspace) {
  NodeHistogramSet stats;
  stats.by_feature.resize(hist.num_cols);

  if (use_gpu) {
    (void)gpu_workspace;
    throw std::invalid_argument(
        "GPU node histogram materialization is no longer supported in the CPU search path");
  }

  for (std::size_t index = row_begin; index < row_end; ++index) {
    const std::size_t row = row_indices[index];
    const double gradient = gradients[row];
    const double hessian = hessians[row];
    const double sample_weight = weights[row];
    stats.total_gradient += sample_weight * gradient;
    stats.total_hessian += sample_weight * hessian;
    stats.gradient_square_sum += sample_weight * gradient * gradient;
    stats.sample_weight_sum += sample_weight;
  }
  stats.gradient_variance = ComputeGradientVariance(
      stats.total_gradient, stats.gradient_square_sum, stats.sample_weight_sum);

  for (std::size_t feature = 0; feature < hist.num_cols; ++feature) {
    const std::size_t feature_bins_count = hist.num_bins(feature);
    BinStatistics feature_stats;
    feature_stats.gradient_sums.assign(feature_bins_count, 0.0);
    feature_stats.hessian_sums.assign(feature_bins_count, 0.0);
    feature_stats.weight_sums.assign(feature_bins_count, 0.0);

    const auto feature_bins = hist.feature_bins(feature);
    for (std::size_t index = row_begin; index < row_end; ++index) {
      const std::size_t row = row_indices[index];
      const std::size_t bin = feature_bins[row];
      const double sample_weight = static_cast<double>(weights[row]);
      feature_stats.gradient_sums[bin] += sample_weight * gradients[row];
      feature_stats.hessian_sums[bin] += sample_weight * hessians[row];
      feature_stats.weight_sums[bin] += sample_weight;
    }

    stats.by_feature[feature] = std::move(feature_stats);
  }

  return stats;
}

NodeHistogramSet SubtractNodeHistogramSet(const NodeHistogramSet& parent,
                                          const NodeHistogramSet& child) {
  if (parent.by_feature.size() != child.by_feature.size()) {
    throw std::invalid_argument("parent and child histogram sets must have the same feature count");
  }

  NodeHistogramSet derived;
  derived.by_feature.resize(parent.by_feature.size());
  derived.sample_weight_sum = parent.sample_weight_sum - child.sample_weight_sum;
  derived.total_gradient = parent.total_gradient - child.total_gradient;
  derived.total_hessian = parent.total_hessian - child.total_hessian;
  derived.gradient_square_sum = parent.gradient_square_sum - child.gradient_square_sum;

  for (std::size_t feature = 0; feature < parent.by_feature.size(); ++feature) {
    const BinStatistics& parent_stats = parent.by_feature[feature];
    const BinStatistics& child_stats = child.by_feature[feature];
    if (parent_stats.gradient_sums.size() != child_stats.gradient_sums.size()) {
      throw std::invalid_argument(
          "parent and child histogram sets must have matching bin counts");
    }

    BinStatistics feature_stats;
    feature_stats.gradient_sums.resize(parent_stats.gradient_sums.size());
    feature_stats.hessian_sums.resize(parent_stats.hessian_sums.size());
    feature_stats.weight_sums.resize(parent_stats.weight_sums.size());
    for (std::size_t bin = 0; bin < parent_stats.gradient_sums.size(); ++bin) {
      feature_stats.gradient_sums[bin] =
          parent_stats.gradient_sums[bin] - child_stats.gradient_sums[bin];
      feature_stats.hessian_sums[bin] = std::max(
          0.0, parent_stats.hessian_sums[bin] - child_stats.hessian_sums[bin]);
      feature_stats.weight_sums[bin] =
          std::max(0.0, parent_stats.weight_sums[bin] - child_stats.weight_sums[bin]);
    }
    derived.by_feature[feature] = std::move(feature_stats);
  }

  derived.sample_weight_sum = std::max(0.0, derived.sample_weight_sum);
  derived.total_hessian = std::max(0.0, derived.total_hessian);
  derived.gradient_square_sum = std::max(0.0, derived.gradient_square_sum);
  derived.gradient_variance = ComputeGradientVariance(
      derived.total_gradient, derived.gradient_square_sum, derived.sample_weight_sum);
  return derived;
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

SplitChoice SelectBestSplit(const BinStatistics& feature_stats,
                            double total_gradient,
                            double total_hessian,
                            double sample_weight_sum,
                            double lambda_l2,
                            int min_data_in_leaf,
                            double min_child_weight,
                            double min_split_gain,
                            bool is_categorical) {
  SplitChoice best;
  const std::size_t feature_bins_count = feature_stats.gradient_sums.size();
  if (feature_bins_count <= 1) {
    return best;
  }

  const double parent_gain = ComputeGain(total_gradient, total_hessian, lambda_l2);

  if (!is_categorical) {
    double left_gradient = 0.0;
    double left_hessian = 0.0;
    double left_count = 0.0;

    for (std::size_t split_bin = 0; split_bin + 1 < feature_bins_count; ++split_bin) {
      left_gradient += feature_stats.gradient_sums[split_bin];
      left_hessian += feature_stats.hessian_sums[split_bin];
      left_count += feature_stats.weight_sums[split_bin];

      const double right_count = sample_weight_sum - left_count;
      if (left_count <= 0.0 || right_count <= 0.0 ||
          left_count < static_cast<double>(min_data_in_leaf) ||
          right_count < static_cast<double>(min_data_in_leaf)) {
        continue;
      }

      const double right_gradient = total_gradient - left_gradient;
      const double right_hessian = total_hessian - left_hessian;
      if (left_hessian < min_child_weight || right_hessian < min_child_weight) {
        continue;
      }
      const double gain = ComputeGain(left_gradient, left_hessian, lambda_l2) +
                          ComputeGain(right_gradient, right_hessian, lambda_l2) - parent_gain;
      if (gain <= min_split_gain) {
        continue;
      }

      if (!best.valid || gain > best.gain) {
        best.valid = true;
        best.split_bin = static_cast<std::uint16_t>(split_bin);
        best.gain = gain;
      }
    }

    return best;
  }

  if (feature_bins_count > kMaxCategoricalRouteBins) {
    throw std::invalid_argument("categorical split routing supports at most 256 bins");
  }

  struct WeightedBin {
    std::uint16_t bin{0};
    double gradient{0.0};
    double hessian{0.0};
    double count{0.0};
    double weight{0.0};
  };

  std::vector<WeightedBin> active_bins;
  active_bins.reserve(feature_bins_count);
  for (std::size_t bin = 0; bin < feature_bins_count; ++bin) {
    if (feature_stats.weight_sums[bin] <= 0.0) {
      continue;
    }

    const double denominator = feature_stats.hessian_sums[bin] + lambda_l2;
    active_bins.push_back(WeightedBin{
        static_cast<std::uint16_t>(bin),
        feature_stats.gradient_sums[bin],
        feature_stats.hessian_sums[bin],
        feature_stats.weight_sums[bin],
        denominator > 0.0 ? feature_stats.gradient_sums[bin] / denominator : 0.0,
    });
  }

  if (active_bins.size() <= 1) {
    return best;
  }

  std::sort(active_bins.begin(), active_bins.end(), [](const WeightedBin& lhs, const WeightedBin& rhs) {
    if (lhs.weight == rhs.weight) {
      return lhs.bin < rhs.bin;
    }
    return lhs.weight < rhs.weight;
  });

  double left_gradient = 0.0;
  double left_hessian = 0.0;
  double left_count = 0.0;

  for (std::size_t split_index = 0; split_index + 1 < active_bins.size(); ++split_index) {
    left_gradient += active_bins[split_index].gradient;
    left_hessian += active_bins[split_index].hessian;
    left_count += active_bins[split_index].count;

    const double right_count = sample_weight_sum - left_count;
    if (left_count <= 0.0 || right_count <= 0.0 ||
        left_count < static_cast<double>(min_data_in_leaf) ||
        right_count < static_cast<double>(min_data_in_leaf)) {
      continue;
    }

    const double right_gradient = total_gradient - left_gradient;
    const double right_hessian = total_hessian - left_hessian;
    if (left_hessian < min_child_weight || right_hessian < min_child_weight) {
      continue;
    }
    const double gain = ComputeGain(left_gradient, left_hessian, lambda_l2) +
                        ComputeGain(right_gradient, right_hessian, lambda_l2) - parent_gain;
    if (gain <= min_split_gain) {
      continue;
    }

    if (!best.valid || gain > best.gain) {
      best.valid = true;
      best.is_categorical = true;
      best.gain = gain;
      best.left_categories.fill(0);
      for (std::size_t left_index = 0; left_index <= split_index; ++left_index) {
        best.left_categories[active_bins[left_index].bin] = 1;
      }
    }
  }

  return best;
}

}  // namespace

void Tree::Build(const HistMatrix& hist,
                 const std::vector<float>& gradients,
                 const std::vector<float>& hessians,
                 const std::vector<float>& weights,
                 const TreeBuildOptions& options,
                 GpuHistogramWorkspace* gpu_workspace,
                 const TrainingProfiler* profiler,
                 std::vector<std::size_t>* row_indices_out,
                 std::vector<LeafRowRange>* leaf_row_ranges_out,
                 const QuantizationSchemaPtr& quantization_schema) {
  if (weights.size() != hist.num_rows) {
    throw std::invalid_argument("weight size must match the histogram row count");
  }
  if (options.use_gpu) {
    if (gpu_workspace == nullptr) {
      throw std::invalid_argument("GPU histogram workspace must be provided when task_type='GPU'");
    }
    if ((!gradients.empty() && gradients.size() != hist.num_rows) ||
        (!hessians.empty() && hessians.size() != hist.num_rows)) {
      throw std::invalid_argument(
          "non-empty gradient and hessian buffers must match the histogram row count");
    }
  } else if (gradients.size() != hist.num_rows || hessians.size() != hist.num_rows) {
    throw std::invalid_argument(
        "gradient, hessian, and weight sizes must match the histogram row count");
  }

  nodes_.clear();
  if (quantization_schema != nullptr) {
    if (quantization_schema->num_cols() != hist.num_cols) {
      throw std::invalid_argument(
          "tree quantization schema feature count must match the histogram feature count");
    }
    quantization_schema_ = quantization_schema;
  } else {
    quantization_schema_ = std::make_shared<QuantizationSchema>(MakeQuantizationSchema(hist));
  }
  feature_importances_.assign(hist.num_cols, 0.0);

  std::vector<std::size_t> row_indices;
  const std::size_t initial_row_count = hist.num_rows;
  if (!options.use_gpu) {
    row_indices.resize(hist.num_rows);
    std::iota(row_indices.begin(), row_indices.end(), 0);
  } else {
    ResetHistogramRowIndicesGpu(gpu_workspace);
  }
  if (leaf_row_ranges_out != nullptr) {
    leaf_row_ranges_out->clear();
  }

  int leaf_count = 1;
  const LinearStatistic statistic_engine;
  BuildNode(hist,
            gradients,
            hessians,
            weights,
            row_indices,
            0,
            initial_row_count,
            0,
            options,
            gpu_workspace,
            nullptr,
            false,
            nullptr,
            0.0,
            profiler,
            statistic_engine,
            leaf_row_ranges_out,
            &leaf_count);
  if (row_indices_out != nullptr) {
    if (options.use_gpu) {
      DownloadHistogramRowIndicesGpu(gpu_workspace, *row_indices_out);
    } else {
      *row_indices_out = std::move(row_indices);
    }
  }
}

int Tree::BuildNode(const HistMatrix& hist,
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
                    const TrainingProfiler* profiler,
                    const LinearStatistic& statistic_engine,
                    std::vector<LeafRowRange>* leaf_row_ranges_out,
                    int* leaf_count) {
  const std::size_t row_count = row_end - row_begin;
  if (options.use_gpu) {
    (void)precomputed_node_stats;
    (void)statistic_engine;

    GpuNodeStatistics gpu_node_stats;
    double histogram_ms = precomputed_histogram_ms;
    if (precomputed_gpu_histogram != nullptr) {
      gpu_node_stats = precomputed_gpu_histogram->node_statistics;
    } else {
      const auto histogram_start = std::chrono::steady_clock::now();
      BuildHistogramsGpu(gpu_workspace, row_begin, row_end, &gpu_node_stats);
      histogram_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - histogram_start)
              .count();
    }
    if (profiler != nullptr && profiler->enabled()) {
      profiler->LogNodeHistogram(depth, row_count, true, histogram_ms);
    }

    Node node;
    node.leaf_weight =
        ComputeLeafWeight(gpu_node_stats.total_gradient, gpu_node_stats.total_hessian, options.lambda_l2);

    const int node_index = static_cast<int>(nodes_.size());
    nodes_.push_back(node);
    if (leaf_row_ranges_out != nullptr && leaf_row_ranges_out->size() < nodes_.size()) {
      leaf_row_ranges_out->resize(nodes_.size());
    }

    if (row_count <= 1 || depth >= options.max_depth ||
        (options.max_leaves > 0 && leaf_count != nullptr && *leaf_count >= options.max_leaves)) {
      if (leaf_row_ranges_out != nullptr) {
        (*leaf_row_ranges_out)[node_index] = LeafRowRange{row_begin, row_end};
      }
      return node_index;
    }

    const auto search_start = std::chrono::steady_clock::now();
    if (precomputed_gpu_histogram != nullptr && !precomputed_gpu_histogram_resident) {
      UploadHistogramSnapshotGpu(gpu_workspace, *precomputed_gpu_histogram);
    }
    GpuNodeSearchResult node_search;
    SearchBestNodeSplitGpu(gpu_workspace,
                           options.allowed_features,
                           options.lambda_l2,
                           options.min_data_in_leaf,
                           options.min_child_weight,
                           options.min_split_gain,
                           &node_search);
    node_search.node_statistics = gpu_node_stats;
    const double feature_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - search_start)
            .count();
    const double split_ms = 0.0;
    if (node_search.feature_id < 0 || node_search.p_value > options.alpha) {
      if (profiler != nullptr && profiler->enabled()) {
        profiler->LogNodeSearch(depth,
                                row_count,
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
      if (leaf_row_ranges_out != nullptr) {
        (*leaf_row_ranges_out)[node_index] = LeafRowRange{row_begin, row_end};
      }
      return node_index;
    }

    if (!node_search.split_valid || node_search.gain <= 0.0) {
      if (profiler != nullptr && profiler->enabled()) {
        profiler->LogNodeSearch(depth,
                                row_count,
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
      if (leaf_row_ranges_out != nullptr) {
        (*leaf_row_ranges_out)[node_index] = LeafRowRange{row_begin, row_end};
      }
      return node_index;
    }

    GpuHistogramSnapshot parent_snapshot;
    if (precomputed_gpu_histogram != nullptr) {
      parent_snapshot = *precomputed_gpu_histogram;
    } else {
      DownloadHistogramSnapshotGpu(gpu_workspace, &parent_snapshot);
      parent_snapshot.node_statistics = gpu_node_stats;
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
                              row_count,
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
    if (left_end == row_begin || left_end == row_end) {
      if (leaf_row_ranges_out != nullptr) {
        (*leaf_row_ranges_out)[node_index] = LeafRowRange{row_begin, row_end};
      }
      return node_index;
    }

    const bool build_left_direct = left_count <= right_count;
    const std::size_t direct_begin = build_left_direct ? row_begin : left_end;
    const std::size_t direct_end = build_left_direct ? left_end : row_end;
    const auto direct_child_start = std::chrono::steady_clock::now();
    GpuHistogramSnapshot direct_child_snapshot;
    BuildHistogramsGpu(
        gpu_workspace, direct_begin, direct_end, &direct_child_snapshot.node_statistics);
    DownloadHistogramSnapshotGpu(gpu_workspace, &direct_child_snapshot);
    const double direct_child_histogram_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - direct_child_start)
            .count();
    const auto subtraction_start = std::chrono::steady_clock::now();
    const GpuHistogramSnapshot sibling_child_snapshot =
        SubtractGpuHistogramSnapshot(parent_snapshot, direct_child_snapshot);
    const double sibling_child_histogram_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - subtraction_start)
            .count();

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
    if (build_left_direct) {
      nodes_[node_index].left_child =
          BuildNode(hist,
                    gradients,
                    hessians,
                    weights,
                    row_indices,
                    row_begin,
                    left_end,
                    depth + 1,
                    options,
                    gpu_workspace,
                    &direct_child_snapshot,
                    true,
                    nullptr,
                    direct_child_histogram_ms,
                    profiler,
                    statistic_engine,
                    leaf_row_ranges_out,
                    leaf_count);
      nodes_[node_index].right_child =
          BuildNode(hist,
                    gradients,
                    hessians,
                    weights,
                    row_indices,
                    left_end,
                    row_end,
                    depth + 1,
                    options,
                    gpu_workspace,
                    &sibling_child_snapshot,
                    false,
                    nullptr,
                    sibling_child_histogram_ms,
                    profiler,
                    statistic_engine,
                    leaf_row_ranges_out,
                    leaf_count);
    } else {
      nodes_[node_index].right_child =
          BuildNode(hist,
                    gradients,
                    hessians,
                    weights,
                    row_indices,
                    left_end,
                    row_end,
                    depth + 1,
                    options,
                    gpu_workspace,
                    &direct_child_snapshot,
                    true,
                    nullptr,
                    direct_child_histogram_ms,
                    profiler,
                    statistic_engine,
                    leaf_row_ranges_out,
                    leaf_count);
      nodes_[node_index].left_child =
          BuildNode(hist,
                    gradients,
                    hessians,
                    weights,
                    row_indices,
                    row_begin,
                    left_end,
                    depth + 1,
                    options,
                    gpu_workspace,
                    &sibling_child_snapshot,
                    false,
                    nullptr,
                    sibling_child_histogram_ms,
                    profiler,
                    statistic_engine,
                    leaf_row_ranges_out,
                    leaf_count);
    }
    return node_index;
  }

  NodeHistogramSet node_stats;
  double histogram_ms = precomputed_histogram_ms;
  if (precomputed_node_stats != nullptr) {
    node_stats = *precomputed_node_stats;
  } else {
    const auto histogram_start = std::chrono::steady_clock::now();
    node_stats = ComputeNodeHistogramSet(
        hist,
        gradients,
        hessians,
        weights,
        row_indices,
        row_begin,
        row_end,
        options.use_gpu,
        gpu_workspace);
    histogram_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - histogram_start)
            .count();
  }
  if (profiler != nullptr && profiler->enabled()) {
    profiler->LogNodeHistogram(depth, row_count, options.use_gpu, histogram_ms);
  }

  Node node;
  node.leaf_weight =
      ComputeLeafWeight(node_stats.total_gradient, node_stats.total_hessian, options.lambda_l2);

  const int node_index = static_cast<int>(nodes_.size());
  nodes_.push_back(node);
  if (leaf_row_ranges_out != nullptr && leaf_row_ranges_out->size() < nodes_.size()) {
    leaf_row_ranges_out->resize(nodes_.size());
  }

  if (row_count <= 1 || depth >= options.max_depth ||
      (options.max_leaves > 0 && leaf_count != nullptr && *leaf_count >= options.max_leaves)) {
    if (leaf_row_ranges_out != nullptr) {
      (*leaf_row_ranges_out)[node_index] = LeafRowRange{row_begin, row_end};
    }
    return node_index;
  }

  const auto feature_start = std::chrono::steady_clock::now();
  const FeatureChoice feature_choice =
      SelectBestFeature(node_stats, statistic_engine, options.allowed_features);
  const double feature_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - feature_start)
          .count();
  if (feature_choice.feature_id < 0 || feature_choice.p_value > options.alpha) {
    if (profiler != nullptr && profiler->enabled()) {
      profiler->LogNodeSearch(depth,
                              row_count,
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
    if (leaf_row_ranges_out != nullptr) {
      (*leaf_row_ranges_out)[node_index] = LeafRowRange{row_begin, row_end};
    }
    return node_index;
  }

  const auto split_start = std::chrono::steady_clock::now();
  const SplitChoice split_choice = SelectBestSplit(
      node_stats.by_feature[static_cast<std::size_t>(feature_choice.feature_id)],
      node_stats.total_gradient,
      node_stats.total_hessian,
      node_stats.sample_weight_sum,
      options.lambda_l2,
      options.min_data_in_leaf,
      options.min_child_weight,
      options.min_split_gain,
      hist.is_categorical(static_cast<std::size_t>(feature_choice.feature_id)));
  const double split_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - split_start)
          .count();
  if (!split_choice.valid || split_choice.gain <= 0.0) {
    if (profiler != nullptr && profiler->enabled()) {
      profiler->LogNodeSearch(depth,
                              row_count,
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
    if (leaf_row_ranges_out != nullptr) {
      (*leaf_row_ranges_out)[node_index] = LeafRowRange{row_begin, row_end};
    }
    return node_index;
  }

  std::size_t left_end = row_begin;
  double partition_ms = 0.0;
  if (options.use_gpu) {
    const auto partition_start = std::chrono::steady_clock::now();
    left_end = PartitionHistogramRowsGpu(gpu_workspace,
                                         row_begin,
                                         row_end,
                                         static_cast<std::size_t>(feature_choice.feature_id),
                                         split_choice.is_categorical,
                                         split_choice.split_bin,
                                         split_choice.left_categories);
    partition_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - partition_start)
            .count();
  } else {
    const auto feature_bins = hist.feature_bins(static_cast<std::size_t>(feature_choice.feature_id));
    const auto left_begin = row_indices.begin() + static_cast<std::ptrdiff_t>(row_begin);
    const auto right_end = row_indices.begin() + static_cast<std::ptrdiff_t>(row_end);
    const auto partition_start = std::chrono::steady_clock::now();
    const auto split_middle = std::partition(left_begin, right_end, [&](std::size_t row) {
      const std::uint16_t bin = feature_bins[row];
      return split_choice.is_categorical ? split_choice.left_categories[bin] != 0
                                         : bin <= split_choice.split_bin;
    });
    partition_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - partition_start)
            .count();
    left_end = row_begin + static_cast<std::size_t>(std::distance(left_begin, split_middle));
  }
  const std::size_t left_count = left_end - row_begin;
  const std::size_t right_count = row_end - left_end;
  if (profiler != nullptr && profiler->enabled()) {
    profiler->LogNodeSearch(depth,
                            row_count,
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
  if (left_end == row_begin || left_end == row_end) {
    if (leaf_row_ranges_out != nullptr) {
      (*leaf_row_ranges_out)[node_index] = LeafRowRange{row_begin, row_end};
    }
    return node_index;
  }

  NodeHistogramSet left_child_stats;
  NodeHistogramSet right_child_stats;
  double left_child_histogram_ms = 0.0;
  double right_child_histogram_ms = 0.0;
  if (left_count <= right_count) {
    const auto direct_child_start = std::chrono::steady_clock::now();
    left_child_stats = ComputeNodeHistogramSet(
        hist,
        gradients,
        hessians,
        weights,
        row_indices,
        row_begin,
        left_end,
        options.use_gpu,
        gpu_workspace);
    left_child_histogram_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - direct_child_start)
            .count();
    const auto subtraction_start = std::chrono::steady_clock::now();
    right_child_stats = SubtractNodeHistogramSet(node_stats, left_child_stats);
    right_child_histogram_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - subtraction_start)
            .count();
  } else {
    const auto direct_child_start = std::chrono::steady_clock::now();
    right_child_stats = ComputeNodeHistogramSet(
        hist,
        gradients,
        hessians,
        weights,
        row_indices,
        left_end,
        row_end,
        options.use_gpu,
        gpu_workspace);
    right_child_histogram_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - direct_child_start)
            .count();
    const auto subtraction_start = std::chrono::steady_clock::now();
    left_child_stats = SubtractNodeHistogramSet(node_stats, right_child_stats);
    left_child_histogram_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - subtraction_start)
            .count();
  }

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
  nodes_[node_index].left_child =
      BuildNode(hist,
                gradients,
                hessians,
                weights,
                row_indices,
                row_begin,
                left_end,
                depth + 1,
                options,
                gpu_workspace,
                nullptr,
                false,
                &left_child_stats,
                left_child_histogram_ms,
                profiler,
                statistic_engine,
                leaf_row_ranges_out,
                leaf_count);
  nodes_[node_index].right_child =
      BuildNode(hist,
                gradients,
                hessians,
                weights,
                row_indices,
                left_end,
                row_end,
                depth + 1,
                options,
                gpu_workspace,
                nullptr,
                false,
                &right_child_stats,
                right_child_histogram_ms,
                profiler,
                statistic_engine,
                leaf_row_ranges_out,
                leaf_count);

  return node_index;
}

float Tree::PredictRow(const Pool& pool, std::size_t row) const {
  const int leaf_index = PredictLeafIndex(pool, row);
  return leaf_index < 0 ? 0.0F : nodes_[leaf_index].leaf_weight;
}

int Tree::PredictLeafIndex(const Pool& pool, std::size_t row) const {
  if (nodes_.empty()) {
    return -1;
  }

  int node_index = 0;
  while (!nodes_[node_index].is_leaf) {
    const Node& node = nodes_[node_index];
    const std::uint16_t bin = BinValue(
        static_cast<std::size_t>(node.split_feature_id),
        pool.feature_value(row, static_cast<std::size_t>(node.split_feature_id)));
    node_index = node.is_categorical_split
                     ? (node.left_categories[bin] != 0 ? node.left_child : node.right_child)
                     : (bin <= node.split_bin_index ? node.left_child : node.right_child);
  }

  return node_index;
}

void Tree::AccumulateContributions(
    const Pool& pool, std::size_t row, float scale, std::vector<float>& row_contributions) const {
  if (row_contributions.empty()) {
    return;
  }
  if (nodes_.empty()) {
    row_contributions.back() += scale * 0.0F;
    return;
  }

  std::vector<int> path_features;
  int node_index = 0;
  while (!nodes_[node_index].is_leaf) {
    const Node& node = nodes_[node_index];
    path_features.push_back(node.split_feature_id);
    const std::uint16_t bin = BinValue(
        static_cast<std::size_t>(node.split_feature_id),
        pool.feature_value(row, static_cast<std::size_t>(node.split_feature_id)));
    node_index = node.is_categorical_split
                     ? (node.left_categories[bin] != 0 ? node.left_child : node.right_child)
                     : (bin <= node.split_bin_index ? node.left_child : node.right_child);
  }

  const float leaf_value = scale * nodes_[node_index].leaf_weight;
  if (path_features.empty()) {
    row_contributions.back() += leaf_value;
    return;
  }

  const float share = leaf_value / static_cast<float>(path_features.size());
  for (const int feature_index : path_features) {
    row_contributions[static_cast<std::size_t>(feature_index)] += share;
  }
}

float Tree::PredictBinnedRow(const HistMatrix& hist, std::size_t row) const {
  const int leaf_index = PredictBinnedLeafIndex(hist, row);
  return leaf_index < 0 ? 0.0F : nodes_[leaf_index].leaf_weight;
}

int Tree::PredictBinnedLeafIndex(const HistMatrix& hist, std::size_t row) const {
  if (nodes_.empty()) {
    return -1;
  }

  int node_index = 0;
  while (!nodes_[node_index].is_leaf) {
    const Node& node = nodes_[node_index];
    const auto feature_bins = hist.feature_bins(static_cast<std::size_t>(node.split_feature_id));
    const std::uint16_t bin = feature_bins[row];
    node_index = node.is_categorical_split
                     ? (node.left_categories[bin] != 0 ? node.left_child : node.right_child)
                     : (bin <= node.split_bin_index ? node.left_child : node.right_child);
  }

  return node_index;
}

std::vector<float> Tree::Predict(const Pool& pool) const {
  std::vector<float> predictions(pool.num_rows(), 0.0F);
  for (std::size_t row = 0; row < pool.num_rows(); ++row) {
    predictions[row] = PredictRow(pool, row);
  }
  return predictions;
}

void Tree::SetLeafWeight(std::size_t node_index, float leaf_weight) {
  if (node_index >= nodes_.size()) {
    throw std::out_of_range("node index is out of bounds");
  }
  if (!nodes_[node_index].is_leaf) {
    throw std::invalid_argument("leaf weight can only be set on leaf nodes");
  }
  nodes_[node_index].leaf_weight = leaf_weight;
}

void Tree::SetQuantizationSchema(const QuantizationSchemaPtr& quantization_schema) {
  quantization_schema_ = quantization_schema;
}

const QuantizationSchemaPtr& Tree::shared_quantization_schema() const noexcept {
  return quantization_schema_;
}

void Tree::LoadState(std::vector<Node> nodes,
                     const QuantizationSchemaPtr& quantization_schema,
                     std::vector<double> feature_importances) {
  nodes_ = std::move(nodes);
  quantization_schema_ = quantization_schema;
  feature_importances_ = std::move(feature_importances);
}

void Tree::LoadState(std::vector<Node> nodes,
                     std::vector<std::uint16_t> num_bins_per_feature,
                     std::vector<std::size_t> cut_offsets,
                     std::vector<float> cut_values,
                     std::vector<std::uint8_t> categorical_mask,
                     std::vector<std::uint8_t> missing_value_mask,
                     std::uint8_t nan_mode,
                     std::vector<double> feature_importances) {
  auto quantization_schema = std::make_shared<QuantizationSchema>();
  quantization_schema->num_bins_per_feature = std::move(num_bins_per_feature);
  quantization_schema->cut_offsets = std::move(cut_offsets);
  quantization_schema->cut_values = std::move(cut_values);
  quantization_schema->categorical_mask = std::move(categorical_mask);
  quantization_schema->missing_value_mask = std::move(missing_value_mask);
  quantization_schema->nan_mode = nan_mode;
  LoadState(std::move(nodes), quantization_schema, std::move(feature_importances));
}

const std::vector<Node>& Tree::nodes() const noexcept { return nodes_; }

const std::vector<std::uint16_t>& Tree::num_bins_per_feature() const {
  return RequireQuantizationSchema(quantization_schema_).num_bins_per_feature;
}

const std::vector<std::size_t>& Tree::cut_offsets() const {
  return RequireQuantizationSchema(quantization_schema_).cut_offsets;
}

const std::vector<float>& Tree::cut_values() const {
  return RequireQuantizationSchema(quantization_schema_).cut_values;
}

const std::vector<std::uint8_t>& Tree::categorical_mask() const {
  return RequireQuantizationSchema(quantization_schema_).categorical_mask;
}

const std::vector<std::uint8_t>& Tree::missing_value_mask() const {
  return RequireQuantizationSchema(quantization_schema_).missing_value_mask;
}

std::uint8_t Tree::nan_mode() const {
  return RequireQuantizationSchema(quantization_schema_).nan_mode;
}

const std::vector<double>& Tree::feature_importances() const noexcept {
  return feature_importances_;
}

std::uint16_t Tree::BinValue(std::size_t feature_index, float value) const {
  return RequireQuantizationSchema(quantization_schema_).bin_value(feature_index, value);
}

}  // namespace ctboost
