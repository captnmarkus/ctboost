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
    if (gpu_workspace == nullptr) {
      throw std::invalid_argument("GPU histogram workspace must be provided when task_type='GPU'");
    }

    std::vector<float> flat_gradient_sums;
    std::vector<float> flat_hessian_sums;
    std::vector<float> flat_weight_sums;
    std::vector<std::size_t> feature_offsets;
    GpuNodeStatistics gpu_node_stats;
    BuildHistogramsGpu(gpu_workspace,
                       row_indices,
                       row_begin,
                       row_end,
                       flat_gradient_sums,
                       flat_hessian_sums,
                       flat_weight_sums,
                       feature_offsets,
                       &gpu_node_stats);
    stats.sample_weight_sum = gpu_node_stats.sample_weight_sum;
    stats.total_gradient = gpu_node_stats.total_gradient;
    stats.total_hessian = gpu_node_stats.total_hessian;
    stats.gradient_square_sum = gpu_node_stats.gradient_square_sum;
    stats.gradient_variance = ComputeGradientVariance(
        stats.total_gradient, stats.gradient_square_sum, stats.sample_weight_sum);

    for (std::size_t feature = 0; feature < hist.num_cols; ++feature) {
      const std::size_t begin = feature_offsets[feature];
      const std::size_t end = feature_offsets[feature + 1];
      BinStatistics feature_stats;
      feature_stats.gradient_sums.reserve(end - begin);
      feature_stats.hessian_sums.reserve(end - begin);
      feature_stats.weight_sums.reserve(end - begin);
      for (std::size_t index = begin; index < end; ++index) {
        feature_stats.gradient_sums.push_back(flat_gradient_sums[index]);
        feature_stats.hessian_sums.push_back(flat_hessian_sums[index]);
        feature_stats.weight_sums.push_back(flat_weight_sums[index]);
      }
      stats.by_feature[feature] = std::move(feature_stats);
    }

    return stats;
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

    const auto* feature_bins = hist.feature_bins(feature);
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
                                const LinearStatistic& statistic_engine) {
  FeatureChoice best;

  for (std::size_t feature = 0; feature < node_stats.by_feature.size(); ++feature) {
    const BinStatistics& feature_stats = node_stats.by_feature[feature];
    if (feature_stats.weight_sums.size() <= 1) {
      continue;
    }

    const auto result = statistic_engine.EvaluateScoreFromBinStatistics(
        feature_stats,
        node_stats.total_gradient,
        node_stats.sample_weight_sum,
        node_stats.gradient_variance);
    if (result.degrees_of_freedom == 0) {
      continue;
    }

    if (best.feature_id < 0 || result.p_value < best.p_value ||
        (std::abs(result.p_value - best.p_value) <= 1e-12 &&
         result.chi_square > best.chi_square)) {
      best.feature_id = static_cast<int>(feature);
      best.p_value = result.p_value;
      best.chi_square = result.chi_square;
    }
  }

  return best;
}

SplitChoice SelectBestSplit(const BinStatistics& feature_stats,
                            double total_gradient,
                            double total_hessian,
                            double sample_weight_sum,
                            double lambda_l2,
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
      if (left_count <= 0.0 || right_count <= 0.0) {
        continue;
      }

      const double right_gradient = total_gradient - left_gradient;
      const double right_hessian = total_hessian - left_hessian;
      const double gain = ComputeGain(left_gradient, left_hessian, lambda_l2) +
                          ComputeGain(right_gradient, right_hessian, lambda_l2) - parent_gain;

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
    if (left_count <= 0.0 || right_count <= 0.0) {
      continue;
    }

    const double right_gradient = total_gradient - left_gradient;
    const double right_hessian = total_hessian - left_hessian;
    const double gain = ComputeGain(left_gradient, left_hessian, lambda_l2) +
                        ComputeGain(right_gradient, right_hessian, lambda_l2) - parent_gain;

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
                 double alpha,
                 int max_depth,
                 double lambda_l2,
                 bool use_gpu,
                 GpuHistogramWorkspace* gpu_workspace,
                 const TrainingProfiler* profiler) {
  if (weights.size() != hist.num_rows) {
    throw std::invalid_argument("weight size must match the histogram row count");
  }
  if (use_gpu) {
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
  num_bins_per_feature_ = hist.num_bins_per_feature;
  cut_offsets_ = hist.cut_offsets;
  cut_values_ = hist.cut_values;
  categorical_mask_ = hist.categorical_mask;
  missing_value_mask_ = hist.missing_value_mask;
  nan_mode_ = hist.nan_mode;
  feature_importances_.assign(hist.num_cols, 0.0);

  std::vector<std::size_t> row_indices(hist.num_rows);
  std::iota(row_indices.begin(), row_indices.end(), 0);

  const LinearStatistic statistic_engine;
  BuildNode(hist,
            gradients,
            hessians,
            weights,
            row_indices,
            0,
            row_indices.size(),
            0,
            alpha,
            max_depth,
            lambda_l2,
            use_gpu,
            gpu_workspace,
            nullptr,
            0.0,
            profiler,
            statistic_engine);
}

int Tree::BuildNode(const HistMatrix& hist,
                    const std::vector<float>& gradients,
                    const std::vector<float>& hessians,
                    const std::vector<float>& weights,
                    std::vector<std::size_t>& row_indices,
                    std::size_t row_begin,
                    std::size_t row_end,
                    int depth,
                    double alpha,
                    int max_depth,
                    double lambda_l2,
                    bool use_gpu,
                    GpuHistogramWorkspace* gpu_workspace,
                    const NodeHistogramSet* precomputed_node_stats,
                    double precomputed_histogram_ms,
                    const TrainingProfiler* profiler,
                    const LinearStatistic& statistic_engine) {
  const std::size_t row_count = row_end - row_begin;
  NodeHistogramSet node_stats;
  double histogram_ms = precomputed_histogram_ms;
  if (precomputed_node_stats != nullptr) {
    node_stats = *precomputed_node_stats;
  } else {
    const auto histogram_start = std::chrono::steady_clock::now();
    node_stats = ComputeNodeHistogramSet(
        hist, gradients, hessians, weights, row_indices, row_begin, row_end, use_gpu, gpu_workspace);
    histogram_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - histogram_start)
            .count();
  }
  if (profiler != nullptr && profiler->enabled()) {
    profiler->LogNodeHistogram(depth, row_count, use_gpu, histogram_ms);
  }

  Node node;
  node.leaf_weight =
      ComputeLeafWeight(node_stats.total_gradient, node_stats.total_hessian, lambda_l2);

  const int node_index = static_cast<int>(nodes_.size());
  nodes_.push_back(node);

  if (row_count <= 1 || depth >= max_depth) {
    return node_index;
  }

  const FeatureChoice feature_choice = SelectBestFeature(node_stats, statistic_engine);
  if (feature_choice.feature_id < 0 || feature_choice.p_value > alpha) {
    return node_index;
  }

  const SplitChoice split_choice = SelectBestSplit(
      node_stats.by_feature[static_cast<std::size_t>(feature_choice.feature_id)],
      node_stats.total_gradient,
      node_stats.total_hessian,
      node_stats.sample_weight_sum,
      lambda_l2,
      hist.is_categorical(static_cast<std::size_t>(feature_choice.feature_id)));
  if (!split_choice.valid || split_choice.gain <= 0.0) {
    return node_index;
  }

  const auto* feature_bins = hist.feature_bins(static_cast<std::size_t>(feature_choice.feature_id));
  const auto left_begin = row_indices.begin() + static_cast<std::ptrdiff_t>(row_begin);
  const auto right_end = row_indices.begin() + static_cast<std::ptrdiff_t>(row_end);
  const auto split_middle = std::partition(left_begin, right_end, [&](std::size_t row) {
    const std::uint16_t bin = feature_bins[row];
    return split_choice.is_categorical ? split_choice.left_categories[bin] != 0
                                       : bin <= split_choice.split_bin;
  });
  const std::size_t left_end =
      row_begin + static_cast<std::size_t>(std::distance(left_begin, split_middle));
  if (left_end == row_begin || left_end == row_end) {
    return node_index;
  }
  const std::size_t left_count = left_end - row_begin;
  const std::size_t right_count = row_end - left_end;

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
        use_gpu,
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
        use_gpu,
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
  nodes_[node_index].left_child =
      BuildNode(hist,
                gradients,
                hessians,
                weights,
                row_indices,
                row_begin,
                left_end,
                depth + 1,
                alpha,
                max_depth,
                lambda_l2,
                use_gpu,
                gpu_workspace,
                &left_child_stats,
                left_child_histogram_ms,
                profiler,
                statistic_engine);
  nodes_[node_index].right_child =
      BuildNode(hist,
                gradients,
                hessians,
                weights,
                row_indices,
                left_end,
                row_end,
                depth + 1,
                alpha,
                max_depth,
                lambda_l2,
                use_gpu,
                gpu_workspace,
                &right_child_stats,
                right_child_histogram_ms,
                profiler,
                statistic_engine);

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
  if (nodes_.empty()) {
    return 0.0F;
  }

  int node_index = 0;
  while (!nodes_[node_index].is_leaf) {
    const Node& node = nodes_[node_index];
    const auto* feature_bins = hist.feature_bins(static_cast<std::size_t>(node.split_feature_id));
    const std::uint16_t bin = feature_bins[row];
    node_index = node.is_categorical_split
                     ? (node.left_categories[bin] != 0 ? node.left_child : node.right_child)
                     : (bin <= node.split_bin_index ? node.left_child : node.right_child);
  }

  return nodes_[node_index].leaf_weight;
}

std::vector<float> Tree::Predict(const Pool& pool) const {
  std::vector<float> predictions(pool.num_rows(), 0.0F);
  for (std::size_t row = 0; row < pool.num_rows(); ++row) {
    predictions[row] = PredictRow(pool, row);
  }
  return predictions;
}

void Tree::LoadState(std::vector<Node> nodes,
                     std::vector<std::uint16_t> num_bins_per_feature,
                     std::vector<std::size_t> cut_offsets,
                     std::vector<float> cut_values,
                     std::vector<std::uint8_t> categorical_mask,
                     std::vector<std::uint8_t> missing_value_mask,
                     std::uint8_t nan_mode,
                     std::vector<double> feature_importances) {
  nodes_ = std::move(nodes);
  num_bins_per_feature_ = std::move(num_bins_per_feature);
  cut_offsets_ = std::move(cut_offsets);
  cut_values_ = std::move(cut_values);
  categorical_mask_ = std::move(categorical_mask);
  missing_value_mask_ = std::move(missing_value_mask);
  nan_mode_ = nan_mode;
  feature_importances_ = std::move(feature_importances);
}

const std::vector<Node>& Tree::nodes() const noexcept { return nodes_; }

const std::vector<std::uint16_t>& Tree::num_bins_per_feature() const noexcept {
  return num_bins_per_feature_;
}

const std::vector<std::size_t>& Tree::cut_offsets() const noexcept { return cut_offsets_; }

const std::vector<float>& Tree::cut_values() const noexcept { return cut_values_; }

const std::vector<std::uint8_t>& Tree::categorical_mask() const noexcept {
  return categorical_mask_;
}

const std::vector<std::uint8_t>& Tree::missing_value_mask() const noexcept {
  return missing_value_mask_;
}

std::uint8_t Tree::nan_mode() const noexcept { return nan_mode_; }

const std::vector<double>& Tree::feature_importances() const noexcept {
  return feature_importances_;
}

std::uint16_t Tree::BinValue(std::size_t feature_index, float value) const {
  HistMatrix hist;
  hist.num_cols = num_bins_per_feature_.size();
  hist.num_bins_per_feature = num_bins_per_feature_;
  hist.cut_offsets = cut_offsets_;
  hist.cut_values = cut_values_;
  hist.categorical_mask = categorical_mask_;
  hist.missing_value_mask = missing_value_mask_;
  hist.nan_mode = nan_mode_;
  return hist.bin_value(feature_index, value);
}

}  // namespace ctboost
