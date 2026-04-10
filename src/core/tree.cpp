#include "ctboost/tree.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include "ctboost/cuda_backend.hpp"

namespace ctboost {
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

struct NodeHistogramSet {
  std::vector<BinStatistics> by_feature;
  std::size_t sample_count{0};
  double total_gradient{0.0};
  double total_hessian{0.0};
  double gradient_variance{0.0};
};

float ComputeLeafWeight(double gradient_sum, double hessian_sum, double lambda_l2) {
  return static_cast<float>(-gradient_sum / (hessian_sum + lambda_l2));
}

double ComputeGain(double gradient_sum, double hessian_sum, double lambda_l2) {
  return (gradient_sum * gradient_sum) / (hessian_sum + lambda_l2);
}

NodeHistogramSet ComputeNodeHistogramSet(const HistMatrix& hist,
                                         const std::vector<float>& gradients,
                                         const std::vector<float>& hessians,
                                         const std::vector<std::size_t>& row_indices,
                                         bool use_gpu) {
  NodeHistogramSet stats;
  stats.by_feature.resize(hist.num_cols);
  stats.sample_count = row_indices.size();

  std::vector<float> node_gradients;
  std::vector<float> node_hessians;
  node_gradients.reserve(row_indices.size());
  node_hessians.reserve(row_indices.size());

  double gradient_square_sum = 0.0;
  for (const std::size_t row : row_indices) {
    const double gradient = gradients[row];
    const double hessian = hessians[row];
    stats.total_gradient += gradient;
    stats.total_hessian += hessian;
    gradient_square_sum += gradient * gradient;
    node_gradients.push_back(gradients[row]);
    node_hessians.push_back(hessians[row]);
  }

  if (!row_indices.empty()) {
    const double sample_count = static_cast<double>(row_indices.size());
    const double mean_gradient = stats.total_gradient / sample_count;
    stats.gradient_variance =
        std::max(0.0, gradient_square_sum / sample_count - mean_gradient * mean_gradient);
  }

  if (use_gpu) {
    std::vector<std::uint16_t> node_bins(hist.num_cols * row_indices.size(), 0);
    for (std::size_t feature = 0; feature < hist.num_cols; ++feature) {
      const auto* feature_bins = hist.feature_bins(feature);
      const std::size_t offset = feature * row_indices.size();
      for (std::size_t i = 0; i < row_indices.size(); ++i) {
        node_bins[offset + i] = feature_bins[row_indices[i]];
      }
    }

    std::vector<float> flat_gradient_sums;
    std::vector<float> flat_hessian_sums;
    std::vector<std::uint32_t> flat_counts;
    std::vector<std::size_t> feature_offsets;
    BuildHistogramsGpu(node_bins,
                       row_indices.size(),
                       hist.num_cols,
                       hist.num_bins_per_feature,
                       node_gradients,
                       node_hessians,
                       flat_gradient_sums,
                       flat_hessian_sums,
                       flat_counts,
                       feature_offsets);

    for (std::size_t feature = 0; feature < hist.num_cols; ++feature) {
      const std::size_t begin = feature_offsets[feature];
      const std::size_t end = feature_offsets[feature + 1];
      BinStatistics feature_stats;
      feature_stats.gradient_sums.reserve(end - begin);
      feature_stats.hessian_sums.reserve(end - begin);
      feature_stats.counts.reserve(end - begin);
      for (std::size_t index = begin; index < end; ++index) {
        feature_stats.gradient_sums.push_back(flat_gradient_sums[index]);
        feature_stats.hessian_sums.push_back(flat_hessian_sums[index]);
        feature_stats.counts.push_back(flat_counts[index]);
      }
      stats.by_feature[feature] = std::move(feature_stats);
    }

    return stats;
  }

  for (std::size_t feature = 0; feature < hist.num_cols; ++feature) {
    const std::size_t feature_bins_count = hist.num_bins(feature);
    BinStatistics feature_stats;
    feature_stats.gradient_sums.assign(feature_bins_count, 0.0);
    feature_stats.hessian_sums.assign(feature_bins_count, 0.0);
    feature_stats.counts.assign(feature_bins_count, 0);

    const auto* feature_bins = hist.feature_bins(feature);
    for (const std::size_t row : row_indices) {
      const std::size_t bin = feature_bins[row];
      feature_stats.gradient_sums[bin] += gradients[row];
      feature_stats.hessian_sums[bin] += hessians[row];
      feature_stats.counts[bin] += 1;
    }

    stats.by_feature[feature] = std::move(feature_stats);
  }

  return stats;
}

FeatureChoice SelectBestFeature(const NodeHistogramSet& node_stats,
                                const LinearStatistic& statistic_engine) {
  FeatureChoice best;

  for (std::size_t feature = 0; feature < node_stats.by_feature.size(); ++feature) {
    const BinStatistics& feature_stats = node_stats.by_feature[feature];
    if (feature_stats.counts.size() <= 1) {
      continue;
    }

    const LinearStatisticResult result = statistic_engine.EvaluateFromBinStatistics(
        feature_stats,
        node_stats.total_gradient,
        node_stats.sample_count,
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
                            std::size_t sample_count,
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
    std::size_t left_count = 0;

    for (std::size_t split_bin = 0; split_bin + 1 < feature_bins_count; ++split_bin) {
      left_gradient += feature_stats.gradient_sums[split_bin];
      left_hessian += feature_stats.hessian_sums[split_bin];
      left_count += feature_stats.counts[split_bin];

      const std::size_t right_count = sample_count - left_count;
      if (left_count == 0 || right_count == 0) {
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
    std::uint32_t count{0};
    double weight{0.0};
  };

  std::vector<WeightedBin> active_bins;
  active_bins.reserve(feature_bins_count);
  for (std::size_t bin = 0; bin < feature_bins_count; ++bin) {
    if (feature_stats.counts[bin] == 0) {
      continue;
    }

    const double denominator = feature_stats.hessian_sums[bin] + lambda_l2;
    active_bins.push_back(WeightedBin{
        static_cast<std::uint16_t>(bin),
        feature_stats.gradient_sums[bin],
        feature_stats.hessian_sums[bin],
        feature_stats.counts[bin],
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
  std::size_t left_count = 0;

  for (std::size_t split_index = 0; split_index + 1 < active_bins.size(); ++split_index) {
    left_gradient += active_bins[split_index].gradient;
    left_hessian += active_bins[split_index].hessian;
    left_count += active_bins[split_index].count;

    const std::size_t right_count = sample_count - left_count;
    if (left_count == 0 || right_count == 0) {
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
                 double alpha,
                 int max_depth,
                 double lambda_l2,
                 bool use_gpu) {
  if (gradients.size() != hist.num_rows || hessians.size() != hist.num_rows) {
    throw std::invalid_argument("gradient and hessian sizes must match the histogram row count");
  }

  nodes_.clear();
  num_bins_per_feature_ = hist.num_bins_per_feature;
  cut_offsets_ = hist.cut_offsets;
  cut_values_ = hist.cut_values;
  categorical_mask_ = hist.categorical_mask;
  feature_importances_.assign(hist.num_cols, 0.0);

  std::vector<std::size_t> row_indices(hist.num_rows);
  std::iota(row_indices.begin(), row_indices.end(), 0);

  const LinearStatistic statistic_engine;
  BuildNode(hist,
            gradients,
            hessians,
            row_indices,
            0,
            alpha,
            max_depth,
            lambda_l2,
            use_gpu,
            statistic_engine);
}

int Tree::BuildNode(const HistMatrix& hist,
                    const std::vector<float>& gradients,
                    const std::vector<float>& hessians,
                    const std::vector<std::size_t>& row_indices,
                    int depth,
                    double alpha,
                    int max_depth,
                    double lambda_l2,
                    bool use_gpu,
                    const LinearStatistic& statistic_engine) {
  const NodeHistogramSet node_stats =
      ComputeNodeHistogramSet(hist, gradients, hessians, row_indices, use_gpu);

  Node node;
  node.leaf_weight =
      ComputeLeafWeight(node_stats.total_gradient, node_stats.total_hessian, lambda_l2);

  const int node_index = static_cast<int>(nodes_.size());
  nodes_.push_back(node);

  if (row_indices.size() <= 1 || depth >= max_depth) {
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
      row_indices.size(),
      lambda_l2,
      hist.is_categorical(static_cast<std::size_t>(feature_choice.feature_id)));
  if (!split_choice.valid || split_choice.gain <= 0.0) {
    return node_index;
  }

  std::vector<std::size_t> left_rows;
  std::vector<std::size_t> right_rows;
  left_rows.reserve(row_indices.size());
  right_rows.reserve(row_indices.size());

  const auto* feature_bins = hist.feature_bins(static_cast<std::size_t>(feature_choice.feature_id));
  for (const std::size_t row : row_indices) {
    const std::uint16_t bin = feature_bins[row];
    const bool goes_left = split_choice.is_categorical
                               ? split_choice.left_categories[bin] != 0
                               : bin <= split_choice.split_bin;
    if (goes_left) {
      left_rows.push_back(row);
    } else {
      right_rows.push_back(row);
    }
  }

  if (left_rows.empty() || right_rows.empty()) {
    return node_index;
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
                left_rows,
                depth + 1,
                alpha,
                max_depth,
                lambda_l2,
                use_gpu,
                statistic_engine);
  nodes_[node_index].right_child =
      BuildNode(hist,
                gradients,
                hessians,
                right_rows,
                depth + 1,
                alpha,
                max_depth,
                lambda_l2,
                use_gpu,
                statistic_engine);

  return node_index;
}

float Tree::PredictRow(const Pool& pool, std::size_t row) const {
  if (nodes_.empty()) {
    return 0.0F;
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

  return nodes_[node_index].leaf_weight;
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

const std::vector<Node>& Tree::nodes() const noexcept { return nodes_; }

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
  return hist.bin_value(feature_index, value);
}

}  // namespace ctboost
