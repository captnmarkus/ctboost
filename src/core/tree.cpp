#include "ctboost/tree.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace ctboost {
namespace {

struct FeatureChoice {
  int feature_id{-1};
  double p_value{1.0};
  double chi_square{-std::numeric_limits<double>::infinity()};
};

struct SplitChoice {
  bool valid{false};
  std::uint16_t split_bin{0};
  double gain{-std::numeric_limits<double>::infinity()};
};

float ComputeLeafWeight(double gradient_sum, double hessian_sum, double lambda_l2) {
  return static_cast<float>(-gradient_sum / (hessian_sum + lambda_l2));
}

double ComputeGain(double gradient_sum, double hessian_sum, double lambda_l2) {
  return (gradient_sum * gradient_sum) / (hessian_sum + lambda_l2);
}

FeatureChoice SelectBestFeature(const HistMatrix& hist,
                                const std::vector<float>& gradients,
                                const std::vector<float>& hessians,
                                const std::vector<std::size_t>& row_indices,
                                const LinearStatistic& statistic_engine) {
  FeatureChoice best;

  for (std::size_t feature = 0; feature < hist.num_cols; ++feature) {
    const std::size_t feature_bins_count = hist.num_bins(feature);
    if (feature_bins_count <= 1) {
      continue;
    }

    const auto* feature_bins = hist.feature_bins(feature);
    std::vector<float> node_gradients;
    std::vector<float> node_hessians;
    std::vector<std::uint16_t> node_bins;
    node_gradients.reserve(row_indices.size());
    node_hessians.reserve(row_indices.size());
    node_bins.reserve(row_indices.size());

    for (const std::size_t row : row_indices) {
      node_gradients.push_back(gradients[row]);
      node_hessians.push_back(hessians[row]);
      node_bins.push_back(feature_bins[row]);
    }

    const LinearStatisticResult result =
        statistic_engine.Evaluate(node_gradients, node_hessians, node_bins, feature_bins_count);
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

SplitChoice SelectBestSplit(const HistMatrix& hist,
                            const std::vector<float>& gradients,
                            const std::vector<float>& hessians,
                            const std::vector<std::size_t>& row_indices,
                            std::size_t feature_id,
                            double lambda_l2) {
  SplitChoice best;
  const std::size_t feature_bins_count = hist.num_bins(feature_id);
  if (feature_bins_count <= 1) {
    return best;
  }

  std::vector<double> gradient_sums(feature_bins_count, 0.0);
  std::vector<double> hessian_sums(feature_bins_count, 0.0);
  std::vector<std::size_t> counts(feature_bins_count, 0);
  const auto* feature_bins = hist.feature_bins(feature_id);

  double total_gradient = 0.0;
  double total_hessian = 0.0;
  for (const std::size_t row : row_indices) {
    const std::size_t bin = feature_bins[row];
    gradient_sums[bin] += gradients[row];
    hessian_sums[bin] += hessians[row];
    counts[bin] += 1;
    total_gradient += gradients[row];
    total_hessian += hessians[row];
  }

  const double parent_gain = ComputeGain(total_gradient, total_hessian, lambda_l2);
  double left_gradient = 0.0;
  double left_hessian = 0.0;
  std::size_t left_count = 0;

  for (std::size_t split_bin = 0; split_bin + 1 < feature_bins_count; ++split_bin) {
    left_gradient += gradient_sums[split_bin];
    left_hessian += hessian_sums[split_bin];
    left_count += counts[split_bin];

    const std::size_t right_count = row_indices.size() - left_count;
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

}  // namespace

void Tree::Build(const HistMatrix& hist,
                 const std::vector<float>& gradients,
                 const std::vector<float>& hessians,
                 double alpha,
                 int max_depth,
                 double lambda_l2) {
  if (gradients.size() != hist.num_rows || hessians.size() != hist.num_rows) {
    throw std::invalid_argument("gradient and hessian sizes must match the histogram row count");
  }

  nodes_.clear();
  num_bins_per_feature_ = hist.num_bins_per_feature;
  cut_offsets_ = hist.cut_offsets;
  cut_values_ = hist.cut_values;
  categorical_mask_ = hist.categorical_mask;

  std::vector<std::size_t> row_indices(hist.num_rows);
  std::iota(row_indices.begin(), row_indices.end(), 0);

  const LinearStatistic statistic_engine;
  BuildNode(hist, gradients, hessians, row_indices, 0, alpha, max_depth, lambda_l2,
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
                    const LinearStatistic& statistic_engine) {
  double gradient_sum = 0.0;
  double hessian_sum = 0.0;
  for (const std::size_t row : row_indices) {
    gradient_sum += gradients[row];
    hessian_sum += hessians[row];
  }

  Node node;
  node.leaf_weight = ComputeLeafWeight(gradient_sum, hessian_sum, lambda_l2);

  const int node_index = static_cast<int>(nodes_.size());
  nodes_.push_back(node);

  if (row_indices.size() <= 1 || depth >= max_depth) {
    return node_index;
  }

  const FeatureChoice feature_choice =
      SelectBestFeature(hist, gradients, hessians, row_indices, statistic_engine);
  if (feature_choice.feature_id < 0 || feature_choice.p_value > alpha) {
    return node_index;
  }

  const SplitChoice split_choice = SelectBestSplit(
      hist, gradients, hessians, row_indices, static_cast<std::size_t>(feature_choice.feature_id),
      lambda_l2);
  if (!split_choice.valid || split_choice.gain <= 0.0) {
    return node_index;
  }

  std::vector<std::size_t> left_rows;
  std::vector<std::size_t> right_rows;
  left_rows.reserve(row_indices.size());
  right_rows.reserve(row_indices.size());

  const auto* feature_bins = hist.feature_bins(static_cast<std::size_t>(feature_choice.feature_id));
  for (const std::size_t row : row_indices) {
    if (feature_bins[row] <= split_choice.split_bin) {
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
  nodes_[node_index].left_child =
      BuildNode(hist, gradients, hessians, left_rows, depth + 1, alpha, max_depth, lambda_l2,
                statistic_engine);
  nodes_[node_index].right_child =
      BuildNode(hist, gradients, hessians, right_rows, depth + 1, alpha, max_depth, lambda_l2,
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
    node_index = bin <= node.split_bin_index ? node.left_child : node.right_child;
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
    node_index = bin <= node.split_bin_index ? node.left_child : node.right_child;
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
