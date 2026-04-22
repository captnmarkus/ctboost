#include "tree_internal.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ctboost::detail {
namespace {

std::uint64_t MixRandomKey(std::uint64_t value) {
  value += 0x9E3779B97F4A7C15ULL;
  value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
  return value ^ (value >> 31U);
}

double UniformFromKey(std::uint64_t key) {
  constexpr double kScale = 1.0 / static_cast<double>(1ULL << 53U);
  return static_cast<double>(MixRandomKey(key) >> 11U) * kScale;
}

double SymmetricNoise(std::uint64_t base_seed, std::uint64_t key, double scale) {
  if (scale <= 0.0) {
    return 0.0;
  }
  const double uniform = UniformFromKey(base_seed ^ key);
  return (2.0 * uniform - 1.0) * scale;
}

double FeatureWeightValue(const std::vector<double>* feature_weights, int feature_id) {
  if (feature_weights == nullptr || feature_id < 0 ||
      static_cast<std::size_t>(feature_id) >= feature_weights->size()) {
    return 1.0;
  }
  return (*feature_weights)[static_cast<std::size_t>(feature_id)];
}

double FirstUsePenaltyValue(const std::vector<double>* first_feature_use_penalties,
                            const std::vector<std::uint8_t>* model_feature_used_mask,
                            int feature_id) {
  if (first_feature_use_penalties == nullptr || model_feature_used_mask == nullptr ||
      feature_id < 0) {
    return 0.0;
  }
  const std::size_t index = static_cast<std::size_t>(feature_id);
  if (index >= first_feature_use_penalties->size() || index >= model_feature_used_mask->size() ||
      (*model_feature_used_mask)[index] != 0U) {
    return 0.0;
  }
  return (*first_feature_use_penalties)[index];
}

}  // namespace

double AdjustedCandidateGain(const TreeBuildOptions& options,
                             int feature_id,
                             double raw_gain,
                             int depth,
                             std::size_t row_begin,
                             std::size_t row_end) {
  const double feature_weight = FeatureWeightValue(options.feature_weights, feature_id);
  if (feature_weight <= 0.0) {
    return -std::numeric_limits<double>::infinity();
  }

  double adjusted_gain = raw_gain * feature_weight;
  adjusted_gain -= FirstUsePenaltyValue(
      options.first_feature_use_penalties, options.model_feature_used_mask, feature_id);
  adjusted_gain += SymmetricNoise(
      options.random_seed,
      static_cast<std::uint64_t>(feature_id + 1) ^
          (static_cast<std::uint64_t>(depth + 1) << 24U) ^
          (static_cast<std::uint64_t>(row_begin + 1) << 1U) ^
          (static_cast<std::uint64_t>(row_end + 1) << 33U),
      options.random_strength);
  return adjusted_gain;
}

SplitChoice SelectBestSplit(const BinStatistics& feature_stats,
                            double total_gradient,
                            double total_hessian,
                            double sample_weight_sum,
                            double lambda_l2,
                            int min_data_in_leaf,
                            double min_child_weight,
                            double min_split_gain,
                            bool is_categorical,
                            int monotone_sign,
                            double leaf_lower_bound,
                            double leaf_upper_bound) {
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
      const double left_leaf_weight = ClampLeafWeight(
          ComputeLeafWeight(left_gradient, left_hessian, lambda_l2), leaf_lower_bound, leaf_upper_bound);
      const double right_leaf_weight = ClampLeafWeight(
          ComputeLeafWeight(right_gradient, right_hessian, lambda_l2), leaf_lower_bound, leaf_upper_bound);
      if ((monotone_sign > 0 && left_leaf_weight > right_leaf_weight + 1e-12) ||
          (monotone_sign < 0 && left_leaf_weight + 1e-12 < right_leaf_weight)) {
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
        best.left_leaf_weight = left_leaf_weight;
        best.right_leaf_weight = right_leaf_weight;
      }
    }
    return best;
  }

  if (monotone_sign != 0) {
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
    const double left_leaf_weight = ClampLeafWeight(
        ComputeLeafWeight(left_gradient, left_hessian, lambda_l2), leaf_lower_bound, leaf_upper_bound);
    const double right_leaf_weight = ClampLeafWeight(
        ComputeLeafWeight(right_gradient, right_hessian, lambda_l2), leaf_lower_bound, leaf_upper_bound);
    const double gain = ComputeGain(left_gradient, left_hessian, lambda_l2) +
                        ComputeGain(right_gradient, right_hessian, lambda_l2) - parent_gain;
    if (gain <= min_split_gain) {
      continue;
    }

    if (!best.valid || gain > best.gain) {
      best.valid = true;
      best.is_categorical = true;
      best.gain = gain;
      best.left_leaf_weight = left_leaf_weight;
      best.right_leaf_weight = right_leaf_weight;
      best.left_categories.fill(0);
      for (std::size_t left_index = 0; left_index <= split_index; ++left_index) {
        best.left_categories[active_bins[left_index].bin] = 1;
      }
    }
  }
  return best;
}

}  // namespace ctboost::detail
