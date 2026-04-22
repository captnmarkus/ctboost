#include "histogram_internal.hpp"

#include <chrono>
#include <cmath>
#include <map>
#include <stdexcept>
#include <unordered_set>

namespace ctboost::detail {
namespace {

constexpr std::size_t kMaxCategoricalBins = 256;

}  // namespace

NanMode ResolveFeatureNanMode(const std::vector<NanMode>& per_feature_modes,
                              const std::vector<std::uint8_t>& per_feature_mode_mask,
                              std::size_t feature_index,
                              NanMode default_mode) {
  if (feature_index < per_feature_mode_mask.size() && per_feature_mode_mask[feature_index] != 0U) {
    return per_feature_modes[feature_index];
  }
  return default_mode;
}

std::size_t ResolveFeatureMaxBins(const std::vector<std::uint16_t>& max_bins_by_feature,
                                  std::size_t feature_index,
                                  std::size_t default_max_bins) {
  if (feature_index < max_bins_by_feature.size() && max_bins_by_feature[feature_index] > 0U) {
    return static_cast<std::size_t>(max_bins_by_feature[feature_index]);
  }
  return default_max_bins;
}

const std::vector<float>* ResolveFeatureBorders(
    const std::vector<std::vector<float>>& feature_borders,
    std::size_t feature_index) noexcept {
  if (feature_index >= feature_borders.size() || feature_borders[feature_index].empty()) {
    return nullptr;
  }
  return &feature_borders[feature_index];
}

std::uint16_t MissingBinIndex(std::size_t bins_for_feature,
                              bool has_missing_values,
                              NanMode nan_mode) {
  if (!has_missing_values) {
    return nan_mode == NanMode::Max && bins_for_feature > 0
               ? static_cast<std::uint16_t>(bins_for_feature - 1)
               : 0;
  }
  return nan_mode == NanMode::Max ? static_cast<std::uint16_t>(bins_for_feature - 1) : 0;
}

FeatureHistogramResult BuildFeatureHistogram(const Pool& pool,
                                             std::size_t feature_max_bins,
                                             NanMode feature_nan_mode,
                                             BorderSelectionMethod border_selection_method,
                                             std::size_t feature,
                                             bool is_categorical,
                                             const HistogramBuildContext& context,
                                             const std::vector<float>* custom_borders) {
  const auto feature_start = std::chrono::steady_clock::now();
  FeatureHistogramResult result;
  result.is_categorical = is_categorical;
  result.nan_mode = feature_nan_mode;
  const std::size_t num_rows = pool.num_rows();

  const float* const contiguous_column = pool.is_sparse() ? nullptr : pool.feature_column_ptr(feature);
  auto feature_at = [&](std::size_t row) -> float {
    return contiguous_column != nullptr ? contiguous_column[row] : pool.feature_value(row, feature);
  };
  bool has_missing_values = false;

  if (result.is_categorical) {
    if (custom_borders != nullptr && !custom_borders->empty()) {
      throw std::invalid_argument(
          "feature_borders can only be specified for numeric features");
    }

    std::map<float, std::uint16_t> category_to_bin;
    for (std::size_t row = 0; row < num_rows; ++row) {
      const float value = feature_at(row);
      if (std::isnan(value)) {
        has_missing_values = true;
        continue;
      }
      category_to_bin.emplace(value, 0);
      if (category_to_bin.size() > kMaxCategoricalBins) {
        throw std::invalid_argument(
            "categorical feature has too many unique categories; maximum supported is 256");
      }
    }

    if (has_missing_values && feature_nan_mode == NanMode::Forbidden) {
      throw std::invalid_argument("NaN values are not allowed when nan_mode='Forbidden'");
    }

    result.missing_value_mask = has_missing_values ? 1U : 0U;
    std::uint16_t next_bin =
        has_missing_values && feature_nan_mode == NanMode::Min ? 1U : 0U;
    result.cut_values.reserve(category_to_bin.size());
    for (auto& [category, mapped_bin] : category_to_bin) {
      mapped_bin = next_bin++;
      result.cut_values.push_back(category);
    }

    result.num_bins =
        static_cast<std::uint16_t>(category_to_bin.size() + (has_missing_values ? 1U : 0U));
    result.elapsed_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - feature_start)
            .count();
    return result;
  }

  std::size_t non_missing_count = 0;
  std::vector<float> approximate_quantile_values;
  approximate_quantile_values.reserve(context.approx_row_sample.size());
  const std::size_t low_cardinality_limit = ResolveExactLowCardinalityLimit(feature_max_bins);
  std::unordered_set<float> observed_unique_values;
  observed_unique_values.reserve(std::min<std::size_t>(low_cardinality_limit + 1U, 4096U));
  bool unique_count_capped = false;
  std::size_t next_sample_index = 0;
  for (std::size_t row = 0; row < num_rows; ++row) {
    const float value = feature_at(row);
    const bool is_missing = std::isnan(value);
    if (is_missing) {
      has_missing_values = true;
    } else {
      ++non_missing_count;
      if (!unique_count_capped) {
        observed_unique_values.insert(value);
        if (observed_unique_values.size() > low_cardinality_limit) {
          unique_count_capped = true;
          observed_unique_values.clear();
        }
      }
    }
    if (next_sample_index < context.approx_row_sample.size() &&
        row == context.approx_row_sample[next_sample_index]) {
      if (!is_missing) {
        approximate_quantile_values.push_back(value);
      }
      ++next_sample_index;
    }
  }

  if (has_missing_values && feature_nan_mode == NanMode::Forbidden) {
    throw std::invalid_argument("NaN values are not allowed when nan_mode='Forbidden'");
  }

  result.missing_value_mask = has_missing_values ? 1U : 0U;
  if (custom_borders != nullptr) {
    result.cut_values = NormalizeCustomBorders(*custom_borders);
    const std::size_t non_missing_bins = non_missing_count == 0 ? 0U : result.cut_values.size() + 1U;
    const std::size_t total_bins = non_missing_bins + (has_missing_values ? 1U : 0U);
    result.num_bins = static_cast<std::uint16_t>(total_bins == 0 ? 1U : total_bins);
    result.elapsed_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - feature_start)
            .count();
    return result;
  }

  std::vector<float> quantile_values;
  const bool use_approximate_quantiles =
      ShouldApproximateQuantiles(context, non_missing_count) &&
      HasSufficientApproximationValues(
          approximate_quantile_values.size(), non_missing_count, feature_max_bins);
  if (use_approximate_quantiles) {
    quantile_values = std::move(approximate_quantile_values);
  } else {
    quantile_values.reserve(non_missing_count);
    for (std::size_t row = 0; row < num_rows; ++row) {
      const float value = feature_at(row);
      if (!std::isnan(value)) {
        quantile_values.push_back(value);
      }
    }
  }

  const ExactQuantileStrategy quantile_strategy =
      use_approximate_quantiles
          ? ExactQuantileStrategy::Sort
          : ResolveExactQuantileStrategy(
                non_missing_count,
                observed_unique_values.size(),
                unique_count_capped,
                feature_max_bins);
  result.cut_values =
      border_selection_method == BorderSelectionMethod::Uniform
          ? ComputeUniformCuts(quantile_values, feature_max_bins)
          : ComputeQuantileCuts(std::move(quantile_values), feature_max_bins, quantile_strategy);
  const std::size_t non_missing_bins = non_missing_count == 0 ? 0U : result.cut_values.size() + 1U;
  const std::size_t total_bins = non_missing_bins + (has_missing_values ? 1U : 0U);

  result.num_bins = static_cast<std::uint16_t>(total_bins == 0 ? 1U : total_bins);
  result.elapsed_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - feature_start)
          .count();
  return result;
}

}  // namespace ctboost::detail
