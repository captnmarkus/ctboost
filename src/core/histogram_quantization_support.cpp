#include "histogram_internal.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <limits>
#include <string>
#include <thread>

namespace ctboost::detail {
namespace {

constexpr std::size_t kApproxQuantileThresholdRowsDefault = 1048576;
constexpr std::size_t kApproxQuantileSampleSizeDefault = 262144;
constexpr std::size_t kApproxQuantileMinValuesPerFeature = 2048;
constexpr std::size_t kExactSelectionThresholdRowsDefault = 1048576;
constexpr std::size_t kExactLowCardinalityBinsMultiplier = 4;
constexpr std::size_t kExactLowCardinalityFloor = 1024;

std::size_t ParseEnvUnsigned(const char* name, std::size_t default_value) {
  const char* raw_value = std::getenv(name);
  if (raw_value == nullptr || *raw_value == '\0') {
    return default_value;
  }

  try {
    return static_cast<std::size_t>(std::stoull(raw_value));
  } catch (const std::exception&) {
    return default_value;
  }
}

std::vector<std::size_t> BuildQuantileCutIndices(std::size_t value_count,
                                                 std::size_t max_bins) {
  if (value_count == 0) {
    return {};
  }

  const std::size_t desired_bins = std::min(max_bins, value_count);
  std::vector<std::size_t> cut_indices;
  cut_indices.reserve(desired_bins > 0 ? desired_bins - 1 : 0);
  std::size_t previous_index = value_count;

  for (std::size_t bin = 1; bin < desired_bins; ++bin) {
    const std::size_t index =
        (bin * value_count) / desired_bins >= value_count ? value_count - 1
                                                          : (bin * value_count) / desired_bins;
    if (index != previous_index) {
      cut_indices.push_back(index);
      previous_index = index;
    }
  }

  return cut_indices;
}

void SelectQuantileValues(std::vector<float>& values,
                          const std::vector<std::size_t>& cut_indices,
                          std::size_t cut_begin,
                          std::size_t cut_end,
                          std::size_t value_begin,
                          std::size_t value_end) {
  if (cut_begin >= cut_end || value_begin >= value_end) {
    return;
  }

  const std::size_t cut_mid = cut_begin + (cut_end - cut_begin) / 2;
  const std::size_t nth_index = cut_indices[cut_mid];
  std::nth_element(values.begin() + static_cast<std::ptrdiff_t>(value_begin),
                   values.begin() + static_cast<std::ptrdiff_t>(nth_index),
                   values.begin() + static_cast<std::ptrdiff_t>(value_end));

  SelectQuantileValues(values, cut_indices, cut_begin, cut_mid, value_begin, nth_index);
  SelectQuantileValues(
      values, cut_indices, cut_mid + 1, cut_end, nth_index + 1, value_end);
}

std::vector<float> ComputeSortedQuantileCuts(std::vector<float> values, std::size_t max_bins) {
  if (values.empty()) {
    return {};
  }

  std::sort(values.begin(), values.end());
  const std::vector<std::size_t> cut_indices = BuildQuantileCutIndices(values.size(), max_bins);
  std::vector<float> cuts;
  cuts.reserve(cut_indices.size());

  for (const std::size_t index : cut_indices) {
    const float cut = values[index];
    if (cuts.empty() || cut > cuts.back()) {
      cuts.push_back(cut);
    }
  }
  return cuts;
}

std::vector<float> ComputeSelectedQuantileCuts(std::vector<float> values, std::size_t max_bins) {
  if (values.empty()) {
    return {};
  }

  const std::vector<std::size_t> cut_indices = BuildQuantileCutIndices(values.size(), max_bins);
  std::vector<float> cuts;
  cuts.reserve(cut_indices.size());
  if (cut_indices.empty()) {
    return cuts;
  }

  SelectQuantileValues(values, cut_indices, 0, cut_indices.size(), 0, values.size());

  for (const std::size_t index : cut_indices) {
    const float cut = values[index];
    if (cuts.empty() || cut > cuts.back()) {
      cuts.push_back(cut);
    }
  }
  return cuts;
}

std::size_t ResolveExactSelectionThresholdRows() {
  return ParseEnvUnsigned(
      "CTBOOST_HIST_EXACT_SELECT_THRESHOLD_ROWS", kExactSelectionThresholdRowsDefault);
}

std::size_t ResolveApproximationThresholdRows() {
  return ParseEnvUnsigned(
      "CTBOOST_HIST_APPROX_THRESHOLD_ROWS", kApproxQuantileThresholdRowsDefault);
}

std::size_t ResolveApproximationSampleSize() {
  return ParseEnvUnsigned(
      "CTBOOST_HIST_APPROX_SAMPLE_SIZE", kApproxQuantileSampleSizeDefault);
}

std::vector<std::size_t> BuildApproximationRowSample(std::size_t num_rows,
                                                     std::size_t sample_size) {
  sample_size = std::min(num_rows, sample_size);
  if (sample_size == 0 || sample_size >= num_rows) {
    return {};
  }

  std::vector<std::size_t> sampled_rows;
  sampled_rows.reserve(sample_size);
  for (std::size_t index = 0; index < sample_size; ++index) {
    sampled_rows.push_back((index * num_rows) / sample_size);
  }
  return sampled_rows;
}

}  // namespace

bool ValidateMaxBins(std::size_t max_bins) {
  return max_bins > 0 &&
         max_bins <= static_cast<std::size_t>(std::numeric_limits<std::uint16_t>::max());
}

std::vector<float> NormalizeCustomBorders(std::vector<float> cuts) {
  if (cuts.empty()) {
    return {};
  }

  for (const float cut : cuts) {
    if (std::isnan(cut)) {
      throw std::invalid_argument("feature_borders cannot contain NaN cut values");
    }
  }

  std::sort(cuts.begin(), cuts.end());
  cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
  return cuts;
}

std::size_t ResolveHistogramThreadCount(std::size_t num_features) {
  if (num_features <= 1) {
    return 1;
  }

  const std::size_t configured_threads = ParseEnvUnsigned("CTBOOST_HIST_THREADS", 0);
  const std::size_t hardware_threads =
      std::max<std::size_t>(1, std::thread::hardware_concurrency());
  const std::size_t thread_count =
      configured_threads == 0 ? hardware_threads : std::max<std::size_t>(1, configured_threads);
  return std::min(num_features, thread_count);
}

HistogramBuildContext ResolveHistogramBuildContext(std::size_t num_rows) {
  HistogramBuildContext context;
  context.approx_threshold_rows = ResolveApproximationThresholdRows();
  context.approx_sample_size = ResolveApproximationSampleSize();
  if (context.approx_threshold_rows == 0 || context.approx_sample_size == 0 ||
      num_rows <= context.approx_threshold_rows || context.approx_sample_size >= num_rows) {
    return context;
  }

  context.approx_row_sample = BuildApproximationRowSample(num_rows, context.approx_sample_size);
  context.approx_sample_size = context.approx_row_sample.size();
  return context;
}

bool ShouldApproximateQuantiles(const HistogramBuildContext& context,
                                std::size_t non_missing_count) {
  return context.approx_threshold_rows > 0 && context.approx_sample_size > 0 &&
         !context.approx_row_sample.empty() && non_missing_count > context.approx_threshold_rows &&
         context.approx_sample_size < non_missing_count;
}

bool HasSufficientApproximationValues(std::size_t sample_count,
                                      std::size_t non_missing_count,
                                      std::size_t max_bins) {
  const std::size_t minimum_values = std::min(
      non_missing_count, std::max(kApproxQuantileMinValuesPerFeature, max_bins * 8U));
  return sample_count >= minimum_values;
}

std::size_t ResolveExactLowCardinalityLimit(std::size_t max_bins) {
  return std::max(kExactLowCardinalityFloor, max_bins * kExactLowCardinalityBinsMultiplier);
}

ExactQuantileStrategy ResolveExactQuantileStrategy(std::size_t non_missing_count,
                                                   std::size_t observed_unique_count,
                                                   bool unique_count_capped,
                                                   std::size_t max_bins) {
  if (non_missing_count < ResolveExactSelectionThresholdRows()) {
    return ExactQuantileStrategy::Sort;
  }

  const std::size_t low_cardinality_limit = ResolveExactLowCardinalityLimit(max_bins);
  if (!unique_count_capped && observed_unique_count <= low_cardinality_limit) {
    return ExactQuantileStrategy::Sort;
  }
  return ExactQuantileStrategy::Selection;
}

std::vector<float> ComputeUniformCuts(const std::vector<float>& values, std::size_t max_bins) {
  if (values.empty()) {
    return {};
  }

  const auto minmax = std::minmax_element(values.begin(), values.end());
  const float min_value = *minmax.first;
  const float max_value = *minmax.second;
  if (!(max_value > min_value)) {
    return {};
  }

  const std::size_t desired_bins = std::min(max_bins, values.size());
  std::vector<float> cuts;
  cuts.reserve(desired_bins > 0 ? desired_bins - 1 : 0);
  const double step =
      (static_cast<double>(max_value) - static_cast<double>(min_value)) /
      static_cast<double>(desired_bins);
  for (std::size_t bin = 1; bin < desired_bins; ++bin) {
    const float cut =
        static_cast<float>(static_cast<double>(min_value) + step * static_cast<double>(bin));
    if (cuts.empty() || cut > cuts.back()) {
      cuts.push_back(cut);
    }
  }
  return cuts;
}

std::vector<float> ComputeQuantileCuts(std::vector<float> values,
                                       std::size_t max_bins,
                                       ExactQuantileStrategy strategy) {
  return strategy == ExactQuantileStrategy::Selection
             ? ComputeSelectedQuantileCuts(std::move(values), max_bins)
             : ComputeSortedQuantileCuts(std::move(values), max_bins);
}

}  // namespace ctboost::detail
