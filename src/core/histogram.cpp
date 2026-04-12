#include "ctboost/histogram.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "ctboost/profiler.hpp"

namespace ctboost {
namespace {

constexpr std::size_t kMaxCategoricalBins = 256;
constexpr std::size_t kApproxQuantileThresholdRowsDefault = 1048576;
constexpr std::size_t kApproxQuantileSampleSizeDefault = 262144;
constexpr std::size_t kApproxQuantileMinValuesPerFeature = 2048;
constexpr std::size_t kExactSelectionThresholdRowsDefault = 1048576;
constexpr std::size_t kExactLowCardinalityBinsMultiplier = 4;
constexpr std::size_t kExactLowCardinalityFloor = 1024;

bool ValidateMaxBins(std::size_t max_bins) {
  return max_bins > 0 &&
         max_bins <= static_cast<std::size_t>(std::numeric_limits<std::uint16_t>::max());
}

std::string NormalizeToken(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return value;
}

std::size_t ParseEnvUnsigned(const char* name, std::size_t default_value);

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

std::size_t ResolveExactLowCardinalityLimit(std::size_t max_bins) {
  return std::max(kExactLowCardinalityFloor, max_bins * kExactLowCardinalityBinsMultiplier);
}

enum class ExactQuantileStrategy : std::uint8_t {
  Sort = 0,
  Selection = 1,
};

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

std::vector<float> ComputeQuantileCuts(std::vector<float> values,
                                       std::size_t max_bins,
                                       ExactQuantileStrategy strategy) {
  return strategy == ExactQuantileStrategy::Selection
             ? ComputeSelectedQuantileCuts(std::move(values), max_bins)
             : ComputeSortedQuantileCuts(std::move(values), max_bins);
}

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

std::size_t ResolveApproximationThresholdRows() {
  return ParseEnvUnsigned(
      "CTBOOST_HIST_APPROX_THRESHOLD_ROWS", kApproxQuantileThresholdRowsDefault);
}

std::size_t ResolveApproximationSampleSize() {
  return ParseEnvUnsigned(
      "CTBOOST_HIST_APPROX_SAMPLE_SIZE", kApproxQuantileSampleSizeDefault);
}

struct HistogramBuildContext {
  std::size_t approx_threshold_rows{0};
  std::size_t approx_sample_size{0};
  std::vector<std::size_t> approx_row_sample;
};

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

std::uint16_t MissingBinIndex(std::size_t bins_for_feature,
                              bool has_missing_values,
                              NanMode nan_mode);

struct FeatureHistogramResult {
  std::vector<float> cut_values;
  std::uint16_t num_bins{0};
  std::uint8_t missing_value_mask{0};
  bool is_categorical{false};
  double elapsed_ms{0.0};
};

FeatureHistogramResult BuildFeatureHistogram(const Pool& pool,
                                             std::size_t max_bins,
                                             NanMode nan_mode,
                                             std::size_t feature,
                                             HistMatrix& hist,
                                             const HistogramBuildContext& context) {
  const auto feature_start = std::chrono::steady_clock::now();
  FeatureHistogramResult result;
  result.is_categorical = hist.categorical_mask[feature] != 0;

  const float* const contiguous_column = pool.feature_column_ptr(feature);
  auto feature_at = [&](std::size_t row) -> float {
    return contiguous_column != nullptr ? contiguous_column[row] : pool.feature_value(row, feature);
  };
  bool has_missing_values = false;

  if (result.is_categorical) {
    std::map<float, std::uint16_t> category_to_bin;
    for (std::size_t row = 0; row < hist.num_rows; ++row) {
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

    if (has_missing_values && nan_mode == NanMode::Forbidden) {
      throw std::invalid_argument("NaN values are not allowed when nan_mode='Forbidden'");
    }

    result.missing_value_mask = has_missing_values ? 1U : 0U;
    std::uint16_t next_bin = has_missing_values && nan_mode == NanMode::Min ? 1U : 0U;
    result.cut_values.reserve(category_to_bin.size());
    for (auto& [category, mapped_bin] : category_to_bin) {
      mapped_bin = next_bin++;
      result.cut_values.push_back(category);
    }

    const std::uint16_t missing_bin =
        MissingBinIndex(category_to_bin.size() + (has_missing_values ? 1U : 0U),
                        has_missing_values,
                        nan_mode);
    for (std::size_t row = 0; row < hist.num_rows; ++row) {
      const float value = feature_at(row);
      if (std::isnan(value)) {
        hist.set_bin_index(feature, row, missing_bin);
      } else {
        hist.set_bin_index(feature, row, category_to_bin.find(value)->second);
      }
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
  const std::size_t low_cardinality_limit = ResolveExactLowCardinalityLimit(max_bins);
  std::unordered_set<float> observed_unique_values;
  observed_unique_values.reserve(std::min<std::size_t>(low_cardinality_limit + 1U, 4096U));
  bool unique_count_capped = false;
  std::size_t next_sample_index = 0;
  for (std::size_t row = 0; row < hist.num_rows; ++row) {
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

  if (has_missing_values && nan_mode == NanMode::Forbidden) {
    throw std::invalid_argument("NaN values are not allowed when nan_mode='Forbidden'");
  }

  result.missing_value_mask = has_missing_values ? 1U : 0U;

  std::vector<float> quantile_values;
  const bool use_approximate_quantiles =
      ShouldApproximateQuantiles(context, non_missing_count) &&
      HasSufficientApproximationValues(
          approximate_quantile_values.size(), non_missing_count, max_bins);
  if (use_approximate_quantiles) {
    quantile_values = std::move(approximate_quantile_values);
  } else {
    quantile_values.reserve(non_missing_count);
    for (std::size_t row = 0; row < hist.num_rows; ++row) {
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
                non_missing_count, observed_unique_values.size(), unique_count_capped, max_bins);
  const std::vector<float> cuts = ComputeQuantileCuts(
      std::move(quantile_values),
      max_bins,
      quantile_strategy);
  result.cut_values = cuts;
  const std::size_t non_missing_bins = non_missing_count == 0 ? 0U : cuts.size() + 1U;
  const std::uint16_t missing_offset =
      has_missing_values && nan_mode == NanMode::Min && non_missing_bins > 0 ? 1U : 0U;
  const std::size_t total_bins = non_missing_bins + (has_missing_values ? 1U : 0U);
  const std::uint16_t missing_bin =
      MissingBinIndex(total_bins == 0 ? 1U : total_bins, has_missing_values, nan_mode);

  for (std::size_t row = 0; row < hist.num_rows; ++row) {
    const float value = feature_at(row);
    if (std::isnan(value)) {
      hist.set_bin_index(feature, row, missing_bin);
    } else {
      const auto bin_it = std::upper_bound(cuts.begin(), cuts.end(), value);
      hist.set_bin_index(feature,
                         row,
                         static_cast<std::uint16_t>(missing_offset +
                                                    std::distance(cuts.begin(), bin_it)));
    }
  }

  result.num_bins = static_cast<std::uint16_t>(total_bins == 0 ? 1U : total_bins);
  result.elapsed_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - feature_start)
          .count();
  return result;
}

std::vector<std::uint8_t> BuildCategoricalMask(const Pool& pool) {
  std::vector<std::uint8_t> mask(pool.num_cols(), 0);
  for (const int feature_index : pool.cat_features()) {
    mask[static_cast<std::size_t>(feature_index)] = 1;
  }
  return mask;
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

}  // namespace

NanMode ParseNanMode(std::string_view name) {
  const std::string normalized = NormalizeToken(std::string(name));
  if (normalized == "forbidden") {
    return NanMode::Forbidden;
  }
  if (normalized == "min") {
    return NanMode::Min;
  }
  if (normalized == "max") {
    return NanMode::Max;
  }
  throw std::invalid_argument("nan_mode must be one of 'Forbidden', 'Min', or 'Max'");
}

const char* NanModeName(NanMode nan_mode) noexcept {
  switch (nan_mode) {
    case NanMode::Forbidden:
      return "Forbidden";
    case NanMode::Min:
      return "Min";
    case NanMode::Max:
      return "Max";
  }
  return "Min";
}

FeatureBinView HistMatrix::feature_bins(std::size_t feature_index) const {
  if (feature_index >= num_cols) {
    throw std::out_of_range("feature index is out of bounds");
  }
  const std::size_t offset = feature_index * num_rows;
  if (bin_index_bytes == 1) {
    return FeatureBinView{compact_bin_indices.data() + offset, nullptr};
  }
  return FeatureBinView{nullptr, bin_indices.data() + offset};
}

std::uint16_t HistMatrix::bin_at(std::size_t feature_index, std::size_t row) const {
  if (feature_index >= num_cols || row >= num_rows) {
    throw std::out_of_range("feature index or row is out of bounds");
  }
  return feature_bins(feature_index)[row];
}

void HistMatrix::set_bin_index(std::size_t feature_index, std::size_t row, std::uint16_t value) {
  if (feature_index >= num_cols || row >= num_rows) {
    throw std::out_of_range("feature index or row is out of bounds");
  }

  const std::size_t offset = feature_index * num_rows + row;
  if (bin_index_bytes == 1) {
    compact_bin_indices[offset] = static_cast<std::uint8_t>(value);
  } else {
    bin_indices[offset] = value;
  }
}

void HistMatrix::CompactBinStorage() {
  std::uint16_t max_feature_bins = 0;
  for (const std::uint16_t feature_bins_count : num_bins_per_feature) {
    max_feature_bins = std::max(max_feature_bins, feature_bins_count);
  }

  if (bin_indices.empty() ||
      max_feature_bins > static_cast<std::uint16_t>(std::numeric_limits<std::uint8_t>::max())) {
    bin_index_bytes = 2;
    compact_bin_indices.clear();
    compact_bin_indices.shrink_to_fit();
    return;
  }

  compact_bin_indices.resize(bin_indices.size(), 0);
  for (std::size_t index = 0; index < bin_indices.size(); ++index) {
    compact_bin_indices[index] = static_cast<std::uint8_t>(bin_indices[index]);
  }
  bin_indices.clear();
  bin_indices.shrink_to_fit();
  bin_index_bytes = 1;
}

bool HistMatrix::uses_compact_bin_storage() const noexcept { return bin_index_bytes == 1; }

std::uint8_t HistMatrix::bin_storage_bytes() const noexcept { return bin_index_bytes; }

std::size_t HistMatrix::num_bins(std::size_t feature_index) const {
  if (feature_index >= num_cols) {
    throw std::out_of_range("feature index is out of bounds");
  }
  return static_cast<std::size_t>(num_bins_per_feature[feature_index]);
}

bool HistMatrix::is_categorical(std::size_t feature_index) const {
  if (feature_index >= num_cols) {
    throw std::out_of_range("feature index is out of bounds");
  }
  return categorical_mask[feature_index] != 0;
}

bool HistMatrix::has_missing_values(std::size_t feature_index) const {
  if (feature_index >= num_cols) {
    throw std::out_of_range("feature index is out of bounds");
  }
  return missing_value_mask[feature_index] != 0;
}

std::uint16_t HistMatrix::bin_value(std::size_t feature_index, float value) const {
  if (feature_index >= num_cols) {
    throw std::out_of_range("feature index is out of bounds");
  }

  const std::size_t bins_for_feature = num_bins(feature_index);
  if (bins_for_feature == 0) {
    return 0;
  }

  const std::size_t cut_begin = cut_offsets[feature_index];
  const std::size_t cut_end = cut_offsets[feature_index + 1];
  const auto begin = cut_values.begin() + static_cast<std::ptrdiff_t>(cut_begin);
  const auto end = cut_values.begin() + static_cast<std::ptrdiff_t>(cut_end);
  const bool feature_has_missing_values = has_missing_values(feature_index);
  const NanMode resolved_nan_mode = static_cast<NanMode>(nan_mode);

  if (std::isnan(value)) {
    if (resolved_nan_mode == NanMode::Forbidden) {
      throw std::invalid_argument("NaN values are not allowed when nan_mode='Forbidden'");
    }
    return MissingBinIndex(bins_for_feature, feature_has_missing_values, resolved_nan_mode);
  }

  if (is_categorical(feature_index)) {
    const std::size_t non_missing_bins =
        bins_for_feature - (feature_has_missing_values ? 1U : 0U);
    if (non_missing_bins == 0) {
      return MissingBinIndex(bins_for_feature, feature_has_missing_values, resolved_nan_mode);
    }

    const auto it = std::lower_bound(begin, end, value);
    const std::size_t insertion = static_cast<std::size_t>(std::distance(begin, it));
    const std::size_t clamped_insertion = std::min(insertion, non_missing_bins - 1);
    const std::uint16_t offset =
        feature_has_missing_values && resolved_nan_mode == NanMode::Min ? 1U : 0U;
    if (it != end && *it == value) {
      return static_cast<std::uint16_t>(offset + std::distance(begin, it));
    }
    return static_cast<std::uint16_t>(offset + clamped_insertion);
  }

  const std::size_t non_missing_bins =
      bins_for_feature - (feature_has_missing_values ? 1U : 0U);
  if (non_missing_bins == 0) {
    return MissingBinIndex(bins_for_feature, feature_has_missing_values, resolved_nan_mode);
  }

  const auto it = std::upper_bound(begin, end, value);
  const std::uint16_t offset =
      feature_has_missing_values && resolved_nan_mode == NanMode::Min ? 1U : 0U;
  return static_cast<std::uint16_t>(offset + std::distance(begin, it));
}

std::size_t HistMatrix::storage_bytes() const noexcept {
  return bin_indices.capacity() * sizeof(std::uint16_t) +
         compact_bin_indices.capacity() * sizeof(std::uint8_t) +
         num_bins_per_feature.capacity() * sizeof(std::uint16_t) +
         cut_offsets.capacity() * sizeof(std::size_t) +
         cut_values.capacity() * sizeof(float) +
         categorical_mask.capacity() * sizeof(std::uint8_t) +
         missing_value_mask.capacity() * sizeof(std::uint8_t);
}

void HistMatrix::ReleaseStorage() noexcept {
  num_rows = 0;
  num_cols = 0;
  bin_indices.clear();
  bin_indices.shrink_to_fit();
  compact_bin_indices.clear();
  compact_bin_indices.shrink_to_fit();
  bin_index_bytes = 2;
  num_bins_per_feature.clear();
  num_bins_per_feature.shrink_to_fit();
  cut_offsets.clear();
  cut_offsets.shrink_to_fit();
  cut_values.clear();
  cut_values.shrink_to_fit();
  categorical_mask.clear();
  categorical_mask.shrink_to_fit();
  missing_value_mask.clear();
  missing_value_mask.shrink_to_fit();
}

HistBuilder::HistBuilder(std::size_t max_bins, std::string nan_mode)
    : max_bins_(max_bins),
      nan_mode_(ParseNanMode(nan_mode)),
      nan_mode_name_(NanModeName(nan_mode_)) {
  if (!ValidateMaxBins(max_bins_)) {
    throw std::invalid_argument("max_bins must be in the range [1, 65535]");
  }
}

HistMatrix HistBuilder::Build(const Pool& pool, const TrainingProfiler* profiler) const {
  const auto hist_build_start = std::chrono::steady_clock::now();
  HistMatrix hist;
  hist.num_rows = pool.num_rows();
  hist.num_cols = pool.num_cols();
  hist.bin_indices.resize(hist.num_rows * hist.num_cols, 0);
  hist.num_bins_per_feature.resize(hist.num_cols, 0);
  hist.cut_offsets.reserve(hist.num_cols + 1);
  hist.cut_offsets.push_back(0);
  hist.categorical_mask = BuildCategoricalMask(pool);
  hist.missing_value_mask.assign(hist.num_cols, 0);
  hist.nan_mode = static_cast<std::uint8_t>(nan_mode_);
  hist.bin_index_bytes = 2;
  const HistogramBuildContext build_context = ResolveHistogramBuildContext(hist.num_rows);

  std::vector<FeatureHistogramResult> feature_results(hist.num_cols);
  std::exception_ptr first_error;
  std::mutex error_mutex;
  std::atomic<bool> failed{false};
  std::atomic<std::size_t> next_feature{0};
  const std::size_t thread_count = ResolveHistogramThreadCount(hist.num_cols);
  std::vector<std::thread> workers;
  workers.reserve(thread_count);

  auto worker = [&]() {
    while (!failed.load(std::memory_order_relaxed)) {
      const std::size_t feature = next_feature.fetch_add(1, std::memory_order_relaxed);
      if (feature >= hist.num_cols) {
        return;
      }

      try {
        feature_results[feature] = BuildFeatureHistogram(
            pool, max_bins_, nan_mode_, feature, hist, build_context);
      } catch (...) {
        failed.store(true, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(error_mutex);
        if (first_error == nullptr) {
          first_error = std::current_exception();
        }
        return;
      }
    }
  };

  for (std::size_t thread_index = 0; thread_index < thread_count; ++thread_index) {
    workers.emplace_back(worker);
  }
  for (std::thread& worker_thread : workers) {
    worker_thread.join();
  }
  if (first_error != nullptr) {
    std::rethrow_exception(first_error);
  }

  for (std::size_t feature = 0; feature < hist.num_cols; ++feature) {
    const FeatureHistogramResult& feature_result = feature_results[feature];
    hist.missing_value_mask[feature] = feature_result.missing_value_mask;
    hist.num_bins_per_feature[feature] = feature_result.num_bins;
    hist.cut_values.insert(
        hist.cut_values.end(), feature_result.cut_values.begin(), feature_result.cut_values.end());
    hist.cut_offsets.push_back(hist.cut_values.size());
    if (profiler != nullptr && profiler->enabled()) {
      profiler->LogHistogramFeature(feature,
                                    hist.num_rows,
                                    static_cast<std::size_t>(feature_result.num_bins),
                                    feature_result.is_categorical,
                                    feature_result.elapsed_ms);
    }
  }

  if (profiler != nullptr && profiler->enabled()) {
    std::size_t total_bins = 0;
    for (const std::uint16_t bins_for_feature : hist.num_bins_per_feature) {
      total_bins += static_cast<std::size_t>(bins_for_feature);
    }
    const double total_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - hist_build_start)
            .count();
    profiler->LogHistogramBuild(hist.num_rows, hist.num_cols, total_bins, total_ms);
  }

  hist.CompactBinStorage();
  return hist;
}

std::size_t HistBuilder::max_bins() const noexcept { return max_bins_; }

NanMode HistBuilder::nan_mode() const noexcept { return nan_mode_; }

const std::string& HistBuilder::nan_mode_name() const noexcept { return nan_mode_name_; }

}  // namespace ctboost
