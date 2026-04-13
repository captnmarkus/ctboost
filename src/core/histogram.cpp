#include "ctboost/histogram.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
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

std::filesystem::path MakeExternalStorageRoot(const std::string& directory) {
  static std::atomic<std::uint64_t> counter{0};

  std::filesystem::path root =
      directory.empty() ? std::filesystem::temp_directory_path() : std::filesystem::path(directory);
  std::error_code error;
  std::filesystem::create_directories(root, error);
  if (error) {
    throw std::runtime_error(
        "failed to create CTBoost external-memory directory: " + error.message());
  }

  const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
  const std::uint64_t token = counter.fetch_add(1, std::memory_order_relaxed);
  root /= "ctboost-hist-" + std::to_string(static_cast<long long>(now)) + "-" +
          std::to_string(static_cast<unsigned long long>(token));
  std::filesystem::create_directories(root, error);
  if (error) {
    throw std::runtime_error(
        "failed to create CTBoost external-memory spill root: " + error.message());
  }
  return root;
}

template <typename BinType>
void WriteFeatureBinsToFile(const std::filesystem::path& path, const std::vector<BinType>& bins) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("failed to open spill file for writing: " + path.string());
  }
  if (!bins.empty()) {
    out.write(reinterpret_cast<const char*>(bins.data()),
              static_cast<std::streamsize>(bins.size() * sizeof(BinType)));
  }
  if (!out) {
    throw std::runtime_error("failed to write spill file: " + path.string());
  }
}

template <typename BinType>
void ReadFeatureBinsFromFile(const std::filesystem::path& path, std::vector<BinType>& bins) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open spill file for reading: " + path.string());
  }
  if (!bins.empty()) {
    in.read(reinterpret_cast<char*>(bins.data()),
            static_cast<std::streamsize>(bins.size() * sizeof(BinType)));
  }
  if (!in && !bins.empty()) {
    throw std::runtime_error("failed to read spill file: " + path.string());
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

std::size_t MaxFeatureBins(const std::vector<std::uint16_t>& num_bins_per_feature) noexcept {
  std::uint16_t max_feature_bins = 0;
  for (const std::uint16_t feature_bins_count : num_bins_per_feature) {
    max_feature_bins = std::max(max_feature_bins, feature_bins_count);
  }
  return static_cast<std::size_t>(max_feature_bins);
}

std::size_t NumBinsChecked(const std::vector<std::uint16_t>& num_bins_per_feature,
                           std::size_t feature_index) {
  if (feature_index >= num_bins_per_feature.size()) {
    throw std::out_of_range("feature index is out of bounds");
  }
  return static_cast<std::size_t>(num_bins_per_feature[feature_index]);
}

bool MaskValueChecked(const std::vector<std::uint8_t>& mask,
                      std::size_t feature_index,
                      const char* name) {
  if (feature_index >= mask.size()) {
    throw std::out_of_range(std::string(name) + " feature index is out of bounds");
  }
  return mask[feature_index] != 0;
}

std::uint16_t BinValueFromSchema(const std::vector<std::uint16_t>& num_bins_per_feature,
                                 const std::vector<std::size_t>& cut_offsets,
                                 const std::vector<float>& cut_values,
                                 const std::vector<std::uint8_t>& categorical_mask,
                                 const std::vector<std::uint8_t>& missing_value_mask,
                                 std::uint8_t nan_mode,
                                 const std::vector<std::uint8_t>& nan_modes,
                                 std::size_t feature_index,
                                 float value) {
  const std::size_t bins_for_feature = NumBinsChecked(num_bins_per_feature, feature_index);
  if (bins_for_feature == 0) {
    return 0;
  }

  const std::size_t cut_begin = cut_offsets[feature_index];
  const std::size_t cut_end = cut_offsets[feature_index + 1];
  const auto begin = cut_values.begin() + static_cast<std::ptrdiff_t>(cut_begin);
  const auto end = cut_values.begin() + static_cast<std::ptrdiff_t>(cut_end);
  const bool feature_is_categorical =
      MaskValueChecked(categorical_mask, feature_index, "categorical mask");
  const bool feature_has_missing_values =
      MaskValueChecked(missing_value_mask, feature_index, "missing value mask");
  const NanMode resolved_nan_mode =
      nan_modes.empty()
          ? static_cast<NanMode>(nan_mode)
          : static_cast<NanMode>(nan_modes.at(feature_index));

  if (std::isnan(value)) {
    if (resolved_nan_mode == NanMode::Forbidden) {
      throw std::invalid_argument("NaN values are not allowed when nan_mode='Forbidden'");
    }
    return MissingBinIndex(bins_for_feature, feature_has_missing_values, resolved_nan_mode);
  }

  if (feature_is_categorical) {
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

struct FeatureHistogramResult {
  std::vector<float> cut_values;
  std::uint16_t num_bins{0};
  std::uint8_t missing_value_mask{0};
  bool is_categorical{false};
  NanMode nan_mode{NanMode::Min};
  double elapsed_ms{0.0};
};

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
  const std::vector<float> cuts =
      border_selection_method == BorderSelectionMethod::Uniform
          ? ComputeUniformCuts(quantile_values, feature_max_bins)
          : ComputeQuantileCuts(std::move(quantile_values), feature_max_bins, quantile_strategy);
  result.cut_values = cuts;
  const std::size_t non_missing_bins = non_missing_count == 0 ? 0U : cuts.size() + 1U;
  const std::size_t total_bins = non_missing_bins + (has_missing_values ? 1U : 0U);

  result.num_bins = static_cast<std::uint16_t>(total_bins == 0 ? 1U : total_bins);
  result.elapsed_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - feature_start)
          .count();
  return result;
}

template <typename BinType>
void MaterializeFeatureBinsToBuffer(const Pool& pool,
                                    std::size_t feature,
                                    const FeatureHistogramResult& feature_result,
                                    BinType* out_bins,
                                    std::size_t out_size) {
  const float* const contiguous_column = pool.is_sparse() ? nullptr : pool.feature_column_ptr(feature);
  auto feature_at = [&](std::size_t row) -> float {
    return contiguous_column != nullptr ? contiguous_column[row] : pool.feature_value(row, feature);
  };

  if (out_size < pool.num_rows()) {
    throw std::invalid_argument("feature bin output buffer is smaller than the row count");
  }

  const bool has_missing_values = feature_result.missing_value_mask != 0U;
  const std::size_t bins_for_feature =
      static_cast<std::size_t>(feature_result.num_bins == 0 ? 1U : feature_result.num_bins);
  const std::uint16_t missing_bin =
      MissingBinIndex(bins_for_feature, has_missing_values, feature_result.nan_mode);

  if (feature_result.is_categorical) {
    const std::uint16_t missing_offset =
        has_missing_values && feature_result.nan_mode == NanMode::Min ? 1U : 0U;
    for (std::size_t row = 0; row < pool.num_rows(); ++row) {
      const float value = feature_at(row);
      if (std::isnan(value)) {
        out_bins[row] = static_cast<BinType>(missing_bin);
        continue;
      }

      const auto it =
          std::lower_bound(feature_result.cut_values.begin(), feature_result.cut_values.end(), value);
      const std::size_t insertion =
          static_cast<std::size_t>(std::distance(feature_result.cut_values.begin(), it));
      out_bins[row] = static_cast<BinType>(missing_offset + insertion);
    }
    return;
  }

  const std::uint16_t missing_offset =
      has_missing_values && feature_result.nan_mode == NanMode::Min ? 1U : 0U;
  for (std::size_t row = 0; row < pool.num_rows(); ++row) {
    const float value = feature_at(row);
    if (std::isnan(value)) {
      out_bins[row] = static_cast<BinType>(missing_bin);
      continue;
    }

    const auto bin_it =
        std::upper_bound(feature_result.cut_values.begin(), feature_result.cut_values.end(), value);
    out_bins[row] = static_cast<BinType>(missing_offset +
                                         std::distance(feature_result.cut_values.begin(), bin_it));
  }
}

void MaterializeFeatureBins(const Pool& pool,
                            std::size_t feature,
                            const FeatureHistogramResult& feature_result,
                            HistMatrix& hist) {
  const std::size_t offset = feature * hist.num_rows;
  if (hist.bin_index_bytes == 1) {
    MaterializeFeatureBinsToBuffer(pool,
                                   feature,
                                   feature_result,
                                   hist.compact_bin_indices.data() + offset,
                                   hist.num_rows);
    return;
  }
  MaterializeFeatureBinsToBuffer(
      pool, feature, feature_result, hist.bin_indices.data() + offset, hist.num_rows);
}

std::vector<std::uint8_t> BuildCategoricalMask(const Pool& pool) {
  std::vector<std::uint8_t> mask(pool.num_cols(), 0);
  for (const int feature_index : pool.cat_features()) {
    mask[static_cast<std::size_t>(feature_index)] = 1;
  }
  return mask;
}

void ValidateForbiddenNanModeHasNoMissingValues(const Pool& pool,
                                                const std::vector<NanMode>& feature_nan_modes) {
  for (std::size_t feature = 0; feature < pool.num_cols(); ++feature) {
    if (feature >= feature_nan_modes.size() || feature_nan_modes[feature] != NanMode::Forbidden) {
      continue;
    }
    const float* const contiguous_column = pool.is_sparse() ? nullptr : pool.feature_column_ptr(feature);
    for (std::size_t row = 0; row < pool.num_rows(); ++row) {
      const float value =
          contiguous_column != nullptr ? contiguous_column[row] : pool.feature_value(row, feature);
      if (std::isnan(value)) {
        throw std::invalid_argument("NaN values are not allowed when nan_mode='Forbidden'");
      }
    }
  }
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

BorderSelectionMethod ParseBorderSelectionMethod(std::string_view name) {
  const std::string normalized = NormalizeToken(std::string(name));
  if (normalized.empty() || normalized == "quantile" || normalized == "median") {
    return BorderSelectionMethod::Quantile;
  }
  if (normalized == "uniform") {
    return BorderSelectionMethod::Uniform;
  }
  throw std::invalid_argument(
      "border_selection_method must be one of 'Quantile' or 'Uniform'");
}

const char* BorderSelectionMethodName(BorderSelectionMethod method) noexcept {
  switch (method) {
    case BorderSelectionMethod::Quantile:
      return "Quantile";
    case BorderSelectionMethod::Uniform:
      return "Uniform";
  }
  return "Quantile";
}

FeatureBinView HistMatrix::feature_bins(std::size_t feature_index) const {
  if (feature_index >= num_cols) {
    throw std::out_of_range("feature index is out of bounds");
  }
  if (uses_external_bin_storage_) {
    if (feature_index >= external_feature_bin_paths.size()) {
      throw std::runtime_error("external histogram spill metadata is incomplete");
    }
    if (external_cached_feature_index != feature_index) {
      const std::filesystem::path feature_path(external_feature_bin_paths[feature_index]);
      if (bin_index_bytes == 1) {
        external_feature_cache_u8.resize(num_rows);
        external_feature_cache_u16.clear();
        external_feature_cache_u16.shrink_to_fit();
        ReadFeatureBinsFromFile(feature_path, external_feature_cache_u8);
      } else {
        external_feature_cache_u16.resize(num_rows);
        external_feature_cache_u8.clear();
        external_feature_cache_u8.shrink_to_fit();
        ReadFeatureBinsFromFile(feature_path, external_feature_cache_u16);
      }
      external_cached_feature_index = feature_index;
    }
    return bin_index_bytes == 1
               ? FeatureBinView{external_feature_cache_u8.data(), nullptr}
               : FeatureBinView{nullptr, external_feature_cache_u16.data()};
  }
  const std::size_t offset = feature_index * num_rows;
  if (bin_index_bytes == 1) {
    if (compact_bin_indices.empty()) {
      throw std::runtime_error("histogram compact bin storage has been released");
    }
    return FeatureBinView{compact_bin_indices.data() + offset, nullptr};
  }
  if (bin_indices.empty()) {
    throw std::runtime_error("histogram bin storage has been released");
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
  if (uses_external_bin_storage_) {
    throw std::runtime_error("cannot mutate external histogram bin storage in place");
  }

  const std::size_t offset = feature_index * num_rows + row;
  if (bin_index_bytes == 1) {
    compact_bin_indices[offset] = static_cast<std::uint8_t>(value);
  } else {
    bin_indices[offset] = value;
  }
}

void HistMatrix::CompactBinStorage() {
  if (uses_external_bin_storage_) {
    return;
  }
  if (bin_index_bytes == 1) {
    bin_indices.clear();
    bin_indices.shrink_to_fit();
    return;
  }

  const std::size_t max_feature_bins = MaxFeatureBins(num_bins_per_feature);

  if (bin_indices.empty() ||
      max_feature_bins > static_cast<std::size_t>(std::numeric_limits<std::uint8_t>::max()) + 1U) {
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

bool HistMatrix::uses_external_bin_storage() const noexcept { return uses_external_bin_storage_; }

std::uint8_t HistMatrix::bin_storage_bytes() const noexcept { return bin_index_bytes; }

std::size_t HistMatrix::num_bins(std::size_t feature_index) const {
  return NumBinsChecked(num_bins_per_feature, feature_index);
}

bool HistMatrix::is_categorical(std::size_t feature_index) const {
  return MaskValueChecked(categorical_mask, feature_index, "categorical mask");
}

bool HistMatrix::has_missing_values(std::size_t feature_index) const {
  return MaskValueChecked(missing_value_mask, feature_index, "missing value mask");
}

NanMode HistMatrix::nan_mode_for_feature(std::size_t feature_index) const {
  if (nan_modes.empty()) {
    if (feature_index >= num_cols) {
      throw std::out_of_range("nan_mode feature index is out of bounds");
    }
    return static_cast<NanMode>(nan_mode);
  }
  if (feature_index >= nan_modes.size()) {
    throw std::out_of_range("nan_mode feature index is out of bounds");
  }
  return static_cast<NanMode>(nan_modes[feature_index]);
}

std::uint16_t HistMatrix::bin_value(std::size_t feature_index, float value) const {
  return BinValueFromSchema(num_bins_per_feature,
                            cut_offsets,
                            cut_values,
                            categorical_mask,
                            missing_value_mask,
                            nan_mode,
                            nan_modes,
                            feature_index,
                            value);
}

std::size_t HistMatrix::storage_bytes() const noexcept {
  return bin_indices.capacity() * sizeof(std::uint16_t) +
         compact_bin_indices.capacity() * sizeof(std::uint8_t) +
         external_feature_cache_u16.capacity() * sizeof(std::uint16_t) +
         external_feature_cache_u8.capacity() * sizeof(std::uint8_t) +
         num_bins_per_feature.capacity() * sizeof(std::uint16_t) +
         cut_offsets.capacity() * sizeof(std::size_t) +
         cut_values.capacity() * sizeof(float) +
         categorical_mask.capacity() * sizeof(std::uint8_t) +
         missing_value_mask.capacity() * sizeof(std::uint8_t) +
         nan_modes.capacity() * sizeof(std::uint8_t) +
         external_bin_storage_dir.capacity() * sizeof(char);
}

void HistMatrix::SpillBinStorage(const std::string& directory) {
  if (uses_external_bin_storage_) {
    return;
  }
  if (num_rows == 0 || num_cols == 0) {
    return;
  }
  if (bin_index_bytes == 1 && compact_bin_indices.empty()) {
    throw std::runtime_error("cannot spill released compact histogram bin storage");
  }
  if (bin_index_bytes != 1 && bin_indices.empty()) {
    throw std::runtime_error("cannot spill released histogram bin storage");
  }

  const std::filesystem::path root = MakeExternalStorageRoot(directory);
  external_feature_bin_paths.assign(num_cols, std::string());
  for (std::size_t feature = 0; feature < num_cols; ++feature) {
    const std::filesystem::path feature_path = root / ("feature_" + std::to_string(feature) + ".bin");
    external_feature_bin_paths[feature] = feature_path.string();
    const std::size_t offset = feature * num_rows;
    if (bin_index_bytes == 1) {
      std::vector<std::uint8_t> feature_bins(num_rows, 0);
      std::copy_n(compact_bin_indices.begin() + static_cast<std::ptrdiff_t>(offset),
                  static_cast<std::ptrdiff_t>(num_rows),
                  feature_bins.begin());
      WriteFeatureBinsToFile(feature_path, feature_bins);
    } else {
      std::vector<std::uint16_t> feature_bins(num_rows, 0);
      std::copy_n(bin_indices.begin() + static_cast<std::ptrdiff_t>(offset),
                  static_cast<std::ptrdiff_t>(num_rows),
                  feature_bins.begin());
      WriteFeatureBinsToFile(feature_path, feature_bins);
    }
  }

  external_bin_storage_dir = root.string();
  uses_external_bin_storage_ = true;
  external_cached_feature_index = static_cast<std::size_t>(-1);
  bin_indices.clear();
  bin_indices.shrink_to_fit();
  compact_bin_indices.clear();
  compact_bin_indices.shrink_to_fit();
}

void HistMatrix::ReleaseBinStorage() noexcept {
  bin_indices.clear();
  bin_indices.shrink_to_fit();
  compact_bin_indices.clear();
  compact_bin_indices.shrink_to_fit();
  external_feature_cache_u16.clear();
  external_feature_cache_u16.shrink_to_fit();
  external_feature_cache_u8.clear();
  external_feature_cache_u8.shrink_to_fit();
  external_cached_feature_index = static_cast<std::size_t>(-1);
  if (uses_external_bin_storage_ && !external_bin_storage_dir.empty()) {
    std::error_code error;
    std::filesystem::remove_all(std::filesystem::path(external_bin_storage_dir), error);
  }
  external_feature_bin_paths.clear();
  external_feature_bin_paths.shrink_to_fit();
  external_bin_storage_dir.clear();
  external_bin_storage_dir.shrink_to_fit();
  uses_external_bin_storage_ = false;
}

void HistMatrix::ReleaseStorage() noexcept {
  num_rows = 0;
  num_cols = 0;
  ReleaseBinStorage();
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
  nan_modes.clear();
  nan_modes.shrink_to_fit();
}

HistBuilder::HistBuilder(std::size_t max_bins,
                         std::string nan_mode,
                         std::vector<std::uint16_t> max_bins_by_feature,
                         std::string border_selection_method,
                         std::vector<std::string> nan_mode_by_feature,
                         std::vector<std::vector<float>> feature_borders,
                         bool external_memory,
                         std::string external_memory_dir)
    : max_bins_(max_bins),
      max_bins_by_feature_(std::move(max_bins_by_feature)),
      nan_mode_(ParseNanMode(nan_mode)),
      nan_mode_name_(NanModeName(nan_mode_)),
      external_memory_(external_memory),
      external_memory_dir_(std::move(external_memory_dir)) {
  if (!ValidateMaxBins(max_bins_)) {
    throw std::invalid_argument("max_bins must be in the range [1, 65535]");
  }
  border_selection_method_ = ParseBorderSelectionMethod(border_selection_method);
  border_selection_method_name_ = BorderSelectionMethodName(border_selection_method_);

  nan_mode_by_feature_.reserve(nan_mode_by_feature.size());
  nan_mode_by_feature_mask_.reserve(nan_mode_by_feature.size());
  nan_mode_by_feature_names_.reserve(nan_mode_by_feature.size());
  for (const std::string& raw_name : nan_mode_by_feature) {
    if (raw_name.empty()) {
      nan_mode_by_feature_.push_back(nan_mode_);
      nan_mode_by_feature_mask_.push_back(0U);
      nan_mode_by_feature_names_.push_back("");
      continue;
    }

    const NanMode parsed_mode = ParseNanMode(raw_name);
    nan_mode_by_feature_.push_back(parsed_mode);
    nan_mode_by_feature_mask_.push_back(1U);
    nan_mode_by_feature_names_.push_back(NanModeName(parsed_mode));
  }

  for (std::size_t feature = 0; feature < max_bins_by_feature_.size(); ++feature) {
    if (max_bins_by_feature_[feature] == 0U) {
      continue;
    }
    if (!ValidateMaxBins(max_bins_by_feature_[feature])) {
      throw std::invalid_argument(
          "max_bin_by_feature entries must be in the range [1, 65535] when provided");
    }
  }

  feature_borders_.reserve(feature_borders.size());
  for (auto& borders : feature_borders) {
    feature_borders_.push_back(NormalizeCustomBorders(std::move(borders)));
  }
}

HistMatrix HistBuilder::Build(const Pool& pool, const TrainingProfiler* profiler) const {
  const auto hist_build_start = std::chrono::steady_clock::now();
  if (!max_bins_by_feature_.empty() && max_bins_by_feature_.size() > pool.num_cols()) {
    throw std::invalid_argument(
        "max_bin_by_feature cannot specify more entries than the pool feature count");
  }
  if (!nan_mode_by_feature_names_.empty() && nan_mode_by_feature_names_.size() > pool.num_cols()) {
    throw std::invalid_argument(
        "nan_mode_by_feature cannot specify more entries than the pool feature count");
  }
  if (!feature_borders_.empty() && feature_borders_.size() > pool.num_cols()) {
    throw std::invalid_argument(
        "feature_borders cannot specify more entries than the pool feature count");
  }

  std::vector<NanMode> resolved_feature_nan_modes(pool.num_cols(), nan_mode_);
  for (std::size_t feature = 0; feature < pool.num_cols(); ++feature) {
    resolved_feature_nan_modes[feature] =
        ResolveFeatureNanMode(nan_mode_by_feature_, nan_mode_by_feature_mask_, feature, nan_mode_);
  }
  if (!resolved_feature_nan_modes.empty()) {
    ValidateForbiddenNanModeHasNoMissingValues(pool, resolved_feature_nan_modes);
  }

  HistMatrix hist;
  hist.num_rows = pool.num_rows();
  hist.num_cols = pool.num_cols();
  hist.num_bins_per_feature.resize(hist.num_cols, 0);
  hist.cut_offsets.reserve(hist.num_cols + 1);
  hist.cut_offsets.push_back(0);
  hist.categorical_mask = BuildCategoricalMask(pool);
  hist.missing_value_mask.assign(hist.num_cols, 0);
  hist.nan_mode = static_cast<std::uint8_t>(nan_mode_);
  hist.nan_modes.resize(hist.num_cols, static_cast<std::uint8_t>(nan_mode_));
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
            pool,
            ResolveFeatureMaxBins(max_bins_by_feature_, feature, max_bins_),
            resolved_feature_nan_modes[feature],
            border_selection_method_,
            feature,
            hist.categorical_mask[feature] != 0U,
            build_context,
            ResolveFeatureBorders(feature_borders_, feature));
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
    hist.nan_modes[feature] = static_cast<std::uint8_t>(feature_result.nan_mode);
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

  if (MaxFeatureBins(hist.num_bins_per_feature) <=
      static_cast<std::size_t>(std::numeric_limits<std::uint8_t>::max()) + 1U) {
    hist.bin_index_bytes = 1;
  } else {
    hist.bin_index_bytes = 2;
  }
  const bool use_external_bin_storage = external_memory_ && hist.num_rows > 0 && hist.num_cols > 0;
  std::filesystem::path external_storage_root;
  if (use_external_bin_storage) {
    external_storage_root = MakeExternalStorageRoot(external_memory_dir_);
    hist.external_bin_storage_dir = external_storage_root.string();
    hist.external_feature_bin_paths.resize(hist.num_cols);
  } else if (hist.bin_index_bytes == 1) {
    hist.compact_bin_indices.resize(hist.num_rows * hist.num_cols, 0);
  } else {
    hist.bin_indices.resize(hist.num_rows * hist.num_cols, 0);
  }

  failed.store(false, std::memory_order_relaxed);
  next_feature.store(0, std::memory_order_relaxed);
  workers.clear();
  workers.reserve(thread_count);
  first_error = nullptr;
  auto materialize_worker = [&]() {
    while (!failed.load(std::memory_order_relaxed)) {
      const std::size_t feature = next_feature.fetch_add(1, std::memory_order_relaxed);
      if (feature >= hist.num_cols) {
        return;
      }

      try {
        if (use_external_bin_storage) {
          const std::filesystem::path feature_path =
              external_storage_root / ("feature_" + std::to_string(feature) + ".bin");
          if (hist.bin_index_bytes == 1) {
            std::vector<std::uint8_t> feature_bins(hist.num_rows, 0);
            MaterializeFeatureBinsToBuffer(pool,
                                           feature,
                                           feature_results[feature],
                                           feature_bins.data(),
                                           feature_bins.size());
            WriteFeatureBinsToFile(feature_path, feature_bins);
          } else {
            std::vector<std::uint16_t> feature_bins(hist.num_rows, 0);
            MaterializeFeatureBinsToBuffer(pool,
                                           feature,
                                           feature_results[feature],
                                           feature_bins.data(),
                                           feature_bins.size());
            WriteFeatureBinsToFile(feature_path, feature_bins);
          }
          hist.external_feature_bin_paths[feature] = feature_path.string();
        } else {
          MaterializeFeatureBins(pool, feature, feature_results[feature], hist);
        }
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
    workers.emplace_back(materialize_worker);
  }
  for (std::thread& worker_thread : workers) {
    worker_thread.join();
  }
  if (first_error != nullptr) {
    std::rethrow_exception(first_error);
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

  if (use_external_bin_storage) {
    hist.uses_external_bin_storage_ = true;
    hist.external_cached_feature_index = static_cast<std::size_t>(-1);
  } else {
    hist.CompactBinStorage();
  }
  return hist;
}

std::size_t QuantizationSchema::num_cols() const noexcept { return num_bins_per_feature.size(); }

std::size_t QuantizationSchema::num_bins(std::size_t feature_index) const {
  return NumBinsChecked(num_bins_per_feature, feature_index);
}

bool QuantizationSchema::is_categorical(std::size_t feature_index) const {
  return MaskValueChecked(categorical_mask, feature_index, "categorical mask");
}

bool QuantizationSchema::has_missing_values(std::size_t feature_index) const {
  return MaskValueChecked(missing_value_mask, feature_index, "missing value mask");
}

NanMode QuantizationSchema::nan_mode_for_feature(std::size_t feature_index) const {
  if (nan_modes.empty()) {
    if (feature_index >= num_bins_per_feature.size()) {
      throw std::out_of_range("nan_mode feature index is out of bounds");
    }
    return static_cast<NanMode>(nan_mode);
  }
  if (feature_index >= nan_modes.size()) {
    throw std::out_of_range("nan_mode feature index is out of bounds");
  }
  return static_cast<NanMode>(nan_modes[feature_index]);
}

std::uint16_t QuantizationSchema::bin_value(std::size_t feature_index, float value) const {
  return BinValueFromSchema(num_bins_per_feature,
                            cut_offsets,
                            cut_values,
                            categorical_mask,
                            missing_value_mask,
                            nan_mode,
                            nan_modes,
                            feature_index,
                            value);
}

std::size_t QuantizationSchema::storage_bytes() const noexcept {
  return num_bins_per_feature.capacity() * sizeof(std::uint16_t) +
         cut_offsets.capacity() * sizeof(std::size_t) +
         cut_values.capacity() * sizeof(float) +
         categorical_mask.capacity() * sizeof(std::uint8_t) +
         missing_value_mask.capacity() * sizeof(std::uint8_t) +
         nan_modes.capacity() * sizeof(std::uint8_t);
}

QuantizationSchema MakeQuantizationSchema(const HistMatrix& hist) {
  QuantizationSchema schema;
  schema.num_bins_per_feature = hist.num_bins_per_feature;
  schema.cut_offsets = hist.cut_offsets;
  schema.cut_values = hist.cut_values;
  schema.categorical_mask = hist.categorical_mask;
  schema.missing_value_mask = hist.missing_value_mask;
  schema.nan_mode = hist.nan_mode;
  schema.nan_modes = hist.nan_modes;
  return schema;
}

void ApplyQuantizationSchema(const QuantizationSchema& schema, HistMatrix& hist) {
  hist.num_cols = schema.num_cols();
  hist.num_bins_per_feature = schema.num_bins_per_feature;
  hist.cut_offsets = schema.cut_offsets;
  hist.cut_values = schema.cut_values;
  hist.categorical_mask = schema.categorical_mask;
  hist.missing_value_mask = schema.missing_value_mask;
  hist.nan_mode = schema.nan_mode;
  hist.nan_modes = schema.nan_modes;
}

std::size_t HistBuilder::max_bins() const noexcept { return max_bins_; }

const std::vector<std::uint16_t>& HistBuilder::max_bins_by_feature() const noexcept {
  return max_bins_by_feature_;
}

NanMode HistBuilder::nan_mode() const noexcept { return nan_mode_; }

const std::string& HistBuilder::nan_mode_name() const noexcept { return nan_mode_name_; }

BorderSelectionMethod HistBuilder::border_selection_method() const noexcept {
  return border_selection_method_;
}

const std::string& HistBuilder::border_selection_method_name() const noexcept {
  return border_selection_method_name_;
}

const std::vector<std::string>& HistBuilder::nan_mode_by_feature_names() const noexcept {
  return nan_mode_by_feature_names_;
}

const std::vector<std::vector<float>>& HistBuilder::feature_borders() const noexcept {
  return feature_borders_;
}

bool HistBuilder::external_memory() const noexcept { return external_memory_; }

const std::string& HistBuilder::external_memory_dir() const noexcept {
  return external_memory_dir_;
}

}  // namespace ctboost
