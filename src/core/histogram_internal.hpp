#pragma once

#include "ctboost/histogram.hpp"

#include <filesystem>

namespace ctboost::detail {

enum class ExactQuantileStrategy : std::uint8_t {
  Sort = 0,
  Selection = 1,
};

bool ValidateMaxBins(std::size_t max_bins);
std::vector<float> NormalizeCustomBorders(std::vector<float> cuts);
std::size_t ResolveHistogramThreadCount(std::size_t num_features);

struct HistogramBuildContext {
  std::size_t approx_threshold_rows{0};
  std::size_t approx_sample_size{0};
  std::vector<std::size_t> approx_row_sample;
};

HistogramBuildContext ResolveHistogramBuildContext(std::size_t num_rows);
bool ShouldApproximateQuantiles(const HistogramBuildContext& context,
                                std::size_t non_missing_count);
bool HasSufficientApproximationValues(std::size_t sample_count,
                                      std::size_t non_missing_count,
                                      std::size_t max_bins);
std::size_t ResolveExactLowCardinalityLimit(std::size_t max_bins);
ExactQuantileStrategy ResolveExactQuantileStrategy(std::size_t non_missing_count,
                                                   std::size_t observed_unique_count,
                                                   bool unique_count_capped,
                                                   std::size_t max_bins);
std::vector<float> ComputeUniformCuts(const std::vector<float>& values, std::size_t max_bins);
std::vector<float> ComputeQuantileCuts(std::vector<float> values,
                                       std::size_t max_bins,
                                       ExactQuantileStrategy strategy);

std::filesystem::path MakeExternalStorageRoot(const std::string& directory);
void WriteFeatureBinsToFile(const std::filesystem::path& path,
                            const std::vector<std::uint8_t>& bins);
void WriteFeatureBinsToFile(const std::filesystem::path& path,
                            const std::vector<std::uint16_t>& bins);

NanMode ResolveFeatureNanMode(const std::vector<NanMode>& per_feature_modes,
                              const std::vector<std::uint8_t>& per_feature_mode_mask,
                              std::size_t feature_index,
                              NanMode default_mode);
std::size_t ResolveFeatureMaxBins(const std::vector<std::uint16_t>& max_bins_by_feature,
                                  std::size_t feature_index,
                                  std::size_t default_max_bins);
const std::vector<float>* ResolveFeatureBorders(
    const std::vector<std::vector<float>>& feature_borders,
    std::size_t feature_index) noexcept;
std::size_t MaxFeatureBins(const std::vector<std::uint16_t>& num_bins_per_feature) noexcept;
std::size_t NumBinsChecked(const std::vector<std::uint16_t>& num_bins_per_feature,
                           std::size_t feature_index);
bool MaskValueChecked(const std::vector<std::uint8_t>& mask,
                      std::size_t feature_index,
                      const char* name);
std::uint16_t MissingBinIndex(std::size_t bins_for_feature,
                              bool has_missing_values,
                              NanMode nan_mode);
std::uint16_t BinValueFromSchema(const std::vector<std::uint16_t>& num_bins_per_feature,
                                 const std::vector<std::size_t>& cut_offsets,
                                 const std::vector<float>& cut_values,
                                 const std::vector<std::uint8_t>& categorical_mask,
                                 const std::vector<std::uint8_t>& missing_value_mask,
                                 std::uint8_t nan_mode,
                                 const std::vector<std::uint8_t>& nan_modes,
                                 std::size_t feature_index,
                                 float value);

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
                                             const std::vector<float>* custom_borders);
void MaterializeFeatureBins(const Pool& pool,
                            std::size_t feature,
                            const FeatureHistogramResult& feature_result,
                            HistMatrix& hist);
void MaterializeFeatureBinsToExternalStorage(const Pool& pool,
                                             std::size_t feature,
                                             const FeatureHistogramResult& feature_result,
                                             std::uint8_t bin_index_bytes,
                                             const std::filesystem::path& feature_path);
std::vector<std::uint8_t> BuildCategoricalMask(const Pool& pool);
void ValidateForbiddenNanModeHasNoMissingValues(const Pool& pool,
                                                const std::vector<NanMode>& feature_nan_modes);

}  // namespace ctboost::detail
