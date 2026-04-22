#include "histogram_internal.hpp"

#include <algorithm>
#include <stdexcept>

namespace ctboost::detail {

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

  const std::size_t non_missing_bins = bins_for_feature - (feature_has_missing_values ? 1U : 0U);
  if (non_missing_bins == 0) {
    return MissingBinIndex(bins_for_feature, feature_has_missing_values, resolved_nan_mode);
  }

  const auto it = std::upper_bound(begin, end, value);
  const std::uint16_t offset =
      feature_has_missing_values && resolved_nan_mode == NanMode::Min ? 1U : 0U;
  return static_cast<std::uint16_t>(offset + std::distance(begin, it));
}

}  // namespace ctboost::detail

namespace ctboost {

std::size_t HistMatrix::num_bins(std::size_t feature_index) const {
  return detail::NumBinsChecked(num_bins_per_feature, feature_index);
}

bool HistMatrix::is_categorical(std::size_t feature_index) const {
  return detail::MaskValueChecked(categorical_mask, feature_index, "categorical mask");
}

bool HistMatrix::has_missing_values(std::size_t feature_index) const {
  return detail::MaskValueChecked(missing_value_mask, feature_index, "missing value mask");
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
  return detail::BinValueFromSchema(num_bins_per_feature,
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

std::size_t QuantizationSchema::num_cols() const noexcept { return num_bins_per_feature.size(); }

std::size_t QuantizationSchema::num_bins(std::size_t feature_index) const {
  return detail::NumBinsChecked(num_bins_per_feature, feature_index);
}

bool QuantizationSchema::is_categorical(std::size_t feature_index) const {
  return detail::MaskValueChecked(categorical_mask, feature_index, "categorical mask");
}

bool QuantizationSchema::has_missing_values(std::size_t feature_index) const {
  return detail::MaskValueChecked(missing_value_mask, feature_index, "missing value mask");
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
  return detail::BinValueFromSchema(num_bins_per_feature,
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

}  // namespace ctboost
