#include "ctboost/histogram.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace ctboost {
namespace {

constexpr std::size_t kMaxCategoricalBins = 256;

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

std::vector<float> ComputeQuantileCuts(const std::vector<float>& values,
                                       std::size_t max_bins) {
  if (values.empty()) {
    return {};
  }

  std::vector<float> sorted(values);
  std::sort(sorted.begin(), sorted.end());

  const std::size_t desired_bins = std::min(max_bins, sorted.size());
  std::vector<float> cuts;
  cuts.reserve(desired_bins > 0 ? desired_bins - 1 : 0);

  for (std::size_t bin = 1; bin < desired_bins; ++bin) {
    const std::size_t index =
        (bin * sorted.size()) / desired_bins >= sorted.size()
            ? sorted.size() - 1
            : (bin * sorted.size()) / desired_bins;
    const float cut = sorted[index];
    if (cuts.empty() || cut > cuts.back()) {
      cuts.push_back(cut);
    }
  }

  return cuts;
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

const std::uint16_t* HistMatrix::feature_bins(std::size_t feature_index) const {
  if (feature_index >= num_cols) {
    throw std::out_of_range("feature index is out of bounds");
  }
  return bin_indices.data() + feature_index * num_rows;
}

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

HistBuilder::HistBuilder(std::size_t max_bins, std::string nan_mode)
    : max_bins_(max_bins),
      nan_mode_(ParseNanMode(nan_mode)),
      nan_mode_name_(NanModeName(nan_mode_)) {
  if (!ValidateMaxBins(max_bins_)) {
    throw std::invalid_argument("max_bins must be in the range [1, 65535]");
  }
}

HistMatrix HistBuilder::Build(const Pool& pool) const {
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

  const auto& feature_data = pool.feature_data();

  for (std::size_t feature = 0; feature < hist.num_cols; ++feature) {
    const float* column = feature_data.data() + feature * hist.num_rows;
    const std::size_t offset = feature * hist.num_rows;
    bool has_missing_values = false;

    if (hist.is_categorical(feature)) {
      std::map<float, std::uint16_t> category_to_bin;
      for (std::size_t row = 0; row < hist.num_rows; ++row) {
        if (std::isnan(column[row])) {
          has_missing_values = true;
          continue;
        }
        category_to_bin.emplace(column[row], 0);
        if (category_to_bin.size() > kMaxCategoricalBins) {
          throw std::invalid_argument(
              "categorical feature has too many unique categories; maximum supported is 256");
        }
      }

      if (has_missing_values && nan_mode_ == NanMode::Forbidden) {
        throw std::invalid_argument("NaN values are not allowed when nan_mode='Forbidden'");
      }

      hist.missing_value_mask[feature] = has_missing_values ? 1U : 0U;

      std::uint16_t next_bin =
          has_missing_values && nan_mode_ == NanMode::Min ? 1U : 0U;
      for (auto& [category, mapped_bin] : category_to_bin) {
        (void)category;
        mapped_bin = next_bin++;
        hist.cut_values.push_back(category);
      }

      const std::uint16_t missing_bin =
          MissingBinIndex(category_to_bin.size() + (has_missing_values ? 1U : 0U),
                          has_missing_values,
                          nan_mode_);
      for (std::size_t row = 0; row < hist.num_rows; ++row) {
        if (std::isnan(column[row])) {
          hist.bin_indices[offset + row] = missing_bin;
        } else {
          hist.bin_indices[offset + row] = category_to_bin.find(column[row])->second;
        }
      }
      hist.num_bins_per_feature[feature] =
          static_cast<std::uint16_t>(category_to_bin.size() + (has_missing_values ? 1U : 0U));
      hist.cut_offsets.push_back(hist.cut_values.size());
      continue;
    }

    std::vector<float> non_missing_values;
    non_missing_values.reserve(hist.num_rows);
    for (std::size_t row = 0; row < hist.num_rows; ++row) {
      if (std::isnan(column[row])) {
        has_missing_values = true;
      } else {
        non_missing_values.push_back(column[row]);
      }
    }

    if (has_missing_values && nan_mode_ == NanMode::Forbidden) {
      throw std::invalid_argument("NaN values are not allowed when nan_mode='Forbidden'");
    }

    hist.missing_value_mask[feature] = has_missing_values ? 1U : 0U;
    const std::vector<float> cuts = ComputeQuantileCuts(non_missing_values, max_bins_);
    hist.cut_values.insert(hist.cut_values.end(), cuts.begin(), cuts.end());
    hist.cut_offsets.push_back(hist.cut_values.size());
    const std::size_t non_missing_bins = non_missing_values.empty() ? 0U : cuts.size() + 1U;
    const std::uint16_t missing_offset =
        has_missing_values && nan_mode_ == NanMode::Min && non_missing_bins > 0 ? 1U : 0U;
    const std::size_t total_bins = non_missing_bins + (has_missing_values ? 1U : 0U);
    const std::uint16_t missing_bin =
        MissingBinIndex(total_bins == 0 ? 1U : total_bins, has_missing_values, nan_mode_);

    for (std::size_t row = 0; row < hist.num_rows; ++row) {
      if (std::isnan(column[row])) {
        hist.bin_indices[offset + row] = missing_bin;
      } else {
        const auto bin_it = std::upper_bound(cuts.begin(), cuts.end(), column[row]);
        hist.bin_indices[offset + row] =
            static_cast<std::uint16_t>(missing_offset + std::distance(cuts.begin(), bin_it));
      }
    }

    hist.num_bins_per_feature[feature] =
        static_cast<std::uint16_t>(total_bins == 0 ? 1U : total_bins);
  }

  return hist;
}

std::size_t HistBuilder::max_bins() const noexcept { return max_bins_; }

NanMode HistBuilder::nan_mode() const noexcept { return nan_mode_; }

const std::string& HistBuilder::nan_mode_name() const noexcept { return nan_mode_name_; }

}  // namespace ctboost
