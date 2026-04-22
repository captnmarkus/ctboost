#include "histogram_internal.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ctboost::detail {
namespace {

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
    out_bins[row] =
        static_cast<BinType>(missing_offset + std::distance(feature_result.cut_values.begin(), bin_it));
  }
}

}  // namespace

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

void MaterializeFeatureBinsToExternalStorage(const Pool& pool,
                                             std::size_t feature,
                                             const FeatureHistogramResult& feature_result,
                                             std::uint8_t bin_index_bytes,
                                             const std::filesystem::path& feature_path) {
  if (bin_index_bytes == 1) {
    std::vector<std::uint8_t> feature_bins(pool.num_rows(), 0);
    MaterializeFeatureBinsToBuffer(
        pool, feature, feature_result, feature_bins.data(), feature_bins.size());
    WriteFeatureBinsToFile(feature_path, feature_bins);
    return;
  }

  std::vector<std::uint16_t> feature_bins(pool.num_rows(), 0);
  MaterializeFeatureBinsToBuffer(
      pool, feature, feature_result, feature_bins.data(), feature_bins.size());
  WriteFeatureBinsToFile(feature_path, feature_bins);
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

}  // namespace ctboost::detail
