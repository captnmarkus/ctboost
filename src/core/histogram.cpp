#include "ctboost/histogram.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

namespace ctboost {
namespace {

constexpr std::size_t kMaxCategoricalBins = 256;

bool ValidateMaxBins(std::size_t max_bins) {
  return max_bins > 0 &&
         max_bins <= static_cast<std::size_t>(std::numeric_limits<std::uint16_t>::max());
}

std::vector<float> ComputeQuantileCuts(const float* column,
                                       std::size_t num_rows,
                                       std::size_t max_bins) {
  if (num_rows == 0) {
    return {};
  }

  std::vector<float> sorted(column, column + num_rows);
  std::sort(sorted.begin(), sorted.end());

  const std::size_t desired_bins = std::min(max_bins, num_rows);
  std::vector<float> cuts;
  cuts.reserve(desired_bins > 0 ? desired_bins - 1 : 0);

  for (std::size_t bin = 1; bin < desired_bins; ++bin) {
    const std::size_t index =
        (bin * num_rows) / desired_bins >= num_rows ? num_rows - 1 : (bin * num_rows) / desired_bins;
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

}  // namespace

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

  if (is_categorical(feature_index)) {
    const auto it = std::lower_bound(begin, end, value);
    if (it != end && *it == value) {
      return static_cast<std::uint16_t>(std::distance(begin, it));
    }
    const std::size_t insertion = static_cast<std::size_t>(std::distance(begin, it));
    return static_cast<std::uint16_t>(std::min(insertion, bins_for_feature - 1));
  }

  const auto it = std::upper_bound(begin, end, value);
  return static_cast<std::uint16_t>(std::distance(begin, it));
}

HistBuilder::HistBuilder(std::size_t max_bins) : max_bins_(max_bins) {
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

  const auto& feature_data = pool.feature_data();

  for (std::size_t feature = 0; feature < hist.num_cols; ++feature) {
    const float* column = feature_data.data() + feature * hist.num_rows;
    const std::size_t offset = feature * hist.num_rows;

    if (hist.is_categorical(feature)) {
      std::map<float, std::uint16_t> category_to_bin;
      for (std::size_t row = 0; row < hist.num_rows; ++row) {
        category_to_bin.emplace(column[row], 0);
        if (category_to_bin.size() > kMaxCategoricalBins) {
          throw std::invalid_argument(
              "categorical feature has too many unique categories; maximum supported is 256");
        }
      }

      std::uint16_t next_bin = 0;
      for (auto& [category, mapped_bin] : category_to_bin) {
        (void)category;
        mapped_bin = next_bin++;
        hist.cut_values.push_back(category);
      }

      for (std::size_t row = 0; row < hist.num_rows; ++row) {
        hist.bin_indices[offset + row] = category_to_bin.find(column[row])->second;
      }

      hist.num_bins_per_feature[feature] =
          static_cast<std::uint16_t>(category_to_bin.size());
      hist.cut_offsets.push_back(hist.cut_values.size());
      continue;
    }

    const std::vector<float> cuts = ComputeQuantileCuts(column, hist.num_rows, max_bins_);
    hist.cut_values.insert(hist.cut_values.end(), cuts.begin(), cuts.end());
    hist.cut_offsets.push_back(hist.cut_values.size());

    for (std::size_t row = 0; row < hist.num_rows; ++row) {
      const auto bin_it = std::upper_bound(cuts.begin(), cuts.end(), column[row]);
      hist.bin_indices[offset + row] =
          static_cast<std::uint16_t>(std::distance(cuts.begin(), bin_it));
    }

    hist.num_bins_per_feature[feature] = static_cast<std::uint16_t>(cuts.size() + 1);
  }

  return hist;
}

std::size_t HistBuilder::max_bins() const noexcept { return max_bins_; }

}  // namespace ctboost
