#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "ctboost/data.hpp"

namespace ctboost {

struct HistMatrix {
  std::size_t num_rows{0};
  std::size_t num_cols{0};
  std::vector<std::uint16_t> bin_indices;
  std::vector<std::uint16_t> num_bins_per_feature;
  std::vector<std::size_t> cut_offsets;
  std::vector<float> cut_values;
  std::vector<std::uint8_t> categorical_mask;

  const std::uint16_t* feature_bins(std::size_t feature_index) const;
  std::size_t num_bins(std::size_t feature_index) const;
  bool is_categorical(std::size_t feature_index) const;
  std::uint16_t bin_value(std::size_t feature_index, float value) const;
};

class HistBuilder {
 public:
  explicit HistBuilder(std::size_t max_bins = 256);

  HistMatrix Build(const Pool& pool) const;
  std::size_t max_bins() const noexcept;

 private:
  std::size_t max_bins_{256};
};

}  // namespace ctboost
