#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "ctboost/data.hpp"

namespace ctboost {

enum class NanMode : std::uint8_t {
  Forbidden = 0,
  Min = 1,
  Max = 2,
};

NanMode ParseNanMode(std::string_view name);
const char* NanModeName(NanMode nan_mode) noexcept;

struct HistMatrix {
  std::size_t num_rows{0};
  std::size_t num_cols{0};
  std::vector<std::uint16_t> bin_indices;
  std::vector<std::uint16_t> num_bins_per_feature;
  std::vector<std::size_t> cut_offsets;
  std::vector<float> cut_values;
  std::vector<std::uint8_t> categorical_mask;
  std::vector<std::uint8_t> missing_value_mask;
  std::uint8_t nan_mode{static_cast<std::uint8_t>(NanMode::Min)};

  const std::uint16_t* feature_bins(std::size_t feature_index) const;
  std::size_t num_bins(std::size_t feature_index) const;
  bool is_categorical(std::size_t feature_index) const;
  bool has_missing_values(std::size_t feature_index) const;
  std::uint16_t bin_value(std::size_t feature_index, float value) const;
};

class HistBuilder {
 public:
  explicit HistBuilder(std::size_t max_bins = 256, std::string nan_mode = "Min");

  HistMatrix Build(const Pool& pool) const;
  std::size_t max_bins() const noexcept;
  NanMode nan_mode() const noexcept;
  const std::string& nan_mode_name() const noexcept;

 private:
  std::size_t max_bins_{256};
  NanMode nan_mode_{NanMode::Min};
  std::string nan_mode_name_{"Min"};
};

}  // namespace ctboost
