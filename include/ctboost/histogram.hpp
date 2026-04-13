#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "ctboost/data.hpp"

namespace ctboost {

struct FeatureBinView {
  const std::uint8_t* data_u8{nullptr};
  const std::uint16_t* data_u16{nullptr};

  std::uint16_t operator[](std::size_t row) const noexcept {
    return data_u8 != nullptr ? static_cast<std::uint16_t>(data_u8[row]) : data_u16[row];
  }
};

class TrainingProfiler;

enum class NanMode : std::uint8_t {
  Forbidden = 0,
  Min = 1,
  Max = 2,
};

enum class BorderSelectionMethod : std::uint8_t {
  Quantile = 0,
  Uniform = 1,
};

NanMode ParseNanMode(std::string_view name);
const char* NanModeName(NanMode nan_mode) noexcept;
BorderSelectionMethod ParseBorderSelectionMethod(std::string_view name);
const char* BorderSelectionMethodName(BorderSelectionMethod method) noexcept;

struct HistMatrix {
  std::size_t num_rows{0};
  std::size_t num_cols{0};
  std::vector<std::uint16_t> bin_indices;
  std::vector<std::uint8_t> compact_bin_indices;
  std::uint8_t bin_index_bytes{2};
  bool uses_external_bin_storage_{false};
  std::string external_bin_storage_dir;
  std::vector<std::string> external_feature_bin_paths;
  mutable std::vector<std::uint16_t> external_feature_cache_u16;
  mutable std::vector<std::uint8_t> external_feature_cache_u8;
  mutable std::size_t external_cached_feature_index{static_cast<std::size_t>(-1)};
  std::vector<std::uint16_t> num_bins_per_feature;
  std::vector<std::size_t> cut_offsets;
  std::vector<float> cut_values;
  std::vector<std::uint8_t> categorical_mask;
  std::vector<std::uint8_t> missing_value_mask;
  std::uint8_t nan_mode{static_cast<std::uint8_t>(NanMode::Min)};
  std::vector<std::uint8_t> nan_modes;

  FeatureBinView feature_bins(std::size_t feature_index) const;
  std::size_t num_bins(std::size_t feature_index) const;
  bool is_categorical(std::size_t feature_index) const;
  bool has_missing_values(std::size_t feature_index) const;
  NanMode nan_mode_for_feature(std::size_t feature_index) const;
  std::uint16_t bin_at(std::size_t feature_index, std::size_t row) const;
  void set_bin_index(std::size_t feature_index, std::size_t row, std::uint16_t value);
  void CompactBinStorage();
  bool uses_compact_bin_storage() const noexcept;
  bool uses_external_bin_storage() const noexcept;
  std::uint8_t bin_storage_bytes() const noexcept;
  std::uint16_t bin_value(std::size_t feature_index, float value) const;
  std::size_t storage_bytes() const noexcept;
  void SpillBinStorage(const std::string& directory = "");
  void ReleaseBinStorage() noexcept;
  void ReleaseStorage() noexcept;
};

struct QuantizationSchema {
  std::vector<std::uint16_t> num_bins_per_feature;
  std::vector<std::size_t> cut_offsets;
  std::vector<float> cut_values;
  std::vector<std::uint8_t> categorical_mask;
  std::vector<std::uint8_t> missing_value_mask;
  std::uint8_t nan_mode{static_cast<std::uint8_t>(NanMode::Min)};
  std::vector<std::uint8_t> nan_modes;

  std::size_t num_cols() const noexcept;
  std::size_t num_bins(std::size_t feature_index) const;
  bool is_categorical(std::size_t feature_index) const;
  bool has_missing_values(std::size_t feature_index) const;
  NanMode nan_mode_for_feature(std::size_t feature_index) const;
  std::uint16_t bin_value(std::size_t feature_index, float value) const;
  std::size_t storage_bytes() const noexcept;
};

QuantizationSchema MakeQuantizationSchema(const HistMatrix& hist);
void ApplyQuantizationSchema(const QuantizationSchema& schema, HistMatrix& hist);
using QuantizationSchemaPtr = std::shared_ptr<const QuantizationSchema>;

class HistBuilder {
 public:
  explicit HistBuilder(std::size_t max_bins = 256,
                       std::string nan_mode = "Min",
                       std::vector<std::uint16_t> max_bins_by_feature = {},
                       std::string border_selection_method = "Quantile",
                       std::vector<std::string> nan_mode_by_feature = {},
                       std::vector<std::vector<float>> feature_borders = {},
                       bool external_memory = false,
                       std::string external_memory_dir = "");

  HistMatrix Build(const Pool& pool, const TrainingProfiler* profiler = nullptr) const;
  std::size_t max_bins() const noexcept;
  const std::vector<std::uint16_t>& max_bins_by_feature() const noexcept;
  NanMode nan_mode() const noexcept;
  const std::string& nan_mode_name() const noexcept;
  BorderSelectionMethod border_selection_method() const noexcept;
  const std::string& border_selection_method_name() const noexcept;
  const std::vector<std::string>& nan_mode_by_feature_names() const noexcept;
  const std::vector<std::vector<float>>& feature_borders() const noexcept;
  bool external_memory() const noexcept;
  const std::string& external_memory_dir() const noexcept;

 private:
  std::size_t max_bins_{256};
  std::vector<std::uint16_t> max_bins_by_feature_;
  NanMode nan_mode_{NanMode::Min};
  std::string nan_mode_name_{"Min"};
  BorderSelectionMethod border_selection_method_{BorderSelectionMethod::Quantile};
  std::string border_selection_method_name_{"Quantile"};
  std::vector<NanMode> nan_mode_by_feature_;
  std::vector<std::uint8_t> nan_mode_by_feature_mask_;
  std::vector<std::string> nan_mode_by_feature_names_;
  std::vector<std::vector<float>> feature_borders_;
  bool external_memory_{false};
  std::string external_memory_dir_;
};

}  // namespace ctboost
