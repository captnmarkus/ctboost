#include "histogram_internal.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace ctboost::detail {
namespace {

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

template <typename BinType>
void WriteFeatureBinsToFileImpl(const std::filesystem::path& path, const std::vector<BinType>& bins) {
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

}  // namespace

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

void WriteFeatureBinsToFile(const std::filesystem::path& path,
                            const std::vector<std::uint8_t>& bins) {
  WriteFeatureBinsToFileImpl(path, bins);
}

void WriteFeatureBinsToFile(const std::filesystem::path& path,
                            const std::vector<std::uint16_t>& bins) {
  WriteFeatureBinsToFileImpl(path, bins);
}

}  // namespace ctboost::detail

namespace ctboost {

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
        detail::ReadFeatureBinsFromFile(feature_path, external_feature_cache_u8);
      } else {
        external_feature_cache_u16.resize(num_rows);
        external_feature_cache_u8.clear();
        external_feature_cache_u8.shrink_to_fit();
        detail::ReadFeatureBinsFromFile(feature_path, external_feature_cache_u16);
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

  const std::size_t max_feature_bins = detail::MaxFeatureBins(num_bins_per_feature);
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

void HistMatrix::SpillBinStorage(const std::string& directory) {
  if (uses_external_bin_storage_ || num_rows == 0 || num_cols == 0) {
    return;
  }
  if (bin_index_bytes == 1 && compact_bin_indices.empty()) {
    throw std::runtime_error("cannot spill released compact histogram bin storage");
  }
  if (bin_index_bytes != 1 && bin_indices.empty()) {
    throw std::runtime_error("cannot spill released histogram bin storage");
  }

  const std::filesystem::path root = detail::MakeExternalStorageRoot(directory);
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
      detail::WriteFeatureBinsToFile(feature_path, feature_bins);
    } else {
      std::vector<std::uint16_t> feature_bins(num_rows, 0);
      std::copy_n(bin_indices.begin() + static_cast<std::ptrdiff_t>(offset),
                  static_cast<std::ptrdiff_t>(num_rows),
                  feature_bins.begin());
      detail::WriteFeatureBinsToFile(feature_path, feature_bins);
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

}  // namespace ctboost
