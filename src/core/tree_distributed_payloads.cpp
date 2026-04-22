#include "tree_internal.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <thread>

namespace ctboost::detail {
namespace {

template <typename T>
void AppendBinary(std::vector<std::uint8_t>& buffer, const T& value) {
  const auto* bytes = reinterpret_cast<const std::uint8_t*>(&value);
  buffer.insert(buffer.end(), bytes, bytes + sizeof(T));
}

template <typename T>
T ReadBinary(const std::vector<std::uint8_t>& buffer, std::size_t& offset) {
  if (offset + sizeof(T) > buffer.size()) {
    throw std::runtime_error("distributed binary payload is truncated");
  }
  T value{};
  std::memcpy(&value, buffer.data() + offset, sizeof(T));
  offset += sizeof(T);
  return value;
}

}  // namespace

std::vector<std::uint8_t> SerializeNodeHistogramSetBinary(const NodeHistogramSet& stats) {
  std::vector<std::uint8_t> buffer;
  const std::uint64_t feature_count = static_cast<std::uint64_t>(stats.by_feature.size());
  AppendBinary(buffer, feature_count);
  AppendBinary(buffer, stats.sample_count);
  AppendBinary(buffer, stats.sample_weight_sum);
  AppendBinary(buffer, stats.total_gradient);
  AppendBinary(buffer, stats.total_hessian);
  AppendBinary(buffer, stats.gradient_square_sum);
  for (const BinStatistics& feature_stats : stats.by_feature) {
    if (feature_stats.hessian_sums.size() != feature_stats.gradient_sums.size() ||
        feature_stats.weight_sums.size() != feature_stats.gradient_sums.size()) {
      throw std::invalid_argument(
          "distributed node histogram bin-stat vectors must have matching sizes");
    }
    const std::uint64_t bin_count =
        static_cast<std::uint64_t>(feature_stats.gradient_sums.size());
    AppendBinary(buffer, bin_count);
    const auto append_array = [&](const std::vector<double>& values) {
      const auto* bytes = reinterpret_cast<const std::uint8_t*>(values.data());
      buffer.insert(buffer.end(), bytes, bytes + values.size() * sizeof(double));
    };
    append_array(feature_stats.gradient_sums);
    append_array(feature_stats.hessian_sums);
    append_array(feature_stats.weight_sums);
  }
  return buffer;
}

NodeHistogramSet DeserializeNodeHistogramSetBinary(const std::vector<std::uint8_t>& buffer) {
  std::size_t offset = 0;
  NodeHistogramSet stats;
  const std::uint64_t feature_count = ReadBinary<std::uint64_t>(buffer, offset);
  stats.sample_count = ReadBinary<std::uint64_t>(buffer, offset);
  stats.sample_weight_sum = ReadBinary<double>(buffer, offset);
  stats.total_gradient = ReadBinary<double>(buffer, offset);
  stats.total_hessian = ReadBinary<double>(buffer, offset);
  stats.gradient_square_sum = ReadBinary<double>(buffer, offset);
  stats.by_feature.resize(static_cast<std::size_t>(feature_count));
  for (std::size_t feature = 0; feature < stats.by_feature.size(); ++feature) {
    const std::uint64_t bin_count = ReadBinary<std::uint64_t>(buffer, offset);
    BinStatistics feature_stats;
    feature_stats.gradient_sums.resize(static_cast<std::size_t>(bin_count));
    feature_stats.hessian_sums.resize(static_cast<std::size_t>(bin_count));
    feature_stats.weight_sums.resize(static_cast<std::size_t>(bin_count));
    const auto read_array = [&](std::vector<double>& values) {
      const std::size_t byte_count = values.size() * sizeof(double);
      if (offset + byte_count > buffer.size()) {
        throw std::runtime_error("distributed node histogram payload is truncated");
      }
      std::memcpy(values.data(), buffer.data() + offset, byte_count);
      offset += byte_count;
    };
    read_array(feature_stats.gradient_sums);
    read_array(feature_stats.hessian_sums);
    read_array(feature_stats.weight_sums);
    stats.by_feature[feature] = std::move(feature_stats);
  }
  stats.gradient_variance = ComputeGradientVariance(
      stats.total_gradient, stats.gradient_square_sum, stats.sample_weight_sum);
  return stats;
}

void WriteNodeHistogramSetBinary(const std::filesystem::path& path,
                                 const NodeHistogramSet& stats) {
  std::error_code error;
  std::filesystem::create_directories(path.parent_path(), error);
  if (error) {
    throw std::runtime_error(
        "failed to create distributed tree directory: " + error.message());
  }
  const std::vector<std::uint8_t> buffer = SerializeNodeHistogramSetBinary(stats);
  const std::filesystem::path temp_path = path.string() + ".tmp";
  std::ofstream out(temp_path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("failed to open distributed histogram file for writing: " +
                             temp_path.string());
  }
  if (!buffer.empty()) {
    out.write(reinterpret_cast<const char*>(buffer.data()),
              static_cast<std::streamsize>(buffer.size()));
  }
  if (!out) {
    throw std::runtime_error("failed to write distributed histogram file: " + temp_path.string());
  }
  out.close();
  std::filesystem::remove(path, error);
  std::filesystem::rename(temp_path, path, error);
  if (error) {
    throw std::runtime_error("failed to publish distributed histogram file: " + error.message());
  }
}

NodeHistogramSet ReadNodeHistogramSetBinary(const std::filesystem::path& path,
                                            double timeout_seconds) {
  const auto deadline = timeout_seconds > 0.0
                            ? std::chrono::steady_clock::now() +
                                  std::chrono::duration<double>(timeout_seconds)
                            : std::chrono::steady_clock::time_point::min();
  std::string last_error =
      "failed to open distributed histogram file for reading: " + path.string();
  while (true) {
    std::ifstream in(path, std::ios::binary);
    if (in) {
      try {
        const std::vector<std::uint8_t> buffer(
            (std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        return DeserializeNodeHistogramSetBinary(buffer);
      } catch (const std::exception& error) {
        last_error = error.what();
      }
    }
    if (timeout_seconds <= 0.0 || std::chrono::steady_clock::now() >= deadline) {
      throw std::runtime_error(last_error);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

void AddNodeHistogramSet(NodeHistogramSet& target, const NodeHistogramSet& source) {
  if (target.by_feature.size() != source.by_feature.size()) {
    throw std::invalid_argument("distributed histogram feature counts must match");
  }
  target.sample_count += source.sample_count;
  target.sample_weight_sum += source.sample_weight_sum;
  target.total_gradient += source.total_gradient;
  target.total_hessian += source.total_hessian;
  target.gradient_square_sum += source.gradient_square_sum;
  for (std::size_t feature = 0; feature < target.by_feature.size(); ++feature) {
    BinStatistics& target_feature = target.by_feature[feature];
    const BinStatistics& source_feature = source.by_feature[feature];
    if (target_feature.gradient_sums.size() != source_feature.gradient_sums.size()) {
      throw std::invalid_argument("distributed histogram feature bin counts must match");
    }
    for (std::size_t bin = 0; bin < target_feature.gradient_sums.size(); ++bin) {
      target_feature.gradient_sums[bin] += source_feature.gradient_sums[bin];
      target_feature.hessian_sums[bin] += source_feature.hessian_sums[bin];
      target_feature.weight_sums[bin] += source_feature.weight_sums[bin];
    }
  }
}

std::vector<std::uint8_t> SerializeGpuHistogramSnapshotBinary(
    const GpuHistogramSnapshot& snapshot) {
  std::vector<std::uint8_t> buffer;
  const std::uint64_t total_bins =
      static_cast<std::uint64_t>(snapshot.gradient_sums.size());
  if (snapshot.hessian_sums.size() != snapshot.gradient_sums.size() ||
      snapshot.weight_sums.size() != snapshot.gradient_sums.size()) {
    throw std::invalid_argument("gpu histogram snapshot buffers must have matching sizes");
  }
  AppendBinary(buffer, total_bins);
  AppendBinary(buffer, snapshot.node_statistics.sample_count);
  AppendBinary(buffer, snapshot.node_statistics.sample_weight_sum);
  AppendBinary(buffer, snapshot.node_statistics.total_gradient);
  AppendBinary(buffer, snapshot.node_statistics.total_hessian);
  AppendBinary(buffer, snapshot.node_statistics.gradient_square_sum);
  const auto append_float_array = [&](const std::vector<float>& values) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(values.data());
    buffer.insert(buffer.end(), bytes, bytes + values.size() * sizeof(float));
  };
  append_float_array(snapshot.gradient_sums);
  append_float_array(snapshot.hessian_sums);
  append_float_array(snapshot.weight_sums);
  return buffer;
}

GpuHistogramSnapshot DeserializeGpuHistogramSnapshotBinary(
    const std::vector<std::uint8_t>& buffer) {
  std::size_t offset = 0;
  GpuHistogramSnapshot snapshot;
  const std::uint64_t total_bins = ReadBinary<std::uint64_t>(buffer, offset);
  snapshot.node_statistics.sample_count = ReadBinary<std::uint64_t>(buffer, offset);
  snapshot.node_statistics.sample_weight_sum = ReadBinary<double>(buffer, offset);
  snapshot.node_statistics.total_gradient = ReadBinary<double>(buffer, offset);
  snapshot.node_statistics.total_hessian = ReadBinary<double>(buffer, offset);
  snapshot.node_statistics.gradient_square_sum = ReadBinary<double>(buffer, offset);
  snapshot.gradient_sums.resize(static_cast<std::size_t>(total_bins));
  snapshot.hessian_sums.resize(static_cast<std::size_t>(total_bins));
  snapshot.weight_sums.resize(static_cast<std::size_t>(total_bins));
  const auto read_float_array = [&](std::vector<float>& values) {
    const std::size_t byte_count = values.size() * sizeof(float);
    if (offset + byte_count > buffer.size()) {
      throw std::runtime_error("distributed gpu histogram payload is truncated");
    }
    std::memcpy(values.data(), buffer.data() + offset, byte_count);
    offset += byte_count;
  };
  read_float_array(snapshot.gradient_sums);
  read_float_array(snapshot.hessian_sums);
  read_float_array(snapshot.weight_sums);
  return snapshot;
}

GpuHistogramSnapshot SubtractGpuHistogramSnapshot(const GpuHistogramSnapshot& parent,
                                                  const GpuHistogramSnapshot& child) {
  if (parent.gradient_sums.size() != child.gradient_sums.size() ||
      parent.hessian_sums.size() != child.hessian_sums.size() ||
      parent.weight_sums.size() != child.weight_sums.size()) {
    throw std::invalid_argument("gpu histogram snapshots must have matching sizes");
  }
  GpuHistogramSnapshot derived = parent;
  derived.node_statistics.sample_count =
      parent.node_statistics.sample_count >= child.node_statistics.sample_count
          ? parent.node_statistics.sample_count - child.node_statistics.sample_count
          : 0U;
  derived.node_statistics.sample_weight_sum =
      std::max(0.0, parent.node_statistics.sample_weight_sum - child.node_statistics.sample_weight_sum);
  derived.node_statistics.total_gradient =
      parent.node_statistics.total_gradient - child.node_statistics.total_gradient;
  derived.node_statistics.total_hessian =
      std::max(0.0, parent.node_statistics.total_hessian - child.node_statistics.total_hessian);
  derived.node_statistics.gradient_square_sum =
      std::max(0.0,
               parent.node_statistics.gradient_square_sum - child.node_statistics.gradient_square_sum);
  for (std::size_t index = 0; index < derived.gradient_sums.size(); ++index) {
    derived.gradient_sums[index] -= child.gradient_sums[index];
    derived.hessian_sums[index] =
        std::max(0.0F, derived.hessian_sums[index] - child.hessian_sums[index]);
    derived.weight_sums[index] =
        std::max(0.0F, derived.weight_sums[index] - child.weight_sums[index]);
  }
  return derived;
}

}  // namespace ctboost::detail
