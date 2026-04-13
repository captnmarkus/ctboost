#include "ctboost/tree.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "ctboost/cuda_backend.hpp"
#include "ctboost/distributed_client.hpp"
#include "ctboost/profiler.hpp"

namespace ctboost {

struct NodeHistogramSet {
  std::vector<BinStatistics> by_feature;
  std::uint64_t sample_count{0};
  double sample_weight_sum{0.0};
  double total_gradient{0.0};
  double total_hessian{0.0};
  double gradient_square_sum{0.0};
  double gradient_variance{0.0};
};

namespace {

struct FeatureChoice {
  int feature_id{-1};
  double p_value{1.0};
  double chi_square{-std::numeric_limits<double>::infinity()};
};

struct SplitChoice {
  bool valid{false};
  bool is_categorical{false};
  std::uint16_t split_bin{0};
  std::array<std::uint8_t, kMaxCategoricalRouteBins> left_categories{};
  double gain{-std::numeric_limits<double>::infinity()};
  double left_leaf_weight{0.0};
  double right_leaf_weight{0.0};
};

struct CandidateSelectionResult {
  FeatureChoice feature_choice;
  SplitChoice split_choice;
  double adjusted_gain{-std::numeric_limits<double>::infinity()};
};

double ComputeLeafWeight(double gradient_sum, double hessian_sum, double lambda_l2) {
  return -gradient_sum / (hessian_sum + lambda_l2);
}

double ComputeGain(double gradient_sum, double hessian_sum, double lambda_l2) {
  return (gradient_sum * gradient_sum) / (hessian_sum + lambda_l2);
}

double ComputeGradientVariance(double weighted_gradient_sum,
                               double weighted_gradient_square_sum,
                               double sample_weight_sum) {
  if (sample_weight_sum <= 0.0) {
    return 0.0;
  }

  const double mean_gradient = weighted_gradient_sum / sample_weight_sum;
  const double second_moment = weighted_gradient_square_sum / sample_weight_sum;
  return std::max(0.0, second_moment - mean_gradient * mean_gradient);
}

const QuantizationSchema& RequireQuantizationSchema(const QuantizationSchemaPtr& schema) {
  if (schema == nullptr) {
    throw std::runtime_error("tree quantization schema is not initialized");
  }
  return *schema;
}

double ClampLeafWeight(double leaf_weight, double lower_bound, double upper_bound) {
  return std::clamp(leaf_weight, lower_bound, upper_bound);
}

std::filesystem::path DistributedTreeRoot(const DistributedCoordinator& coordinator) {
  return std::filesystem::path(coordinator.root) / coordinator.run_id /
         ("tree_" + std::to_string(coordinator.tree_index));
}

std::filesystem::path DistributedOperationDir(DistributedCoordinator& coordinator,
                                              const char* label) {
  std::ostringstream stream;
  stream << label << "_" << std::setw(10) << std::setfill('0') << coordinator.operation_counter++;
  return DistributedTreeRoot(coordinator) / stream.str();
}

void WaitForDistributedPath(const std::filesystem::path& path, double timeout_seconds) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_seconds);
  while (!std::filesystem::exists(path)) {
    if (std::chrono::steady_clock::now() >= deadline) {
      throw std::runtime_error("timed out waiting for distributed artifact: " + path.string());
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

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

NodeHistogramSet ReadNodeHistogramSetBinary(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open distributed histogram file for reading: " +
                             path.string());
  }
  const std::vector<std::uint8_t> buffer(
      (std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  return DeserializeNodeHistogramSetBinary(buffer);
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

NodeHistogramSet AllReduceNodeHistogramSet(DistributedCoordinator* coordinator,
                                           const NodeHistogramSet& local_stats) {
  if (coordinator == nullptr || coordinator->world_size <= 1) {
    return local_stats;
  }

  if (DistributedRootUsesTcp(coordinator->root)) {
    const std::filesystem::path operation_dir = DistributedOperationDir(*coordinator, "node_hist");
    const std::vector<std::uint8_t> response = DistributedTcpRequest(
        coordinator->root,
        coordinator->timeout_seconds,
        "node_hist_reduce",
        operation_dir.generic_string(),
        coordinator->rank,
        coordinator->world_size,
        SerializeNodeHistogramSetBinary(local_stats));
    NodeHistogramSet reduced = DeserializeNodeHistogramSetBinary(response);
    reduced.total_hessian = std::max(0.0, reduced.total_hessian);
    reduced.gradient_square_sum = std::max(0.0, reduced.gradient_square_sum);
    reduced.gradient_variance = ComputeGradientVariance(
        reduced.total_gradient, reduced.gradient_square_sum, reduced.sample_weight_sum);
    return reduced;
  }

  const std::filesystem::path operation_dir = DistributedOperationDir(*coordinator, "node_hist");
  const auto rank_file_name = [&](int rank) {
    std::ostringstream stream;
    stream << "rank_" << std::setw(5) << std::setfill('0') << rank << ".bin";
    return stream.str();
  };
  const std::filesystem::path local_path = operation_dir / rank_file_name(coordinator->rank);
  WriteNodeHistogramSetBinary(local_path, local_stats);
  const std::filesystem::path result_path = operation_dir / "result.bin";

  if (coordinator->rank == 0) {
    NodeHistogramSet reduced;
    bool initialized = false;
    for (int rank = 0; rank < coordinator->world_size; ++rank) {
      const std::filesystem::path rank_path = operation_dir / rank_file_name(rank);
      WaitForDistributedPath(rank_path, coordinator->timeout_seconds);
      const NodeHistogramSet rank_stats = ReadNodeHistogramSetBinary(rank_path);
      if (!initialized) {
        reduced = rank_stats;
        initialized = true;
      } else {
        AddNodeHistogramSet(reduced, rank_stats);
      }
    }
    reduced.total_hessian = std::max(0.0, reduced.total_hessian);
    reduced.gradient_square_sum = std::max(0.0, reduced.gradient_square_sum);
    reduced.gradient_variance = ComputeGradientVariance(
        reduced.total_gradient, reduced.gradient_square_sum, reduced.sample_weight_sum);
    WriteNodeHistogramSetBinary(result_path, reduced);
    return reduced;
  }

  WaitForDistributedPath(result_path, coordinator->timeout_seconds);
  return ReadNodeHistogramSetBinary(result_path);
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

GpuHistogramSnapshot AllReduceGpuHistogramSnapshot(DistributedCoordinator* coordinator,
                                                   const GpuHistogramSnapshot& local_snapshot) {
  if (coordinator == nullptr || coordinator->world_size <= 1) {
    return local_snapshot;
  }
  if (!DistributedRootUsesTcp(coordinator->root)) {
    throw std::invalid_argument(
        "distributed gpu histogram reduction currently requires a tcp:// coordinator root");
  }
  const std::filesystem::path operation_dir = DistributedOperationDir(*coordinator, "gpu_hist");
  const std::vector<std::uint8_t> response = DistributedTcpRequest(
      coordinator->root,
      coordinator->timeout_seconds,
      "gpu_snapshot_reduce",
      operation_dir.generic_string(),
      coordinator->rank,
      coordinator->world_size,
      SerializeGpuHistogramSnapshotBinary(local_snapshot));
  return DeserializeGpuHistogramSnapshotBinary(response);
}

std::uint64_t MixRandomKey(std::uint64_t value) {
  value += 0x9E3779B97F4A7C15ULL;
  value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
  return value ^ (value >> 31U);
}

double UniformFromKey(std::uint64_t key) {
  constexpr double kScale = 1.0 / static_cast<double>(1ULL << 53U);
  return static_cast<double>(MixRandomKey(key) >> 11U) * kScale;
}

double SymmetricNoise(std::uint64_t base_seed, std::uint64_t key, double scale) {
  if (scale <= 0.0) {
    return 0.0;
  }
  const double uniform = UniformFromKey(base_seed ^ key);
  return (2.0 * uniform - 1.0) * scale;
}

double FeatureWeightValue(const std::vector<double>* feature_weights, int feature_id) {
  if (feature_weights == nullptr || feature_id < 0 ||
      static_cast<std::size_t>(feature_id) >= feature_weights->size()) {
    return 1.0;
  }
  return (*feature_weights)[static_cast<std::size_t>(feature_id)];
}

double FirstUsePenaltyValue(const std::vector<double>* first_feature_use_penalties,
                            const std::vector<std::uint8_t>* model_feature_used_mask,
                            int feature_id) {
  if (first_feature_use_penalties == nullptr || model_feature_used_mask == nullptr ||
      feature_id < 0) {
    return 0.0;
  }
  const std::size_t index = static_cast<std::size_t>(feature_id);
  if (index >= first_feature_use_penalties->size() || index >= model_feature_used_mask->size() ||
      (*model_feature_used_mask)[index] != 0U) {
    return 0.0;
  }
  return (*first_feature_use_penalties)[index];
}

bool UsesCandidateRegularization(const TreeBuildOptions& options) {
  return options.random_strength > 0.0 ||
         options.feature_weights != nullptr ||
         options.first_feature_use_penalties != nullptr;
}

std::vector<FeatureChoice> RankFeaturesByStatistic(const NodeHistogramSet& node_stats,
                                                   const LinearStatistic& statistic_engine,
                                                   const std::vector<int>* allowed_features) {
  std::vector<FeatureChoice> ranked_features;

  const auto evaluate_feature = [&](std::size_t feature) {
    const BinStatistics& feature_stats = node_stats.by_feature[feature];
    if (feature_stats.weight_sums.size() <= 1) {
      return;
    }

    const auto result = statistic_engine.EvaluateScoreFromBinStatistics(
        feature_stats,
        node_stats.total_gradient,
        node_stats.sample_weight_sum,
        node_stats.gradient_variance);
    if (result.degrees_of_freedom == 0) {
      return;
    }

    ranked_features.push_back(
        FeatureChoice{
            static_cast<int>(feature),
            result.p_value,
            result.chi_square,
        });
  };

  if (allowed_features != nullptr && !allowed_features->empty()) {
    for (int feature_id : *allowed_features) {
      if (feature_id < 0 ||
          static_cast<std::size_t>(feature_id) >= node_stats.by_feature.size()) {
        continue;
      }
      evaluate_feature(static_cast<std::size_t>(feature_id));
    }
  } else {
    for (std::size_t feature = 0; feature < node_stats.by_feature.size(); ++feature) {
      evaluate_feature(feature);
    }
  }

  std::sort(ranked_features.begin(),
            ranked_features.end(),
            [](const FeatureChoice& lhs, const FeatureChoice& rhs) {
              if (std::abs(lhs.p_value - rhs.p_value) > 1e-12) {
                return lhs.p_value < rhs.p_value;
              }
              return lhs.chi_square > rhs.chi_square;
            });
  return ranked_features;
}

bool FeatureAllowedByInteraction(int feature_id,
                                 const InteractionConstraintSet& constraints,
                                 const std::vector<int>* active_groups) {
  if (feature_id < 0 ||
      static_cast<std::size_t>(feature_id) >= constraints.feature_to_groups.size()) {
    return false;
  }
  if (constraints.constrained_feature_mask[static_cast<std::size_t>(feature_id)] == 0U) {
    return true;
  }
  if (active_groups == nullptr || active_groups->empty()) {
    return true;
  }
  const auto& feature_groups = constraints.feature_to_groups[static_cast<std::size_t>(feature_id)];
  for (const int group_id : feature_groups) {
    if (std::find(active_groups->begin(), active_groups->end(), group_id) != active_groups->end()) {
      return true;
    }
  }
  return false;
}

std::vector<int> FilterAllowedFeaturesForInteraction(
    std::size_t num_features,
    const std::vector<int>* parent_allowed_features,
    const InteractionConstraintSet& constraints,
    const std::vector<int>* active_groups) {
  std::vector<int> filtered_features;
  if (parent_allowed_features != nullptr && !parent_allowed_features->empty()) {
    filtered_features.reserve(parent_allowed_features->size());
    for (const int feature_id : *parent_allowed_features) {
      if (FeatureAllowedByInteraction(feature_id, constraints, active_groups)) {
        filtered_features.push_back(feature_id);
      }
    }
    return filtered_features;
  }

  filtered_features.reserve(num_features);
  for (std::size_t feature = 0; feature < num_features; ++feature) {
    const int feature_id = static_cast<int>(feature);
    if (FeatureAllowedByInteraction(feature_id, constraints, active_groups)) {
      filtered_features.push_back(feature_id);
    }
  }
  return filtered_features;
}

std::vector<int> IntersectSortedVectors(const std::vector<int>& lhs, const std::vector<int>& rhs) {
  std::vector<int> intersection;
  std::set_intersection(lhs.begin(),
                        lhs.end(),
                        rhs.begin(),
                        rhs.end(),
                        std::back_inserter(intersection));
  return intersection;
}

NodeHistogramSet ComputeNodeHistogramSet(const HistMatrix& hist,
                                         const std::vector<float>& gradients,
                                         const std::vector<float>& hessians,
                                         const std::vector<float>& weights,
                                         const std::vector<std::size_t>& row_indices,
                                         std::size_t row_begin,
                                         std::size_t row_end,
                                         bool use_gpu,
                                         GpuHistogramWorkspace* gpu_workspace) {
  NodeHistogramSet stats;
  stats.by_feature.resize(hist.num_cols);

  if (use_gpu) {
    (void)gpu_workspace;
    throw std::invalid_argument(
        "GPU node histogram materialization is no longer supported in the CPU search path");
  }

  for (std::size_t index = row_begin; index < row_end; ++index) {
    const std::size_t row = row_indices[index];
    const double gradient = gradients[row];
    const double hessian = hessians[row];
    const double sample_weight = weights[row];
    ++stats.sample_count;
    stats.total_gradient += sample_weight * gradient;
    stats.total_hessian += sample_weight * hessian;
    stats.gradient_square_sum += sample_weight * gradient * gradient;
    stats.sample_weight_sum += sample_weight;
  }
  stats.gradient_variance = ComputeGradientVariance(
      stats.total_gradient, stats.gradient_square_sum, stats.sample_weight_sum);

  for (std::size_t feature = 0; feature < hist.num_cols; ++feature) {
    const std::size_t feature_bins_count = hist.num_bins(feature);
    BinStatistics feature_stats;
    feature_stats.gradient_sums.assign(feature_bins_count, 0.0);
    feature_stats.hessian_sums.assign(feature_bins_count, 0.0);
    feature_stats.weight_sums.assign(feature_bins_count, 0.0);

    const auto feature_bins = hist.feature_bins(feature);
    for (std::size_t index = row_begin; index < row_end; ++index) {
      const std::size_t row = row_indices[index];
      const std::size_t bin = feature_bins[row];
      const double sample_weight = static_cast<double>(weights[row]);
      feature_stats.gradient_sums[bin] += sample_weight * gradients[row];
      feature_stats.hessian_sums[bin] += sample_weight * hessians[row];
      feature_stats.weight_sums[bin] += sample_weight;
    }

    stats.by_feature[feature] = std::move(feature_stats);
  }

  return stats;
}

NodeHistogramSet SubtractNodeHistogramSet(const NodeHistogramSet& parent,
                                          const NodeHistogramSet& child) {
  if (parent.by_feature.size() != child.by_feature.size()) {
    throw std::invalid_argument("parent and child histogram sets must have the same feature count");
  }

  NodeHistogramSet derived;
  derived.by_feature.resize(parent.by_feature.size());
  derived.sample_count =
      parent.sample_count >= child.sample_count ? parent.sample_count - child.sample_count : 0U;
  derived.sample_weight_sum = parent.sample_weight_sum - child.sample_weight_sum;
  derived.total_gradient = parent.total_gradient - child.total_gradient;
  derived.total_hessian = parent.total_hessian - child.total_hessian;
  derived.gradient_square_sum = parent.gradient_square_sum - child.gradient_square_sum;

  for (std::size_t feature = 0; feature < parent.by_feature.size(); ++feature) {
    const BinStatistics& parent_stats = parent.by_feature[feature];
    const BinStatistics& child_stats = child.by_feature[feature];
    if (parent_stats.gradient_sums.size() != child_stats.gradient_sums.size()) {
      throw std::invalid_argument(
          "parent and child histogram sets must have matching bin counts");
    }

    BinStatistics feature_stats;
    feature_stats.gradient_sums.resize(parent_stats.gradient_sums.size());
    feature_stats.hessian_sums.resize(parent_stats.hessian_sums.size());
    feature_stats.weight_sums.resize(parent_stats.weight_sums.size());
    for (std::size_t bin = 0; bin < parent_stats.gradient_sums.size(); ++bin) {
      feature_stats.gradient_sums[bin] =
          parent_stats.gradient_sums[bin] - child_stats.gradient_sums[bin];
      feature_stats.hessian_sums[bin] = std::max(
          0.0, parent_stats.hessian_sums[bin] - child_stats.hessian_sums[bin]);
      feature_stats.weight_sums[bin] =
          std::max(0.0, parent_stats.weight_sums[bin] - child_stats.weight_sums[bin]);
    }
    derived.by_feature[feature] = std::move(feature_stats);
  }

  derived.sample_weight_sum = std::max(0.0, derived.sample_weight_sum);
  derived.total_hessian = std::max(0.0, derived.total_hessian);
  derived.gradient_square_sum = std::max(0.0, derived.gradient_square_sum);
  derived.gradient_variance = ComputeGradientVariance(
      derived.total_gradient, derived.gradient_square_sum, derived.sample_weight_sum);
  return derived;
}

FeatureChoice SelectBestFeature(const NodeHistogramSet& node_stats,
                                const LinearStatistic& statistic_engine,
                                const std::vector<int>* allowed_features) {
  FeatureChoice best;

  const auto evaluate_feature = [&](std::size_t feature) {
    const BinStatistics& feature_stats = node_stats.by_feature[feature];
    if (feature_stats.weight_sums.size() <= 1) {
      return;
    }

    const auto result = statistic_engine.EvaluateScoreFromBinStatistics(
        feature_stats,
        node_stats.total_gradient,
        node_stats.sample_weight_sum,
        node_stats.gradient_variance);
    if (result.degrees_of_freedom == 0) {
      return;
    }

    if (best.feature_id < 0 || result.p_value < best.p_value ||
        (std::abs(result.p_value - best.p_value) <= 1e-12 &&
         result.chi_square > best.chi_square)) {
      best.feature_id = static_cast<int>(feature);
      best.p_value = result.p_value;
      best.chi_square = result.chi_square;
    }
  };

  if (allowed_features != nullptr && !allowed_features->empty()) {
    for (int feature_id : *allowed_features) {
      if (feature_id < 0 ||
          static_cast<std::size_t>(feature_id) >= node_stats.by_feature.size()) {
        continue;
      }
      evaluate_feature(static_cast<std::size_t>(feature_id));
    }
    return best;
  }

  for (std::size_t feature = 0; feature < node_stats.by_feature.size(); ++feature) {
    evaluate_feature(feature);
  }

  return best;
}

SplitChoice SelectBestSplit(const BinStatistics& feature_stats,
                            double total_gradient,
                            double total_hessian,
                            double sample_weight_sum,
                            double lambda_l2,
                            int min_data_in_leaf,
                            double min_child_weight,
                            double min_split_gain,
                            bool is_categorical,
                            int monotone_sign,
                            double leaf_lower_bound,
                            double leaf_upper_bound) {
  SplitChoice best;
  const std::size_t feature_bins_count = feature_stats.gradient_sums.size();
  if (feature_bins_count <= 1) {
    return best;
  }

  const double parent_gain = ComputeGain(total_gradient, total_hessian, lambda_l2);

  if (!is_categorical) {
    double left_gradient = 0.0;
    double left_hessian = 0.0;
    double left_count = 0.0;

    for (std::size_t split_bin = 0; split_bin + 1 < feature_bins_count; ++split_bin) {
      left_gradient += feature_stats.gradient_sums[split_bin];
      left_hessian += feature_stats.hessian_sums[split_bin];
      left_count += feature_stats.weight_sums[split_bin];

      const double right_count = sample_weight_sum - left_count;
      if (left_count <= 0.0 || right_count <= 0.0 ||
          left_count < static_cast<double>(min_data_in_leaf) ||
          right_count < static_cast<double>(min_data_in_leaf)) {
        continue;
      }

      const double right_gradient = total_gradient - left_gradient;
      const double right_hessian = total_hessian - left_hessian;
      if (left_hessian < min_child_weight || right_hessian < min_child_weight) {
        continue;
      }
      const double left_leaf_weight = ClampLeafWeight(
          ComputeLeafWeight(left_gradient, left_hessian, lambda_l2), leaf_lower_bound, leaf_upper_bound);
      const double right_leaf_weight = ClampLeafWeight(
          ComputeLeafWeight(right_gradient, right_hessian, lambda_l2), leaf_lower_bound, leaf_upper_bound);
      if ((monotone_sign > 0 && left_leaf_weight > right_leaf_weight + 1e-12) ||
          (monotone_sign < 0 && left_leaf_weight + 1e-12 < right_leaf_weight)) {
        continue;
      }
      const double gain = ComputeGain(left_gradient, left_hessian, lambda_l2) +
                          ComputeGain(right_gradient, right_hessian, lambda_l2) - parent_gain;
      if (gain <= min_split_gain) {
        continue;
      }

      if (!best.valid || gain > best.gain) {
        best.valid = true;
        best.split_bin = static_cast<std::uint16_t>(split_bin);
        best.gain = gain;
        best.left_leaf_weight = left_leaf_weight;
        best.right_leaf_weight = right_leaf_weight;
      }
    }

    return best;
  }

  if (monotone_sign != 0) {
    return best;
  }

  if (feature_bins_count > kMaxCategoricalRouteBins) {
    throw std::invalid_argument("categorical split routing supports at most 256 bins");
  }

  struct WeightedBin {
    std::uint16_t bin{0};
    double gradient{0.0};
    double hessian{0.0};
    double count{0.0};
    double weight{0.0};
  };

  std::vector<WeightedBin> active_bins;
  active_bins.reserve(feature_bins_count);
  for (std::size_t bin = 0; bin < feature_bins_count; ++bin) {
    if (feature_stats.weight_sums[bin] <= 0.0) {
      continue;
    }

    const double denominator = feature_stats.hessian_sums[bin] + lambda_l2;
    active_bins.push_back(WeightedBin{
        static_cast<std::uint16_t>(bin),
        feature_stats.gradient_sums[bin],
        feature_stats.hessian_sums[bin],
        feature_stats.weight_sums[bin],
        denominator > 0.0 ? feature_stats.gradient_sums[bin] / denominator : 0.0,
    });
  }

  if (active_bins.size() <= 1) {
    return best;
  }

  std::sort(active_bins.begin(), active_bins.end(), [](const WeightedBin& lhs, const WeightedBin& rhs) {
    if (lhs.weight == rhs.weight) {
      return lhs.bin < rhs.bin;
    }
    return lhs.weight < rhs.weight;
  });

  double left_gradient = 0.0;
  double left_hessian = 0.0;
  double left_count = 0.0;

  for (std::size_t split_index = 0; split_index + 1 < active_bins.size(); ++split_index) {
    left_gradient += active_bins[split_index].gradient;
    left_hessian += active_bins[split_index].hessian;
    left_count += active_bins[split_index].count;

    const double right_count = sample_weight_sum - left_count;
    if (left_count <= 0.0 || right_count <= 0.0 ||
        left_count < static_cast<double>(min_data_in_leaf) ||
        right_count < static_cast<double>(min_data_in_leaf)) {
      continue;
    }

    const double right_gradient = total_gradient - left_gradient;
    const double right_hessian = total_hessian - left_hessian;
    if (left_hessian < min_child_weight || right_hessian < min_child_weight) {
      continue;
    }
    const double left_leaf_weight = ClampLeafWeight(
        ComputeLeafWeight(left_gradient, left_hessian, lambda_l2), leaf_lower_bound, leaf_upper_bound);
    const double right_leaf_weight = ClampLeafWeight(
        ComputeLeafWeight(right_gradient, right_hessian, lambda_l2), leaf_lower_bound, leaf_upper_bound);
    const double gain = ComputeGain(left_gradient, left_hessian, lambda_l2) +
                        ComputeGain(right_gradient, right_hessian, lambda_l2) - parent_gain;
    if (gain <= min_split_gain) {
      continue;
    }

    if (!best.valid || gain > best.gain) {
      best.valid = true;
      best.is_categorical = true;
      best.gain = gain;
      best.left_leaf_weight = left_leaf_weight;
      best.right_leaf_weight = right_leaf_weight;
      best.left_categories.fill(0);
      for (std::size_t left_index = 0; left_index <= split_index; ++left_index) {
        best.left_categories[active_bins[left_index].bin] = 1;
      }
    }
  }

  return best;
}

double AdjustedCandidateGain(const TreeBuildOptions& options,
                             int feature_id,
                             double raw_gain,
                             int depth,
                             std::size_t row_begin,
                             std::size_t row_end) {
  const double feature_weight = FeatureWeightValue(options.feature_weights, feature_id);
  if (feature_weight <= 0.0) {
    return -std::numeric_limits<double>::infinity();
  }

  double adjusted_gain = raw_gain * feature_weight;
  adjusted_gain -= FirstUsePenaltyValue(
      options.first_feature_use_penalties, options.model_feature_used_mask, feature_id);
  adjusted_gain += SymmetricNoise(
      options.random_seed,
      static_cast<std::uint64_t>(feature_id + 1) ^
          (static_cast<std::uint64_t>(depth + 1) << 24U) ^
          (static_cast<std::uint64_t>(row_begin + 1) << 1U) ^
          (static_cast<std::uint64_t>(row_end + 1) << 33U),
      options.random_strength);
  return adjusted_gain;
}

CandidateSelectionResult SelectBestCandidateSplit(const HistMatrix& hist,
                                                  const NodeHistogramSet& node_stats,
                                                  const TreeBuildOptions& options,
                                                  const LinearStatistic& statistic_engine,
                                                  const std::vector<int>* node_allowed_features,
                                                  double leaf_lower_bound,
                                                  double leaf_upper_bound,
                                                  int depth,
                                                  std::size_t row_begin,
                                                  std::size_t row_end) {
  CandidateSelectionResult best;
  const bool constrained_search =
      (options.monotone_constraints != nullptr && !options.monotone_constraints->empty()) ||
      options.interaction_constraints != nullptr;
  const bool use_ranked_search = constrained_search || UsesCandidateRegularization(options);

  if (!use_ranked_search) {
    best.feature_choice = SelectBestFeature(node_stats, statistic_engine, node_allowed_features);
    if (best.feature_choice.feature_id < 0 || best.feature_choice.p_value > options.alpha) {
      return best;
    }

    best.split_choice = SelectBestSplit(
        node_stats.by_feature[static_cast<std::size_t>(best.feature_choice.feature_id)],
        node_stats.total_gradient,
        node_stats.total_hessian,
        node_stats.sample_weight_sum,
        options.lambda_l2,
        options.min_data_in_leaf,
        options.min_child_weight,
        options.min_split_gain,
        hist.is_categorical(static_cast<std::size_t>(best.feature_choice.feature_id)),
        0,
        leaf_lower_bound,
        leaf_upper_bound);
    best.adjusted_gain = best.split_choice.gain;
    return best;
  }

  const std::vector<FeatureChoice> ranked_features =
      RankFeaturesByStatistic(node_stats, statistic_engine, node_allowed_features);
  if (!ranked_features.empty()) {
    best.feature_choice = ranked_features.front();
  }

  for (const FeatureChoice& candidate : ranked_features) {
    if (candidate.p_value > options.alpha) {
      break;
    }

    const int monotone_sign =
        options.monotone_constraints == nullptr ||
                static_cast<std::size_t>(candidate.feature_id) >= options.monotone_constraints->size()
            ? 0
            : (*options.monotone_constraints)[static_cast<std::size_t>(candidate.feature_id)];
    const SplitChoice candidate_split = SelectBestSplit(
        node_stats.by_feature[static_cast<std::size_t>(candidate.feature_id)],
        node_stats.total_gradient,
        node_stats.total_hessian,
        node_stats.sample_weight_sum,
        options.lambda_l2,
        options.min_data_in_leaf,
        options.min_child_weight,
        options.min_split_gain,
        hist.is_categorical(static_cast<std::size_t>(candidate.feature_id)),
        monotone_sign,
        leaf_lower_bound,
        leaf_upper_bound);
    if (!candidate_split.valid || candidate_split.gain <= 0.0) {
      continue;
    }

    const std::size_t adjusted_row_begin = options.distributed == nullptr ? row_begin : 0U;
    const std::size_t adjusted_row_end = options.distributed == nullptr
                                             ? row_end
                                             : static_cast<std::size_t>(node_stats.sample_count);
    const double adjusted_gain = AdjustedCandidateGain(options,
                                                       candidate.feature_id,
                                                       candidate_split.gain,
                                                       depth,
                                                       adjusted_row_begin,
                                                       adjusted_row_end);
    if (!best.split_choice.valid || adjusted_gain > best.adjusted_gain) {
      best.feature_choice = candidate;
      best.split_choice = candidate_split;
      best.adjusted_gain = adjusted_gain;
    }
  }

  return best;
}

}  // namespace

void Tree::Build(const HistMatrix& hist,
                 const std::vector<float>& gradients,
                 const std::vector<float>& hessians,
                 const std::vector<float>& weights,
                 const TreeBuildOptions& options,
                 GpuHistogramWorkspace* gpu_workspace,
                 const TrainingProfiler* profiler,
                 std::vector<std::size_t>* row_indices_out,
                 std::vector<LeafRowRange>* leaf_row_ranges_out,
                 const QuantizationSchemaPtr& quantization_schema) {
  if (weights.size() != hist.num_rows) {
    throw std::invalid_argument("weight size must match the histogram row count");
  }
  if (options.use_gpu) {
    if (gpu_workspace == nullptr) {
      throw std::invalid_argument("GPU histogram workspace must be provided when task_type='GPU'");
    }
    if ((!gradients.empty() && gradients.size() != hist.num_rows) ||
        (!hessians.empty() && hessians.size() != hist.num_rows)) {
      throw std::invalid_argument(
          "non-empty gradient and hessian buffers must match the histogram row count");
    }
  } else if (gradients.size() != hist.num_rows || hessians.size() != hist.num_rows) {
    throw std::invalid_argument(
        "gradient, hessian, and weight sizes must match the histogram row count");
  }

  nodes_.clear();
  if (quantization_schema != nullptr) {
    if (quantization_schema->num_cols() != hist.num_cols) {
      throw std::invalid_argument(
          "tree quantization schema feature count must match the histogram feature count");
    }
    quantization_schema_ = quantization_schema;
  } else {
    quantization_schema_ = std::make_shared<QuantizationSchema>(MakeQuantizationSchema(hist));
  }
  feature_importances_.assign(hist.num_cols, 0.0);

  std::vector<std::size_t> row_indices;
  const std::size_t initial_row_count = hist.num_rows;
  if (!options.use_gpu) {
    row_indices.resize(hist.num_rows);
    std::iota(row_indices.begin(), row_indices.end(), 0);
  } else {
    ResetHistogramRowIndicesGpu(gpu_workspace);
  }
  if (leaf_row_ranges_out != nullptr) {
    leaf_row_ranges_out->clear();
  }

  int leaf_count = 1;
  const LinearStatistic statistic_engine;
  const double root_leaf_lower_bound =
      options.max_leaf_weight > 0.0 ? -options.max_leaf_weight : -std::numeric_limits<double>::infinity();
  const double root_leaf_upper_bound =
      options.max_leaf_weight > 0.0 ? options.max_leaf_weight : std::numeric_limits<double>::infinity();
  BuildNode(hist,
            gradients,
            hessians,
            weights,
            row_indices,
            0,
            initial_row_count,
            0,
            options,
            gpu_workspace,
            nullptr,
            false,
            nullptr,
            0.0,
            options.allowed_features,
            nullptr,
            root_leaf_lower_bound,
            root_leaf_upper_bound,
            profiler,
            statistic_engine,
            leaf_row_ranges_out,
            &leaf_count);
  if (row_indices_out != nullptr) {
    if (options.use_gpu) {
      DownloadHistogramRowIndicesGpu(gpu_workspace, *row_indices_out);
    } else {
      *row_indices_out = std::move(row_indices);
    }
  }
}

int Tree::BuildNode(const HistMatrix& hist,
                    const std::vector<float>& gradients,
                    const std::vector<float>& hessians,
                    const std::vector<float>& weights,
                    std::vector<std::size_t>& row_indices,
                    std::size_t row_begin,
                    std::size_t row_end,
                    int depth,
                    const TreeBuildOptions& options,
                    GpuHistogramWorkspace* gpu_workspace,
                    const GpuHistogramSnapshot* precomputed_gpu_histogram,
                    bool precomputed_gpu_histogram_resident,
                    const NodeHistogramSet* precomputed_node_stats,
                    double precomputed_histogram_ms,
                    const std::vector<int>* node_allowed_features,
                    const std::vector<int>* active_interaction_groups,
                    double leaf_lower_bound,
                    double leaf_upper_bound,
                    const TrainingProfiler* profiler,
                    const LinearStatistic& statistic_engine,
                    std::vector<LeafRowRange>* leaf_row_ranges_out,
                    int* leaf_count) {
  const std::size_t row_count = row_end - row_begin;
  if (options.use_gpu) {
    (void)precomputed_node_stats;
    (void)statistic_engine;

    GpuNodeStatistics gpu_node_stats;
    GpuHistogramSnapshot parent_snapshot;
    double histogram_ms = precomputed_histogram_ms;
    if (precomputed_gpu_histogram != nullptr) {
      gpu_node_stats = precomputed_gpu_histogram->node_statistics;
      parent_snapshot = *precomputed_gpu_histogram;
      if (!precomputed_gpu_histogram_resident) {
        UploadHistogramSnapshotGpu(gpu_workspace, parent_snapshot);
      }
    } else {
      const auto histogram_start = std::chrono::steady_clock::now();
      GpuHistogramSnapshot local_snapshot;
      BuildHistogramsGpu(gpu_workspace, row_begin, row_end, &local_snapshot.node_statistics);
      DownloadHistogramSnapshotGpu(gpu_workspace, &local_snapshot);
      if (options.distributed != nullptr) {
        parent_snapshot = AllReduceGpuHistogramSnapshot(options.distributed, local_snapshot);
        UploadHistogramSnapshotGpu(gpu_workspace, parent_snapshot);
      } else {
        parent_snapshot = std::move(local_snapshot);
      }
      gpu_node_stats = parent_snapshot.node_statistics;
      histogram_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - histogram_start)
              .count();
    }
    const std::size_t effective_row_count =
        options.distributed == nullptr ? row_count
                                       : static_cast<std::size_t>(gpu_node_stats.sample_count);
    if (profiler != nullptr && profiler->enabled()) {
      profiler->LogNodeHistogram(depth, effective_row_count, true, histogram_ms);
    }

    Node node;
    node.leaf_weight = static_cast<float>(ClampLeafWeight(
        ComputeLeafWeight(gpu_node_stats.total_gradient, gpu_node_stats.total_hessian, options.lambda_l2),
        leaf_lower_bound,
        leaf_upper_bound));

    const int node_index = static_cast<int>(nodes_.size());
    nodes_.push_back(node);
    if (leaf_row_ranges_out != nullptr && leaf_row_ranges_out->size() < nodes_.size()) {
      leaf_row_ranges_out->resize(nodes_.size());
    }

    if (effective_row_count < static_cast<std::size_t>(options.min_samples_split) ||
        depth >= options.max_depth ||
        (options.max_leaves > 0 && leaf_count != nullptr && *leaf_count >= options.max_leaves)) {
      if (leaf_row_ranges_out != nullptr) {
        (*leaf_row_ranges_out)[node_index] = LeafRowRange{row_begin, row_end};
      }
      return node_index;
    }

    const auto search_start = std::chrono::steady_clock::now();
    GpuNodeSearchResult node_search;
    const std::size_t adjusted_row_begin = options.distributed == nullptr ? row_begin : 0U;
    const std::size_t adjusted_row_end =
        options.distributed == nullptr ? row_end : effective_row_count;
    SearchBestNodeSplitGpu(gpu_workspace,
                           node_allowed_features,
                           options.lambda_l2,
                           options.min_data_in_leaf,
                           options.min_child_weight,
                           options.min_split_gain,
                           options.alpha,
                           depth,
                           adjusted_row_begin,
                           adjusted_row_end,
                           leaf_lower_bound,
                           leaf_upper_bound,
                           options.random_seed,
                           options.random_strength,
                           &node_search);
    node_search.node_statistics = gpu_node_stats;
    const double feature_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - search_start)
            .count();
    const double split_ms = 0.0;
    if (node_search.feature_id < 0 || node_search.p_value > options.alpha) {
      if (profiler != nullptr && profiler->enabled()) {
        profiler->LogNodeSearch(depth,
                                effective_row_count,
                                node_search.feature_id,
                                node_search.p_value,
                                node_search.chi_square,
                                false,
                                node_search.is_categorical,
                                node_search.gain,
                                0,
                                0,
                                feature_ms,
                                split_ms,
                                0.0);
      }
      if (leaf_row_ranges_out != nullptr) {
        (*leaf_row_ranges_out)[node_index] = LeafRowRange{row_begin, row_end};
      }
      return node_index;
    }

    if (!node_search.split_valid || node_search.gain <= 0.0) {
      if (profiler != nullptr && profiler->enabled()) {
        profiler->LogNodeSearch(depth,
                                effective_row_count,
                                node_search.feature_id,
                                node_search.p_value,
                                node_search.chi_square,
                                false,
                                node_search.is_categorical,
                                node_search.gain,
                                0,
                                0,
                                feature_ms,
                                split_ms,
                                0.0);
      }
      if (leaf_row_ranges_out != nullptr) {
        (*leaf_row_ranges_out)[node_index] = LeafRowRange{row_begin, row_end};
      }
      return node_index;
    }
    const auto partition_start = std::chrono::steady_clock::now();
    const std::size_t left_end = PartitionHistogramRowsGpu(gpu_workspace,
                                                           row_begin,
                                                           row_end,
                                                           static_cast<std::size_t>(node_search.feature_id),
                                                           node_search.is_categorical,
                                                           node_search.split_bin,
                                                           node_search.left_categories);
    const double partition_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - partition_start)
            .count();
    const std::size_t left_count = left_end - row_begin;
    const std::size_t right_count = row_end - left_end;
    if (profiler != nullptr && profiler->enabled()) {
      profiler->LogNodeSearch(depth,
                              effective_row_count,
                              node_search.feature_id,
                              node_search.p_value,
                              node_search.chi_square,
                              true,
                              node_search.is_categorical,
                              node_search.gain,
                              left_count,
                              right_count,
                              feature_ms,
                              split_ms,
                              partition_ms);
    }
    if (options.distributed == nullptr && (left_end == row_begin || left_end == row_end)) {
      if (leaf_row_ranges_out != nullptr) {
        (*leaf_row_ranges_out)[node_index] = LeafRowRange{row_begin, row_end};
      }
      return node_index;
    }

    double left_child_lower_bound = leaf_lower_bound;
    double left_child_upper_bound = leaf_upper_bound;
    double right_child_lower_bound = leaf_lower_bound;
    double right_child_upper_bound = leaf_upper_bound;
    const int monotone_sign =
        options.monotone_constraints == nullptr ||
                static_cast<std::size_t>(node_search.feature_id) >= options.monotone_constraints->size()
            ? 0
            : (*options.monotone_constraints)[static_cast<std::size_t>(node_search.feature_id)];
    if (monotone_sign > 0) {
      const double midpoint =
          0.5 * (node_search.left_leaf_weight + node_search.right_leaf_weight);
      left_child_upper_bound = std::min(left_child_upper_bound, midpoint);
      right_child_lower_bound = std::max(right_child_lower_bound, midpoint);
    } else if (monotone_sign < 0) {
      const double midpoint =
          0.5 * (node_search.left_leaf_weight + node_search.right_leaf_weight);
      left_child_lower_bound = std::max(left_child_lower_bound, midpoint);
      right_child_upper_bound = std::min(right_child_upper_bound, midpoint);
    }

    const bool build_left_direct = options.distributed != nullptr || left_count <= right_count;
    const std::size_t direct_begin = build_left_direct ? row_begin : left_end;
    const std::size_t direct_end = build_left_direct ? left_end : row_end;
    const auto direct_child_start = std::chrono::steady_clock::now();
    GpuHistogramSnapshot direct_child_snapshot;
    GpuHistogramSnapshot local_direct_child_snapshot;
    BuildHistogramsGpu(
        gpu_workspace, direct_begin, direct_end, &local_direct_child_snapshot.node_statistics);
    DownloadHistogramSnapshotGpu(gpu_workspace, &local_direct_child_snapshot);
    if (options.distributed != nullptr) {
      direct_child_snapshot =
          AllReduceGpuHistogramSnapshot(options.distributed, local_direct_child_snapshot);
    } else {
      direct_child_snapshot = std::move(local_direct_child_snapshot);
    }
    const double direct_child_histogram_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - direct_child_start)
            .count();
    const auto subtraction_start = std::chrono::steady_clock::now();
    const GpuHistogramSnapshot sibling_child_snapshot =
        SubtractGpuHistogramSnapshot(parent_snapshot, direct_child_snapshot);
    const double sibling_child_histogram_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - subtraction_start)
            .count();

    nodes_[node_index].is_leaf = false;
    nodes_[node_index].split_feature_id = node_search.feature_id;
    nodes_[node_index].split_bin_index = node_search.split_bin;
    nodes_[node_index].is_categorical_split = node_search.is_categorical;
    if (node_search.is_categorical) {
      nodes_[node_index].left_categories = node_search.left_categories;
    }
    feature_importances_[static_cast<std::size_t>(node_search.feature_id)] += node_search.gain;
    if (leaf_count != nullptr) {
      ++(*leaf_count);
    }
    std::vector<int> child_active_groups_storage;
    std::vector<int> child_allowed_features_storage;
    const std::vector<int>* child_active_groups = active_interaction_groups;
    const std::vector<int>* child_allowed_features = node_allowed_features;
    if (options.interaction_constraints != nullptr) {
      const auto& constraints = *options.interaction_constraints;
      const auto& feature_groups =
          constraints.feature_to_groups[static_cast<std::size_t>(node_search.feature_id)];
      if (!feature_groups.empty()) {
        if (active_interaction_groups == nullptr || active_interaction_groups->empty()) {
          child_active_groups_storage = feature_groups;
        } else {
          child_active_groups_storage =
              IntersectSortedVectors(*active_interaction_groups, feature_groups);
        }
        child_active_groups =
            child_active_groups_storage.empty() ? nullptr : &child_active_groups_storage;
        child_allowed_features_storage = FilterAllowedFeaturesForInteraction(
            hist.num_cols, node_allowed_features, constraints, child_active_groups);
        child_allowed_features = &child_allowed_features_storage;
      }
    }

    bool build_left_first = true;
    bool left_snapshot_resident = options.distributed == nullptr && build_left_direct;
    bool right_snapshot_resident = options.distributed == nullptr && !build_left_direct;
    if (options.grow_policy == GrowPolicy::LeafWise && options.max_leaves > 0) {
      GpuNodeSearchResult left_selection;
      GpuNodeSearchResult right_selection;
      if (build_left_direct) {
        UploadHistogramSnapshotGpu(gpu_workspace, direct_child_snapshot);
        SearchBestNodeSplitGpu(gpu_workspace,
                               child_allowed_features,
                               options.lambda_l2,
                               options.min_data_in_leaf,
                               options.min_child_weight,
                               options.min_split_gain,
                               options.alpha,
                               depth + 1,
                               options.distributed == nullptr
                                   ? row_begin
                                   : 0U,
                               options.distributed == nullptr
                                   ? left_end
                                   : static_cast<std::size_t>(
                                         direct_child_snapshot.node_statistics.sample_count),
                               left_child_lower_bound,
                               left_child_upper_bound,
                               options.random_seed,
                               options.random_strength,
                               &left_selection);
        UploadHistogramSnapshotGpu(gpu_workspace, sibling_child_snapshot);
        SearchBestNodeSplitGpu(gpu_workspace,
                               child_allowed_features,
                               options.lambda_l2,
                               options.min_data_in_leaf,
                               options.min_child_weight,
                               options.min_split_gain,
                               options.alpha,
                               depth + 1,
                               options.distributed == nullptr
                                   ? left_end
                                   : 0U,
                               options.distributed == nullptr
                                   ? row_end
                                   : static_cast<std::size_t>(
                                         sibling_child_snapshot.node_statistics.sample_count),
                               right_child_lower_bound,
                               right_child_upper_bound,
                               options.random_seed,
                               options.random_strength,
                               &right_selection);
      } else {
        UploadHistogramSnapshotGpu(gpu_workspace, direct_child_snapshot);
        SearchBestNodeSplitGpu(gpu_workspace,
                               child_allowed_features,
                               options.lambda_l2,
                               options.min_data_in_leaf,
                               options.min_child_weight,
                               options.min_split_gain,
                               options.alpha,
                               depth + 1,
                               options.distributed == nullptr
                                   ? left_end
                                   : 0U,
                               options.distributed == nullptr
                                   ? row_end
                                   : static_cast<std::size_t>(
                                         direct_child_snapshot.node_statistics.sample_count),
                               right_child_lower_bound,
                               right_child_upper_bound,
                               options.random_seed,
                               options.random_strength,
                               &right_selection);
        UploadHistogramSnapshotGpu(gpu_workspace, sibling_child_snapshot);
        SearchBestNodeSplitGpu(gpu_workspace,
                               child_allowed_features,
                               options.lambda_l2,
                               options.min_data_in_leaf,
                               options.min_child_weight,
                               options.min_split_gain,
                               options.alpha,
                               depth + 1,
                               options.distributed == nullptr
                                   ? row_begin
                                   : 0U,
                               options.distributed == nullptr
                                   ? left_end
                                   : static_cast<std::size_t>(
                                         sibling_child_snapshot.node_statistics.sample_count),
                               left_child_lower_bound,
                               left_child_upper_bound,
                               options.random_seed,
                               options.random_strength,
                               &left_selection);
      }
      if (right_selection.adjusted_gain > left_selection.adjusted_gain) {
        build_left_first = false;
      }
      left_snapshot_resident = false;
      right_snapshot_resident = false;
    }
    const GpuHistogramSnapshot* left_snapshot =
        build_left_direct ? &direct_child_snapshot : &sibling_child_snapshot;
    const GpuHistogramSnapshot* right_snapshot =
        build_left_direct ? &sibling_child_snapshot : &direct_child_snapshot;
    const double left_histogram_ms =
        build_left_direct ? direct_child_histogram_ms : sibling_child_histogram_ms;
    const double right_histogram_ms =
        build_left_direct ? sibling_child_histogram_ms : direct_child_histogram_ms;

    if (build_left_first) {
      nodes_[node_index].left_child =
          BuildNode(hist,
                    gradients,
                    hessians,
                    weights,
                    row_indices,
                    row_begin,
                    left_end,
                    depth + 1,
                    options,
                    gpu_workspace,
                    left_snapshot,
                    left_snapshot_resident,
                    nullptr,
                    left_histogram_ms,
                    child_allowed_features,
                    child_active_groups,
                    left_child_lower_bound,
                    left_child_upper_bound,
                    profiler,
                    statistic_engine,
                    leaf_row_ranges_out,
                    leaf_count);
      nodes_[node_index].right_child =
          BuildNode(hist,
                    gradients,
                    hessians,
                    weights,
                    row_indices,
                    left_end,
                    row_end,
                    depth + 1,
                    options,
                    gpu_workspace,
                    right_snapshot,
                    right_snapshot_resident,
                    nullptr,
                    right_histogram_ms,
                    child_allowed_features,
                    child_active_groups,
                    right_child_lower_bound,
                    right_child_upper_bound,
                    profiler,
                    statistic_engine,
                    leaf_row_ranges_out,
                    leaf_count);
    } else {
      nodes_[node_index].right_child =
          BuildNode(hist,
                    gradients,
                    hessians,
                    weights,
                    row_indices,
                    left_end,
                    row_end,
                    depth + 1,
                    options,
                    gpu_workspace,
                    right_snapshot,
                    right_snapshot_resident,
                    nullptr,
                    right_histogram_ms,
                    child_allowed_features,
                    child_active_groups,
                    right_child_lower_bound,
                    right_child_upper_bound,
                    profiler,
                    statistic_engine,
                    leaf_row_ranges_out,
                    leaf_count);
      nodes_[node_index].left_child =
          BuildNode(hist,
                    gradients,
                    hessians,
                    weights,
                    row_indices,
                    row_begin,
                    left_end,
                    depth + 1,
                    options,
                    gpu_workspace,
                    left_snapshot,
                    left_snapshot_resident,
                    nullptr,
                    left_histogram_ms,
                    child_allowed_features,
                    child_active_groups,
                    left_child_lower_bound,
                    left_child_upper_bound,
                    profiler,
                    statistic_engine,
                    leaf_row_ranges_out,
                    leaf_count);
    }
    return node_index;
  }

  NodeHistogramSet node_stats;
  double histogram_ms = precomputed_histogram_ms;
  if (precomputed_node_stats != nullptr) {
    node_stats = *precomputed_node_stats;
  } else {
    const auto histogram_start = std::chrono::steady_clock::now();
    NodeHistogramSet local_node_stats = ComputeNodeHistogramSet(
        hist,
        gradients,
        hessians,
        weights,
        row_indices,
        row_begin,
        row_end,
        options.use_gpu,
        gpu_workspace);
    node_stats = AllReduceNodeHistogramSet(options.distributed, local_node_stats);
    histogram_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - histogram_start)
            .count();
  }
  const std::size_t effective_row_count =
      options.distributed == nullptr ? row_count : static_cast<std::size_t>(node_stats.sample_count);
  if (profiler != nullptr && profiler->enabled()) {
    profiler->LogNodeHistogram(depth, effective_row_count, options.use_gpu, histogram_ms);
  }

  Node node;
  node.leaf_weight = static_cast<float>(ClampLeafWeight(
      ComputeLeafWeight(node_stats.total_gradient, node_stats.total_hessian, options.lambda_l2),
      leaf_lower_bound,
      leaf_upper_bound));

  const int node_index = static_cast<int>(nodes_.size());
  nodes_.push_back(node);
  if (leaf_row_ranges_out != nullptr && leaf_row_ranges_out->size() < nodes_.size()) {
    leaf_row_ranges_out->resize(nodes_.size());
  }

  if (effective_row_count < static_cast<std::size_t>(options.min_samples_split) ||
      depth >= options.max_depth ||
      (options.max_leaves > 0 && leaf_count != nullptr && *leaf_count >= options.max_leaves)) {
    if (leaf_row_ranges_out != nullptr) {
      (*leaf_row_ranges_out)[node_index] = LeafRowRange{row_begin, row_end};
    }
    return node_index;
  }

  const auto feature_start = std::chrono::steady_clock::now();
  const CandidateSelectionResult selection = SelectBestCandidateSplit(hist,
                                                                      node_stats,
                                                                      options,
                                                                      statistic_engine,
                                                                      node_allowed_features,
                                                                      leaf_lower_bound,
                                                                      leaf_upper_bound,
                                                                      depth,
                                                                      row_begin,
                                                                      row_end);
  const FeatureChoice& feature_choice = selection.feature_choice;
  const double feature_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - feature_start)
          .count();
  if (feature_choice.feature_id < 0 || feature_choice.p_value > options.alpha) {
    if (profiler != nullptr && profiler->enabled()) {
      profiler->LogNodeSearch(depth,
                              effective_row_count,
                              feature_choice.feature_id,
                              feature_choice.p_value,
                              feature_choice.chi_square,
                              false,
                              false,
                              0.0,
                              0,
                              0,
                              feature_ms,
                              0.0,
                              0.0);
    }
    if (leaf_row_ranges_out != nullptr) {
      (*leaf_row_ranges_out)[node_index] = LeafRowRange{row_begin, row_end};
    }
    return node_index;
  }

  const auto split_start = std::chrono::steady_clock::now();
  const SplitChoice& split_choice = selection.split_choice;
  const double split_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - split_start)
          .count();
  if (!split_choice.valid || split_choice.gain <= 0.0) {
    if (profiler != nullptr && profiler->enabled()) {
      profiler->LogNodeSearch(depth,
                              effective_row_count,
                              feature_choice.feature_id,
                              feature_choice.p_value,
                              feature_choice.chi_square,
                              false,
                              split_choice.is_categorical,
                              split_choice.gain,
                              0,
                              0,
                              feature_ms,
                              split_ms,
                              0.0);
    }
    if (leaf_row_ranges_out != nullptr) {
      (*leaf_row_ranges_out)[node_index] = LeafRowRange{row_begin, row_end};
    }
    return node_index;
  }

  std::size_t left_end = row_begin;
  double partition_ms = 0.0;
  if (options.use_gpu) {
    const auto partition_start = std::chrono::steady_clock::now();
    left_end = PartitionHistogramRowsGpu(gpu_workspace,
                                         row_begin,
                                         row_end,
                                         static_cast<std::size_t>(feature_choice.feature_id),
                                         split_choice.is_categorical,
                                         split_choice.split_bin,
                                         split_choice.left_categories);
    partition_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - partition_start)
            .count();
  } else {
    const auto feature_bins = hist.feature_bins(static_cast<std::size_t>(feature_choice.feature_id));
    const auto left_begin = row_indices.begin() + static_cast<std::ptrdiff_t>(row_begin);
    const auto right_end = row_indices.begin() + static_cast<std::ptrdiff_t>(row_end);
    const auto partition_start = std::chrono::steady_clock::now();
    const auto split_middle = std::partition(left_begin, right_end, [&](std::size_t row) {
      const std::uint16_t bin = feature_bins[row];
      return split_choice.is_categorical ? split_choice.left_categories[bin] != 0
                                         : bin <= split_choice.split_bin;
    });
    partition_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - partition_start)
            .count();
    left_end = row_begin + static_cast<std::size_t>(std::distance(left_begin, split_middle));
  }
  const std::size_t left_count = left_end - row_begin;
  const std::size_t right_count = row_end - left_end;
  if (profiler != nullptr && profiler->enabled()) {
    profiler->LogNodeSearch(depth,
                            effective_row_count,
                            feature_choice.feature_id,
                            feature_choice.p_value,
                            feature_choice.chi_square,
                            true,
                            split_choice.is_categorical,
                            split_choice.gain,
                            left_count,
                            right_count,
                            feature_ms,
                            split_ms,
                            partition_ms);
  }
  if (options.distributed == nullptr && (left_end == row_begin || left_end == row_end)) {
    if (leaf_row_ranges_out != nullptr) {
      (*leaf_row_ranges_out)[node_index] = LeafRowRange{row_begin, row_end};
    }
    return node_index;
  }

  double left_child_lower_bound = leaf_lower_bound;
  double left_child_upper_bound = leaf_upper_bound;
  double right_child_lower_bound = leaf_lower_bound;
  double right_child_upper_bound = leaf_upper_bound;
  const int monotone_sign =
      options.monotone_constraints == nullptr ||
              static_cast<std::size_t>(feature_choice.feature_id) >= options.monotone_constraints->size()
          ? 0
          : (*options.monotone_constraints)[static_cast<std::size_t>(feature_choice.feature_id)];
  if (monotone_sign > 0) {
    const double midpoint = 0.5 * (split_choice.left_leaf_weight + split_choice.right_leaf_weight);
    left_child_upper_bound = std::min(left_child_upper_bound, midpoint);
    right_child_lower_bound = std::max(right_child_lower_bound, midpoint);
  } else if (monotone_sign < 0) {
    const double midpoint = 0.5 * (split_choice.left_leaf_weight + split_choice.right_leaf_weight);
    left_child_lower_bound = std::max(left_child_lower_bound, midpoint);
    right_child_upper_bound = std::min(right_child_upper_bound, midpoint);
  }

  NodeHistogramSet left_child_stats;
  NodeHistogramSet right_child_stats;
  double left_child_histogram_ms = 0.0;
  double right_child_histogram_ms = 0.0;
  const bool build_left_direct = options.distributed != nullptr || left_count <= right_count;
  if (build_left_direct) {
    const auto direct_child_start = std::chrono::steady_clock::now();
    NodeHistogramSet local_left_child_stats = ComputeNodeHistogramSet(
        hist,
        gradients,
        hessians,
        weights,
        row_indices,
        row_begin,
        left_end,
        options.use_gpu,
        gpu_workspace);
    left_child_stats = AllReduceNodeHistogramSet(options.distributed, local_left_child_stats);
    left_child_histogram_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - direct_child_start)
            .count();
    const auto subtraction_start = std::chrono::steady_clock::now();
    right_child_stats = SubtractNodeHistogramSet(node_stats, left_child_stats);
    right_child_histogram_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - subtraction_start)
            .count();
  } else {
    const auto direct_child_start = std::chrono::steady_clock::now();
    NodeHistogramSet local_right_child_stats = ComputeNodeHistogramSet(
        hist,
        gradients,
        hessians,
        weights,
        row_indices,
        left_end,
        row_end,
        options.use_gpu,
        gpu_workspace);
    right_child_stats = AllReduceNodeHistogramSet(options.distributed, local_right_child_stats);
    right_child_histogram_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - direct_child_start)
            .count();
    const auto subtraction_start = std::chrono::steady_clock::now();
    left_child_stats = SubtractNodeHistogramSet(node_stats, right_child_stats);
    left_child_histogram_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - subtraction_start)
            .count();
  }

  nodes_[node_index].is_leaf = false;
  nodes_[node_index].split_feature_id = feature_choice.feature_id;
  nodes_[node_index].split_bin_index = split_choice.split_bin;
  nodes_[node_index].is_categorical_split = split_choice.is_categorical;
  if (split_choice.is_categorical) {
    nodes_[node_index].left_categories = split_choice.left_categories;
  }
  feature_importances_[static_cast<std::size_t>(feature_choice.feature_id)] += split_choice.gain;
  if (leaf_count != nullptr) {
    ++(*leaf_count);
  }
  std::vector<int> child_active_groups_storage;
  std::vector<int> child_allowed_features_storage;
  const std::vector<int>* child_active_groups = active_interaction_groups;
  const std::vector<int>* child_allowed_features = node_allowed_features;
  if (options.interaction_constraints != nullptr) {
    const auto& constraints = *options.interaction_constraints;
    const auto& feature_groups =
        constraints.feature_to_groups[static_cast<std::size_t>(feature_choice.feature_id)];
    if (!feature_groups.empty()) {
      if (active_interaction_groups == nullptr || active_interaction_groups->empty()) {
        child_active_groups_storage = feature_groups;
      } else {
        child_active_groups_storage =
            IntersectSortedVectors(*active_interaction_groups, feature_groups);
      }
      child_active_groups = child_active_groups_storage.empty() ? nullptr : &child_active_groups_storage;
      child_allowed_features_storage = FilterAllowedFeaturesForInteraction(
          hist.num_cols, node_allowed_features, constraints, child_active_groups);
      child_allowed_features = &child_allowed_features_storage;
    }
  }
  bool build_left_first = true;
  if (options.grow_policy == GrowPolicy::LeafWise && options.max_leaves > 0) {
    const CandidateSelectionResult left_selection = SelectBestCandidateSplit(hist,
                                                                             left_child_stats,
                                                                             options,
                                                                             statistic_engine,
                                                                             child_allowed_features,
                                                                             left_child_lower_bound,
                                                                             left_child_upper_bound,
                                                                             depth + 1,
                                                                             row_begin,
                                                                             left_end);
    const CandidateSelectionResult right_selection = SelectBestCandidateSplit(hist,
                                                                              right_child_stats,
                                                                              options,
                                                                              statistic_engine,
                                                                              child_allowed_features,
                                                                              right_child_lower_bound,
                                                                              right_child_upper_bound,
                                                                              depth + 1,
                                                                              left_end,
                                                                              row_end);
    if (right_selection.adjusted_gain > left_selection.adjusted_gain) {
      build_left_first = false;
    }
  }

  if (build_left_first) {
    nodes_[node_index].left_child =
        BuildNode(hist,
                  gradients,
                  hessians,
                  weights,
                  row_indices,
                  row_begin,
                  left_end,
                  depth + 1,
                  options,
                  gpu_workspace,
                  nullptr,
                  false,
                  &left_child_stats,
                  left_child_histogram_ms,
                  child_allowed_features,
                  child_active_groups,
                  left_child_lower_bound,
                  left_child_upper_bound,
                  profiler,
                  statistic_engine,
                  leaf_row_ranges_out,
                  leaf_count);
    nodes_[node_index].right_child =
        BuildNode(hist,
                  gradients,
                  hessians,
                  weights,
                  row_indices,
                  left_end,
                  row_end,
                  depth + 1,
                  options,
                  gpu_workspace,
                  nullptr,
                  false,
                  &right_child_stats,
                  right_child_histogram_ms,
                  child_allowed_features,
                  child_active_groups,
                  right_child_lower_bound,
                  right_child_upper_bound,
                  profiler,
                  statistic_engine,
                  leaf_row_ranges_out,
                  leaf_count);
  } else {
    nodes_[node_index].right_child =
        BuildNode(hist,
                  gradients,
                  hessians,
                  weights,
                  row_indices,
                  left_end,
                  row_end,
                  depth + 1,
                  options,
                  gpu_workspace,
                  nullptr,
                  false,
                  &right_child_stats,
                  right_child_histogram_ms,
                  child_allowed_features,
                  child_active_groups,
                  right_child_lower_bound,
                  right_child_upper_bound,
                  profiler,
                  statistic_engine,
                  leaf_row_ranges_out,
                  leaf_count);
    nodes_[node_index].left_child =
        BuildNode(hist,
                  gradients,
                  hessians,
                  weights,
                  row_indices,
                  row_begin,
                  left_end,
                  depth + 1,
                  options,
                  gpu_workspace,
                  nullptr,
                  false,
                  &left_child_stats,
                  left_child_histogram_ms,
                  child_allowed_features,
                  child_active_groups,
                  left_child_lower_bound,
                  left_child_upper_bound,
                  profiler,
                  statistic_engine,
                  leaf_row_ranges_out,
                  leaf_count);
  }

  return node_index;
}

float Tree::PredictRow(const Pool& pool, std::size_t row) const {
  const int leaf_index = PredictLeafIndex(pool, row);
  return leaf_index < 0 ? 0.0F : nodes_[leaf_index].leaf_weight;
}

int Tree::PredictLeafIndex(const Pool& pool, std::size_t row) const {
  if (nodes_.empty()) {
    return -1;
  }

  int node_index = 0;
  while (!nodes_[node_index].is_leaf) {
    const Node& node = nodes_[node_index];
    const std::uint16_t bin = BinValue(
        static_cast<std::size_t>(node.split_feature_id),
        pool.feature_value(row, static_cast<std::size_t>(node.split_feature_id)));
    node_index = node.is_categorical_split
                     ? (node.left_categories[bin] != 0 ? node.left_child : node.right_child)
                     : (bin <= node.split_bin_index ? node.left_child : node.right_child);
  }

  return node_index;
}

void Tree::AccumulateContributions(
    const Pool& pool, std::size_t row, float scale, std::vector<float>& row_contributions) const {
  if (row_contributions.empty()) {
    return;
  }
  if (nodes_.empty()) {
    row_contributions.back() += scale * 0.0F;
    return;
  }

  std::vector<int> path_features;
  int node_index = 0;
  while (!nodes_[node_index].is_leaf) {
    const Node& node = nodes_[node_index];
    path_features.push_back(node.split_feature_id);
    const std::uint16_t bin = BinValue(
        static_cast<std::size_t>(node.split_feature_id),
        pool.feature_value(row, static_cast<std::size_t>(node.split_feature_id)));
    node_index = node.is_categorical_split
                     ? (node.left_categories[bin] != 0 ? node.left_child : node.right_child)
                     : (bin <= node.split_bin_index ? node.left_child : node.right_child);
  }

  const float leaf_value = scale * nodes_[node_index].leaf_weight;
  if (path_features.empty()) {
    row_contributions.back() += leaf_value;
    return;
  }

  const float share = leaf_value / static_cast<float>(path_features.size());
  for (const int feature_index : path_features) {
    row_contributions[static_cast<std::size_t>(feature_index)] += share;
  }
}

float Tree::PredictBinnedRow(const HistMatrix& hist, std::size_t row) const {
  const int leaf_index = PredictBinnedLeafIndex(hist, row);
  return leaf_index < 0 ? 0.0F : nodes_[leaf_index].leaf_weight;
}

int Tree::PredictBinnedLeafIndex(const HistMatrix& hist, std::size_t row) const {
  if (nodes_.empty()) {
    return -1;
  }

  int node_index = 0;
  while (!nodes_[node_index].is_leaf) {
    const Node& node = nodes_[node_index];
    const auto feature_bins = hist.feature_bins(static_cast<std::size_t>(node.split_feature_id));
    const std::uint16_t bin = feature_bins[row];
    node_index = node.is_categorical_split
                     ? (node.left_categories[bin] != 0 ? node.left_child : node.right_child)
                     : (bin <= node.split_bin_index ? node.left_child : node.right_child);
  }

  return node_index;
}

std::vector<float> Tree::Predict(const Pool& pool) const {
  std::vector<float> predictions(pool.num_rows(), 0.0F);
  for (std::size_t row = 0; row < pool.num_rows(); ++row) {
    predictions[row] = PredictRow(pool, row);
  }
  return predictions;
}

void Tree::SetLeafWeight(std::size_t node_index, float leaf_weight) {
  if (node_index >= nodes_.size()) {
    throw std::out_of_range("node index is out of bounds");
  }
  if (!nodes_[node_index].is_leaf) {
    throw std::invalid_argument("leaf weight can only be set on leaf nodes");
  }
  nodes_[node_index].leaf_weight = leaf_weight;
}

void Tree::SetQuantizationSchema(const QuantizationSchemaPtr& quantization_schema) {
  quantization_schema_ = quantization_schema;
}

const QuantizationSchemaPtr& Tree::shared_quantization_schema() const noexcept {
  return quantization_schema_;
}

void Tree::LoadState(std::vector<Node> nodes,
                     const QuantizationSchemaPtr& quantization_schema,
                     std::vector<double> feature_importances) {
  nodes_ = std::move(nodes);
  quantization_schema_ = quantization_schema;
  feature_importances_ = std::move(feature_importances);
}

void Tree::LoadState(std::vector<Node> nodes,
                     std::vector<std::uint16_t> num_bins_per_feature,
                     std::vector<std::size_t> cut_offsets,
                     std::vector<float> cut_values,
                     std::vector<std::uint8_t> categorical_mask,
                     std::vector<std::uint8_t> missing_value_mask,
                     std::uint8_t nan_mode,
                     std::vector<double> feature_importances) {
  auto quantization_schema = std::make_shared<QuantizationSchema>();
  quantization_schema->num_bins_per_feature = std::move(num_bins_per_feature);
  quantization_schema->cut_offsets = std::move(cut_offsets);
  quantization_schema->cut_values = std::move(cut_values);
  quantization_schema->categorical_mask = std::move(categorical_mask);
  quantization_schema->missing_value_mask = std::move(missing_value_mask);
  quantization_schema->nan_mode = nan_mode;
  LoadState(std::move(nodes), quantization_schema, std::move(feature_importances));
}

const std::vector<Node>& Tree::nodes() const noexcept { return nodes_; }

const std::vector<std::uint16_t>& Tree::num_bins_per_feature() const {
  return RequireQuantizationSchema(quantization_schema_).num_bins_per_feature;
}

const std::vector<std::size_t>& Tree::cut_offsets() const {
  return RequireQuantizationSchema(quantization_schema_).cut_offsets;
}

const std::vector<float>& Tree::cut_values() const {
  return RequireQuantizationSchema(quantization_schema_).cut_values;
}

const std::vector<std::uint8_t>& Tree::categorical_mask() const {
  return RequireQuantizationSchema(quantization_schema_).categorical_mask;
}

const std::vector<std::uint8_t>& Tree::missing_value_mask() const {
  return RequireQuantizationSchema(quantization_schema_).missing_value_mask;
}

std::uint8_t Tree::nan_mode() const {
  return RequireQuantizationSchema(quantization_schema_).nan_mode;
}

const std::vector<double>& Tree::feature_importances() const noexcept {
  return feature_importances_;
}

std::uint16_t Tree::BinValue(std::size_t feature_index, float value) const {
  return RequireQuantizationSchema(quantization_schema_).bin_value(feature_index, value);
}

}  // namespace ctboost
