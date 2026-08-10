#include "booster_internal.hpp"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace ctboost::booster_detail {
namespace {

template <typename T>
void AppendBinary(std::vector<std::uint8_t>& buffer, const T& value) {
  const auto* bytes = reinterpret_cast<const std::uint8_t*>(&value);
  buffer.insert(buffer.end(), bytes, bytes + sizeof(T));
}

template <typename T>
T ReadBinary(const std::vector<std::uint8_t>& buffer, std::size_t& offset) {
  if (offset + sizeof(T) > buffer.size()) {
    throw std::runtime_error("distributed metric payload is truncated");
  }
  T value{};
  std::memcpy(&value, buffer.data() + offset, sizeof(T));
  offset += sizeof(T);
  return value;
}

std::vector<std::uint8_t> SerializeDistributedMetricInputs(
    const DistributedMetricInputs& inputs) {
  if (inputs.labels.size() != inputs.weights.size()) {
    throw std::invalid_argument("distributed metric labels and weights must have matching sizes");
  }
  if (inputs.labels.empty() ? !inputs.predictions.empty()
                            : inputs.predictions.size() % inputs.labels.size() != 0U) {
    throw std::invalid_argument(
        "distributed metric prediction size must be a multiple of the label count");
  }
  if (inputs.has_group_ids && inputs.group_ids.size() != inputs.labels.size()) {
    throw std::invalid_argument(
        "distributed metric group_ids must match the label count when provided");
  }
  std::vector<std::uint8_t> buffer;
  const std::uint64_t num_rows = static_cast<std::uint64_t>(inputs.labels.size());
  const std::uint64_t prediction_size = static_cast<std::uint64_t>(inputs.predictions.size());
  const std::uint8_t has_group_ids = inputs.has_group_ids ? 1U : 0U;
  AppendBinary(buffer, num_rows);
  AppendBinary(buffer, prediction_size);
  AppendBinary(buffer, has_group_ids);
  if (!inputs.predictions.empty()) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(inputs.predictions.data());
    buffer.insert(buffer.end(), bytes, bytes + inputs.predictions.size() * sizeof(float));
  }
  if (!inputs.labels.empty()) {
    const auto* label_bytes = reinterpret_cast<const std::uint8_t*>(inputs.labels.data());
    buffer.insert(buffer.end(), label_bytes, label_bytes + inputs.labels.size() * sizeof(float));
    const auto* weight_bytes = reinterpret_cast<const std::uint8_t*>(inputs.weights.data());
    buffer.insert(buffer.end(), weight_bytes, weight_bytes + inputs.weights.size() * sizeof(float));
  }
  if (has_group_ids != 0U && !inputs.group_ids.empty()) {
    const auto* group_bytes = reinterpret_cast<const std::uint8_t*>(inputs.group_ids.data());
    buffer.insert(buffer.end(),
                  group_bytes,
                  group_bytes + inputs.group_ids.size() * sizeof(std::int64_t));
  }
  return buffer;
}

DistributedMetricInputs DeserializeDistributedMetricInputs(
    const std::vector<std::uint8_t>& buffer) {
  std::size_t offset = 0;
  const std::uint64_t num_rows = ReadBinary<std::uint64_t>(buffer, offset);
  const std::uint64_t prediction_size = ReadBinary<std::uint64_t>(buffer, offset);
  const std::uint8_t has_group_ids = ReadBinary<std::uint8_t>(buffer, offset);
  DistributedMetricInputs inputs;
  inputs.has_group_ids = has_group_ids != 0U;
  inputs.predictions.resize(static_cast<std::size_t>(prediction_size));
  inputs.labels.resize(static_cast<std::size_t>(num_rows));
  inputs.weights.resize(static_cast<std::size_t>(num_rows));
  const auto read_float_array = [&](std::vector<float>& values) {
    const std::size_t byte_count = values.size() * sizeof(float);
    if (offset + byte_count > buffer.size()) {
      throw std::runtime_error("distributed metric payload is truncated");
    }
    if (byte_count != 0U) {
      std::memcpy(values.data(), buffer.data() + offset, byte_count);
    }
    offset += byte_count;
  };
  read_float_array(inputs.predictions);
  read_float_array(inputs.labels);
  read_float_array(inputs.weights);
  if (inputs.has_group_ids) {
    inputs.group_ids.resize(static_cast<std::size_t>(num_rows));
    const std::size_t byte_count = inputs.group_ids.size() * sizeof(std::int64_t);
    if (offset + byte_count > buffer.size()) {
      throw std::runtime_error("distributed metric group_id payload is truncated");
    }
    if (byte_count != 0U) {
      std::memcpy(inputs.group_ids.data(), buffer.data() + offset, byte_count);
    }
    offset += byte_count;
  }
  return inputs;
}

std::vector<std::vector<std::uint8_t>> DeserializeGatheredPayloads(
    const std::vector<std::uint8_t>& buffer) {
  std::size_t offset = 0;
  const std::uint64_t payload_count = ReadBinary<std::uint64_t>(buffer, offset);
  std::vector<std::vector<std::uint8_t>> payloads;
  payloads.reserve(static_cast<std::size_t>(payload_count));
  for (std::size_t index = 0; index < static_cast<std::size_t>(payload_count); ++index) {
    const std::uint64_t payload_size = ReadBinary<std::uint64_t>(buffer, offset);
    if (offset + payload_size > buffer.size()) {
      throw std::runtime_error("distributed allgather payload is truncated");
    }
    payloads.emplace_back(buffer.begin() + static_cast<std::ptrdiff_t>(offset),
                          buffer.begin() + static_cast<std::ptrdiff_t>(offset + payload_size));
    offset += static_cast<std::size_t>(payload_size);
  }
  return payloads;
}

DistributedMetricInputs MergeDistributedMetricInputs(
    const std::vector<std::vector<std::uint8_t>>& payloads) {
  DistributedMetricInputs merged;
  bool initialized = false;
  for (const auto& payload : payloads) {
    DistributedMetricInputs shard = DeserializeDistributedMetricInputs(payload);
    if (!initialized) {
      merged.has_group_ids = shard.has_group_ids;
      initialized = true;
    } else if (merged.has_group_ids != shard.has_group_ids) {
      throw std::invalid_argument(
          "distributed metric shards must either all include group_ids or all omit them");
    }
    merged.predictions.insert(
        merged.predictions.end(), shard.predictions.begin(), shard.predictions.end());
    merged.labels.insert(merged.labels.end(), shard.labels.begin(), shard.labels.end());
    merged.weights.insert(merged.weights.end(), shard.weights.begin(), shard.weights.end());
    if (merged.has_group_ids) {
      merged.group_ids.insert(merged.group_ids.end(), shard.group_ids.begin(), shard.group_ids.end());
    }
  }
  return merged;
}

std::filesystem::path DistributedMetricOperationDir(
    const DistributedCoordinator& coordinator,
    const char* label) {
  return std::filesystem::path(coordinator.root) / coordinator.run_id /
         ("tree_" + std::to_string(coordinator.tree_index)) /
         (std::string("booster_") + label);
}

std::string DistributedMetricRankName(int rank) {
  std::ostringstream stream;
  stream << "rank_" << std::setw(5) << std::setfill('0') << rank;
  return stream.str();
}

std::string FreshDistributedNonce() {
  std::random_device random;
  std::ostringstream stream;
  stream << std::hex << std::setfill('0');
  for (int index = 0; index < 4; ++index) {
    stream << std::setw(8) << random();
  }
  return stream.str();
}

void WriteDistributedMetricPayload(const std::filesystem::path& path,
                                   const std::vector<std::uint8_t>& payload) {
  std::error_code error;
  std::filesystem::create_directories(path.parent_path(), error);
  if (error) {
    throw std::runtime_error("failed to create distributed metric directory: " +
                             error.message());
  }
  const std::filesystem::path temp_path = path.string() + ".tmp";
  std::ofstream output(temp_path, std::ios::binary | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("failed to open distributed metric payload for writing");
  }
  if (!payload.empty()) {
    output.write(reinterpret_cast<const char*>(payload.data()),
                 static_cast<std::streamsize>(payload.size()));
  }
  if (!output) {
    throw std::runtime_error("failed to write distributed metric payload");
  }
  output.close();
  std::filesystem::remove(path, error);
  error.clear();
  std::filesystem::rename(temp_path, path, error);
  if (error) {
    throw std::runtime_error("failed to publish distributed metric payload: " +
                             error.message());
  }
}

bool TryReadDistributedMetricPayload(const std::filesystem::path& path,
                                     std::vector<std::uint8_t>& payload) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return false;
  }
  payload.assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
  return true;
}

std::filesystem::path FindDistributedMetricRequest(
    const std::filesystem::path& directory,
    const std::string& prefix) {
  std::error_code error;
  std::filesystem::path selected;
  std::filesystem::file_time_type selected_time{};
  for (const auto& entry : std::filesystem::directory_iterator(directory, error)) {
    if (error || !entry.is_regular_file()) {
      continue;
    }
    const std::string filename = entry.path().filename().string();
    if (filename.rfind(prefix, 0) != 0U || filename.size() < 4U ||
        filename.substr(filename.size() - 4U) != ".bin") {
      continue;
    }
    const auto write_time = entry.last_write_time(error);
    if (!error && (selected.empty() || write_time > selected_time)) {
      selected = entry.path();
      selected_time = write_time;
    }
    error.clear();
  }
  return selected;
}

DistributedMetricInputs AllGatherFilesystemMetricInputs(
    const DistributedCoordinator& coordinator,
    const char* label,
    const DistributedMetricInputs& local_inputs) {
  const std::filesystem::path directory =
      DistributedMetricOperationDir(coordinator, label);
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::duration<double>(coordinator.timeout_seconds);
  const std::string local_nonce = FreshDistributedNonce();
  const std::filesystem::path challenge_path = directory / "challenge.bin";
  if (coordinator.rank == 0) {
    const std::string root_nonce = FreshDistributedNonce();
    WriteDistributedMetricPayload(
        challenge_path, std::vector<std::uint8_t>(root_nonce.begin(), root_nonce.end()));
    std::vector<std::vector<std::uint8_t>> payloads(
        static_cast<std::size_t>(coordinator.world_size));
    std::vector<std::filesystem::path> request_paths(
        static_cast<std::size_t>(coordinator.world_size));
    payloads[0] = SerializeDistributedMetricInputs(local_inputs);
    for (int rank = 1; rank < coordinator.world_size; ++rank) {
      const std::string prefix =
          "request_" + DistributedMetricRankName(rank) + "_" + root_nonce + "_";
      while (request_paths[static_cast<std::size_t>(rank)].empty()) {
        request_paths[static_cast<std::size_t>(rank)] =
            FindDistributedMetricRequest(directory, prefix);
        if (!request_paths[static_cast<std::size_t>(rank)].empty()) {
          break;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
          throw std::runtime_error("timed out waiting for distributed metric request");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
      if (!TryReadDistributedMetricPayload(
              request_paths[static_cast<std::size_t>(rank)],
              payloads[static_cast<std::size_t>(rank)])) {
        throw std::runtime_error("failed to read distributed metric request");
      }
    }
    DistributedMetricInputs merged = MergeDistributedMetricInputs(payloads);
    const std::vector<std::uint8_t> merged_payload =
        SerializeDistributedMetricInputs(merged);
    for (int rank = 1; rank < coordinator.world_size; ++rank) {
      const std::string request_name =
          request_paths[static_cast<std::size_t>(rank)].filename().string();
      WriteDistributedMetricPayload(
          directory / ("result_" + request_name.substr(std::string("request_").size())),
          merged_payload);
    }
    return merged;
  }
  std::string active_root_nonce;
  std::filesystem::path result_path;
  while (std::chrono::steady_clock::now() < deadline) {
    std::vector<std::uint8_t> challenge;
    if (TryReadDistributedMetricPayload(challenge_path, challenge)) {
      const std::string root_nonce(challenge.begin(), challenge.end());
      if (!root_nonce.empty() && root_nonce != active_root_nonce) {
        active_root_nonce = root_nonce;
        const std::string suffix = DistributedMetricRankName(coordinator.rank) + "_" +
                                   active_root_nonce + "_" + local_nonce + ".bin";
        WriteDistributedMetricPayload(
            directory / ("request_" + suffix),
            SerializeDistributedMetricInputs(local_inputs));
        result_path = directory / ("result_" + suffix);
      }
    }
    std::vector<std::uint8_t> result;
    if (!result_path.empty() && TryReadDistributedMetricPayload(result_path, result)) {
      return DeserializeDistributedMetricInputs(result);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  throw std::runtime_error("timed out waiting for distributed metric result");
}

std::vector<std::uint8_t> SerializeLeafStatistics(
    const LeafStatistics& statistics) {
  if (statistics.gradient_sums.size() != statistics.hessian_sums.size()) {
    throw std::invalid_argument(
        "leaf gradient and hessian statistic buffers must have matching sizes");
  }
  std::vector<std::uint8_t> buffer;
  const std::uint64_t statistic_count =
      static_cast<std::uint64_t>(statistics.gradient_sums.size());
  buffer.reserve(sizeof(statistic_count) +
                 statistics.gradient_sums.size() * 2U * sizeof(double));
  AppendBinary(buffer, statistic_count);
  for (const double value : statistics.gradient_sums) {
    AppendBinary(buffer, value);
  }
  for (const double value : statistics.hessian_sums) {
    AppendBinary(buffer, value);
  }
  return buffer;
}

LeafStatistics DeserializeLeafStatistics(
    const std::vector<std::uint8_t>& buffer) {
  std::size_t offset = 0;
  const std::uint64_t statistic_count = ReadBinary<std::uint64_t>(buffer, offset);
  if (statistic_count >
      static_cast<std::uint64_t>((buffer.size() - offset) / (2U * sizeof(double)))) {
    throw std::runtime_error("distributed leaf statistic payload is truncated");
  }
  LeafStatistics statistics;
  statistics.gradient_sums.resize(static_cast<std::size_t>(statistic_count));
  statistics.hessian_sums.resize(static_cast<std::size_t>(statistic_count));
  for (double& value : statistics.gradient_sums) {
    value = ReadBinary<double>(buffer, offset);
  }
  for (double& value : statistics.hessian_sums) {
    value = ReadBinary<double>(buffer, offset);
  }
  if (offset != buffer.size()) {
    throw std::runtime_error("distributed leaf statistic payload has trailing bytes");
  }
  return statistics;
}

LeafStatistics ReduceLeafStatisticPayloads(
    const std::vector<std::vector<std::uint8_t>>& payloads) {
  LeafStatistics reduced;
  bool initialized = false;
  for (const auto& payload : payloads) {
    const LeafStatistics shard = DeserializeLeafStatistics(payload);
    if (!initialized) {
      reduced.gradient_sums.assign(shard.gradient_sums.size(), 0.0);
      reduced.hessian_sums.assign(shard.hessian_sums.size(), 0.0);
      initialized = true;
    } else if (shard.gradient_sums.size() != reduced.gradient_sums.size()) {
      throw std::runtime_error(
          "distributed leaf statistic shards must have matching sizes");
    }
    for (std::size_t index = 0; index < shard.gradient_sums.size(); ++index) {
      reduced.gradient_sums[index] += shard.gradient_sums[index];
      reduced.hessian_sums[index] += shard.hessian_sums[index];
    }
  }
  return reduced;
}

LeafStatistics AllReduceFilesystemLeafStatistics(
    const DistributedCoordinator& coordinator,
    const char* label,
    const LeafStatistics& local_statistics) {
  const std::filesystem::path directory =
      DistributedMetricOperationDir(coordinator, label);
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::duration<double>(coordinator.timeout_seconds);
  const std::string local_nonce = FreshDistributedNonce();
  const std::filesystem::path challenge_path = directory / "challenge.bin";
  if (coordinator.rank == 0) {
    const std::string root_nonce = FreshDistributedNonce();
    WriteDistributedMetricPayload(
        challenge_path,
        std::vector<std::uint8_t>(root_nonce.begin(), root_nonce.end()));
    std::vector<std::vector<std::uint8_t>> payloads(
        static_cast<std::size_t>(coordinator.world_size));
    std::vector<std::filesystem::path> request_paths(
        static_cast<std::size_t>(coordinator.world_size));
    payloads[0] = SerializeLeafStatistics(local_statistics);
    for (int rank = 1; rank < coordinator.world_size; ++rank) {
      const std::string prefix =
          "request_" + DistributedMetricRankName(rank) + "_" + root_nonce + "_";
      while (request_paths[static_cast<std::size_t>(rank)].empty()) {
        request_paths[static_cast<std::size_t>(rank)] =
            FindDistributedMetricRequest(directory, prefix);
        if (!request_paths[static_cast<std::size_t>(rank)].empty()) {
          break;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
          throw std::runtime_error(
              "timed out waiting for distributed leaf statistic request");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
      if (!TryReadDistributedMetricPayload(
              request_paths[static_cast<std::size_t>(rank)],
              payloads[static_cast<std::size_t>(rank)])) {
        throw std::runtime_error(
            "failed to read distributed leaf statistic request");
      }
    }
    const LeafStatistics reduced = ReduceLeafStatisticPayloads(payloads);
    const std::vector<std::uint8_t> reduced_payload =
        SerializeLeafStatistics(reduced);
    for (int rank = 1; rank < coordinator.world_size; ++rank) {
      const std::string request_name =
          request_paths[static_cast<std::size_t>(rank)].filename().string();
      WriteDistributedMetricPayload(
          directory /
              ("result_" + request_name.substr(std::string("request_").size())),
          reduced_payload);
    }
    return reduced;
  }

  std::string active_root_nonce;
  std::filesystem::path result_path;
  while (std::chrono::steady_clock::now() < deadline) {
    std::vector<std::uint8_t> challenge;
    if (TryReadDistributedMetricPayload(challenge_path, challenge)) {
      const std::string root_nonce(challenge.begin(), challenge.end());
      if (!root_nonce.empty() && root_nonce != active_root_nonce) {
        active_root_nonce = root_nonce;
        const std::string suffix = DistributedMetricRankName(coordinator.rank) + "_" +
                                   active_root_nonce + "_" + local_nonce + ".bin";
        WriteDistributedMetricPayload(
            directory / ("request_" + suffix),
            SerializeLeafStatistics(local_statistics));
        result_path = directory / ("result_" + suffix);
      }
    }
    std::vector<std::uint8_t> result;
    if (!result_path.empty() &&
        TryReadDistributedMetricPayload(result_path, result)) {
      return DeserializeLeafStatistics(result);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  throw std::runtime_error(
      "timed out waiting for distributed leaf statistic result");
}

}  // namespace

std::vector<std::uint8_t> SerializeDistributedMetricControl(
    const DistributedMetricControl& control) {
  std::vector<std::uint8_t> buffer(sizeof(DistributedMetricControl), 0U);
  std::memcpy(buffer.data(), &control, sizeof(DistributedMetricControl));
  return buffer;
}

DistributedMetricControl DeserializeDistributedMetricControl(
    const std::vector<std::uint8_t>& buffer) {
  if (buffer.size() != sizeof(DistributedMetricControl)) {
    throw std::runtime_error("distributed metric control payload size mismatch");
  }
  DistributedMetricControl control{};
  std::memcpy(&control, buffer.data(), sizeof(DistributedMetricControl));
  return control;
}

DistributedMetricControl BroadcastDistributedMetricControl(
    const DistributedCoordinator* coordinator,
    const char* label,
    const DistributedMetricControl* root_control) {
  if (coordinator == nullptr || coordinator->world_size <= 1) {
    return root_control == nullptr ? DistributedMetricControl{} : *root_control;
  }
  if (!DistributedRootUsesTcp(coordinator->root)) {
    return root_control == nullptr ? DistributedMetricControl{} : *root_control;
  }
  std::vector<std::uint8_t> payload;
  if (coordinator->rank == 0 && root_control != nullptr) {
    payload = SerializeDistributedMetricControl(*root_control);
  }
  const std::string key =
      coordinator->run_id + "/" + std::to_string(coordinator->tree_index) + "/" + label;
  return DeserializeDistributedMetricControl(DistributedTcpRequest(coordinator->root,
                                                                   coordinator->timeout_seconds,
                                                                   "broadcast",
                                                                   key,
                                                                   coordinator->rank,
                                                                   coordinator->world_size,
                                                                   payload));
}

DistributedMetricInputs AllGatherDistributedMetricInputs(
    const DistributedCoordinator* coordinator,
    const char* label,
    const DistributedMetricInputs& local_inputs) {
  if (coordinator == nullptr || coordinator->world_size <= 1 ||
      coordinator->root.empty()) {
    return local_inputs;
  }
  if (!DistributedRootUsesTcp(coordinator->root)) {
    return AllGatherFilesystemMetricInputs(*coordinator, label, local_inputs);
  }
  const std::string key =
      coordinator->run_id + "/" + std::to_string(coordinator->tree_index) + "/" + label;
  const std::vector<std::uint8_t> response = DistributedTcpRequest(
      coordinator->root,
      coordinator->timeout_seconds,
      "allgather",
      key,
      coordinator->rank,
      coordinator->world_size,
      SerializeDistributedMetricInputs(local_inputs));
  return MergeDistributedMetricInputs(DeserializeGatheredPayloads(response));
}

LeafStatistics AllReduceLeafStatistics(const DistributedCoordinator* coordinator,
                                       const char* label,
                                       const LeafStatistics& local_statistics) {
  if (local_statistics.gradient_sums.size() != local_statistics.hessian_sums.size()) {
    throw std::invalid_argument(
        "leaf gradient and hessian statistic buffers must have matching sizes");
  }
  if (coordinator == nullptr || coordinator->world_size <= 1) {
    return local_statistics;
  }
  if (!DistributedRootUsesTcp(coordinator->root)) {
    return AllReduceFilesystemLeafStatistics(
        *coordinator, label, local_statistics);
  }
  const std::string key = coordinator->run_id + "/" +
                          std::to_string(coordinator->tree_index) + "/" + label;
  const std::vector<std::uint8_t> response = DistributedTcpRequest(
      coordinator->root,
      coordinator->timeout_seconds,
      "allgather",
      key,
      coordinator->rank,
      coordinator->world_size,
      SerializeLeafStatistics(local_statistics));
  const std::vector<std::vector<std::uint8_t>> payloads =
      DeserializeGatheredPayloads(response);
  if (payloads.size() != static_cast<std::size_t>(coordinator->world_size)) {
    throw std::runtime_error("distributed leaf statistic rank count mismatch");
  }
  return ReduceLeafStatisticPayloads(payloads);
}

}  // namespace ctboost::booster_detail
