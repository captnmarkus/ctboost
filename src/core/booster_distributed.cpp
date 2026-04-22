#include "booster_internal.hpp"

#include <cstring>
#include <stdexcept>

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
      !DistributedRootUsesTcp(coordinator->root)) {
    return local_inputs;
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

}  // namespace ctboost::booster_detail
