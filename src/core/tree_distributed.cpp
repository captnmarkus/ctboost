#include "tree_internal.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <thread>

#include "ctboost/distributed_client.hpp"

namespace ctboost::detail {
namespace {

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

}  // namespace

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
      const NodeHistogramSet rank_stats =
          ReadNodeHistogramSetBinary(rank_path, coordinator->timeout_seconds);
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
  return ReadNodeHistogramSetBinary(result_path, coordinator->timeout_seconds);
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

}  // namespace ctboost::detail
