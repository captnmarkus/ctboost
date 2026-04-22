#pragma once

#include "ctboost/tree.hpp"

#include <filesystem>
#include <limits>
#include <vector>

#include "ctboost/cuda_backend.hpp"

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

namespace detail {

struct ChildLeafBounds {
  double left_lower_bound{0.0};
  double left_upper_bound{0.0};
  double right_lower_bound{0.0};
  double right_upper_bound{0.0};
};

struct ChildInteractionState {
  std::vector<int> active_groups_storage;
  std::vector<int> allowed_features_storage;
  const std::vector<int>* active_groups{nullptr};
  const std::vector<int>* allowed_features{nullptr};
};

struct CpuChildHistogramState {
  NodeHistogramSet left_stats;
  NodeHistogramSet right_stats;
  double left_histogram_ms{0.0};
  double right_histogram_ms{0.0};
};

struct GpuChildHistogramState {
  GpuHistogramSnapshot left_snapshot;
  GpuHistogramSnapshot right_snapshot;
  double left_histogram_ms{0.0};
  double right_histogram_ms{0.0};
  bool left_snapshot_resident{false};
  bool right_snapshot_resident{false};
};

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

double ComputeLeafWeight(double gradient_sum, double hessian_sum, double lambda_l2);
double ComputeGain(double gradient_sum, double hessian_sum, double lambda_l2);
double ComputeGradientVariance(double weighted_gradient_sum,
                               double weighted_gradient_square_sum,
                               double sample_weight_sum);
ChildLeafBounds ComputeChildLeafBounds(const TreeBuildOptions& options,
                                       int feature_id,
                                       double left_leaf_weight,
                                       double right_leaf_weight,
                                       double leaf_lower_bound,
                                       double leaf_upper_bound);
const QuantizationSchema& RequireQuantizationSchema(const QuantizationSchemaPtr& schema);
double ClampLeafWeight(double leaf_weight, double lower_bound, double upper_bound);
ChildInteractionState ResolveChildInteractionState(
    const HistMatrix& hist,
    const TreeBuildOptions& options,
    int feature_id,
    const std::vector<int>* node_allowed_features,
    const std::vector<int>* active_interaction_groups);

NodeHistogramSet ComputeNodeHistogramSet(const HistMatrix& hist,
                                         const std::vector<float>& gradients,
                                         const std::vector<float>& hessians,
                                         const std::vector<float>& weights,
                                         const std::vector<std::size_t>& row_indices,
                                         std::size_t row_begin,
                                         std::size_t row_end,
                                         bool use_gpu,
                                         GpuHistogramWorkspace* gpu_workspace);
NodeHistogramSet SubtractNodeHistogramSet(const NodeHistogramSet& parent,
                                          const NodeHistogramSet& child);

std::vector<std::uint8_t> SerializeNodeHistogramSetBinary(const NodeHistogramSet& stats);
NodeHistogramSet DeserializeNodeHistogramSetBinary(const std::vector<std::uint8_t>& buffer);
void WriteNodeHistogramSetBinary(const std::filesystem::path& path,
                                 const NodeHistogramSet& stats);
NodeHistogramSet ReadNodeHistogramSetBinary(const std::filesystem::path& path,
                                            double timeout_seconds = 0.0);
void AddNodeHistogramSet(NodeHistogramSet& target, const NodeHistogramSet& source);

std::vector<std::uint8_t> SerializeGpuHistogramSnapshotBinary(
    const GpuHistogramSnapshot& snapshot);
GpuHistogramSnapshot DeserializeGpuHistogramSnapshotBinary(
    const std::vector<std::uint8_t>& buffer);
GpuHistogramSnapshot SubtractGpuHistogramSnapshot(const GpuHistogramSnapshot& parent,
                                                  const GpuHistogramSnapshot& child);

NodeHistogramSet AllReduceNodeHistogramSet(DistributedCoordinator* coordinator,
                                           const NodeHistogramSet& local_stats);
GpuHistogramSnapshot AllReduceGpuHistogramSnapshot(DistributedCoordinator* coordinator,
                                                   const GpuHistogramSnapshot& local_snapshot);
CpuChildHistogramState BuildCpuChildHistogramState(
    const HistMatrix& hist,
    const std::vector<float>& gradients,
    const std::vector<float>& hessians,
    const std::vector<float>& weights,
    const std::vector<std::size_t>& row_indices,
    std::size_t row_begin,
    std::size_t left_end,
    std::size_t row_end,
    bool build_left_direct,
    const TreeBuildOptions& options,
    const NodeHistogramSet& node_stats);
bool ChooseCpuFirstChild(const HistMatrix& hist,
                         const CpuChildHistogramState& child_histograms,
                         const TreeBuildOptions& options,
                         const LinearStatistic& statistic_engine,
                         const std::vector<int>* child_allowed_features,
                         const ChildLeafBounds& child_bounds,
                         int depth,
                         std::size_t row_begin,
                         std::size_t left_end,
                         std::size_t row_end);
GpuChildHistogramState BuildGpuChildHistogramState(const TreeBuildOptions& options,
                                                   GpuHistogramWorkspace* gpu_workspace,
                                                   std::size_t row_begin,
                                                   std::size_t left_end,
                                                   std::size_t row_end,
                                                   const GpuHistogramSnapshot& parent_snapshot,
                                                   bool build_left_direct);
bool ChooseGpuFirstChild(const TreeBuildOptions& options,
                         GpuHistogramWorkspace* gpu_workspace,
                         const std::vector<int>* child_allowed_features,
                         const ChildLeafBounds& child_bounds,
                         int depth,
                         std::size_t row_begin,
                         std::size_t left_end,
                         std::size_t row_end,
                         GpuChildHistogramState* child_histograms);

std::vector<int> FilterAllowedFeaturesForInteraction(
    std::size_t num_features,
    const std::vector<int>* parent_allowed_features,
    const InteractionConstraintSet& constraints,
    const std::vector<int>* active_groups);
std::vector<int> IntersectSortedVectors(const std::vector<int>& lhs, const std::vector<int>& rhs);
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
                            double leaf_upper_bound);
double AdjustedCandidateGain(const TreeBuildOptions& options,
                             int feature_id,
                             double raw_gain,
                             int depth,
                             std::size_t row_begin,
                             std::size_t row_end);
CandidateSelectionResult SelectBestCandidateSplit(const HistMatrix& hist,
                                                  const NodeHistogramSet& node_stats,
                                                  const TreeBuildOptions& options,
                                                  const LinearStatistic& statistic_engine,
                                                  const std::vector<int>* node_allowed_features,
                                                  double leaf_lower_bound,
                                                  double leaf_upper_bound,
                                                  int depth,
                                                  std::size_t row_begin,
                                                  std::size_t row_end);

}  // namespace detail
}  // namespace ctboost
