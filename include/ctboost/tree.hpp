#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "ctboost/data.hpp"
#include "ctboost/histogram.hpp"
#include "ctboost/statistics.hpp"

namespace ctboost {

inline constexpr std::size_t kMaxCategoricalRouteBins = 256;

class TrainingProfiler;
struct GpuHistogramWorkspace;
struct GpuHistogramSnapshot;
struct NodeHistogramSet;

struct InteractionConstraintSet {
  std::vector<std::vector<int>> groups;
  std::vector<std::vector<int>> feature_to_groups;
  std::vector<std::uint8_t> constrained_feature_mask;
};

struct Node {
  bool is_leaf{true};
  bool is_categorical_split{false};
  int split_feature_id{-1};
  std::uint16_t split_bin_index{0};
  int left_child{-1};
  int right_child{-1};
  float leaf_weight{0.0F};
  std::array<std::uint8_t, kMaxCategoricalRouteBins> left_categories{};
};

struct LeafRowRange {
  std::size_t begin{0};
  std::size_t end{0};
};

struct TreeBuildOptions {
  double alpha{0.05};
  int max_depth{6};
  double lambda_l2{1.0};
  bool use_gpu{false};
  int max_leaves{0};
  int min_data_in_leaf{0};
  double min_child_weight{0.0};
  double min_split_gain{0.0};
  const std::vector<int>* allowed_features{nullptr};
  const std::vector<int>* monotone_constraints{nullptr};
  const InteractionConstraintSet* interaction_constraints{nullptr};
};

class Tree {
 public:
  void Build(const HistMatrix& hist,
             const std::vector<float>& gradients,
             const std::vector<float>& hessians,
             const std::vector<float>& weights,
             const TreeBuildOptions& options,
             GpuHistogramWorkspace* gpu_workspace = nullptr,
             const TrainingProfiler* profiler = nullptr,
             std::vector<std::size_t>* row_indices_out = nullptr,
             std::vector<LeafRowRange>* leaf_row_ranges_out = nullptr,
             const QuantizationSchemaPtr& quantization_schema = nullptr);

  float PredictRow(const Pool& pool, std::size_t row) const;
  float PredictBinnedRow(const HistMatrix& hist, std::size_t row) const;
  int PredictBinnedLeafIndex(const HistMatrix& hist, std::size_t row) const;
  int PredictLeafIndex(const Pool& pool, std::size_t row) const;
  void AccumulateContributions(
      const Pool& pool, std::size_t row, float scale, std::vector<float>& row_contributions) const;
  std::vector<float> Predict(const Pool& pool) const;
  void SetLeafWeight(std::size_t node_index, float leaf_weight);
  void SetQuantizationSchema(const QuantizationSchemaPtr& quantization_schema);
  const QuantizationSchemaPtr& shared_quantization_schema() const noexcept;
  void LoadState(std::vector<Node> nodes,
                 const QuantizationSchemaPtr& quantization_schema,
                 std::vector<double> feature_importances);
  void LoadState(std::vector<Node> nodes,
                 std::vector<std::uint16_t> num_bins_per_feature,
                 std::vector<std::size_t> cut_offsets,
                 std::vector<float> cut_values,
                 std::vector<std::uint8_t> categorical_mask,
                 std::vector<std::uint8_t> missing_value_mask,
                 std::uint8_t nan_mode,
                 std::vector<double> feature_importances);
  const std::vector<Node>& nodes() const noexcept;
  const std::vector<std::uint16_t>& num_bins_per_feature() const;
  const std::vector<std::size_t>& cut_offsets() const;
  const std::vector<float>& cut_values() const;
  const std::vector<std::uint8_t>& categorical_mask() const;
  const std::vector<std::uint8_t>& missing_value_mask() const;
  std::uint8_t nan_mode() const;
  const std::vector<double>& feature_importances() const noexcept;

 private:
  int BuildNode(const HistMatrix& hist,
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
                int* leaf_count);

  std::uint16_t BinValue(std::size_t feature_index, float value) const;

  std::vector<Node> nodes_;
  QuantizationSchemaPtr quantization_schema_;
  std::vector<double> feature_importances_;
};

}  // namespace ctboost
