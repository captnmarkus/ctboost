#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "ctboost/data.hpp"
#include "ctboost/histogram.hpp"
#include "ctboost/statistics.hpp"

namespace ctboost {

inline constexpr std::size_t kMaxCategoricalRouteBins = 256;

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

class Tree {
 public:
  void Build(const HistMatrix& hist,
             const std::vector<float>& gradients,
             const std::vector<float>& hessians,
             const std::vector<float>& weights,
             double alpha,
             int max_depth,
             double lambda_l2,
             bool use_gpu);

  float PredictRow(const Pool& pool, std::size_t row) const;
  float PredictBinnedRow(const HistMatrix& hist, std::size_t row) const;
  int PredictLeafIndex(const Pool& pool, std::size_t row) const;
  void AccumulateContributions(
      const Pool& pool, std::size_t row, float scale, std::vector<float>& row_contributions) const;
  std::vector<float> Predict(const Pool& pool) const;
  void LoadState(std::vector<Node> nodes,
                 std::vector<std::uint16_t> num_bins_per_feature,
                 std::vector<std::size_t> cut_offsets,
                 std::vector<float> cut_values,
                 std::vector<std::uint8_t> categorical_mask,
                 std::vector<std::uint8_t> missing_value_mask,
                 std::uint8_t nan_mode,
                 std::vector<double> feature_importances);
  const std::vector<Node>& nodes() const noexcept;
  const std::vector<std::uint16_t>& num_bins_per_feature() const noexcept;
  const std::vector<std::size_t>& cut_offsets() const noexcept;
  const std::vector<float>& cut_values() const noexcept;
  const std::vector<std::uint8_t>& categorical_mask() const noexcept;
  const std::vector<std::uint8_t>& missing_value_mask() const noexcept;
  std::uint8_t nan_mode() const noexcept;
  const std::vector<double>& feature_importances() const noexcept;

 private:
  int BuildNode(const HistMatrix& hist,
                const std::vector<float>& gradients,
                const std::vector<float>& hessians,
                const std::vector<float>& weights,
                const std::vector<std::size_t>& row_indices,
                int depth,
                double alpha,
                int max_depth,
                double lambda_l2,
                bool use_gpu,
                const LinearStatistic& statistic_engine);

  std::uint16_t BinValue(std::size_t feature_index, float value) const;

  std::vector<Node> nodes_;
  std::vector<std::uint16_t> num_bins_per_feature_;
  std::vector<std::size_t> cut_offsets_;
  std::vector<float> cut_values_;
  std::vector<std::uint8_t> categorical_mask_;
  std::vector<std::uint8_t> missing_value_mask_;
  std::uint8_t nan_mode_{static_cast<std::uint8_t>(NanMode::Min)};
  std::vector<double> feature_importances_;
};

}  // namespace ctboost
