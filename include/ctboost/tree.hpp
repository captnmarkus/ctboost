#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "ctboost/data.hpp"
#include "ctboost/histogram.hpp"
#include "ctboost/statistics.hpp"

namespace ctboost {

struct Node {
  bool is_leaf{true};
  int split_feature_id{-1};
  std::uint16_t split_bin_index{0};
  int left_child{-1};
  int right_child{-1};
  float leaf_weight{0.0F};
};

class Tree {
 public:
  void Build(const HistMatrix& hist,
             const std::vector<float>& gradients,
             const std::vector<float>& hessians,
             double alpha,
             int max_depth,
             double lambda_l2);

  float PredictRow(const Pool& pool, std::size_t row) const;
  float PredictBinnedRow(const HistMatrix& hist, std::size_t row) const;
  std::vector<float> Predict(const Pool& pool) const;
  const std::vector<Node>& nodes() const noexcept;

 private:
  int BuildNode(const HistMatrix& hist,
                const std::vector<float>& gradients,
                const std::vector<float>& hessians,
                const std::vector<std::size_t>& row_indices,
                int depth,
                double alpha,
                int max_depth,
                double lambda_l2,
                const LinearStatistic& statistic_engine);

  std::uint16_t BinValue(std::size_t feature_index, float value) const;

  std::vector<Node> nodes_;
  std::vector<std::uint16_t> num_bins_per_feature_;
  std::vector<std::size_t> cut_offsets_;
  std::vector<float> cut_values_;
  std::vector<std::uint8_t> categorical_mask_;
};

}  // namespace ctboost
