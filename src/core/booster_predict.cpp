#include "booster_internal.hpp"

#include <algorithm>

namespace ctboost {

std::vector<float> GradientBooster::Predict(const Pool& pool, int num_iteration) const {
  std::size_t tree_limit = trees_.size();
  if (num_iteration >= 0) {
    tree_limit = std::min(
        trees_.size(),
        static_cast<std::size_t>(num_iteration) * static_cast<std::size_t>(prediction_dimension_));
  }

  std::vector<float> predictions(pool.num_rows() * static_cast<std::size_t>(prediction_dimension_), 0.0F);
  if (tree_limit == 0 || pool.num_rows() == 0) {
    booster_detail::AddPoolBaselineToPredictions(pool, prediction_dimension_, predictions);
    return predictions;
  }
  if (use_gpu_ && CudaBackendCompiled()) {
    const HistMatrix hist =
        booster_detail::BuildPredictionHist(pool, booster_detail::RequireQuantizationSchema(quantization_schema_));
    std::vector<std::int32_t> tree_offsets;
    const std::vector<GpuTreeNode> flattened_nodes = booster_detail::FlattenTreesForGpu(
        trees_, tree_limit, tree_learning_rates_, learning_rate_, prediction_dimension_, tree_offsets);
    PredictRawGpu(hist, flattened_nodes, tree_offsets, 1.0F, prediction_dimension_, predictions, devices_);
    booster_detail::AddPoolBaselineToPredictions(pool, prediction_dimension_, predictions);
    return predictions;
  }

  for (std::size_t tree_index = 0; tree_index < tree_limit; ++tree_index) {
    const double tree_learning_rate = booster_detail::ResolveIterationLearningRate(
        tree_learning_rates_, tree_index, prediction_dimension_, learning_rate_);
    const int class_index =
        prediction_dimension_ == 1 ? 0 : static_cast<int>(tree_index % static_cast<std::size_t>(prediction_dimension_));
    booster_detail::UpdatePredictions(
        trees_[tree_index], pool, tree_learning_rate, prediction_dimension_, class_index, predictions);
  }
  booster_detail::AddPoolBaselineToPredictions(pool, prediction_dimension_, predictions);
  return predictions;
}

std::vector<std::int32_t> GradientBooster::PredictLeafIndices(const Pool& pool,
                                                              int num_iteration) const {
  std::size_t tree_limit = trees_.size();
  if (num_iteration >= 0) {
    tree_limit = std::min(
        trees_.size(),
        static_cast<std::size_t>(num_iteration) * static_cast<std::size_t>(prediction_dimension_));
  }
  std::vector<std::int32_t> leaf_indices(pool.num_rows() * tree_limit, -1);
  for (std::size_t tree_index = 0; tree_index < tree_limit; ++tree_index) {
    for (std::size_t row = 0; row < pool.num_rows(); ++row) {
      leaf_indices[row * tree_limit + tree_index] = trees_[tree_index].PredictLeafIndex(pool, row);
    }
  }
  return leaf_indices;
}

std::vector<float> GradientBooster::PredictContributions(const Pool& pool, int num_iteration) const {
  std::size_t tree_limit = trees_.size();
  if (num_iteration >= 0) {
    tree_limit = std::min(
        trees_.size(),
        static_cast<std::size_t>(num_iteration) * static_cast<std::size_t>(prediction_dimension_));
  }
  const std::size_t row_width = static_cast<std::size_t>(prediction_dimension_) * (pool.num_cols() + 1);
  std::vector<float> contributions(pool.num_rows() * row_width, 0.0F);
  for (std::size_t tree_index = 0; tree_index < tree_limit; ++tree_index) {
    const std::size_t class_index =
        prediction_dimension_ == 1 ? 0 : tree_index % static_cast<std::size_t>(prediction_dimension_);
    const float tree_learning_rate = static_cast<float>(booster_detail::ResolveIterationLearningRate(
        tree_learning_rates_, tree_index, prediction_dimension_, learning_rate_));
    for (std::size_t row = 0; row < pool.num_rows(); ++row) {
      std::vector<float> row_buffer(pool.num_cols() + 1, 0.0F);
      trees_[tree_index].AccumulateContributions(pool, row, tree_learning_rate, row_buffer);
      const std::size_t row_offset = row * row_width + class_index * (pool.num_cols() + 1);
      for (std::size_t feature = 0; feature < row_buffer.size(); ++feature) {
        contributions[row_offset + feature] += row_buffer[feature];
      }
    }
  }
  return contributions;
}

}  // namespace ctboost
