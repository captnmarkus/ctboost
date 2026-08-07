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

  std::vector<float> predictions(
      pool.num_rows() * static_cast<std::size_t>(prediction_dimension_), 0.0F);
  if (tree_limit == 0) {
    booster_detail::AddPoolBaselineToPredictions(pool, prediction_dimension_, predictions);
    return predictions;
  }
  const auto& quantization_schema =
      booster_detail::RequireQuantizationSchema(quantization_schema_);
  std::vector<std::uint8_t> active_features(quantization_schema.num_cols(), 0U);
  for (std::size_t tree_index = 0; tree_index < tree_limit; ++tree_index) {
    booster_detail::MarkUsedFeatures(trees_[tree_index], active_features);
  }
  const HistMatrix hist = booster_detail::BuildPredictionHist(
      pool,
      quantization_schema,
      &active_features);
  predictions = booster_detail::PredictFromHist(trees_,
                                                hist,
                                                tree_limit,
                                                tree_learning_rates_,
                                                learning_rate_,
                                                use_gpu_,
                                                prediction_dimension_,
                                                devices_);
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
  if (tree_limit == 0) {
    return leaf_indices;
  }
  const auto& quantization_schema =
      booster_detail::RequireQuantizationSchema(quantization_schema_);
  std::vector<std::uint8_t> active_features(quantization_schema.num_cols(), 0U);
  for (std::size_t tree_index = 0; tree_index < tree_limit; ++tree_index) {
    booster_detail::MarkUsedFeatures(trees_[tree_index], active_features);
  }
  const HistMatrix hist = booster_detail::BuildPredictionHist(
      pool,
      quantization_schema,
      &active_features);
  for (std::size_t tree_index = 0; tree_index < tree_limit; ++tree_index) {
    for (std::size_t row = 0; row < pool.num_rows(); ++row) {
      leaf_indices[row * tree_limit + tree_index] =
          trees_[tree_index].PredictBinnedLeafIndex(hist, row);
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
  if (tree_limit == 0) {
    return contributions;
  }
  const auto& quantization_schema =
      booster_detail::RequireQuantizationSchema(quantization_schema_);
  std::vector<std::uint8_t> active_features(quantization_schema.num_cols(), 0U);
  for (std::size_t tree_index = 0; tree_index < tree_limit; ++tree_index) {
    booster_detail::MarkUsedFeatures(trees_[tree_index], active_features);
  }
  const HistMatrix hist = booster_detail::BuildPredictionHist(
      pool,
      quantization_schema,
      &active_features);
  std::vector<float> row_buffer(pool.num_cols() + 1, 0.0F);
  for (std::size_t tree_index = 0; tree_index < tree_limit; ++tree_index) {
    const std::size_t class_index =
        prediction_dimension_ == 1 ? 0 : tree_index % static_cast<std::size_t>(prediction_dimension_);
    const float tree_learning_rate = static_cast<float>(booster_detail::ResolveIterationLearningRate(
        tree_learning_rates_, tree_index, prediction_dimension_, learning_rate_));
    for (std::size_t row = 0; row < pool.num_rows(); ++row) {
      std::fill(row_buffer.begin(), row_buffer.end(), 0.0F);
      trees_[tree_index].AccumulateBinnedContributions(
          hist, row, tree_learning_rate, row_buffer);
      const std::size_t row_offset = row * row_width + class_index * (pool.num_cols() + 1);
      for (std::size_t feature = 0; feature < row_buffer.size(); ++feature) {
        contributions[row_offset + feature] += row_buffer[feature];
      }
    }
  }
  return contributions;
}

}  // namespace ctboost
