#include "booster_internal.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace ctboost {

void GradientBooster::SetIterations(int iterations) {
  if (iterations <= 0) {
    throw std::invalid_argument("iterations must be positive");
  }
  iterations_ = iterations;
}

void GradientBooster::SetLearningRate(double learning_rate) {
  if (learning_rate <= 0.0) {
    throw std::invalid_argument("learning_rate must be positive");
  }
  learning_rate_ = learning_rate;
}

void GradientBooster::LoadState(std::vector<Tree> trees,
                                QuantizationSchemaPtr quantization_schema,
                                std::vector<double> loss_history,
                                std::vector<double> eval_loss_history,
                                std::vector<double> tree_learning_rates,
                                std::vector<double> feature_importance_sums,
                                int best_iteration,
                                double best_score,
                                bool use_gpu,
                                std::uint64_t rng_state) {
  trees_ = std::move(trees);
  tree_learning_rates_ = std::move(tree_learning_rates);
  if (quantization_schema == nullptr && !trees_.empty()) {
    auto mutable_quantization_schema = std::make_shared<QuantizationSchema>();
    mutable_quantization_schema->num_bins_per_feature = trees_.front().num_bins_per_feature();
    mutable_quantization_schema->cut_offsets = trees_.front().cut_offsets();
    mutable_quantization_schema->cut_values = trees_.front().cut_values();
    mutable_quantization_schema->categorical_mask = trees_.front().categorical_mask();
    mutable_quantization_schema->missing_value_mask = trees_.front().missing_value_mask();
    mutable_quantization_schema->nan_mode = trees_.front().nan_mode();
    quantization_schema = mutable_quantization_schema;
  }
  quantization_schema_ = quantization_schema;
  if (quantization_schema_ != nullptr) {
    for (Tree& tree : trees_) {
      tree.SetQuantizationSchema(quantization_schema_);
    }
  }
  const std::size_t trained_iterations = num_iterations_trained();
  if (!tree_learning_rates_.empty() && tree_learning_rates_.size() != trained_iterations) {
    throw std::invalid_argument("tree_learning_rates must match the trained iteration count");
  }
  if (tree_learning_rates_.empty() && trained_iterations > 0) {
    tree_learning_rates_.assign(trained_iterations, learning_rate_);
  }
  loss_history_ = std::move(loss_history);
  eval_loss_history_ = std::move(eval_loss_history);
  if (feature_importance_sums.empty()) {
    const std::size_t num_features = quantization_schema_ == nullptr ? 0 : quantization_schema_->num_cols();
    booster_detail::RecomputeFeatureImportances(trees_, num_features, feature_importance_sums_);
  } else {
    feature_importance_sums_ = std::move(feature_importance_sums);
  }
  best_iteration_ = best_iteration;
  best_score_ = best_score;
  use_gpu_ = use_gpu;
  if (rng_state != 0) {
    rng_state_ = booster_detail::NormalizeRngState(rng_state);
  }
}

void GradientBooster::LoadQuantizationSchema(QuantizationSchemaPtr quantization_schema) {
  quantization_schema_ = std::move(quantization_schema);
  if (quantization_schema_ != nullptr) {
    for (Tree& tree : trees_) {
      tree.SetQuantizationSchema(quantization_schema_);
    }
  }
}

std::vector<float> GradientBooster::get_feature_importances() const {
  std::vector<float> importances(feature_importance_sums_.size(), 0.0F);
  const double total_importance = std::accumulate(feature_importance_sums_.begin(),
                                                  feature_importance_sums_.end(),
                                                  0.0);
  if (total_importance <= 0.0) {
    return importances;
  }
  const double scale = 100.0 / total_importance;
  for (std::size_t feature = 0; feature < feature_importance_sums_.size(); ++feature) {
    importances[feature] = static_cast<float>(feature_importance_sums_[feature] * scale);
  }
  return importances;
}

}  // namespace ctboost
