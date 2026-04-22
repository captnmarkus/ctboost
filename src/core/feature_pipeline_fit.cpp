#include "ctboost/feature_pipeline.hpp"

#include "feature_pipeline_internal.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace ctboost {

std::vector<int> NativeFeaturePipeline::ResolveIndices(const py::object& selectors) const {
  if (selectors.is_none()) {
    return {};
  }

  std::unordered_map<std::string, int> name_to_index;
  if (feature_names_in_.has_value()) {
    for (std::size_t index = 0; index < feature_names_in_->size(); ++index) {
      name_to_index.emplace((*feature_names_in_)[index], static_cast<int>(index));
    }
  }

  std::vector<int> resolved;
  for (const py::handle selector : selectors) {
    if (py::isinstance<py::str>(selector)) {
      const std::string name = py::str(selector).cast<std::string>();
      const auto it = name_to_index.find(name);
      if (it == name_to_index.end()) {
        throw std::invalid_argument("unknown feature selector: " + name);
      }
      resolved.push_back(it->second);
      continue;
    }

    const int index = py::cast<int>(selector);
    if (index < 0 || index >= n_features_in_) {
      throw std::invalid_argument("feature selector is out of bounds");
    }
    resolved.push_back(index);
  }

  std::sort(resolved.begin(), resolved.end());
  resolved.erase(std::unique(resolved.begin(), resolved.end()), resolved.end());
  return resolved;
}

void NativeFeaturePipeline::RefreshCombinationSourceIndices() {
  combination_source_indices_.clear();
  if (n_features_in_ < 0) {
    return;
  }

  if (!categorical_combinations_.is_none()) {
    for (const py::handle combination : categorical_combinations_) {
      py::list selectors;
      for (const py::handle selector : combination) {
        selectors.append(py::reinterpret_borrow<py::object>(selector));
      }
      std::vector<int> indices = ResolveIndices(std::move(selectors));
      if (indices.size() >= 2) {
        combination_source_indices_.push_back(std::move(indices));
      }
    }
  }

  if (pairwise_categorical_combinations_) {
    const std::vector<int> cat_indices = ResolveIndices(cat_features_);
    for (std::size_t left = 0; left < cat_indices.size(); ++left) {
      for (std::size_t right = left + 1; right < cat_indices.size(); ++right) {
        combination_source_indices_.push_back({cat_indices[left], cat_indices[right]});
      }
    }
  }

  std::sort(combination_source_indices_.begin(), combination_source_indices_.end());
  combination_source_indices_.erase(
      std::unique(combination_source_indices_.begin(), combination_source_indices_.end()),
      combination_source_indices_.end());
}

void NativeFeaturePipeline::FitInternal(py::array raw_matrix,
                                        py::array_t<float, py::array::forcecast> labels,
                                        py::object feature_names) {
  py::array object_matrix = detail::EnsureObjectMatrix(std::move(raw_matrix));
  const detail::MatrixView matrix = detail::MakeMatrixView(object_matrix);
  const std::vector<float> label_values = detail::ArrayToFloatVector(std::move(labels), "labels");
  if (label_values.size() != matrix.rows) {
    throw std::invalid_argument("label size must match the number of rows");
  }

  n_features_in_ = static_cast<int>(matrix.cols);
  if (feature_names.is_none()) {
    feature_names_in_.reset();
  } else {
    feature_names_in_ = py::cast<std::vector<std::string>>(feature_names);
  }

  const std::vector<int> cat_indices = ResolveIndices(cat_features_);
  const std::vector<int> text_indices = ResolveIndices(text_features_);
  const std::vector<int> embedding_indices = ResolveIndices(embedding_features_);
  const std::unordered_set<int> text_index_set(text_indices.begin(), text_indices.end());
  for (int feature_index : embedding_indices) {
    if (text_index_set.find(feature_index) != text_index_set.end()) {
      throw std::invalid_argument("text_features and embedding_features cannot overlap");
    }
  }

  output_feature_names_.clear();
  cat_feature_indices_.clear();
  one_hot_states_.clear();
  categorical_states_.clear();
  combination_states_.clear();
  ctr_states_.clear();
  text_states_.clear();
  embedding_states_.clear();
  training_ctr_columns_.clear();
  text_hash_cache_.clear();

  FitCoreFeatureState(object_matrix, label_values, cat_indices, text_indices, embedding_indices);
  FitCtrState(object_matrix, label_values);
  FitTextAndEmbeddingState(text_indices, embedding_indices);
}

}  // namespace ctboost
