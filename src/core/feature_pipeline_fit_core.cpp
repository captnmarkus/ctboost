#include "ctboost/feature_pipeline.hpp"

#include "feature_pipeline_internal.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace ctboost {

void NativeFeaturePipeline::FitCoreFeatureState(py::array object_matrix,
                                                const std::vector<float>&,
                                                const std::vector<int>& cat_indices,
                                                const std::vector<int>& text_indices,
                                                const std::vector<int>& embedding_indices) {
  const detail::MatrixView matrix = detail::MakeMatrixView(std::move(object_matrix));
  const std::unordered_set<int> text_reserved(text_indices.begin(), text_indices.end());
  const std::unordered_set<int> embedding_reserved(embedding_indices.begin(), embedding_indices.end());
  numeric_indices_.clear();
  for (int feature_index = 0; feature_index < n_features_in_; ++feature_index) {
    if (text_reserved.find(feature_index) == text_reserved.end() &&
        embedding_reserved.find(feature_index) == embedding_reserved.end()) {
      numeric_indices_.push_back(feature_index);
    }
  }

  auto feature_name = [this](int index) {
    if (!feature_names_in_.has_value()) {
      return std::string("f") + std::to_string(index);
    }
    return (*feature_names_in_)[static_cast<std::size_t>(index)];
  };

  const auto has_explicit_ctr_for_source = [this](const std::vector<int>& source_indices) {
    if (per_feature_ctr_.is_none()) {
      return false;
    }
    for (const auto& item : py::cast<py::dict>(per_feature_ctr_)) {
      py::list selectors;
      py::object key_object = py::reinterpret_borrow<py::object>(item.first);
      if (py::isinstance<py::list>(key_object) || py::isinstance<py::tuple>(key_object)) {
        for (const py::handle selector : key_object) {
          selectors.append(py::reinterpret_borrow<py::object>(selector));
        }
      } else {
        selectors.append(std::move(key_object));
      }
      if (ResolveIndices(std::move(selectors)) == source_indices) {
        return true;
      }
    }
    return false;
  };

  const std::unordered_set<int> cat_index_set(cat_indices.begin(), cat_indices.end());
  for (int feature_index : numeric_indices_) {
    const std::string name = feature_name(feature_index);
    if (cat_index_set.find(feature_index) == cat_index_set.end()) {
      output_feature_names_.push_back(name);
      continue;
    }

    std::unordered_map<std::string, std::size_t> key_counts;
    key_counts.reserve(matrix.rows);
    for (std::size_t row = 0; row < matrix.rows; ++row) {
      ++key_counts[detail::NormalizeKey(
          detail::MatrixValue(matrix, row, static_cast<std::size_t>(feature_index)))];
    }

    bool has_other_bucket = false;
    const std::vector<std::string> bucket_keys =
        detail::BuildBucketKeys(key_counts, max_cat_threshold_, &has_other_bucket);

    if (one_hot_max_size_ > 0 &&
        bucket_keys.size() <= static_cast<std::size_t>(one_hot_max_size_) &&
        !has_explicit_ctr_for_source({feature_index})) {
      OneHotEncoderState state;
      state.source_index = feature_index;
      state.prefix = name;
      state.category_keys = bucket_keys;
      state.has_other_bucket = has_other_bucket ? 1U : 0U;
      state.output_names.reserve(bucket_keys.size());
      for (const std::string& key : bucket_keys) {
        state.output_names.push_back(detail::OneHotOutputName(name, key));
        output_feature_names_.push_back(state.output_names.back());
      }
      one_hot_states_.push_back(std::move(state));
      continue;
    }

    CategoricalEncoderState state;
    state.source_index = feature_index;
    state.output_name = name;
    state.has_other_bucket = has_other_bucket ? 1U : 0U;
    for (std::size_t code = 0; code < bucket_keys.size(); ++code) {
      state.mapping.emplace(bucket_keys[code], static_cast<float>(code));
      if (bucket_keys[code] == detail::kOtherKey) {
        state.other_value = static_cast<float>(code);
      }
    }
    categorical_states_.push_back(std::move(state));
    cat_feature_indices_.push_back(static_cast<int>(output_feature_names_.size()));
    output_feature_names_.push_back(name);
  }

  RefreshCombinationSourceIndices();
  for (const auto& source_indices : combination_source_indices_) {
    std::unordered_map<std::string, std::size_t> key_counts;
    key_counts.reserve(matrix.rows);
    for (std::size_t row = 0; row < matrix.rows; ++row) {
      ++key_counts[detail::JoinNormalizedKey(matrix, row, source_indices)];
    }

    bool has_other_bucket = false;
    const std::vector<std::string> bucket_keys =
        detail::BuildBucketKeys(key_counts, max_cat_threshold_, &has_other_bucket);

    CategoricalEncoderState state;
    for (std::size_t index = 0; index < source_indices.size(); ++index) {
      if (index > 0) {
        state.output_name += "_x_";
      }
      state.output_name += feature_name(source_indices[index]);
    }
    state.has_other_bucket = has_other_bucket ? 1U : 0U;
    for (std::size_t code = 0; code < bucket_keys.size(); ++code) {
      state.mapping.emplace(bucket_keys[code], static_cast<float>(code));
      if (bucket_keys[code] == detail::kOtherKey) {
        state.other_value = static_cast<float>(code);
      }
    }
    combination_states_.push_back(std::move(state));
    cat_feature_indices_.push_back(static_cast<int>(output_feature_names_.size()));
    output_feature_names_.push_back(combination_states_.back().output_name);
  }
}

void NativeFeaturePipeline::FitTextAndEmbeddingState(const std::vector<int>& text_indices,
                                                     const std::vector<int>& embedding_indices) {
  auto feature_name = [this](int index) {
    if (!feature_names_in_.has_value()) {
      return std::string("f") + std::to_string(index);
    }
    return (*feature_names_in_)[static_cast<std::size_t>(index)];
  };

  for (int feature_index : text_indices) {
    TextState state;
    state.source_index = feature_index;
    state.prefix = feature_name(feature_index);
    text_states_.push_back(state);
    for (int hash_index = 0; hash_index < text_hash_dim_; ++hash_index) {
      output_feature_names_.push_back(state.prefix + "_hash" + std::to_string(hash_index));
    }
  }

  const std::vector<std::string> embedding_stats = detail::NormalizeEmbeddingStats(embedding_stats_);
  for (int feature_index : embedding_indices) {
    EmbeddingState state;
    state.source_index = feature_index;
    state.prefix = feature_name(feature_index);
    state.stats = embedding_stats;
    embedding_states_.push_back(state);
    for (const std::string& stat : state.stats) {
      output_feature_names_.push_back(state.prefix + "_" + stat);
    }
  }
}

}  // namespace ctboost
