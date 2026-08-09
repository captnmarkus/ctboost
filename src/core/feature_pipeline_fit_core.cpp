#include "ctboost/feature_pipeline.hpp"

#include "feature_pipeline_internal.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
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

void NativeFeaturePipeline::FitTextAndEmbeddingState(py::array object_matrix,
                                                     const std::vector<float>& label_values,
                                                     const std::vector<int>& text_indices,
                                                     const std::vector<int>& embedding_indices) {
  const detail::MatrixView matrix = detail::MakeMatrixView(std::move(object_matrix));
  auto feature_name = [this](int index) {
    if (!feature_names_in_.has_value()) {
      return std::string("f") + std::to_string(index);
    }
    return (*feature_names_in_)[static_cast<std::size_t>(index)];
  };

  auto tokenize = [this](const py::handle& raw_value) {
    if (detail::IsMissing(raw_value)) {
      return std::vector<std::string>{};
    }
    return detail::ExtractTextTokens(py::str(raw_value).cast<std::string>(),
                                     text_tokenizer_,
                                     text_ngram_min_,
                                     text_ngram_max_,
                                     text_lowercase_);
  };

  auto hash_token = [this](const std::string& token) {
    auto cache_it = text_hash_cache_.find(token);
    if (cache_it == text_hash_cache_.end()) {
      const py::bytes digest =
          detail::HashlibModule()
              .attr("blake2b")(py::bytes(token), py::arg("digest_size") = 8)
              .attr("digest")()
              .cast<py::bytes>();
      cache_it = text_hash_cache_.emplace(token, detail::BytesToLittleEndianU64(digest)).first;
    }
    return static_cast<int>(cache_it->second % static_cast<std::uint64_t>(text_hash_dim_));
  };

  for (int feature_index : text_indices) {
    TextState state;
    state.source_index = feature_index;
    state.prefix = feature_name(feature_index);

    std::unordered_map<std::string, std::size_t> token_counts;
    if (text_max_dictionary_size_ > 0 || text_min_token_count_ > 1) {
      for (std::size_t row = 0; row < matrix.rows; ++row) {
        for (const std::string& token : tokenize(detail::MatrixValue(
                 matrix, row, static_cast<std::size_t>(feature_index)))) {
          ++token_counts[token];
        }
      }
    }

    std::vector<std::pair<std::string, std::size_t>> ranked_tokens;
    ranked_tokens.reserve(token_counts.size());
    for (const auto& [token, count] : token_counts) {
      if (count >= static_cast<std::size_t>(text_min_token_count_)) {
        ranked_tokens.emplace_back(token, count);
      }
    }
    std::sort(ranked_tokens.begin(), ranked_tokens.end(), [](const auto& lhs, const auto& rhs) {
      if (lhs.second != rhs.second) {
        return lhs.second > rhs.second;
      }
      return lhs.first < rhs.first;
    });

    state.uses_dictionary = text_max_dictionary_size_ > 0 ? 1U : 0U;
    state.filters_tokens = text_min_token_count_ > 1 ? 1U : 0U;
    if (state.uses_dictionary != 0U) {
      const std::size_t dictionary_size = std::min<std::size_t>(
          ranked_tokens.size(), static_cast<std::size_t>(text_max_dictionary_size_));
      state.vocabulary.reserve(dictionary_size);
      for (std::size_t index = 0; index < dictionary_size; ++index) {
        state.vocabulary.push_back(ranked_tokens[index].first);
        state.vocabulary_indices.emplace(ranked_tokens[index].first, static_cast<int>(index));
      }
      state.output_dim = static_cast<int>(state.vocabulary.size());
    } else {
      state.output_dim = text_hash_dim_;
      if (state.filters_tokens != 0U) {
        state.vocabulary.reserve(ranked_tokens.size());
        for (const auto& [token, _] : ranked_tokens) {
          (void)_;
          state.vocabulary.push_back(token);
          state.vocabulary_indices.emplace(token, 0);
        }
      }
    }

    if (text_feature_calcer_ == "tfidf") {
      std::vector<std::size_t> document_frequency(
          static_cast<std::size_t>(state.output_dim), 0U);
      for (std::size_t row = 0; row < matrix.rows; ++row) {
        std::unordered_set<int> seen_columns;
        for (const std::string& token : tokenize(detail::MatrixValue(
                 matrix, row, static_cast<std::size_t>(feature_index)))) {
          int token_index = -1;
          if (state.uses_dictionary != 0U) {
            const auto vocabulary_it = state.vocabulary_indices.find(token);
            if (vocabulary_it == state.vocabulary_indices.end()) {
              continue;
            }
            token_index = vocabulary_it->second;
          } else {
            if (state.filters_tokens != 0U &&
                state.vocabulary_indices.find(token) == state.vocabulary_indices.end()) {
              continue;
            }
            token_index = hash_token(token);
          }
          seen_columns.insert(token_index);
        }
        for (int token_index : seen_columns) {
          ++document_frequency[static_cast<std::size_t>(token_index)];
        }
      }
      state.idf_values.resize(static_cast<std::size_t>(state.output_dim), 1.0F);
      for (std::size_t index = 0; index < state.idf_values.size(); ++index) {
        state.idf_values[index] = static_cast<float>(
            std::log((1.0 + static_cast<double>(matrix.rows)) /
                     (1.0 + static_cast<double>(document_frequency[index]))) +
            1.0);
      }
    }

    if (state.uses_dictionary != 0U) {
      for (std::size_t index = 0; index < state.vocabulary.size(); ++index) {
        output_feature_names_.push_back(
            detail::TextOutputName(state.prefix, state.vocabulary[index], index));
      }
    } else {
      for (int hash_index = 0; hash_index < state.output_dim; ++hash_index) {
        output_feature_names_.push_back(state.prefix + "_hash" + std::to_string(hash_index));
      }
    }
    text_states_.push_back(std::move(state));
  }

  const std::vector<std::string> embedding_stats = detail::NormalizeEmbeddingStats(embedding_stats_);
  for (int feature_index : embedding_indices) {
    EmbeddingState state;
    state.source_index = feature_index;
    state.prefix = feature_name(feature_index);
    state.stats = embedding_stats;
    for (const std::string& stat : state.stats) {
      output_feature_names_.push_back(state.prefix + "_" + stat);
    }

    if (embedding_target_features_) {
      std::vector<std::size_t> valid_rows;
      std::vector<std::vector<float>> embeddings;
      std::size_t embedding_dimension = 0U;
      for (std::size_t row = 0; row < matrix.rows; ++row) {
        std::vector<float> values = detail::EmbeddingValues(detail::MatrixValue(
            matrix, row, static_cast<std::size_t>(feature_index)));
        if (values.empty()) {
          continue;
        }
        if (embedding_dimension == 0U) {
          embedding_dimension = values.size();
        } else if (values.size() != embedding_dimension) {
          throw std::invalid_argument(
              "embedding_target_features requires a consistent embedding dimension");
        }
        valid_rows.push_back(row);
        embeddings.push_back(std::move(values));
      }

      state.center.assign(embedding_dimension, 0.0F);
      for (const auto& values : embeddings) {
        for (std::size_t dimension = 0; dimension < embedding_dimension; ++dimension) {
          state.center[dimension] += values[dimension];
        }
      }
      if (!embeddings.empty()) {
        for (float& value : state.center) {
          value /= static_cast<float>(embeddings.size());
        }
      }

      const auto [detected_num_classes, _] = detail::FitTargetMode(label_values);
      (void)_;
      const int num_classes = embedding_target_mode_ == "regression"
                                  ? 1
                                  : detected_num_classes;
      const int projection_count = num_classes > 1 ? num_classes : 1;
      state.target_projection_weights.reserve(static_cast<std::size_t>(projection_count));
      state.target_output_names.reserve(static_cast<std::size_t>(projection_count));
      for (int projection_index = 0; projection_index < projection_count; ++projection_index) {
        std::vector<float> responses;
        responses.reserve(valid_rows.size());
        for (std::size_t row : valid_rows) {
          responses.push_back(num_classes > 1
                                  ? (static_cast<int>(std::llround(label_values[row])) ==
                                             projection_index
                                         ? 1.0F
                                         : 0.0F)
                                  : label_values[row]);
        }
        const float response_mean = responses.empty()
                                        ? 0.0F
                                        : std::accumulate(responses.begin(), responses.end(), 0.0F) /
                                              static_cast<float>(responses.size());
        std::vector<float> numerator(embedding_dimension, 0.0F);
        std::vector<float> denominator(
            embedding_dimension, static_cast<float>(embedding_target_regularization_));
        for (std::size_t row_index = 0; row_index < embeddings.size(); ++row_index) {
          const float centered_response = responses[row_index] - response_mean;
          for (std::size_t dimension = 0; dimension < embedding_dimension; ++dimension) {
            const float centered_value = embeddings[row_index][dimension] - state.center[dimension];
            numerator[dimension] += centered_value * centered_response;
            denominator[dimension] += centered_value * centered_value;
          }
        }
        std::vector<float> weights(embedding_dimension, 0.0F);
        for (std::size_t dimension = 0; dimension < embedding_dimension; ++dimension) {
          if (denominator[dimension] > 0.0F) {
            weights[dimension] = numerator[dimension] / denominator[dimension];
          }
        }
        state.target_projection_weights.push_back(std::move(weights));
        const std::string output_name =
            num_classes > 1
                ? state.prefix + "_target_projection_class" + std::to_string(projection_index)
                : state.prefix + "_target_projection";
        state.target_output_names.push_back(output_name);
        output_feature_names_.push_back(output_name);
      }
    }
    embedding_states_.push_back(std::move(state));
  }
}

}  // namespace ctboost
