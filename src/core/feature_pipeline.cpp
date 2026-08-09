#include "ctboost/feature_pipeline.hpp"

#include "feature_pipeline_internal.hpp"

#include <stdexcept>
#include <utility>

namespace py = pybind11;

namespace ctboost {

NativeFeaturePipeline::NativeFeaturePipeline(py::object cat_features,
                                             bool ordered_ctr,
                                             int one_hot_max_size,
                                             int max_cat_threshold,
                                             py::object categorical_combinations,
                                             bool pairwise_categorical_combinations,
                                             py::object simple_ctr,
                                             py::object combinations_ctr,
                                             py::object per_feature_ctr,
                                             py::object text_features,
                                             int text_hash_dim,
                                             std::string text_tokenizer,
                                             py::object text_ngram_range,
                                             bool text_lowercase,
                                             int text_min_token_count,
                                             int text_max_dictionary_size,
                                             std::string text_feature_calcer,
                                             py::object embedding_features,
                                             py::object embedding_stats,
                                             bool embedding_target_features,
                                             double embedding_target_regularization,
                                             std::string embedding_target_mode,
                                             double ctr_prior_strength,
                                             int random_seed)
    : cat_features_(detail::NormalizeOptionalSequence(std::move(cat_features))),
      ordered_ctr_(ordered_ctr),
      one_hot_max_size_(one_hot_max_size),
      max_cat_threshold_(max_cat_threshold),
      categorical_combinations_(detail::NormalizeOptionalCombinations(std::move(categorical_combinations))),
      pairwise_categorical_combinations_(pairwise_categorical_combinations),
      simple_ctr_(detail::NormalizeOptionalCtrTypes(std::move(simple_ctr))),
      combinations_ctr_(detail::NormalizeOptionalCtrTypes(std::move(combinations_ctr))),
      per_feature_ctr_(per_feature_ctr.is_none() ? py::none() : std::move(per_feature_ctr)),
      text_features_(detail::NormalizeOptionalSequence(std::move(text_features))),
      text_hash_dim_(text_hash_dim),
      text_tokenizer_(detail::NormalizeTextTokenizer(std::move(text_tokenizer))),
      text_lowercase_(text_lowercase),
      text_min_token_count_(text_min_token_count),
      text_max_dictionary_size_(text_max_dictionary_size),
      text_feature_calcer_(detail::NormalizeTextFeatureCalcer(std::move(text_feature_calcer))),
      embedding_features_(detail::NormalizeOptionalSequence(std::move(embedding_features))),
      embedding_stats_(detail::VectorToPyList(detail::NormalizeEmbeddingStats(std::move(embedding_stats)))),
      embedding_target_features_(embedding_target_features),
      embedding_target_regularization_(embedding_target_regularization),
      embedding_target_mode_(
          detail::NormalizeEmbeddingTargetMode(std::move(embedding_target_mode))),
      ctr_prior_strength_(ctr_prior_strength),
      random_seed_(random_seed) {
  const auto ngram_range = detail::NormalizeTextNgramRange(std::move(text_ngram_range));
  text_ngram_min_ = ngram_range.first;
  text_ngram_max_ = ngram_range.second;
  if (one_hot_max_size_ < 0) {
    throw std::invalid_argument("one_hot_max_size must be non-negative");
  }
  if (max_cat_threshold_ < 0) {
    throw std::invalid_argument("max_cat_threshold must be non-negative");
  }
  if (text_hash_dim_ <= 0) {
    throw std::invalid_argument("text_hash_dim must be positive");
  }
  if (text_min_token_count_ <= 0) {
    throw std::invalid_argument("text_min_token_count must be positive");
  }
  if (text_max_dictionary_size_ < 0) {
    throw std::invalid_argument("text_max_dictionary_size must be non-negative");
  }
  if (!(embedding_target_regularization_ >= 0.0)) {
    throw std::invalid_argument("embedding_target_regularization must be non-negative");
  }
}

void NativeFeaturePipeline::fit_array(py::array raw_matrix,
                                      py::array_t<float, py::array::forcecast> labels,
                                      py::object feature_names) {
  FitInternal(std::move(raw_matrix), std::move(labels), std::move(feature_names));
}

py::tuple NativeFeaturePipeline::fit_transform_array(
    py::array raw_matrix,
    py::array_t<float, py::array::forcecast> labels,
    py::object feature_names) {
  FitInternal(raw_matrix, std::move(labels), feature_names);
  return TransformInternal(std::move(raw_matrix), std::move(feature_names), true);
}

}  // namespace ctboost
