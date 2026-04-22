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
                                             py::object embedding_features,
                                             py::object embedding_stats,
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
      embedding_features_(detail::NormalizeOptionalSequence(std::move(embedding_features))),
      embedding_stats_(detail::VectorToPyList(detail::NormalizeEmbeddingStats(std::move(embedding_stats)))),
                                             ctr_prior_strength_(ctr_prior_strength),
                                             random_seed_(random_seed) {
  if (one_hot_max_size_ < 0) {
    throw std::invalid_argument("one_hot_max_size must be non-negative");
  }
  if (max_cat_threshold_ < 0) {
    throw std::invalid_argument("max_cat_threshold must be non-negative");
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
