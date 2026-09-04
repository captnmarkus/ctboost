#include "ctboost/feature_pipeline.hpp"

#include "feature_pipeline_internal.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace py = pybind11;

namespace ctboost {

namespace {

void ValidateSourceIndex(int source_index, int feature_count, const char* state_name) {
  if (source_index < 0 || source_index >= feature_count) {
    throw std::invalid_argument(std::string("serialized ") + state_name +
                                " source index is out of bounds");
  }
}

void ValidateSortedUniqueIndices(const std::vector<int>& indices,
                                 int feature_count,
                                 const char* state_name,
                                 bool allow_empty = true) {
  if (!allow_empty && indices.empty()) {
    throw std::invalid_argument(std::string("serialized ") + state_name +
                                " source indices must not be empty");
  }
  if (!std::is_sorted(indices.begin(), indices.end()) ||
      std::adjacent_find(indices.begin(), indices.end()) != indices.end()) {
    throw std::invalid_argument(std::string("serialized ") + state_name +
                                " source indices must be sorted and unique");
  }
  for (int source_index : indices) {
    ValidateSourceIndex(source_index, feature_count, state_name);
  }
}

void ValidateFinite(float value, const char* field_name) {
  if (!std::isfinite(value)) {
    throw std::invalid_argument(std::string("serialized ") + field_name +
                                " must contain only finite values");
  }
}

void ValidateFinite(double value, const char* field_name) {
  if (!std::isfinite(value)) {
    throw std::invalid_argument(std::string("serialized ") + field_name +
                                " must be finite");
  }
}

std::size_t CheckedAdd(std::size_t lhs, std::size_t rhs, const char* field_name) {
  if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
    throw std::invalid_argument(std::string("serialized ") + field_name +
                                " size overflows the platform limit");
  }
  return lhs + rhs;
}

void ValidateCategoricalCodes(const std::unordered_map<std::string, float>& mapping,
                              float other_value,
                              const char* state_name) {
  std::vector<std::uint8_t> seen(mapping.size(), 0U);
  for (const auto& [_, value] : mapping) {
    (void)_;
    ValidateFinite(value, state_name);
    const double resolved_value = static_cast<double>(value);
    if (resolved_value < 0.0 || std::floor(resolved_value) != resolved_value ||
        resolved_value >= static_cast<double>(mapping.size())) {
      throw std::invalid_argument(std::string("serialized ") + state_name +
                                  " codes must be contiguous non-negative integers");
    }
    const std::size_t code = static_cast<std::size_t>(value);
    if (seen[code] != 0U) {
      throw std::invalid_argument(std::string("serialized ") + state_name +
                                  " codes must be unique");
    }
    seen[code] = 1U;
  }
  ValidateFinite(other_value, state_name);
}

template <typename T>
void RequireUnique(const std::vector<T>& values, const char* field_name) {
  const std::unordered_set<T> unique(values.begin(), values.end());
  if (unique.size() != values.size()) {
    throw std::invalid_argument(std::string("serialized ") + field_name +
                                " must be unique");
  }
}

}  // namespace

void NativeFeaturePipeline::LoadState(const py::dict& state) {
  const int format_version = state.contains("feature_pipeline_format_version")
                                 ? py::cast<int>(state["feature_pipeline_format_version"])
                                 : 1;
  if (format_version < 1 || format_version > detail::kCurrentFeaturePipelineFormatVersion) {
    throw std::invalid_argument("unsupported feature pipeline format version: " +
                                std::to_string(format_version));
  }
  categorical_key_encoding_version_ =
      state.contains("categorical_key_encoding_version")
          ? py::cast<int>(state["categorical_key_encoding_version"])
          : detail::kLegacyCategoricalKeyEncodingVersion;
  if (categorical_key_encoding_version_ != detail::kLegacyCategoricalKeyEncodingVersion &&
      categorical_key_encoding_version_ != detail::kCurrentCategoricalKeyEncodingVersion) {
    throw std::invalid_argument("unsupported categorical key encoding version: " +
                                std::to_string(categorical_key_encoding_version_));
  }
  if (format_version < detail::kCurrentFeaturePipelineFormatVersion &&
      categorical_key_encoding_version_ != detail::kLegacyCategoricalKeyEncodingVersion) {
    throw std::invalid_argument(
        "feature pipeline formats before version 3 require categorical key encoding version 1");
  }

  cat_features_ = detail::NormalizeOptionalSequence(
      state.contains("cat_features") ? py::reinterpret_borrow<py::object>(state["cat_features"])
                                     : py::none());
  ordered_ctr_ = state.contains("ordered_ctr") ? py::cast<bool>(state["ordered_ctr"]) : false;
  one_hot_max_size_ =
      state.contains("one_hot_max_size") ? py::cast<int>(state["one_hot_max_size"]) : 0;
  max_cat_threshold_ =
      state.contains("max_cat_threshold") ? py::cast<int>(state["max_cat_threshold"]) : 0;
  categorical_combinations_ =
      detail::NormalizeOptionalCombinations(state.contains("categorical_combinations")
                                                ? py::reinterpret_borrow<py::object>(
                                                      state["categorical_combinations"])
                                                : py::none());
  pairwise_categorical_combinations_ =
      state.contains("pairwise_categorical_combinations")
          ? py::cast<bool>(state["pairwise_categorical_combinations"])
          : false;
  simple_ctr_ = detail::NormalizeOptionalCtrTypes(
      state.contains("simple_ctr") ? py::reinterpret_borrow<py::object>(state["simple_ctr"]) : py::none());
  combinations_ctr_ = detail::NormalizeOptionalCtrTypes(state.contains("combinations_ctr")
                                                            ? py::reinterpret_borrow<py::object>(
                                                                  state["combinations_ctr"])
                                                            : py::none());
  per_feature_ctr_ =
      state.contains("per_feature_ctr") ? py::reinterpret_borrow<py::object>(state["per_feature_ctr"])
                                        : py::none();
  text_features_ = detail::NormalizeOptionalSequence(
      state.contains("text_features") ? py::reinterpret_borrow<py::object>(state["text_features"])
                                      : py::none());
  text_hash_dim_ = state.contains("text_hash_dim") ? py::cast<int>(state["text_hash_dim"]) : 64;
  text_tokenizer_ = detail::NormalizeTextTokenizer(
      state.contains("text_tokenizer") ? py::cast<std::string>(state["text_tokenizer"])
                                       : std::string("word"));
  const auto ngram_range = detail::NormalizeTextNgramRange(
      state.contains("text_ngram_range")
          ? py::reinterpret_borrow<py::object>(state["text_ngram_range"])
          : py::none());
  text_ngram_min_ = ngram_range.first;
  text_ngram_max_ = ngram_range.second;
  text_lowercase_ = state.contains("text_lowercase")
                        ? py::cast<bool>(state["text_lowercase"])
                        : true;
  text_min_token_count_ = state.contains("text_min_token_count")
                              ? py::cast<int>(state["text_min_token_count"])
                              : 1;
  text_max_dictionary_size_ = state.contains("text_max_dictionary_size")
                                  ? py::cast<int>(state["text_max_dictionary_size"])
                                  : 0;
  text_feature_calcer_ = detail::NormalizeTextFeatureCalcer(
      state.contains("text_feature_calcer")
          ? py::cast<std::string>(state["text_feature_calcer"])
          : std::string("count"));
  embedding_features_ =
      detail::NormalizeOptionalSequence(state.contains("embedding_features")
                                            ? py::reinterpret_borrow<py::object>(state["embedding_features"])
                                            : py::none());
  embedding_stats_ = detail::VectorToPyList(
      detail::NormalizeEmbeddingStats(state.contains("embedding_stats")
                                          ? py::reinterpret_borrow<py::object>(state["embedding_stats"])
                                          : py::none()));
  embedding_target_features_ = state.contains("embedding_target_features")
                                   ? py::cast<bool>(state["embedding_target_features"])
                                   : false;
  embedding_target_regularization_ = state.contains("embedding_target_regularization")
                                         ? py::cast<double>(
                                               state["embedding_target_regularization"])
                                         : 1.0;
  embedding_target_mode_ = detail::NormalizeEmbeddingTargetMode(
      state.contains("embedding_target_mode")
          ? py::cast<std::string>(state["embedding_target_mode"])
          : std::string("auto"));
  ctr_prior_strength_ =
      state.contains("ctr_prior_strength") ? py::cast<double>(state["ctr_prior_strength"]) : 1.0;
  random_seed_ = state.contains("random_seed") ? py::cast<int>(state["random_seed"]) : 0;

  if (state.contains("feature_names_in_") && !state["feature_names_in_"].is_none()) {
    feature_names_in_ = py::cast<std::vector<std::string>>(state["feature_names_in_"]);
  } else {
    feature_names_in_.reset();
  }
  n_features_in_ = (state.contains("n_features_in_") && !state["n_features_in_"].is_none())
                       ? py::cast<int>(state["n_features_in_"])
                       : -1;
  cat_feature_indices_ = state.contains("cat_feature_indices_")
                             ? py::cast<std::vector<int>>(state["cat_feature_indices_"])
                             : std::vector<int>{};
  output_feature_names_ = state.contains("output_feature_names_")
                              ? py::cast<std::vector<std::string>>(state["output_feature_names_"])
                              : std::vector<std::string>{};
  if (categorical_key_encoding_version_ ==
      detail::kCurrentCategoricalKeyEncodingVersion) {
    const std::unordered_set<std::string> unique_output_names(
        output_feature_names_.begin(), output_feature_names_.end());
    if (unique_output_names.size() != output_feature_names_.size()) {
      throw std::invalid_argument(
          "serialized version-2 categorical pipeline requires globally unique output names");
    }
  }
  numeric_indices_ = state.contains("numeric_indices")
                         ? py::cast<std::vector<int>>(state["numeric_indices"])
                         : std::vector<int>{};

  one_hot_states_.clear();
  if (state.contains("one_hot_states")) {
    for (const py::handle item_handle : py::cast<py::list>(state["one_hot_states"])) {
      const py::dict item = item_handle.cast<py::dict>();
      OneHotEncoderState one_hot_state;
      one_hot_state.source_index = py::cast<int>(item["source_index"]);
      one_hot_state.prefix = py::cast<std::string>(item["prefix"]);
      one_hot_state.category_keys = py::cast<std::vector<std::string>>(item["category_keys"]);
      one_hot_state.output_names = py::cast<std::vector<std::string>>(item["output_names"]);
      one_hot_state.has_other_bucket =
          item.contains("has_other_bucket") && py::cast<bool>(item["has_other_bucket"]) ? 1U : 0U;
      if (one_hot_state.category_keys.size() != one_hot_state.output_names.size()) {
        throw std::invalid_argument(
            "serialized one-hot category keys and output names must have matching sizes");
      }
      if (categorical_key_encoding_version_ ==
          detail::kCurrentCategoricalKeyEncodingVersion) {
        const std::unordered_set<std::string> unique_keys(one_hot_state.category_keys.begin(),
                                                           one_hot_state.category_keys.end());
        if (unique_keys.size() != one_hot_state.category_keys.size()) {
          throw std::invalid_argument(
              "serialized version-2 categorical keys must be unique");
        }
        const std::unordered_set<std::string> unique_names(one_hot_state.output_names.begin(),
                                                            one_hot_state.output_names.end());
        if (unique_names.size() != one_hot_state.output_names.size()) {
          throw std::invalid_argument(
              "serialized version-2 categorical keys require unique one-hot output names");
        }
        const std::size_t other_count = static_cast<std::size_t>(std::count(
            one_hot_state.category_keys.begin(),
            one_hot_state.category_keys.end(),
            std::string(detail::kCodec2OtherKey)));
        if (other_count != (one_hot_state.has_other_bucket != 0U ? 1U : 0U)) {
          throw std::invalid_argument(
              "serialized version-2 one-hot other-bucket marker is inconsistent");
        }
      }
      one_hot_states_.push_back(std::move(one_hot_state));
    }
  }

  categorical_states_.clear();
  if (state.contains("categorical_states")) {
    for (const py::handle item_handle : py::cast<py::list>(state["categorical_states"])) {
      const py::dict item = item_handle.cast<py::dict>();
      CategoricalEncoderState categorical_state;
      categorical_state.source_index = py::cast<int>(item["source_index"]);
      categorical_state.output_name = py::cast<std::string>(item["output_name"]);
      categorical_state.has_other_bucket =
          item.contains("has_other_bucket") && py::cast<bool>(item["has_other_bucket"]) ? 1U : 0U;
      categorical_state.other_value =
          item.contains("other_value") ? py::cast<float>(item["other_value"]) : 0.0F;
      for (const auto& mapping_item : py::cast<py::dict>(item["mapping"])) {
        categorical_state.mapping.emplace(py::cast<std::string>(mapping_item.first),
                                          py::cast<float>(mapping_item.second));
      }
      if (categorical_key_encoding_version_ ==
          detail::kCurrentCategoricalKeyEncodingVersion) {
        const auto other_it = categorical_state.mapping.find(detail::kCodec2OtherKey);
        if ((other_it != categorical_state.mapping.end()) !=
            (categorical_state.has_other_bucket != 0U)) {
          throw std::invalid_argument(
              "serialized version-2 categorical other-bucket marker is inconsistent");
        }
        if (other_it != categorical_state.mapping.end() &&
            other_it->second != categorical_state.other_value) {
          throw std::invalid_argument(
              "serialized version-2 categorical other-bucket value is inconsistent");
        }
      }
      categorical_states_.push_back(std::move(categorical_state));
    }
  }

  combination_states_.clear();
  if (state.contains("combination_states")) {
    for (const py::handle item_handle : py::cast<py::list>(state["combination_states"])) {
      const py::dict item = item_handle.cast<py::dict>();
      CategoricalEncoderState combination_state;
      combination_state.output_name = py::cast<std::string>(item["output_name"]);
      combination_state.has_other_bucket =
          item.contains("has_other_bucket") && py::cast<bool>(item["has_other_bucket"]) ? 1U : 0U;
      combination_state.other_value =
          item.contains("other_value") ? py::cast<float>(item["other_value"]) : 0.0F;
      for (const auto& mapping_item : py::cast<py::dict>(item["mapping"])) {
        combination_state.mapping.emplace(py::cast<std::string>(mapping_item.first),
                                          py::cast<float>(mapping_item.second));
      }
      if (categorical_key_encoding_version_ ==
          detail::kCurrentCategoricalKeyEncodingVersion) {
        const auto other_it = combination_state.mapping.find(detail::kCodec2OtherKey);
        if ((other_it != combination_state.mapping.end()) !=
            (combination_state.has_other_bucket != 0U)) {
          throw std::invalid_argument(
              "serialized version-2 combination other-bucket marker is inconsistent");
        }
        if (other_it != combination_state.mapping.end() &&
            other_it->second != combination_state.other_value) {
          throw std::invalid_argument(
              "serialized version-2 combination other-bucket value is inconsistent");
        }
      }
      combination_states_.push_back(std::move(combination_state));
    }
  }
  ctr_states_.clear();
  if (state.contains("ctr_states")) {
    for (const py::handle item_handle : py::cast<py::list>(state["ctr_states"])) {
      const py::dict item = item_handle.cast<py::dict>();
      CtrState ctr_state;
      ctr_state.source_indices = py::cast<std::vector<int>>(item["source_indices"]);
      ctr_state.output_names = py::cast<std::vector<std::string>>(item["output_names"]);
      ctr_state.ctr_type =
          item.contains("ctr_type") ? py::cast<std::string>(item["ctr_type"]) : std::string("Mean");
      ctr_state.prior_values = py::cast<std::vector<float>>(item["prior_values"]);
      ctr_state.total_rows =
          item.contains("total_rows") ? py::cast<std::size_t>(item["total_rows"]) : 0U;
      for (const auto& count_item : py::cast<py::dict>(item["total_counts"])) {
        ctr_state.total_counts.emplace(py::cast<std::string>(count_item.first),
                                       py::cast<int>(count_item.second));
      }
      for (const auto& sum_item : py::cast<py::dict>(item["total_sums"])) {
        ctr_state.total_sums.emplace(py::cast<std::string>(sum_item.first),
                                     py::cast<std::vector<float>>(sum_item.second));
      }
      ctr_states_.push_back(std::move(ctr_state));
    }
  }

  text_states_.clear();
  if (state.contains("text_states")) {
    for (const py::handle item_handle : py::cast<py::list>(state["text_states"])) {
      const py::dict item = item_handle.cast<py::dict>();
      TextState text_state;
      text_state.source_index = py::cast<int>(item["source_index"]);
      text_state.prefix = py::cast<std::string>(item["prefix"]);
      text_state.output_dim =
          item.contains("output_dim") ? py::cast<int>(item["output_dim"]) : text_hash_dim_;
      text_state.uses_dictionary =
          item.contains("uses_dictionary") && py::cast<bool>(item["uses_dictionary"]) ? 1U : 0U;
      text_state.filters_tokens =
          item.contains("filters_tokens") && py::cast<bool>(item["filters_tokens"]) ? 1U : 0U;
      text_state.vocabulary = item.contains("vocabulary")
                                  ? py::cast<std::vector<std::string>>(item["vocabulary"])
                                  : std::vector<std::string>{};
      for (std::size_t index = 0; index < text_state.vocabulary.size(); ++index) {
        text_state.vocabulary_indices.emplace(
            text_state.vocabulary[index],
            text_state.uses_dictionary != 0U ? static_cast<int>(index) : 0);
      }
      text_state.idf_values = item.contains("idf_values")
                                  ? py::cast<std::vector<float>>(item["idf_values"])
                                  : std::vector<float>{};
      text_states_.push_back(std::move(text_state));
    }
  }

  embedding_states_.clear();
  if (state.contains("embedding_states")) {
    for (const py::handle item_handle : py::cast<py::list>(state["embedding_states"])) {
      const py::dict item = item_handle.cast<py::dict>();
      EmbeddingState embedding_state;
      embedding_state.source_index = py::cast<int>(item["source_index"]);
      embedding_state.prefix = py::cast<std::string>(item["prefix"]);
      embedding_state.stats = py::cast<std::vector<std::string>>(item["stats"]);
      embedding_state.center = item.contains("center")
                                   ? py::cast<std::vector<float>>(item["center"])
                                   : std::vector<float>{};
      if (item.contains("target_projection_weights")) {
        for (const py::handle weights : py::cast<py::list>(item["target_projection_weights"])) {
          embedding_state.target_projection_weights.push_back(
              py::cast<std::vector<float>>(weights));
        }
      }
      embedding_state.target_output_names = item.contains("target_output_names")
                                                ? py::cast<std::vector<std::string>>(
                                                      item["target_output_names"])
                                                : std::vector<std::string>{};
      embedding_states_.push_back(std::move(embedding_state));
    }
  }

  training_ctr_columns_.clear();
  allocated_output_feature_names_.clear();
  text_hash_cache_.clear();

  // Bound all O(n_features) and pairwise expansion work by compact state that
  // has already been parsed from the artifact. This rejects tiny allocation
  // bombs before reserve/looping over an attacker-controlled feature count.
  if (n_features_in_ >= 0) {
    if (feature_names_in_.has_value() &&
        feature_names_in_->size() != static_cast<std::size_t>(n_features_in_)) {
      throw std::invalid_argument(
          "serialized feature names must match the input feature count");
    }
    ValidateSortedUniqueIndices(numeric_indices_, n_features_in_, "numeric");
    const std::vector<int> text_sources = ResolveIndices(text_features_);
    const std::vector<int> embedding_sources = ResolveIndices(embedding_features_);
    std::unordered_set<int> occupied(numeric_indices_.begin(), numeric_indices_.end());
    for (int source_index : text_sources) {
      if (!occupied.insert(source_index).second) {
        throw std::invalid_argument(
            "serialized numeric and text feature sources must be disjoint");
      }
    }
    for (int source_index : embedding_sources) {
      if (!occupied.insert(source_index).second) {
        throw std::invalid_argument(
            "serialized numeric, text, and embedding feature sources must be disjoint");
      }
    }
    if (occupied.size() != static_cast<std::size_t>(n_features_in_)) {
      throw std::invalid_argument(
          "serialized feature sources do not cover the input feature count");
    }
    if (pairwise_categorical_combinations_) {
      const std::size_t categorical_count = ResolveIndices(cat_features_).size();
      if (categorical_count > 1U) {
        if (categorical_count - 1U >
            std::numeric_limits<std::size_t>::max() / categorical_count) {
          throw std::invalid_argument(
              "serialized pairwise categorical combination count overflows");
        }
        const std::size_t pair_count =
            categorical_count * (categorical_count - 1U) / 2U;
        if (pair_count > combination_states_.size()) {
          throw std::invalid_argument(
              "serialized pairwise categorical combinations exceed fitted state");
        }
      }
    }
  }
  RefreshCombinationSourceIndices();
  ValidateFittedState();
}

void NativeFeaturePipeline::ValidateFittedState() const {
  if (n_features_in_ < -1) {
    throw std::invalid_argument("serialized input feature count must be non-negative");
  }
  if (n_features_in_ == -1) {
    if (feature_names_in_.has_value() || !cat_feature_indices_.empty() ||
        !output_feature_names_.empty() || !numeric_indices_.empty() ||
        !one_hot_states_.empty() || !categorical_states_.empty() ||
        !combination_states_.empty() || !ctr_states_.empty() ||
        !text_states_.empty() || !embedding_states_.empty() ||
        !training_ctr_columns_.empty()) {
      throw std::invalid_argument(
          "serialized unfitted feature pipeline contains fitted state");
    }
    return;
  }

  if (feature_names_in_.has_value()) {
    if (feature_names_in_->size() != static_cast<std::size_t>(n_features_in_)) {
      throw std::invalid_argument(
          "serialized feature names must match the input feature count");
    }
  }
  if (one_hot_max_size_ < 0 || max_cat_threshold_ < 0) {
    throw std::invalid_argument(
        "serialized categorical thresholds must be non-negative");
  }
  if (text_hash_dim_ <= 0 || text_min_token_count_ <= 0 ||
      text_max_dictionary_size_ < 0) {
    throw std::invalid_argument("serialized text configuration is invalid");
  }
  ValidateFinite(embedding_target_regularization_,
                 "embedding target regularization");
  ValidateFinite(ctr_prior_strength_, "CTR prior strength");
  if (embedding_target_regularization_ < 0.0 || ctr_prior_strength_ < 0.0 ||
      !std::isfinite(static_cast<float>(embedding_target_regularization_)) ||
      !std::isfinite(static_cast<float>(ctr_prior_strength_))) {
    throw std::invalid_argument(
        "serialized regularization strengths must be finite non-negative float values");
  }

  const std::vector<int> categorical_sources = ResolveIndices(cat_features_);
  const std::vector<int> text_sources = ResolveIndices(text_features_);
  const std::vector<int> embedding_sources = ResolveIndices(embedding_features_);
  std::unordered_set<int> reserved_sources(text_sources.begin(), text_sources.end());
  for (int source_index : embedding_sources) {
    if (!reserved_sources.insert(source_index).second) {
      throw std::invalid_argument(
          "serialized text and embedding feature sources must not overlap");
    }
  }

  std::vector<int> expected_numeric_indices;
  expected_numeric_indices.reserve(static_cast<std::size_t>(n_features_in_));
  for (int feature_index = 0; feature_index < n_features_in_; ++feature_index) {
    if (reserved_sources.find(feature_index) == reserved_sources.end()) {
      expected_numeric_indices.push_back(feature_index);
    }
  }
  if (numeric_indices_ != expected_numeric_indices) {
    throw std::invalid_argument(
        "serialized numeric feature indices do not match text/embedding reservations");
  }

  std::unordered_map<int, const OneHotEncoderState*> one_hot_by_source;
  for (const auto& state : one_hot_states_) {
    ValidateSourceIndex(state.source_index, n_features_in_, "one-hot");
    if (!std::binary_search(numeric_indices_.begin(), numeric_indices_.end(),
                            state.source_index) ||
        !one_hot_by_source.emplace(state.source_index, &state).second) {
      throw std::invalid_argument(
          "serialized one-hot source must be a unique numeric input feature");
    }
    if (state.category_keys.size() != state.output_names.size()) {
      throw std::invalid_argument(
          "serialized one-hot category keys and output names must have matching sizes");
    }
    if (categorical_key_encoding_version_ ==
        detail::kCurrentCategoricalKeyEncodingVersion) {
      RequireUnique(state.category_keys, "one-hot category keys");
      RequireUnique(state.output_names, "one-hot output names");
    }
  }

  std::unordered_map<int, const CategoricalEncoderState*> categorical_by_source;
  for (const auto& state : categorical_states_) {
    ValidateSourceIndex(state.source_index, n_features_in_, "categorical");
    if (!std::binary_search(numeric_indices_.begin(), numeric_indices_.end(),
                            state.source_index) ||
        one_hot_by_source.find(state.source_index) != one_hot_by_source.end() ||
        !categorical_by_source.emplace(state.source_index, &state).second) {
      throw std::invalid_argument(
          "serialized categorical source must be unique and disjoint from one-hot sources");
    }
    if (state.output_name.empty()) {
      throw std::invalid_argument(
          "serialized categorical output name must not be empty");
    }
    ValidateCategoricalCodes(state.mapping, state.other_value,
                             "categorical mapping");
  }

  std::unordered_set<int> expected_categorical_sources;
  for (int source_index : categorical_sources) {
    if (std::binary_search(numeric_indices_.begin(), numeric_indices_.end(),
                           source_index)) {
      expected_categorical_sources.insert(source_index);
    }
  }
  std::unordered_set<int> observed_categorical_sources;
  for (const auto& [source_index, _] : one_hot_by_source) {
    (void)_;
    observed_categorical_sources.insert(source_index);
  }
  for (const auto& [source_index, _] : categorical_by_source) {
    (void)_;
    observed_categorical_sources.insert(source_index);
  }
  if (observed_categorical_sources != expected_categorical_sources) {
    throw std::invalid_argument(
        "serialized categorical encoders do not match categorical feature selectors");
  }

  if (combination_states_.size() != combination_source_indices_.size()) {
    throw std::invalid_argument(
        "serialized categorical combination state count is inconsistent");
  }
  for (std::size_t index = 0; index < combination_states_.size(); ++index) {
    ValidateSortedUniqueIndices(combination_source_indices_[index], n_features_in_,
                                "categorical combination", false);
    const auto& state = combination_states_[index];
    if (state.output_name.empty()) {
      throw std::invalid_argument(
          "serialized combination output name must not be empty");
    }
    ValidateCategoricalCodes(state.mapping, state.other_value,
                             "combination mapping");
  }

  std::unordered_set<std::string> ctr_identities;
  std::optional<std::size_t> fitted_row_count;
  std::size_t ctr_output_count = 0U;
  for (const auto& state : ctr_states_) {
    ValidateSortedUniqueIndices(state.source_indices, n_features_in_, "CTR", false);
    const bool scalar_source =
        state.source_indices.size() == 1U &&
        categorical_by_source.find(state.source_indices[0]) != categorical_by_source.end();
    const bool combination_source =
        std::find(combination_source_indices_.begin(),
                  combination_source_indices_.end(), state.source_indices) !=
        combination_source_indices_.end();
    if (!scalar_source && !combination_source) {
      throw std::invalid_argument(
          "serialized CTR source does not reference a fitted categorical encoder");
    }
    if (state.ctr_type != "Mean" && state.ctr_type != "Frequency") {
      throw std::invalid_argument("serialized CTR type is unsupported");
    }
    std::string identity = state.ctr_type;
    for (int source_index : state.source_indices) {
      identity += ":" + std::to_string(source_index);
    }
    if (!ctr_identities.insert(std::move(identity)).second) {
      throw std::invalid_argument("serialized CTR states must be unique");
    }
    if (state.output_names.empty()) {
      throw std::invalid_argument("serialized CTR output names must not be empty");
    }
    RequireUnique(state.output_names, "CTR output names");
    if (state.prior_values.empty()) {
      throw std::invalid_argument("serialized CTR priors must not be empty");
    }
    for (float value : state.prior_values) {
      ValidateFinite(value, "CTR priors");
    }
    if (state.ctr_type == "Mean") {
      if (state.output_names.size() != state.prior_values.size() ||
          state.total_sums.size() != state.total_counts.size()) {
        throw std::invalid_argument(
            "serialized mean CTR dimensions are inconsistent");
      }
    } else if (state.output_names.size() != 1U || !state.total_sums.empty()) {
      throw std::invalid_argument(
          "serialized frequency CTR dimensions are inconsistent");
    }
    std::size_t total_count = 0U;
    for (const auto& [key, count] : state.total_counts) {
      if (count < 0 || static_cast<std::size_t>(count) > state.total_rows) {
        throw std::invalid_argument("serialized CTR counts must be non-negative");
      }
      total_count = CheckedAdd(total_count, static_cast<std::size_t>(count),
                               "CTR count");
      if (state.ctr_type == "Mean") {
        const auto sums_it = state.total_sums.find(key);
        if (sums_it == state.total_sums.end() ||
            sums_it->second.size() != state.output_names.size()) {
          throw std::invalid_argument(
              "serialized mean CTR sum dimensions are inconsistent");
        }
        for (float value : sums_it->second) {
          ValidateFinite(value, "CTR sums");
        }
      }
    }
    if (total_count != state.total_rows) {
      throw std::invalid_argument(
          "serialized CTR counts do not match the fitted row count");
    }
    if (!fitted_row_count.has_value()) {
      fitted_row_count = state.total_rows;
    } else if (*fitted_row_count != state.total_rows) {
      throw std::invalid_argument(
          "serialized CTR states disagree on the fitted row count");
    }
    ctr_output_count = CheckedAdd(ctr_output_count, state.output_names.size(),
                                  "CTR output");
  }

  std::vector<int> observed_text_sources;
  observed_text_sources.reserve(text_states_.size());
  for (const auto& state : text_states_) {
    ValidateSourceIndex(state.source_index, n_features_in_, "text");
    observed_text_sources.push_back(state.source_index);
    if (state.output_dim < 0) {
      throw std::invalid_argument(
          "serialized text output dimension must be non-negative");
    }
    RequireUnique(state.vocabulary, "text vocabulary");
    if (state.uses_dictionary != 0U) {
      if (text_max_dictionary_size_ <= 0 ||
          state.output_dim != static_cast<int>(state.vocabulary.size()) ||
          state.vocabulary.size() >
              static_cast<std::size_t>(text_max_dictionary_size_)) {
        throw std::invalid_argument(
            "serialized text dictionary dimensions are inconsistent");
      }
    } else {
      if (text_max_dictionary_size_ != 0 || state.output_dim != text_hash_dim_) {
        throw std::invalid_argument(
            "serialized text hash dimensions are inconsistent");
      }
      if (state.filters_tokens == 0U && !state.vocabulary.empty()) {
        throw std::invalid_argument(
            "serialized unfiltered text hash state must not contain a vocabulary");
      }
    }
    if ((state.filters_tokens != 0U) != (text_min_token_count_ > 1)) {
      throw std::invalid_argument(
          "serialized text token filter flag is inconsistent");
    }
    const std::size_t output_dim = static_cast<std::size_t>(state.output_dim);
    if (text_feature_calcer_ == "tfidf") {
      if (state.idf_values.size() != output_dim) {
        throw std::invalid_argument(
            "serialized TF-IDF vector dimension is inconsistent");
      }
      for (float value : state.idf_values) {
        ValidateFinite(value, "TF-IDF values");
        if (value <= 0.0F) {
          throw std::invalid_argument(
              "serialized TF-IDF values must be positive");
        }
      }
    } else if (!state.idf_values.empty()) {
      throw std::invalid_argument(
          "serialized non-TF-IDF text state must not contain IDF values");
    }
  }
  if (observed_text_sources != text_sources) {
    throw std::invalid_argument(
        "serialized text states do not match text feature selectors");
  }

  const std::vector<std::string> configured_embedding_stats =
      detail::NormalizeEmbeddingStats(embedding_stats_);
  std::vector<int> observed_embedding_sources;
  observed_embedding_sources.reserve(embedding_states_.size());
  for (const auto& state : embedding_states_) {
    ValidateSourceIndex(state.source_index, n_features_in_, "embedding");
    observed_embedding_sources.push_back(state.source_index);
    if (state.stats != configured_embedding_stats) {
      throw std::invalid_argument(
          "serialized embedding stats do not match the fitted configuration");
    }
    for (float value : state.center) {
      ValidateFinite(value, "embedding center");
    }
    if (state.target_projection_weights.size() !=
        state.target_output_names.size()) {
      throw std::invalid_argument(
          "serialized embedding projection dimensions are inconsistent");
    }
    for (const auto& weights : state.target_projection_weights) {
      if (weights.size() != state.center.size()) {
        throw std::invalid_argument(
            "serialized embedding projection width is inconsistent");
      }
      for (float value : weights) {
        ValidateFinite(value, "embedding projection weights");
      }
    }
    if (!embedding_target_features_ &&
        (!state.center.empty() || !state.target_projection_weights.empty() ||
         !state.target_output_names.empty())) {
      throw std::invalid_argument(
          "serialized embedding target state is disabled by configuration");
    }
    if (embedding_target_features_ && state.target_projection_weights.empty()) {
      throw std::invalid_argument(
          "serialized embedding target state has no projections");
    }
    if (embedding_target_mode_ == "regression" &&
        state.target_projection_weights.size() != 1U) {
      throw std::invalid_argument(
          "serialized regression embedding state must have one projection");
    }
  }
  if (observed_embedding_sources != embedding_sources) {
    throw std::invalid_argument(
        "serialized embedding states do not match embedding feature selectors");
  }

  std::size_t output_count = 0U;
  std::vector<int> expected_categorical_output_indices;
  auto require_output_name = [this](std::size_t index, const std::string& expected,
                                    const char* state_name) {
    if (index >= output_feature_names_.size() ||
        output_feature_names_[index] != expected) {
      throw std::invalid_argument(std::string("serialized ") + state_name +
                                  " output layout is inconsistent");
    }
  };
  for (int feature_index : numeric_indices_) {
    const auto one_hot_it = one_hot_by_source.find(feature_index);
    if (one_hot_it != one_hot_by_source.end()) {
      for (const std::string& output_name : one_hot_it->second->output_names) {
        require_output_name(output_count, output_name, "one-hot");
        output_count = CheckedAdd(output_count, 1U, "feature-pipeline output");
      }
      continue;
    }
    const auto categorical_it = categorical_by_source.find(feature_index);
    if (categorical_it != categorical_by_source.end()) {
      if (output_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
            "serialized categorical output index exceeds the supported range");
      }
      expected_categorical_output_indices.push_back(
          static_cast<int>(output_count));
      require_output_name(output_count, categorical_it->second->output_name,
                          "categorical");
    }
    output_count = CheckedAdd(output_count, 1U, "feature-pipeline output");
  }
  for (const auto& state : combination_states_) {
    if (output_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      throw std::invalid_argument(
          "serialized categorical output index exceeds the supported range");
    }
    expected_categorical_output_indices.push_back(static_cast<int>(output_count));
    require_output_name(output_count, state.output_name, "combination");
    output_count = CheckedAdd(output_count, 1U, "feature-pipeline output");
  }
  for (const auto& state : ctr_states_) {
    for (const std::string& output_name : state.output_names) {
      require_output_name(output_count, output_name, "CTR");
      output_count = CheckedAdd(output_count, 1U, "feature-pipeline output");
    }
  }
  for (const auto& state : text_states_) {
    output_count = CheckedAdd(output_count,
                              static_cast<std::size_t>(state.output_dim),
                              "text output");
  }
  for (const auto& state : embedding_states_) {
    output_count = CheckedAdd(output_count, state.stats.size(),
                              "embedding stats output");
    for (std::size_t index = 0; index < state.target_output_names.size(); ++index) {
      const std::size_t target_index = CheckedAdd(
          output_count, index, "embedding target output");
      require_output_name(target_index, state.target_output_names[index],
                          "embedding target");
    }
    output_count = CheckedAdd(output_count, state.target_output_names.size(),
                              "embedding target output");
  }
  if (output_count != output_feature_names_.size()) {
    throw std::invalid_argument(
        "serialized feature-pipeline output dimension is inconsistent");
  }
  if (expected_categorical_output_indices != cat_feature_indices_) {
    throw std::invalid_argument(
        "serialized categorical output indices are inconsistent");
  }
  if (categorical_key_encoding_version_ ==
      detail::kCurrentCategoricalKeyEncodingVersion) {
    RequireUnique(output_feature_names_, "feature-pipeline output names");
  }

  if (!training_ctr_columns_.empty()) {
    if (training_ctr_columns_.size() != ctr_output_count ||
        !fitted_row_count.has_value()) {
      throw std::invalid_argument(
          "fitted CTR training columns have inconsistent dimensions");
    }
    for (const auto& column : training_ctr_columns_) {
      if (column.size() != *fitted_row_count) {
        throw std::invalid_argument(
            "fitted CTR training column row count is inconsistent");
      }
      for (float value : column) {
        ValidateFinite(value, "CTR training values");
      }
    }
  }
}

}  // namespace ctboost
