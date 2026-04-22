#include "ctboost/feature_pipeline.hpp"

#include "feature_pipeline_internal.hpp"

namespace py = pybind11;

namespace ctboost {

void NativeFeaturePipeline::LoadState(const py::dict& state) {
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
  embedding_features_ =
      detail::NormalizeOptionalSequence(state.contains("embedding_features")
                                            ? py::reinterpret_borrow<py::object>(state["embedding_features"])
                                            : py::none());
  embedding_stats_ = detail::VectorToPyList(
      detail::NormalizeEmbeddingStats(state.contains("embedding_stats")
                                          ? py::reinterpret_borrow<py::object>(state["embedding_stats"])
                                          : py::none()));
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
      combination_states_.push_back(std::move(combination_state));
    }
  }
  RefreshCombinationSourceIndices();

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
      embedding_states_.push_back(std::move(embedding_state));
    }
  }

  training_ctr_columns_.clear();
  text_hash_cache_.clear();
}

}  // namespace ctboost
