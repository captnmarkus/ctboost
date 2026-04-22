#include "ctboost/feature_pipeline.hpp"

#include "feature_pipeline_internal.hpp"

namespace py = pybind11;

namespace ctboost {

py::dict NativeFeaturePipeline::to_state() const {
  py::dict state;
  state["cat_features"] = cat_features_;
  state["ordered_ctr"] = ordered_ctr_;
  state["one_hot_max_size"] = one_hot_max_size_;
  state["max_cat_threshold"] = max_cat_threshold_;
  state["categorical_combinations"] = categorical_combinations_;
  state["pairwise_categorical_combinations"] = pairwise_categorical_combinations_;
  state["simple_ctr"] = simple_ctr_;
  state["combinations_ctr"] = combinations_ctr_;
  state["per_feature_ctr"] = per_feature_ctr_;
  state["text_features"] = text_features_;
  state["text_hash_dim"] = text_hash_dim_;
  state["embedding_features"] = embedding_features_;
  state["embedding_stats"] = embedding_stats_;
  state["ctr_prior_strength"] = ctr_prior_strength_;
  state["random_seed"] = random_seed_;
  state["feature_names_in_"] =
      feature_names_in_.has_value() ? detail::VectorToPyList(feature_names_in_.value()) : py::none();
  state["n_features_in_"] = n_features_in_ >= 0 ? py::int_(n_features_in_) : py::none();
  state["cat_feature_indices_"] = detail::VectorToPyList(cat_feature_indices_);
  state["output_feature_names_"] = detail::VectorToPyList(output_feature_names_);
  state["numeric_indices"] = detail::VectorToPyList(numeric_indices_);

  py::list one_hot_states;
  for (const auto& one_hot_state : one_hot_states_) {
    py::dict item;
    item["source_index"] = one_hot_state.source_index;
    item["prefix"] = one_hot_state.prefix;
    item["category_keys"] = detail::VectorToPyList(one_hot_state.category_keys);
    item["output_names"] = detail::VectorToPyList(one_hot_state.output_names);
    item["has_other_bucket"] = one_hot_state.has_other_bucket != 0U;
    one_hot_states.append(std::move(item));
  }
  state["one_hot_states"] = std::move(one_hot_states);

  py::list categorical_states;
  for (const auto& categorical_state : categorical_states_) {
    py::dict item;
    item["source_index"] = categorical_state.source_index;
    item["output_name"] = categorical_state.output_name;
    item["has_other_bucket"] = categorical_state.has_other_bucket != 0U;
    item["other_value"] = categorical_state.other_value;
    py::dict mapping;
    for (const auto& [key, value] : categorical_state.mapping) {
      mapping[py::str(key)] = value;
    }
    item["mapping"] = std::move(mapping);
    categorical_states.append(std::move(item));
  }
  state["categorical_states"] = std::move(categorical_states);

  py::list combination_states;
  for (const auto& combination_state : combination_states_) {
    py::dict item;
    item["output_name"] = combination_state.output_name;
    item["has_other_bucket"] = combination_state.has_other_bucket != 0U;
    item["other_value"] = combination_state.other_value;
    py::dict mapping;
    for (const auto& [key, value] : combination_state.mapping) {
      mapping[py::str(key)] = value;
    }
    item["mapping"] = std::move(mapping);
    combination_states.append(std::move(item));
  }
  state["combination_states"] = std::move(combination_states);

  py::list ctr_states;
  for (const auto& ctr_state : ctr_states_) {
    py::dict item;
    item["source_indices"] = detail::VectorToPyList(ctr_state.source_indices);
    item["output_names"] = detail::VectorToPyList(ctr_state.output_names);
    item["ctr_type"] = ctr_state.ctr_type;
    item["prior_values"] = detail::VectorToPyList(ctr_state.prior_values);
    item["total_rows"] = ctr_state.total_rows;
    py::dict total_counts;
    for (const auto& [key, value] : ctr_state.total_counts) {
      total_counts[py::str(key)] = value;
    }
    item["total_counts"] = std::move(total_counts);
    py::dict total_sums;
    for (const auto& [key, values] : ctr_state.total_sums) {
      total_sums[py::str(key)] = detail::VectorToPyList(values);
    }
    item["total_sums"] = std::move(total_sums);
    ctr_states.append(std::move(item));
  }
  state["ctr_states"] = std::move(ctr_states);

  py::list text_states;
  for (const auto& text_state : text_states_) {
    py::dict item;
    item["source_index"] = text_state.source_index;
    item["prefix"] = text_state.prefix;
    text_states.append(std::move(item));
  }
  state["text_states"] = std::move(text_states);

  py::list embedding_states;
  for (const auto& embedding_state : embedding_states_) {
    py::dict item;
    item["source_index"] = embedding_state.source_index;
    item["prefix"] = embedding_state.prefix;
    item["stats"] = detail::VectorToPyList(embedding_state.stats);
    embedding_states.append(std::move(item));
  }
  state["embedding_states"] = std::move(embedding_states);
  return state;
}

NativeFeaturePipeline NativeFeaturePipeline::FromState(const py::dict& state) {
  NativeFeaturePipeline pipeline;
  pipeline.LoadState(state);
  return pipeline;
}

}  // namespace ctboost
