#include "module_internal.hpp"

namespace ctboost::bindings {

void BindNativeFeaturePipeline(py::module_& m) {
  py::class_<ctboost::NativeFeaturePipeline>(m, "NativeFeaturePipeline")
      .def(py::init<py::object,
                    bool,
                    int,
                    int,
                    py::object,
                    bool,
                    py::object,
                    py::object,
                    py::object,
                    py::object,
                    int,
                    std::string,
                    py::object,
                    bool,
                    int,
                    int,
                    std::string,
                    py::object,
                    py::object,
                    bool,
                    double,
                    std::string,
                    double,
                    int>(),
           py::arg("cat_features") = py::none(),
           py::arg("ordered_ctr") = false,
           py::arg("one_hot_max_size") = 0,
           py::arg("max_cat_threshold") = 0,
           py::arg("categorical_combinations") = py::none(),
           py::arg("pairwise_categorical_combinations") = false,
           py::arg("simple_ctr") = py::none(),
           py::arg("combinations_ctr") = py::none(),
           py::arg("per_feature_ctr") = py::none(),
           py::arg("text_features") = py::none(),
           py::arg("text_hash_dim") = 64,
           py::arg("text_tokenizer") = "word",
           py::arg("text_ngram_range") = py::none(),
           py::arg("text_lowercase") = true,
           py::arg("text_min_token_count") = 1,
           py::arg("text_max_dictionary_size") = 0,
           py::arg("text_feature_calcer") = "count",
           py::arg("embedding_features") = py::none(),
           py::arg("embedding_stats") = py::none(),
           py::arg("embedding_target_features") = false,
           py::arg("embedding_target_regularization") = 1.0,
           py::arg("embedding_target_mode") = "auto",
           py::arg("ctr_prior_strength") = 1.0,
           py::arg("random_seed") = 0)
      .def("fit_array",
           &ctboost::NativeFeaturePipeline::fit_array,
           py::arg("raw_matrix"),
           py::arg("labels"),
           py::arg("feature_names") = py::none())
      .def("fit_transform_array",
           &ctboost::NativeFeaturePipeline::fit_transform_array,
           py::arg("raw_matrix"),
           py::arg("labels"),
           py::arg("feature_names") = py::none())
      .def("transform_array",
           &ctboost::NativeFeaturePipeline::transform_array,
           py::arg("raw_matrix"),
           py::arg("feature_names") = py::none())
      .def("to_state", &ctboost::NativeFeaturePipeline::to_state)
      .def_static("from_state", &ctboost::NativeFeaturePipeline::FromState);
}

}  // namespace ctboost::bindings
