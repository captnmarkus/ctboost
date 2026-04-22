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
                    py::object,
                    py::object,
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
           py::arg("embedding_features") = py::none(),
           py::arg("embedding_stats") = py::none(),
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
