#include "module_internal.hpp"

#include <cstring>

namespace ctboost::bindings {
namespace {

py::array_t<float, py::array::forcecast> ResolvePoolWeights(
    py::array_t<float, py::array::forcecast> label,
    py::object weight) {
  if (!weight.is_none()) {
    return weight.cast<py::array_t<float, py::array::forcecast>>();
  }

  const py::buffer_info label_info = label.request();
  py::array_t<float> ones(static_cast<py::ssize_t>(label_info.shape[0]));
  auto mutable_weights = ones.mutable_unchecked<1>();
  for (py::ssize_t index = 0; index < label_info.shape[0]; ++index) {
    mutable_weights(index) = 1.0F;
  }
  return ones;
}

}  // namespace

void BindPool(py::module_& m) {
  py::class_<ctboost::Pool>(m, "Pool")
      .def(py::init([](py::array_t<float, py::array::forcecast> data,
                       py::array_t<float, py::array::forcecast> label,
                       std::vector<int> cat_features,
                       py::object weight,
                       py::object group_id,
                       py::object group_weight,
                       py::object subgroup_id,
                       py::object baseline,
                       py::object pairs,
                       py::object pairs_weight) {
             py::array_t<float, py::array::forcecast> resolved_weight =
                 ResolvePoolWeights(label, weight);
             py::array_t<std::int64_t, py::array::forcecast> resolved_group_id;
             if (!group_id.is_none()) {
               resolved_group_id = group_id.cast<py::array_t<std::int64_t, py::array::forcecast>>();
             }
             py::array_t<float, py::array::forcecast> resolved_group_weight;
             if (!group_weight.is_none()) {
               resolved_group_weight = group_weight.cast<py::array_t<float, py::array::forcecast>>();
             }
             py::array_t<std::int64_t, py::array::forcecast> resolved_subgroup_id;
             if (!subgroup_id.is_none()) {
               resolved_subgroup_id =
                   subgroup_id.cast<py::array_t<std::int64_t, py::array::forcecast>>();
             }
             py::array_t<float, py::array::forcecast> resolved_baseline;
             if (!baseline.is_none()) {
               resolved_baseline = baseline.cast<py::array_t<float, py::array::forcecast>>();
             }
             py::array_t<std::int64_t, py::array::forcecast> resolved_pairs;
             if (!pairs.is_none()) {
               resolved_pairs = pairs.cast<py::array_t<std::int64_t, py::array::forcecast>>();
             }
             py::array_t<float, py::array::forcecast> resolved_pairs_weight;
             if (!pairs_weight.is_none()) {
               resolved_pairs_weight = pairs_weight.cast<py::array_t<float, py::array::forcecast>>();
             }
             return ctboost::Pool(data,
                                  label,
                                  std::move(cat_features),
                                  resolved_weight,
                                  resolved_group_id,
                                  resolved_group_weight,
                                  resolved_subgroup_id,
                                  resolved_baseline,
                                  resolved_pairs,
                                  resolved_pairs_weight);
           }),
           py::arg("data"),
           py::arg("label"),
           py::arg("cat_features") = std::vector<int>{},
           py::arg("weight") = py::none(),
           py::arg("group_id") = py::none(),
           py::arg("group_weight") = py::none(),
           py::arg("subgroup_id") = py::none(),
           py::arg("baseline") = py::none(),
           py::arg("pairs") = py::none(),
           py::arg("pairs_weight") = py::none())
      .def("num_rows", &ctboost::Pool::num_rows)
      .def("num_cols", &ctboost::Pool::num_cols)
      .def("feature_data", [](const ctboost::Pool& pool) {
        return VectorToArray(pool.feature_data());
      })
      .def("label", [](const ctboost::Pool& pool) {
        return VectorToArray(pool.labels());
      })
      .def("weight", [](const ctboost::Pool& pool) {
        return VectorToArray(pool.weights());
      })
      .def("group_id", [](const ctboost::Pool& pool) -> py::object {
        if (!pool.has_group_ids()) {
          return py::none();
        }
        py::array_t<std::int64_t> result(pool.group_ids().size());
        if (!pool.group_ids().empty()) {
          std::memcpy(
              result.mutable_data(), pool.group_ids().data(), pool.group_ids().size() * sizeof(std::int64_t));
        }
        return result;
      })
      .def("cat_features", [](const ctboost::Pool& pool) {
        return pool.cat_features();
      })
      .def("is_sparse", &ctboost::Pool::is_sparse)
      .def_static("from_csc",
                  [](py::array_t<float, py::array::forcecast> sparse_data,
                     py::array_t<std::int64_t, py::array::forcecast> sparse_indices,
                     py::array_t<std::int64_t, py::array::forcecast> sparse_indptr,
                     std::size_t num_rows,
                     std::size_t num_cols,
                     py::array_t<float, py::array::forcecast> label,
                     std::vector<int> cat_features,
                     py::object weight,
                     py::object group_id,
                     py::object group_weight,
                     py::object subgroup_id,
                     py::object baseline,
                     py::object pairs,
                     py::object pairs_weight) {
                    py::array_t<float, py::array::forcecast> resolved_weight =
                        ResolvePoolWeights(label, weight);
                    py::array_t<std::int64_t, py::array::forcecast> resolved_group_id;
                    if (!group_id.is_none()) {
                      resolved_group_id =
                          group_id.cast<py::array_t<std::int64_t, py::array::forcecast>>();
                    }
                    py::array_t<float, py::array::forcecast> resolved_group_weight;
                    if (!group_weight.is_none()) {
                      resolved_group_weight =
                          group_weight.cast<py::array_t<float, py::array::forcecast>>();
                    }
                    py::array_t<std::int64_t, py::array::forcecast> resolved_subgroup_id;
                    if (!subgroup_id.is_none()) {
                      resolved_subgroup_id =
                          subgroup_id.cast<py::array_t<std::int64_t, py::array::forcecast>>();
                    }
                    py::array_t<float, py::array::forcecast> resolved_baseline;
                    if (!baseline.is_none()) {
                      resolved_baseline = baseline.cast<py::array_t<float, py::array::forcecast>>();
                    }
                    py::array_t<std::int64_t, py::array::forcecast> resolved_pairs;
                    if (!pairs.is_none()) {
                      resolved_pairs = pairs.cast<py::array_t<std::int64_t, py::array::forcecast>>();
                    }
                    py::array_t<float, py::array::forcecast> resolved_pairs_weight;
                    if (!pairs_weight.is_none()) {
                      resolved_pairs_weight =
                          pairs_weight.cast<py::array_t<float, py::array::forcecast>>();
                    }
                    return ctboost::Pool(sparse_data,
                                         sparse_indices,
                                         sparse_indptr,
                                         num_rows,
                                         num_cols,
                                         label,
                                         std::move(cat_features),
                                         resolved_weight,
                                         resolved_group_id,
                                         resolved_group_weight,
                                         resolved_subgroup_id,
                                         resolved_baseline,
                                         resolved_pairs,
                                         resolved_pairs_weight);
                  },
                  py::arg("sparse_data"),
                  py::arg("sparse_indices"),
                  py::arg("sparse_indptr"),
                  py::arg("num_rows"),
                  py::arg("num_cols"),
                  py::arg("label"),
                  py::arg("cat_features") = std::vector<int>{},
                  py::arg("weight") = py::none(),
                  py::arg("group_id") = py::none(),
                  py::arg("group_weight") = py::none(),
                  py::arg("subgroup_id") = py::none(),
                  py::arg("baseline") = py::none(),
                  py::arg("pairs") = py::none(),
                  py::arg("pairs_weight") = py::none())
      .def("set_feature_storage_releasable", &ctboost::Pool::SetFeatureStorageReleasable);
}

}  // namespace ctboost::bindings
