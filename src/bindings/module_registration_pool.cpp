#include "module_internal.hpp"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

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

std::size_t CheckedCudaDimension(const py::handle& value, const char* name) {
  const py::ssize_t dimension = py::cast<py::ssize_t>(value);
  if (dimension < 0) {
    throw std::invalid_argument(std::string("CUDA quantized ") + name + " must be non-negative");
  }
  return static_cast<std::size_t>(dimension);
}

std::size_t CheckedMultiply(std::size_t lhs, std::size_t rhs, const char* name) {
  if (lhs != 0U && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    throw std::invalid_argument(std::string("CUDA quantized ") + name + " overflows size_t");
  }
  return lhs * rhs;
}

ctboost::CudaQuantizedMatrixView ParseCudaQuantizedMatrixView(const py::handle& owner) {
  if (!py::hasattr(owner, "__cuda_array_interface__")) {
    throw std::invalid_argument(
        "CUDA quantized data must expose __cuda_array_interface__ version 3");
  }
  const py::dict interface = owner.attr("__cuda_array_interface__").cast<py::dict>();
  for (const char* required : {"shape", "typestr", "data", "version"}) {
    if (!interface.contains(required)) {
      throw std::invalid_argument(
          std::string("CUDA Array Interface is missing required field '") + required + "'");
    }
  }
  const int version = py::cast<int>(interface["version"]);
  if (version != 3) {
    throw std::invalid_argument("CUDA quantized data requires CUDA Array Interface version 3");
  }
  if (interface.contains("mask") && !interface["mask"].is_none()) {
    throw std::invalid_argument("masked CUDA quantized arrays are not supported");
  }

  const py::tuple shape = interface["shape"].cast<py::tuple>();
  if (shape.size() != 2U) {
    throw std::invalid_argument("CUDA quantized data must be a 2D array");
  }
  const std::size_t num_rows = CheckedCudaDimension(shape[0], "row count");
  const std::size_t num_cols = CheckedCudaDimension(shape[1], "column count");

  const std::string typestr = py::cast<std::string>(interface["typestr"]);
  std::uint8_t element_bytes = 0U;
  if (typestr == "|u1" || typestr == "<u1" || typestr == "=u1") {
    element_bytes = 1U;
  } else if (typestr == "<u2" || typestr == "=u2") {
    element_bytes = 2U;
  } else {
    throw std::invalid_argument(
        "CUDA quantized data must have little-endian uint8 or uint16 dtype");
  }

  const py::tuple data = interface["data"].cast<py::tuple>();
  if (data.size() != 2U) {
    throw std::invalid_argument("CUDA Array Interface data must be a (pointer, read_only) pair");
  }
  const std::uintptr_t data_pointer = py::cast<std::uintptr_t>(data[0]);
  (void)py::cast<bool>(data[1]);
  if (num_rows != 0U && num_cols != 0U && data_pointer == 0U) {
    throw std::invalid_argument("non-empty CUDA quantized data must have a non-null pointer");
  }
  if (data_pointer % element_bytes != 0U) {
    throw std::invalid_argument("CUDA quantized data pointer is not aligned for its dtype");
  }

  const std::size_t c_row_stride = CheckedMultiply(num_cols, element_bytes, "row stride");
  const std::size_t f_col_stride = CheckedMultiply(num_rows, element_bytes, "column stride");
  if (c_row_stride > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
      f_col_stride > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::invalid_argument("CUDA quantized contiguous strides exceed int64 range");
  }
  std::int64_t row_stride = static_cast<std::int64_t>(c_row_stride);
  std::int64_t col_stride = static_cast<std::int64_t>(element_bytes);
  if (interface.contains("strides") && !interface["strides"].is_none()) {
    const py::tuple strides = interface["strides"].cast<py::tuple>();
    if (strides.size() != 2U) {
      throw std::invalid_argument("CUDA quantized data strides must contain two entries");
    }
    row_stride = py::cast<std::int64_t>(strides[0]);
    col_stride = py::cast<std::int64_t>(strides[1]);
  }
  if (row_stride < 0 || col_stride < 0 ||
      row_stride % static_cast<std::int64_t>(element_bytes) != 0 ||
      col_stride % static_cast<std::int64_t>(element_bytes) != 0) {
    throw std::invalid_argument(
        "CUDA quantized data strides must be non-negative and dtype-aligned");
  }
  const bool empty = num_rows == 0U || num_cols == 0U;
  const bool c_contiguous =
      col_stride == static_cast<std::int64_t>(element_bytes) &&
      (num_rows <= 1U || row_stride == static_cast<std::int64_t>(c_row_stride));
  const bool f_contiguous =
      row_stride == static_cast<std::int64_t>(element_bytes) &&
      (num_cols <= 1U || col_stride == static_cast<std::int64_t>(f_col_stride));
  if (!empty && !c_contiguous && !f_contiguous) {
    throw std::invalid_argument(
        "CUDA quantized data must be C-contiguous or Fortran-contiguous");
  }

  if (!empty) {
    const std::size_t row_offset = CheckedMultiply(
        num_rows - 1U, static_cast<std::size_t>(row_stride), "row address range");
    const std::size_t col_offset = CheckedMultiply(
        num_cols - 1U, static_cast<std::size_t>(col_stride), "column address range");
    if (row_offset > std::numeric_limits<std::size_t>::max() - col_offset) {
      throw std::invalid_argument("CUDA quantized data address range overflows uintptr_t");
    }
    const std::size_t address_span = row_offset + col_offset;
    const std::uintptr_t trailing_bytes = static_cast<std::uintptr_t>(element_bytes - 1U);
    if (data_pointer > std::numeric_limits<std::uintptr_t>::max() - trailing_bytes ||
        address_span >
            std::numeric_limits<std::uintptr_t>::max() - data_pointer - trailing_bytes) {
      throw std::invalid_argument("CUDA quantized data address range overflows uintptr_t");
    }
  }

  std::uintptr_t producer_stream = 0U;
  if (interface.contains("stream") && !interface["stream"].is_none()) {
    producer_stream = py::cast<std::uintptr_t>(interface["stream"]);
    if (producer_stream == 0U) {
      throw std::invalid_argument(
          "CUDA Array Interface stream must be None, 1, or a non-zero named cudaStream_t");
    }
    if (producer_stream == 2U) {
      throw std::invalid_argument(
          "CUDA Array Interface per-thread default stream marker 2 is not supported by "
          "deferred CUDA quantized Pool consumption; use the legacy default or a named stream");
    }
  }

  return ctboost::CudaQuantizedMatrixView{
      data_pointer,
      num_rows,
      num_cols,
      row_stride,
      col_stride,
      element_bytes,
      producer_stream,
  };
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
      .def("has_cuda_quantized_features", &ctboost::Pool::has_cuda_quantized_features)
      .def_static("from_cuda_quantized",
                  [](py::object cuda_quantized_data,
                     const py::dict& quantization_schema,
                     py::array_t<float, py::array::forcecast> label,
                     std::vector<int> cat_features,
                     py::object weight,
                     py::object group_id,
                     py::object group_weight,
                     py::object subgroup_id,
                     py::object baseline,
                     py::object pairs,
                     py::object pairs_weight) {
                    const ctboost::CudaQuantizedMatrixView view =
                        ParseCudaQuantizedMatrixView(cuda_quantized_data);
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
                      resolved_baseline =
                          baseline.cast<py::array_t<float, py::array::forcecast>>();
                    }
                    py::array_t<std::int64_t, py::array::forcecast> resolved_pairs;
                    if (!pairs.is_none()) {
                      resolved_pairs =
                          pairs.cast<py::array_t<std::int64_t, py::array::forcecast>>();
                    }
                    py::array_t<float, py::array::forcecast> resolved_pairs_weight;
                    if (!pairs_weight.is_none()) {
                      resolved_pairs_weight =
                          pairs_weight.cast<py::array_t<float, py::array::forcecast>>();
                    }
                    return ctboost::Pool(view,
                                         std::move(cuda_quantized_data),
                                         QuantizationSchemaFromStateDict(quantization_schema),
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
                  py::arg("data"),
                  py::arg("quantization_schema"),
                  py::arg("label"),
                  py::arg("cat_features") = std::vector<int>{},
                  py::arg("weight") = py::none(),
                  py::arg("group_id") = py::none(),
                  py::arg("group_weight") = py::none(),
                  py::arg("subgroup_id") = py::none(),
                  py::arg("baseline") = py::none(),
                  py::arg("pairs") = py::none(),
                  py::arg("pairs_weight") = py::none())
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
