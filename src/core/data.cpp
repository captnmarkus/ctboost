#include "ctboost/data.hpp"

#include "data_internal.hpp"

#include <stdexcept>
#include <utility>

namespace py = pybind11;

namespace ctboost {

Pool::Pool(py::array_t<float, py::array::forcecast> data,
           py::array_t<float, py::array::forcecast> label,
           std::vector<int> cat_features,
           py::array_t<float, py::array::forcecast> weight,
           py::array_t<std::int64_t, py::array::forcecast> group_id,
           py::array_t<float, py::array::forcecast> group_weight,
           py::array_t<std::int64_t, py::array::forcecast> subgroup_id,
           py::array_t<float, py::array::forcecast> baseline,
           py::array_t<std::int64_t, py::array::forcecast> pairs,
           py::array_t<float, py::array::forcecast> pairs_weight)
    : cat_features_(std::move(cat_features)) {
  feature_owner_ = py::reinterpret_borrow<py::object>(data);
  const py::buffer_info data_info = data.request();

  if (data_info.ndim != 2) {
    throw std::invalid_argument("data must be a 2D NumPy array");
  }

  num_rows_ = static_cast<std::size_t>(data_info.shape[0]);
  num_cols_ = static_cast<std::size_t>(data_info.shape[1]);
  detail::ValidateFeatureIndices(cat_features_, num_cols_);

  feature_data_ptr_ = static_cast<const float*>(data_info.ptr);
  feature_row_stride_ = detail::ValidateFloatStride(data_info.strides[0], "data");
  feature_col_stride_ = detail::ValidateFloatStride(data_info.strides[1], "data");
  bool has_label = false;
  bool has_weight = false;
  detail::CopyFloatVector1D(label, num_rows_, "label", labels_, has_label);
  detail::CopyFloatVector1D(weight, num_rows_, "weight", weights_, has_weight);
  (void)has_label;
  (void)has_weight;
  detail::CopyInt64Vector1D(group_id, num_rows_, "group_id", group_ids_, has_group_ids_);
  detail::CopyFloatVector1D(
      group_weight, num_rows_, "group_weight", group_weights_, has_group_weights_);
  detail::CopyInt64Vector1D(
      subgroup_id, num_rows_, "subgroup_id", subgroup_ids_, has_subgroup_ids_);
  detail::CopyBaseline(baseline, num_rows_, baseline_, has_baseline_, baseline_dimension_);
  detail::CopyPairs(pairs, pairs_weight, num_rows_, pairs_, has_pairs_);
  detail::ValidatePoolMetadata(
      weights_, group_ids_, has_group_ids_, group_weights_, has_group_weights_, subgroup_ids_,
      has_subgroup_ids_, pairs_, has_pairs_);
}

Pool::Pool(py::array_t<float, py::array::forcecast> sparse_data,
           py::array_t<std::int64_t, py::array::forcecast> sparse_indices,
           py::array_t<std::int64_t, py::array::forcecast> sparse_indptr,
           std::size_t num_rows,
           std::size_t num_cols,
           py::array_t<float, py::array::forcecast> label,
           std::vector<int> cat_features,
           py::array_t<float, py::array::forcecast> weight,
           py::array_t<std::int64_t, py::array::forcecast> group_id,
           py::array_t<float, py::array::forcecast> group_weight,
           py::array_t<std::int64_t, py::array::forcecast> subgroup_id,
           py::array_t<float, py::array::forcecast> baseline,
           py::array_t<std::int64_t, py::array::forcecast> pairs,
           py::array_t<float, py::array::forcecast> pairs_weight)
    : num_rows_(num_rows), num_cols_(num_cols), cat_features_(std::move(cat_features)), is_sparse_(true) {
  detail::ValidateFeatureIndices(cat_features_, num_cols_);

  sparse_data_owner_ = py::reinterpret_borrow<py::object>(sparse_data);
  sparse_indices_owner_ = py::reinterpret_borrow<py::object>(sparse_indices);
  sparse_indptr_owner_ = py::reinterpret_borrow<py::object>(sparse_indptr);
  const py::buffer_info data_info = sparse_data.request();
  const py::buffer_info indices_info = sparse_indices.request();
  const py::buffer_info indptr_info = sparse_indptr.request();
  if (data_info.ndim != 1) {
    throw std::invalid_argument("sparse_data must be a 1D NumPy array");
  }
  if (indices_info.ndim != 1) {
    throw std::invalid_argument("sparse_indices must be a 1D NumPy array");
  }
  if (indptr_info.ndim != 1) {
    throw std::invalid_argument("sparse_indptr must be a 1D NumPy array");
  }
  if (static_cast<std::size_t>(indices_info.shape[0]) != static_cast<std::size_t>(data_info.shape[0])) {
    throw std::invalid_argument("sparse_indices size must match sparse_data size");
  }
  if (static_cast<std::size_t>(indptr_info.shape[0]) != num_cols_ + 1U) {
    throw std::invalid_argument("sparse_indptr size must equal num_cols + 1");
  }

  const py::ssize_t data_stride = detail::ValidateFloatStride(data_info.strides[0], "sparse_data");
  const py::ssize_t indices_stride =
      detail::ValidateInt64Stride(indices_info.strides[0], "sparse_indices");
  const py::ssize_t indptr_stride =
      detail::ValidateInt64Stride(indptr_info.strides[0], "sparse_indptr");
  if (data_stride != 1 || indices_stride != 1 || indptr_stride != 1) {
    throw std::invalid_argument("sparse CSC buffers must be contiguous");
  }

  sparse_data_ptr_ = static_cast<const float*>(data_info.ptr);
  sparse_indices_ptr_ = static_cast<const std::int64_t*>(indices_info.ptr);
  sparse_indptr_ptr_ = static_cast<const std::int64_t*>(indptr_info.ptr);
  sparse_nnz_ = static_cast<std::size_t>(data_info.shape[0]);

  if (sparse_indptr_ptr_[0] != 0) {
    throw std::invalid_argument("sparse_indptr must start with zero");
  }
  if (static_cast<std::size_t>(sparse_indptr_ptr_[num_cols_]) != sparse_nnz_) {
    throw std::invalid_argument("sparse_indptr must end at sparse_data size");
  }
  for (std::size_t col = 0; col < num_cols_; ++col) {
    const std::int64_t begin = sparse_indptr_ptr_[col];
    const std::int64_t end = sparse_indptr_ptr_[col + 1];
    if (begin < 0 || end < begin || static_cast<std::size_t>(end) > sparse_nnz_) {
      throw std::invalid_argument("sparse_indptr must be a non-decreasing CSC column pointer array");
    }
  }
  for (std::size_t index = 0; index < sparse_nnz_; ++index) {
    const std::int64_t row_index = sparse_indices_ptr_[index];
    if (row_index < 0 || static_cast<std::size_t>(row_index) >= num_rows_) {
      throw std::invalid_argument("sparse row index is out of bounds");
    }
  }

  bool has_label = false;
  bool has_weight = false;
  detail::CopyFloatVector1D(label, num_rows_, "label", labels_, has_label);
  detail::CopyFloatVector1D(weight, num_rows_, "weight", weights_, has_weight);
  (void)has_label;
  (void)has_weight;
  detail::CopyInt64Vector1D(group_id, num_rows_, "group_id", group_ids_, has_group_ids_);
  detail::CopyFloatVector1D(
      group_weight, num_rows_, "group_weight", group_weights_, has_group_weights_);
  detail::CopyInt64Vector1D(
      subgroup_id, num_rows_, "subgroup_id", subgroup_ids_, has_subgroup_ids_);
  detail::CopyBaseline(baseline, num_rows_, baseline_, has_baseline_, baseline_dimension_);
  detail::CopyPairs(pairs, pairs_weight, num_rows_, pairs_, has_pairs_);
  detail::ValidatePoolMetadata(
      weights_, group_ids_, has_group_ids_, group_weights_, has_group_weights_, subgroup_ids_,
      has_subgroup_ids_, pairs_, has_pairs_);
}

}  // namespace ctboost
