#include "ctboost/data.hpp"

#include "ctboost/histogram.hpp"

#include "data_internal.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace py = pybind11;

namespace ctboost {
namespace {

void ValidateCudaQuantizationSchema(const CudaQuantizedMatrixView& view,
                                    const QuantizationSchema& schema,
                                    const std::vector<int>& cat_features) {
  if (view.element_bytes != 1U && view.element_bytes != 2U) {
    throw std::invalid_argument("CUDA quantized data must use uint8 or uint16 elements");
  }
  if (schema.num_cols() != view.num_cols) {
    throw std::invalid_argument(
        "CUDA quantization schema feature count must match the device matrix column count");
  }
  if (schema.cut_offsets.size() != view.num_cols + 1U || schema.cut_offsets.front() != 0U ||
      schema.cut_offsets.back() != schema.cut_values.size()) {
    throw std::invalid_argument("CUDA quantization schema has invalid cut offsets");
  }
  if (schema.categorical_mask.size() != view.num_cols ||
      schema.missing_value_mask.size() != view.num_cols ||
      (!schema.nan_modes.empty() && schema.nan_modes.size() != view.num_cols)) {
    throw std::invalid_argument("CUDA quantization schema masks must have one entry per feature");
  }
  if (schema.nan_mode > static_cast<std::uint8_t>(NanMode::Max)) {
    throw std::invalid_argument("CUDA quantization schema has an invalid nan_mode");
  }

  std::vector<int> schema_categorical_features;
  schema_categorical_features.reserve(cat_features.size());
  for (std::size_t feature = 0; feature < view.num_cols; ++feature) {
    if (schema.categorical_mask[feature] > 1U || schema.missing_value_mask[feature] > 1U) {
      throw std::invalid_argument("CUDA quantization schema masks must contain only zero or one");
    }
    if (!schema.nan_modes.empty() &&
        schema.nan_modes[feature] > static_cast<std::uint8_t>(NanMode::Max)) {
      throw std::invalid_argument("CUDA quantization schema has an invalid per-feature nan_mode");
    }
    const std::uint8_t feature_nan_mode =
        schema.nan_modes.empty() ? schema.nan_mode : schema.nan_modes[feature];
    if (schema.missing_value_mask[feature] != 0U &&
        feature_nan_mode == static_cast<std::uint8_t>(NanMode::Forbidden)) {
      throw std::invalid_argument(
          "CUDA quantization schema cannot declare missing values with nan_mode='Forbidden'");
    }
    if (schema.categorical_mask[feature] != 0U) {
      schema_categorical_features.push_back(static_cast<int>(feature));
    }

    const std::size_t bins = static_cast<std::size_t>(schema.num_bins_per_feature[feature]);
    if (bins == 0U) {
      throw std::invalid_argument("CUDA quantization schema features must have at least one bin");
    }
    if (view.element_bytes == 1U && bins > 256U) {
      throw std::invalid_argument(
          "uint8 CUDA quantized data cannot represent a feature with more than 256 bins");
    }
    if (schema.categorical_mask[feature] != 0U && bins > 256U) {
      throw std::invalid_argument("categorical CUDA quantization schema cannot exceed 256 bins");
    }

    const std::size_t cut_begin = schema.cut_offsets[feature];
    const std::size_t cut_end = schema.cut_offsets[feature + 1U];
    if (cut_begin > cut_end || cut_end > schema.cut_values.size()) {
      throw std::invalid_argument("CUDA quantization schema cut offsets must be non-decreasing");
    }
    const std::size_t cut_count = cut_end - cut_begin;
    const std::size_t non_missing_bins =
        bins - (schema.missing_value_mask[feature] != 0U ? 1U : 0U);
    const bool cut_count_valid =
        schema.categorical_mask[feature] != 0U
            ? cut_count == non_missing_bins
            : (non_missing_bins == 0U ? cut_count == 0U
                                      : cut_count + 1U == non_missing_bins);
    if (!cut_count_valid) {
      throw std::invalid_argument(
          "CUDA quantization schema cut count is inconsistent with its feature bin count");
    }
    for (std::size_t cut = cut_begin; cut < cut_end; ++cut) {
      if (std::isnan(schema.cut_values[cut])) {
        throw std::invalid_argument("CUDA quantization schema cuts cannot contain NaN");
      }
      if (cut > cut_begin && !(schema.cut_values[cut] > schema.cut_values[cut - 1U])) {
        throw std::invalid_argument(
            "CUDA quantization schema cuts must be strictly increasing within each feature");
      }
    }
  }
  if (schema_categorical_features != cat_features) {
    throw std::invalid_argument(
        "cat_features must exactly match the CUDA quantization schema categorical mask");
  }
}

}  // namespace

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
    std::int64_t previous_row = -1;
    for (std::int64_t index = begin; index < end; ++index) {
      const std::int64_t row_index = sparse_indices_ptr_[static_cast<std::size_t>(index)];
      if (row_index < 0 || static_cast<std::size_t>(row_index) >= num_rows_) {
        throw std::invalid_argument("sparse row index is out of bounds");
      }
      if (row_index <= previous_row) {
        throw std::invalid_argument(
            "sparse CSC row indices must be sorted and unique within each column");
      }
      previous_row = row_index;
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

Pool::Pool(CudaQuantizedMatrixView cuda_quantized_data,
           py::object cuda_quantized_owner,
           std::shared_ptr<const QuantizationSchema> quantization_schema,
           py::array_t<float, py::array::forcecast> label,
           std::vector<int> cat_features,
           py::array_t<float, py::array::forcecast> weight,
           py::array_t<std::int64_t, py::array::forcecast> group_id,
           py::array_t<float, py::array::forcecast> group_weight,
           py::array_t<std::int64_t, py::array::forcecast> subgroup_id,
           py::array_t<float, py::array::forcecast> baseline,
           py::array_t<std::int64_t, py::array::forcecast> pairs,
           py::array_t<float, py::array::forcecast> pairs_weight)
    : num_rows_(cuda_quantized_data.num_rows),
      num_cols_(cuda_quantized_data.num_cols),
      cuda_quantized_owner_(std::move(cuda_quantized_owner)),
      cuda_quantized_view_(cuda_quantized_data),
      cuda_quantization_schema_(std::move(quantization_schema)),
      cat_features_(std::move(cat_features)),
      has_cuda_quantized_features_(true) {
  if (cuda_quantized_owner_.is_none()) {
    throw std::invalid_argument("CUDA quantized data owner must not be None");
  }
  if (cuda_quantization_schema_ == nullptr) {
    throw std::invalid_argument("CUDA quantized data requires a quantization schema");
  }
  detail::ValidateFeatureIndices(cat_features_, num_cols_);
  ValidateCudaQuantizationSchema(
      cuda_quantized_view_, *cuda_quantization_schema_, cat_features_);

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
