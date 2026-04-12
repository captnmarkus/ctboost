#include "ctboost/data.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace py = pybind11;

namespace ctboost {
namespace {

py::ssize_t ValidateFloatStride(py::ssize_t stride_bytes, const char* name) {
  if (stride_bytes % static_cast<py::ssize_t>(sizeof(float)) != 0) {
    throw std::invalid_argument(std::string(name) + " must have float-compatible strides");
  }
  return stride_bytes / static_cast<py::ssize_t>(sizeof(float));
}

}  // namespace

Pool::Pool(py::array_t<float, py::array::forcecast> data,
           py::array_t<float, py::array::forcecast> label,
           std::vector<int> cat_features,
           py::array_t<float, py::array::forcecast> weight,
           py::array_t<std::int64_t, py::array::forcecast> group_id)
    : cat_features_(std::move(cat_features)) {
  feature_owner_ = py::reinterpret_borrow<py::object>(data);
  const py::buffer_info data_info = data.request();
  const py::buffer_info label_info = label.request();
  const py::buffer_info weight_info = weight.request();
  const py::buffer_info group_info = group_id.request();

  if (data_info.ndim != 2) {
    throw std::invalid_argument("data must be a 2D NumPy array");
  }
  if (label_info.ndim != 1) {
    throw std::invalid_argument("label must be a 1D NumPy array");
  }
  if (weight_info.ndim != 1) {
    throw std::invalid_argument("weight must be a 1D NumPy array");
  }
  if (group_info.ndim != 0 && group_info.ndim != 1) {
    throw std::invalid_argument("group_id must be a 1D NumPy array");
  }

  num_rows_ = static_cast<std::size_t>(data_info.shape[0]);
  num_cols_ = static_cast<std::size_t>(data_info.shape[1]);

  if (static_cast<std::size_t>(label_info.shape[0]) != num_rows_) {
    throw std::invalid_argument("label size must match the number of data rows");
  }
  if (static_cast<std::size_t>(weight_info.shape[0]) != num_rows_) {
    throw std::invalid_argument("weight size must match the number of data rows");
  }
  const bool has_group_ids = group_info.ndim == 1 && group_info.shape[0] > 0;
  if (has_group_ids && static_cast<std::size_t>(group_info.shape[0]) != num_rows_) {
    throw std::invalid_argument("group_id size must match the number of data rows");
  }

  for (const int feature_index : cat_features_) {
    if (feature_index < 0 ||
        static_cast<std::size_t>(feature_index) >= num_cols_) {
      throw std::invalid_argument("categorical feature index is out of bounds");
    }
  }

  feature_data_ptr_ = static_cast<const float*>(data_info.ptr);
  const auto* label_ptr = static_cast<const float*>(label_info.ptr);
  const auto* weight_ptr = static_cast<const float*>(weight_info.ptr);
  const auto* group_ptr = static_cast<const std::int64_t*>(group_info.ptr);

  feature_row_stride_ = ValidateFloatStride(data_info.strides[0], "data");
  feature_col_stride_ = ValidateFloatStride(data_info.strides[1], "data");
  const py::ssize_t label_stride = ValidateFloatStride(label_info.strides[0], "label");
  const py::ssize_t weight_stride = ValidateFloatStride(weight_info.strides[0], "weight");
  if (has_group_ids &&
      group_info.strides[0] % static_cast<py::ssize_t>(sizeof(std::int64_t)) != 0) {
    throw std::invalid_argument("group_id must have integer-compatible strides");
  }
  const py::ssize_t group_stride =
      has_group_ids ? group_info.strides[0] / static_cast<py::ssize_t>(sizeof(std::int64_t)) : 0;

  labels_.resize(num_rows_);
  weights_.resize(num_rows_);
  if (has_group_ids) {
    group_ids_.resize(num_rows_);
    has_group_ids_ = true;
  }

  if (!labels_.empty()) {
    if (label_stride == 1) {
      std::memcpy(labels_.data(), label_ptr, labels_.size() * sizeof(float));
    } else {
      for (std::size_t row = 0; row < num_rows_; ++row) {
        labels_[row] = *(label_ptr + static_cast<py::ssize_t>(row) * label_stride);
      }
    }
  }

  if (!weights_.empty()) {
    if (weight_stride == 1) {
      std::memcpy(weights_.data(), weight_ptr, weights_.size() * sizeof(float));
    } else {
      for (std::size_t row = 0; row < num_rows_; ++row) {
        weights_[row] = *(weight_ptr + static_cast<py::ssize_t>(row) * weight_stride);
      }
    }
  }

  if (has_group_ids_) {
    for (std::size_t row = 0; row < num_rows_; ++row) {
      group_ids_[row] = *(group_ptr + static_cast<py::ssize_t>(row) * group_stride);
    }
  }

  for (const float sample_weight : weights_) {
    if (!std::isfinite(sample_weight) || sample_weight < 0.0F) {
      throw std::invalid_argument("weight values must be finite and non-negative");
    }
  }
}

std::size_t Pool::num_rows() const noexcept { return num_rows_; }

std::size_t Pool::num_cols() const noexcept { return num_cols_; }

const std::vector<float>& Pool::feature_data() const {
  if (!feature_data_cache_.empty()) {
    return feature_data_cache_;
  }
  if (feature_data_ptr_ == nullptr) {
    throw std::runtime_error("feature storage has been released from this pool");
  }

  feature_data_cache_.resize(num_rows_ * num_cols_);
  if (feature_data_cache_.empty()) {
    return feature_data_cache_;
  }

  const py::ssize_t fortran_row_stride = 1;
  const py::ssize_t fortran_col_stride = static_cast<py::ssize_t>(num_rows_);
  if (feature_row_stride_ == fortran_row_stride && feature_col_stride_ == fortran_col_stride) {
    std::memcpy(
        feature_data_cache_.data(), feature_data_ptr_, feature_data_cache_.size() * sizeof(float));
    return feature_data_cache_;
  }

  for (std::size_t col = 0; col < num_cols_; ++col) {
    for (std::size_t row = 0; row < num_rows_; ++row) {
      const py::ssize_t offset = static_cast<py::ssize_t>(row) * feature_row_stride_ +
                                 static_cast<py::ssize_t>(col) * feature_col_stride_;
      feature_data_cache_[col * num_rows_ + row] = *(feature_data_ptr_ + offset);
    }
  }
  return feature_data_cache_;
}

const std::vector<float>& Pool::labels() const noexcept { return labels_; }

const std::vector<float>& Pool::weights() const noexcept { return weights_; }

const std::vector<std::int64_t>& Pool::group_ids() const noexcept { return group_ids_; }

bool Pool::has_group_ids() const noexcept { return has_group_ids_; }

const std::vector<int>& Pool::cat_features() const noexcept { return cat_features_; }

float Pool::feature_value(std::size_t row, std::size_t col) const {
  if (row >= num_rows_ || col >= num_cols_) {
    throw std::out_of_range("feature index is out of bounds");
  }
  if (feature_data_ptr_ != nullptr) {
    const py::ssize_t offset = static_cast<py::ssize_t>(row) * feature_row_stride_ +
                               static_cast<py::ssize_t>(col) * feature_col_stride_;
    return *(feature_data_ptr_ + offset);
  }
  if (!feature_data_cache_.empty()) {
    return feature_data_cache_[col * num_rows_ + row];
  }
  throw std::runtime_error("feature storage has been released from this pool");
}

bool Pool::is_column_major_contiguous() const noexcept {
  return feature_data_ptr_ != nullptr && feature_row_stride_ == 1 &&
         feature_col_stride_ == static_cast<py::ssize_t>(num_rows_);
}

const float* Pool::feature_column_ptr(std::size_t col) const {
  if (col >= num_cols_) {
    throw std::out_of_range("feature index is out of bounds");
  }
  if (is_column_major_contiguous()) {
    return feature_data_ptr_ + col * num_rows_;
  }
  if (!feature_data_cache_.empty()) {
    return feature_data_cache_.data() + col * num_rows_;
  }
  return nullptr;
}

std::size_t Pool::dense_feature_bytes() const noexcept {
  const std::size_t dense_bytes = num_rows_ * num_cols_ * sizeof(float);
  std::size_t total_bytes = 0;
  if (feature_data_ptr_ != nullptr) {
    total_bytes += dense_bytes;
  }
  if (!feature_data_cache_.empty()) {
    total_bytes += feature_data_cache_.capacity() * sizeof(float);
  }
  return total_bytes;
}

bool Pool::ReleaseFeatureStorage() noexcept {
  if (!feature_storage_releasable_) {
    return false;
  }

  feature_owner_ = py::object();
  feature_data_ptr_ = nullptr;
  feature_row_stride_ = 0;
  feature_col_stride_ = 0;
  feature_data_cache_.clear();
  feature_data_cache_.shrink_to_fit();
  return true;
}

bool Pool::feature_storage_releasable() const noexcept { return feature_storage_releasable_; }

void Pool::SetFeatureStorageReleasable(bool releasable) noexcept {
  feature_storage_releasable_ = releasable;
}

}  // namespace ctboost
