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

  const auto* data_ptr = static_cast<const float*>(data_info.ptr);
  const auto* label_ptr = static_cast<const float*>(label_info.ptr);
  const auto* weight_ptr = static_cast<const float*>(weight_info.ptr);
  const auto* group_ptr = static_cast<const std::int64_t*>(group_info.ptr);

  const py::ssize_t data_row_stride = ValidateFloatStride(data_info.strides[0], "data");
  const py::ssize_t data_col_stride = ValidateFloatStride(data_info.strides[1], "data");
  const py::ssize_t label_stride = ValidateFloatStride(label_info.strides[0], "label");
  const py::ssize_t weight_stride = ValidateFloatStride(weight_info.strides[0], "weight");
  if (has_group_ids &&
      group_info.strides[0] % static_cast<py::ssize_t>(sizeof(std::int64_t)) != 0) {
    throw std::invalid_argument("group_id must have integer-compatible strides");
  }
  const py::ssize_t group_stride =
      has_group_ids ? group_info.strides[0] / static_cast<py::ssize_t>(sizeof(std::int64_t)) : 0;

  feature_data_.resize(num_rows_ * num_cols_);
  labels_.resize(num_rows_);
  weights_.resize(num_rows_);
  if (has_group_ids) {
    group_ids_.resize(num_rows_);
    has_group_ids_ = true;
  }

  if (!feature_data_.empty()) {
    const py::ssize_t fortran_row_stride = 1;
    const py::ssize_t fortran_col_stride = static_cast<py::ssize_t>(num_rows_);
    if (data_row_stride == fortran_row_stride && data_col_stride == fortran_col_stride) {
      std::memcpy(feature_data_.data(), data_ptr, feature_data_.size() * sizeof(float));
    } else {
      for (std::size_t col = 0; col < num_cols_; ++col) {
        for (std::size_t row = 0; row < num_rows_; ++row) {
          const py::ssize_t offset = static_cast<py::ssize_t>(row) * data_row_stride +
                                     static_cast<py::ssize_t>(col) * data_col_stride;
          feature_data_[col * num_rows_ + row] = *(data_ptr + offset);
        }
      }
    }
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

const std::vector<float>& Pool::feature_data() const noexcept { return feature_data_; }

const std::vector<float>& Pool::labels() const noexcept { return labels_; }

const std::vector<float>& Pool::weights() const noexcept { return weights_; }

const std::vector<std::int64_t>& Pool::group_ids() const noexcept { return group_ids_; }

bool Pool::has_group_ids() const noexcept { return has_group_ids_; }

const std::vector<int>& Pool::cat_features() const noexcept { return cat_features_; }

float Pool::feature_value(std::size_t row, std::size_t col) const {
  if (row >= num_rows_ || col >= num_cols_) {
    throw std::out_of_range("feature index is out of bounds");
  }
  return feature_data_[col * num_rows_ + row];
}

}  // namespace ctboost
