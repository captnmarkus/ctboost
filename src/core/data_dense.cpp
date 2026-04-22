#include "ctboost/data.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace py = pybind11;

namespace ctboost {

const std::vector<float>& Pool::feature_data() const {
  if (!feature_data_cache_.empty()) {
    return feature_data_cache_;
  }
  if (is_sparse_) {
    if (sparse_data_ptr_ == nullptr || sparse_indices_ptr_ == nullptr || sparse_indptr_ptr_ == nullptr) {
      throw std::runtime_error("feature storage has been released from this pool");
    }
    feature_data_cache_.assign(num_rows_ * num_cols_, 0.0F);
    for (std::size_t col = 0; col < num_cols_; ++col) {
      const std::size_t begin = static_cast<std::size_t>(sparse_indptr_ptr_[col]);
      const std::size_t end = static_cast<std::size_t>(sparse_indptr_ptr_[col + 1]);
      const std::size_t col_offset = col * num_rows_;
      for (std::size_t index = begin; index < end; ++index) {
        feature_data_cache_[col_offset + static_cast<std::size_t>(sparse_indices_ptr_[index])] =
            sparse_data_ptr_[index];
      }
    }
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

float Pool::feature_value(std::size_t row, std::size_t col) const {
  if (row >= num_rows_ || col >= num_cols_) {
    throw std::out_of_range("feature index is out of bounds");
  }
  if (is_sparse_) {
    if (!feature_data_cache_.empty()) {
      return feature_data_cache_[col * num_rows_ + row];
    }
    if (sparse_data_ptr_ == nullptr || sparse_indices_ptr_ == nullptr || sparse_indptr_ptr_ == nullptr) {
      throw std::runtime_error("feature storage has been released from this pool");
    }
    const std::size_t begin = static_cast<std::size_t>(sparse_indptr_ptr_[col]);
    const std::size_t end = static_cast<std::size_t>(sparse_indptr_ptr_[col + 1]);
    const auto* search_begin = sparse_indices_ptr_ + begin;
    const auto* search_end = sparse_indices_ptr_ + end;
    const std::int64_t row_index = static_cast<std::int64_t>(row);
    const auto* it = std::lower_bound(search_begin, search_end, row_index);
    if (it != search_end && *it == row_index) {
      return sparse_data_ptr_[begin + static_cast<std::size_t>(it - search_begin)];
    }
    return 0.0F;
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

bool Pool::is_sparse() const noexcept { return is_sparse_; }

bool Pool::is_column_major_contiguous() const noexcept {
  return !is_sparse_ && feature_data_ptr_ != nullptr && feature_row_stride_ == 1 &&
         feature_col_stride_ == static_cast<py::ssize_t>(num_rows_);
}

const float* Pool::feature_column_ptr(std::size_t col) const {
  if (col >= num_cols_) {
    throw std::out_of_range("feature index is out of bounds");
  }
  if (is_sparse_) {
    if (!feature_data_cache_.empty()) {
      return feature_data_cache_.data() + col * num_rows_;
    }
    if (sparse_data_ptr_ == nullptr || sparse_indices_ptr_ == nullptr || sparse_indptr_ptr_ == nullptr) {
      return nullptr;
    }
    if (sparse_column_cache_.size() != num_rows_) {
      sparse_column_cache_.assign(num_rows_, 0.0F);
      sparse_cached_column_ = static_cast<std::size_t>(-1);
    }
    if (sparse_cached_column_ != col) {
      std::fill(sparse_column_cache_.begin(), sparse_column_cache_.end(), 0.0F);
      const std::size_t begin = static_cast<std::size_t>(sparse_indptr_ptr_[col]);
      const std::size_t end = static_cast<std::size_t>(sparse_indptr_ptr_[col + 1]);
      for (std::size_t index = begin; index < end; ++index) {
        sparse_column_cache_[static_cast<std::size_t>(sparse_indices_ptr_[index])] =
            sparse_data_ptr_[index];
      }
      sparse_cached_column_ = col;
    }
    return sparse_column_cache_.data();
  }
  if (is_column_major_contiguous()) {
    return feature_data_ptr_ + col * num_rows_;
  }
  if (!feature_data_cache_.empty()) {
    return feature_data_cache_.data() + col * num_rows_;
  }
  return nullptr;
}

}  // namespace ctboost
