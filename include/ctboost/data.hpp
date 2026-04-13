#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <pybind11/numpy.h>

namespace ctboost {

class Pool {
 public:
  Pool(pybind11::array_t<float, pybind11::array::forcecast> data,
       pybind11::array_t<float, pybind11::array::forcecast> label,
       std::vector<int> cat_features = {},
       pybind11::array_t<float, pybind11::array::forcecast> weight = pybind11::array_t<float>(),
       pybind11::array_t<std::int64_t, pybind11::array::forcecast> group_id =
           pybind11::array_t<std::int64_t>());
  Pool(pybind11::array_t<float, pybind11::array::forcecast> sparse_data,
       pybind11::array_t<std::int64_t, pybind11::array::forcecast> sparse_indices,
       pybind11::array_t<std::int64_t, pybind11::array::forcecast> sparse_indptr,
       std::size_t num_rows,
       std::size_t num_cols,
       pybind11::array_t<float, pybind11::array::forcecast> label,
       std::vector<int> cat_features = {},
       pybind11::array_t<float, pybind11::array::forcecast> weight = pybind11::array_t<float>(),
       pybind11::array_t<std::int64_t, pybind11::array::forcecast> group_id =
           pybind11::array_t<std::int64_t>());

  std::size_t num_rows() const noexcept;
  std::size_t num_cols() const noexcept;
  const std::vector<float>& feature_data() const;
  const std::vector<float>& labels() const noexcept;
  const std::vector<float>& weights() const noexcept;
  const std::vector<std::int64_t>& group_ids() const noexcept;
  bool has_group_ids() const noexcept;
  const std::vector<int>& cat_features() const noexcept;
  float feature_value(std::size_t row, std::size_t col) const;
  bool is_sparse() const noexcept;
  bool is_column_major_contiguous() const noexcept;
  const float* feature_column_ptr(std::size_t col) const;
  std::size_t dense_feature_bytes() const noexcept;
  bool ReleaseFeatureStorage() noexcept;
  bool feature_storage_releasable() const noexcept;
  void SetFeatureStorageReleasable(bool releasable) noexcept;

 private:
  std::size_t num_rows_{0};
  std::size_t num_cols_{0};
  pybind11::object feature_owner_;
  const float* feature_data_ptr_{nullptr};
  pybind11::ssize_t feature_row_stride_{0};
  pybind11::ssize_t feature_col_stride_{0};
  mutable std::vector<float> feature_data_cache_;
  pybind11::object sparse_data_owner_;
  pybind11::object sparse_indices_owner_;
  pybind11::object sparse_indptr_owner_;
  const float* sparse_data_ptr_{nullptr};
  const std::int64_t* sparse_indices_ptr_{nullptr};
  const std::int64_t* sparse_indptr_ptr_{nullptr};
  std::size_t sparse_nnz_{0};
  mutable std::vector<float> sparse_column_cache_;
  mutable std::size_t sparse_cached_column_{static_cast<std::size_t>(-1)};
  std::vector<float> labels_;
  std::vector<float> weights_;
  std::vector<std::int64_t> group_ids_;
  std::vector<int> cat_features_;
  bool has_group_ids_{false};
  bool is_sparse_{false};
  bool feature_storage_releasable_{false};
};

}  // namespace ctboost
