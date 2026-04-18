#include "ctboost/data.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
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

py::ssize_t ValidateInt64Stride(py::ssize_t stride_bytes, const char* name) {
  if (stride_bytes % static_cast<py::ssize_t>(sizeof(std::int64_t)) != 0) {
    throw std::invalid_argument(std::string(name) + " must have int64-compatible strides");
  }
  return stride_bytes / static_cast<py::ssize_t>(sizeof(std::int64_t));
}

void ValidateFeatureIndices(const std::vector<int>& cat_features, std::size_t num_cols) {
  for (const int feature_index : cat_features) {
    if (feature_index < 0 || static_cast<std::size_t>(feature_index) >= num_cols) {
      throw std::invalid_argument("categorical feature index is out of bounds");
    }
  }
}

void ValidateWeights(const std::vector<float>& weights) {
  for (const float sample_weight : weights) {
    if (!std::isfinite(sample_weight) || sample_weight < 0.0F) {
      throw std::invalid_argument("weight values must be finite and non-negative");
    }
  }
}

bool HasArrayValues(const py::buffer_info& info) {
  if (info.ndim == 0) {
    return false;
  }
  if (info.ndim == 1) {
    return info.shape[0] > 0;
  }
  if (info.ndim == 2) {
    return info.shape[0] > 0 && info.shape[1] > 0;
  }
  return false;
}

void CopyFloatVector1D(py::array_t<float, py::array::forcecast> values,
                       std::size_t expected_size,
                       const char* name,
                       std::vector<float>& out,
                       bool& has_values) {
  const py::buffer_info info = values.request();
  if (info.ndim != 0 && info.ndim != 1) {
    throw std::invalid_argument(std::string(name) + " must be a 1D NumPy array");
  }
  if (!HasArrayValues(info)) {
    out.clear();
    has_values = false;
    return;
  }
  if (static_cast<std::size_t>(info.shape[0]) != expected_size) {
    throw std::invalid_argument(std::string(name) + " size must match the number of data rows");
  }

  const auto* ptr = static_cast<const float*>(info.ptr);
  const py::ssize_t stride = ValidateFloatStride(info.strides[0], name);
  out.resize(expected_size);
  if (stride == 1) {
    std::memcpy(out.data(), ptr, out.size() * sizeof(float));
  } else {
    for (std::size_t row = 0; row < expected_size; ++row) {
      out[row] = *(ptr + static_cast<py::ssize_t>(row) * stride);
    }
  }
  has_values = true;
}

void CopyInt64Vector1D(py::array_t<std::int64_t, py::array::forcecast> values,
                       std::size_t expected_size,
                       const char* name,
                       std::vector<std::int64_t>& out,
                       bool& has_values) {
  const py::buffer_info info = values.request();
  if (info.ndim != 0 && info.ndim != 1) {
    throw std::invalid_argument(std::string(name) + " must be a 1D NumPy array");
  }
  if (!HasArrayValues(info)) {
    out.clear();
    has_values = false;
    return;
  }
  if (static_cast<std::size_t>(info.shape[0]) != expected_size) {
    throw std::invalid_argument(std::string(name) + " size must match the number of data rows");
  }

  const auto* ptr = static_cast<const std::int64_t*>(info.ptr);
  const py::ssize_t stride = ValidateInt64Stride(info.strides[0], name);
  out.resize(expected_size);
  if (stride == 1) {
    std::memcpy(out.data(), ptr, out.size() * sizeof(std::int64_t));
  } else {
    for (std::size_t row = 0; row < expected_size; ++row) {
      out[row] = *(ptr + static_cast<py::ssize_t>(row) * stride);
    }
  }
  has_values = true;
}

void CopyBaseline(py::array_t<float, py::array::forcecast> baseline,
                  std::size_t num_rows,
                  std::vector<float>& out,
                  bool& has_baseline,
                  int& baseline_dimension) {
  const py::buffer_info info = baseline.request();
  if (info.ndim != 0 && info.ndim != 1 && info.ndim != 2) {
    throw std::invalid_argument("baseline must be a 1D or 2D NumPy array");
  }
  if (!HasArrayValues(info)) {
    out.clear();
    has_baseline = false;
    baseline_dimension = 0;
    return;
  }

  const auto* ptr = static_cast<const float*>(info.ptr);
  if (info.ndim == 1) {
    if (static_cast<std::size_t>(info.shape[0]) != num_rows) {
      throw std::invalid_argument("baseline size must match the number of data rows");
    }
    const py::ssize_t stride = ValidateFloatStride(info.strides[0], "baseline");
    out.resize(num_rows);
    if (stride == 1) {
      std::memcpy(out.data(), ptr, out.size() * sizeof(float));
    } else {
      for (std::size_t row = 0; row < num_rows; ++row) {
        out[row] = *(ptr + static_cast<py::ssize_t>(row) * stride);
      }
    }
    has_baseline = true;
    baseline_dimension = 1;
    return;
  }

  if (static_cast<std::size_t>(info.shape[0]) != num_rows) {
    throw std::invalid_argument("baseline row count must match the number of data rows");
  }
  if (info.shape[1] <= 0) {
    throw std::invalid_argument("baseline must have at least one prediction column");
  }
  const py::ssize_t row_stride = ValidateFloatStride(info.strides[0], "baseline");
  const py::ssize_t col_stride = ValidateFloatStride(info.strides[1], "baseline");
  out.resize(num_rows * static_cast<std::size_t>(info.shape[1]));
  std::size_t offset = 0;
  for (py::ssize_t row = 0; row < info.shape[0]; ++row) {
    for (py::ssize_t col = 0; col < info.shape[1]; ++col) {
      out[offset++] = *(ptr + row * row_stride + col * col_stride);
    }
  }
  has_baseline = true;
  baseline_dimension = static_cast<int>(info.shape[1]);
}

void CopyPairs(py::array_t<std::int64_t, py::array::forcecast> pairs,
               py::array_t<float, py::array::forcecast> pairs_weight,
               std::size_t num_rows,
               std::vector<RankingPair>& out,
               bool& has_pairs) {
  const py::buffer_info pair_info = pairs.request();
  const py::buffer_info weight_info = pairs_weight.request();
  const bool empty_pair_vector =
      pair_info.ndim == 1 && pair_info.shape.size() == 1 && pair_info.shape[0] == 0;
  if (pair_info.ndim != 0 && pair_info.ndim != 2 && !empty_pair_vector) {
    throw std::invalid_argument("pairs must be a 2D NumPy array with shape (n_pairs, 2)");
  }
  if (weight_info.ndim != 0 && weight_info.ndim != 1) {
    throw std::invalid_argument("pairs_weight must be a 1D NumPy array");
  }
  if (!HasArrayValues(pair_info)) {
    if (HasArrayValues(weight_info)) {
      throw std::invalid_argument("pairs_weight requires pairs");
    }
    out.clear();
    has_pairs = false;
    return;
  }
  if (pair_info.shape[1] != 2) {
    throw std::invalid_argument("pairs must have shape (n_pairs, 2)");
  }

  const std::size_t pair_count = static_cast<std::size_t>(pair_info.shape[0]);
  const auto* pair_ptr = static_cast<const std::int64_t*>(pair_info.ptr);
  const py::ssize_t pair_row_stride = ValidateInt64Stride(pair_info.strides[0], "pairs");
  const py::ssize_t pair_col_stride = ValidateInt64Stride(pair_info.strides[1], "pairs");

  std::vector<float> resolved_weights;
  bool has_pair_weights = false;
  CopyFloatVector1D(pairs_weight, pair_count, "pairs_weight", resolved_weights, has_pair_weights);

  out.resize(pair_count);
  for (std::size_t pair_index = 0; pair_index < pair_count; ++pair_index) {
    const py::ssize_t row_offset = static_cast<py::ssize_t>(pair_index) * pair_row_stride;
    const std::int64_t winner = *(pair_ptr + row_offset);
    const std::int64_t loser = *(pair_ptr + row_offset + pair_col_stride);
    if (winner < 0 || loser < 0 || static_cast<std::size_t>(winner) >= num_rows ||
        static_cast<std::size_t>(loser) >= num_rows) {
      throw std::invalid_argument("pairs must reference valid row indices");
    }
    out[pair_index] = RankingPair{
        winner,
        loser,
        has_pair_weights ? resolved_weights[pair_index] : 1.0F,
    };
  }
  has_pairs = true;
}

void ValidatePoolMetadata(const std::vector<float>& weights,
                          const std::vector<std::int64_t>& group_ids,
                          bool has_group_ids,
                          const std::vector<float>& group_weights,
                          bool has_group_weights,
                          const std::vector<std::int64_t>& subgroup_ids,
                          bool has_subgroup_ids,
                          const std::vector<RankingPair>& pairs,
                          bool has_pairs) {
  ValidateWeights(weights);
  if (has_group_weights) {
    if (!has_group_ids) {
      throw std::invalid_argument("group_weight requires group_id values");
    }
    std::unordered_map<std::int64_t, float> group_weight_by_id;
    group_weight_by_id.reserve(group_ids.size());
    for (std::size_t row = 0; row < group_weights.size(); ++row) {
      const float value = group_weights[row];
      if (!std::isfinite(value) || value < 0.0F) {
        throw std::invalid_argument("group_weight values must be finite and non-negative");
      }
      const auto [it, inserted] = group_weight_by_id.emplace(group_ids[row], value);
      if (!inserted && std::fabs(static_cast<double>(it->second) - value) > 1e-6) {
        throw std::invalid_argument("group_weight must be constant within each group_id");
      }
    }
  }
  if (has_subgroup_ids && !has_group_ids) {
    throw std::invalid_argument("subgroup_id requires group_id values");
  }
  if (has_pairs) {
    if (!has_group_ids) {
      throw std::invalid_argument("pairs require group_id values");
    }
    for (const RankingPair& pair : pairs) {
      if (!std::isfinite(pair.weight) || pair.weight < 0.0F) {
        throw std::invalid_argument("pairs_weight values must be finite and non-negative");
      }
      if (pair.winner == pair.loser) {
        throw std::invalid_argument("pairs must reference distinct winner and loser rows");
      }
      if (group_ids[static_cast<std::size_t>(pair.winner)] !=
          group_ids[static_cast<std::size_t>(pair.loser)]) {
        throw std::invalid_argument("pairs must stay within a single group_id");
      }
    }
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
  ValidateFeatureIndices(cat_features_, num_cols_);

  feature_data_ptr_ = static_cast<const float*>(data_info.ptr);
  feature_row_stride_ = ValidateFloatStride(data_info.strides[0], "data");
  feature_col_stride_ = ValidateFloatStride(data_info.strides[1], "data");
  bool has_label = false;
  bool has_weight = false;
  CopyFloatVector1D(label, num_rows_, "label", labels_, has_label);
  CopyFloatVector1D(weight, num_rows_, "weight", weights_, has_weight);
  (void)has_label;
  (void)has_weight;
  CopyInt64Vector1D(group_id, num_rows_, "group_id", group_ids_, has_group_ids_);
  CopyFloatVector1D(group_weight, num_rows_, "group_weight", group_weights_, has_group_weights_);
  CopyInt64Vector1D(subgroup_id, num_rows_, "subgroup_id", subgroup_ids_, has_subgroup_ids_);
  CopyBaseline(baseline, num_rows_, baseline_, has_baseline_, baseline_dimension_);
  CopyPairs(pairs, pairs_weight, num_rows_, pairs_, has_pairs_);
  ValidatePoolMetadata(
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
  ValidateFeatureIndices(cat_features_, num_cols_);

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

  const py::ssize_t data_stride = ValidateFloatStride(data_info.strides[0], "sparse_data");
  const py::ssize_t indices_stride = ValidateInt64Stride(indices_info.strides[0], "sparse_indices");
  const py::ssize_t indptr_stride = ValidateInt64Stride(indptr_info.strides[0], "sparse_indptr");
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
  CopyFloatVector1D(label, num_rows_, "label", labels_, has_label);
  CopyFloatVector1D(weight, num_rows_, "weight", weights_, has_weight);
  (void)has_label;
  (void)has_weight;
  CopyInt64Vector1D(group_id, num_rows_, "group_id", group_ids_, has_group_ids_);
  CopyFloatVector1D(group_weight, num_rows_, "group_weight", group_weights_, has_group_weights_);
  CopyInt64Vector1D(subgroup_id, num_rows_, "subgroup_id", subgroup_ids_, has_subgroup_ids_);
  CopyBaseline(baseline, num_rows_, baseline_, has_baseline_, baseline_dimension_);
  CopyPairs(pairs, pairs_weight, num_rows_, pairs_, has_pairs_);
  ValidatePoolMetadata(
      weights_, group_ids_, has_group_ids_, group_weights_, has_group_weights_, subgroup_ids_,
      has_subgroup_ids_, pairs_, has_pairs_);
}

std::size_t Pool::num_rows() const noexcept { return num_rows_; }

std::size_t Pool::num_cols() const noexcept { return num_cols_; }

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

const std::vector<float>& Pool::labels() const noexcept { return labels_; }

const std::vector<float>& Pool::weights() const noexcept { return weights_; }

const std::vector<std::int64_t>& Pool::group_ids() const noexcept { return group_ids_; }

bool Pool::has_group_ids() const noexcept { return has_group_ids_; }

const std::vector<float>& Pool::group_weights() const noexcept { return group_weights_; }

bool Pool::has_group_weights() const noexcept { return has_group_weights_; }

const std::vector<std::int64_t>& Pool::subgroup_ids() const noexcept { return subgroup_ids_; }

bool Pool::has_subgroup_ids() const noexcept { return has_subgroup_ids_; }

const std::vector<RankingPair>& Pool::pairs() const noexcept { return pairs_; }

bool Pool::has_pairs() const noexcept { return has_pairs_; }

const std::vector<float>& Pool::baseline() const noexcept { return baseline_; }

bool Pool::has_baseline() const noexcept { return has_baseline_; }

int Pool::baseline_dimension() const noexcept { return baseline_dimension_; }

RankingMetadataView Pool::ranking_metadata() const noexcept {
  return RankingMetadataView{
      has_group_ids_ ? &group_ids_ : nullptr,
      has_subgroup_ids_ ? &subgroup_ids_ : nullptr,
      has_group_weights_ ? &group_weights_ : nullptr,
      has_pairs_ ? &pairs_ : nullptr,
  };
}

const std::vector<int>& Pool::cat_features() const noexcept { return cat_features_; }

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
        sparse_column_cache_[static_cast<std::size_t>(sparse_indices_ptr_[index])] = sparse_data_ptr_[index];
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

std::size_t Pool::dense_feature_bytes() const noexcept {
  std::size_t total_bytes = 0;
  if (is_sparse_) {
    if (sparse_data_ptr_ != nullptr) {
      total_bytes += sparse_nnz_ * sizeof(float);
      total_bytes += sparse_nnz_ * sizeof(std::int64_t);
      total_bytes += (num_cols_ + 1U) * sizeof(std::int64_t);
    }
    total_bytes += sparse_column_cache_.capacity() * sizeof(float);
  } else {
    const std::size_t dense_bytes = num_rows_ * num_cols_ * sizeof(float);
    if (feature_data_ptr_ != nullptr) {
      total_bytes += dense_bytes;
    }
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
  sparse_data_owner_ = py::object();
  sparse_indices_owner_ = py::object();
  sparse_indptr_owner_ = py::object();
  feature_data_ptr_ = nullptr;
  sparse_data_ptr_ = nullptr;
  sparse_indices_ptr_ = nullptr;
  sparse_indptr_ptr_ = nullptr;
  feature_row_stride_ = 0;
  feature_col_stride_ = 0;
  feature_data_cache_.clear();
  feature_data_cache_.shrink_to_fit();
  sparse_column_cache_.clear();
  sparse_column_cache_.shrink_to_fit();
  sparse_cached_column_ = static_cast<std::size_t>(-1);
  return true;
}

bool Pool::feature_storage_releasable() const noexcept { return feature_storage_releasable_; }

void Pool::SetFeatureStorageReleasable(bool releasable) noexcept {
  feature_storage_releasable_ = releasable;
}

}  // namespace ctboost
