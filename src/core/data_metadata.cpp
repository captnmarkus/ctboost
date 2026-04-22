#include "ctboost/data.hpp"

namespace py = pybind11;

namespace ctboost {

std::size_t Pool::num_rows() const noexcept { return num_rows_; }

std::size_t Pool::num_cols() const noexcept { return num_cols_; }

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
