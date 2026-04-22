#pragma once

#include "ctboost/data.hpp"

namespace ctboost::detail {

pybind11::ssize_t ValidateFloatStride(pybind11::ssize_t stride_bytes, const char* name);
pybind11::ssize_t ValidateInt64Stride(pybind11::ssize_t stride_bytes, const char* name);
void ValidateFeatureIndices(const std::vector<int>& cat_features, std::size_t num_cols);
void CopyFloatVector1D(pybind11::array_t<float, pybind11::array::forcecast> values,
                       std::size_t expected_size,
                       const char* name,
                       std::vector<float>& out,
                       bool& has_values);
void CopyInt64Vector1D(pybind11::array_t<std::int64_t, pybind11::array::forcecast> values,
                       std::size_t expected_size,
                       const char* name,
                       std::vector<std::int64_t>& out,
                       bool& has_values);
void CopyBaseline(pybind11::array_t<float, pybind11::array::forcecast> baseline,
                  std::size_t num_rows,
                  std::vector<float>& out,
                  bool& has_baseline,
                  int& baseline_dimension);
void CopyPairs(pybind11::array_t<std::int64_t, pybind11::array::forcecast> pairs,
               pybind11::array_t<float, pybind11::array::forcecast> pairs_weight,
               std::size_t num_rows,
               std::vector<RankingPair>& out,
               bool& has_pairs);
void ValidatePoolMetadata(const std::vector<float>& weights,
                          const std::vector<std::int64_t>& group_ids,
                          bool has_group_ids,
                          const std::vector<float>& group_weights,
                          bool has_group_weights,
                          const std::vector<std::int64_t>& subgroup_ids,
                          bool has_subgroup_ids,
                          const std::vector<RankingPair>& pairs,
                          bool has_pairs);

}  // namespace ctboost::detail
