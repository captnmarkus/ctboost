#include "ctboost/data.hpp"

#include "data_internal.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace py = pybind11;

namespace ctboost::detail {
namespace {

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

}  // namespace

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

}  // namespace ctboost::detail
