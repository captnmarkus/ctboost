#include "module_internal.hpp"

#include <cstring>
#include <limits>
#include <stdexcept>

namespace ctboost::bindings {

py::array_t<float> VectorToArray(const std::vector<float>& values) {
  py::array_t<float> result(values.size());
  if (!values.empty()) {
    std::memcpy(result.mutable_data(), values.data(), values.size() * sizeof(float));
  }
  return result;
}

py::array_t<std::int32_t> IntVectorToArray(const std::vector<std::int32_t>& values) {
  py::array_t<std::int32_t> result(values.size());
  if (!values.empty()) {
    std::memcpy(result.mutable_data(), values.data(), values.size() * sizeof(std::int32_t));
  }
  return result;
}

std::vector<float> ArrayToVector(py::array_t<float, py::array::forcecast> values, const char* name) {
  const py::buffer_info info = values.request();
  if (info.ndim != 1) {
    throw std::invalid_argument(std::string(name) + " must be a 1D NumPy array");
  }
  if (info.strides[0] % static_cast<py::ssize_t>(sizeof(float)) != 0) {
    throw std::invalid_argument(std::string(name) + " must have float-compatible strides");
  }

  const auto* ptr = static_cast<const float*>(info.ptr);
  const py::ssize_t stride = info.strides[0] / static_cast<py::ssize_t>(sizeof(float));
  std::vector<float> out(static_cast<std::size_t>(info.shape[0]));
  for (std::size_t index = 0; index < out.size(); ++index) {
    out[index] = *(ptr + static_cast<py::ssize_t>(index) * stride);
  }
  return out;
}

std::vector<float> ArrayToFlatFloatVector(py::array_t<float, py::array::forcecast> values,
                                          const char* name) {
  const py::buffer_info info = values.request();
  if (info.ndim != 1 && info.ndim != 2) {
    throw std::invalid_argument(std::string(name) + " must be a 1D or 2D NumPy array");
  }
  if (info.itemsize != static_cast<py::ssize_t>(sizeof(float))) {
    throw std::invalid_argument(std::string(name) + " must have float32-compatible items");
  }

  const auto* ptr = static_cast<const float*>(info.ptr);
  std::vector<float> out(static_cast<std::size_t>(info.size));
  if (info.size == 0) {
    return out;
  }

  if (info.ndim == 1) {
    if (info.strides[0] % static_cast<py::ssize_t>(sizeof(float)) != 0) {
      throw std::invalid_argument(std::string(name) + " must have float-compatible strides");
    }
    const py::ssize_t stride = info.strides[0] / static_cast<py::ssize_t>(sizeof(float));
    for (std::size_t index = 0; index < out.size(); ++index) {
      out[index] = *(ptr + static_cast<py::ssize_t>(index) * stride);
    }
    return out;
  }

  if (info.strides[0] % static_cast<py::ssize_t>(sizeof(float)) != 0 ||
      info.strides[1] % static_cast<py::ssize_t>(sizeof(float)) != 0) {
    throw std::invalid_argument(std::string(name) + " must have float-compatible strides");
  }
  const py::ssize_t row_stride = info.strides[0] / static_cast<py::ssize_t>(sizeof(float));
  const py::ssize_t col_stride = info.strides[1] / static_cast<py::ssize_t>(sizeof(float));
  std::size_t offset = 0;
  for (py::ssize_t row = 0; row < info.shape[0]; ++row) {
    for (py::ssize_t col = 0; col < info.shape[1]; ++col) {
      out[offset++] = *(ptr + row * row_stride + col * col_stride);
    }
  }
  return out;
}

std::vector<std::int64_t> ArrayToInt64Vector(
    py::array_t<std::int64_t, py::array::forcecast> values,
    const char* name) {
  const py::buffer_info info = values.request();
  if (info.ndim != 1) {
    throw std::invalid_argument(std::string(name) + " must be a 1D NumPy array");
  }
  if (info.strides[0] % static_cast<py::ssize_t>(sizeof(std::int64_t)) != 0) {
    throw std::invalid_argument(std::string(name) + " must have integer-compatible strides");
  }

  const auto* ptr = static_cast<const std::int64_t*>(info.ptr);
  const py::ssize_t stride = info.strides[0] / static_cast<py::ssize_t>(sizeof(std::int64_t));
  std::vector<std::int64_t> out(static_cast<std::size_t>(info.shape[0]));
  for (std::size_t index = 0; index < out.size(); ++index) {
    out[index] = *(ptr + static_cast<py::ssize_t>(index) * stride);
  }
  return out;
}

std::vector<ctboost::RankingPair> ArrayToRankingPairs(
    py::array_t<std::int64_t, py::array::forcecast> pairs,
    py::object pairs_weight,
    std::size_t num_rows) {
  const py::buffer_info pair_info = pairs.request();
  if (pair_info.ndim != 2 || pair_info.shape[1] != 2) {
    throw std::invalid_argument("pairs must be a 2D NumPy array with shape (n_pairs, 2)");
  }
  if (pair_info.strides[0] % static_cast<py::ssize_t>(sizeof(std::int64_t)) != 0 ||
      pair_info.strides[1] % static_cast<py::ssize_t>(sizeof(std::int64_t)) != 0) {
    throw std::invalid_argument("pairs must have integer-compatible strides");
  }

  std::vector<float> resolved_weights;
  if (!pairs_weight.is_none()) {
    resolved_weights = ArrayToVector(
        pairs_weight.cast<py::array_t<float, py::array::forcecast>>(), "pairs_weight");
  }

  const std::size_t pair_count = static_cast<std::size_t>(pair_info.shape[0]);
  if (!resolved_weights.empty() && resolved_weights.size() != pair_count) {
    throw std::invalid_argument("pairs_weight size must match the number of pairs");
  }

  const auto* pair_ptr = static_cast<const std::int64_t*>(pair_info.ptr);
  const py::ssize_t row_stride = pair_info.strides[0] / static_cast<py::ssize_t>(sizeof(std::int64_t));
  const py::ssize_t col_stride = pair_info.strides[1] / static_cast<py::ssize_t>(sizeof(std::int64_t));
  std::vector<ctboost::RankingPair> out(pair_count);
  for (std::size_t index = 0; index < pair_count; ++index) {
    const py::ssize_t row_offset = static_cast<py::ssize_t>(index) * row_stride;
    const std::int64_t winner = *(pair_ptr + row_offset);
    const std::int64_t loser = *(pair_ptr + row_offset + col_stride);
    if (winner < 0 || loser < 0 || static_cast<std::size_t>(winner) >= num_rows ||
        static_cast<std::size_t>(loser) >= num_rows) {
      throw std::invalid_argument("pairs must reference valid row indices");
    }
    out[index] = ctboost::RankingPair{
        winner,
        loser,
        resolved_weights.empty() ? 1.0F : resolved_weights[index],
    };
  }
  return out;
}

std::vector<std::uint16_t> ArrayToBinVector(
    py::array_t<std::int64_t, py::array::forcecast> values,
    const char* name) {
  const py::buffer_info info = values.request();
  if (info.ndim != 1) {
    throw std::invalid_argument(std::string(name) + " must be a 1D NumPy array");
  }
  if (info.strides[0] % static_cast<py::ssize_t>(sizeof(std::int64_t)) != 0) {
    throw std::invalid_argument(std::string(name) + " must have integer-compatible strides");
  }

  const auto* ptr = static_cast<const std::int64_t*>(info.ptr);
  const py::ssize_t stride = info.strides[0] / static_cast<py::ssize_t>(sizeof(std::int64_t));
  std::vector<std::uint16_t> out(static_cast<std::size_t>(info.shape[0]));
  for (std::size_t index = 0; index < out.size(); ++index) {
    const std::int64_t value = *(ptr + static_cast<py::ssize_t>(index) * stride);
    if (value < 0 || value > static_cast<std::int64_t>(std::numeric_limits<std::uint16_t>::max())) {
      throw std::invalid_argument(std::string(name) + " must contain values in [0, 65535]");
    }
    out[index] = static_cast<std::uint16_t>(value);
  }
  return out;
}

}  // namespace ctboost::bindings
