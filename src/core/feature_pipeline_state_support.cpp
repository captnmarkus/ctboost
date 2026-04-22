#include "feature_pipeline_internal.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace ctboost::detail {

std::vector<float> ArrayToFloatVector(py::array_t<float, py::array::forcecast> values,
                                      const char* name) {
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

std::string JoinNormalizedKey(const MatrixView& matrix,
                              std::size_t row,
                              const std::vector<int>& source_indices) {
  std::string key;
  for (std::size_t index = 0; index < source_indices.size(); ++index) {
    if (index > 0) {
      key += "||";
    }
    key += NormalizeKey(MatrixValue(matrix, row, static_cast<std::size_t>(source_indices[index])));
  }
  return key;
}

std::uint64_t BytesToLittleEndianU64(const py::bytes& digest) {
  const std::string bytes = digest;
  if (bytes.size() != 8) {
    throw std::runtime_error("unexpected BLAKE2b digest size");
  }
  std::uint64_t value = 0;
  for (std::size_t index = 0; index < bytes.size(); ++index) {
    value |= static_cast<std::uint64_t>(static_cast<unsigned char>(bytes[index])) << (index * 8U);
  }
  return value;
}

std::vector<float> EmbeddingValues(const py::handle& value) {
  if (IsMissing(value)) {
    return {};
  }
  py::array array = NumpyModule()
                        .attr("asarray")(py::reinterpret_borrow<py::object>(value),
                                         py::arg("dtype") = py::dtype::of<float>())
                        .attr("reshape")(-1)
                        .cast<py::array>();
  return ArrayToFloatVector(array.cast<py::array_t<float, py::array::forcecast>>(), "embedding");
}

std::pair<int, std::vector<float>> FitTargetMode(const std::vector<float>& labels) {
  if (labels.empty()) {
    return {1, std::vector<float>{0.0F}};
  }

  std::vector<float> unique = labels;
  std::sort(unique.begin(), unique.end());
  unique.erase(std::unique(unique.begin(), unique.end()), unique.end());

  const bool all_integral = std::all_of(unique.begin(), unique.end(), [](float value) {
    return std::fabs(value - std::round(value)) <= 1.0e-6F;
  });
  if (unique.size() > 2 && all_integral && unique.front() >= 0.0F) {
    const int num_classes = static_cast<int>(std::llround(unique.back())) + 1;
    if (num_classes == static_cast<int>(unique.size())) {
      bool contiguous = true;
      for (int class_index = 0; class_index < num_classes; ++class_index) {
        if (static_cast<int>(std::llround(unique[static_cast<std::size_t>(class_index)])) != class_index) {
          contiguous = false;
          break;
        }
      }
      if (contiguous) {
        std::vector<float> prior(static_cast<std::size_t>(num_classes), 0.0F);
        for (float label : labels) {
          prior[static_cast<std::size_t>(std::llround(label))] += 1.0F;
        }
        for (float& value : prior) {
          value /= static_cast<float>(labels.size());
        }
        return {num_classes, prior};
      }
    }
  }

  const float mean = std::accumulate(labels.begin(), labels.end(), 0.0F) /
                     static_cast<float>(std::max<std::size_t>(labels.size(), 1U));
  return {1, std::vector<float>{mean}};
}

std::vector<std::string> ExtractAsciiTokens(std::string text) {
  for (char& ch : text) {
    if (ch >= 'A' && ch <= 'Z') {
      ch = static_cast<char>(ch - 'A' + 'a');
    }
  }

  std::vector<std::string> tokens;
  std::string current;
  for (unsigned char ch : text) {
    const bool is_token_char =
        (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') || ch == '_';
    if (is_token_char) {
      current.push_back(static_cast<char>(ch));
    } else if (!current.empty()) {
      tokens.push_back(std::move(current));
      current.clear();
    }
  }
  if (!current.empty()) {
    tokens.push_back(std::move(current));
  }
  return tokens;
}

}  // namespace ctboost::detail
