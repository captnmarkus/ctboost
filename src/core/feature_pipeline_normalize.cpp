#include "feature_pipeline_internal.hpp"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

namespace ctboost::detail {
namespace {

std::string SanitizeNameToken(std::string token) {
  if (token == kMissingKey) {
    return "missing";
  }
  if (token == kOtherKey) {
    return "other";
  }
  for (char& ch : token) {
    const bool ascii_alnum = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
                             (ch >= '0' && ch <= '9');
    if (!ascii_alnum && ch != '_') {
      ch = '_';
    }
  }
  while (!token.empty() && token.front() == '_') {
    token.erase(token.begin());
  }
  while (!token.empty() && token.back() == '_') {
    token.pop_back();
  }
  return token.empty() ? "value" : token;
}

}  // namespace

py::module_& NumpyModule() {
  static py::module_* numpy = new py::module_(py::module_::import("numpy"));
  return *numpy;
}

py::module_& HashlibModule() {
  static py::module_* hashlib = new py::module_(py::module_::import("hashlib"));
  return *hashlib;
}

py::object& NumpyFloatingType() {
  static py::object* numpy_floating = new py::object(NumpyModule().attr("floating"));
  return *numpy_floating;
}

py::array EnsureObjectMatrix(py::array raw_matrix) {
  py::array object_array =
      NumpyModule().attr("asarray")(raw_matrix, py::arg("dtype") = py::dtype("O")).cast<py::array>();
  const py::buffer_info info = object_array.request();
  if (info.ndim != 2) {
    throw std::invalid_argument("feature pipelines expect a 2D array-like input");
  }
  return object_array;
}

MatrixView MakeMatrixView(py::array object_array) {
  MatrixView view;
  view.array = std::move(object_array);
  view.info = view.array.request();
  view.data = static_cast<PyObject**>(view.info.ptr);
  view.row_stride = view.info.strides[0] / static_cast<py::ssize_t>(sizeof(PyObject*));
  view.col_stride = view.info.strides[1] / static_cast<py::ssize_t>(sizeof(PyObject*));
  view.rows = static_cast<std::size_t>(view.info.shape[0]);
  view.cols = static_cast<std::size_t>(view.info.shape[1]);
  return view;
}

py::handle MatrixValue(const MatrixView& matrix, std::size_t row, std::size_t col) {
  return py::handle(*(matrix.data + static_cast<py::ssize_t>(row) * matrix.row_stride +
                      static_cast<py::ssize_t>(col) * matrix.col_stride));
}

bool IsMissing(const py::handle& value) {
  if (value.is_none()) {
    return true;
  }
  if (py::isinstance<py::array>(value)) {
    return py::reinterpret_borrow<py::array>(value).size() == 0;
  }
  if (py::isinstance<py::float_>(value) || py::isinstance(value, NumpyFloatingType())) {
    return std::isnan(py::cast<double>(value));
  }
  return false;
}

std::string NormalizeKey(const py::handle& value) {
  if (IsMissing(value)) {
    return kMissingKey;
  }
  return py::str(value).cast<std::string>();
}

std::vector<std::string> NormalizeEmbeddingStats(py::object embedding_stats) {
  const std::vector<std::string> defaults = {"mean", "std", "min", "max", "l2"};
  if (embedding_stats.is_none()) {
    return defaults;
  }

  static const std::unordered_set<std::string> supported = {
      "mean", "std", "min", "max", "l2", "sum", "dim"};
  std::vector<std::string> resolved;
  for (const py::handle value : embedding_stats) {
    std::string stat = py::str(value).cast<std::string>();
    std::transform(stat.begin(), stat.end(), stat.begin(), [](unsigned char ch) {
      return static_cast<char>(std::tolower(ch));
    });
    if (supported.find(stat) == supported.end()) {
      throw std::invalid_argument("unsupported embedding stat: " + stat);
    }
    resolved.push_back(std::move(stat));
  }
  return resolved;
}

py::object NormalizeOptionalSequence(py::object values) {
  if (values.is_none()) {
    return py::none();
  }
  py::list normalized;
  for (const py::handle value : values) {
    normalized.append(py::reinterpret_borrow<py::object>(value));
  }
  return std::move(normalized);
}

py::object NormalizeOptionalCombinations(py::object values) {
  if (values.is_none()) {
    return py::none();
  }
  py::list normalized;
  for (const py::handle combination : values) {
    py::list resolved;
    for (const py::handle value : combination) {
      resolved.append(py::reinterpret_borrow<py::object>(value));
    }
    normalized.append(std::move(resolved));
  }
  return std::move(normalized);
}

std::string CanonicalCtrType(std::string ctr_type) {
  std::transform(ctr_type.begin(), ctr_type.end(), ctr_type.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  if (ctr_type == "mean" || ctr_type == "targetmean" || ctr_type == "target_mean") {
    return "Mean";
  }
  if (ctr_type == "frequency" || ctr_type == "featurefreq" || ctr_type == "feature_freq" ||
      ctr_type == "counter" || ctr_type == "count" || ctr_type == "freq") {
    return "Frequency";
  }
  throw std::invalid_argument("unsupported CTR type: " + ctr_type);
}

py::object NormalizeOptionalCtrTypes(py::object values) {
  if (values.is_none()) {
    return py::none();
  }
  py::list normalized;
  for (const py::handle value : values) {
    normalized.append(CanonicalCtrType(py::str(value).cast<std::string>()));
  }
  return std::move(normalized);
}

std::string OneHotOutputName(const std::string& prefix, const std::string& key) {
  return prefix + "_is_" + SanitizeNameToken(key);
}

std::vector<std::string> ResolveCtrTypeList(const py::object& values, bool default_mean) {
  std::vector<std::string> resolved;
  if (!values.is_none()) {
    for (const py::handle value : values) {
      resolved.push_back(CanonicalCtrType(py::str(value).cast<std::string>()));
    }
  }
  if (resolved.empty() && default_mean) {
    resolved.push_back("Mean");
  }
  std::sort(resolved.begin(), resolved.end());
  resolved.erase(std::unique(resolved.begin(), resolved.end()), resolved.end());
  return resolved;
}

std::vector<std::string> BuildBucketKeys(
    const std::unordered_map<std::string, std::size_t>& key_counts,
    int max_cat_threshold,
    bool* out_has_other_bucket) {
  if (out_has_other_bucket != nullptr) {
    *out_has_other_bucket = false;
  }
  if (key_counts.empty()) {
    return {};
  }

  if (max_cat_threshold <= 1 || key_counts.size() <= static_cast<std::size_t>(max_cat_threshold)) {
    std::vector<std::string> keys;
    keys.reserve(key_counts.size());
    for (const auto& [key, _] : key_counts) {
      (void)_;
      keys.push_back(key);
    }
    std::sort(keys.begin(), keys.end());
    return keys;
  }

  std::vector<std::pair<std::string, std::size_t>> ranked_keys(key_counts.begin(), key_counts.end());
  std::sort(ranked_keys.begin(),
            ranked_keys.end(),
            [](const auto& lhs, const auto& rhs) {
              if (lhs.second != rhs.second) {
                return lhs.second > rhs.second;
              }
              return lhs.first < rhs.first;
            });

  const std::size_t keep_count = static_cast<std::size_t>(std::max(1, max_cat_threshold - 1));
  std::vector<std::string> keys;
  keys.reserve(keep_count + 1U);
  for (std::size_t index = 0; index < ranked_keys.size() && index < keep_count; ++index) {
    keys.push_back(ranked_keys[index].first);
  }
  std::sort(keys.begin(), keys.end());
  keys.push_back(kOtherKey);
  if (out_has_other_bucket != nullptr) {
    *out_has_other_bucket = true;
  }
  return keys;
}

py::list VectorToPyList(const std::vector<int>& values) {
  py::list result;
  for (int value : values) {
    result.append(value);
  }
  return result;
}

py::list VectorToPyList(const std::vector<std::string>& values) {
  py::list result;
  for (const std::string& value : values) {
    result.append(value);
  }
  return result;
}

py::list VectorToPyList(const std::vector<float>& values) {
  py::list result;
  for (float value : values) {
    result.append(value);
  }
  return result;
}

}  // namespace ctboost::detail
