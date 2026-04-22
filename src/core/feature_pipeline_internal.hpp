#pragma once

#include "ctboost/feature_pipeline.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <pybind11/stl.h>

namespace py = pybind11;

namespace ctboost::detail {

inline constexpr const char* kMissingKey = "__ctboost_missing__";
inline constexpr const char* kOtherKey = "__ctboost_other__";

struct MatrixView {
  py::array array;
  py::buffer_info info;
  PyObject** data{nullptr};
  py::ssize_t row_stride{0};
  py::ssize_t col_stride{0};
  std::size_t rows{0};
  std::size_t cols{0};
};

py::module_& NumpyModule();
py::module_& HashlibModule();
py::object& NumpyFloatingType();
py::array EnsureObjectMatrix(py::array raw_matrix);
MatrixView MakeMatrixView(py::array object_array);
py::handle MatrixValue(const MatrixView& matrix, std::size_t row, std::size_t col);
bool IsMissing(const py::handle& value);
std::string NormalizeKey(const py::handle& value);
std::vector<std::string> NormalizeEmbeddingStats(py::object embedding_stats);
py::object NormalizeOptionalSequence(py::object values);
py::object NormalizeOptionalCombinations(py::object values);
std::string CanonicalCtrType(std::string ctr_type);
py::object NormalizeOptionalCtrTypes(py::object values);
std::string OneHotOutputName(const std::string& prefix, const std::string& key);
std::vector<std::string> ResolveCtrTypeList(const py::object& values, bool default_mean);
std::vector<std::string> BuildBucketKeys(
    const std::unordered_map<std::string, std::size_t>& key_counts,
    int max_cat_threshold,
    bool* out_has_other_bucket);
std::vector<float> ArrayToFloatVector(py::array_t<float, py::array::forcecast> values,
                                      const char* name);
std::string JoinNormalizedKey(const MatrixView& matrix,
                              std::size_t row,
                              const std::vector<int>& source_indices);
std::uint64_t BytesToLittleEndianU64(const py::bytes& digest);
std::vector<float> EmbeddingValues(const py::handle& value);
std::pair<int, std::vector<float>> FitTargetMode(const std::vector<float>& labels);
std::vector<std::string> ExtractAsciiTokens(std::string text);
py::list VectorToPyList(const std::vector<int>& values);
py::list VectorToPyList(const std::vector<std::string>& values);
py::list VectorToPyList(const std::vector<float>& values);

}  // namespace ctboost::detail
