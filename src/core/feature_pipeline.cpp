#include "ctboost/feature_pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <pybind11/stl.h>

namespace py = pybind11;

namespace ctboost {

namespace {

constexpr const char* kMissingKey = "__ctboost_missing__";

struct MatrixView {
  py::array array;
  py::buffer_info info;
  PyObject** data{nullptr};
  py::ssize_t row_stride{0};
  py::ssize_t col_stride{0};
  std::size_t rows{0};
  std::size_t cols{0};
};

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

}  // namespace

NativeFeaturePipeline::NativeFeaturePipeline(py::object cat_features,
                                             bool ordered_ctr,
                                             py::object categorical_combinations,
                                             bool pairwise_categorical_combinations,
                                             py::object text_features,
                                             int text_hash_dim,
                                             py::object embedding_features,
                                             py::object embedding_stats,
                                             double ctr_prior_strength,
                                             int random_seed)
    : cat_features_(NormalizeOptionalSequence(std::move(cat_features))),
      ordered_ctr_(ordered_ctr),
      categorical_combinations_(NormalizeOptionalCombinations(std::move(categorical_combinations))),
      pairwise_categorical_combinations_(pairwise_categorical_combinations),
      text_features_(NormalizeOptionalSequence(std::move(text_features))),
      text_hash_dim_(text_hash_dim),
      embedding_features_(NormalizeOptionalSequence(std::move(embedding_features))),
      embedding_stats_(VectorToPyList(NormalizeEmbeddingStats(std::move(embedding_stats)))),
      ctr_prior_strength_(ctr_prior_strength),
      random_seed_(random_seed) {}

void NativeFeaturePipeline::fit_array(py::array raw_matrix,
                                      py::array_t<float, py::array::forcecast> labels,
                                      py::object feature_names) {
  FitInternal(std::move(raw_matrix), std::move(labels), std::move(feature_names));
}

py::tuple NativeFeaturePipeline::fit_transform_array(
    py::array raw_matrix,
    py::array_t<float, py::array::forcecast> labels,
    py::object feature_names) {
  FitInternal(raw_matrix, std::move(labels), feature_names);
  return TransformInternal(std::move(raw_matrix), std::move(feature_names), true);
}

py::tuple NativeFeaturePipeline::transform_array(py::array raw_matrix,
                                                 py::object feature_names) const {
  return TransformInternal(std::move(raw_matrix), std::move(feature_names), false);
}

std::vector<int> NativeFeaturePipeline::ResolveIndices(const py::object& selectors) const {
  if (selectors.is_none()) {
    return {};
  }

  std::unordered_map<std::string, int> name_to_index;
  if (feature_names_in_.has_value()) {
    for (std::size_t index = 0; index < feature_names_in_->size(); ++index) {
      name_to_index.emplace((*feature_names_in_)[index], static_cast<int>(index));
    }
  }

  std::vector<int> resolved;
  for (const py::handle selector : selectors) {
    if (py::isinstance<py::str>(selector)) {
      const std::string name = py::str(selector).cast<std::string>();
      const auto it = name_to_index.find(name);
      if (it == name_to_index.end()) {
        throw std::invalid_argument("unknown feature selector: " + name);
      }
      resolved.push_back(it->second);
      continue;
    }

    const int index = py::cast<int>(selector);
    if (index < 0 || index >= n_features_in_) {
      throw std::invalid_argument("feature selector is out of bounds");
    }
    resolved.push_back(index);
  }

  std::sort(resolved.begin(), resolved.end());
  resolved.erase(std::unique(resolved.begin(), resolved.end()), resolved.end());
  return resolved;
}

void NativeFeaturePipeline::RefreshCombinationSourceIndices() {
  combination_source_indices_.clear();
  if (n_features_in_ < 0) {
    return;
  }

  if (!categorical_combinations_.is_none()) {
    for (const py::handle combination : categorical_combinations_) {
      py::list selectors;
      for (const py::handle selector : combination) {
        selectors.append(py::reinterpret_borrow<py::object>(selector));
      }
      std::vector<int> indices = ResolveIndices(std::move(selectors));
      if (indices.size() >= 2) {
        combination_source_indices_.push_back(std::move(indices));
      }
    }
  }

  if (pairwise_categorical_combinations_) {
    const std::vector<int> cat_indices = ResolveIndices(cat_features_);
    for (std::size_t left = 0; left < cat_indices.size(); ++left) {
      for (std::size_t right = left + 1; right < cat_indices.size(); ++right) {
        combination_source_indices_.push_back({cat_indices[left], cat_indices[right]});
      }
    }
  }

  std::sort(combination_source_indices_.begin(), combination_source_indices_.end());
  combination_source_indices_.erase(
      std::unique(combination_source_indices_.begin(), combination_source_indices_.end()),
      combination_source_indices_.end());
}

void NativeFeaturePipeline::FitInternal(py::array raw_matrix,
                                        py::array_t<float, py::array::forcecast> labels,
                                        py::object feature_names) {
  const MatrixView matrix = MakeMatrixView(EnsureObjectMatrix(std::move(raw_matrix)));
  const std::vector<float> label_values = ArrayToFloatVector(std::move(labels), "labels");
  if (label_values.size() != matrix.rows) {
    throw std::invalid_argument("label size must match the number of rows");
  }

  n_features_in_ = static_cast<int>(matrix.cols);
  if (feature_names.is_none()) {
    feature_names_in_.reset();
  } else {
    feature_names_in_ = py::cast<std::vector<std::string>>(feature_names);
  }

  const std::vector<int> cat_indices = ResolveIndices(cat_features_);
  const std::vector<int> text_indices = ResolveIndices(text_features_);
  const std::vector<int> embedding_indices = ResolveIndices(embedding_features_);

  {
    const std::unordered_set<int> text_index_set(text_indices.begin(), text_indices.end());
    for (int feature_index : embedding_indices) {
      if (text_index_set.find(feature_index) != text_index_set.end()) {
        throw std::invalid_argument("text_features and embedding_features cannot overlap");
      }
    }
  }

  const std::unordered_set<int> text_reserved(text_indices.begin(), text_indices.end());
  const std::unordered_set<int> embedding_reserved(embedding_indices.begin(), embedding_indices.end());
  numeric_indices_.clear();
  for (int feature_index = 0; feature_index < n_features_in_; ++feature_index) {
    if (text_reserved.find(feature_index) == text_reserved.end() &&
        embedding_reserved.find(feature_index) == embedding_reserved.end()) {
      numeric_indices_.push_back(feature_index);
    }
  }

  auto feature_name = [this](int index) {
    if (!feature_names_in_.has_value()) {
      return std::string("f") + std::to_string(index);
    }
    return (*feature_names_in_)[static_cast<std::size_t>(index)];
  };

  output_feature_names_.clear();
  cat_feature_indices_.clear();
  categorical_states_.clear();
  combination_states_.clear();
  ctr_states_.clear();
  text_states_.clear();
  embedding_states_.clear();
  training_ctr_columns_.clear();
  text_hash_cache_.clear();

  const std::unordered_set<int> cat_index_set(cat_indices.begin(), cat_indices.end());
  for (int feature_index : numeric_indices_) {
    const std::string name = feature_name(feature_index);
    if (cat_index_set.find(feature_index) == cat_index_set.end()) {
      output_feature_names_.push_back(name);
      continue;
    }

    std::vector<std::string> keys(matrix.rows);
    for (std::size_t row = 0; row < matrix.rows; ++row) {
      keys[row] = NormalizeKey(MatrixValue(matrix, row, static_cast<std::size_t>(feature_index)));
    }
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());

    CategoricalEncoderState state;
    state.source_index = feature_index;
    state.output_name = name;
    for (std::size_t code = 0; code < keys.size(); ++code) {
      state.mapping.emplace(keys[code], static_cast<float>(code));
    }
    categorical_states_.push_back(std::move(state));
    cat_feature_indices_.push_back(static_cast<int>(output_feature_names_.size()));
    output_feature_names_.push_back(name);
  }

  RefreshCombinationSourceIndices();
  for (const auto& source_indices : combination_source_indices_) {
    std::vector<std::string> keys(matrix.rows);
    for (std::size_t row = 0; row < matrix.rows; ++row) {
      keys[row] = JoinNormalizedKey(matrix, row, source_indices);
    }
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());

    CategoricalEncoderState state;
    for (std::size_t index = 0; index < source_indices.size(); ++index) {
      if (index > 0) {
        state.output_name += "_x_";
      }
      state.output_name += feature_name(source_indices[index]);
    }
    for (std::size_t code = 0; code < keys.size(); ++code) {
      state.mapping.emplace(keys[code], static_cast<float>(code));
    }
    combination_states_.push_back(std::move(state));
    cat_feature_indices_.push_back(static_cast<int>(output_feature_names_.size()));
    output_feature_names_.push_back(combination_states_.back().output_name);
  }

  const auto [target_width, target_prior] = FitTargetMode(label_values);
  if (ordered_ctr_) {
    std::vector<std::size_t> permutation(matrix.rows);
    std::iota(permutation.begin(), permutation.end(), 0U);
    std::mt19937_64 rng(static_cast<std::uint64_t>(random_seed_));
    std::shuffle(permutation.begin(), permutation.end(), rng);

    std::vector<std::pair<std::vector<int>, std::string>> ctr_sources;
    for (const auto& state : categorical_states_) {
      ctr_sources.push_back({{state.source_index}, state.output_name});
    }
    for (std::size_t index = 0; index < combination_states_.size(); ++index) {
      ctr_sources.push_back({combination_source_indices_[index], combination_states_[index].output_name});
    }

    for (const auto& source : ctr_sources) {
      std::vector<std::string> output_names;
      if (target_width == 1) {
        output_names.push_back(source.second + "_ctr");
      } else {
        for (int class_index = 0; class_index < target_width; ++class_index) {
          output_names.push_back(source.second + "_ctr_class" + std::to_string(class_index));
        }
      }

      std::unordered_map<std::string, int> total_counts;
      std::unordered_map<std::string, std::vector<float>> total_sums;
      std::unordered_map<std::string, int> running_counts;
      std::unordered_map<std::string, std::vector<float>> running_sums;
      std::vector<std::vector<float>> training_columns(
          output_names.size(), std::vector<float>(matrix.rows, 0.0F));

      for (std::size_t row : permutation) {
        const std::string key = JoinNormalizedKey(matrix, row, source.first);
        const float current_count = static_cast<float>(running_counts[key]);
        auto& current_sums = running_sums[key];
        if (current_sums.empty()) {
          current_sums.assign(static_cast<std::size_t>(target_width), 0.0F);
        }

        const float denominator = current_count + static_cast<float>(ctr_prior_strength_);
        for (int output_index = 0; output_index < target_width; ++output_index) {
          const float numerator = current_sums[static_cast<std::size_t>(output_index)] +
                                  static_cast<float>(ctr_prior_strength_) *
                                      target_prior[static_cast<std::size_t>(output_index)];
          training_columns[static_cast<std::size_t>(output_index)][row] =
              numerator / std::max(denominator, 1.0F);
        }

        total_counts[key] += 1;
        auto& total_sum_values = total_sums[key];
        if (total_sum_values.empty()) {
          total_sum_values.assign(static_cast<std::size_t>(target_width), 0.0F);
        }
        if (target_width == 1) {
          total_sum_values[0] += label_values[row];
          current_sums[0] += label_values[row];
        } else {
          const int class_index = static_cast<int>(std::llround(label_values[row]));
          total_sum_values[static_cast<std::size_t>(class_index)] += 1.0F;
          current_sums[static_cast<std::size_t>(class_index)] += 1.0F;
        }
        running_counts[key] += 1;
      }

      CtrState state;
      state.source_indices = source.first;
      state.output_names = output_names;
      state.prior_values = target_prior;
      state.total_counts = std::move(total_counts);
      state.total_sums = std::move(total_sums);
      ctr_states_.push_back(std::move(state));
      for (auto& column : training_columns) {
        training_ctr_columns_.push_back(std::move(column));
      }
      output_feature_names_.insert(output_feature_names_.end(), output_names.begin(), output_names.end());
    }
  }

  for (int feature_index : text_indices) {
    TextState state;
    state.source_index = feature_index;
    state.prefix = feature_name(feature_index);
    text_states_.push_back(state);
    for (int hash_index = 0; hash_index < text_hash_dim_; ++hash_index) {
      output_feature_names_.push_back(state.prefix + "_hash" + std::to_string(hash_index));
    }
  }

  const std::vector<std::string> embedding_stats = NormalizeEmbeddingStats(embedding_stats_);
  for (int feature_index : embedding_indices) {
    EmbeddingState state;
    state.source_index = feature_index;
    state.prefix = feature_name(feature_index);
    state.stats = embedding_stats;
    embedding_states_.push_back(state);
    for (const std::string& stat : state.stats) {
      output_feature_names_.push_back(state.prefix + "_" + stat);
    }
  }
}

py::tuple NativeFeaturePipeline::TransformInternal(py::array raw_matrix,
                                                   py::object feature_names,
                                                   bool use_training_ctr_columns) const {
  if (n_features_in_ < 0) {
    throw std::invalid_argument("feature pipeline must be fitted before calling transform");
  }

  const MatrixView matrix = MakeMatrixView(EnsureObjectMatrix(std::move(raw_matrix)));
  if (static_cast<int>(matrix.cols) != n_features_in_) {
    throw std::invalid_argument("input feature count does not match the fitted feature pipeline");
  }
  if (feature_names_in_.has_value() && !feature_names.is_none()) {
    if (py::cast<std::vector<std::string>>(feature_names) != feature_names_in_.value()) {
      throw std::invalid_argument("input feature names do not match the fitted feature pipeline");
    }
  }

  const std::size_t row_count = matrix.rows;
  const std::size_t column_count = output_feature_names_.size();
  float* data = new float[std::max<std::size_t>(row_count * column_count, 1U)];
  std::fill(data, data + (row_count * column_count), 0.0F);
  py::capsule owner(data, [](void* ptr) { delete[] static_cast<float*>(ptr); });
  py::array_t<float> transformed(
      {static_cast<py::ssize_t>(row_count), static_cast<py::ssize_t>(column_count)},
      {static_cast<py::ssize_t>(sizeof(float)),
       static_cast<py::ssize_t>(row_count * sizeof(float))},
      data,
      owner);

  auto write_column_value = [data, row_count](std::size_t row, std::size_t col, float value) {
    data[col * row_count + row] = value;
  };

  std::unordered_map<int, const CategoricalEncoderState*> categorical_by_index;
  for (const auto& state : categorical_states_) {
    categorical_by_index.emplace(state.source_index, &state);
  }

  std::size_t column_index = 0;
  for (int feature_index : numeric_indices_) {
    const auto categorical_it = categorical_by_index.find(feature_index);
    if (categorical_it == categorical_by_index.end()) {
      for (std::size_t row = 0; row < row_count; ++row) {
        write_column_value(row,
                           column_index,
                           py::cast<float>(MatrixValue(matrix, row, static_cast<std::size_t>(feature_index))));
      }
    } else {
      const auto& mapping = categorical_it->second->mapping;
      for (std::size_t row = 0; row < row_count; ++row) {
        const std::string key = NormalizeKey(MatrixValue(matrix, row, static_cast<std::size_t>(feature_index)));
        const auto code_it = mapping.find(key);
        write_column_value(
            row,
            column_index,
            code_it == mapping.end() ? std::numeric_limits<float>::quiet_NaN() : code_it->second);
      }
    }
    ++column_index;
  }

  for (std::size_t combination_index = 0; combination_index < combination_states_.size(); ++combination_index) {
    const auto& state = combination_states_[combination_index];
    const auto& source_indices = combination_source_indices_[combination_index];
    for (std::size_t row = 0; row < row_count; ++row) {
      const std::string key = JoinNormalizedKey(matrix, row, source_indices);
      const auto code_it = state.mapping.find(key);
      write_column_value(
          row,
          column_index,
          code_it == state.mapping.end() ? std::numeric_limits<float>::quiet_NaN() : code_it->second);
    }
    ++column_index;
  }

  if (use_training_ctr_columns) {
    for (const auto& column : training_ctr_columns_) {
      for (std::size_t row = 0; row < row_count; ++row) {
        write_column_value(row, column_index, column[row]);
      }
      ++column_index;
    }
  } else {
    for (const auto& state : ctr_states_) {
      for (std::size_t output_index = 0; output_index < state.output_names.size(); ++output_index) {
        for (std::size_t row = 0; row < row_count; ++row) {
          const std::string key = JoinNormalizedKey(matrix, row, state.source_indices);
          const auto count_it = state.total_counts.find(key);
          const auto sums_it = state.total_sums.find(key);
          const float count =
              count_it == state.total_counts.end() ? 0.0F : static_cast<float>(count_it->second);
          const float summed =
              sums_it == state.total_sums.end() ? 0.0F : sums_it->second[output_index];
          const float denominator = count + static_cast<float>(ctr_prior_strength_);
          const float numerator =
              summed + static_cast<float>(ctr_prior_strength_) * state.prior_values[output_index];
          write_column_value(row, column_index, numerator / std::max(denominator, 1.0F));
        }
        ++column_index;
      }
    }
  }

  for (const auto& state : text_states_) {
    const std::size_t text_column_start = column_index;
    for (std::size_t row = 0; row < row_count; ++row) {
      const py::handle raw_value = MatrixValue(matrix, row, static_cast<std::size_t>(state.source_index));
      if (IsMissing(raw_value)) {
        continue;
      }
      const std::string text = py::str(raw_value).cast<std::string>();
      for (const std::string& token : ExtractAsciiTokens(text)) {
        auto cache_it = text_hash_cache_.find(token);
        if (cache_it == text_hash_cache_.end()) {
          const py::bytes digest =
              HashlibModule()
                  .attr("blake2b")(py::bytes(token), py::arg("digest_size") = 8)
                  .attr("digest")()
                  .cast<py::bytes>();
          cache_it = text_hash_cache_.emplace(token, BytesToLittleEndianU64(digest)).first;
        }
        const std::size_t bucket =
            static_cast<std::size_t>(cache_it->second % static_cast<std::uint64_t>(text_hash_dim_));
        data[(text_column_start + bucket) * row_count + row] += 1.0F;
      }
    }
    column_index += static_cast<std::size_t>(text_hash_dim_);
  }

  for (const auto& state : embedding_states_) {
    const std::size_t embedding_column_start = column_index;
    for (std::size_t row = 0; row < row_count; ++row) {
      const std::vector<float> values =
          EmbeddingValues(MatrixValue(matrix, row, static_cast<std::size_t>(state.source_index)));
      if (values.empty()) {
        continue;
      }

      const auto [min_it, max_it] = std::minmax_element(values.begin(), values.end());
      const float sum = std::accumulate(values.begin(), values.end(), 0.0F);
      const float mean = sum / static_cast<float>(values.size());
      float sum_squared = 0.0F;
      float l2 = 0.0F;
      for (float value : values) {
        const float delta = value - mean;
        sum_squared += delta * delta;
        l2 += value * value;
      }
      const float stddev = std::sqrt(sum_squared / static_cast<float>(values.size()));
      l2 = std::sqrt(l2);

      for (std::size_t stat_index = 0; stat_index < state.stats.size(); ++stat_index) {
        float stat_value = 0.0F;
        const std::string& stat = state.stats[stat_index];
        if (stat == "mean") {
          stat_value = mean;
        } else if (stat == "std") {
          stat_value = stddev;
        } else if (stat == "min") {
          stat_value = *min_it;
        } else if (stat == "max") {
          stat_value = *max_it;
        } else if (stat == "l2") {
          stat_value = l2;
        } else if (stat == "sum") {
          stat_value = sum;
        } else if (stat == "dim") {
          stat_value = static_cast<float>(values.size());
        }
        write_column_value(row, embedding_column_start + stat_index, stat_value);
      }
    }
    column_index += state.stats.size();
  }

  return py::make_tuple(std::move(transformed),
                        VectorToPyList(cat_feature_indices_),
                        VectorToPyList(output_feature_names_));
}

py::dict NativeFeaturePipeline::to_state() const {
  py::dict state;
  state["cat_features"] = cat_features_;
  state["ordered_ctr"] = ordered_ctr_;
  state["categorical_combinations"] = categorical_combinations_;
  state["pairwise_categorical_combinations"] = pairwise_categorical_combinations_;
  state["text_features"] = text_features_;
  state["text_hash_dim"] = text_hash_dim_;
  state["embedding_features"] = embedding_features_;
  state["embedding_stats"] = embedding_stats_;
  state["ctr_prior_strength"] = ctr_prior_strength_;
  state["random_seed"] = random_seed_;
  if (feature_names_in_.has_value()) {
    state["feature_names_in_"] = VectorToPyList(feature_names_in_.value());
  } else {
    state["feature_names_in_"] = py::none();
  }
  if (n_features_in_ >= 0) {
    state["n_features_in_"] = py::int_(n_features_in_);
  } else {
    state["n_features_in_"] = py::none();
  }
  state["cat_feature_indices_"] = VectorToPyList(cat_feature_indices_);
  state["output_feature_names_"] = VectorToPyList(output_feature_names_);
  state["numeric_indices"] = VectorToPyList(numeric_indices_);

  py::list categorical_states;
  for (const auto& categorical_state : categorical_states_) {
    py::dict item;
    item["source_index"] = categorical_state.source_index;
    item["output_name"] = categorical_state.output_name;
    py::dict mapping;
    for (const auto& [key, value] : categorical_state.mapping) {
      mapping[py::str(key)] = value;
    }
    item["mapping"] = std::move(mapping);
    categorical_states.append(std::move(item));
  }
  state["categorical_states"] = std::move(categorical_states);

  py::list combination_states;
  for (const auto& combination_state : combination_states_) {
    py::dict item;
    item["output_name"] = combination_state.output_name;
    py::dict mapping;
    for (const auto& [key, value] : combination_state.mapping) {
      mapping[py::str(key)] = value;
    }
    item["mapping"] = std::move(mapping);
    combination_states.append(std::move(item));
  }
  state["combination_states"] = std::move(combination_states);

  py::list ctr_states;
  for (const auto& ctr_state : ctr_states_) {
    py::dict item;
    item["source_indices"] = VectorToPyList(ctr_state.source_indices);
    item["output_names"] = VectorToPyList(ctr_state.output_names);
    item["prior_values"] = VectorToPyList(ctr_state.prior_values);
    py::dict total_counts;
    for (const auto& [key, value] : ctr_state.total_counts) {
      total_counts[py::str(key)] = value;
    }
    item["total_counts"] = std::move(total_counts);
    py::dict total_sums;
    for (const auto& [key, values] : ctr_state.total_sums) {
      total_sums[py::str(key)] = VectorToPyList(values);
    }
    item["total_sums"] = std::move(total_sums);
    ctr_states.append(std::move(item));
  }
  state["ctr_states"] = std::move(ctr_states);

  py::list text_states;
  for (const auto& text_state : text_states_) {
    py::dict item;
    item["source_index"] = text_state.source_index;
    item["prefix"] = text_state.prefix;
    text_states.append(std::move(item));
  }
  state["text_states"] = std::move(text_states);

  py::list embedding_states;
  for (const auto& embedding_state : embedding_states_) {
    py::dict item;
    item["source_index"] = embedding_state.source_index;
    item["prefix"] = embedding_state.prefix;
    item["stats"] = VectorToPyList(embedding_state.stats);
    embedding_states.append(std::move(item));
  }
  state["embedding_states"] = std::move(embedding_states);
  return state;
}

void NativeFeaturePipeline::LoadState(const py::dict& state) {
  cat_features_ = NormalizeOptionalSequence(
      state.contains("cat_features") ? py::reinterpret_borrow<py::object>(state["cat_features"])
                                     : py::none());
  ordered_ctr_ = state.contains("ordered_ctr") ? py::cast<bool>(state["ordered_ctr"]) : false;
  categorical_combinations_ =
      NormalizeOptionalCombinations(state.contains("categorical_combinations")
                                        ? py::reinterpret_borrow<py::object>(
                                              state["categorical_combinations"])
                                        : py::none());
  pairwise_categorical_combinations_ =
      state.contains("pairwise_categorical_combinations")
          ? py::cast<bool>(state["pairwise_categorical_combinations"])
          : false;
  text_features_ = NormalizeOptionalSequence(
      state.contains("text_features") ? py::reinterpret_borrow<py::object>(state["text_features"])
                                      : py::none());
  text_hash_dim_ = state.contains("text_hash_dim") ? py::cast<int>(state["text_hash_dim"]) : 64;
  embedding_features_ =
      NormalizeOptionalSequence(state.contains("embedding_features")
                                    ? py::reinterpret_borrow<py::object>(state["embedding_features"])
                                    : py::none());
  embedding_stats_ = VectorToPyList(
      NormalizeEmbeddingStats(state.contains("embedding_stats")
                                  ? py::reinterpret_borrow<py::object>(state["embedding_stats"])
                                  : py::none()));
  ctr_prior_strength_ =
      state.contains("ctr_prior_strength") ? py::cast<double>(state["ctr_prior_strength"]) : 1.0;
  random_seed_ = state.contains("random_seed") ? py::cast<int>(state["random_seed"]) : 0;

  if (state.contains("feature_names_in_") && !state["feature_names_in_"].is_none()) {
    feature_names_in_ = py::cast<std::vector<std::string>>(state["feature_names_in_"]);
  } else {
    feature_names_in_.reset();
  }
  n_features_in_ = (state.contains("n_features_in_") && !state["n_features_in_"].is_none())
                       ? py::cast<int>(state["n_features_in_"])
                       : -1;
  cat_feature_indices_ = state.contains("cat_feature_indices_")
                             ? py::cast<std::vector<int>>(state["cat_feature_indices_"])
                             : std::vector<int>{};
  output_feature_names_ = state.contains("output_feature_names_")
                              ? py::cast<std::vector<std::string>>(state["output_feature_names_"])
                              : std::vector<std::string>{};
  numeric_indices_ = state.contains("numeric_indices")
                         ? py::cast<std::vector<int>>(state["numeric_indices"])
                         : std::vector<int>{};

  categorical_states_.clear();
  if (state.contains("categorical_states")) {
    for (const py::handle item_handle : py::cast<py::list>(state["categorical_states"])) {
      const py::dict item = item_handle.cast<py::dict>();
      CategoricalEncoderState categorical_state;
      categorical_state.source_index = py::cast<int>(item["source_index"]);
      categorical_state.output_name = py::cast<std::string>(item["output_name"]);
      for (const auto& mapping_item : py::cast<py::dict>(item["mapping"])) {
        categorical_state.mapping.emplace(py::cast<std::string>(mapping_item.first),
                                          py::cast<float>(mapping_item.second));
      }
      categorical_states_.push_back(std::move(categorical_state));
    }
  }

  combination_states_.clear();
  if (state.contains("combination_states")) {
    for (const py::handle item_handle : py::cast<py::list>(state["combination_states"])) {
      const py::dict item = item_handle.cast<py::dict>();
      CategoricalEncoderState combination_state;
      combination_state.output_name = py::cast<std::string>(item["output_name"]);
      for (const auto& mapping_item : py::cast<py::dict>(item["mapping"])) {
        combination_state.mapping.emplace(py::cast<std::string>(mapping_item.first),
                                          py::cast<float>(mapping_item.second));
      }
      combination_states_.push_back(std::move(combination_state));
    }
  }
  RefreshCombinationSourceIndices();

  ctr_states_.clear();
  if (state.contains("ctr_states")) {
    for (const py::handle item_handle : py::cast<py::list>(state["ctr_states"])) {
      const py::dict item = item_handle.cast<py::dict>();
      CtrState ctr_state;
      ctr_state.source_indices = py::cast<std::vector<int>>(item["source_indices"]);
      ctr_state.output_names = py::cast<std::vector<std::string>>(item["output_names"]);
      ctr_state.prior_values = py::cast<std::vector<float>>(item["prior_values"]);
      for (const auto& count_item : py::cast<py::dict>(item["total_counts"])) {
        ctr_state.total_counts.emplace(py::cast<std::string>(count_item.first),
                                       py::cast<int>(count_item.second));
      }
      for (const auto& sum_item : py::cast<py::dict>(item["total_sums"])) {
        ctr_state.total_sums.emplace(py::cast<std::string>(sum_item.first),
                                     py::cast<std::vector<float>>(sum_item.second));
      }
      ctr_states_.push_back(std::move(ctr_state));
    }
  }

  text_states_.clear();
  if (state.contains("text_states")) {
    for (const py::handle item_handle : py::cast<py::list>(state["text_states"])) {
      const py::dict item = item_handle.cast<py::dict>();
      TextState text_state;
      text_state.source_index = py::cast<int>(item["source_index"]);
      text_state.prefix = py::cast<std::string>(item["prefix"]);
      text_states_.push_back(std::move(text_state));
    }
  }

  embedding_states_.clear();
  if (state.contains("embedding_states")) {
    for (const py::handle item_handle : py::cast<py::list>(state["embedding_states"])) {
      const py::dict item = item_handle.cast<py::dict>();
      EmbeddingState embedding_state;
      embedding_state.source_index = py::cast<int>(item["source_index"]);
      embedding_state.prefix = py::cast<std::string>(item["prefix"]);
      embedding_state.stats = py::cast<std::vector<std::string>>(item["stats"]);
      embedding_states_.push_back(std::move(embedding_state));
    }
  }

  training_ctr_columns_.clear();
  text_hash_cache_.clear();
}

NativeFeaturePipeline NativeFeaturePipeline::FromState(const py::dict& state) {
  NativeFeaturePipeline pipeline;
  pipeline.LoadState(state);
  return pipeline;
}

}  // namespace ctboost
