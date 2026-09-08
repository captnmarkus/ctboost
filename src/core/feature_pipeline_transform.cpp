#include "ctboost/feature_pipeline.hpp"

#include "feature_pipeline_internal.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace ctboost {

namespace {

float CheckedCtrFloat(double value) {
  if (!std::isfinite(value) ||
      value < -static_cast<double>(std::numeric_limits<float>::max()) ||
      value > static_cast<double>(std::numeric_limits<float>::max())) {
    throw std::invalid_argument("computed CTR value exceeds the finite float range");
  }
  return static_cast<float>(value);
}

}  // namespace

py::tuple NativeFeaturePipeline::transform_array(py::array raw_matrix,
                                                 py::object feature_names) const {
  return TransformInternal(std::move(raw_matrix), std::move(feature_names), false);
}

py::tuple NativeFeaturePipeline::TransformInternal(py::array raw_matrix,
                                                   py::object feature_names,
                                                   bool use_training_ctr_columns) const {
  if (n_features_in_ < 0) {
    throw std::invalid_argument("feature pipeline must be fitted before calling transform");
  }

  if (raw_matrix.ndim() != 2) {
    throw std::invalid_argument("feature pipelines expect a 2D array-like input");
  }
  if (raw_matrix.shape(1) != n_features_in_) {
    throw std::invalid_argument("input feature count does not match the fitted feature pipeline");
  }
  if (feature_names_in_.has_value() && !feature_names.is_none()) {
    if (py::cast<std::vector<std::string>>(feature_names) != feature_names_in_.value()) {
      throw std::invalid_argument("input feature names do not match the fitted feature pipeline");
    }
  }

  const std::size_t row_count = static_cast<std::size_t>(raw_matrix.shape(0));
  const std::size_t column_count = output_feature_names_.size();
  // The token cache is only an optimization within one transform request.
  // Do not retain attacker-controlled token strings across requests.
  text_hash_cache_.clear();
  if (row_count > static_cast<std::size_t>(std::numeric_limits<py::ssize_t>::max()) ||
      column_count > static_cast<std::size_t>(std::numeric_limits<py::ssize_t>::max()) ||
      (row_count != 0U &&
       column_count > std::numeric_limits<std::size_t>::max() / row_count)) {
    throw std::invalid_argument("feature-pipeline output dimensions are too large");
  }
  const std::size_t value_count = row_count * column_count;
  float* data = new float[std::max<std::size_t>(value_count, 1U)];
  std::fill(data, data + value_count, 0.0F);
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

  const char kind = raw_matrix.dtype().kind();
  // float16 must retain scalar conversion's platform-specific error policy.
  const bool plain_numeric = kind == 'b' || kind == 'i' || kind == 'u' ||
                             (kind == 'f' && (raw_matrix.itemsize() == 4 || raw_matrix.itemsize() == 8));
  if (plain_numeric && categorical_states_.empty() && one_hot_states_.empty() &&
      combination_states_.empty() && ctr_states_.empty() && text_states_.empty() &&
      embedding_states_.empty() && numeric_indices_.size() == column_count) {
    if (kind == 'f' && raw_matrix.itemsize() == sizeof(float)) {
      // Inspect float32 bits before any NumPy widening: a signaling NaN must
      // use the legacy scalar conversion, including its warning/error policy.
      const py::buffer_info info = raw_matrix.request();
      const auto* source = static_cast<const char*>(info.ptr);
      const std::uint16_t endian_probe = 1U;
      const bool little_endian = *reinterpret_cast<const unsigned char*>(&endian_probe) == 1U;
      const char byte_order = raw_matrix.dtype().byteorder();
      const bool swap_bytes = (byte_order == '>' && little_endian) ||
                              (byte_order == '<' && !little_endian);
      bool scalar_fallback = false;
      for (std::size_t column = 0; column < column_count && !scalar_fallback; ++column) {
        const py::ssize_t source_column = numeric_indices_[column];
        for (std::size_t row = 0; row < row_count; ++row) {
          std::uint32_t bits;
          std::memcpy(&bits, source + static_cast<py::ssize_t>(row) * info.strides[0] +
                                 source_column * info.strides[1], sizeof(bits));
          if (swap_bytes) {
            bits = (bits >> 24U) | ((bits >> 8U) & 0x0000ff00U) |
                   ((bits << 8U) & 0x00ff0000U) | (bits << 24U);
          }
          if ((bits & 0x7f800000U) == 0x7f800000U && (bits & 0x007fffffU) != 0U &&
              (bits & 0x00400000U) == 0U) {
            scalar_fallback = true;
            break;
          }
          float value;
          std::memcpy(&value, &bits, sizeof(value));
          write_column_value(row, column, value);
        }
      }
      if (!scalar_fallback) {
        return py::make_tuple(std::move(transformed),
                              detail::VectorToPyList(cat_feature_indices_),
                              detail::VectorToPyList(output_feature_names_));
      }
    } else {
      // Python numeric scalars are converted through double by py::cast<float>.
      // Retain that rounding order, including for int64/uint64, without boxing
      // every input value. forcecast also handles non-native byte order.
      const auto numeric = py::array_t<double, py::array::forcecast>::ensure(raw_matrix);
      if (numeric) {
        const py::buffer_info info = numeric.request();
        const auto* source = static_cast<const char*>(info.ptr);
        for (std::size_t column = 0; column < column_count; ++column) {
          const py::ssize_t source_column = numeric_indices_[column];
          for (std::size_t row = 0; row < row_count; ++row) {
            // NumPy permits unaligned views; memcpy avoids undefined accesses.
            double value;
            std::memcpy(&value, source + static_cast<py::ssize_t>(row) * info.strides[0] +
                                    source_column * info.strides[1], sizeof(value));
            write_column_value(row, column, static_cast<float>(value));
          }
        }
        return py::make_tuple(std::move(transformed),
                              detail::VectorToPyList(cat_feature_indices_),
                              detail::VectorToPyList(output_feature_names_));
      }
      // NumPy's vectorized cast can reject signaling NaNs under np.errstate.
      // Preserve the original scalar conversion's result or exception instead.
    }
  }

  const detail::MatrixView matrix =
      detail::MakeMatrixView(detail::EnsureObjectMatrix(std::move(raw_matrix)));
  std::unordered_map<int, const CategoricalEncoderState*> categorical_by_index;
  for (const auto& state : categorical_states_) {
    categorical_by_index.emplace(state.source_index, &state);
  }
  std::unordered_map<int, const OneHotEncoderState*> one_hot_by_index;
  for (const auto& state : one_hot_states_) {
    one_hot_by_index.emplace(state.source_index, &state);
  }

  std::size_t column_index = 0;
  for (int feature_index : numeric_indices_) {
    const auto one_hot_it = one_hot_by_index.find(feature_index);
    if (one_hot_it != one_hot_by_index.end()) {
      const auto& category_keys = one_hot_it->second->category_keys;
      const std::string other_key =
          one_hot_it->second->has_other_bucket != 0U
              ? detail::OtherKey(categorical_key_encoding_version_)
              : "";
      for (std::size_t row = 0; row < row_count; ++row) {
        const std::string raw_key =
            detail::NormalizeKey(
                detail::MatrixValue(matrix, row, static_cast<std::size_t>(feature_index)),
                categorical_key_encoding_version_);
        std::string bucket_key = raw_key;
        if (one_hot_it->second->has_other_bucket != 0U &&
            std::find(category_keys.begin(), category_keys.end(), bucket_key) == category_keys.end()) {
          bucket_key = other_key;
        }
        for (std::size_t offset = 0; offset < category_keys.size(); ++offset) {
          write_column_value(
              row, column_index + offset, category_keys[offset] == bucket_key ? 1.0F : 0.0F);
        }
      }
      column_index += category_keys.size();
      continue;
    }

    const auto categorical_it = categorical_by_index.find(feature_index);
    if (categorical_it == categorical_by_index.end()) {
      for (std::size_t row = 0; row < row_count; ++row) {
        write_column_value(
            row,
            column_index,
            py::cast<float>(detail::MatrixValue(matrix, row, static_cast<std::size_t>(feature_index))));
      }
    } else {
      const auto& mapping = categorical_it->second->mapping;
      for (std::size_t row = 0; row < row_count; ++row) {
        const std::string key = detail::NormalizeKey(
            detail::MatrixValue(matrix, row, static_cast<std::size_t>(feature_index)),
            categorical_key_encoding_version_);
        const auto code_it = mapping.find(key);
        write_column_value(
            row,
            column_index,
            code_it == mapping.end()
                ? (categorical_it->second->has_other_bucket != 0U
                       ? categorical_it->second->other_value
                       : std::numeric_limits<float>::quiet_NaN())
                : code_it->second);
      }
    }
    ++column_index;
  }

  for (std::size_t combination_index = 0; combination_index < combination_states_.size(); ++combination_index) {
    const auto& state = combination_states_[combination_index];
    const auto& source_indices = combination_source_indices_[combination_index];
    for (std::size_t row = 0; row < row_count; ++row) {
      const std::string key = detail::JoinNormalizedKey(
          matrix, row, source_indices, categorical_key_encoding_version_);
      const auto code_it = state.mapping.find(key);
      write_column_value(
          row,
          column_index,
          code_it == state.mapping.end()
              ? (state.has_other_bucket != 0U ? state.other_value : std::numeric_limits<float>::quiet_NaN())
              : code_it->second);
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
      for (std::size_t row = 0; row < row_count; ++row) {
        // All class outputs use the same category and fitted sufficient statistics.
        // Resolve them once per row while keeping each output's arithmetic intact.
        const std::string key = detail::JoinNormalizedKey(
            matrix, row, state.source_indices, categorical_key_encoding_version_);
        const auto count_it = state.total_counts.find(key);
        const float count =
            count_it == state.total_counts.end() ? 0.0F : static_cast<float>(count_it->second);
        const auto sums_it = state.ctr_type == "Mean"
            ? state.total_sums.find(key) : state.total_sums.end();
        for (std::size_t output_index = 0; output_index < state.output_names.size(); ++output_index) {
          float value = 0.0F;
          if (state.ctr_type == "Mean") {
            const float summed =
                sums_it == state.total_sums.end() ? 0.0F : sums_it->second[output_index];
            const double denominator =
                static_cast<double>(count) + ctr_prior_strength_;
            const double numerator = static_cast<double>(summed) +
                                     ctr_prior_strength_ * static_cast<double>(
                                         state.prior_values[output_index]);
            value = CheckedCtrFloat(
                numerator / (ctr_smoothing_version_ == 1
                                 ? std::max(denominator, 1.0)
                                 : (denominator > 0.0 ? denominator : 1.0)));
          } else {
            const float total_rows = static_cast<float>(std::max<std::size_t>(state.total_rows, 1U));
            const float global_frequency = count / total_rows;
            const double denominator =
                static_cast<double>(total_rows) + ctr_prior_strength_;
            const double numerator = static_cast<double>(count) +
                                     ctr_prior_strength_ * static_cast<double>(
                                         global_frequency);
            value = CheckedCtrFloat(
                numerator / std::max(denominator, 1.0));
          }
          write_column_value(row, column_index + output_index, value);
        }
      }
      column_index += state.output_names.size();
    }
  }

  for (const auto& state : text_states_) {
    const std::size_t text_column_start = column_index;
    for (std::size_t row = 0; row < row_count; ++row) {
      const py::handle raw_value = detail::MatrixValue(matrix, row, static_cast<std::size_t>(state.source_index));
      if (detail::IsMissing(raw_value)) {
        continue;
      }
      const std::string text = py::str(raw_value).cast<std::string>();
      for (const std::string& token : detail::ExtractTextTokens(text,
                                                               text_tokenizer_,
                                                               text_ngram_min_,
                                                               text_ngram_max_,
                                                               text_lowercase_)) {
        int bucket = -1;
        if (state.uses_dictionary != 0U) {
          const auto vocabulary_it = state.vocabulary_indices.find(token);
          if (vocabulary_it == state.vocabulary_indices.end()) {
            continue;
          }
          bucket = vocabulary_it->second;
        } else {
          if (state.filters_tokens != 0U &&
              state.vocabulary_indices.find(token) == state.vocabulary_indices.end()) {
            continue;
          }
          const auto cache_it = text_hash_cache_.find(token);
          std::uint64_t token_hash = 0U;
          if (cache_it == text_hash_cache_.end()) {
            const py::bytes digest =
                detail::HashlibModule()
                    .attr("blake2b")(py::bytes(token), py::arg("digest_size") = 8)
                    .attr("digest")()
                    .cast<py::bytes>();
            token_hash = detail::BytesToLittleEndianU64(digest);
            constexpr std::size_t kMaximumCachedTextTokens = 65536U;
            if (text_hash_cache_.size() < kMaximumCachedTextTokens) {
              text_hash_cache_.emplace(token, token_hash);
            }
          } else {
            token_hash = cache_it->second;
          }
          bucket = static_cast<int>(
              token_hash % static_cast<std::uint64_t>(text_hash_dim_));
        }
        float& value = data[(text_column_start + static_cast<std::size_t>(bucket)) * row_count + row];
        if (text_feature_calcer_ == "binary") {
          value = 1.0F;
        } else {
          value += 1.0F;
        }
      }
      if (text_feature_calcer_ == "tfidf") {
        for (int offset = 0; offset < state.output_dim; ++offset) {
          data[(text_column_start + static_cast<std::size_t>(offset)) * row_count + row] *=
              state.idf_values[static_cast<std::size_t>(offset)];
        }
      }
    }
    column_index += static_cast<std::size_t>(state.output_dim);
  }

  for (const auto& state : embedding_states_) {
    const std::size_t embedding_column_start = column_index;
    for (std::size_t row = 0; row < row_count; ++row) {
      const std::vector<float> values = detail::EmbeddingValues(
          detail::MatrixValue(matrix, row, static_cast<std::size_t>(state.source_index)));
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

      const std::size_t target_column_start = embedding_column_start + state.stats.size();
      if (!state.target_projection_weights.empty()) {
        if (values.size() != state.center.size()) {
          throw std::invalid_argument(
              "embedding dimension does not match the fitted target projection");
        }
        for (std::size_t projection_index = 0;
             projection_index < state.target_projection_weights.size(); ++projection_index) {
          float projection = 0.0F;
          const auto& weights = state.target_projection_weights[projection_index];
          for (std::size_t dimension = 0; dimension < values.size(); ++dimension) {
            projection += (values[dimension] - state.center[dimension]) * weights[dimension];
          }
          write_column_value(
              row, target_column_start + projection_index, projection);
        }
      }
    }
    column_index += state.stats.size() + state.target_projection_weights.size();
  }

  if (column_index != column_count) {
    throw std::runtime_error(
        "feature-pipeline transform output layout failed its validated contract");
  }

  return py::make_tuple(std::move(transformed),
                        detail::VectorToPyList(cat_feature_indices_),
                        detail::VectorToPyList(output_feature_names_));
}

}  // namespace ctboost
