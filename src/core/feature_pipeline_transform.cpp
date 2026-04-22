#include "ctboost/feature_pipeline.hpp"

#include "feature_pipeline_internal.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace ctboost {

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

  const detail::MatrixView matrix = detail::MakeMatrixView(detail::EnsureObjectMatrix(std::move(raw_matrix)));
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
  std::unordered_map<int, const OneHotEncoderState*> one_hot_by_index;
  for (const auto& state : one_hot_states_) {
    one_hot_by_index.emplace(state.source_index, &state);
  }

  std::size_t column_index = 0;
  for (int feature_index : numeric_indices_) {
    const auto one_hot_it = one_hot_by_index.find(feature_index);
    if (one_hot_it != one_hot_by_index.end()) {
      const auto& category_keys = one_hot_it->second->category_keys;
      const std::string other_key = one_hot_it->second->has_other_bucket != 0U ? detail::kOtherKey : "";
      for (std::size_t row = 0; row < row_count; ++row) {
        const std::string raw_key =
            detail::NormalizeKey(detail::MatrixValue(matrix, row, static_cast<std::size_t>(feature_index)));
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
            detail::MatrixValue(matrix, row, static_cast<std::size_t>(feature_index)));
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
      const std::string key = detail::JoinNormalizedKey(matrix, row, source_indices);
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
      for (std::size_t output_index = 0; output_index < state.output_names.size(); ++output_index) {
        for (std::size_t row = 0; row < row_count; ++row) {
          const std::string key = detail::JoinNormalizedKey(matrix, row, state.source_indices);
          const auto count_it = state.total_counts.find(key);
          const float count =
              count_it == state.total_counts.end() ? 0.0F : static_cast<float>(count_it->second);
          float value = 0.0F;
          if (state.ctr_type == "Mean") {
            const auto sums_it = state.total_sums.find(key);
            const float summed =
                sums_it == state.total_sums.end() ? 0.0F : sums_it->second[output_index];
            const float denominator = count + static_cast<float>(ctr_prior_strength_);
            const float numerator =
                summed + static_cast<float>(ctr_prior_strength_) * state.prior_values[output_index];
            value = numerator / std::max(denominator, 1.0F);
          } else {
            const float total_rows = static_cast<float>(std::max<std::size_t>(state.total_rows, 1U));
            const float global_frequency = count / total_rows;
            const float denominator = total_rows + static_cast<float>(ctr_prior_strength_);
            const float numerator =
                count + static_cast<float>(ctr_prior_strength_) * global_frequency;
            value = numerator / std::max(denominator, 1.0F);
          }
          write_column_value(row, column_index, value);
        }
        ++column_index;
      }
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
      for (const std::string& token : detail::ExtractAsciiTokens(text)) {
        auto cache_it = text_hash_cache_.find(token);
        if (cache_it == text_hash_cache_.end()) {
          const py::bytes digest =
              detail::HashlibModule()
                  .attr("blake2b")(py::bytes(token), py::arg("digest_size") = 8)
                  .attr("digest")()
                  .cast<py::bytes>();
          cache_it = text_hash_cache_.emplace(token, detail::BytesToLittleEndianU64(digest)).first;
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
    }
    column_index += state.stats.size();
  }

  return py::make_tuple(std::move(transformed),
                        detail::VectorToPyList(cat_feature_indices_),
                        detail::VectorToPyList(output_feature_names_));
}

}  // namespace ctboost
