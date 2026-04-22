#include "ctboost/feature_pipeline.hpp"

#include "feature_pipeline_internal.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ctboost {

void NativeFeaturePipeline::FitCtrState(pybind11::array object_matrix,
                                        const std::vector<float>& label_values) {
  const auto [target_width, target_prior] = detail::FitTargetMode(label_values);
  const bool ctr_enabled =
      ordered_ctr_ || !simple_ctr_.is_none() || !combinations_ctr_.is_none() || !per_feature_ctr_.is_none();
  if (!ctr_enabled) {
    return;
  }

  const detail::MatrixView matrix = detail::MakeMatrixView(std::move(object_matrix));
  const auto resolve_ctr_types_for_source = [this](const std::vector<int>& source_indices,
                                                   bool is_combination) {
    if (!per_feature_ctr_.is_none()) {
      for (const auto& item : pybind11::cast<pybind11::dict>(per_feature_ctr_)) {
        pybind11::list selectors;
        pybind11::object key_object = pybind11::reinterpret_borrow<pybind11::object>(item.first);
        if (pybind11::isinstance<pybind11::list>(key_object) || pybind11::isinstance<pybind11::tuple>(key_object)) {
          for (const pybind11::handle selector : key_object) {
            selectors.append(pybind11::reinterpret_borrow<pybind11::object>(selector));
          }
        } else {
          selectors.append(std::move(key_object));
        }
        if (ResolveIndices(std::move(selectors)) == source_indices) {
          return detail::ResolveCtrTypeList(pybind11::reinterpret_borrow<pybind11::object>(item.second), false);
        }
      }
    }
    return detail::ResolveCtrTypeList(is_combination ? combinations_ctr_ : simple_ctr_, ordered_ctr_);
  };

  std::vector<std::size_t> permutation(matrix.rows);
  std::iota(permutation.begin(), permutation.end(), 0U);
  std::mt19937_64 rng(static_cast<std::uint64_t>(random_seed_));
  std::shuffle(permutation.begin(), permutation.end(), rng);

  struct CtrSourceSpec {
    std::vector<int> source_indices;
    std::string output_prefix;
    bool is_combination{false};
  };

  std::vector<CtrSourceSpec> ctr_sources;
  for (const auto& state : categorical_states_) {
    ctr_sources.push_back(CtrSourceSpec{{state.source_index}, state.output_name, false});
  }
  for (std::size_t index = 0; index < combination_states_.size(); ++index) {
    ctr_sources.push_back(
        CtrSourceSpec{combination_source_indices_[index], combination_states_[index].output_name, true});
  }

  for (const auto& source : ctr_sources) {
    const std::vector<std::string> ctr_types =
        resolve_ctr_types_for_source(source.source_indices, source.is_combination);
    if (ctr_types.empty()) {
      continue;
    }

    std::unordered_map<std::string, int> feature_counts;
    feature_counts.reserve(matrix.rows);
    for (std::size_t row = 0; row < matrix.rows; ++row) {
      ++feature_counts[detail::JoinNormalizedKey(matrix, row, source.source_indices)];
    }

    for (const std::string& ctr_type : ctr_types) {
      std::vector<std::string> output_names;
      if (ctr_type == "Mean") {
        if (target_width == 1) {
          output_names.push_back(source.output_prefix + "_ctr");
        } else {
          for (int class_index = 0; class_index < target_width; ++class_index) {
            output_names.push_back(source.output_prefix + "_ctr_class" + std::to_string(class_index));
          }
        }
      } else {
        output_names.push_back(source.output_prefix + "_freq_ctr");
      }

      std::unordered_map<std::string, int> total_counts;
      std::unordered_map<std::string, std::vector<float>> total_sums;
      std::unordered_map<std::string, int> running_counts;
      std::unordered_map<std::string, std::vector<float>> running_sums;
      std::vector<std::vector<float>> training_columns(
          output_names.size(), std::vector<float>(matrix.rows, 0.0F));
      std::size_t seen_rows = 0;

      for (std::size_t row : permutation) {
        const std::string key = detail::JoinNormalizedKey(matrix, row, source.source_indices);
        const float current_count = static_cast<float>(running_counts[key]);
        auto& current_sums = running_sums[key];
        if (ctr_type == "Mean" && current_sums.empty()) {
          current_sums.assign(static_cast<std::size_t>(target_width), 0.0F);
        }

        if (ctr_type == "Mean") {
          const float denominator = current_count + static_cast<float>(ctr_prior_strength_);
          for (int output_index = 0; output_index < target_width; ++output_index) {
            const float numerator = current_sums[static_cast<std::size_t>(output_index)] +
                                    static_cast<float>(ctr_prior_strength_) *
                                        target_prior[static_cast<std::size_t>(output_index)];
            training_columns[static_cast<std::size_t>(output_index)][row] =
                numerator / std::max(denominator, 1.0F);
          }
        } else {
          const float global_frequency =
              matrix.rows == 0
                  ? 0.0F
                  : static_cast<float>(feature_counts[key]) / static_cast<float>(matrix.rows);
          const float denominator = static_cast<float>(seen_rows) + static_cast<float>(ctr_prior_strength_);
          const float numerator =
              current_count + static_cast<float>(ctr_prior_strength_) * global_frequency;
          training_columns[0][row] = numerator / std::max(denominator, 1.0F);
        }

        total_counts[key] += 1;
        if (ctr_type == "Mean") {
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
        }
        running_counts[key] += 1;
        ++seen_rows;
      }

      CtrState state;
      state.source_indices = source.source_indices;
      state.output_names = output_names;
      state.ctr_type = ctr_type;
      state.prior_values = target_prior;
      state.total_counts = std::move(total_counts);
      state.total_sums = std::move(total_sums);
      state.total_rows = matrix.rows;
      ctr_states_.push_back(std::move(state));
      for (auto& column : training_columns) {
        training_ctr_columns_.push_back(std::move(column));
      }
      output_feature_names_.insert(output_feature_names_.end(), output_names.begin(), output_names.end());
    }
  }
}

}  // namespace ctboost
