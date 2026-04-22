#include "histogram_internal.hpp"

#include <atomic>
#include <chrono>
#include <exception>
#include <filesystem>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "ctboost/profiler.hpp"

namespace ctboost {

HistBuilder::HistBuilder(std::size_t max_bins,
                         std::string nan_mode,
                         std::vector<std::uint16_t> max_bins_by_feature,
                         std::string border_selection_method,
                         std::vector<std::string> nan_mode_by_feature,
                         std::vector<std::vector<float>> feature_borders,
                         bool external_memory,
                         std::string external_memory_dir)
    : max_bins_(max_bins),
      max_bins_by_feature_(std::move(max_bins_by_feature)),
      nan_mode_(ParseNanMode(nan_mode)),
      nan_mode_name_(NanModeName(nan_mode_)),
      external_memory_(external_memory),
      external_memory_dir_(std::move(external_memory_dir)) {
  if (!detail::ValidateMaxBins(max_bins_)) {
    throw std::invalid_argument("max_bins must be in the range [1, 65535]");
  }
  border_selection_method_ = ParseBorderSelectionMethod(border_selection_method);
  border_selection_method_name_ = BorderSelectionMethodName(border_selection_method_);

  nan_mode_by_feature_.reserve(nan_mode_by_feature.size());
  nan_mode_by_feature_mask_.reserve(nan_mode_by_feature.size());
  nan_mode_by_feature_names_.reserve(nan_mode_by_feature.size());
  for (const std::string& raw_name : nan_mode_by_feature) {
    if (raw_name.empty()) {
      nan_mode_by_feature_.push_back(nan_mode_);
      nan_mode_by_feature_mask_.push_back(0U);
      nan_mode_by_feature_names_.push_back("");
      continue;
    }

    const NanMode parsed_mode = ParseNanMode(raw_name);
    nan_mode_by_feature_.push_back(parsed_mode);
    nan_mode_by_feature_mask_.push_back(1U);
    nan_mode_by_feature_names_.push_back(NanModeName(parsed_mode));
  }

  for (std::size_t feature = 0; feature < max_bins_by_feature_.size(); ++feature) {
    if (max_bins_by_feature_[feature] == 0U) {
      continue;
    }
    if (!detail::ValidateMaxBins(max_bins_by_feature_[feature])) {
      throw std::invalid_argument(
          "max_bin_by_feature entries must be in the range [1, 65535] when provided");
    }
  }

  feature_borders_.reserve(feature_borders.size());
  for (auto& borders : feature_borders) {
    feature_borders_.push_back(detail::NormalizeCustomBorders(std::move(borders)));
  }
}

HistMatrix HistBuilder::Build(const Pool& pool, const TrainingProfiler* profiler) const {
  const auto hist_build_start = std::chrono::steady_clock::now();
  if (!max_bins_by_feature_.empty() && max_bins_by_feature_.size() > pool.num_cols()) {
    throw std::invalid_argument(
        "max_bin_by_feature cannot specify more entries than the pool feature count");
  }
  if (!nan_mode_by_feature_names_.empty() && nan_mode_by_feature_names_.size() > pool.num_cols()) {
    throw std::invalid_argument(
        "nan_mode_by_feature cannot specify more entries than the pool feature count");
  }
  if (!feature_borders_.empty() && feature_borders_.size() > pool.num_cols()) {
    throw std::invalid_argument(
        "feature_borders cannot specify more entries than the pool feature count");
  }

  std::vector<NanMode> resolved_feature_nan_modes(pool.num_cols(), nan_mode_);
  for (std::size_t feature = 0; feature < pool.num_cols(); ++feature) {
    resolved_feature_nan_modes[feature] = detail::ResolveFeatureNanMode(
        nan_mode_by_feature_, nan_mode_by_feature_mask_, feature, nan_mode_);
  }
  if (!resolved_feature_nan_modes.empty()) {
    detail::ValidateForbiddenNanModeHasNoMissingValues(pool, resolved_feature_nan_modes);
  }

  HistMatrix hist;
  hist.num_rows = pool.num_rows();
  hist.num_cols = pool.num_cols();
  hist.num_bins_per_feature.resize(hist.num_cols, 0);
  hist.cut_offsets.reserve(hist.num_cols + 1);
  hist.cut_offsets.push_back(0);
  hist.categorical_mask = detail::BuildCategoricalMask(pool);
  hist.missing_value_mask.assign(hist.num_cols, 0);
  hist.nan_mode = static_cast<std::uint8_t>(nan_mode_);
  hist.nan_modes.resize(hist.num_cols, static_cast<std::uint8_t>(nan_mode_));
  const detail::HistogramBuildContext build_context =
      detail::ResolveHistogramBuildContext(hist.num_rows);

  std::vector<detail::FeatureHistogramResult> feature_results(hist.num_cols);
  std::exception_ptr first_error;
  std::mutex error_mutex;
  std::atomic<bool> failed{false};
  std::atomic<std::size_t> next_feature{0};
  const std::size_t thread_count = detail::ResolveHistogramThreadCount(hist.num_cols);
  std::vector<std::thread> workers;
  workers.reserve(thread_count);

  auto worker = [&]() {
    while (!failed.load(std::memory_order_relaxed)) {
      const std::size_t feature = next_feature.fetch_add(1, std::memory_order_relaxed);
      if (feature >= hist.num_cols) {
        return;
      }

      try {
        feature_results[feature] = detail::BuildFeatureHistogram(
            pool,
            detail::ResolveFeatureMaxBins(max_bins_by_feature_, feature, max_bins_),
            resolved_feature_nan_modes[feature],
            border_selection_method_,
            feature,
            hist.categorical_mask[feature] != 0U,
            build_context,
            detail::ResolveFeatureBorders(feature_borders_, feature));
      } catch (...) {
        failed.store(true, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(error_mutex);
        if (first_error == nullptr) {
          first_error = std::current_exception();
        }
        return;
      }
    }
  };

  for (std::size_t thread_index = 0; thread_index < thread_count; ++thread_index) {
    workers.emplace_back(worker);
  }
  for (std::thread& worker_thread : workers) {
    worker_thread.join();
  }
  if (first_error != nullptr) {
    std::rethrow_exception(first_error);
  }

  for (std::size_t feature = 0; feature < hist.num_cols; ++feature) {
    const detail::FeatureHistogramResult& feature_result = feature_results[feature];
    hist.missing_value_mask[feature] = feature_result.missing_value_mask;
    hist.nan_modes[feature] = static_cast<std::uint8_t>(feature_result.nan_mode);
    hist.num_bins_per_feature[feature] = feature_result.num_bins;
    hist.cut_values.insert(
        hist.cut_values.end(), feature_result.cut_values.begin(), feature_result.cut_values.end());
    hist.cut_offsets.push_back(hist.cut_values.size());
    if (profiler != nullptr && profiler->enabled()) {
      profiler->LogHistogramFeature(feature,
                                    hist.num_rows,
                                    static_cast<std::size_t>(feature_result.num_bins),
                                    feature_result.is_categorical,
                                    feature_result.elapsed_ms);
    }
  }

  hist.bin_index_bytes =
      detail::MaxFeatureBins(hist.num_bins_per_feature) <=
              static_cast<std::size_t>(std::numeric_limits<std::uint8_t>::max()) + 1U
          ? 1
          : 2;
  const bool use_external_bin_storage = external_memory_ && hist.num_rows > 0 && hist.num_cols > 0;
  std::filesystem::path external_storage_root;
  if (use_external_bin_storage) {
    external_storage_root = detail::MakeExternalStorageRoot(external_memory_dir_);
    hist.external_bin_storage_dir = external_storage_root.string();
    hist.external_feature_bin_paths.resize(hist.num_cols);
  } else if (hist.bin_index_bytes == 1) {
    hist.compact_bin_indices.resize(hist.num_rows * hist.num_cols, 0);
  } else {
    hist.bin_indices.resize(hist.num_rows * hist.num_cols, 0);
  }

  failed.store(false, std::memory_order_relaxed);
  next_feature.store(0, std::memory_order_relaxed);
  workers.clear();
  workers.reserve(thread_count);
  first_error = nullptr;
  auto materialize_worker = [&]() {
    while (!failed.load(std::memory_order_relaxed)) {
      const std::size_t feature = next_feature.fetch_add(1, std::memory_order_relaxed);
      if (feature >= hist.num_cols) {
        return;
      }

      try {
        if (use_external_bin_storage) {
          const std::filesystem::path feature_path =
              external_storage_root / ("feature_" + std::to_string(feature) + ".bin");
          detail::MaterializeFeatureBinsToExternalStorage(
              pool, feature, feature_results[feature], hist.bin_index_bytes, feature_path);
          hist.external_feature_bin_paths[feature] = feature_path.string();
        } else {
          detail::MaterializeFeatureBins(pool, feature, feature_results[feature], hist);
        }
      } catch (...) {
        failed.store(true, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(error_mutex);
        if (first_error == nullptr) {
          first_error = std::current_exception();
        }
        return;
      }
    }
  };

  for (std::size_t thread_index = 0; thread_index < thread_count; ++thread_index) {
    workers.emplace_back(materialize_worker);
  }
  for (std::thread& worker_thread : workers) {
    worker_thread.join();
  }
  if (first_error != nullptr) {
    std::rethrow_exception(first_error);
  }

  if (profiler != nullptr && profiler->enabled()) {
    std::size_t total_bins = 0;
    for (const std::uint16_t bins_for_feature : hist.num_bins_per_feature) {
      total_bins += static_cast<std::size_t>(bins_for_feature);
    }
    const double total_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - hist_build_start)
            .count();
    profiler->LogHistogramBuild(hist.num_rows, hist.num_cols, total_bins, total_ms);
  }

  if (use_external_bin_storage) {
    hist.uses_external_bin_storage_ = true;
    hist.external_cached_feature_index = static_cast<std::size_t>(-1);
  } else {
    hist.CompactBinStorage();
  }
  return hist;
}

}  // namespace ctboost
