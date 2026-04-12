#include "ctboost/profiler.hpp"

#include <algorithm>
#include <cctype>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ctboost {
namespace {

bool EnvVarTruthy(const char* value) noexcept {
  if (value == nullptr || *value == '\0') {
    return false;
  }

  char normalized[16]{};
  const std::size_t length = std::min<std::size_t>(std::strlen(value), sizeof(normalized) - 1U);
  for (std::size_t index = 0; index < length; ++index) {
    const unsigned char ch = static_cast<unsigned char>(value[index]);
    normalized[index] = static_cast<char>(std::tolower(ch));
  }

  return std::strcmp(normalized, "1") == 0 || std::strcmp(normalized, "true") == 0 ||
         std::strcmp(normalized, "yes") == 0 || std::strcmp(normalized, "on") == 0;
}

void LogLine(const char* format, ...) {
  std::va_list args;
  va_start(args, format);
  std::fputs("[ctboost] ", stderr);
  std::vfprintf(stderr, format, args);
  std::fputc('\n', stderr);
  std::fflush(stderr);
  va_end(args);
}

}  // namespace

TrainingProfiler::TrainingProfiler(bool enabled) noexcept : enabled_(enabled) {}

bool TrainingProfiler::ResolveEnabled(bool requested) noexcept {
  return requested || EnvVarTruthy(std::getenv("CTBOOST_PROFILE"));
}

bool TrainingProfiler::enabled() const noexcept { return enabled_; }

void TrainingProfiler::LogFitStart(std::size_t rows,
                                   std::size_t cols,
                                   int iterations,
                                   bool use_gpu,
                                   int prediction_dimension) const {
  if (!enabled_) {
    return;
  }
  LogLine("fit_start rows=%zu cols=%zu iterations=%d task=%s prediction_dimension=%d",
          rows,
          cols,
          iterations,
          use_gpu ? "GPU" : "CPU",
          prediction_dimension);
}

void TrainingProfiler::LogHistogramFeature(std::size_t feature_index,
                                           std::size_t rows,
                                           std::size_t bins,
                                           bool is_categorical,
                                           double elapsed_ms) const {
  if (!enabled_) {
    return;
  }
  LogLine("hist_feature feature=%zu rows=%zu bins=%zu kind=%s elapsed_ms=%.3f",
          feature_index,
          rows,
          bins,
          is_categorical ? "categorical" : "numeric",
          elapsed_ms);
}

void TrainingProfiler::LogHistogramBuild(std::size_t rows,
                                         std::size_t cols,
                                         std::size_t total_bins,
                                         double elapsed_ms) const {
  if (!enabled_) {
    return;
  }
  LogLine("hist_build rows=%zu cols=%zu total_bins=%zu elapsed_ms=%.3f",
          rows,
          cols,
          total_bins,
          elapsed_ms);
}

void TrainingProfiler::LogNodeHistogram(int depth,
                                        std::size_t rows,
                                        bool use_gpu,
                                        double elapsed_ms) const {
  if (!enabled_) {
    return;
  }
  LogLine("node_hist depth=%d rows=%zu backend=%s elapsed_ms=%.3f",
          depth,
          rows,
          use_gpu ? "GPU" : "CPU",
          elapsed_ms);
}

void TrainingProfiler::LogNodeSearch(int depth,
                                     std::size_t rows,
                                     int feature_id,
                                     double p_value,
                                     double chi_square,
                                     bool split_valid,
                                     bool is_categorical,
                                     double gain,
                                     std::size_t left_rows,
                                     std::size_t right_rows,
                                     double feature_ms,
                                     double split_ms,
                                     double partition_ms) const {
  if (!enabled_) {
    return;
  }
  LogLine(
      "node_search depth=%d rows=%zu feature=%d p_value=%.6g chi_square=%.6g split_valid=%d categorical=%d gain=%.6g left_rows=%zu right_rows=%zu feature_ms=%.3f split_ms=%.3f partition_ms=%.3f",
      depth,
      rows,
      feature_id,
      p_value,
      chi_square,
      split_valid ? 1 : 0,
      is_categorical ? 1 : 0,
      gain,
      left_rows,
      right_rows,
      feature_ms,
      split_ms,
      partition_ms);
}

void TrainingProfiler::LogTreeBuild(int iteration,
                                    int total_iterations,
                                    int class_index,
                                    int prediction_dimension,
                                    double elapsed_ms) const {
  if (!enabled_) {
    return;
  }
  LogLine("tree_build iteration=%d/%d class=%d/%d elapsed_ms=%.3f",
          iteration,
          total_iterations,
          class_index + 1,
          prediction_dimension,
          elapsed_ms);
}

void TrainingProfiler::LogIteration(int iteration,
                                    int total_iterations,
                                    double gradient_ms,
                                    double tree_ms,
                                    double prediction_ms,
                                    double metric_ms,
                                    double eval_ms,
                                    double total_ms) const {
  if (!enabled_) {
    return;
  }
  LogLine("iteration iteration=%d/%d gradient_ms=%.3f tree_ms=%.3f prediction_ms=%.3f metric_ms=%.3f eval_ms=%.3f total_ms=%.3f",
          iteration,
          total_iterations,
          gradient_ms,
          tree_ms,
          prediction_ms,
          metric_ms,
          eval_ms,
          total_ms);
}

void TrainingProfiler::LogFitSummary(double hist_build_ms, double total_fit_ms) const {
  if (!enabled_) {
    return;
  }
  LogLine("fit_summary hist_build_ms=%.3f total_fit_ms=%.3f", hist_build_ms, total_fit_ms);
}

}  // namespace ctboost
