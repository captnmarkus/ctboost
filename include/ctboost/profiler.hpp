#pragma once

#include <cstddef>

namespace ctboost {

class TrainingProfiler {
 public:
  explicit TrainingProfiler(bool enabled = false) noexcept;

  static bool ResolveEnabled(bool requested) noexcept;

  bool enabled() const noexcept;
  void LogFitStart(std::size_t rows,
                   std::size_t cols,
                   int iterations,
                   bool use_gpu,
                   int prediction_dimension) const;
  void LogHistogramFeature(std::size_t feature_index,
                           std::size_t rows,
                           std::size_t bins,
                           bool is_categorical,
                           double elapsed_ms) const;
  void LogHistogramBuild(std::size_t rows,
                         std::size_t cols,
                         std::size_t total_bins,
                         double elapsed_ms) const;
  void LogNodeHistogram(int depth, std::size_t rows, bool use_gpu, double elapsed_ms) const;
  void LogNodeSearch(int depth,
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
                     double partition_ms) const;
  void LogTreeBuild(int iteration,
                    int total_iterations,
                    int class_index,
                    int prediction_dimension,
                    double elapsed_ms) const;
  void LogIteration(int iteration,
                    int total_iterations,
                    double gradient_ms,
                    double tree_ms,
                    double prediction_ms,
                    double metric_ms,
                    double eval_ms,
                    double total_ms) const;
  void LogFitSummary(double hist_build_ms, double total_fit_ms) const;

 private:
  bool enabled_{false};
};

}  // namespace ctboost
