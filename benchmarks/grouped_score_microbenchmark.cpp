// Standalone: compile with statistics.cpp and statistics_gamma.cpp, C++17 /O2.
// This isolates unchanged node-score work; it does not compare fitted trees.
#include "ctboost/statistics.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
volatile double sink = 0.0;

struct Input {
  ctboost::BinStatistics stats;
  double gradient = 0.0;
  double weight = 0.0;
  double variance = 0.0;
};

Input MakeInput(std::size_t bins) {
  Input input;
  std::mt19937 generator(20260907);
  std::normal_distribution<double> normal;
  input.stats.gradient_sums.assign(bins, 0.0);
  input.stats.hessian_sums.assign(bins, 0.0);
  input.stats.weight_sums.assign(bins, 0.0);
  double square_sum = 0.0;
  for (std::size_t row = 0; row < 4096; ++row) {
    const std::size_t bin = row % bins;
    const double gradient = normal(generator) + 0.1 * static_cast<double>(bin % 7);
    input.stats.gradient_sums[bin] += gradient;
    input.stats.hessian_sums[bin] += 1.0;
    input.stats.weight_sums[bin] += 1.0;
    input.gradient += gradient;
    input.weight += 1.0;
    square_sum += gradient * gradient;
  }
  const double mean = input.gradient / input.weight;
  input.variance = square_sum / input.weight - mean * mean;
  return input;
}

ctboost::LinearStatisticScore Evaluate(const ctboost::LinearStatistic& engine,
                                       const Input& input,
                                       std::size_t groups,
                                       std::size_t missing,
                                       bool optimized) {
  if (optimized) {
    return engine.EvaluateGroupedScoreFromBinStatistics(
        input.stats, input.gradient, input.weight, input.variance, groups, missing);
  }
  const auto grouped = ctboost::GroupOrderedBinStatistics(input.stats, groups, missing);
  return engine.EvaluateScoreFromBinStatistics(
      grouped, input.gradient, input.weight, input.variance);
}

double Time(const ctboost::LinearStatistic& engine, const Input& input,
             std::size_t groups, std::size_t missing, bool optimized,
             std::size_t iterations) {
  const auto start = std::chrono::steady_clock::now();
  double sum = 0.0;
  for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
    sum += Evaluate(engine, input, groups, missing, optimized).chi_square;
  }
  sink = sum;
  return std::chrono::duration<double, std::nano>(
      std::chrono::steady_clock::now() - start).count() / static_cast<double>(iterations);
}
}  // namespace

int main() {
  const ctboost::LinearStatistic engine;
  constexpr std::size_t iterations = 100000;
  std::cout << std::setprecision(10)
            << "{\"iterations_per_round\":" << iterations
            << ",\"rounds\":7,\"cases\":[";
  bool first = true;
  for (std::size_t bins : {32U, 128U, 256U}) {
    const Input input = MakeInput(bins);
    for (std::size_t groups : {8U, 64U}) {
      for (bool missing : {false, true}) {
        const std::size_t missing_bin = missing ? bins - 1U : ctboost::kNoMissingStatisticBin;
        const auto original = Evaluate(engine, input, groups, missing_bin, false);
        const auto candidate = Evaluate(engine, input, groups, missing_bin, true);
        if (original.chi_square != candidate.chi_square || original.p_value != candidate.p_value ||
            original.degrees_of_freedom != candidate.degrees_of_freedom) {
          throw std::runtime_error("Grouped score changed");
        }
        Time(engine, input, groups, missing_bin, false, 1000);
        Time(engine, input, groups, missing_bin, true, 1000);
        std::vector<double> legacy_times, optimized_times;
        for (std::size_t round = 0; round < 7; ++round) {
          if (round % 2 == 0) {
            legacy_times.push_back(Time(engine, input, groups, missing_bin, false, iterations));
            optimized_times.push_back(Time(engine, input, groups, missing_bin, true, iterations));
          } else {
            optimized_times.push_back(Time(engine, input, groups, missing_bin, true, iterations));
            legacy_times.push_back(Time(engine, input, groups, missing_bin, false, iterations));
          }
        }
        std::sort(legacy_times.begin(), legacy_times.end());
        std::sort(optimized_times.begin(), optimized_times.end());
        if (!first) std::cout << ',';
        first = false;
        std::cout << "{\"bins\":" << bins << ",\"groups\":" << groups
                  << ",\"missing\":" << (missing ? "true" : "false")
                  << ",\"exact\":true,\"legacy_ns\":" << legacy_times[3]
                  << ",\"optimized_ns\":" << optimized_times[3]
                  << ",\"ratio\":" << optimized_times[3] / legacy_times[3] << '}';
      }
    }
  }
  std::cout << "]}\n";
}
