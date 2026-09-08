// Direct C++ coverage: Python state loading clears use_gpu in CPU-only builds,
// while native LoadState can exercise GPU leaf-index traversal without a GPU.
#include "ctboost/booster.hpp"

#include <pybind11/embed.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

int checks = 0;
int failures = 0;

void Check(bool condition, const char* name) {
  ++checks;
  if (!condition) {
    ++failures;
    std::printf("FAIL %s\n", name);
  }
}

template <typename Function>
void CheckInvalidArgument(Function function, const char* name) {
  bool threw = false;
  try {
    function();
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  Check(threw, name);
}

ctboost::Pool MakePool(std::size_t rows, std::size_t columns,
                       const std::vector<float>& values = {}) {
  pybind11::array_t<float> data({static_cast<pybind11::ssize_t>(rows),
                                static_cast<pybind11::ssize_t>(columns)});
  if (!values.empty()) {
    std::copy(values.begin(), values.end(), data.mutable_data());
  } else {
    std::fill_n(data.mutable_data(), rows * columns, 0.0F);
  }
  return ctboost::Pool(data, pybind11::array_t<float>());
}

void CheckPredictionArithmetic() {
  auto schema = std::make_shared<ctboost::QuantizationSchema>();
  schema->num_bins_per_feature = {2};
  schema->cut_offsets = {0, 1};
  schema->cut_values = {0};
  schema->categorical_mask = {0};
  schema->missing_value_mask = {0};
  const auto from_bits = [](std::uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  };
  const float initial = from_bits(0xc18a2628U);
  const float weight = from_bits(0x4129776dU);
  auto pool = MakePool(8, 1, {-1, 1, 0, 2, -3, -1, 1, 4});
  ctboost::GradientBooster booster;
  for (bool root_only : {true, false}) {
    std::vector<ctboost::Node> nodes(root_only ? 1 : 3);
    if (!root_only) {
      nodes[0].is_leaf = false;
      nodes[0].split_feature_id = 0;
      nodes[0].left_child = 1;
      nodes[0].right_child = 2;
    }
    for (auto& node : nodes) node.leaf_weight = weight;
    ctboost::Tree tree;
    tree.LoadState(nodes, schema, {});
    booster.LoadState({tree}, schema, {}, {}, {0.123456789}, {}, -1, 0, false,
                       0, {static_cast<double>(initial)});
    const auto expected = booster.PredictUncached(pool);
    for (int call = 0; call < 2; ++call) {
      const auto actual = booster.Predict(pool);
      Check(actual.size() == expected.size() &&
                std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0,
            "same-compiler legacy arithmetic, cold/warm root/split");
    }
  }
}

void CheckGpuLeaves(bool wide) {
  auto schema = std::make_shared<ctboost::QuantizationSchema>();
  const std::size_t numeric_cuts = wide ? 298 : 1;
  schema->num_bins_per_feature = {static_cast<std::uint16_t>(numeric_cuts + 2), 3, 2};
  schema->cut_offsets = {0, numeric_cuts, numeric_cuts + 2, numeric_cuts + 3};
  for (std::size_t index = 0; index < numeric_cuts; ++index) {
    schema->cut_values.push_back(static_cast<float>(index));
  }
  schema->cut_values.insert(schema->cut_values.end(), {0.0F, 2.0F, 0.0F});
  schema->categorical_mask = {0, 1, 0};
  schema->missing_value_mask = {1, 1, 0};
  schema->nan_modes = {static_cast<std::uint8_t>(ctboost::NanMode::Min),
                       static_cast<std::uint8_t>(ctboost::NanMode::Min),
                       static_cast<std::uint8_t>(ctboost::NanMode::Forbidden)};
  std::vector<ctboost::Node> nodes(5);
  nodes[0].is_leaf = false;
  nodes[0].split_feature_id = 0;
  nodes[0].split_bin_index = 1;
  nodes[0].left_child = 1;
  nodes[0].right_child = 2;
  nodes[2].is_leaf = false;
  nodes[2].is_categorical_split = true;
  nodes[2].split_feature_id = 1;
  nodes[2].left_categories[0] = 1;
  nodes[2].left_categories[1] = 1;
  nodes[2].left_child = 3;
  nodes[2].right_child = 4;
  ctboost::Tree routed;
  routed.LoadState(nodes, schema, {});
  ctboost::Tree empty;
  empty.LoadState({}, schema, {});
  ctboost::Tree constant;
  constant.LoadState(std::vector<ctboost::Node>(1), schema, {});
  nodes.resize(3);
  nodes[0].split_feature_id = 2;
  nodes[0].split_bin_index = 0;
  nodes[2] = ctboost::Node{};
  ctboost::Tree late;
  late.LoadState(nodes, schema, {});
  const std::vector<ctboost::Tree> trees{routed, empty, constant, late};
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const std::vector<float> values{-1, 0, -1, 1, 0, 1, 1, 2, -1,
                                 nan, 2, 1, 400, nan, 0, 400, 9, 1};
  auto pool = MakePool(6, 3, values);
  auto no_rows = MakePool(0, 3);
  auto wrong_columns = MakePool(0, 2);
  ctboost::GradientBooster cpu;
  ctboost::GradientBooster gpu;
  cpu.LoadState(trees, schema, {}, {}, {}, {}, -1, 0, false);
  gpu.LoadState(trees, schema, {}, {}, {}, {}, -1, 0, true);
  Check(gpu.use_gpu(), "native GPU state remains enabled without a device");
  for (int prefix : {1, -1, 0, 3, 99, 2, 1}) {
    const auto actual = gpu.PredictLeafIndices(pool, prefix);
    Check(actual == cpu.PredictLeafIndices(pool, prefix), "GPU/CPU leaf prefix parity");
    const std::size_t limit = prefix < 0 ? trees.size()
        : std::min(trees.size(), static_cast<std::size_t>(prefix));
    const std::vector<int> first_leaves{1, 3, 4, 1, 3, 4};
    for (std::size_t row = 0; row < 6; ++row) {
      if (limit > 0) Check(actual[row * limit] == first_leaves[row], "known route");
      if (limit > 1) Check(actual[row * limit + 1] == -1, "empty tree leaf sentinel");
      if (limit > 2) Check(actual[row * limit + 2] == 0, "root-only leaf index");
    }
    Check(gpu.PredictLeafIndices(no_rows, prefix).empty(), "empty GPU leaf query");
  }
  for (auto* model : {&cpu, &gpu}) {
    CheckInvalidArgument([&] { model->PredictLeafIndices(wrong_columns); },
                         "empty query retains schema validation");
    Check(model->PredictLeafIndices(wrong_columns, 0).empty(),
          "zero prefix retains schema-free behavior");
    auto missing_late = values;
    missing_late[2] = nan;
    auto prefix_pool = MakePool(6, 3, missing_late);
    Check(model->PredictLeafIndices(prefix_pool, 3).size() == 18,
          "later forbidden NaN feature not quantized for prefix");
    CheckInvalidArgument([&] { model->PredictLeafIndices(prefix_pool); },
                         "active forbidden NaN rejected");
  }
  // Replacing a GPU state must neither reuse an old CPU cache nor retain old
  // tree columns when the replacement is shorter.
  gpu.LoadState({constant}, schema, {}, {}, {}, {}, -1, 0, true);
  Check(gpu.PredictLeafIndices(pool) == std::vector<std::int32_t>(6, 0),
        "GPU state replacement leaf indices");
  gpu.LoadState(trees, schema, {}, {}, {}, {}, -1, 0, false);
  Check(gpu.PredictLeafIndices(pool) == cpu.PredictLeafIndices(pool),
        "CPU state replacement after GPU prediction");
}

void CheckDirectTreeBounds(int bytes) {
  auto schema = std::make_shared<ctboost::QuantizationSchema>();
  schema->num_bins_per_feature = {2, 2};
  schema->cut_offsets = {0, 1, 2};
  schema->cut_values = {0, 0};
  schema->categorical_mask = {0, 0};
  schema->missing_value_mask = {0, 0};
  std::vector<ctboost::Node> nodes(3);
  nodes[0].is_leaf = false;
  nodes[0].split_feature_id = 1;
  nodes[0].left_child = 1;
  nodes[0].right_child = 2;
  nodes[1].leaf_weight = 2;
  nodes[2].leaf_weight = -3;
  ctboost::Tree tree;
  tree.LoadState(nodes, schema, {});
  ctboost::HistMatrix hist;
  hist.num_rows = 2;
  hist.num_cols = 2;
  hist.bin_index_bytes = static_cast<std::uint8_t>(bytes);
  if (bytes == 1) hist.compact_bin_indices = {0, 0, 0, 1};
  else hist.bin_indices = {0, 0, 0, 1};
  Check(tree.PredictBinnedLeafIndex(hist, 0) == 1, "direct normal left leaf");
  Check(tree.PredictBinnedRow(hist, 1) == -3, "direct normal row value");
  std::vector<float> contributions(3, 0);
  tree.AccumulateBinnedContributions(hist, 1, 0.5F, contributions);
  Check(contributions == std::vector<float>({0, -1.5F, 0}), "direct normal contributions");
  // Keep allocated padding so a missing logical-bounds check fails safely,
  // without making the regression fixture access unallocated memory.
  hist.num_cols = 1;
  const auto check_bounds = [&](auto function) {
    bool threw = false;
    try {
      function();
    } catch (const std::out_of_range&) {
      threw = true;
    }
    Check(threw, "direct tree rejects missing feature column");
  };
  check_bounds([&] { tree.PredictBinnedLeafIndex(hist, 0); });
  check_bounds([&] { tree.PredictBinnedRow(hist, 0); });
  check_bounds([&] { tree.AccumulateBinnedContributions(hist, 0, 1, contributions); });
  nodes[0].split_feature_id = 0;
  tree.LoadState(nodes, schema, {});
  Check(tree.PredictBinnedLeafIndex(hist, 0) == 1,
        "direct narrower histogram works for an available feature");
  contributions.assign(3, 0);
  tree.AccumulateBinnedContributions(hist, 0, 0.5F, contributions);
  Check(contributions == std::vector<float>({1, 0, 0}),
        "direct narrower histogram contributions");
  nodes[0].split_feature_id = 1;
  tree.LoadState(nodes, nullptr, {});
  hist.num_cols = 2;
  Check(tree.PredictBinnedLeafIndex(hist, 1) == 2, "direct schema-free binned tree");
  contributions.assign(3, 0);
  tree.AccumulateBinnedContributions(hist, 1, 0.5F, contributions);
  Check(contributions == std::vector<float>({0, -1.5F, 0}),
        "direct schema-free contributions");
}

}  // namespace

int main(int argc, char** argv) {
  const bool arithmetic_only = argc == 2 && std::strcmp(argv[1], "--arithmetic-only") == 0;
  if (argc > 1 && !arithmetic_only) {
    std::printf("Usage: prediction_cache_smoke [--arithmetic-only]\n");
    return 2;
  }
  try {
    pybind11::scoped_interpreter interpreter{};
    CheckPredictionArithmetic();
    if (!arithmetic_only) {
      CheckGpuLeaves(false);
      CheckGpuLeaves(true);
      CheckDirectTreeBounds(1);
      CheckDirectTreeBounds(2);
    }
  } catch (const std::exception& error) {
    std::printf("FAIL unexpected exception: %s\n", error.what());
    return 1;
  }
  std::printf("checks=%d failures=%d\n", checks, failures);
  return failures == 0 ? 0 : 1;
}
