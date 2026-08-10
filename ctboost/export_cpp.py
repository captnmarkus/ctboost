"""Dependency-free C++17/C ABI code generation for CTBoost inference."""

from __future__ import annotations

import json
import math
from typing import Any, Iterable, Mapping, Sequence


def _float_literal(value: Any) -> str:
    resolved = float(value)
    if math.isnan(resolved):
        return "std::numeric_limits<float>::quiet_NaN()"
    if math.isinf(resolved):
        return (
            "std::numeric_limits<float>::infinity()"
            if resolved > 0.0
            else "-std::numeric_limits<float>::infinity()"
        )
    return format(resolved, ".9g") + "f"


def _integer_literal(value: Any) -> str:
    return str(int(value))


def _array(
    cpp_type: str,
    name: str,
    values: Iterable[Any],
    formatter: Any,
) -> str:
    resolved = list(values)
    storage = resolved if resolved else [0]
    body = ", ".join(formatter(value) for value in storage)
    return "static constexpr %s %s[] = {%s};" % (cpp_type, name, body)


def _flatten_model(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    schema = payload["quantization_schema"]
    nodes = []
    tree_offsets = []
    tree_sizes = []
    category_routes = []
    for tree in payload["trees"]:
        tree_offsets.append(len(nodes))
        tree_nodes = list(tree["nodes"])
        tree_sizes.append(len(tree_nodes))
        for node in tree_nodes:
            routes = [int(value) for value in node.get("left_categories", ())]
            category_offset = len(category_routes)
            category_routes.extend(routes)
            nodes.append(
                {
                    "split_feature": int(node["split_feature_id"]),
                    "split_bin": int(node["split_bin_index"]),
                    "left_child": int(node["left_child"]),
                    "right_child": int(node["right_child"]),
                    "leaf_weight": float(node["leaf_weight"]),
                    "category_offset": category_offset,
                    "category_size": len(routes),
                    "is_leaf": bool(node["is_leaf"]),
                    "is_categorical": bool(node["is_categorical_split"]),
                }
            )
    return {
        "schema": schema,
        "nodes": nodes,
        "tree_offsets": tree_offsets,
        "tree_sizes": tree_sizes,
        "category_routes": category_routes,
    }


def standalone_cpp_source(payload: Mapping[str, Any]) -> str:
    """Generate one C++17 translation unit with a small stable C ABI."""

    flattened = _flatten_model(payload)
    schema = flattened["schema"]
    nodes = flattened["nodes"]
    prediction_dimension = int(payload["prediction_dimension"])
    base_score = [
        float(value)
        for value in payload.get("base_score", [0.0] * prediction_dimension)
    ]
    if len(base_score) != prediction_dimension:
        raise ValueError("predictor base_score dimension mismatch")
    tree_count = len(flattened["tree_offsets"])
    default_learning_rate = float(payload["learning_rate"])
    learning_rates = [float(value) for value in payload.get("tree_learning_rates", ())]
    tree_scales = []
    for tree_index in range(tree_count):
        iteration = tree_index // prediction_dimension
        tree_scales.append(
            learning_rates[iteration]
            if iteration < len(learning_rates)
            else default_learning_rate
        )
    node_literals = []
    for node in nodes:
        node_literals.append(
            "{%d, %d, %d, %d, %s, %dU, %dU, %s, %s}"
            % (
                node["split_feature"],
                node["split_bin"],
                node["left_child"],
                node["right_child"],
                _float_literal(node["leaf_weight"]),
                node["category_offset"],
                node["category_size"],
                "true" if node["is_leaf"] else "false",
                "true" if node["is_categorical"] else "false",
            )
        )
    if not node_literals:
        node_literals = ["{0, 0, 0, 0, 0.0f, 0U, 0U, true, false}"]
    manifest_json = json.dumps(
        payload.get("inference_manifest"),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    if ")ctboost\"" in manifest_json:
        raise ValueError("inference manifest cannot be represented in generated C++ source")
    objective = str(payload["objective_name"]).lower()
    binary = objective in {"logloss", "binary_logloss", "binary:logistic"}
    multiclass = objective in {"multiclass", "softmax", "softmaxloss"}
    probability_dimension = 2 if binary else prediction_dimension if multiclass else 0

    declarations = "\n".join(
        [
            _array(
                "std::uint16_t",
                "kNumBins",
                schema["num_bins_per_feature"],
                _integer_literal,
            ),
            _array("std::uint64_t", "kCutOffsets", schema["cut_offsets"], _integer_literal),
            _array("float", "kCutValues", schema["cut_values"], _float_literal),
            _array(
                "std::uint8_t",
                "kCategoricalMask",
                schema["categorical_mask"],
                _integer_literal,
            ),
            _array(
                "std::uint8_t",
                "kMissingMask",
                schema["missing_value_mask"],
                _integer_literal,
            ),
            _array(
                "std::uint8_t",
                "kNanModes",
                schema.get("nan_modes", ()),
                _integer_literal,
            ),
            _array(
                "std::uint32_t",
                "kTreeOffsets",
                flattened["tree_offsets"],
                _integer_literal,
            ),
            _array(
                "std::uint32_t",
                "kTreeSizes",
                flattened["tree_sizes"],
                _integer_literal,
            ),
            _array("float", "kTreeScales", tree_scales, _float_literal),
            _array("float", "kBaseScore", base_score, _float_literal),
            _array(
                "std::uint8_t",
                "kCategoryRoutes",
                flattened["category_routes"],
                _integer_literal,
            ),
        ]
    )
    node_declaration = "static constexpr Node kNodes[] = {\n  %s\n};" % ",\n  ".join(
        node_literals
    )
    nan_mode_count = len(schema.get("nan_modes", ()))
    cut_value_count = len(schema["cut_values"])
    category_route_count = len(flattened["category_routes"])

    return r'''// Generated by CTBoost. Compile as C++17 or newer.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(_WIN32)
#define CTBOOST_EXPORT extern "C" __declspec(dllexport)
#else
#define CTBOOST_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {
struct Node {
  std::int32_t split_feature;
  std::int32_t split_bin;
  std::int32_t left_child;
  std::int32_t right_child;
  float leaf_weight;
  std::uint32_t category_offset;
  std::uint32_t category_size;
  bool is_leaf;
  bool is_categorical;
};

static constexpr std::size_t kNumFeatures = %(num_features)dU;
static constexpr std::size_t kPredictionDimension = %(prediction_dimension)dU;
static constexpr std::size_t kProbabilityDimension = %(probability_dimension)dU;
static constexpr std::size_t kTreeCount = %(tree_count)dU;
static constexpr std::size_t kCutValueCount = %(cut_value_count)dU;
static constexpr std::size_t kNanModeCount = %(nan_mode_count)dU;
static constexpr std::size_t kCategoryRouteCount = %(category_route_count)dU;
static constexpr std::uint8_t kDefaultNanMode = %(default_nan_mode)dU;
%(declarations)s
%(node_declaration)s

std::uint16_t BinValue(std::size_t feature, float value) noexcept {
  const std::uint16_t bins = kNumBins[feature];
  if (bins == 0U) return 0U;
  const bool has_missing = kMissingMask[feature] != 0U;
  const std::uint8_t nan_mode = kNanModeCount == 0U ? kDefaultNanMode : kNanModes[feature];
  if (std::isnan(value)) return nan_mode == 2U ? static_cast<std::uint16_t>(bins - 1U) : 0U;
  const std::size_t cut_begin = static_cast<std::size_t>(kCutOffsets[feature]);
  const std::size_t cut_end = static_cast<std::size_t>(kCutOffsets[feature + 1U]);
  const std::size_t non_missing_bins = bins - (has_missing ? 1U : 0U);
  if (non_missing_bins == 0U) return nan_mode == 2U ? static_cast<std::uint16_t>(bins - 1U) : 0U;
  const std::uint16_t offset = has_missing && nan_mode == 1U ? 1U : 0U;
  const float* begin = kCutValues + cut_begin;
  const float* end = kCutValues + cut_end;
  if (kCategoricalMask[feature] != 0U) {
    const float* position = std::lower_bound(begin, end, value);
    std::size_t insertion = static_cast<std::size_t>(position - begin);
    insertion = std::min(insertion, non_missing_bins - 1U);
    return static_cast<std::uint16_t>(offset + insertion);
  }
  return static_cast<std::uint16_t>(offset + static_cast<std::size_t>(std::upper_bound(begin, end, value) - begin));
}

void PredictRow(const float* row, double* output) noexcept {
  std::uint16_t bins[kNumFeatures > 0U ? kNumFeatures : 1U] = {};
  for (std::size_t feature = 0; feature < kNumFeatures; ++feature) bins[feature] = BinValue(feature, row[feature]);
  for (std::size_t output_index = 0; output_index < kPredictionDimension; ++output_index) {
    output[output_index] = static_cast<double>(kBaseScore[output_index]);
  }
  for (std::size_t tree_index = 0; tree_index < kTreeCount; ++tree_index) {
    const Node* tree = kNodes + kTreeOffsets[tree_index];
    std::int32_t node_index = 0;
    while (!tree[node_index].is_leaf) {
      const Node& node = tree[node_index];
      const std::uint16_t bin = bins[static_cast<std::size_t>(node.split_feature)];
      bool left = false;
      if (node.is_categorical) {
        left = static_cast<std::size_t>(node.category_offset) + bin < kCategoryRouteCount &&
               kCategoryRoutes[static_cast<std::size_t>(node.category_offset) + bin] != 0U;
      } else {
        left = bin <= static_cast<std::uint16_t>(node.split_bin);
      }
      node_index = left ? node.left_child : node.right_child;
    }
    output[tree_index %% kPredictionDimension] +=
        static_cast<double>(kTreeScales[tree_index]) * static_cast<double>(tree[node_index].leaf_weight);
  }
}
}  // namespace

CTBOOST_EXPORT std::size_t ctboost_num_features() noexcept { return kNumFeatures; }
CTBOOST_EXPORT std::size_t ctboost_prediction_dimension() noexcept { return kPredictionDimension; }
CTBOOST_EXPORT std::size_t ctboost_probability_dimension() noexcept { return kProbabilityDimension; }
CTBOOST_EXPORT const char* ctboost_inference_manifest_json() noexcept {
  return R"ctboost(%(manifest_json)s)ctboost";
}

CTBOOST_EXPORT int ctboost_predict(const float* data,
                                   std::size_t rows,
                                   std::size_t columns,
                                   float* output,
                                   std::size_t output_size) noexcept {
  if (data == nullptr || output == nullptr) return 1;
  if (columns != kNumFeatures) return 2;
  if (kPredictionDimension != 0U && rows > output_size / kPredictionDimension) return 3;
  if (columns != 0U && rows > std::numeric_limits<std::size_t>::max() / columns) return 5;
  double scores[kPredictionDimension > 0U ? kPredictionDimension : 1U] = {};
  for (std::size_t row = 0; row < rows; ++row) {
    PredictRow(data + row * columns, scores);
    for (std::size_t output_index = 0; output_index < kPredictionDimension; ++output_index) {
      output[row * kPredictionDimension + output_index] = static_cast<float>(scores[output_index]);
    }
  }
  return 0;
}

CTBOOST_EXPORT int ctboost_predict_proba(const float* data,
                                         std::size_t rows,
                                         std::size_t columns,
                                         float* output,
                                         std::size_t output_size) noexcept {
  if (kProbabilityDimension == 0U) return 4;
  if (data == nullptr || output == nullptr) return 1;
  if (columns != kNumFeatures) return 2;
  if (kProbabilityDimension != 0U && rows > output_size / kProbabilityDimension) return 3;
  if (columns != 0U && rows > std::numeric_limits<std::size_t>::max() / columns) return 5;
  double scores[kPredictionDimension > 0U ? kPredictionDimension : 1U] = {};
  for (std::size_t row = 0; row < rows; ++row) {
    PredictRow(data + row * columns, scores);
    if (kProbabilityDimension == 2U && kPredictionDimension == 1U) {
      const double positive = scores[0] >= 0.0
          ? 1.0 / (1.0 + std::exp(-scores[0]))
          : std::exp(scores[0]) / (1.0 + std::exp(scores[0]));
      output[row * 2U] = static_cast<float>(1.0 - positive);
      output[row * 2U + 1U] = static_cast<float>(positive);
    } else {
      double maximum = scores[0];
      for (std::size_t index = 1; index < kPredictionDimension; ++index) maximum = std::max(maximum, scores[index]);
      double normalizer = 0.0;
      for (std::size_t index = 0; index < kPredictionDimension; ++index) normalizer += std::exp(scores[index] - maximum);
      for (std::size_t index = 0; index < kPredictionDimension; ++index) {
        output[row * kPredictionDimension + index] = static_cast<float>(std::exp(scores[index] - maximum) / normalizer);
      }
    }
  }
  return 0;
}
''' % {
        "num_features": int(payload["num_features"]),
        "prediction_dimension": prediction_dimension,
        "probability_dimension": probability_dimension,
        "tree_count": tree_count,
        "cut_value_count": cut_value_count,
        "nan_mode_count": nan_mode_count,
        "category_route_count": category_route_count,
        "default_nan_mode": int(schema["nan_mode"]),
        "declarations": declarations,
        "node_declaration": node_declaration,
        "manifest_json": manifest_json,
    }


__all__ = ["standalone_cpp_source"]
