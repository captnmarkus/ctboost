"""Exact ONNX graph export for CTBoost's quantized arbitrary trees."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


class _GraphBuilder:
    def __init__(self, helper: Any, numpy_helper: Any, tensor_proto: Any) -> None:
        self.helper = helper
        self.numpy_helper = numpy_helper
        self.tensor_proto = tensor_proto
        self.nodes: List[Any] = []
        self.initializers: List[Any] = []
        self.counter = 0

    def name(self, prefix: str) -> str:
        self.counter += 1
        return "%s_%d" % (prefix, self.counter)

    def constant(self, prefix: str, value: Any, dtype: Any) -> str:
        name = self.name(prefix)
        self.initializers.append(
            self.numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name)
        )
        return name

    def node(self, operation: str, inputs: Sequence[str], prefix: str, **attributes: Any) -> str:
        output = self.name(prefix)
        self.nodes.append(
            self.helper.make_node(
                operation,
                list(inputs),
                [output],
                name=self.name(operation.lower()),
                **attributes,
            )
        )
        return output


def _feature_bins(
    builder: _GraphBuilder,
    schema: Mapping[str, Any],
    feature_index: int,
) -> str:
    column_index = builder.constant("feature_index", [feature_index], np.int64)
    column = builder.node("Gather", ["features", column_index], "feature", axis=1)
    cut_offsets = schema["cut_offsets"]
    cut_begin = int(cut_offsets[feature_index])
    cut_end = int(cut_offsets[feature_index + 1])
    cuts = np.asarray(schema["cut_values"][cut_begin:cut_end], dtype=np.float32)
    categorical = bool(schema["categorical_mask"][feature_index])
    bins_for_feature = int(schema["num_bins_per_feature"][feature_index])
    has_missing = bool(schema["missing_value_mask"][feature_index])
    nan_modes = schema.get("nan_modes", ())
    nan_mode = int(nan_modes[feature_index]) if len(nan_modes) else int(schema["nan_mode"])
    non_missing_bins = bins_for_feature - (1 if has_missing else 0)

    if cuts.size:
        cut_name = builder.constant("cuts", cuts.reshape((1, -1)), np.float32)
        comparison = builder.node(
            "Greater" if categorical else "GreaterOrEqual",
            [column, cut_name],
            "cut_comparison",
        )
        cast = builder.node(
            "Cast",
            [comparison],
            "cut_count_values",
            to=builder.tensor_proto.INT64,
        )
        axes = builder.constant("reduce_axes", [1], np.int64)
        computed = builder.node(
            "ReduceSum",
            [cast, axes],
            "bin_count",
            keepdims=1,
        )
    else:
        is_nan_for_zero = builder.node("IsNaN", [column], "nan_for_zero")
        computed = builder.node(
            "Cast",
            [is_nan_for_zero],
            "zero_bin_cast",
            to=builder.tensor_proto.INT64,
        )
        zero = builder.constant("zero", [0], np.int64)
        computed = builder.node("Mul", [computed, zero], "zero_bin")

    if categorical and non_missing_bins > 0:
        maximum = builder.constant("maximum_bin", [non_missing_bins - 1], np.int64)
        computed = builder.node("Min", [computed, maximum], "clamped_bin")
    offset = 1 if has_missing and nan_mode == 1 else 0
    if offset:
        offset_name = builder.constant("bin_offset", [offset], np.int64)
        computed = builder.node("Add", [computed, offset_name], "offset_bin")
    missing_bin = bins_for_feature - 1 if nan_mode == 2 and bins_for_feature else 0
    missing_name = builder.constant("missing_bin", [missing_bin], np.int64)
    is_nan = builder.node("IsNaN", [column], "is_nan")
    return builder.node("Where", [is_nan, missing_name, computed], "feature_bins")


def _left_condition(
    builder: _GraphBuilder,
    bins: str,
    node: Mapping[str, Any],
) -> str:
    if not bool(node["is_categorical_split"]):
        split = builder.constant("split_bin", [int(node["split_bin_index"])], np.int64)
        return builder.node("LessOrEqual", [bins, split], "left_numeric")
    left_bins = [
        index
        for index, value in enumerate(node.get("left_categories", ()))
        if int(value) != 0
    ]
    if not left_bins:
        impossible = builder.constant("impossible_category", [-1], np.int64)
        return builder.node("Equal", [bins, impossible], "left_empty_category")
    conditions = []
    for category_bin in left_bins:
        category = builder.constant("left_category", [category_bin], np.int64)
        conditions.append(builder.node("Equal", [bins, category], "left_category_match"))
    condition = conditions[0]
    for additional in conditions[1:]:
        condition = builder.node("Or", [condition, additional], "left_category_union")
    return condition


def _leaf_paths(tree: Mapping[str, Any]) -> List[Tuple[float, List[Tuple[Mapping[str, Any], bool]]]]:
    nodes = list(tree["nodes"])
    leaves: List[Tuple[float, List[Tuple[Mapping[str, Any], bool]]]] = []

    def visit(node_index: int, path: List[Tuple[Mapping[str, Any], bool]]) -> None:
        node = nodes[node_index]
        if bool(node["is_leaf"]):
            leaves.append((float(node["leaf_weight"]), list(path)))
            return
        visit(int(node["left_child"]), path + [(node, True)])
        visit(int(node["right_child"]), path + [(node, False)])

    if nodes:
        visit(0, [])
    return leaves


def build_onnx_model(payload: Mapping[str, Any]) -> Any:
    """Build an exact ONNX model operating on prepared float features."""

    try:
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError as exc:  # pragma: no cover - dependency-specific
        raise ImportError("ONNX export requires 'pip install ctboost[onnx]'") from exc

    builder = _GraphBuilder(helper, numpy_helper, TensorProto)
    schema = payload["quantization_schema"]
    trees = list(payload["trees"])
    feature_count = int(payload["num_features"])
    prediction_dimension = int(payload["prediction_dimension"])
    used_features = sorted(
        {
            int(node["split_feature_id"])
            for tree in trees
            for node in tree["nodes"]
            if not bool(node["is_leaf"])
        }
    )
    bins_by_feature = {
        feature: _feature_bins(builder, schema, feature) for feature in used_features
    }
    learning_rates = [float(value) for value in payload.get("tree_learning_rates", ())]
    default_learning_rate = float(payload["learning_rate"])
    output_terms: List[List[str]] = [[] for _ in range(prediction_dimension)]

    for tree_index, tree in enumerate(trees):
        iteration = tree_index // prediction_dimension
        scale = learning_rates[iteration] if iteration < len(learning_rates) else default_learning_rate
        output_index = tree_index % prediction_dimension
        for leaf_value, path in _leaf_paths(tree):
            conditions = []
            for split_node, goes_left in path:
                feature = int(split_node["split_feature_id"])
                condition = _left_condition(builder, bins_by_feature[feature], split_node)
                if not goes_left:
                    condition = builder.node("Not", [condition], "right_condition")
                conditions.append(condition)
            if conditions:
                mask = conditions[0]
                for additional in conditions[1:]:
                    mask = builder.node("And", [mask, additional], "leaf_path")
            else:
                any_bins = next(iter(bins_by_feature.values()), None)
                if any_bins is None:
                    feature_index = builder.constant("feature_index", [0], np.int64)
                    column = builder.node("Gather", ["features", feature_index], "feature", axis=1)
                    nan = builder.node("IsNaN", [column], "root_nan")
                    mask = builder.node("Equal", [nan, nan], "root_mask")
                else:
                    mask = builder.node("Equal", [any_bins, any_bins], "root_mask")
            mask_float = builder.node(
                "Cast", [mask], "leaf_mask_float", to=TensorProto.FLOAT
            )
            value = builder.constant("leaf_value", [scale * leaf_value], np.float32)
            output_terms[output_index].append(
                builder.node("Mul", [mask_float, value], "leaf_contribution")
            )

    dimension_outputs = []
    for output_index, terms in enumerate(output_terms):
        if not terms:
            raise ValueError("ONNX export found an output dimension without trees")
        summed = terms[0] if len(terms) == 1 else builder.node("Sum", terms, "tree_sum")
        dimension_outputs.append(summed)
    if prediction_dimension == 1:
        squeeze_axes = builder.constant("squeeze_axes", [1], np.int64)
        raw_output = builder.node("Squeeze", [dimension_outputs[0], squeeze_axes], "raw")
    else:
        raw_output = builder.node("Concat", dimension_outputs, "raw", axis=1)
    builder.nodes.append(helper.make_node("Identity", [raw_output], ["raw_predictions"], name="raw_predictions"))

    outputs = [
        helper.make_tensor_value_info(
            "raw_predictions",
            TensorProto.FLOAT,
            [None] if prediction_dimension == 1 else [None, prediction_dimension],
        )
    ]
    objective = str(payload["objective_name"]).lower()
    if objective in {"logloss", "binary_logloss", "binary:logistic"}:
        positive = builder.node("Sigmoid", ["raw_predictions"], "positive_probability")
        one = builder.constant("one", [1.0], np.float32)
        negative = builder.node("Sub", [one, positive], "negative_probability")
        axes = builder.constant("unsqueeze_axes", [1], np.int64)
        negative_column = builder.node("Unsqueeze", [negative, axes], "negative_column")
        positive_column = builder.node("Unsqueeze", [positive, axes], "positive_column")
        probabilities = builder.node(
            "Concat", [negative_column, positive_column], "probability_matrix", axis=1
        )
        builder.nodes.append(
            helper.make_node("Identity", [probabilities], ["probabilities"], name="probabilities")
        )
        outputs.append(
            helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [None, 2])
        )
    elif objective in {"multiclass", "softmax", "softmaxloss"}:
        probabilities = builder.node("Softmax", ["raw_predictions"], "probability_matrix", axis=1)
        builder.nodes.append(
            helper.make_node("Identity", [probabilities], ["probabilities"], name="probabilities")
        )
        outputs.append(
            helper.make_tensor_value_info(
                "probabilities", TensorProto.FLOAT, [None, prediction_dimension]
            )
        )

    graph = helper.make_graph(
        builder.nodes,
        "ctboost_inference",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, feature_count])],
        outputs,
        builder.initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="ctboost",
        producer_version=str(payload.get("ctboost_version", "")),
        opset_imports=[helper.make_opsetid("", 18)],
    )
    # The graph uses no post-IR9 features and keeping this conservative makes
    # the artifact usable by older production ONNX Runtime installations.
    model.ir_version = min(int(model.ir_version), 9)
    manifest = json.dumps(payload.get("inference_manifest"), sort_keys=True, separators=(",", ":"))
    helper.set_model_props(
        model,
        {
            "ctboost.inference_manifest": manifest,
            "ctboost.objective": str(payload["objective_name"]),
            "ctboost.prepared_features": str(bool(payload["expects_prepared_features"])).lower(),
        },
    )
    return model


def save_onnx_model(path: Any, payload: Mapping[str, Any]) -> None:
    try:
        import onnx
    except ImportError as exc:  # pragma: no cover - dependency-specific
        raise ImportError("ONNX export requires 'pip install ctboost[onnx]'") from exc
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(build_onnx_model(payload), str(destination))


__all__ = ["build_onnx_model", "save_onnx_model"]
