"""Explanation and visualization helpers for fitted CTBoost ensembles.

The TreeSHAP routines implement interventional values directly over
CTBoost's arbitrary binary trees.  A background row defines whether a split
uses the explained value (feature present in the coalition) or the background
value (feature absent).  Shapley linearity then lets us average the exact
single-reference result over a weighted empirical background distribution.

This is intentionally separate from ``predict_contrib``.  That older method
is a fast path decomposition; it is additive, but it is not a SHAP value.

Object influence is exposed separately as a clearly labeled shared-leaf
approximation. It does not claim the exactness of the TreeSHAP implementation
and does not refit the model.
"""

from __future__ import annotations

from math import comb
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from .core import Pool
from .training._pool_build import _resolve_num_iteration
from .training.schema import _baseline_matrix_for_prediction


def _iter_mask_indices(mask: int) -> Iterable[int]:
    while mask:
        lowest = mask & -mask
        yield lowest.bit_length() - 1
        mask ^= lowest


def _quantize_feature(
    values: np.ndarray,
    schema: Mapping[str, Any],
    feature_index: int,
) -> np.ndarray:
    """Match ``QuantizationSchema::bin_value`` for one feature column."""
    bins_for_feature = int(schema["num_bins_per_feature"][feature_index])
    result = np.zeros(values.shape[0], dtype=np.uint16)
    if bins_for_feature == 0 or values.size == 0:
        return result

    cut_offsets = schema["cut_offsets"]
    cut_begin = int(cut_offsets[feature_index])
    cut_end = int(cut_offsets[feature_index + 1])
    cuts = np.asarray(schema["cut_values"][cut_begin:cut_end], dtype=np.float32)
    categorical = bool(schema["categorical_mask"][feature_index])
    has_missing = bool(schema["missing_value_mask"][feature_index])
    nan_modes = schema.get("nan_modes", ())
    nan_mode = int(nan_modes[feature_index]) if len(nan_modes) else int(schema["nan_mode"])

    missing = np.isnan(values)
    if missing.any() and nan_mode == 0:
        raise ValueError("NaN values are not allowed when nan_mode='Forbidden'")
    missing_bin = bins_for_feature - 1 if nan_mode == 2 else 0
    result[missing] = np.uint16(missing_bin)

    present = ~missing
    non_missing_bins = bins_for_feature - (1 if has_missing else 0)
    if not present.any() or non_missing_bins == 0:
        return result

    non_missing_values = values[present]
    offset = 1 if has_missing and nan_mode == 1 else 0
    if categorical:
        insertion = np.searchsorted(cuts, non_missing_values, side="left")
        clamped = np.minimum(insertion, non_missing_bins - 1)
        exact = np.zeros(insertion.shape, dtype=bool)
        within = insertion < cuts.size
        exact[within] = cuts[insertion[within]] == non_missing_values[within]
        resolved = np.where(exact, insertion, clamped)
    else:
        resolved = np.searchsorted(cuts, non_missing_values, side="right")
    result[present] = np.asarray(offset + resolved, dtype=np.uint16)
    return result


def _used_features(trees: Sequence[Mapping[str, Any]]) -> Tuple[int, ...]:
    return tuple(
        sorted(
            {
                int(node["split_feature_id"])
                for tree in trees
                for node in tree["nodes"]
                if not bool(node["is_leaf"])
            }
        )
    )


def _quantize_used_features(
    matrix: np.ndarray,
    schema: Mapping[str, Any],
    features: Sequence[int],
) -> Dict[int, np.ndarray]:
    return {
        feature: _quantize_feature(matrix[:, feature], schema, feature)
        for feature in features
    }


def _left_routes(
    nodes: Sequence[Mapping[str, Any]],
    quantized: Mapping[int, np.ndarray],
) -> Dict[int, np.ndarray]:
    routes: Dict[int, np.ndarray] = {}
    for node_index, node in enumerate(nodes):
        if bool(node["is_leaf"]):
            continue
        feature = int(node["split_feature_id"])
        bins = quantized[feature]
        if bool(node["is_categorical_split"]):
            category_routes = np.asarray(node["left_categories"], dtype=np.uint8)
            routes[node_index] = category_routes[bins] != 0
        else:
            routes[node_index] = bins <= int(node["split_bin_index"])
    return routes


def _accumulate_leaf_shap(
    leaf_value: float,
    present_mask: int,
    absent_mask: int,
    phi: np.ndarray,
    interactions: Optional[np.ndarray],
) -> float:
    """Accumulate one leaf indicator game's exact Shapley values.

    A reachable leaf is characterized by features that must use the explained
    row (``present_mask``) and features that must use the reference row
    (``absent_mask``).  All other features are dummies.  Closed-form Shapley
    weights for this indicator avoid enumerating feature coalitions.
    """
    present = tuple(_iter_mask_indices(present_mask))
    absent = tuple(_iter_mask_indices(absent_mask))
    present_count = len(present)
    absent_count = len(absent)
    active_count = present_count + absent_count

    if present_count == 0:
        expected_value = leaf_value
    else:
        expected_value = 0.0

    coalition_count = comb(active_count, present_count)
    if present_count:
        coefficient = leaf_value / (present_count * coalition_count)
        for feature in present:
            phi[feature] += coefficient
    if absent_count:
        coefficient = -leaf_value / (absent_count * coalition_count)
        for feature in absent:
            phi[feature] += coefficient

    if interactions is None or active_count < 2:
        return expected_value

    if present_count >= 2:
        coefficient = leaf_value / (
            2.0 * (present_count - 1) * comb(active_count - 1, present_count - 1)
        )
        for offset, first in enumerate(present):
            for second in present[offset + 1 :]:
                interactions[first, second] += coefficient
                interactions[second, first] += coefficient

    if absent_count >= 2:
        coefficient = leaf_value / (
            2.0 * (absent_count - 1) * comb(active_count - 1, present_count)
        )
        for offset, first in enumerate(absent):
            for second in absent[offset + 1 :]:
                interactions[first, second] += coefficient
                interactions[second, first] += coefficient

    if present_count and absent_count:
        coefficient = -leaf_value / (
            2.0 * absent_count * comb(active_count - 1, present_count - 1)
        )
        for first in present:
            for second in absent:
                interactions[first, second] += coefficient
                interactions[second, first] += coefficient
    return expected_value


def _explain_tree_pair(
    nodes: Sequence[Mapping[str, Any]],
    foreground_routes: Mapping[int, np.ndarray],
    background_routes: Mapping[int, np.ndarray],
    foreground_row: int,
    background_row: int,
    scale: float,
    phi: np.ndarray,
    interactions: Optional[np.ndarray],
) -> float:
    expected_value = 0.0

    def visit(node_index: int, present_mask: int, absent_mask: int) -> None:
        nonlocal expected_value
        node = nodes[node_index]
        if bool(node["is_leaf"]):
            expected_value += _accumulate_leaf_shap(
                scale * float(node["leaf_weight"]),
                present_mask,
                absent_mask,
                phi,
                interactions,
            )
            return

        feature = int(node["split_feature_id"])
        feature_bit = 1 << feature
        foreground_child = (
            int(node["left_child"])
            if bool(foreground_routes[node_index][foreground_row])
            else int(node["right_child"])
        )
        background_child = (
            int(node["left_child"])
            if bool(background_routes[node_index][background_row])
            else int(node["right_child"])
        )

        if foreground_child == background_child:
            visit(foreground_child, present_mask, absent_mask)
        elif present_mask & feature_bit:
            visit(foreground_child, present_mask, absent_mask)
        elif absent_mask & feature_bit:
            visit(background_child, present_mask, absent_mask)
        else:
            visit(foreground_child, present_mask | feature_bit, absent_mask)
            visit(background_child, present_mask, absent_mask | feature_bit)

    if nodes:
        visit(0, 0, 0)
    return expected_value


def _background_weights(pool: Pool) -> np.ndarray:
    if pool.num_rows == 0:
        raise ValueError("background must contain at least one row")
    if pool.weight is None:
        return np.full(pool.num_rows, 1.0 / pool.num_rows, dtype=np.float64)
    weights = np.asarray(pool.weight, dtype=np.float64).reshape(-1)
    if weights.size != pool.num_rows:
        raise ValueError("background weights must match the number of background rows")
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("background weights must be finite and non-negative")
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("background weights must have a positive sum")
    return weights / total


def explain_booster(
    booster: Any,
    data: Any,
    background: Any,
    *,
    num_iteration: Optional[int] = None,
    interaction_values: bool = False,
) -> np.ndarray:
    """Return exact empirical interventional TreeSHAP values.

    The final column is the expected raw model output over ``background``.
    Interaction output follows XGBoost's convention: the final row/column are
    reserved for the bias, with the expected value at ``[..., -1, -1]``.
    """
    foreground_pool = booster._prediction_pool(data)
    background_pool = booster._prediction_pool(background)
    if foreground_pool.num_cols != background_pool.num_cols:
        raise ValueError("data and background must have the same transformed feature count")

    state = dict(booster._handle.export_state())
    schema = state.get("quantization_schema")
    if schema is None:
        raise ValueError("the fitted booster does not contain a quantization schema")
    feature_count = len(schema["num_bins_per_feature"])
    if foreground_pool.num_cols != feature_count:
        raise ValueError(
            f"data has {foreground_pool.num_cols} transformed features, but the model expects "
            f"{feature_count}"
        )

    resolved_iteration = _resolve_num_iteration(num_iteration)
    prediction_dimension = int(booster.prediction_dimension)
    all_trees = list(state.get("trees", ()))
    tree_limit = len(all_trees)
    if resolved_iteration >= 0:
        tree_limit = min(tree_limit, resolved_iteration * prediction_dimension)
    trees = all_trees[:tree_limit]

    foreground = np.asarray(foreground_pool.data, dtype=np.float32)
    reference = np.asarray(background_pool.data, dtype=np.float32)
    features = _used_features(trees)
    foreground_bins = _quantize_used_features(foreground, schema, features)
    background_bins = _quantize_used_features(reference, schema, features)
    weights = _background_weights(background_pool)

    sample_count = foreground_pool.num_rows
    shap_values = np.zeros(
        (sample_count, prediction_dimension, feature_count + 1), dtype=np.float64
    )
    base_score = np.asarray(
        state.get("base_score", [0.0] * prediction_dimension), dtype=np.float64
    ).reshape(-1)
    if base_score.size != prediction_dimension:
        raise ValueError("persisted base_score dimension does not match the model")
    shap_values[:, :, -1] += base_score[np.newaxis, :]
    interactions = (
        np.zeros(
            (
                sample_count,
                prediction_dimension,
                feature_count + 1,
                feature_count + 1,
            ),
            dtype=np.float64,
        )
        if interaction_values
        else None
    )

    default_learning_rate = float(state["learning_rate"])
    learning_rates = [float(value) for value in state.get("tree_learning_rates", ())]
    for tree_index, tree in enumerate(trees):
        nodes = list(tree["nodes"])
        foreground_routes = _left_routes(nodes, foreground_bins)
        background_routes = _left_routes(nodes, background_bins)
        output_index = 0 if prediction_dimension == 1 else tree_index % prediction_dimension
        iteration_index = tree_index // prediction_dimension
        learning_rate = (
            learning_rates[iteration_index]
            if iteration_index < len(learning_rates)
            else default_learning_rate
        )
        for foreground_row in range(sample_count):
            phi = shap_values[foreground_row, output_index, :-1]
            interaction_matrix = (
                None
                if interactions is None
                else interactions[foreground_row, output_index, :-1, :-1]
            )
            for background_row, background_weight in enumerate(weights):
                pair_scale = learning_rate * float(background_weight)
                shap_values[foreground_row, output_index, -1] += _explain_tree_pair(
                    nodes,
                    foreground_routes,
                    background_routes,
                    foreground_row,
                    background_row,
                    pair_scale,
                    phi,
                    interaction_matrix,
                )

    prediction_baseline = _baseline_matrix_for_prediction(
        foreground_pool, prediction_dimension
    )
    if prediction_baseline is not None:
        shap_values[:, :, -1] += prediction_baseline

    if interactions is not None:
        feature_interactions = interactions[:, :, :-1, :-1]
        for feature in range(feature_count):
            off_diagonal_sum = feature_interactions[:, :, feature, :].sum(axis=-1)
            feature_interactions[:, :, feature, feature] = (
                shap_values[:, :, feature] - off_diagonal_sum
            )
        interactions[:, :, -1, -1] = shap_values[:, :, -1]
        if prediction_dimension == 1:
            return interactions[:, 0]
        return interactions

    if prediction_dimension == 1:
        return shap_values[:, 0]
    return shap_values


def _leaf_influence_weights(pool: Pool) -> np.ndarray:
    if pool.num_rows == 0:
        raise ValueError("reference_data must contain at least one row")
    weights = (
        np.ones(pool.num_rows, dtype=np.float64)
        if pool.weight is None
        else np.asarray(pool.weight, dtype=np.float64).reshape(-1)
    )
    if weights.size != pool.num_rows:
        raise ValueError("reference weights must match the number of reference rows")
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("reference weights must be finite and non-negative")
    if float(weights.sum()) <= 0.0:
        raise ValueError("reference weights must have a positive sum")
    return weights


def calc_leaf_influence(
    booster: Any,
    data: Any,
    reference_data: Any,
    *,
    num_iteration: Optional[int] = None,
    return_coverage: bool = False,
) -> Any:
    """Approximate object influence by distributing shared-leaf output.

    For each tree, its signed leaf contribution for an explained row is
    distributed among reference rows that reach the same leaf. Distribution
    is uniform unless ``reference_data`` is a weighted :class:`Pool`, in
    which case it is proportional to those weights.

    This is a *leaf co-membership approximation*. It performs no model refits,
    does not differentiate the training loss, and must not be interpreted as
    the counterfactual effect of deleting or upweighting a training object.
    When ``reference_data`` contains the original training rows, summing the
    scores over reference objects reconstructs the tree component of the raw
    prediction. Pool baselines are intentionally not attributed.

    The single-output result has shape ``(rows, reference_rows)``. Multiclass
    output has shape ``(rows, prediction_dimension, reference_rows)``.
    ``return_coverage=True`` also returns the fraction of selected trees for
    which at least one positive-weight reference reached the explained leaf.
    """
    foreground_pool = booster._prediction_pool(data)
    reference_pool = booster._prediction_pool(reference_data)
    if foreground_pool.num_cols != reference_pool.num_cols:
        raise ValueError("data and reference_data must have the same transformed feature count")
    reference_weights = _leaf_influence_weights(reference_pool)

    state = dict(booster._handle.export_state())
    prediction_dimension = int(booster.prediction_dimension)
    resolved_iteration = _resolve_num_iteration(num_iteration)
    all_trees = list(state.get("trees", ()))
    tree_limit = len(all_trees)
    if resolved_iteration >= 0:
        tree_limit = min(tree_limit, resolved_iteration * prediction_dimension)
    trees = all_trees[:tree_limit]

    foreground_leaves = booster.predict_leaf_index(
        foreground_pool, num_iteration=num_iteration
    )
    reference_leaves = booster.predict_leaf_index(
        reference_pool, num_iteration=num_iteration
    )
    if foreground_leaves.shape[1] != tree_limit or reference_leaves.shape[1] != tree_limit:
        raise RuntimeError("native leaf-index output does not match the selected tree count")

    scores = np.zeros(
        (foreground_pool.num_rows, prediction_dimension, reference_pool.num_rows),
        dtype=np.float64,
    )
    covered_tree_count = np.zeros(
        (foreground_pool.num_rows, prediction_dimension), dtype=np.int64
    )
    output_tree_count = np.zeros(prediction_dimension, dtype=np.int64)
    default_learning_rate = float(state["learning_rate"])
    learning_rates = [float(value) for value in state.get("tree_learning_rates", ())]

    for tree_index, tree in enumerate(trees):
        output_index = 0 if prediction_dimension == 1 else tree_index % prediction_dimension
        iteration_index = tree_index // prediction_dimension
        output_tree_count[output_index] += 1
        learning_rate = (
            learning_rates[iteration_index]
            if iteration_index < len(learning_rates)
            else default_learning_rate
        )
        nodes = list(tree["nodes"])
        foreground_column = foreground_leaves[:, tree_index]
        reference_column = reference_leaves[:, tree_index]
        for leaf_index in np.unique(foreground_column):
            explained_rows = np.flatnonzero(foreground_column == leaf_index)
            reference_rows = np.flatnonzero(reference_column == leaf_index)
            if reference_rows.size == 0:
                continue
            leaf_weights = reference_weights[reference_rows]
            total_leaf_weight = float(leaf_weights.sum())
            if total_leaf_weight <= 0.0:
                continue
            node_index = int(leaf_index)
            if node_index < 0 or node_index >= len(nodes) or not bool(
                nodes[node_index].get("is_leaf", False)
            ):
                raise RuntimeError("native leaf index does not identify a fitted leaf node")
            contribution = learning_rate * float(nodes[node_index]["leaf_weight"])
            shares = contribution * leaf_weights / total_leaf_weight
            scores[
                explained_rows[:, np.newaxis],
                output_index,
                reference_rows[np.newaxis, :],
            ] += shares[np.newaxis, :]
            covered_tree_count[explained_rows, output_index] += 1

    coverage = np.divide(
        covered_tree_count,
        output_tree_count[np.newaxis, :],
        out=np.ones_like(covered_tree_count, dtype=np.float64),
        where=output_tree_count[np.newaxis, :] > 0,
    )
    if prediction_dimension == 1:
        resolved_scores: np.ndarray = scores[:, 0]
        resolved_coverage: np.ndarray = coverage[:, 0]
    else:
        resolved_scores = scores
        resolved_coverage = coverage
    if return_coverage:
        return resolved_scores, resolved_coverage
    return resolved_scores


def _top_influence_indices(scores: np.ndarray, top_size: int) -> np.ndarray:
    order = np.argsort(-np.abs(scores), axis=-1, kind="stable")
    return order[..., :top_size]


def get_object_importance(
    booster: Any,
    data: Any,
    reference_data: Any,
    *,
    top_size: int = -1,
    importance_type: str = "Average",
    prediction_dimension: Optional[int] = None,
    num_iteration: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rank reference objects using :func:`calc_leaf_influence` scores.

    ``importance_type='Average'`` averages signed scores across explained
    rows before ranking. ``'PerObject'`` ranks separately for every explained
    row. Ranking uses absolute magnitude but returned scores retain their sign.
    For multiclass models, ``prediction_dimension`` is required.

    Despite the familiar convenience name, these are not exact leave-one-out
    or model-refit importances; they are the leaf approximation documented by
    :func:`calc_leaf_influence`.
    """
    influence = np.asarray(
        calc_leaf_influence(
            booster,
            data,
            reference_data,
            num_iteration=num_iteration,
        ),
        dtype=np.float64,
    )
    if influence.ndim == 3:
        if prediction_dimension is None:
            raise ValueError(
                "prediction_dimension is required for multiclass object importance"
            )
        resolved_dimension = int(prediction_dimension)
        if resolved_dimension < 0 or resolved_dimension >= influence.shape[1]:
            raise ValueError("prediction_dimension is out of range")
        influence = influence[:, resolved_dimension, :]
    if influence.shape[0] == 0:
        raise ValueError("data must contain at least one row for ranked object importance")

    if isinstance(top_size, bool) or not isinstance(top_size, (int, np.integer)):
        raise TypeError("top_size must be an integer")
    resolved_top_size = int(top_size)
    if resolved_top_size == -1:
        resolved_top_size = influence.shape[1]
    elif resolved_top_size <= 0:
        raise ValueError("top_size must be -1 or a positive integer")
    resolved_top_size = min(resolved_top_size, influence.shape[1])

    normalized_type = str(importance_type).replace("_", "").lower()
    if normalized_type == "average":
        average_scores = influence.mean(axis=0)
        indices = _top_influence_indices(average_scores, resolved_top_size)
        return indices.astype(np.int64, copy=False), average_scores[indices]
    if normalized_type == "perobject":
        indices = _top_influence_indices(influence, resolved_top_size)
        return indices.astype(np.int64, copy=False), np.take_along_axis(
            influence, indices, axis=1
        )
    raise ValueError("importance_type must be one of: Average, PerObject")


def _resolve_tree_state(booster: Any, tree_index: int) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    if isinstance(tree_index, bool) or not isinstance(tree_index, (int, np.integer)):
        raise TypeError("tree_index must be an integer")
    state = dict(booster._handle.export_state())
    trees = list(state.get("trees", ()))
    resolved_index = int(tree_index)
    if resolved_index < 0 or resolved_index >= len(trees):
        raise IndexError(
            f"tree_index {resolved_index} is out of range for an ensemble with {len(trees)} trees"
        )
    return state, trees[resolved_index]


def _dot_escape(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _tree_feature_names(booster: Any, feature_count: int) -> Tuple[str, ...]:
    names = booster.feature_names
    if names is None or len(names) != feature_count:
        return tuple(f"f{index}" for index in range(feature_count))
    return tuple(str(name) for name in names)


def _split_label(
    node: Mapping[str, Any],
    feature_names: Sequence[str],
    *,
    precision: int,
) -> str:
    feature_index = int(node["split_feature_id"])
    feature_name = feature_names[feature_index]
    if bool(node["is_categorical_split"]):
        left_bins = [
            str(index)
            for index, goes_left in enumerate(node["left_categories"])
            if int(goes_left) != 0
        ]
        if len(left_bins) > 12:
            left_bins = left_bins[:12] + ["..."]
        condition = "left bins = {" + ", ".join(left_bins) + "}"
    else:
        condition = f"bin <= {int(node['split_bin_index'])}"
    return f"{feature_name} [#{feature_index}]\n{condition}"


def tree_to_dot(
    booster: Any,
    tree_index: int = 0,
    *,
    rankdir: str = "TB",
    precision: int = 6,
) -> str:
    """Return one fitted tree as dependency-free Graphviz DOT source."""
    state, tree = _resolve_tree_state(booster, tree_index)
    schema = state.get("quantization_schema", {})
    feature_count = len(schema.get("num_bins_per_feature", ()))
    feature_names = _tree_feature_names(booster, feature_count)
    prediction_dimension = int(booster.prediction_dimension)
    iteration_index = int(tree_index) // prediction_dimension
    learning_rates = [float(value) for value in state.get("tree_learning_rates", ())]
    learning_rate = (
        learning_rates[iteration_index]
        if iteration_index < len(learning_rates)
        else float(state["learning_rate"])
    )

    lines = [
        "digraph CTBoostTree {",
        f'  graph [rankdir="{_dot_escape(rankdir)}"];',
        '  node [fontname="Helvetica", fontsize="10"];',
        '  edge [fontname="Helvetica", fontsize="9"];',
    ]
    nodes = list(tree["nodes"])
    for node_index, node in enumerate(nodes):
        if bool(node["is_leaf"]):
            weight = float(node["leaf_weight"])
            contribution = learning_rate * weight
            label = (
                f"leaf {node_index}\\nweight={weight:.{precision}g}"
                f"\\ncontribution={contribution:.{precision}g}"
            )
            lines.append(
                f'  n{node_index} [shape="ellipse", label="{_dot_escape(label)}"];'
            )
            continue
        label = _split_label(node, feature_names, precision=precision).replace("\n", "\\n")
        lines.append(f'  n{node_index} [shape="box", label="{_dot_escape(label)}"];')
        lines.append(
            f'  n{node_index} -> n{int(node["left_child"])} [label="yes / left"];'
        )
        lines.append(
            f'  n{node_index} -> n{int(node["right_child"])} [label="no / right"];'
        )
    lines.append("}")
    return "\n".join(lines) + "\n"


def plot_tree(
    booster: Any,
    tree_index: int = 0,
    *,
    ax: Any = None,
    figsize: Tuple[float, float] = (12.0, 7.0),
    precision: int = 4,
) -> Any:
    """Plot one fitted tree with matplotlib and return its ``Axes``."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            "plot_tree requires matplotlib; install it with 'pip install matplotlib'"
        ) from exc

    state, tree = _resolve_tree_state(booster, tree_index)
    schema = state.get("quantization_schema", {})
    feature_count = len(schema.get("num_bins_per_feature", ()))
    feature_names = _tree_feature_names(booster, feature_count)
    nodes = list(tree["nodes"])
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    positions: Dict[int, Tuple[float, float]] = {}
    next_leaf = [0.0]

    def layout(node_index: int, depth: int) -> float:
        node = nodes[node_index]
        if bool(node["is_leaf"]):
            x_position = next_leaf[0]
            next_leaf[0] += 1.0
        else:
            left_position = layout(int(node["left_child"]), depth + 1)
            right_position = layout(int(node["right_child"]), depth + 1)
            x_position = (left_position + right_position) / 2.0
        positions[node_index] = (x_position, -float(depth))
        return x_position

    layout(0, 0)
    for node_index, node in enumerate(nodes):
        x_position, y_position = positions[node_index]
        if bool(node["is_leaf"]):
            label = f"leaf\n{float(node['leaf_weight']):.{precision}g}"
            box = {"boxstyle": "round", "facecolor": "#e8f3e8", "edgecolor": "#367c36"}
        else:
            label = _split_label(node, feature_names, precision=precision)
            box = {"boxstyle": "round", "facecolor": "#eaf1fb", "edgecolor": "#315f91"}
            for child, edge_label in (
                (int(node["left_child"]), "yes"),
                (int(node["right_child"]), "no"),
            ):
                child_x, child_y = positions[child]
                ax.plot(
                    [x_position, child_x],
                    [y_position, child_y],
                    color="#606060",
                    linewidth=1.0,
                    zorder=1,
                )
                ax.text(
                    (x_position + child_x) / 2.0,
                    (y_position + child_y) / 2.0,
                    edge_label,
                    fontsize=8,
                    color="#505050",
                )
        ax.text(
            x_position,
            y_position,
            label,
            ha="center",
            va="center",
            fontsize=9,
            bbox=box,
            zorder=2,
        )

    ax.set_title(f"CTBoost tree {int(tree_index)}")
    ax.margins(x=0.08, y=0.15)
    ax.axis("off")
    return ax


def _resolve_stat_features(
    booster: Any,
    feature: Any,
    feature_count: int,
) -> Tuple[int, ...]:
    feature_names = _tree_feature_names(booster, feature_count)
    name_to_index = {name: index for index, name in enumerate(feature_names)}
    if feature is None:
        values = list(range(feature_count))
    elif isinstance(feature, (str, int, np.integer)):
        values = [feature]
    else:
        values = list(feature)
    resolved = []
    for value in values:
        if isinstance(value, str):
            if value not in name_to_index:
                raise ValueError(f"unknown feature name {value!r}")
            index = name_to_index[value]
        elif isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            index = int(value)
        else:
            raise TypeError("feature entries must be feature indices or names")
        if index < 0 or index >= feature_count:
            raise ValueError(f"feature index {index} is out of range")
        if index not in resolved:
            resolved.append(index)
    if not resolved:
        raise ValueError("feature cannot be empty")
    return tuple(resolved)


def calc_feature_statistics(
    booster: Any,
    data: Any,
    target: Any = None,
    *,
    feature: Any = None,
    prediction_dimension: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """Aggregate prediction and optional target statistics by fitted feature bin."""

    pool = booster._prediction_pool(data)
    state = dict(booster._handle.export_state())
    schema = state.get("quantization_schema")
    if schema is None:
        raise ValueError("the fitted booster does not contain a quantization schema")
    feature_count = len(schema["num_bins_per_feature"])
    if pool.num_cols != feature_count:
        raise ValueError(
            f"data has {pool.num_cols} transformed features, but the model expects {feature_count}"
        )
    selected_features = _resolve_stat_features(booster, feature, feature_count)
    matrix = np.asarray(pool.data, dtype=np.float32)
    predictions = np.asarray(booster.predict(pool), dtype=np.float64)
    if predictions.ndim == 1:
        resolved_predictions = predictions
    else:
        if prediction_dimension is None:
            raise ValueError(
                "prediction_dimension is required for multi-output feature statistics"
            )
        resolved_dimension = int(prediction_dimension)
        if resolved_dimension < 0 or resolved_dimension >= predictions.shape[1]:
            raise ValueError("prediction_dimension is out of range")
        resolved_predictions = predictions[:, resolved_dimension]

    labels = None
    if target is not None:
        labels = np.asarray(target, dtype=np.float64).reshape(-1)
    elif np.asarray(pool.label).shape[0] == pool.num_rows:
        labels = np.asarray(pool.label, dtype=np.float64).reshape(-1)
    if labels is not None and labels.shape[0] != pool.num_rows:
        raise ValueError("target size must match the number of data rows")
    weights = (
        np.ones(pool.num_rows, dtype=np.float64)
        if pool.weight is None
        else np.asarray(pool.weight, dtype=np.float64).reshape(-1)
    )
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("data weights must be finite and non-negative")

    names = _tree_feature_names(booster, feature_count)
    cut_offsets = schema["cut_offsets"]
    statistics: Dict[str, Dict[str, Any]] = {}
    for feature_index in selected_features:
        bins = _quantize_feature(matrix[:, feature_index], schema, feature_index)
        bin_count = int(schema["num_bins_per_feature"][feature_index])
        object_count = np.bincount(bins, minlength=bin_count).astype(np.int64)
        weight_sum = np.bincount(bins, weights=weights, minlength=bin_count).astype(np.float64)
        prediction_sum = np.bincount(
            bins,
            weights=weights * resolved_predictions,
            minlength=bin_count,
        ).astype(np.float64)
        mean_prediction = np.divide(
            prediction_sum,
            weight_sum,
            out=np.full(bin_count, np.nan, dtype=np.float64),
            where=weight_sum > 0.0,
        )
        finite_feature = np.isfinite(matrix[:, feature_index])
        feature_weight = weights * finite_feature
        feature_sum = np.bincount(
            bins,
            weights=feature_weight * np.nan_to_num(matrix[:, feature_index], nan=0.0),
            minlength=bin_count,
        ).astype(np.float64)
        finite_weight_sum = np.bincount(
            bins, weights=feature_weight, minlength=bin_count
        ).astype(np.float64)
        mean_feature_value = np.divide(
            feature_sum,
            finite_weight_sum,
            out=np.full(bin_count, np.nan, dtype=np.float64),
            where=finite_weight_sum > 0.0,
        )
        cut_begin = int(cut_offsets[feature_index])
        cut_end = int(cut_offsets[feature_index + 1])
        item: Dict[str, Any] = {
            "feature_index": int(feature_index),
            "feature_name": names[feature_index],
            "is_categorical": bool(schema["categorical_mask"][feature_index]),
            "borders": np.asarray(schema["cut_values"][cut_begin:cut_end], dtype=np.float32),
            "bin_index": np.arange(bin_count, dtype=np.int32),
            "object_count": object_count,
            "weight_sum": weight_sum,
            "mean_feature_value": mean_feature_value,
            "mean_prediction": mean_prediction,
        }
        if labels is not None:
            target_sum = np.bincount(
                bins,
                weights=weights * labels,
                minlength=bin_count,
            ).astype(np.float64)
            item["mean_target"] = np.divide(
                target_sum,
                weight_sum,
                out=np.full(bin_count, np.nan, dtype=np.float64),
                where=weight_sum > 0.0,
            )
        statistics[names[feature_index]] = item
    return statistics


def _plot_target_values(
    pool: Pool,
    target: Any,
    *,
    prediction_dimension: Optional[int],
) -> Optional[np.ndarray]:
    if target is None:
        pool_label = np.asarray(pool.label)
        if pool_label.ndim != 1 or pool_label.shape[0] != pool.num_rows:
            return None
        values = pool_label
    else:
        values = np.asarray(target)
    if values.ndim == 2:
        if prediction_dimension is None:
            if values.shape[1] != 1:
                raise ValueError(
                    "prediction_dimension is required for multi-output plot targets"
                )
            values = values[:, 0]
        else:
            dimension = int(prediction_dimension)
            if dimension < 0 or dimension >= values.shape[1]:
                raise ValueError("prediction_dimension is out of range for target")
            values = values[:, dimension]
    elif values.ndim != 1:
        raise ValueError("target must be a 1D array or a 2D multi-output array")
    if values.shape[0] != pool.num_rows:
        raise ValueError("target size must match the number of data rows")
    try:
        return np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("prediction plots require numeric target values") from exc


def plot_predictions(
    booster: Any,
    data: Any,
    target: Any = None,
    *,
    kind: str = "auto",
    prediction_dimension: Optional[int] = None,
    num_iteration: Optional[int] = None,
    ax: Any = None,
    figsize: Tuple[float, float] = (7.0, 5.0),
) -> Any:
    """Plot raw predictions, actual-vs-predicted values, or residuals.

    ``kind='auto'`` selects ``'actual_vs_predicted'`` when a numeric target is
    available and ``'prediction'`` otherwise. The other accepted values are
    ``'prediction'``, ``'actual_vs_predicted'``, and ``'residual'``. This plots
    raw model output; classifier probabilities are deliberately not implied.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            "plot_predictions requires matplotlib; install it with 'pip install matplotlib'"
        ) from exc

    pool = booster._prediction_pool(data)
    if pool.num_rows == 0:
        raise ValueError("data must contain at least one row")
    predictions = np.asarray(
        booster.predict(pool, num_iteration=num_iteration), dtype=np.float64
    )
    if predictions.ndim == 2:
        if prediction_dimension is None:
            raise ValueError(
                "prediction_dimension is required for multiclass prediction plots"
            )
        resolved_dimension = int(prediction_dimension)
        if resolved_dimension < 0 or resolved_dimension >= predictions.shape[1]:
            raise ValueError("prediction_dimension is out of range")
        values = predictions[:, resolved_dimension]
    else:
        if prediction_dimension not in (None, 0):
            raise ValueError("prediction_dimension is out of range")
        values = predictions.reshape(-1)
    targets = _plot_target_values(
        pool,
        target,
        prediction_dimension=prediction_dimension,
    )

    normalized_kind = str(kind).replace("-", "_").lower()
    if normalized_kind == "auto":
        normalized_kind = "actual_vs_predicted" if targets is not None else "prediction"
    aliases = {"actual": "actual_vs_predicted", "residuals": "residual"}
    normalized_kind = aliases.get(normalized_kind, normalized_kind)
    if normalized_kind not in {"prediction", "actual_vs_predicted", "residual"}:
        raise ValueError(
            "kind must be one of: auto, prediction, actual_vs_predicted, residual"
        )
    if normalized_kind != "prediction" and targets is None:
        raise ValueError(f"kind={normalized_kind!r} requires target values")
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if normalized_kind == "prediction":
        row_index = np.arange(values.size)
        ax.plot(row_index, values, label="raw prediction")
        if targets is not None:
            ax.plot(row_index, targets, label="target", alpha=0.8)
        ax.set(xlabel="row", ylabel="raw value", title="CTBoost predictions")
        ax.legend()
    elif normalized_kind == "actual_vs_predicted":
        assert targets is not None
        ax.scatter(targets, values, alpha=0.75)
        finite = np.isfinite(targets) & np.isfinite(values)
        if finite.any():
            lower = float(min(targets[finite].min(), values[finite].min()))
            upper = float(max(targets[finite].max(), values[finite].max()))
            ax.plot([lower, upper], [lower, upper], linestyle="--", color="#555555")
        ax.set(
            xlabel="target",
            ylabel="raw prediction",
            title="CTBoost actual vs predicted",
        )
    else:
        assert targets is not None
        ax.scatter(values, targets - values, alpha=0.75)
        ax.axhline(0.0, linestyle="--", color="#555555")
        ax.set(
            xlabel="raw prediction",
            ylabel="target - prediction",
            title="CTBoost residuals",
        )
    ax.grid(alpha=0.2)
    return ax


def plot_feature_statistics(
    statistics: Mapping[str, Mapping[str, Any]],
    *,
    axes: Any = None,
    figsize: Optional[Tuple[float, float]] = None,
    show_object_count: bool = True,
) -> Any:
    """Plot prediction/target curves and optional object counts by fitted bin."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            "plot_feature_statistics requires matplotlib; install it with 'pip install matplotlib'"
        ) from exc
    items = list(statistics.items())
    if not items:
        raise ValueError("statistics cannot be empty")
    if axes is None:
        _, axes = plt.subplots(
            len(items),
            1,
            squeeze=False,
            figsize=figsize or (9.0, 3.5 * len(items)),
        )
        resolved_axes = list(axes[:, 0])
    elif isinstance(axes, np.ndarray):
        resolved_axes = list(axes.reshape(-1))
    elif isinstance(axes, (list, tuple)):
        resolved_axes = list(axes)
    else:
        resolved_axes = [axes]
    if len(resolved_axes) < len(items):
        raise ValueError("not enough axes were provided for the requested features")
    for axis, (name, item) in zip(resolved_axes, items):
        bins = np.asarray(item["bin_index"])
        axis.plot(bins, item["mean_prediction"], marker="o", label="mean prediction")
        if "mean_target" in item:
            axis.plot(bins, item["mean_target"], marker="s", label="mean target")
        if show_object_count:
            count_axis = axis.twinx()
            count_axis.bar(
                bins,
                item["object_count"],
                color="#777777",
                alpha=0.14,
                width=0.8,
                label="object count",
            )
            count_axis.set_ylabel("object count", color="#666666")
            count_axis.tick_params(axis="y", colors="#666666")
        axis.set(
            xlabel="fitted bin",
            ylabel="value",
            title=f"CTBoost feature statistics: {name}",
        )
        axis.legend()
    return resolved_axes[0] if len(items) == 1 else resolved_axes[: len(items)]


__all__ = [
    "calc_leaf_influence",
    "calc_feature_statistics",
    "explain_booster",
    "get_object_importance",
    "plot_feature_statistics",
    "plot_predictions",
    "plot_tree",
    "tree_to_dot",
]
