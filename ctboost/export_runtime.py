"""Runtime helpers for standalone CTBoost exports."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
import json
import math
from pathlib import Path
from typing import Any, Union

PathLike = Union[str, Path]


class ExportedPredictor:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = dict(payload)
        self.objective_name = str(self.payload["objective_name"])
        self.learning_rate = float(self.payload["learning_rate"])
        tree_learning_rates = self.payload.get("tree_learning_rates")
        self.tree_learning_rates = (
            [] if tree_learning_rates is None else [float(value) for value in tree_learning_rates]
        )
        self.prediction_dimension = int(self.payload["prediction_dimension"])
        self.num_features = int(self.payload["num_features"])
        self.expects_prepared_features = bool(self.payload["expects_prepared_features"])
        self.quantization_schema = dict(self.payload["quantization_schema"])
        self.trees = list(self.payload["trees"])

    @staticmethod
    def _is_nan(value: Any) -> bool:
        try:
            return math.isnan(value)
        except TypeError:
            return False

    @staticmethod
    def _coerce_value(value: Any) -> float:
        if value is None:
            return float("nan")
        return float(value)

    @staticmethod
    def _missing_bin_index(bins_for_feature: int, has_missing_values: bool, nan_mode: int) -> int:
        if not has_missing_values:
            return bins_for_feature - 1 if nan_mode == 2 and bins_for_feature > 0 else 0
        return bins_for_feature - 1 if nan_mode == 2 else 0

    def _bin_value(self, feature_index: int, value: Any) -> int:
        num_bins_per_feature = self.quantization_schema["num_bins_per_feature"]
        cut_offsets = self.quantization_schema["cut_offsets"]
        cut_values = self.quantization_schema["cut_values"]
        categorical_mask = self.quantization_schema["categorical_mask"]
        missing_value_mask = self.quantization_schema["missing_value_mask"]
        nan_modes = self.quantization_schema.get("nan_modes") or []
        default_nan_mode = int(self.quantization_schema["nan_mode"])

        bins_for_feature = int(num_bins_per_feature[feature_index])
        if bins_for_feature == 0:
            return 0

        resolved_value = self._coerce_value(value)
        feature_is_categorical = bool(categorical_mask[feature_index])
        feature_has_missing_values = bool(missing_value_mask[feature_index])
        resolved_nan_mode = int(nan_modes[feature_index]) if nan_modes else default_nan_mode
        if self._is_nan(resolved_value):
            return self._missing_bin_index(
                bins_for_feature,
                feature_has_missing_values,
                resolved_nan_mode,
            )

        cut_begin = int(cut_offsets[feature_index])
        cut_end = int(cut_offsets[feature_index + 1])
        cuts = cut_values[cut_begin:cut_end]
        non_missing_bins = bins_for_feature - (1 if feature_has_missing_values else 0)
        if non_missing_bins == 0:
            return self._missing_bin_index(
                bins_for_feature,
                feature_has_missing_values,
                resolved_nan_mode,
            )

        offset = 1 if feature_has_missing_values and resolved_nan_mode == 1 else 0
        if feature_is_categorical:
            insertion = bisect_left(cuts, resolved_value)
            clamped_insertion = min(insertion, non_missing_bins - 1)
            if insertion < len(cuts) and cuts[insertion] == resolved_value:
                return offset + insertion
            return offset + clamped_insertion

        return offset + bisect_right(cuts, resolved_value)

    def _row_scores(self, row: list[Any]) -> Any:
        if len(row) != self.num_features:
            raise ValueError(f"expected {self.num_features} features per row, got {len(row)}")
        bins = [self._bin_value(index, value) for index, value in enumerate(row)]
        scores = [0.0] * self.prediction_dimension
        for tree_index, tree in enumerate(self.trees):
            nodes = tree["nodes"]
            iteration_index = tree_index // self.prediction_dimension
            tree_learning_rate = (
                self.tree_learning_rates[iteration_index]
                if iteration_index < len(self.tree_learning_rates)
                else self.learning_rate
            )
            node_index = 0
            while True:
                node = nodes[node_index]
                if node["is_leaf"]:
                    break
                split_feature = int(node["split_feature_id"])
                split_bin = bins[split_feature]
                if node["is_categorical_split"]:
                    go_left = node["left_categories"][split_bin] != 0
                else:
                    go_left = split_bin <= int(node["split_bin_index"])
                node_index = int(node["left_child"] if go_left else node["right_child"])
            scores[tree_index % self.prediction_dimension] += tree_learning_rate * float(node["leaf_weight"])
        return scores[0] if self.prediction_dimension == 1 else scores

    def _coerce_rows(self, data: Any) -> tuple[list[list[Any]], bool]:
        if hasattr(data, "tolist"):
            data = data.tolist()
        if isinstance(data, tuple):
            data = list(data)
        if not isinstance(data, list):
            raise TypeError("data must be a 1D or 2D array-like object")
        if not data:
            return [], False
        first = data[0]
        if hasattr(first, "tolist"):
            first = first.tolist()
        if isinstance(first, tuple):
            first = list(first)
        if isinstance(first, list):
            rows = []
            for row in data:
                if hasattr(row, "tolist"):
                    row = row.tolist()
                elif isinstance(row, tuple):
                    row = list(row)
                rows.append(list(row))
            return rows, False
        return [list(data)], True

    def predict_raw(self, data: Any) -> Any:
        rows, is_single_row = self._coerce_rows(data)
        predictions = [self._row_scores(row) for row in rows]
        if is_single_row:
            return predictions[0]
        return predictions

    def predict(self, data: Any) -> Any:
        return self.predict_raw(data)

    def predict_proba(self, data: Any) -> Any:
        objective_name = self.objective_name.lower()
        raw = self.predict_raw(data)
        if objective_name in {"logloss", "binary_logloss", "binary:logistic"}:
            is_single_row = isinstance(raw, (int, float))
            rows = [raw] if is_single_row else raw
            probabilities = []
            for value in rows:
                positive = _sigmoid(float(value))
                probabilities.append([1.0 - positive, positive])
            return probabilities[0] if is_single_row else probabilities
        if objective_name in {"multiclass", "softmax", "softmaxloss"}:
            is_single_row = bool(raw) and isinstance(raw[0], (int, float))
            rows = [raw] if is_single_row else raw
            probabilities = []
            for row in rows:
                max_score = max(float(score) for score in row)
                exp_scores = [math.exp(float(score) - max_score) for score in row]
                normalizer = sum(exp_scores)
                probabilities.append([score / normalizer for score in exp_scores])
            return probabilities[0] if is_single_row else probabilities
        raise RuntimeError(
            f"predict_proba is only available for classification objectives, got {self.objective_name!r}"
        )

    def predict_class(self, data: Any) -> Any:
        objective_name = self.objective_name.lower()
        if objective_name in {"logloss", "binary_logloss", "binary:logistic"}:
            raw = self.predict_raw(data)
            if isinstance(raw, (int, float)):
                return 1 if float(raw) >= 0.0 else 0
            return [1 if float(value) >= 0.0 else 0 for value in raw]
        if objective_name in {"multiclass", "softmax", "softmaxloss"}:
            raw = self.predict_raw(data)
            is_single_row = bool(raw) and isinstance(raw[0], (int, float))
            rows = [raw] if is_single_row else raw
            classes = []
            for row in rows:
                best_index = 0
                best_score = float(row[0])
                for index, score in enumerate(row[1:], start=1):
                    if float(score) > best_score:
                        best_index = index
                        best_score = float(score)
                classes.append(best_index)
            return classes[0] if is_single_row else classes
        raise RuntimeError(
            f"predict_class is only available for classification objectives, got {self.objective_name!r}"
        )


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        exp_value = math.exp(-value)
        return 1.0 / (1.0 + exp_value)
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def load_exported_predictor(path: PathLike) -> ExportedPredictor:
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return ExportedPredictor(payload)
