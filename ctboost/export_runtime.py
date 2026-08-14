"""Runtime helpers for standalone CTBoost exports."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
import json
import math
from pathlib import Path
from typing import Any, Union

from .export_payload import JSON_PREDICTOR_FORMAT, JSON_PREDICTOR_FORMAT_VERSION
from .inference_manifest import _fingerprint, _model_fingerprint, validate_inference_manifest

PathLike = Union[str, Path]
_DEFAULT_MAX_ARTIFACT_BYTES = 512 * 1024 * 1024


def _require_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        qualifier = "" if minimum is None else f" >= {minimum}"
        raise ValueError(f"{name} must be an integer{qualifier}")
    return value


def _require_finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        resolved = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be a finite number")
    return resolved


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"predictor JSON contains non-finite value {value!r}")


def _reject_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"predictor JSON contains duplicate key {key!r}")
        result[key] = value
    return result


class ExportedPredictor:
    def __init__(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            raise ValueError("predictor document must be a JSON object")
        self.payload = dict(payload)
        artifact_format = self.payload.get("format")
        if artifact_format is not None and artifact_format != JSON_PREDICTOR_FORMAT:
            raise ValueError(f"unsupported predictor format: {artifact_format!r}")
        format_version = self.payload.get("format_version", 1)
        if type(format_version) is not int or format_version not in {
            1,
            JSON_PREDICTOR_FORMAT_VERSION,
        }:
            raise ValueError(f"unsupported predictor format version: {format_version!r}")
        self.format_version = format_version
        if self.format_version == JSON_PREDICTOR_FORMAT_VERSION and (
            artifact_format != JSON_PREDICTOR_FORMAT
        ):
            raise ValueError("predictor format version 2 requires an explicit format identifier")
        objective_name = self.payload["objective_name"]
        if not isinstance(objective_name, str) or not objective_name:
            raise ValueError("objective_name must be a non-empty string")
        self.objective_name = objective_name
        self.learning_rate = _require_finite_number(
            self.payload["learning_rate"], "learning_rate"
        )
        tree_learning_rates = self.payload.get("tree_learning_rates")
        if tree_learning_rates is not None and not isinstance(tree_learning_rates, list):
            raise ValueError("tree_learning_rates must be an array")
        self.tree_learning_rates = [] if tree_learning_rates is None else [
            _require_finite_number(value, "tree_learning_rates")
            for value in tree_learning_rates
        ]
        self.prediction_dimension = _require_int(
            self.payload["prediction_dimension"],
            "prediction_dimension",
            minimum=1,
        )
        trees = self.payload["trees"]
        if not isinstance(trees, list) or not trees:
            raise ValueError("trees must be a non-empty array")
        if len(trees) % self.prediction_dimension != 0:
            raise ValueError("tree count must be divisible by prediction_dimension")
        self.trees = list(trees)
        base_score_value = self.payload.get(
            "base_score", [0.0] * self.prediction_dimension
        )
        if not isinstance(base_score_value, list):
            raise ValueError("base_score must be an array")
        self.base_score = [
            _require_finite_number(value, "base_score")
            for value in base_score_value
        ]
        if len(self.base_score) != self.prediction_dimension:
            raise ValueError("predictor base_score dimension mismatch")
        self.num_features = _require_int(
            self.payload["num_features"], "num_features", minimum=0
        )
        prepared_value = self.payload["expects_prepared_features"]
        if type(prepared_value) is not bool:
            raise ValueError("expects_prepared_features must be a JSON boolean")
        self.expects_prepared_features = prepared_value
        pipeline_state = self.payload.get("feature_pipeline_state")
        if pipeline_state is not None and not isinstance(pipeline_state, dict):
            raise ValueError("feature_pipeline_state must be an object or null")
        self.feature_pipeline_state = None if pipeline_state is None else dict(pipeline_state)
        if not self.expects_prepared_features:
            if self.format_version != JSON_PREDICTOR_FORMAT_VERSION:
                raise ValueError("predictor format version 1 supports prepared features only")
            if self.feature_pipeline_state is None:
                raise ValueError("raw-feature predictor is missing feature_pipeline_state")
            if self.feature_pipeline_state.get("feature_pipeline_format_version") != 3:
                raise ValueError(
                    "raw-feature predictor exports require feature-pipeline format version 3"
                )
            if self.feature_pipeline_state.get("categorical_key_encoding_version") != 2:
                raise ValueError(
                    "raw-feature predictor exports require categorical key encoding version 2"
                )
        quantization_schema = self.payload["quantization_schema"]
        if not isinstance(quantization_schema, dict):
            raise ValueError("quantization_schema must be an object")
        self.quantization_schema = dict(quantization_schema)
        self._validate_quantization_layout()
        self._validate_tree_model()
        class_labels = self.payload.get("class_labels")
        if class_labels is not None and not isinstance(class_labels, list):
            raise ValueError("class_labels must be an array or null")
        self.class_labels = None if class_labels is None else list(class_labels)
        self._validate_output_labels()
        manifest = self.payload.get("inference_manifest")
        if manifest is not None and not isinstance(manifest, dict):
            raise ValueError("inference_manifest must be an object or null")
        self.inference_manifest = (
            None if manifest is None else validate_inference_manifest(manifest)
        )
        if (
            self.inference_manifest is not None
            and self.inference_manifest["model"]["fingerprint"]
            != _model_fingerprint(self.payload, self.class_labels)
        ):
            raise ValueError("predictor model fingerprint mismatch")
        if self.inference_manifest is not None and self.feature_pipeline_state is not None:
            preprocessing = self.inference_manifest["input"]["preprocessing"]
            if not isinstance(preprocessing, dict):
                raise ValueError("predictor preprocessing manifest is missing")
            if preprocessing.get("fingerprint") != _fingerprint(
                self.feature_pipeline_state
            ):
                raise ValueError("predictor feature-pipeline fingerprint mismatch")
        self._validate_input_envelope()
        self._feature_pipeline = None
        if self.feature_pipeline_state is not None and not self.expects_prepared_features:
            if self.inference_manifest is None:
                raise ValueError(
                    "raw-feature predictor exports require an inference manifest"
                )
            if self.feature_pipeline_state.get("feature_pipeline_format_version") != 3:
                raise ValueError(
                    "raw-feature predictor exports require feature-pipeline format version 3"
                )
            if self.feature_pipeline_state.get("categorical_key_encoding_version") != 2:
                raise ValueError(
                    "raw-feature predictor exports require categorical key encoding version 2"
                )
            output_names = self.feature_pipeline_state.get("output_feature_names_")
            if not isinstance(output_names, list) or len(output_names) != self.num_features:
                raise ValueError(
                    "predictor feature-pipeline output dimension mismatch"
                )

            # Native state construction happens only after the manifest checksum and
            # exported dimensions have been cross-checked. Native LoadState performs
            # full structural validation before the pipeline can transform data. The
            # checksum detects accidental/tampered-envelope drift; it is not a signature.
            from .feature_pipeline import FeaturePipeline

            self._feature_pipeline = FeaturePipeline.from_state(
                self.feature_pipeline_state
            )

    def _validate_quantization_layout(self) -> None:
        schema = self.quantization_schema
        arrays = {}
        for key in (
            "num_bins_per_feature",
            "categorical_mask",
            "missing_value_mask",
            "cut_offsets",
            "cut_values",
        ):
            value = schema.get(key)
            if not isinstance(value, list):
                raise ValueError(f"quantization_schema.{key} must be an array")
            arrays[key] = value
        feature_count = self.num_features
        for key in (
            "num_bins_per_feature",
            "categorical_mask",
            "missing_value_mask",
        ):
            if len(arrays[key]) != feature_count:
                raise ValueError(f"quantization_schema.{key} dimension mismatch")
        nan_modes_value = schema.get("nan_modes")
        nan_modes = [] if nan_modes_value is None else nan_modes_value
        if not isinstance(nan_modes, list) or (
            nan_modes and len(nan_modes) != feature_count
        ):
            raise ValueError("quantization_schema.nan_modes dimension mismatch")
        offsets = arrays["cut_offsets"]
        if len(offsets) != feature_count + 1 or offsets[0] != 0:
            raise ValueError("quantization_schema.cut_offsets dimension mismatch")
        previous = -1
        for offset in offsets:
            if type(offset) is not int or offset < previous:
                raise ValueError("quantization_schema.cut_offsets must be monotone integers")
            previous = offset
        if offsets[-1] != len(arrays["cut_values"]):
            raise ValueError("quantization_schema cut value dimension mismatch")
        bins = arrays["num_bins_per_feature"]
        if any(type(value) is not int or not 0 <= value <= 65535 for value in bins):
            raise ValueError("quantization_schema bin counts must be uint16 integers")
        for key in ("categorical_mask", "missing_value_mask"):
            if any(
                not (
                    type(value) is bool
                    or type(value) is int and value in (0, 1)
                )
                for value in arrays[key]
            ):
                raise ValueError(f"quantization_schema.{key} must contain only 0/1 integers")
        default_nan_mode = schema.get("nan_mode")
        if type(default_nan_mode) is not int or default_nan_mode not in (0, 1, 2):
            raise ValueError("quantization_schema.nan_mode must be 0, 1, or 2")
        if any(type(value) is not int or value not in (0, 1, 2) for value in nan_modes):
            raise ValueError("quantization_schema.nan_modes must contain only 0, 1, or 2")
        cuts = [
            _require_finite_number(value, "quantization_schema.cut_values")
            for value in arrays["cut_values"]
        ]
        for feature_index in range(feature_count):
            begin = offsets[feature_index]
            end = offsets[feature_index + 1]
            feature_cuts = cuts[begin:end]
            if any(
                feature_cuts[index] <= feature_cuts[index - 1]
                for index in range(1, len(feature_cuts))
            ):
                raise ValueError("quantization_schema cuts must be strictly increasing per feature")
            non_missing_bins = bins[feature_index] - arrays["missing_value_mask"][feature_index]
            if non_missing_bins < 0:
                raise ValueError("quantization_schema missing-value bin count is inconsistent")
            expected_cut_count = (
                non_missing_bins
                if arrays["categorical_mask"][feature_index] == 1
                else max(non_missing_bins - 1, 0)
            )
            if len(feature_cuts) != expected_cut_count:
                raise ValueError("quantization_schema feature cut count is inconsistent")

    def _validate_tree_model(self) -> None:
        if len(self.trees) % self.prediction_dimension != 0:
            raise ValueError("tree count must be divisible by prediction_dimension")
        iteration_count = len(self.trees) // self.prediction_dimension
        if len(self.tree_learning_rates) > iteration_count:
            raise ValueError("tree_learning_rates dimension mismatch")
        categorical_mask = self.quantization_schema["categorical_mask"]
        bins = self.quantization_schema["num_bins_per_feature"]
        for tree_index, tree in enumerate(self.trees):
            if not isinstance(tree, dict):
                raise ValueError(f"trees[{tree_index}] must be an object")
            nodes = tree.get("nodes")
            if not isinstance(nodes, list) or not nodes:
                raise ValueError(f"trees[{tree_index}] must contain nodes")
            for node_index, node in enumerate(nodes):
                context = f"trees[{tree_index}].nodes[{node_index}]"
                if not isinstance(node, dict):
                    raise ValueError(f"{context} must be an object")
                is_leaf = node.get("is_leaf")
                is_categorical = node.get("is_categorical_split")
                if type(is_leaf) is not bool or type(is_categorical) is not bool:
                    raise ValueError(f"{context} flags must be JSON booleans")
                feature = _require_int(node.get("split_feature_id"), f"{context}.split_feature_id")
                split_bin = _require_int(node.get("split_bin_index"), f"{context}.split_bin_index")
                left = _require_int(node.get("left_child"), f"{context}.left_child")
                right = _require_int(node.get("right_child"), f"{context}.right_child")
                _require_finite_number(node.get("leaf_weight"), f"{context}.leaf_weight")
                routes = node.get("left_categories")
                if not isinstance(routes, list) or any(
                    not (
                        type(value) is bool
                        or type(value) is int and value in (0, 1)
                    )
                    for value in routes
                ):
                    raise ValueError(f"{context}.left_categories must be a 0/1 array")
                if is_leaf:
                    if left != -1 or right != -1:
                        raise ValueError(f"{context} leaf fields are inconsistent")
                    continue
                if not 0 <= feature < self.num_features:
                    raise ValueError(f"{context} split feature is out of bounds")
                if not 0 <= split_bin < bins[feature]:
                    raise ValueError(f"{context} split bin is out of bounds")
                if (
                    not 0 <= left < len(nodes)
                    or not 0 <= right < len(nodes)
                    or left == right
                ):
                    raise ValueError(f"{context} child index is invalid")
                if is_categorical != bool(categorical_mask[feature]):
                    raise ValueError(f"{context} categorical split flag is inconsistent")
                if is_categorical:
                    if len(routes) < bins[feature]:
                        raise ValueError(f"{context} categorical route dimension mismatch")

            visited: set[int] = set()
            pending = [0]
            while pending:
                node_index = pending.pop()
                if node_index in visited:
                    raise ValueError(
                        f"trees[{tree_index}] contains a cycle or shared child"
                    )
                visited.add(node_index)
                node = nodes[node_index]
                if not node["is_leaf"]:
                    pending.extend((node["right_child"], node["left_child"]))
            if len(visited) != len(nodes):
                raise ValueError(f"trees[{tree_index}] contains unreachable nodes")

    def _validate_output_labels(self) -> None:
        objective = self.objective_name.strip().lower()
        if objective in {"logloss", "binary_logloss", "binary:logistic"}:
            if self.prediction_dimension != 1:
                raise ValueError("binary objectives require prediction_dimension=1")
            expected_labels = 2
        elif objective in {"multiclass", "softmax", "softmaxloss"}:
            if self.prediction_dimension < 2:
                raise ValueError("multiclass objectives require multiple prediction dimensions")
            expected_labels = self.prediction_dimension
        else:
            if self.class_labels is not None:
                raise ValueError("class_labels are only valid for classification objectives")
            return
        if self.class_labels is not None and len(self.class_labels) != expected_labels:
            raise ValueError("class_labels dimension mismatch")

    def _validate_input_envelope(self) -> None:
        if self.format_version == 1:
            if not self.expects_prepared_features:
                raise ValueError("predictor format version 1 supports prepared features only")
            return
        if self.inference_manifest is None:
            if not self.expects_prepared_features:
                raise ValueError(
                    "raw-feature predictor format version 2 requires an inference manifest"
                )
            return

        manifest = self.inference_manifest
        artifact = manifest["artifact"]
        model = manifest["model"]
        input_contract = manifest["input"]
        output = manifest["output"]
        preprocessing = input_contract.get("preprocessing")
        if not isinstance(preprocessing, dict):
            raise ValueError("predictor preprocessing manifest is missing")
        if artifact.get("kind") != "json_predictor":
            raise ValueError("predictor manifest artifact kind mismatch")
        if model.get("objective") != self.objective_name:
            raise ValueError("predictor manifest objective mismatch")
        if type(model.get("prediction_dimension")) is not int or (
            model.get("prediction_dimension") != self.prediction_dimension
        ):
            raise ValueError("predictor manifest prediction dimension mismatch")
        if type(model.get("tree_count")) is not int or model.get("tree_count") != len(self.trees):
            raise ValueError("predictor manifest tree count mismatch")
        manifest_base_score = model.get("base_score")
        if not isinstance(manifest_base_score, list) or [
            _require_finite_number(value, "manifest model.base_score")
            for value in manifest_base_score
        ] != self.base_score:
            raise ValueError("predictor manifest base score mismatch")
        if output.get("objective") != self.objective_name:
            raise ValueError("predictor output objective mismatch")
        if type(output.get("prediction_dimension")) is not int or (
            output.get("prediction_dimension") != self.prediction_dimension
        ):
            raise ValueError("predictor output dimension mismatch")
        if type(input_contract.get("model_feature_count")) is not int or (
            input_contract.get("model_feature_count") != self.num_features
        ):
            raise ValueError("predictor manifest model feature count mismatch")

        representation = input_contract.get("representation")
        runtime_required = artifact.get("ctboost_runtime_required")
        if not self.expects_prepared_features:
            state = self.feature_pipeline_state
            if state is None:
                raise ValueError("raw-feature predictor is missing feature_pipeline_state")
            if state.get("feature_pipeline_format_version") != 3:
                raise ValueError(
                    "raw-feature predictor exports require feature-pipeline format version 3"
                )
            if state.get("categorical_key_encoding_version") != 2:
                raise ValueError(
                    "raw-feature predictor exports require categorical key encoding version 2"
                )
            raw_feature_count = state.get("n_features_in_")
            output_names = state.get("output_feature_names_")
            if type(raw_feature_count) is not int or raw_feature_count < 0:
                raise ValueError("raw-feature predictor input feature count is invalid")
            if not isinstance(output_names, list) or len(output_names) != self.num_features:
                raise ValueError("predictor feature-pipeline output dimension mismatch")
            expected = {
                "representation": "raw_features",
                "runtime_required": True,
                "input_features": raw_feature_count,
                "raw_features": raw_feature_count,
                "transformed_features": self.num_features,
                "preprocessing_kind": "ctboost_feature_pipeline",
                "external_preprocessing": False,
                "key_encoding": 2,
            }
            observed = {
                "representation": representation,
                "runtime_required": runtime_required,
                "input_features": input_contract.get("num_features"),
                "raw_features": preprocessing.get("raw_feature_count"),
                "transformed_features": preprocessing.get("transformed_feature_count"),
                "preprocessing_kind": preprocessing.get("kind"),
                "external_preprocessing": preprocessing.get(
                    "external_preprocessing_required"
                ),
                "key_encoding": preprocessing.get(
                    "categorical_key_encoding_version"
                ),
            }
            exact_integer_fields = (
                input_contract.get("num_features"),
                preprocessing.get("raw_feature_count"),
                preprocessing.get("transformed_feature_count"),
                preprocessing.get("categorical_key_encoding_version"),
            )
            if (
                any(type(value) is not int for value in exact_integer_fields)
                or type(runtime_required) is not bool
                or type(preprocessing.get("external_preprocessing_required")) is not bool
                or observed != expected
            ):
                raise ValueError("raw-feature predictor manifest is inconsistent")
            if state.get("categorical_key_encoding_version") != 2:
                raise ValueError("raw-feature predictor categorical key encoding mismatch")
            categorical_indices = [
                index
                for index, value in enumerate(
                    self.quantization_schema["categorical_mask"]
                )
                if bool(value)
            ]
            if state.get("cat_feature_indices_") != categorical_indices:
                raise ValueError(
                    "predictor feature-pipeline categorical layout mismatch"
                )
            source_categorical_indices = sorted(
                {
                    _require_int(item.get("source_index"), "categorical source index", minimum=0)
                    for key in ("one_hot_states", "categorical_states")
                    for item in state.get(key, [])
                    if isinstance(item, dict)
                }
            )
            self._validate_manifest_features(
                input_contract,
                raw_feature_count,
                source_categorical_indices,
            )
            return

        if type(runtime_required) is not bool or runtime_required is not False:
            raise ValueError("prepared-feature predictor must not require the CTBoost runtime")
        if representation not in {
            "prepared_numeric_features",
            "numeric_or_preencoded_categorical_features",
        }:
            raise ValueError("prepared-feature predictor representation is inconsistent")
        if type(input_contract.get("num_features")) is not int or (
            input_contract.get("num_features") != self.num_features
        ):
            raise ValueError("prepared-feature predictor input feature count mismatch")
        if representation == "numeric_or_preencoded_categorical_features":
            if preprocessing.get("kind") != "none" or preprocessing.get(
                "external_preprocessing_required"
            ) is not False:
                raise ValueError("prepared-feature predictor preprocessing is inconsistent")
        else:
            if preprocessing.get("kind") != "ctboost_feature_pipeline" or preprocessing.get(
                "external_preprocessing_required"
            ) is not True:
                raise ValueError("prepared-feature predictor preprocessing is inconsistent")
            if type(preprocessing.get("transformed_feature_count")) is not int or (
                preprocessing.get("transformed_feature_count") != self.num_features
            ):
                raise ValueError("prepared-feature predictor transformed feature count mismatch")
            if self.feature_pipeline_state is not None:
                if type(preprocessing.get("raw_feature_count")) is not int or (
                    preprocessing.get("raw_feature_count")
                    != self.feature_pipeline_state.get("n_features_in_")
                ):
                    raise ValueError("prepared-feature predictor raw feature count mismatch")
                if type(preprocessing.get("categorical_key_encoding_version")) is not int or (
                    preprocessing.get("categorical_key_encoding_version")
                    != self.feature_pipeline_state.get(
                        "categorical_key_encoding_version"
                    )
                ):
                    raise ValueError("prepared-feature predictor key encoding mismatch")
        self._validate_manifest_features(
            input_contract,
            self.num_features,
            [
                index
                for index, value in enumerate(
                    self.quantization_schema["categorical_mask"]
                )
                if value == 1
            ],
        )

    @staticmethod
    def _validate_manifest_features(
        input_contract: dict[str, Any],
        feature_count: int,
        categorical_indices: list[int],
    ) -> None:
        observed_indices = input_contract.get("categorical_feature_indices")
        if (
            not isinstance(observed_indices, list)
            or any(type(value) is not int for value in observed_indices)
            or observed_indices != categorical_indices
        ):
            raise ValueError("predictor manifest categorical feature indices mismatch")
        features = input_contract.get("features")
        if not isinstance(features, list) or len(features) != feature_count:
            raise ValueError("predictor manifest feature dimension mismatch")
        categorical_set = set(categorical_indices)
        for index, feature in enumerate(features):
            if (
                not isinstance(feature, dict)
                or type(feature.get("index")) is not int
                or feature.get("index") != index
                or type(feature.get("categorical")) is not bool
                or feature.get("categorical") is not (index in categorical_set)
            ):
                raise ValueError("predictor manifest feature metadata mismatch")

    def get_inference_manifest(self) -> Union[dict[str, Any], None]:
        """Return a defensive copy of the embedded deployment contract, if present."""
        if self.inference_manifest is None:
            return None
        return validate_inference_manifest(self.inference_manifest)

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
        scores = list(self.base_score)
        for tree_index, tree in enumerate(self.trees):
            nodes = tree["nodes"]
            iteration_index = tree_index // self.prediction_dimension
            tree_learning_rate = (
                self.tree_learning_rates[iteration_index]
                if iteration_index < len(self.tree_learning_rates)
                else self.learning_rate
            )
            node_index = 0
            for _ in range(len(nodes)):
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
            else:  # Defensive if callers mutate a validated payload after construction.
                raise RuntimeError("validated tree traversal exceeded its node count")
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
        raw_single_row = False
        if self._feature_pipeline is not None:
            if isinstance(data, (list, tuple)) and not data:
                return []
            dimension = getattr(data, "ndim", None)
            if dimension == 1:
                data = [data.tolist() if hasattr(data, "tolist") else list(data)]
                raw_single_row = True
            elif isinstance(data, (list, tuple)):
                raw_feature_count = int(self.feature_pipeline_state["n_features_in_"])

                def is_row_like(value: Any) -> bool:
                    if isinstance(value, (str, bytes)):
                        return False
                    if isinstance(value, (list, tuple)):
                        return len(value) == raw_feature_count
                    if getattr(value, "ndim", None) == 1:
                        try:
                            return len(value) == raw_feature_count
                        except TypeError:
                            return False
                    return False

                if not all(is_row_like(value) for value in data) and (
                    len(data) == raw_feature_count
                ):
                    embedding_sources = {
                        int(state["source_index"])
                        for state in self.feature_pipeline_state.get(
                            "embedding_states", []
                        )
                    }
                    nested_positions = {
                        index
                        for index, value in enumerate(data)
                        if isinstance(value, (list, tuple))
                        or getattr(value, "ndim", 0) == 1
                    }
                    if nested_positions.issubset(embedding_sources):
                        data = [list(data)]
                        raw_single_row = True
            data, _, _ = self._feature_pipeline.transform_array(data)
        rows, is_single_row = self._coerce_rows(data)
        predictions = [self._row_scores(row) for row in rows]
        if is_single_row or raw_single_row:
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
                index = 1 if float(raw) >= 0.0 else 0
                return index if self.class_labels is None else self.class_labels[index]
            indices = [1 if float(value) >= 0.0 else 0 for value in raw]
            return (
                indices
                if self.class_labels is None
                else [self.class_labels[index] for index in indices]
            )
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
            if self.class_labels is not None:
                classes = [self.class_labels[index] for index in classes]
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


def load_exported_predictor(
    path: PathLike,
    *,
    max_artifact_bytes: int = _DEFAULT_MAX_ARTIFACT_BYTES,
) -> ExportedPredictor:
    if type(max_artifact_bytes) is not int or max_artifact_bytes <= 0:
        raise ValueError("max_artifact_bytes must be a positive integer")
    with Path(path).open("rb") as stream:
        encoded = stream.read(max_artifact_bytes + 1)
    if len(encoded) > max_artifact_bytes:
        raise ValueError("predictor artifact exceeds the configured size limit")
    try:
        text = encoded.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("predictor artifact must be valid UTF-8 JSON") from error
    payload = json.loads(
        text,
        parse_constant=_reject_nonfinite_json,
        object_pairs_hook=_reject_duplicate_object,
    )
    if not isinstance(payload, dict):
        raise ValueError("predictor document must be a JSON object")
    return ExportedPredictor(payload)
