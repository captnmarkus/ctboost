"""Production-oriented command line interface for CTBoost.

The CLI intentionally delegates model behavior to CTBoost's public Python
estimators and Booster APIs. File handling and argument validation live here so
deployment scripts receive deterministic output and concise actionable errors.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


class CLIError(RuntimeError):
    """An expected user-facing command line error."""


@dataclass
class LoadedDataset:
    values: Any
    columns: Optional[List[str]]
    arrays: Optional[Dict[str, np.ndarray]] = None
    matrix_key: Optional[str] = None


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _dump_json(value: Any, stream: Any) -> None:
    json.dump(
        _json_ready(value),
        stream,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    stream.write("\n")


def _load_json_params(source: Optional[str]) -> Dict[str, Any]:
    if source is None:
        return {}
    text = source
    if source.startswith("@"):
        path = Path(source[1:])
        if not path.is_file():
            raise CLIError("JSON parameter file does not exist: %s" % path)
        text = path.read_text(encoding="utf-8")
    else:
        try:
            candidate = Path(source)
            if candidate.is_file():
                text = candidate.read_text(encoding="utf-8")
        except OSError:
            # Long inline JSON strings are not necessarily valid OS paths.
            pass
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CLIError(
            "invalid JSON parameters at line %d column %d: %s"
            % (exc.lineno, exc.colno, exc.msg)
        ) from exc
    if not isinstance(value, dict):
        raise CLIError("JSON parameters must decode to an object")
    return dict(value)


def _import_pandas(reason: str) -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise CLIError(
            "%s requires pandas; install it with 'pip install \"ctboost[cli]\"'" % reason
        ) from exc
    return pd


def _select_npz_matrix(
    arrays: Mapping[str, np.ndarray], array_key: Optional[str], path: Path
) -> str:
    if array_key is not None:
        if array_key not in arrays:
            raise CLIError(
                "array key %r was not found in %s; available keys: %s"
                % (array_key, path, ", ".join(sorted(arrays)))
            )
        return array_key
    for preferred in ("data", "X", "features"):
        if preferred in arrays and np.asarray(arrays[preferred]).ndim == 2:
            return preferred
    two_dimensional = sorted(
        key for key, value in arrays.items() if np.asarray(value).ndim == 2
    )
    if len(two_dimensional) == 1:
        return two_dimensional[0]
    if len(arrays) == 1:
        return next(iter(arrays))
    raise CLIError(
        "%s contains multiple arrays; select the feature matrix with --array-key. "
        "Available keys: %s" % (path, ", ".join(sorted(arrays)))
    )


def _load_dataset(path_value: str, *, array_key: Optional[str] = None) -> LoadedDataset:
    path = Path(path_value)
    if not path.is_file():
        raise CLIError("input file does not exist: %s" % path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        try:
            values = np.load(str(path), allow_pickle=False)
        except ValueError as exc:
            raise CLIError(
                "unsafe/object NumPy arrays are not loaded; use numeric .npy or a table format"
            ) from exc
        return LoadedDataset(np.asarray(values), None)
    if suffix == ".npz":
        try:
            with np.load(str(path), allow_pickle=False) as archive:
                arrays = {key: np.asarray(archive[key]) for key in archive.files}
        except ValueError as exc:
            raise CLIError(
                "unsafe/object NPZ arrays are not loaded; use numeric arrays or a table format"
            ) from exc
        if not arrays:
            raise CLIError("NPZ input contains no arrays: %s" % path)
        matrix_key = _select_npz_matrix(arrays, array_key, path)
        return LoadedDataset(arrays[matrix_key], None, arrays, matrix_key)
    if suffix in {".csv", ".tsv"}:
        pd = _import_pandas("CSV/TSV input")
        separator = "\t" if suffix == ".tsv" else ","
        try:
            frame = pd.read_csv(path, sep=separator)
        except Exception as exc:
            raise CLIError("failed to read %s: %s" % (path, exc)) from exc
        return LoadedDataset(frame, [str(column) for column in frame.columns])
    if suffix in {".parquet", ".pq", ".feather"}:
        pd = _import_pandas("Parquet/Feather input")
        try:
            frame = pd.read_feather(path) if suffix == ".feather" else pd.read_parquet(path)
        except ImportError as exc:
            raise CLIError(
                "Parquet/Feather input requires pyarrow; install 'ctboost[cli]'"
            ) from exc
        except Exception as exc:
            raise CLIError("failed to read %s: %s" % (path, exc)) from exc
        return LoadedDataset(frame, [str(column) for column in frame.columns])
    raise CLIError(
        "unsupported input format %r; use .npy, .npz, .csv, .tsv, .parquet, or .feather"
        % suffix
    )


def _row_count(values: Any) -> int:
    shape = getattr(values, "shape", None)
    if shape is None or len(shape) == 0:
        raise CLIError("input must contain a row dimension")
    return int(shape[0])


def _resolve_matrix_column(
    selector: str,
    columns: Optional[Sequence[str]],
    column_count: int,
    role: str,
) -> Tuple[int, Optional[str]]:
    if columns is not None and selector in columns:
        return list(columns).index(selector), selector
    try:
        index = int(selector)
    except (TypeError, ValueError) as exc:
        if columns is None:
            raise CLIError("%s must be an integer column index for NumPy input" % role) from exc
        raise CLIError(
            "unknown %s column %r; available columns: %s"
            % (role, selector, ", ".join(columns))
        ) from exc
    if index < 0:
        index += column_count
    if index < 0 or index >= column_count:
        raise CLIError("%s column index %d is out of range" % (role, index))
    return index, None if columns is None else str(columns[index])


def _extract_roles(
    dataset: LoadedDataset,
    selectors: Mapping[str, Optional[str]],
) -> Tuple[Any, Optional[List[str]], Dict[str, np.ndarray]]:
    values = dataset.values
    shape = getattr(values, "shape", ())
    if len(shape) != 2:
        raise CLIError("training/prediction feature input must be a 2D matrix or table")
    row_count = int(shape[0])
    column_count = int(shape[1])
    vectors: Dict[str, np.ndarray] = {}
    dropped_indices: Dict[int, str] = {}
    for role, selector in selectors.items():
        if selector is None:
            continue
        if (
            dataset.arrays is not None
            and selector in dataset.arrays
            and selector != dataset.matrix_key
        ):
            vector = np.asarray(dataset.arrays[selector]).reshape(-1)
        else:
            index, _ = _resolve_matrix_column(selector, dataset.columns, column_count, role)
            if index in dropped_indices:
                raise CLIError(
                    "%s and %s select the same input column"
                    % (dropped_indices[index], role)
                )
            dropped_indices[index] = role
            if dataset.columns is None:
                vector = np.asarray(values)[:, index]
            else:
                vector = np.asarray(values.iloc[:, index])
        if vector.shape[0] != row_count:
            raise CLIError("%s size must match the number of feature rows" % role)
        vectors[role] = vector

    keep = [index for index in range(column_count) if index not in dropped_indices]
    if not keep:
        raise CLIError("no feature columns remain after removing target/metadata columns")
    if dataset.columns is None:
        features = np.asarray(values)[:, keep]
        feature_columns = None
    else:
        features = values.iloc[:, keep].copy()
        feature_columns = [str(dataset.columns[index]) for index in keep]
    return features, feature_columns, vectors


def _load_vector(path_value: str, *, key: Optional[str], role: str) -> np.ndarray:
    dataset = _load_dataset(path_value, array_key=key)
    values = dataset.values
    shape = getattr(values, "shape", ())
    if len(shape) == 1:
        return np.asarray(values).reshape(-1)
    if len(shape) != 2:
        raise CLIError("%s file must contain a vector or a one-column table" % role)
    if key is not None and dataset.columns is not None:
        index, _ = _resolve_matrix_column(key, dataset.columns, int(shape[1]), role)
        return np.asarray(values.iloc[:, index]).reshape(-1)
    if int(shape[1]) != 1:
        raise CLIError(
            "%s file has %d columns; select one with --%s-key"
            % (role, int(shape[1]), role.replace("_", "-"))
        )
    if dataset.columns is None:
        return np.asarray(values)[:, 0]
    return np.asarray(values.iloc[:, 0])


def _split_list(values: Optional[Iterable[str]]) -> List[str]:
    resolved: List[str] = []
    for entry in values or ():
        for item in str(entry).split(","):
            item = item.strip()
            if item and item not in resolved:
                resolved.append(item)
    return resolved


def _resolve_feature_selectors(
    selectors: Sequence[str], features: Any, columns: Optional[Sequence[str]], role: str
) -> List[Any]:
    shape = getattr(features, "shape", ())
    if len(shape) != 2:
        raise CLIError("features must be two-dimensional")
    resolved: List[Any] = []
    for selector in selectors:
        index, name = _resolve_matrix_column(selector, columns, int(shape[1]), role)
        value: Any = index if name is None else name
        if value not in resolved:
            resolved.append(value)
    return resolved


def _drop_columns(dataset: LoadedDataset, selectors: Sequence[str]) -> LoadedDataset:
    if not selectors:
        return dataset
    role_selectors = {"excluded column %d" % index: value for index, value in enumerate(selectors)}
    features, columns, _ = _extract_roles(dataset, role_selectors)
    return LoadedDataset(features, columns)


def _ensure_output_path(path_value: str, *, force: bool) -> Path:
    path = Path(path_value)
    if path.exists() and not force:
        raise CLIError("output already exists: %s (pass --force to overwrite)" % path)
    if path.exists() and path.is_dir():
        raise CLIError("output path is a directory: %s" % path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _prediction_headers(width: int, prefix: str) -> List[str]:
    if width == 1:
        return [prefix]
    return ["%s_%d" % (prefix, index) for index in range(width)]


def _prediction_matrix(predictions: Any) -> Tuple[np.ndarray, bool]:
    array = np.asarray(predictions)
    if array.ndim == 1:
        return array.reshape((-1, 1)), True
    if array.ndim == 2:
        return array, False
    raise CLIError("predictions must be a 1D or 2D array")


def _format_cell(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return format(value, ".17g")
    if value is None:
        return ""
    return str(value)


def _write_predictions(
    path_value: str,
    predictions: Any,
    *,
    prefix: str,
    force: bool,
) -> Path:
    path = _ensure_output_path(path_value, force=force)
    suffix = path.suffix.lower()
    array = np.asarray(predictions)
    matrix, _ = _prediction_matrix(array)
    headers = _prediction_headers(matrix.shape[1], prefix)
    if suffix == ".npy":
        if array.dtype.hasobject:
            raise CLIError(
                "object-valued predictions cannot be written safely as .npy; "
                "use CSV, TSV, or JSON for string/mixed class labels"
            )
        with path.open("wb") as stream:
            np.save(stream, array, allow_pickle=False)
        return path
    if suffix == ".npz":
        if array.dtype.hasobject:
            raise CLIError(
                "object-valued predictions cannot be written safely as .npz; "
                "use CSV, TSV, or JSON for string/mixed class labels"
            )
        with path.open("wb") as stream:
            np.savez(stream, predictions=array)
        return path
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter=delimiter, lineterminator="\n")
            writer.writerow(headers)
            for row in matrix:
                writer.writerow([_format_cell(value) for value in row])
        return path
    if suffix == ".json":
        with path.open("w", encoding="utf-8") as stream:
            _dump_json(array.tolist(), stream)
        return path
    if suffix in {".parquet", ".pq", ".feather"}:
        pd = _import_pandas("Parquet/Feather output")
        frame = pd.DataFrame(matrix, columns=headers)
        try:
            if suffix == ".feather":
                frame.to_feather(path)
            else:
                frame.to_parquet(path, index=False)
        except ImportError as exc:
            raise CLIError(
                "Parquet/Feather output requires pyarrow; install 'ctboost[cli]'"
            ) from exc
        return path
    raise CLIError(
        "unsupported prediction output format %r; use .npy, .npz, .csv, .tsv, "
        ".json, .parquet, or .feather" % suffix
    )


def _load_any_model(
    path_value: str,
    *,
    allow_unsafe_pickle: bool = False,
) -> Tuple[Any, str]:
    import ctboost

    path = Path(path_value)
    if not path.is_file():
        raise CLIError("model file does not exist: %s" % path)
    try:
        with path.open("rb") as stream:
            looks_like_json = stream.read(64).lstrip().startswith(b"{")
    except OSError as exc:
        raise CLIError("failed to read model file %s: %s" % (path, exc)) from exc
    pickle_suffixes = {".pkl", ".pickle"}
    json_suffixes = {".ctb", ".ctboost", ".json"}
    suffix = path.suffix.lower()
    if looks_like_json and suffix in pickle_suffixes:
        raise CLIError(
            "JSON model content cannot use a pickle filename; rename it to .ctb or .json"
        )
    if not looks_like_json and suffix in json_suffixes:
        raise CLIError(
            "non-JSON model content cannot use a .ctb/.ctboost/.json filename"
        )
    if not looks_like_json and not allow_unsafe_pickle:
        raise CLIError(
            "refusing to load a non-JSON model because pickle may execute arbitrary code; "
            "use a trusted .ctb/.json model or pass --allow-unsafe-pickle"
        )

    estimator_names = ("CTBoostClassifier", "CTBoostRegressor", "CTBoostRanker")
    if not looks_like_json:
        # Deserialize a trusted pickle exactly once. Probing each public model
        # loader separately would execute pickle reducers repeatedly.
        import pickle

        try:
            with path.open("rb") as stream:
                loaded = pickle.load(stream)
        except Exception as exc:
            raise CLIError("failed to load trusted pickle model: %s" % exc) from exc

        try:
            estimator_types = tuple(getattr(ctboost, name) for name in estimator_names)
        except ImportError as exc:
            estimator_types = ()
            estimator_import_error: Optional[ImportError] = exc
        else:
            estimator_import_error = None
        if estimator_types and isinstance(loaded, estimator_types):
            loaded._ensure_input_metadata_compatibility()
            return loaded, type(loaded).__name__
        if isinstance(loaded, ctboost.Booster):
            return loaded, "Booster"
        if isinstance(loaded, ctboost._core.GradientBooster):
            return ctboost.Booster(loaded), "Booster"
        if isinstance(loaded, dict) and loaded.get("artifact_type") == "ctboost.booster":
            from ctboost._serialization import (
                MODEL_SCHEMA_VERSION,
                _deserialize_json_value,
            )

            if loaded.get("schema_version") != MODEL_SCHEMA_VERSION:
                raise CLIError("unsupported CTBoost model schema version")
            pipeline_state = loaded.get("feature_pipeline_state")
            training_state = loaded.get("training_state")
            booster = ctboost.Booster(
                ctboost._core.GradientBooster.from_state(loaded["booster_state"]),
                feature_pipeline=(
                    None
                    if pipeline_state is None
                    else ctboost.FeaturePipeline.from_state(
                        _deserialize_json_value(pipeline_state)
                    )
                ),
                training_metadata=(
                    None
                    if training_state is None
                    else _deserialize_json_value(training_state)
                ),
            )
            return booster, "Booster"
        if estimator_import_error is not None:
            raise CLIError(
                "estimator pickle loading requires scikit-learn; install 'ctboost[cli]'"
            ) from estimator_import_error
        raise CLIError("trusted pickle does not contain a supported CTBoost model")

    errors: List[str] = []
    try:
        return ctboost.load_model(path), "Booster"
    except Exception as exc:
        errors.append("Booster: %s" % exc)

    try:
        estimators = [getattr(ctboost, name) for name in estimator_names]
    except ImportError as exc:
        raise CLIError(
            "the artifact is not a low-level Booster and estimator loading requires "
            "scikit-learn; install 'ctboost[cli]'"
        ) from exc
    for estimator in estimators:
        try:
            return estimator.load_model(path), estimator.__name__
        except Exception as exc:
            errors.append("%s: %s" % (estimator.__name__, exc))
    raise CLIError(
        "could not load model through any supported public model API (%s)"
        % "; ".join(errors)
    )


def _model_booster(model: Any) -> Any:
    getter = getattr(model, "get_booster", None)
    return getter() if callable(getter) else model


def _objective_token(model: Any) -> str:
    booster = _model_booster(model)
    return str(getattr(booster, "objective_name", "")).lower()


def _binary_probabilities(raw: np.ndarray) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float64).reshape(-1)
    positive = np.empty_like(values)
    nonnegative = values >= 0.0
    positive[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exp_values = np.exp(values[~nonnegative])
    positive[~nonnegative] = exp_values / (1.0 + exp_values)
    return np.column_stack([1.0 - positive, positive])


def _multiclass_probabilities(raw: np.ndarray) -> np.ndarray:
    scores = np.asarray(raw, dtype=np.float64)
    if scores.ndim != 2:
        raise CLIError("multiclass raw predictions are not a 2D score matrix")
    shifted = scores - scores.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _predict_model(model: Any, data: Any, prediction_type: str, num_iteration: Optional[int]) -> Any:
    kwargs = {} if num_iteration is None else {"num_iteration": int(num_iteration)}
    normalized = prediction_type.lower()
    if normalized == "raw":
        return _model_booster(model).predict(data, **kwargs)
    if normalized == "probability" and hasattr(model, "predict_proba"):
        return model.predict_proba(data, **kwargs)
    if normalized == "class" and hasattr(model, "predict_proba"):
        return model.predict(data, **kwargs)

    raw = np.asarray(_model_booster(model).predict(data, **kwargs))
    objective = _objective_token(model)
    binary_tokens = {"logloss", "binary_logloss", "binary:logistic"}
    multiclass_tokens = {"multiclass", "softmax", "softmaxloss"}
    if objective in binary_tokens:
        probabilities = _binary_probabilities(raw)
    elif objective in multiclass_tokens:
        probabilities = _multiclass_probabilities(raw)
    else:
        raise CLIError(
            "prediction type %r is only valid for classification models; model objective is %r"
            % (prediction_type, getattr(_model_booster(model), "objective_name", "unknown"))
        )
    if normalized == "probability":
        return probabilities
    return (
        (probabilities[:, 1] >= 0.5).astype(np.int64)
        if objective in binary_tokens
        else np.argmax(probabilities, axis=1).astype(np.int64)
    )


def _normalize_numeric_target(values: Any, role: str) -> np.ndarray:
    try:
        target = np.asarray(values, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise CLIError("%s must contain numeric values" % role) from exc
    if not np.isfinite(target).all():
        raise CLIError("%s must contain only finite values" % role)
    return target


def _normalize_group_id(values: Any) -> np.ndarray:
    raw = np.asarray(values).reshape(-1)
    if np.issubdtype(raw.dtype, np.integer):
        return np.asarray(raw, dtype=np.int64)
    try:
        # Preserve arbitrary numeric/string identifiers instead of truncating
        # floating-point group IDs (for example, 1.2 and 1.8) to the same int.
        _, inverse = np.unique(raw, return_inverse=True)
        return np.asarray(inverse, dtype=np.int64)
    except TypeError:
        # Mixed object arrays are not orderable on modern NumPy.  A typed key
        # keeps values such as integer 1 and string "1" distinct.
        mapping: Dict[Tuple[str, str, str], int] = {}
        encoded = np.empty(raw.size, dtype=np.int64)
        for index, value in enumerate(raw):
            resolved = value.item() if isinstance(value, np.generic) else value
            key = (type(resolved).__module__, type(resolved).__qualname__, repr(resolved))
            if key not in mapping:
                mapping[key] = len(mapping)
            encoded[index] = mapping[key]
        return encoded


def _training_parameters(args: argparse.Namespace) -> Dict[str, Any]:
    params = _load_json_params(args.params)
    if "objective" in params:
        if "loss_function" in params and params["loss_function"] != params["objective"]:
            raise CLIError("JSON parameters contain conflicting objective and loss_function")
        params["loss_function"] = params.pop("objective")
    overrides = {
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "leaf_estimation_iterations": args.leaf_estimation_iterations,
        "feature_test": args.feature_test,
        "feature_test_bins": args.feature_test_bins,
        "feature_test_adjustment": args.feature_test_adjustment,
        "random_seed": args.random_seed,
        "task_type": args.task_type,
        "devices": args.devices,
    }
    for key, value in overrides.items():
        if value is not None:
            params[key] = value
    return params


def _command_train(args: argparse.Namespace) -> int:
    import ctboost

    # Fail before loading data or starting an expensive fit when the requested
    # destination is protected by the default no-overwrite policy.
    destination = _ensure_output_path(args.model, force=args.force)
    suffix = destination.suffix.lower()
    if args.model_format == "json" and suffix in {".pkl", ".pickle"}:
        raise CLIError(
            "--model-format json cannot use a pickle filename; use .ctb, .ctboost, or .json"
        )
    if args.model_format == "pickle" and suffix in {".ctb", ".ctboost", ".json"}:
        raise CLIError(
            "--model-format pickle cannot use a JSON model filename; use .pkl or .pickle"
        )
    pickle_requested = args.model_format == "pickle" or (
        args.model_format is None and suffix in {".pkl", ".pickle"}
    )
    if pickle_requested and not args.allow_unsafe_pickle:
        raise CLIError(
            "pickle models are unsafe to load from untrusted sources; use a .ctb/.json "
            "destination or explicitly pass --allow-unsafe-pickle"
        )
    dataset = _load_dataset(args.input, array_key=args.array_key)
    if args.target_file is not None and args.target is not None:
        raise CLIError("use either --target or --target-file, not both")
    if args.target_file is None and args.target is None:
        raise CLIError("training requires --target or --target-file")
    if args.target_key is not None and args.target_file is None:
        raise CLIError("--target-key requires --target-file")
    if args.group_file is not None and args.group is not None:
        raise CLIError("use either --group or --group-file, not both")
    if args.group_key is not None and args.group_file is None:
        raise CLIError("--group-key requires --group-file")
    if args.task != "ranking" and (args.group is not None or args.group_file is not None):
        raise CLIError("--group and --group-file are only valid for ranking training")
    selectors = {
        "target": None if args.target_file is not None else args.target,
        "group": None if args.group_file is not None else args.group,
    }
    features, feature_columns, vectors = _extract_roles(dataset, selectors)
    target = (
        _load_vector(args.target_file, key=args.target_key, role="target")
        if args.target_file is not None
        else vectors["target"]
    )
    group = None
    if args.group_file is not None:
        group = _load_vector(args.group_file, key=args.group_key, role="group")
    elif "group" in vectors:
        group = vectors["group"]
    row_count = _row_count(features)
    if np.asarray(target).reshape(-1).shape[0] != row_count:
        raise CLIError("target size must match the number of feature rows")

    params = _training_parameters(args)
    categorical = _resolve_feature_selectors(
        _split_list(args.categorical), features, feature_columns, "categorical"
    )
    if categorical:
        if params.get("cat_features") not in (None, [], ()):
            raise CLIError("categorical features were provided in both --categorical and JSON params")
        params["cat_features"] = categorical

    try:
        if args.task == "classification":
            model = ctboost.CTBoostClassifier(**params)
            model.fit(features, np.asarray(target).reshape(-1))
        elif args.task == "ranking":
            if group is None:
                raise CLIError("ranking training requires --group or --group-file")
            resolved_target = _normalize_numeric_target(target, "target")
            resolved_group = _normalize_group_id(group)
            if resolved_group.shape[0] != row_count:
                raise CLIError("group size must match the number of feature rows")
            model = ctboost.CTBoostRanker(**params)
            model.fit(features, resolved_target, group_id=resolved_group)
        else:
            model = ctboost.CTBoostRegressor(**params)
            model.fit(features, _normalize_numeric_target(target, "target"))
    except ImportError as exc:
        raise CLIError(
            "CLI training requires scikit-learn; install 'ctboost[cli]'"
        ) from exc
    except CLIError:
        raise
    except Exception as exc:
        raise CLIError("training failed: %s" % exc) from exc

    try:
        model.save_model(destination, model_format=args.model_format)
    except Exception as exc:
        raise CLIError("failed to save model: %s" % exc) from exc
    summary: Dict[str, Any] = {
        "command": "train",
        "model": str(destination.resolve()),
        "model_type": type(model).__name__,
        "rows": row_count,
        "features": int(getattr(features, "shape")[1]),
        "iterations_trained": int(model.get_booster().num_iterations_trained),
        "objective": str(model.get_booster().objective_name),
    }
    if feature_columns is not None:
        summary["feature_columns"] = feature_columns
    if hasattr(model, "classes_"):
        summary["class_labels"] = np.asarray(model.classes_).tolist()
    _dump_json(summary, sys.stdout)
    return 0


def _command_predict(args: argparse.Namespace) -> int:
    model, model_type = _load_any_model(
        args.model,
        allow_unsafe_pickle=args.allow_unsafe_pickle,
    )
    dataset = _load_dataset(args.input, array_key=args.array_key)
    dataset = _drop_columns(dataset, _split_list(args.drop_column))
    values = dataset.values
    shape = getattr(values, "shape", ())
    if len(shape) != 2:
        raise CLIError("prediction input must be a 2D feature matrix or table")
    try:
        predictions = _predict_model(
            model,
            values,
            args.prediction_type,
            args.num_iteration,
        )
    except CLIError:
        raise
    except Exception as exc:
        raise CLIError("prediction failed: %s" % exc) from exc
    output = _write_predictions(
        args.output,
        predictions,
        prefix=args.prediction_type,
        force=args.force,
    )
    summary = {
        "command": "predict",
        "model_type": model_type,
        "output": str(output.resolve()),
        "prediction_type": args.prediction_type,
        "rows": int(shape[0]),
        "shape": list(np.asarray(predictions).shape),
    }
    _dump_json(summary, sys.stdout)
    return 0


def _model_inspection(model: Any, model_type: str) -> Dict[str, Any]:
    booster = _model_booster(model)
    document: Dict[str, Any] = {
        "model_type": model_type,
        "objective": str(getattr(booster, "objective_name", "")),
        "native_objective": str(getattr(booster, "native_objective_name", "")),
        "iterations_trained": int(getattr(booster, "num_iterations_trained")),
        "prediction_dimension": int(getattr(booster, "prediction_dimension")),
        "learning_rate": float(getattr(booster, "learning_rate")),
        "best_iteration": int(getattr(booster, "best_iteration")),
        "feature_names": getattr(booster, "feature_names"),
        "data_schema": getattr(booster, "data_schema"),
    }
    if hasattr(model, "classes_"):
        document["class_labels"] = np.asarray(model.classes_).tolist()
    try:
        document["inference_manifest"] = model.get_inference_manifest()
    except Exception as exc:
        document["inference_manifest_error"] = str(exc)
    return document


def _write_json_document(
    path_value: Optional[str],
    value: Any,
    *,
    command: str,
    force: bool,
) -> None:
    if path_value is None:
        _dump_json(value, sys.stdout)
        return
    destination = _ensure_output_path(path_value, force=force)
    with destination.open("w", encoding="utf-8") as stream:
        _dump_json(value, stream)
    _dump_json({"command": command, "output": str(destination.resolve())}, sys.stdout)


def _command_inspect(args: argparse.Namespace) -> int:
    model, model_type = _load_any_model(
        args.model,
        allow_unsafe_pickle=args.allow_unsafe_pickle,
    )
    _write_json_document(
        args.output,
        _model_inspection(model, model_type),
        command="inspect",
        force=args.force,
    )
    return 0


def _command_info(args: argparse.Namespace) -> int:
    import ctboost

    document = {
        "build": ctboost.build_info(),
        "ctboost_version": ctboost.__version__,
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    _write_json_document(args.output, document, command="info", force=args.force)
    return 0


def _command_export(args: argparse.Namespace) -> int:
    model, model_type = _load_any_model(
        args.model,
        allow_unsafe_pickle=args.allow_unsafe_pickle,
    )
    destination = _ensure_output_path(args.output, force=args.force)
    try:
        if args.format == "manifest":
            model.export_inference_manifest(
                destination,
                prepared_features=args.prepared_features,
            )
        else:
            model.export_model(
                destination,
                export_format=args.format,
                prepared_features=args.prepared_features,
            )
    except Exception as exc:
        raise CLIError("export failed: %s" % exc) from exc
    _dump_json(
        {
            "command": "export",
            "format": args.format,
            "model_type": model_type,
            "output": str(destination.resolve()),
        },
        sys.stdout,
    )
    return 0


def _add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, help="Input .npy/.npz/table file")
    parser.add_argument(
        "--array-key",
        help="Feature-matrix key for NPZ input (defaults to data, X, features, or the only 2D array)",
    )


def _add_force_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output file",
    )


def _add_pickle_safety_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--allow-unsafe-pickle",
        action="store_true",
        help="Allow trusted pickle models (pickle loading can execute arbitrary code)",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build and return the public CLI parser (useful for embedding/tests)."""
    from ._version import __version__

    parser = argparse.ArgumentParser(
        prog="ctboost",
        description="Train, inspect, predict with, and export CTBoost models.",
    )
    parser.add_argument("--version", action="version", version="ctboost %s" % __version__)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show a traceback for unexpected command failures",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and save a CTBoost estimator")
    _add_input_arguments(train_parser)
    train_parser.add_argument("--model", required=True, help="Destination model file")
    train_parser.add_argument(
        "--task",
        choices=("regression", "classification", "ranking"),
        default="regression",
        help="Training task (default: regression)",
    )
    train_parser.add_argument(
        "--target",
        help="Target column name/index, or a separate array key in the input NPZ",
    )
    train_parser.add_argument("--target-file", help="Separate target vector/table file")
    train_parser.add_argument("--target-key", help="Array key or table column in --target-file")
    train_parser.add_argument(
        "--group",
        help="Ranking group column name/index, or a separate array key in the input NPZ",
    )
    train_parser.add_argument("--group-file", help="Separate ranking group vector/table file")
    train_parser.add_argument("--group-key", help="Array key or table column in --group-file")
    train_parser.add_argument(
        "--categorical",
        action="append",
        help="Categorical feature names/indices; repeat or use comma-separated values",
    )
    train_parser.add_argument(
        "--params",
        help="JSON object inline, a JSON file path, or @path (objective aliases loss_function)",
    )
    train_parser.add_argument("--iterations", type=int, help="Override iterations")
    train_parser.add_argument("--learning-rate", type=float, help="Override learning_rate")
    train_parser.add_argument("--max-depth", type=int, help="Override max_depth")
    train_parser.add_argument(
        "--leaf-estimation-iterations",
        type=int,
        choices=range(1, 6),
        help="Override fixed-structure leaf estimation steps for single-output objectives (1-5)",
    )
    train_parser.add_argument(
        "--feature-test",
        choices=("quadratic", "grouped"),
        help="Override the conditional-inference feature test",
    )
    train_parser.add_argument(
        "--feature-test-bins",
        type=int,
        choices=range(2, 65),
        help="Override grouped-test numeric bins (2-64)",
    )
    train_parser.add_argument(
        "--feature-test-adjustment",
        choices=("none", "bonferroni"),
        help="Override the alpha stopping multiplicity adjustment",
    )
    train_parser.add_argument("--random-seed", type=int, help="Override random_seed")
    train_parser.add_argument("--task-type", choices=("CPU", "GPU"), help="Override task_type")
    train_parser.add_argument("--devices", help="Override GPU devices")
    train_parser.add_argument(
        "--model-format",
        choices=("json", "pickle"),
        help="Model serialization format (normally inferred from suffix)",
    )
    _add_pickle_safety_argument(train_parser)
    _add_force_argument(train_parser)
    train_parser.set_defaults(handler=_command_train)

    predict_parser = subparsers.add_parser("predict", help="Run deterministic batch prediction")
    _add_input_arguments(predict_parser)
    predict_parser.add_argument("--model", required=True, help="Saved CTBoost model")
    predict_parser.add_argument("--output", required=True, help="Prediction output file")
    predict_parser.add_argument(
        "--prediction-type",
        choices=("raw", "probability", "class"),
        default="raw",
        help="Output semantics (default: raw model output)",
    )
    predict_parser.add_argument(
        "--num-iteration", type=int, help="Use only the first N boosting iterations"
    )
    predict_parser.add_argument(
        "--drop-column",
        action="append",
        help="Drop table columns before prediction; repeat or use comma-separated values",
    )
    _add_pickle_safety_argument(predict_parser)
    _add_force_argument(predict_parser)
    predict_parser.set_defaults(handler=_command_predict)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect a saved model as JSON")
    inspect_parser.add_argument("--model", required=True, help="Saved CTBoost model")
    inspect_parser.add_argument("--output", help="Write JSON to a file instead of stdout")
    _add_pickle_safety_argument(inspect_parser)
    _add_force_argument(inspect_parser)
    inspect_parser.set_defaults(handler=_command_inspect)

    info_parser = subparsers.add_parser("info", help="Show CTBoost build/runtime information")
    info_parser.add_argument("--output", help="Write JSON to a file instead of stdout")
    _add_force_argument(info_parser)
    info_parser.set_defaults(handler=_command_info)

    export_parser = subparsers.add_parser("export", help="Export a deployment artifact")
    export_parser.add_argument("--model", required=True, help="Saved CTBoost model")
    export_parser.add_argument("--output", required=True, help="Export destination")
    export_parser.add_argument(
        "--format",
        required=True,
        choices=("json_predictor", "python", "cpp", "onnx", "manifest"),
        help="Deployment artifact format",
    )
    export_parser.add_argument(
        "--prepared-features",
        action="store_true",
        help="Export a scorer that expects already-transformed numeric features",
    )
    _add_pickle_safety_argument(export_parser)
    _add_force_argument(export_parser)
    export_parser.set_defaults(handler=_command_export)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(None if argv is None else list(argv))
    try:
        return int(args.handler(args))
    except CLIError as exc:
        print("ctboost %s: error: %s" % (args.command, exc), file=sys.stderr)
        return 2
    except Exception as exc:
        if args.debug:
            raise
        print("ctboost %s: error: %s" % (args.command, exc), file=sys.stderr)
        print("rerun with --debug before the subcommand for a traceback", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess tests
    raise SystemExit(main())


__all__ = ["CLIError", "build_parser", "main"]
