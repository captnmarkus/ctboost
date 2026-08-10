"""Atomic I/O, source provenance, and OpenML staging for the external panel."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import pickle
import platform
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from ._external_panel_protocol import (
    FOLDS,
    REPEAT,
    RESULT_SCHEMA_VERSION,
    array_digest,
    assert_no_absolute_paths,
    json_default,
    sha256_json,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.{}.tmp".format(path.name, uuid.uuid4().hex))
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                value,
                stream,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
                default=json_default,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.{}.tmp".format(path.name, uuid.uuid4().hex))
    try:
        with temporary.open("wb") as stream:
            pickle.dump(value, stream, protocol=5)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _git_output(root: Path, arguments: Sequence[str]) -> Optional[bytes]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root)] + list(arguments),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return None
    return completed.stdout if completed.returncode == 0 else None


def _git_working_tree_identity(root: Path) -> Optional[Dict[str, Any]]:
    head = _git_output(root, ["rev-parse", "HEAD"])
    listed = _git_output(root, ["ls-files", "-co", "--exclude-standard", "-z"])
    status = _git_output(root, ["status", "--porcelain=v1", "-z"])
    if head is None or listed is None or status is None:
        return None
    digest = hashlib.sha256()
    file_count = 0
    for raw_name in sorted(name for name in listed.split(b"\0") if name):
        relative_name = raw_name.decode("utf-8", errors="surrogateescape")
        path = root / relative_name
        if not path.is_file():
            continue
        digest.update(raw_name)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        file_count += 1
    return {
        "git_commit": head.decode("ascii").strip(),
        "working_tree_sha256": digest.hexdigest(),
        "working_tree_file_count": file_count,
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status).hexdigest(),
    }


def _package_fingerprint(package_root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in package_root.rglob("*")
        if path.is_file() and (path.suffix == ".py" or path.name.startswith("_core."))
    )
    for path in files:
        digest.update(path.relative_to(package_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def collect_source_identity(source_root: Optional[Path] = None) -> Dict[str, Any]:
    ctboost = importlib.import_module("ctboost")
    native = importlib.import_module("ctboost._core")
    package_root = Path(ctboost.__file__).resolve().parent
    native_path = Path(native.__file__).resolve()
    root = (
        Path(__file__).resolve().parents[2]
        if source_root is None
        else Path(source_root).resolve()
    )
    versions: Dict[str, Optional[str]] = {}
    for distribution in ("ctboost", "numpy", "pandas", "scikit-learn", "openml"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    identity = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "distributions": versions,
        "ctboost_version": str(ctboost.__version__),
        "ctboost_package_sha256": _package_fingerprint(package_root),
        "native_extension_sha256": sha256_file(native_path),
        "native_build_info": dict(ctboost.build_info()),
        "git": _git_working_tree_identity(root),
    }
    identity["source_identity_sha256"] = sha256_json(identity)
    assert_no_absolute_paths(identity)
    return identity


def probe_native_feature_test_api() -> Dict[str, Any]:
    native = importlib.import_module("ctboost._core")
    explicit = native.GradientBooster(
        feature_test="quadratic",
        feature_test_bins=8,
        feature_test_adjustment="none",
    )
    candidate = native.GradientBooster(
        feature_test="grouped",
        feature_test_bins=8,
        feature_test_adjustment="none",
    )
    observed = {
        "explicit": {
            "feature_test": explicit.feature_test(),
            "feature_test_bins": int(explicit.feature_test_bins()),
            "feature_test_adjustment": explicit.feature_test_adjustment(),
        },
        "candidate": {
            "feature_test": candidate.feature_test(),
            "feature_test_bins": int(candidate.feature_test_bins()),
            "feature_test_adjustment": candidate.feature_test_adjustment(),
        },
    }
    expected = {
        "explicit": {
            "feature_test": "quadratic",
            "feature_test_bins": 8,
            "feature_test_adjustment": "none",
        },
        "candidate": {
            "feature_test": "grouped",
            "feature_test_bins": 8,
            "feature_test_adjustment": "none",
        },
    }
    if observed != expected:
        raise RuntimeError("native grouped feature-test API returned unexpected values")
    return observed


def _category_values_sha256(series: Any) -> str:
    values = [
        {
            "python_type": "{}.{}".format(
                type(value).__module__, type(value).__qualname__
            ),
            "repr": repr(value),
        }
        for value in series.cat.categories.tolist()
    ]
    return sha256_json(values)


def dataframe_schema(frame: Any) -> list:
    import pandas as pd

    schema = []
    for name, series in frame.items():
        entry: Dict[str, Any] = {
            "name": str(name),
            "dtype": str(series.dtype),
            "missing_count": int(series.isna().sum()),
            "categorical": isinstance(series.dtype, pd.CategoricalDtype),
        }
        if entry["categorical"]:
            entry["category_count"] = int(len(series.cat.categories))
            entry["ordered"] = bool(series.cat.ordered)
            entry["categories_sha256"] = _category_values_sha256(series)
        schema.append(entry)
    return schema


def _dataframe_digest(frame: Any) -> str:
    import pandas as pd

    digest = hashlib.sha256()
    digest.update(json.dumps(dataframe_schema(frame), sort_keys=True).encode("utf-8"))
    digest.update(
        json.dumps([str(value) for value in frame.index.tolist()]).encode("utf-8")
    )
    hashes = pd.util.hash_pandas_object(frame, index=True, categorize=True).to_numpy(
        dtype="uint64", copy=False
    )
    digest.update(hashes.astype("<u8", copy=False).tobytes())
    return digest.hexdigest()


def _series_digest(series: Any) -> str:
    import pandas as pd

    descriptor: Dict[str, Any] = {
        "name": None if series.name is None else str(series.name),
        "dtype": str(series.dtype),
        "missing_count": int(series.isna().sum()),
    }
    if isinstance(series.dtype, pd.CategoricalDtype):
        descriptor["category_count"] = int(len(series.cat.categories))
        descriptor["ordered"] = bool(series.cat.ordered)
        descriptor["categories_sha256"] = _category_values_sha256(series)
    digest = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode("utf-8"))
    hashes = pd.util.hash_pandas_object(series, index=True, categorize=True).to_numpy(
        dtype="uint64", copy=False
    )
    digest.update(hashes.astype("<u8", copy=False).tobytes())
    return digest.hexdigest()


def preserve_openml_frame(frame: Any, categorical_indicator: Sequence[bool]) -> Any:
    import pandas as pd

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("OpenML must return a pandas DataFrame")
    if not frame.columns.is_unique:
        raise ValueError("OpenML feature names must be unique")
    if len(categorical_indicator) != frame.shape[1]:
        raise ValueError("OpenML categorical indicator does not match feature count")
    preserved = frame.copy(deep=False)
    for index, is_categorical in enumerate(categorical_indicator):
        if not bool(is_categorical):
            continue
        series = preserved.iloc[:, index]
        if not isinstance(series.dtype, pd.CategoricalDtype):
            preserved = preserved.copy(deep=False)
            preserved[preserved.columns[index]] = series.astype("category")
    return preserved


def validate_outer_split_indices(
    train_indices: Any,
    test_indices: Any,
    *,
    row_count: int,
    context: str,
) -> tuple:
    import numpy as np

    normalized = []
    for name, values in (("train", train_indices), ("test", test_indices)):
        array = np.asarray(values)
        if array.ndim != 1:
            raise ValueError("{} {} indices must be 1D".format(context, name))
        if array.dtype.kind not in {"i", "u"}:
            raise TypeError(
                "{} {} indices must have an integer dtype".format(context, name)
            )
        if array.size == 0:
            raise ValueError("{} {} indices must not be empty".format(context, name))
        if np.any(array < 0) or np.any(array >= int(row_count)):
            raise ValueError(
                "{} {} indices are outside the row range".format(context, name)
            )
        values_int64 = np.asarray(array, dtype=np.int64)
        if np.unique(values_int64).size != values_int64.size:
            raise ValueError("{} {} indices must be unique".format(context, name))
        normalized.append(values_int64)
    train, test = normalized
    if np.intersect1d(train, test).size:
        raise ValueError("{} train and test indices overlap".format(context))
    observed = np.sort(np.concatenate([train, test]))
    expected = np.arange(int(row_count), dtype=np.int64)
    if not np.array_equal(observed, expected):
        raise ValueError(
            "{} train and test must partition every row exactly".format(context)
        )
    return train, test


def dataset_identity(payload: Mapping[str, Any]) -> Dict[str, Any]:
    import numpy as np
    import pandas as pd

    frame = payload["X"]
    target = payload["y"]
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("staged features must remain a pandas DataFrame")
    if not isinstance(target, pd.Series):
        target = pd.Series(target, name=payload["metadata"].get("target_name"))
    categorical_indicator = payload.get("categorical_indicator")
    if not isinstance(categorical_indicator, (list, tuple)):
        raise TypeError("categorical_indicator must be an ordered sequence")
    if len(categorical_indicator) != frame.shape[1]:
        raise ValueError("categorical_indicator does not match the feature count")
    if any(not isinstance(value, (bool, np.bool_)) for value in categorical_indicator):
        raise TypeError("categorical_indicator entries must be booleans")
    categorical_indicator = [bool(value) for value in categorical_indicator]
    splits = []
    for fold in FOLDS:
        split = payload["outer_splits"][fold]
        train, test = validate_outer_split_indices(
            split["train"],
            split["test"],
            row_count=int(frame.shape[0]),
            context="OpenML repeat 0 fold {} sample 0".format(fold),
        )
        splits.append(
            {
                "repeat": REPEAT,
                "fold": fold,
                "sample": 0,
                "train_count": int(train.size),
                "test_count": int(test.size),
                "train_indices_sha256": array_digest(train.astype("<i8", copy=False)),
                "test_indices_sha256": array_digest(test.astype("<i8", copy=False)),
            }
        )
    core = {
        "openml": dict(payload["metadata"]),
        "row_count": int(frame.shape[0]),
        "feature_count": int(frame.shape[1]),
        "categorical_indicator": categorical_indicator,
        "feature_schema": dataframe_schema(frame),
        "features_sha256": _dataframe_digest(frame),
        "target_sha256": _series_digest(target),
        "published_splits": splits,
        "pandas_version": str(pd.__version__),
    }
    core["data_identity_sha256"] = sha256_json(core)
    return core


def _openml_metadata(
    task: Any, dataset: Any, task_spec: Mapping[str, Any]
) -> Dict[str, Any]:
    dataset_id = getattr(dataset, "dataset_id", getattr(task, "dataset_id", None))
    return {
        "task_id": int(task_spec["task_id"]),
        "dataset_id": None if dataset_id is None else int(dataset_id),
        "dataset_version": getattr(dataset, "version", None),
        "dataset_md5_checksum": getattr(dataset, "md5_checksum", None),
        "target_name": str(task.target_name),
        "problem": str(task_spec["problem"]),
        "stress_only": bool(task_spec["stress_only"]),
        "frozen_display_name": str(task_spec["name"]),
        "identity_basis": "OpenML task ID and published repeat/fold/sample split",
        "repeat": REPEAT,
        "sample": 0,
    }


def load_openml_payload(
    task_spec: Mapping[str, Any], openml_module: Any
) -> Dict[str, Any]:
    import pandas as pd

    task_id = int(task_spec["task_id"])
    task = openml_module.tasks.get_task(task_id, download_data=True)
    if int(getattr(task, "task_id", task_id)) != task_id:
        raise RuntimeError("OpenML returned a different task ID")
    dataset = task.get_dataset()
    X, y, categorical_indicator, _ = dataset.get_data(
        target=task.target_name,
        dataset_format="dataframe",
    )
    X = preserve_openml_frame(X, categorical_indicator)
    if not isinstance(y, pd.Series):
        y = pd.Series(y, name=str(task.target_name))
    if len(X) != len(y):
        raise ValueError("OpenML features and target have different row counts")
    if y.isna().any():
        raise ValueError("OpenML task target contains missing values")
    outer_splits = {}
    for fold in FOLDS:
        train, test = task.get_train_test_split_indices(
            repeat=REPEAT, fold=fold, sample=0
        )
        train, test = validate_outer_split_indices(
            train,
            test,
            row_count=int(X.shape[0]),
            context="OpenML repeat 0 fold {} sample 0 staging".format(fold),
        )
        outer_splits[fold] = {
            "train": train,
            "test": test,
        }
    payload = {
        "metadata": _openml_metadata(task, dataset, task_spec),
        "categorical_indicator": [bool(value) for value in categorical_indicator],
        "X": X,
        "y": y,
        "outer_splits": outer_splits,
    }
    payload["data_identity"] = dataset_identity(payload)
    return payload


def _stage_manifest_path(cache_dir: Path, task_id: int) -> Path:
    return cache_dir / "staged" / "task-{}.json".format(task_id)


def _read_staged_payload(
    cache_dir: Path, task_spec: Mapping[str, Any]
) -> Optional[Dict[str, Any]]:
    manifest_path = _stage_manifest_path(cache_dir, int(task_spec["task_id"]))
    if not manifest_path.exists():
        return None
    manifest = load_json(manifest_path)
    stage_name = manifest.get("stage_file")
    if not isinstance(stage_name, str) or Path(stage_name).name != stage_name:
        raise ValueError("invalid staged OpenML cache filename")
    stage_path = manifest_path.parent / stage_name
    if not stage_path.is_file():
        raise FileNotFoundError("staged OpenML payload is missing")
    if sha256_file(stage_path) != manifest.get("stage_file_sha256"):
        raise ValueError("staged OpenML payload checksum mismatch")
    with stage_path.open("rb") as stream:
        payload = pickle.load(stream)
    identity = dataset_identity(payload)
    if identity != manifest.get("data_identity"):
        raise ValueError("staged OpenML payload identity mismatch")
    payload["data_identity"] = identity
    payload["_stage_path"] = stage_path
    payload["_stage_file_sha256"] = manifest["stage_file_sha256"]
    return payload


def stage_openml_payload(
    cache_dir: Path, task_spec: Mapping[str, Any], openml_module: Any
) -> Dict[str, Any]:
    cached = _read_staged_payload(cache_dir, task_spec)
    if cached is not None:
        return cached
    payload = load_openml_payload(task_spec, openml_module)
    digest = payload["data_identity"]["data_identity_sha256"]
    stage_path = (
        cache_dir
        / "staged"
        / "task-{}-{}.pickle".format(task_spec["task_id"], digest[:16])
    )
    serializable = dict(payload)
    serializable.pop("_stage_path", None)
    serializable.pop("_stage_file_sha256", None)
    atomic_write_pickle(stage_path, serializable)
    stage_sha256 = sha256_file(stage_path)
    manifest = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "task_id": int(task_spec["task_id"]),
        "stage_file": stage_path.name,
        "stage_file_sha256": stage_sha256,
        "data_identity": payload["data_identity"],
    }
    assert_no_absolute_paths(manifest)
    atomic_write_json(
        _stage_manifest_path(cache_dir, int(task_spec["task_id"])), manifest
    )
    payload["_stage_path"] = stage_path
    payload["_stage_file_sha256"] = stage_sha256
    return payload
