"""Single-fit worker for the frozen external panel."""

from __future__ import annotations

import ctypes
import hashlib
import math
import os
import pickle
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from ._external_panel_data import (
    atomic_write_json,
    collect_source_identity,
    dataframe_schema,
    dataset_identity,
    load_json,
    sha256_file,
    validate_outer_split_indices,
)
from ._external_panel_protocol import (
    HISTOGRAM_THREADS,
    RESULT_SCHEMA_VERSION,
    array_digest,
    assert_no_absolute_paths,
    canonical_json,
    encode_target,
    redact_absolute_paths,
    seal_record,
    sha256_json,
    validate_full_job_identity,
    validate_inner_split,
)


def peak_rss_bytes() -> int:
    if os.name == "nt":
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        get_current_process = ctypes.windll.kernel32.GetCurrentProcess
        get_current_process.argtypes = []
        get_current_process.restype = wintypes.HANDLE
        get_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory_info.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        get_memory_info.restype = wintypes.BOOL
        if not get_memory_info(
            get_current_process(), ctypes.byref(counters), counters.cb
        ):
            raise OSError("GetProcessMemoryInfo failed")
        return int(counters.PeakWorkingSetSize)
    import resource

    maximum = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return maximum if sys.platform == "darwin" else maximum * 1024


def stable_raw_prediction_digest(predictions: Any) -> Tuple[str, List[int], str]:
    import numpy as np

    original = np.asarray(predictions)
    if original.dtype.kind != "f":
        raise TypeError("raw predictions must have a floating-point dtype")
    canonical_dtype = np.dtype("<f{}".format(original.dtype.itemsize))
    values = np.ascontiguousarray(original.astype(canonical_dtype, copy=False))
    descriptor = {
        "dtype": canonical_dtype.str,
        "shape": [int(value) for value in values.shape],
    }
    digest = hashlib.sha256()
    digest.update(canonical_json(descriptor))
    digest.update(b"\0")
    digest.update(values.tobytes(order="C"))
    return (
        digest.hexdigest(),
        descriptor["shape"],
        descriptor["dtype"],
    )


def canonical_tree_digest(booster: Any) -> str:
    state = booster._handle.export_state()
    return hashlib.sha256(canonical_json(state["trees"])).hexdigest()


def prediction_metrics(
    problem: str,
    raw_predictions: Any,
    target: Any,
    outer_train_target: Any,
    class_count: Optional[int],
) -> Dict[str, Any]:
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        log_loss,
        mean_squared_error,
        roc_auc_score,
    )

    raw = np.asarray(raw_predictions, dtype=np.float64)
    y_true = np.asarray(target)
    if not np.isfinite(raw).all():
        raise ValueError("model produced non-finite raw predictions")
    if problem == "binary":
        if raw.shape != (y_true.shape[0],):
            raise ValueError("binary raw prediction shape is incorrect")
        probabilities = np.empty_like(raw)
        nonnegative = raw >= 0.0
        probabilities[nonnegative] = 1.0 / (1.0 + np.exp(-raw[nonnegative]))
        exp_values = np.exp(raw[~nonnegative])
        probabilities[~nonnegative] = exp_values / (1.0 + exp_values)
        auc = float(roc_auc_score(y_true, probabilities))
        diagnostic = float(
            log_loss(
                y_true,
                np.column_stack([1.0 - probabilities, probabilities]),
                labels=[0, 1],
            )
        )
        return {
            "primary_name": "one_minus_roc_auc",
            "primary_loss": 1.0 - auc,
            "roc_auc": auc,
            "log_loss": diagnostic,
        }
    if problem == "multiclass":
        if class_count is None or raw.shape != (y_true.shape[0], int(class_count)):
            raise ValueError("multiclass raw prediction shape is incorrect")
        shifted = raw - np.max(raw, axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
        loss = float(
            log_loss(y_true, probabilities, labels=list(range(int(class_count))))
        )
        accuracy = float(accuracy_score(y_true, np.argmax(probabilities, axis=1)))
        return {
            "primary_name": "log_loss",
            "primary_loss": loss,
            "log_loss": loss,
            "accuracy": accuracy,
        }
    if raw.shape != (y_true.shape[0],):
        raise ValueError("regression raw prediction shape is incorrect")
    rmse = float(math.sqrt(mean_squared_error(y_true, raw)))
    scale = float(np.std(np.asarray(outer_train_target, dtype=np.float64), ddof=0))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("outer training target standard deviation is not positive")
    return {
        "primary_name": "normalized_rmse",
        "primary_loss": rmse / scale,
        "rmse": rmse,
        "outer_train_target_std_ddof0": scale,
    }


def _load_worker_payload(job: Mapping[str, Any]) -> Mapping[str, Any]:
    stage_path = Path(str(job["stage_file"]))
    if not stage_path.is_file():
        raise FileNotFoundError("staged OpenML payload is missing")
    if sha256_file(stage_path) != job["stage_file_sha256"]:
        raise ValueError("staged OpenML payload checksum mismatch in worker")
    with stage_path.open("rb") as stream:
        payload = pickle.load(stream)
    if dataset_identity(payload) != job["data_identity"]:
        raise ValueError("worker observed a different OpenML data identity")
    return payload


def _assert_slice_dtypes_preserved(original: Any, subsets: Tuple[Any, ...]) -> None:
    original_schema = dataframe_schema(original)
    for subset in subsets:
        expected = [
            {**entry, "missing_count": int(subset.iloc[:, index].isna().sum())}
            for index, entry in enumerate(original_schema)
        ]
        if dataframe_schema(subset) != expected:
            raise RuntimeError("pandas feature dtypes changed while slicing a fold")


def execute_worker_job(
    job: Mapping[str, Any],
    work_dir: Path,
    *,
    source_collector: Callable[[], Dict[str, Any]] = collect_source_identity,
    ctboost_module: Any = None,
) -> Dict[str, Any]:
    """Execute one fit; the caller must already be an isolated subprocess."""
    import numpy as np

    if os.environ.get("CTBOOST_HIST_THREADS") != str(HISTOGRAM_THREADS):
        raise RuntimeError("worker CTBOOST_HIST_THREADS is not the frozen value 8")
    validate_full_job_identity(job)
    observed_source = source_collector()
    if observed_source != job["source_identity"]:
        raise RuntimeError(
            "worker source/native identity differs from the scheduled job"
        )
    payload = _load_worker_payload(job)
    config = job["config"]
    problem = str(config["problem"])
    fold = int(config["fold"])
    outer_split = payload["outer_splits"][fold]
    outer_train, outer_test = validate_outer_split_indices(
        outer_split["train"],
        outer_split["test"],
        row_count=int(payload["X"].shape[0]),
        context="worker repeat 0 fold {} sample 0".format(fold),
    )
    inner_train = np.asarray(job["inner_train_indices"], dtype=np.int64)
    inner_valid = np.asarray(job["inner_validation_indices"], dtype=np.int64)
    validate_inner_split(outer_train, outer_test, inner_train, inner_valid)
    if (
        array_digest(inner_train.astype("<i8", copy=False))
        != config["inner_train_indices_sha256"]
    ):
        raise RuntimeError("worker inner-training split identity mismatch")
    if (
        array_digest(inner_valid.astype("<i8", copy=False))
        != config["inner_validation_indices_sha256"]
    ):
        raise RuntimeError("worker inner-validation split identity mismatch")

    X = payload["X"]
    encoded_target, class_labels = encode_target(payload["y"], problem)
    if class_labels != job.get("class_labels"):
        raise RuntimeError("worker classification label identity mismatch")
    class_count = None if class_labels is None else len(class_labels)
    train_frame = X.iloc[inner_train]
    valid_frame = X.iloc[inner_valid]
    test_frame = X.iloc[outer_test]
    _assert_slice_dtypes_preserved(X, (train_frame, valid_frame, test_frame))

    if ctboost_module is None:
        import ctboost as ctboost_module

    categorical_features = [
        index
        for index, value in enumerate(payload["categorical_indicator"])
        if bool(value)
    ]
    train_pool = ctboost_module.Pool(
        train_frame, encoded_target[inner_train], cat_features=categorical_features
    )
    valid_pool = ctboost_module.Pool(
        valid_frame, encoded_target[inner_valid], cat_features=categorical_features
    )
    test_pool = ctboost_module.Pool(
        test_frame, encoded_target[outer_test], cat_features=categorical_features
    )
    params = dict(config["params"])
    iterations = int(params.pop("iterations", config["iterations"]))
    fit_started = time.perf_counter()
    booster = ctboost_module.train(
        train_pool,
        params,
        num_boost_round=iterations,
        eval_set=valid_pool,
        early_stopping_rounds=int(config["early_stopping_rounds"]),
    )
    fit_seconds = time.perf_counter() - fit_started

    raw_predictions = booster.predict(test_pool)
    prediction_sha, prediction_shape, prediction_dtype = stable_raw_prediction_digest(
        raw_predictions
    )
    metrics = prediction_metrics(
        problem,
        raw_predictions,
        encoded_target[outer_test],
        encoded_target[outer_train],
        class_count,
    )
    numeric_metrics = [value for key, value in metrics.items() if key != "primary_name"]
    if not all(math.isfinite(float(value)) for value in numeric_metrics):
        raise ValueError("model produced non-finite metrics")
    model_path = work_dir / "model.ctboost"
    booster.save_model(model_path)
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "job_key": str(job["job_key"]),
        "status": "success",
        "config": dict(config),
        "source_identity_sha256": observed_source["source_identity_sha256"],
        "data_identity_sha256": job["data_identity"]["data_identity_sha256"],
        "config_identity_sha256": sha256_json(config),
        "fit_seconds": float(fit_seconds),
        "peak_process_rss_bytes": peak_rss_bytes(),
        "peak_process_rss_source": "worker_process",
        "serialized_model_bytes": int(model_path.stat().st_size),
        "serialized_model_sha256": sha256_file(model_path),
        "best_iteration": int(booster.best_iteration),
        "iterations_trained": int(booster.num_iterations_trained),
        "raw_prediction_sha256": prediction_sha,
        "raw_prediction_shape": prediction_shape,
        "raw_prediction_dtype": prediction_dtype,
        "canonical_tree_sha256": canonical_tree_digest(booster),
        "metrics": metrics,
        "finite_predictions": True,
        "correct_prediction_shape": True,
        "categorical_feature_count": len(categorical_features),
        "feature_missing_value_count": int(X.isna().sum().sum()),
        "histogram_threads": HISTOGRAM_THREADS,
        "fit_timer_excludes_openml_cache_and_pool_construction": True,
    }
    model_path.unlink()
    result = seal_record(result, "result_sha256")
    assert_no_absolute_paths(result)
    return result


def failure_result(
    job: Mapping[str, Any],
    exc: BaseException,
    *,
    record_process_rss: bool = True,
) -> Dict[str, Any]:
    message = redact_absolute_paths(str(exc))
    try:
        peak_rss = peak_rss_bytes() if record_process_rss else None
        peak_rss_error = (
            None
            if record_process_rss
            else "child process RSS unavailable because no worker record was produced"
        )
    except Exception as rss_exc:  # preserve the original worker failure record
        peak_rss = None
        peak_rss_error = redact_absolute_paths(
            "{}: {}".format(type(rss_exc).__name__, rss_exc)
        )
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "job_key": str(job.get("job_key", "unknown")),
        "status": "failure",
        "config": dict(job.get("config", {})),
        "source_identity_sha256": job.get("source_identity", {}).get(
            "source_identity_sha256"
        ),
        "data_identity_sha256": job.get("data_identity", {}).get(
            "data_identity_sha256"
        ),
        "config_identity_sha256": sha256_json(job.get("config", {})),
        "error_type": type(exc).__name__,
        "error_message": message,
        "peak_process_rss_bytes": peak_rss,
        "peak_process_rss_source": (
            "worker_process" if peak_rss is not None else "unavailable"
        ),
        "peak_process_rss_error": peak_rss_error,
    }
    result = seal_record(result, "result_sha256")
    assert_no_absolute_paths(result)
    return result


def worker_main(job_path: Path, output_path: Path) -> int:
    job: Dict[str, Any] = {}
    try:
        job = load_json(job_path)
        with tempfile.TemporaryDirectory(prefix="ctboost-external-fit-") as temporary:
            result = execute_worker_job(job, Path(temporary))
    except BaseException as exc:
        result = failure_result(job, exc)
        atomic_write_json(output_path, result)
        traceback.print_exc()
        return 1
    atomic_write_json(output_path, result)
    return 0
