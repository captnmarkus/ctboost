"""Validate, sanitize, and summarize the complete 306-artifact scout."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import ntpath
import re
import shutil
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .constants import (
    EXPECTED_ARTIFACTS,
    EXPECTED_CHILD_FITS,
    MEMORY_LIMIT_GB,
    NUM_CONFIGS,
    NUM_CPUS,
    NUM_GPUS,
    TASKS,
    TIME_LIMIT_SECONDS,
    TREATMENTS,
    config_id,
    experiment_name,
)
from .models import base_p50, canonical_json_bytes, paired_configs

_FILE_URI = re.compile(r"(?i)(?<![A-Za-z0-9+.-])file://")
_WINDOWS_ABSOLUTE_PATH = re.compile(r"(?i)(?<![A-Z0-9])[A-Z]:[\\/]")
_WINDOWS_ROOTED_PATH = re.compile(r"(?<![A-Za-z0-9._~%-])\\(?!\\)")
_UNC_ABSOLUTE_PATH = re.compile(r"(?:\\\\|(?<!:)//)[^\\/\s]+[\\/][^\\/\s]+")
_POSIX_ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z0-9._~%+\-/])/(?!/)")

_PROVENANCE_REPORT = "scout_provenance.json"
_RUN_MANIFEST_REPORT = "run_manifest.shard-00000-of-00001.json"
_SUCCESS_REPORTS = frozenset(
    {
        "sanitized/scout_summary.json",
        "sanitized/paired_configs.json",
        "sanitized/config_results.csv",
        "sanitized/endpoint_results.csv",
    }
)
_FAILURE_REPORTS = frozenset({"sanitized/scout_failure.json"})


def _expected_raw_paths() -> frozenset[str]:
    return frozenset(
        (
            Path("data")
            / experiment_name(treatment, index)
            / str(task_id)
            / "0_0"
            / "results.pkl"
        ).as_posix()
        for task_id, _dataset, _problem, _metric in TASKS
        for treatment in ("quadratic", "grouped")
        for index in range(NUM_CONFIGS)
    )


def _is_link_or_junction(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(callable(is_junction) and is_junction())


def _namespace_files(root: Path) -> tuple[set[str], set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()
    linked_or_special: set[str] = set()
    if _is_link_or_junction(root):
        return files, directories, {"."}
    if not root.exists():
        return files, directories, linked_or_special
    if not root.is_dir():
        return files, directories, {"."}
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if _is_link_or_junction(path):
            linked_or_special.add(relative)
        elif path.is_file():
            files.add(relative)
            if path.stat().st_nlink != 1:
                linked_or_special.add(relative)
        elif path.is_dir():
            directories.add(relative)
        else:
            linked_or_special.add(relative)
    return files, directories, linked_or_special


def _parent_directories(relative_paths: Iterable[str]) -> set[str]:
    directories: set[str] = set()
    for relative in relative_paths:
        parent = Path(relative).parent
        while parent != Path("."):
            directories.add(parent.as_posix())
            parent = parent.parent
    return directories


def namespace_inventory(
    *, raw_dir: Path, report_dir: Path, phase: str
) -> dict[str, Any]:
    expected_raw = _expected_raw_paths()
    raw_files, raw_directories, invalid_raw = _namespace_files(raw_dir)
    report_files, report_directories, invalid_report = _namespace_files(report_dir)
    base_reports = {_PROVENANCE_REPORT, _RUN_MANIFEST_REPORT}

    if phase == "run_input":
        allowed_reports = base_reports
        required_reports = {_PROVENANCE_REPORT}
        required_raw: set[str] = set()
    elif phase in {"run_output", "summarize_input"}:
        allowed_reports = base_reports
        required_reports = base_reports
        required_raw = set(expected_raw)
    elif phase == "summary_success":
        allowed_reports = base_reports | set(_SUCCESS_REPORTS)
        required_reports = set(allowed_reports)
        required_raw = set(expected_raw)
    elif phase == "summary_failure":
        allowed_reports = base_reports | set(_FAILURE_REPORTS)
        required_reports = set(allowed_reports)
        required_raw = set(expected_raw)
    elif phase == "preflight":
        allowed_reports = set(base_reports)
        required_reports = {_PROVENANCE_REPORT} if raw_files or report_files else set()
        sanitized = report_files - base_reports
        if sanitized == set(_SUCCESS_REPORTS):
            allowed_reports.update(_SUCCESS_REPORTS)
        elif sanitized == set(_FAILURE_REPORTS):
            allowed_reports.update(_FAILURE_REPORTS)
        elif sanitized:
            invalid_report.update(sanitized)
        required_raw = set(expected_raw) if sanitized else set()
        if sanitized:
            required_reports.add(_RUN_MANIFEST_REPORT)
    else:
        raise ValueError(f"unknown scout namespace phase: {phase}")

    allowed_raw_directories = _parent_directories(expected_raw)
    allowed_report_directories = _parent_directories(allowed_reports)
    stale = {
        *(f"raw/{path}" for path in raw_files - expected_raw),
        *(f"raw/{path}" for path in raw_directories - allowed_raw_directories),
        *(f"raw/{path}" for path in invalid_raw),
        *(f"report/{path}" for path in report_files - allowed_reports),
        *(f"report/{path}" for path in report_directories - allowed_report_directories),
        *(f"report/{path}" for path in invalid_report),
    }
    missing = {
        *(f"raw/{path}" for path in required_raw - raw_files),
        *(f"report/{path}" for path in required_reports - report_files),
    }
    observed_expected = len(raw_files & expected_raw)
    return {
        "phase": phase,
        "expected_outer_artifacts": len(expected_raw),
        "observed_outer_artifacts": observed_expected,
        "raw_file_count": len(raw_files),
        "raw_directory_count": len(raw_directories),
        "report_file_count": len(report_files),
        "report_directory_count": len(report_directories),
        "stale_or_unexpected": len(stale),
        "missing_required": len(missing),
        "complete": not stale and not missing,
    }


def validate_namespace(
    *, raw_dir: Path, report_dir: Path, phase: str
) -> dict[str, Any]:
    inventory = namespace_inventory(raw_dir=raw_dir, report_dir=report_dir, phase=phase)
    if inventory["stale_or_unexpected"]:
        raise RuntimeError(
            "sealed scout namespace contains stale, unexpected, or linked files"
        )
    if inventory["missing_required"]:
        raise RuntimeError("sealed scout namespace is missing required phase outputs")
    return inventory


@dataclass(frozen=True)
class ArtifactRecord:
    treatment: str
    config_index: int
    config_id: str
    method: str
    task_id: int
    dataset: str
    problem_type: str
    metric: str
    metric_error: float
    metric_error_val: float
    time_train_s: float
    time_infer_s: float
    peak_rss_bytes: int
    incremental_peak_rss_bytes: int
    model_bytes: int
    artifact_sha256: str

    def csv_row(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _finite_float(value: Any, field: str) -> float:
    resolved = float(value)
    if not math.isfinite(resolved):
        raise RuntimeError(f"artifact has non-finite {field}")
    return resolved


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"artifact has invalid {field}")
    resolved = int(value)
    if resolved < 0 or float(value) != resolved:
        raise RuntimeError(f"artifact has invalid {field}")
    return resolved


def _metric_error(
    problem_type: str, y_true: np.ndarray, prediction: np.ndarray
) -> float:
    from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score

    if problem_type == "multiclass":
        return float(
            log_loss(y_true, prediction, labels=list(range(prediction.shape[1])))
        )
    if problem_type == "binary":
        return float(1.0 - roc_auc_score(y_true, prediction))
    if problem_type == "regression":
        return float(math.sqrt(mean_squared_error(y_true, prediction)))
    raise RuntimeError(f"unexpected problem type: {problem_type}")


def _validate_predictions(result: dict[str, Any]) -> None:
    artifacts = dict(result.get("simulation_artifacts", {}))
    problem_type = str(result.get("problem_type"))
    for split, reported_field in (
        ("val", "metric_error_val"),
        ("test", "metric_error"),
    ):
        y_true = np.asarray(artifacts.get(f"y_{split}"))
        prediction = np.asarray(artifacts.get(f"pred_{split}"))
        if y_true.ndim != 1 or y_true.size == 0:
            raise RuntimeError(f"artifact has invalid {split} target shape")
        if problem_type == "multiclass":
            classes = int(artifacts.get("num_classes", 0))
            if (
                prediction.ndim != 2
                or prediction.shape != (y_true.size, classes)
                or classes < 2
            ):
                raise RuntimeError(
                    f"artifact has invalid multiclass {split} prediction shape"
                )
            if np.any(prediction < 0.0) or np.any(prediction > 1.0):
                raise RuntimeError(
                    f"artifact has out-of-range multiclass {split} probabilities"
                )
            if not np.allclose(prediction.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6):
                raise RuntimeError(
                    f"artifact has unnormalized multiclass {split} probabilities"
                )
        else:
            if prediction.ndim != 1 or prediction.shape != y_true.shape:
                raise RuntimeError(
                    f"artifact has invalid {problem_type} {split} prediction shape"
                )
            if problem_type == "binary" and (
                np.any(prediction < 0.0) or np.any(prediction > 1.0)
            ):
                raise RuntimeError(
                    f"artifact has out-of-range binary {split} probabilities"
                )
        if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(prediction)):
            raise RuntimeError(f"artifact has non-finite {split} values")
        recomputed = _metric_error(problem_type, y_true, prediction)
        reported = _finite_float(result.get(reported_field), reported_field)
        if not math.isclose(recomputed, reported, rel_tol=2e-6, abs_tol=1e-10):
            raise RuntimeError(
                f"artifact {reported_field} does not match its predictions"
            )


def _strip_runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in config.items()
        if key not in {"ag_args", "ag_args_ensemble"} and not key.startswith("ag.")
    }


def _validate_method_metadata(
    result: dict[str, Any],
    *,
    treatment: str,
    index: int,
    expected_config: dict[str, Any],
) -> tuple[int, int, int]:
    identity = TREATMENTS[treatment]
    metadata = dict(result.get("method_metadata", {}))
    if metadata.get("model_cls") != identity["model_class"]:
        raise RuntimeError("artifact model class identity drift")
    if metadata.get("model_type") != identity["ag_key"]:
        raise RuntimeError("artifact model key identity drift")
    if metadata.get("name_prefix") != identity["ag_name"]:
        raise RuntimeError("artifact model name identity drift")
    for key, expected in (
        ("num_cpus", NUM_CPUS),
        ("num_cpus_child", NUM_CPUS),
        ("num_gpus", NUM_GPUS),
        ("num_gpus_child", NUM_GPUS),
    ):
        if int(metadata.get(key, -1)) != expected:
            raise RuntimeError(f"artifact resource field {key} drift")

    config = dict(metadata.get("model_hyperparameters", {}))
    if _strip_runtime_config(config) != expected_config:
        raise RuntimeError("artifact effective treatment configuration drift")
    expected_suffix = f"_{config_id(index)}_default"
    if config.get("ag_args") != {"name_suffix": expected_suffix}:
        raise RuntimeError("artifact configuration suffix drift")
    ensemble_args = dict(config.get("ag_args_ensemble", {}))
    if int(ensemble_args.get("model_random_seed", -1)) != index * 8:
        raise RuntimeError("artifact config-wise seed drift")
    if ensemble_args.get("vary_seed_across_folds") is not True:
        raise RuntimeError("artifact fold-wise seed variation drift")
    if int(ensemble_args.get("ag.max_time_limit", -1)) != TIME_LIMIT_SECONDS:
        raise RuntimeError("artifact fit-time limit drift")
    if ensemble_args.get("fold_fitting_strategy") != "sequential_local":
        raise RuntimeError("artifact used a Ray/parallel fold-fitting strategy")
    allowed_ensemble_args = {
        "model_random_seed",
        "vary_seed_across_folds",
        "ag.max_time_limit",
        "fold_fitting_strategy",
    }
    if set(ensemble_args) != allowed_ensemble_args:
        raise RuntimeError("artifact has unexpected seed/ensemble arguments")

    fit_kwargs = dict(metadata.get("fit_kwargs_extra", {}))
    for key, expected in (
        ("num_cpus", NUM_CPUS),
        ("num_gpus", NUM_GPUS),
        ("memory_limit", MEMORY_LIMIT_GB),
        ("num_bag_folds", 8),
        ("num_bag_sets", 1),
    ):
        if int(fit_kwargs.get(key, -1)) != expected:
            raise RuntimeError(f"artifact fit argument {key} drift")
    if fit_kwargs.get("raise_on_model_failure") is not True:
        raise RuntimeError("artifact did not fail closed on child-model failure")

    info = dict(metadata.get("info", {}))
    bagged = dict(info.get("bagged_info", {}))
    if int(bagged.get("num_child_models", -1)) != 8:
        raise RuntimeError("artifact does not contain eight bagged child models")
    children = dict(info.get("children_info", {}))
    expected_children = [f"S1F{fold}" for fold in range(1, 9)]
    if list(children) != expected_children:
        raise RuntimeError("artifact child-fold identities drift")
    for offset, child_name in enumerate(expected_children):
        child = dict(children[child_name])
        child_params = dict(child.get("hyperparameters", {}))
        if int(child_params.get("random_seed", -1)) != index * 8 + offset:
            raise RuntimeError("artifact child-fold seed drift")
        for field in ("feature_test", "feature_test_bins", "feature_test_adjustment"):
            if child_params.get(field) != expected_config[field]:
                raise RuntimeError(f"artifact child treatment field {field} drift")
        if (
            int(child.get("num_cpus", -1)) != NUM_CPUS
            or int(child.get("num_gpus", -1)) != 0
        ):
            raise RuntimeError("artifact child resource contract drift")

    model_bytes = _nonnegative_int(metadata.get("disk_usage"), "model bytes")
    memory = dict(result.get("memory_usage", {}))
    if bool(memory.get("gpu_tracking_enabled", False)):
        raise RuntimeError("CPU scout unexpectedly enabled GPU tracking")
    for field in (
        "peak_mem_gpu",
        "min_mem_gpu",
        "peak_mem_gpu_reserved",
        "min_mem_gpu_reserved",
    ):
        if memory.get(field) not in (None, 0):
            raise RuntimeError("CPU scout recorded nonzero GPU memory")
    peak = _nonnegative_int(memory.get("peak_mem_cpu"), "absolute peak RSS")
    minimum = _nonnegative_int(memory.get("min_mem_cpu"), "minimum RSS")
    if peak < minimum:
        raise RuntimeError("artifact absolute peak RSS is below its baseline RSS")
    return peak, peak - minimum, model_bytes


def _artifact_set_sha256(paths: Iterable[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_and_validate_artifacts(
    raw_dir: Path,
) -> tuple[list[Any], list[ArtifactRecord], str, dict[str, Any]]:
    from tabarena.benchmark.result import raw_loading

    raw_dir = raw_dir.resolve()
    expected_paths = _expected_raw_paths()
    raw_files, invalid_raw = _namespace_files(raw_dir)
    stale = (raw_files - expected_paths) | invalid_raw
    missing = expected_paths - raw_files
    raw_inventory = {
        "expected_outer_artifacts": len(expected_paths),
        "observed_outer_artifacts": len(raw_files & expected_paths),
        "stale_or_unexpected": len(stale),
        "missing_required": len(missing),
        "complete": not stale and not missing,
    }
    if stale:
        raise RuntimeError("raw namespace contains stale, unexpected, or linked files")
    if missing:
        raise RuntimeError(
            "coverage mismatch: expected "
            f"{EXPECTED_ARTIFACTS} artifacts, observed "
            f"{raw_inventory['observed_outer_artifacts']}"
        )
    paths = [raw_dir / Path(relative) for relative in sorted(expected_paths)]
    if any(path.is_symlink() or path.stat().st_nlink != 1 for path in paths):
        raise RuntimeError("raw namespace contains a linked artifact")

    expected_configs = paired_configs()
    task_by_id = {
        task_id: (dataset, problem, metric)
        for task_id, dataset, problem, metric in TASKS
    }
    method_lookup = {
        experiment_name(treatment, index): (treatment, index)
        for treatment in ("quadratic", "grouped")
        for index in range(NUM_CONFIGS)
    }
    seen: set[tuple[str, int]] = set()
    loaded: list[Any] = []
    records: list[ArtifactRecord] = []
    for path in paths:
        relative = path.relative_to(raw_dir)
        parts = relative.parts
        if len(parts) != 5 or parts[0] != "data" or parts[4] != "results.pkl":
            raise RuntimeError("raw artifact has an unexpected cache path")
        method, task_text, split = parts[1], parts[2], parts[3]
        if method not in method_lookup or split != "0_0" or not task_text.isdigit():
            raise RuntimeError(
                "raw artifact cache identity is outside the frozen schedule"
            )
        task_id = int(task_text)
        if task_id not in task_by_id:
            raise RuntimeError("raw artifact has an unexpected task id")
        identity_key = (method, task_id)
        if identity_key in seen:
            raise RuntimeError("raw namespace contains a duplicate artifact identity")
        seen.add(identity_key)

        artifact = raw_loading.load_and_align(path)
        result = dict(artifact.result)
        task = dict(result.get("task_metadata", {}))
        dataset, problem_type, metric = task_by_id[task_id]
        if str(result.get("framework")) != method:
            raise RuntimeError("artifact framework differs from its cache path")
        if (
            int(task.get("tid", -1)) != task_id
            or str(task.get("name")) != dataset
            or int(task.get("repeat", -1)) != 0
            or int(task.get("fold", -1)) != 0
            or int(task.get("sample", -1)) != 0
        ):
            raise RuntimeError("artifact task metadata differs from its cache identity")
        if (
            str(result.get("problem_type")) != problem_type
            or str(result.get("metric")) != metric
        ):
            raise RuntimeError("artifact task metric/problem type drift")
        _validate_predictions(result)

        treatment, index = method_lookup[method]
        peak, incremental, model_bytes = _validate_method_metadata(
            result,
            treatment=treatment,
            index=index,
            expected_config=expected_configs[treatment][index],
        )
        time_train = _finite_float(result.get("time_train_s"), "training time")
        time_infer = _finite_float(result.get("time_infer_s"), "inference time")
        if time_train < 0.0 or time_infer < 0.0:
            raise RuntimeError("artifact contains a negative resource time")
        records.append(
            ArtifactRecord(
                treatment=treatment,
                config_index=index,
                config_id=config_id(index),
                method=method,
                task_id=task_id,
                dataset=dataset,
                problem_type=problem_type,
                metric=metric,
                metric_error=_finite_float(result.get("metric_error"), "metric_error"),
                metric_error_val=_finite_float(
                    result.get("metric_error_val"), "metric_error_val"
                ),
                time_train_s=time_train,
                time_infer_s=time_infer,
                peak_rss_bytes=peak,
                incremental_peak_rss_bytes=incremental,
                model_bytes=model_bytes,
                artifact_sha256=_sha256_file(path),
            )
        )
        loaded.append(artifact)

    if len(seen) != EXPECTED_ARTIFACTS:
        raise RuntimeError(
            "raw namespace does not contain every frozen schedule identity"
        )
    records.sort(key=lambda row: (row.dataset, row.config_index, row.treatment))
    return loaded, records, _artifact_set_sha256(paths, raw_dir), raw_inventory


def _endpoint_rows(
    loaded: list[Any],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], list[str]]]:
    from tabarena.contexts import TabArenaContext
    from tabarena.end_to_end import EndToEnd

    context = TabArenaContext()
    processed = EndToEnd.from_raw(
        results_lst=loaded,
        task_metadata=context.task_metadata_collection,
        cache=False,
        backend="native",
        verbose=False,
    )
    frame = processed.get_results(new_result_prefix=None)
    endpoint_names = {
        "default": "default",
        "tuned": "tuned",
        "tuned_ensemble": "tuned + ensemble",
    }
    expected_method_to_treatment = {
        identity["ag_key"]: treatment for treatment, identity in TREATMENTS.items()
    }
    rows: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        subtype = str(record.get("method_subtype"))
        if subtype not in endpoint_names:
            raise RuntimeError(f"unexpected TabArena endpoint subtype: {subtype}")
        config_type = str(record.get("config_type"))
        if config_type not in expected_method_to_treatment:
            raise RuntimeError(
                f"unexpected TabArena endpoint config type: {config_type}"
            )
        rows.append(
            {
                "treatment": expected_method_to_treatment[config_type],
                "dataset": str(record["dataset"]),
                "fold": int(record["fold"]),
                "endpoint": endpoint_names[subtype],
                "metric_error": _finite_float(
                    record["metric_error"], "endpoint metric_error"
                ),
                "metric_error_val": _finite_float(
                    record["metric_error_val"], "endpoint metric_error_val"
                ),
                "time_train_s": _finite_float(
                    record["time_train_s"], "endpoint training time"
                ),
                "time_infer_s": _finite_float(
                    record["time_infer_s"], "endpoint inference time"
                ),
                "selected_config_ids": [],
            }
        )
    expected_rows = 2 * 3 * len(TASKS)
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"expected {expected_rows} endpoint summaries, observed {len(rows)}"
        )
    keys = [(row["treatment"], row["dataset"], row["endpoint"]) for row in rows]
    if len(set(keys)) != expected_rows or any(row["fold"] != 0 for row in rows):
        raise RuntimeError("TabArena endpoint summaries are missing or duplicated")

    selected_configs: dict[tuple[str, str, str], list[str]] = {}
    for method_results in processed.method_results_lst:
        treatment = expected_method_to_treatment.get(
            method_results.method_metadata.model_key
        )
        if treatment is None or method_results.repo is None:
            raise RuntimeError("processed TabArena method identity/repository drift")
        configs = method_results.repo.configs()
        if len(configs) != NUM_CONFIGS:
            raise RuntimeError(
                "processed TabArena repository does not contain all 51 configs"
            )
        name_to_id = {
            experiment_name(treatment, index): config_id(index)
            for index in range(NUM_CONFIGS)
        }
        if set(configs) != set(name_to_id):
            raise RuntimeError("processed TabArena repository config names drifted")
        for endpoint, ensemble_size in (("tuned", 1), ("tuned + ensemble", 40)):
            _ensemble_frame, weights = method_results.repo.evaluate_ensembles(
                configs=configs,
                ensemble_size=ensemble_size,
                fit_order="original",
                seed=0,
                backend="native",
            )
            for index, weight_row in weights.iterrows():
                dataset = str(index[0] if isinstance(index, tuple) else index)
                selected = [
                    name_to_id[name]
                    for name, weight in weight_row.items()
                    if float(weight) != 0.0
                ]
                selected_configs[(treatment, dataset, endpoint)] = selected
    return rows, selected_configs


def _decorate_endpoint_selections(
    endpoint_rows: list[dict[str, Any]],
    records: list[ArtifactRecord],
    selected_configs: dict[tuple[str, str, str], list[str]],
) -> None:
    by_task: dict[tuple[str, str], list[ArtifactRecord]] = {}
    for record in records:
        by_task.setdefault((record.treatment, record.dataset), []).append(record)
    endpoint_lookup = {
        (row["treatment"], row["dataset"], row["endpoint"]): row
        for row in endpoint_rows
    }
    for key, task_records in by_task.items():
        ordered = sorted(task_records, key=lambda row: row.config_index)
        default = ordered[0]
        tuned_ids = selected_configs[(*key, "tuned")]
        if len(tuned_ids) != 1:
            raise RuntimeError(
                "TabArena tuned endpoint did not select exactly one config"
            )
        tuned = next(record for record in ordered if record.config_id == tuned_ids[0])
        best_validation_error = min(record.metric_error_val for record in ordered)
        if not math.isclose(
            tuned.metric_error_val,
            best_validation_error,
            rel_tol=2e-6,
            abs_tol=1e-10,
        ):
            raise RuntimeError(
                "TabArena tuned endpoint did not select a validation-optimal config"
            )
        endpoint_lookup[(*key, "default")]["selected_config_ids"] = [default.config_id]
        endpoint_lookup[(*key, "tuned")]["selected_config_ids"] = tuned_ids
        endpoint_lookup[(*key, "tuned + ensemble")]["selected_config_ids"] = (
            selected_configs[(*key, "tuned + ensemble")]
        )
        if not math.isclose(
            endpoint_lookup[(*key, "default")]["metric_error"],
            default.metric_error,
            rel_tol=2e-6,
            abs_tol=1e-10,
        ):
            raise RuntimeError("TabArena default endpoint does not match c1")
        if not math.isclose(
            endpoint_lookup[(*key, "tuned")]["metric_error"],
            tuned.metric_error,
            rel_tol=2e-6,
            abs_tol=1e-10,
        ):
            raise RuntimeError(
                "TabArena tuned endpoint does not match validation selection"
            )
    endpoint_rows.sort(
        key=lambda row: (row["dataset"], row["endpoint"], row["treatment"])
    )


def _relative_improvement(quadratic: float, grouped: float) -> float:
    if quadratic == 0.0:
        if grouped == 0.0:
            return 0.0
        raise RuntimeError(
            "relative improvement is undefined because quadratic error is zero"
        )
    return (quadratic - grouped) / quadratic


def _comparison_outcome(improvement: float) -> str:
    if abs(improvement) < 0.001:
        return "tie"
    return "win" if improvement > 0.0 else "loss"


def _evaluate_decision_gates(
    *,
    primary_wins: int,
    primary_macro_median: float,
    primary_worst: float,
    tuned_macro_median: float,
    tuned_worst: float,
    median_paired_training_ratio: float,
) -> dict[str, bool]:
    """Apply only the frozen decision boundaries from the tracked protocol."""
    return {
        "integration_complete": True,
        "primary_wins_at_least_2": primary_wins >= 2,
        "primary_macro_median_at_least_0_0025": primary_macro_median >= 0.0025,
        "primary_no_task_worse_than_0_01": primary_worst >= -0.01,
        "tuned_median_nonnegative": tuned_macro_median >= 0.0,
        "tuned_no_task_worse_than_0_02": tuned_worst >= -0.02,
        "median_paired_training_ratio_at_most_1_15": median_paired_training_ratio
        <= 1.15,
    }


def _decision_summary(
    endpoint_rows: list[dict[str, Any]], records: list[ArtifactRecord]
) -> dict[str, Any]:
    endpoint_lookup = {
        (row["dataset"], row["endpoint"], row["treatment"]): row
        for row in endpoint_rows
    }
    comparisons = []
    for _task_id, dataset, _problem, _metric in TASKS:
        for endpoint in ("default", "tuned", "tuned + ensemble"):
            quadratic = endpoint_lookup[(dataset, endpoint, "quadratic")][
                "metric_error"
            ]
            grouped = endpoint_lookup[(dataset, endpoint, "grouped")]["metric_error"]
            improvement = _relative_improvement(quadratic, grouped)
            outcome = _comparison_outcome(improvement)
            comparisons.append(
                {
                    "dataset": dataset,
                    "endpoint": endpoint,
                    "quadratic_error": quadratic,
                    "grouped_error": grouped,
                    "relative_improvement": improvement,
                    "outcome": outcome,
                }
            )

    primary = [row for row in comparisons if row["endpoint"] == "tuned + ensemble"]
    tuned = [row for row in comparisons if row["endpoint"] == "tuned"]
    paired_ratios = []
    for dataset in (task[1] for task in TASKS):
        for index in range(NUM_CONFIGS):
            pair = [
                record
                for record in records
                if record.dataset == dataset and record.config_index == index
            ]
            by_treatment = {record.treatment: record for record in pair}
            denominator = by_treatment["quadratic"].time_train_s
            if denominator <= 0.0:
                raise RuntimeError("quadratic artifact has nonpositive training time")
            paired_ratios.append(by_treatment["grouped"].time_train_s / denominator)

    primary_improvements = [row["relative_improvement"] for row in primary]
    tuned_improvements = [row["relative_improvement"] for row in tuned]
    observed = {
        "primary_wins": sum(row["outcome"] == "win" for row in primary),
        "primary_task_macro_median_improvement": statistics.median(
            primary_improvements
        ),
        "primary_worst_task_improvement": min(primary_improvements),
        "tuned_task_macro_median_improvement": statistics.median(tuned_improvements),
        "tuned_worst_task_improvement": min(tuned_improvements),
        "median_paired_training_time_ratio": statistics.median(paired_ratios),
    }
    gates = _evaluate_decision_gates(
        primary_wins=observed["primary_wins"],
        primary_macro_median=observed["primary_task_macro_median_improvement"],
        primary_worst=observed["primary_worst_task_improvement"],
        tuned_macro_median=observed["tuned_task_macro_median_improvement"],
        tuned_worst=observed["tuned_worst_task_improvement"],
        median_paired_training_ratio=observed["median_paired_training_time_ratio"],
    )
    supported = all(gates.values())
    return {
        "comparisons": comparisons,
        "gates": gates,
        "observed": observed,
        "decision": "full grouped ablation supported"
        if supported
        else "performance not supportive",
        "full_grouped_ablation_supported": supported,
    }


def _resource_summary(records: list[ArtifactRecord]) -> dict[str, Any]:
    output: dict[str, Any] = {"overall": {}, "by_task": {}}
    for label, selected in [
        ("overall", records),
        *[
            (dataset, [record for record in records if record.dataset == dataset])
            for _task, dataset, _problem, _metric in TASKS
        ],
    ]:
        block: dict[str, Any] = {}
        for treatment in ("quadratic", "grouped"):
            rows = [record for record in selected if record.treatment == treatment]
            block[treatment] = {
                "total_training_time_s": sum(row.time_train_s for row in rows),
                "median_training_time_s": statistics.median(
                    row.time_train_s for row in rows
                ),
                "total_inference_time_s": sum(row.time_infer_s for row in rows),
                "median_inference_time_s": statistics.median(
                    row.time_infer_s for row in rows
                ),
                "median_absolute_peak_rss_bytes": statistics.median(
                    row.peak_rss_bytes for row in rows
                ),
                "maximum_absolute_peak_rss_bytes": max(
                    row.peak_rss_bytes for row in rows
                ),
                "median_incremental_peak_rss_bytes": statistics.median(
                    row.incremental_peak_rss_bytes for row in rows
                ),
                "maximum_incremental_peak_rss_bytes": max(
                    row.incremental_peak_rss_bytes for row in rows
                ),
                "total_model_bytes": sum(row.model_bytes for row in rows),
                "median_model_bytes": statistics.median(
                    row.model_bytes for row in rows
                ),
            }
        quadratic_time = block["quadratic"]["total_training_time_s"]
        block["grouped_to_quadratic_total_training_time_ratio"] = (
            block["grouped"]["total_training_time_s"] / quadratic_time
        )
        if label == "overall":
            output["overall"] = block
        else:
            output["by_task"][label] = block
    return output


def _paired_config_document() -> dict[str, Any]:
    base = base_p50()
    paired = paired_configs()
    entries = []
    for index in range(NUM_CONFIGS):
        entries.append(
            {
                "config_id": config_id(index),
                "base": base[index],
                "quadratic": paired["quadratic"][index],
                "grouped": paired["grouped"][index],
                "differing_fields": ["feature_test"],
            }
        )
    return {
        "schema_version": 1,
        "count": NUM_CONFIGS,
        "entries": entries,
        "sha256": hashlib.sha256(canonical_json_bytes(entries)).hexdigest(),
    }


def _contains_absolute_path(value: str) -> bool:
    return ntpath.isabs(value) or any(
        pattern.search(value)
        for pattern in (
            _FILE_URI,
            _WINDOWS_ABSOLUTE_PATH,
            _WINDOWS_ROOTED_PATH,
            _UNC_ABSOLUTE_PATH,
            _POSIX_ABSOLUTE_PATH,
        )
    )


def _assert_no_absolute_paths(value: Any) -> None:
    def walk(item: Any) -> Iterable[str]:
        if isinstance(item, str):
            yield item
        elif isinstance(item, dict):
            for key, nested in item.items():
                yield str(key)
                yield from walk(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                yield from walk(nested)

    offending = [text for text in walk(value) if _contains_absolute_path(text)]
    if offending:
        raise RuntimeError("sanitized output contains an absolute path")


def _write_json(path: Path, value: Any) -> None:
    _assert_no_absolute_paths(value)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            encoded = {
                field: json.dumps(row[field], sort_keys=True, separators=(",", ":"))
                if isinstance(row.get(field), (list, dict))
                else row.get(field)
                for field in fields
            }
            _assert_no_absolute_paths(encoded)
            writer.writerow(encoded)


def summarize(
    *,
    raw_dir: Path,
    output_dir: Path,
    provenance: dict[str, Any],
    schedule: dict[str, Any],
    namespace_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if namespace_state is None:
        namespace_state = validate_namespace(
            raw_dir=raw_dir,
            report_dir=output_dir.parent,
            phase="summarize_input",
        )
    loaded, records, artifact_set_sha256, raw_inventory = load_and_validate_artifacts(
        raw_dir
    )
    stale_or_unexpected = max(
        int(namespace_state["stale_or_unexpected"]),
        int(raw_inventory["stale_or_unexpected"]),
    )
    endpoint_rows, selected_configs = _endpoint_rows(loaded)
    _decorate_endpoint_selections(endpoint_rows, records, selected_configs)
    decision = _decision_summary(endpoint_rows, records)
    paired_document = _paired_config_document()
    summary = {
        "schema_version": 1,
        "label": "local three-task provisional grouped-statistic scout",
        "not_official_tabarena_full": True,
        "coverage": {
            "expected_outer_artifacts": EXPECTED_ARTIFACTS,
            "observed_outer_artifacts": len(records),
            "expected_bagged_child_fits": EXPECTED_CHILD_FITS,
            "failures": 0,
            "timeouts": 0,
            "duplicates": 0,
            "stale_or_unexpected": stale_or_unexpected,
            "complete": bool(namespace_state["complete"])
            and bool(raw_inventory["complete"]),
        },
        "provenance": provenance,
        "schedule": schedule,
        "artifact_set_sha256": artifact_set_sha256,
        "paired_configs_sha256": paired_document["sha256"],
        "endpoint_results": endpoint_rows,
        "decision_evaluation": decision,
        "resources": _resource_summary(records),
    }
    summary["summary_sha256"] = hashlib.sha256(
        canonical_json_bytes(summary)
    ).hexdigest()
    _assert_no_absolute_paths(summary)

    if output_dir.exists():
        raise RuntimeError("sanitized success output directory must not already exist")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".g8s1-success-", dir=str(output_dir.parent))
    )
    try:
        _write_json(staging / "scout_summary.json", summary)
        _write_json(staging / "paired_configs.json", paired_document)
        _write_csv(
            staging / "config_results.csv",
            [record.csv_row() for record in records],
            list(ArtifactRecord.__dataclass_fields__),
        )
        endpoint_csv_rows = [
            {**row, "selected_config_ids": row["selected_config_ids"]}
            for row in endpoint_rows
        ]
        _write_csv(
            staging / "endpoint_results.csv",
            endpoint_csv_rows,
            [
                "treatment",
                "dataset",
                "fold",
                "endpoint",
                "metric_error",
                "metric_error_val",
                "time_train_s",
                "time_infer_s",
                "selected_config_ids",
            ],
        )
        staging.replace(output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return summary


def write_failure_summary(
    *,
    output_dir: Path,
    provenance: dict[str, Any],
    schedule: dict[str, Any],
    error: Exception,
) -> None:
    message = str(error)
    if _contains_absolute_path(message):
        message = (
            "validation failed with a path-bearing diagnostic; inspect private logs"
        )
    failure = {
        "schema_version": 1,
        "label": "local three-task provisional grouped-statistic scout",
        "not_official_tabarena_full": True,
        "status": "integration failure",
        "error": message,
        "provenance": provenance,
        "schedule": schedule,
    }
    failure["summary_sha256"] = hashlib.sha256(
        canonical_json_bytes(failure)
    ).hexdigest()
    if output_dir.exists():
        raise RuntimeError("failure output requires a clean sanitized directory")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".g8s1-failure-", dir=str(output_dir.parent))
    )
    try:
        _write_json(staging / "scout_failure.json", failure)
        staging.replace(output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
