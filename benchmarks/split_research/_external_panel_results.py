"""Atomic result ledger and frozen aggregation for the external panel."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ._external_panel_data import atomic_write_json, load_json
from ._external_panel_protocol import (
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_REPETITIONS,
    BOOTSTRAP_SEED,
    DATASETS,
    FOLDS,
    HISTOGRAM_THREADS,
    MAX_DATASETS_WORSE_THAN_ONE_PERCENT,
    MAX_FIT_TIME_RATIO,
    MINIMUM_MEDIAN_IMPROVEMENT,
    PROFILES,
    PROTOCOL_NAME,
    RESULT_SCHEMA_VERSION,
    TIE_THRESHOLD,
    assert_no_absolute_paths,
    protocol_manifest,
    seal_record,
    sha256_json,
    validate_full_job_identity,
    validate_record_seal,
    validate_self_hashed_identity,
)


def new_ledger(protocol_sha256: str) -> Dict[str, Any]:
    return seal_record(
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "protocol_sha256": protocol_sha256,
            "jobs": {},
        },
        "ledger_sha256",
    )


def _validate_result_record(result: Mapping[str, Any]) -> None:
    if not isinstance(result, Mapping):
        raise ValueError("result ledger entries must be objects")
    validate_record_seal(result, "result_sha256")
    if result.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError("result entry has an unsupported schema version")
    if result.get("status") not in {"success", "failure"}:
        raise ValueError("result entry has an invalid status")
    if not isinstance(result.get("job_key"), str):
        raise ValueError("result entry is missing its job key")
    if not isinstance(result.get("config"), Mapping):
        raise ValueError("result entry is missing its config")
    if result.get("config_identity_sha256") != sha256_json(result["config"]):
        raise ValueError("result config identity mismatch")
    assert_no_absolute_paths(result)


def validate_result_for_job(result: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    """Validate a sealed worker record against its complete scheduled identity."""
    validate_full_job_identity(job)
    _validate_result_record(result)
    if result["job_key"] != job["job_key"]:
        raise ValueError("result job key differs from the scheduled full identity")
    if dict(result["config"]) != dict(job["config"]):
        raise ValueError("result config differs from the scheduled config")
    expected_hashes = {
        "source_identity_sha256": job["source_identity"]["source_identity_sha256"],
        "data_identity_sha256": job["data_identity"]["data_identity_sha256"],
        "config_identity_sha256": sha256_json(job["config"]),
    }
    for name, expected in expected_hashes.items():
        if result.get(name) != expected:
            raise ValueError("result {} mismatch".format(name))
    if result["status"] == "success":
        required = (
            "metrics",
            "fit_seconds",
            "peak_process_rss_bytes",
            "serialized_model_bytes",
            "raw_prediction_sha256",
            "canonical_tree_sha256",
            "best_iteration",
        )
        missing = [name for name in required if name not in result]
        if missing:
            raise ValueError(
                "successful result is missing fields: {}".format(
                    ", ".join(sorted(missing))
                )
            )
        if result.get("peak_process_rss_source") != "worker_process":
            raise ValueError("successful result RSS is not attributed to its worker")
        if not isinstance(result["metrics"], Mapping):
            raise ValueError("successful result metrics must be an object")
        if (
            not math.isfinite(float(result["fit_seconds"]))
            or float(result["fit_seconds"]) < 0.0
        ):
            raise ValueError("successful result fit time is invalid")
        if (
            not isinstance(result["peak_process_rss_bytes"], int)
            or result["peak_process_rss_bytes"] < 0
        ):
            raise ValueError("successful result worker RSS is invalid")
        if (
            not isinstance(result["serialized_model_bytes"], int)
            or result["serialized_model_bytes"] < 0
        ):
            raise ValueError("successful result model size is invalid")


def validate_ledger_entries(
    ledger: Mapping[str, Any], jobs: Sequence[Mapping[str, Any]]
) -> None:
    expected = {str(job["job_key"]): job for job in jobs}
    if len(expected) != len(jobs):
        raise ValueError("scheduled jobs contain duplicate full identities")
    for stored_key, result in ledger.get("jobs", {}).items():
        _validate_result_record(result)
        if stored_key != result["job_key"]:
            raise ValueError("result is stored under a different job key")
        job = expected.get(stored_key)
        if job is not None:
            validate_result_for_job(result, job)


def load_ledger(path: Path, protocol_sha256: str) -> Dict[str, Any]:
    if not path.exists():
        return new_ledger(protocol_sha256)
    ledger = load_json(path)
    validate_record_seal(ledger, "ledger_sha256")
    if ledger.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError("unsupported external-panel result schema")
    if ledger.get("protocol_sha256") != protocol_sha256:
        raise ValueError("result ledger belongs to a different frozen protocol")
    if not isinstance(ledger.get("jobs"), dict):
        raise ValueError("result ledger jobs must be an object")
    for stored_key, result in ledger["jobs"].items():
        _validate_result_record(result)
        if stored_key != result["job_key"]:
            raise ValueError("result is stored under a different job key")
    assert_no_absolute_paths(ledger)
    return ledger


def store_result(
    ledger_path: Path,
    ledger: Dict[str, Any],
    result: Mapping[str, Any],
    expected_job: Optional[Mapping[str, Any]] = None,
) -> None:
    _validate_result_record(result)
    if expected_job is not None:
        validate_result_for_job(result, expected_job)
    ledger["jobs"][str(result["job_key"])] = dict(result)
    sealed = seal_record(ledger, "ledger_sha256")
    assert_no_absolute_paths(sealed)
    ledger.clear()
    ledger.update(sealed)
    atomic_write_json(ledger_path, sealed)


def _result_lookup(
    jobs: Sequence[Mapping[str, Any]], ledger: Mapping[str, Any]
) -> Dict[Tuple[int, int, str, str, str], Mapping[str, Any]]:
    lookup = {}
    completed = ledger.get("jobs", {})
    for job in jobs:
        result = completed.get(job["job_key"])
        if result is None:
            continue
        config = result["config"]
        key = (
            int(config["task_id"]),
            int(config["fold"]),
            str(config["profile"]),
            str(config["treatment"]),
            str(config["role"]),
        )
        lookup[key] = result
    return lookup


def exact_control_checks(
    jobs: Sequence[Mapping[str, Any]], ledger: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    lookup = _result_lookup(jobs, ledger)
    checks = []
    for dataset in DATASETS:
        task_id = int(dataset["task_id"])
        for profile in PROFILES:
            profile_name = str(profile["name"])
            explicit = lookup.get((task_id, 0, profile_name, "control", "treatment"))
            implicit = lookup.get(
                (task_id, 0, profile_name, "implicit-control", "implicit_control_check")
            )
            fields = (
                "raw_prediction_sha256",
                "canonical_tree_sha256",
                "best_iteration",
            )
            complete = bool(
                explicit
                and implicit
                and explicit.get("status") == "success"
                and implicit.get("status") == "success"
            )
            exact = complete and all(
                explicit.get(field) == implicit.get(field) for field in fields
            )
            checks.append(
                {
                    "task_id": task_id,
                    "profile": profile_name,
                    "stress_only": bool(dataset["stress_only"]),
                    "complete": complete,
                    "exact": bool(exact),
                    "matched_fields": list(fields),
                }
            )
    return checks


def _median(values: Sequence[float]) -> float:
    import numpy as np

    return float(np.median(np.asarray(values, dtype=np.float64)))


def bootstrap_interval(task_improvements: Sequence[float]) -> Optional[List[float]]:
    import numpy as np

    values = np.asarray(task_improvements, dtype=np.float64)
    if values.size == 0:
        return None
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    indices = generator.integers(
        0, values.size, size=(BOOTSTRAP_REPETITIONS, values.size)
    )
    medians = np.median(values[indices], axis=1)
    tail = (1.0 - BOOTSTRAP_CONFIDENCE) / 2.0
    return [
        float(np.quantile(medians, tail)),
        float(np.quantile(medians, 1.0 - tail)),
    ]


def _task_aggregation(
    lookup: Mapping[Tuple[int, int, str, str, str], Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[float], List[float]]:
    per_task = []
    decision_task_improvements: List[float] = []
    decision_pair_time_ratios: List[float] = []
    for dataset in DATASETS:
        task_id = int(dataset["task_id"])
        per_profile = []
        profile_improvements = []
        for profile in PROFILES:
            profile_name = str(profile["name"])
            control_losses = []
            candidate_losses = []
            profile_time_ratios = []
            for fold in FOLDS:
                control = lookup.get(
                    (task_id, fold, profile_name, "control", "treatment")
                )
                candidate = lookup.get(
                    (task_id, fold, profile_name, "candidate", "treatment")
                )
                if not control or not candidate:
                    continue
                if (
                    control.get("status") != "success"
                    or candidate.get("status") != "success"
                ):
                    continue
                control_seconds = float(control["fit_seconds"])
                candidate_seconds = float(candidate["fit_seconds"])
                if control_seconds <= 0.0 or candidate_seconds < 0.0:
                    continue
                control_losses.append(float(control["metrics"]["primary_loss"]))
                candidate_losses.append(float(candidate["metrics"]["primary_loss"]))
                ratio = candidate_seconds / control_seconds
                profile_time_ratios.append(ratio)
                if not dataset["stress_only"]:
                    decision_pair_time_ratios.append(ratio)
            complete = len(control_losses) == len(FOLDS)
            row: Dict[str, Any] = {"profile": profile_name, "complete": complete}
            if complete:
                control_median = _median(control_losses)
                candidate_median = _median(candidate_losses)
                if control_median <= 0.0:
                    row["complete"] = False
                    row["aggregation_error"] = (
                        "control median primary loss is not positive"
                    )
                else:
                    improvement = (control_median - candidate_median) / control_median
                    row.update(
                        {
                            "control_median_primary_loss": control_median,
                            "candidate_median_primary_loss": candidate_median,
                            "relative_primary_loss_improvement": improvement,
                            "median_fit_time_ratio": _median(profile_time_ratios),
                        }
                    )
                    profile_improvements.append(improvement)
            per_profile.append(row)
        task_complete = len(profile_improvements) == len(PROFILES)
        task_row: Dict[str, Any] = {
            "task_id": task_id,
            "dataset": str(dataset["name"]),
            "stress_only": bool(dataset["stress_only"]),
            "complete": task_complete,
            "profiles": per_profile,
        }
        if task_complete:
            task_improvement = _median(profile_improvements)
            outcome = (
                "win"
                if task_improvement >= TIE_THRESHOLD
                else "loss"
                if task_improvement <= -TIE_THRESHOLD
                else "tie"
            )
            task_row.update(
                {
                    "median_relative_primary_loss_improvement": task_improvement,
                    "outcome": outcome,
                }
            )
            if not dataset["stress_only"]:
                decision_task_improvements.append(task_improvement)
        per_task.append(task_row)
    return per_task, decision_task_improvements, decision_pair_time_ratios


def summarize_results(
    jobs: Sequence[Mapping[str, Any]],
    ledger: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    data_identities: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    validate_self_hashed_identity(source_identity, "source_identity_sha256")
    for data_identity in data_identities:
        validate_self_hashed_identity(data_identity, "data_identity_sha256")
    validate_ledger_entries(ledger, jobs)
    expected_keys = {str(job["job_key"]) for job in jobs}
    completed = {
        key: value
        for key, value in ledger.get("jobs", {}).items()
        if key in expected_keys
    }
    successes = {
        key: value for key, value in completed.items() if value["status"] == "success"
    }
    failures = {
        key: value for key, value in completed.items() if value["status"] != "success"
    }
    lookup = _result_lookup(jobs, ledger)
    checks = exact_control_checks(jobs, ledger)
    decision_checks = [item for item in checks if not item["stress_only"]]
    expected_decision_checks = sum(not item["stress_only"] for item in DATASETS) * len(
        PROFILES
    )
    all_decision_checks_exact = len(
        decision_checks
    ) == expected_decision_checks and all(
        item["complete"] and item["exact"] for item in decision_checks
    )

    per_task, task_improvements, pair_time_ratios = _task_aggregation(lookup)
    decision_rows = [row for row in per_task if not row["stress_only"]]
    decision_complete = all(row["complete"] for row in decision_rows)
    outcomes = {
        name: sum(row.get("outcome") == name for row in decision_rows)
        for name in ("win", "tie", "loss")
    }
    median_improvement = _median(task_improvements) if decision_complete else None
    median_time_ratio = _median(pair_time_ratios) if decision_complete else None
    worse_than_one_percent = (
        sum(value < -0.01 for value in task_improvements) if decision_complete else None
    )
    decision_treatment_jobs = [
        job
        for job in jobs
        if job["config"]["role"] == "treatment" and not job["config"]["stress_only"]
    ]
    expected_decision_fits = (
        sum(not item["stress_only"] for item in DATASETS)
        * len(FOLDS)
        * len(PROFILES)
        * 2
    )
    all_decision_fits_valid = (
        len(decision_treatment_jobs) == expected_decision_fits
        and len({job["job_key"] for job in decision_treatment_jobs})
        == expected_decision_fits
        and all(
            job["job_key"] in successes
            and successes[job["job_key"]].get("finite_predictions") is True
            and successes[job["job_key"]].get("correct_prediction_shape") is True
            for job in decision_treatment_jobs
        )
    )
    frozen_gates = {
        "all_decision_fits_complete_finite_and_shaped": all_decision_fits_valid,
        "wins_at_least_7_of_12": decision_complete and outcomes["win"] >= 7,
        "median_improvement_at_least_0_25_percent": (
            decision_complete
            and median_improvement is not None
            and median_improvement >= MINIMUM_MEDIAN_IMPROVEMENT
        ),
        "at_most_3_datasets_worse_than_1_percent": (
            decision_complete
            and worse_than_one_percent is not None
            and worse_than_one_percent <= MAX_DATASETS_WORSE_THAN_ONE_PERCENT
        ),
        "median_fit_time_ratio_at_most_1_15": (
            decision_complete
            and median_time_ratio is not None
            and median_time_ratio <= MAX_FIT_TIME_RATIO
        ),
    }
    summary = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "benchmark": PROTOCOL_NAME,
        "protocol_sha256": protocol_manifest()["protocol_sha256"],
        "source_identity": dict(source_identity),
        "data_identities": [dict(value) for value in data_identities],
        "coverage": {
            "expected_jobs": len(jobs),
            "completed_jobs": len(completed),
            "successful_jobs": len(successes),
            "failed_jobs": len(failures),
            "missing_jobs": len(expected_keys - set(completed)),
            "unexpected_ledger_jobs_ignored": len(
                set(ledger.get("jobs", {})) - expected_keys
            ),
        },
        "resources": {
            "fit_seconds_sum": sum(
                float(value["fit_seconds"]) for value in successes.values()
            ),
            "peak_process_rss_bytes_max": max(
                (
                    int(value["peak_process_rss_bytes"])
                    for value in completed.values()
                    if value.get("peak_process_rss_bytes") is not None
                    and value.get("peak_process_rss_source") == "worker_process"
                ),
                default=None,
            ),
            "serialized_model_bytes_sum": sum(
                int(value["serialized_model_bytes"]) for value in successes.values()
            ),
            "histogram_threads_per_fit": HISTOGRAM_THREADS,
            "one_isolated_subprocess_per_fit": True,
        },
        "control_identity_checks": {
            "expected": len(checks),
            "complete": sum(item["complete"] for item in checks),
            "exact": sum(item["exact"] for item in checks),
            "all_decision_checks_exact": all_decision_checks_exact,
            "stress_checks_do_not_affect_promotion": True,
            "checks": checks,
        },
        "task_aggregation": per_task,
        "decision_aggregation": {
            "complete": decision_complete,
            "win_tie_loss": outcomes,
            "median_relative_primary_loss_improvement": median_improvement,
            "task_bootstrap_interval": (
                bootstrap_interval(task_improvements) if decision_complete else None
            ),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "datasets_worse_than_one_percent": worse_than_one_percent,
            "median_paired_fit_time_ratio": median_time_ratio,
        },
        "protocol_valid": all_decision_fits_valid and all_decision_checks_exact,
        "frozen_promotion_gates": frozen_gates,
        "grouped_8_advances": (
            all_decision_fits_valid
            and all_decision_checks_exact
            and all(frozen_gates.values())
        ),
        "failures": [
            {
                "job_key": key,
                "config": value["config"],
                "error_type": value.get("error_type"),
                "error_message": value.get("error_message"),
            }
            for key, value in sorted(failures.items())
        ],
    }
    assert_no_absolute_paths(summary)
    return summary
