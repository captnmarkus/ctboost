"""Pure validation-only evaluation of the frozen CTBoost 0.1.59 pilot.

The caller supplies records, not result directories. This module never loads
datasets, predictions, test scores or prior leaderboard results.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from typing import Any, Iterable, Mapping


def expected_pilot_jobs(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return canonical jobs; scheduling order may differ without changing IDs."""
    jobs = []
    splits = protocol["splits"]
    for dataset in protocol["datasets"]:
        for fold in splits["inner_fold_indices"]:
            for variant in protocol["variants_by_problem_type"][dataset["problem_type"]]:
                jobs.append({
                    "job_id": (
                        f"{dataset['dataset_name']}::r{splits['outer_repeat']}"
                        f"f{splits['outer_fold']}::inner{fold}::{variant}"
                    ),
                    "dataset": dataset["dataset_name"],
                    "problem_type": dataset["problem_type"],
                    "variant": variant,
                    "inner_fold": fold,
                    "seed": splits["model_seed_by_inner_fold"][str(fold)],
                })
    return jobs


def _key(record: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    return record.get("dataset"), record.get("variant"), record.get("inner_fold")


def _finite_number(value: Any, *, positive: bool = False) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and (value > 0 if positive else value >= 0)
    )


def _sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _percentile_linear(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def evaluate_pilot(
    protocol: Mapping[str, Any],
    records: Iterable[Mapping[str, Any]],
    *,
    protocol_sha256: str,
) -> dict[str, Any]:
    """Evaluate all arms without modifying inputs or selecting on outer tests.

    ``protocol_sha256`` is the SHA256 of the frozen JSON file bytes, supplied by
    the caller that read it. Every expected job must have one terminal record.
    A missing/duplicate/foreign record or provenance mismatch blocks the entire
    decision. A valid terminal failure blocks the affected comparison while
    remaining visible in the report; it cannot disappear through filtering.

    Required record fields are the canonical job fields (``job_id`` optional),
    ``status``, ``protocol_sha256``, ``split_sha256``, ``preprocessing_sha256`` and
    ``resource_contract_sha256``. Successful records additionally require finite
    nonnegative ``metric_error``, positive ``fit_seconds``/``peak_rss_bytes`` and
    boolean ``deadline_stopped``. Failure records may retain partial metrics.
    """
    if not _sha256(protocol_sha256):
        raise ValueError("protocol_sha256 must be the frozen file's lowercase SHA256")
    expected = expected_pilot_jobs(protocol)
    if len(expected) != protocol["compute"]["expected_fit_count"]:
        raise ValueError("protocol expected_fit_count does not match its task/arm/fold grid")
    expected_by_key = {_key(job): job for job in expected}
    if len(expected_by_key) != len(expected):
        raise ValueError("protocol contains duplicate jobs")
    result_records = list(records)
    global_errors: list[str] = []
    observed: dict[tuple[Any, Any, Any], list[Mapping[str, Any]]] = {}
    allowed_statuses = {"ok", "failed", "resource_limit", "timeout", "not_applicable"}
    if any(not isinstance(record, Mapping) for record in result_records):
        global_errors.append("non_mapping_record")
    for record in result_records:
        if not isinstance(record, Mapping):
            continue
        identity = _key(record)
        if not all(isinstance(value, (str, int)) and not isinstance(value, bool) for value in identity):
            global_errors.append("malformed_job_identity")
            continue
        observed.setdefault(identity, []).append(record)
        if identity not in expected_by_key:
            global_errors.append(f"unexpected_job:{identity!r}")
            continue
        job = expected_by_key[identity]
        for name in ("problem_type", "seed"):
            if record.get(name) != job[name] or isinstance(record.get(name), bool):
                global_errors.append(f"{job['job_id']}:{name}_mismatch")
        if "job_id" in record and record["job_id"] != job["job_id"]:
            global_errors.append(f"{job['job_id']}:job_id_mismatch")
        if record.get("protocol_sha256") != protocol_sha256:
            global_errors.append(f"{job['job_id']}:protocol_sha256_mismatch")
        for name in ("split_sha256", "preprocessing_sha256", "resource_contract_sha256"):
            if not _sha256(record.get(name)):
                global_errors.append(f"{job['job_id']}:{name}_missing_or_invalid")
        if record.get("status") not in allowed_statuses:
            global_errors.append(f"{job['job_id']}:invalid_status")
        if record.get("status") == "ok":
            for name in ("metric_error", "fit_seconds", "peak_rss_bytes"):
                if not _finite_number(record.get(name), positive=name != "metric_error"):
                    global_errors.append(f"{job['job_id']}:{name}_invalid")
            if not isinstance(record.get("deadline_stopped"), bool):
                global_errors.append(f"{job['job_id']}:deadline_stopped_invalid")
            if job["problem_type"] == "binary" and _finite_number(record.get("metric_error")) and record["metric_error"] > 1:
                global_errors.append(f"{job['job_id']}:binary_error_out_of_range")
    missing = [job["job_id"] for identity, job in expected_by_key.items() if identity not in observed]
    duplicate = [expected_by_key[k]["job_id"] if k in expected_by_key else repr(k) for k, values in observed.items() if len(values) != 1]
    if missing:
        global_errors.append("missing_expected_jobs")
    if duplicate:
        global_errors.append("duplicate_jobs")
    unique = {identity: values[0] for identity, values in observed.items() if len(values) == 1 and identity in expected_by_key}
    resource_hashes = {record.get("resource_contract_sha256") for record in unique.values() if _sha256(record.get("resource_contract_sha256"))}
    if len(resource_hashes) > 1:
        global_errors.append("resource_contract_sha256_not_constant")
    for dataset in protocol["datasets"]:
        for fold in protocol["splits"]["inner_fold_indices"]:
            block = [r for (name, _, inner_fold), r in unique.items() if name == dataset["dataset_name"] and inner_fold == fold]
            for name in ("split_sha256", "preprocessing_sha256"):
                values = {r[name] for r in block if _sha256(r.get(name))}
                if len(values) > 1:
                    global_errors.append(f"{dataset['dataset_name']}::inner{fold}:{name}_not_paired")

    gate = protocol["decision_gate"]
    max_rss = protocol["compute"]["max_process_rss_gib"] * 1024**3
    max_fit_time = protocol["compute"]["fit_time_limit_seconds"] + protocol["compute"]["hard_fit_grace_seconds"]
    family_reports = {}
    for family, variants in protocol["variants_by_problem_type"].items():
        datasets = [d["dataset_name"] for d in protocol["datasets"] if d["problem_type"] == family]
        variant_reports = {}
        for variant in variants:
            if variant == "baseline":
                continue
            reasons = []
            dataset_comparisons = []
            required = [
                unique.get((dataset, arm, fold))
                for dataset in datasets
                for fold in protocol["splits"]["inner_fold_indices"]
                for arm in ("baseline", variant)
            ]
            if any(record is None for record in required):
                reasons.append("missing_or_duplicate_paired_fit")
            if any(record is not None and record.get("status") != "ok" for record in required):
                reasons.append("failed_or_inapplicable_paired_fit")
            if any(record is not None and record.get("status") == "ok" and not all(_finite_number(record.get(name), positive=name != "metric_error") for name in ("metric_error", "fit_seconds", "peak_rss_bytes")) for record in required):
                reasons.append("nonfinite_or_invalid_paired_metric")
            if any(record is not None and record.get("status") == "ok" and not isinstance(record.get("deadline_stopped"), bool) for record in required):
                reasons.append("invalid_paired_deadline_record")
            if any(record is not None and _finite_number(record.get("peak_rss_bytes")) and record["peak_rss_bytes"] > max_rss for record in required):
                reasons.append("peak_process_memory_limit_exceeded")
            if any(record is not None and _finite_number(record.get("fit_seconds")) and record["fit_seconds"] > max_fit_time for record in required):
                reasons.append("fit_hard_grace_limit_exceeded")
            if not reasons:
                for dataset in datasets:
                    baseline = [unique[(dataset, "baseline", fold)] for fold in protocol["splits"]["inner_fold_indices"]]
                    candidate = [unique[(dataset, variant, fold)] for fold in protocol["splits"]["inner_fold_indices"]]
                    baseline_error = statistics.mean(r["metric_error"] for r in baseline)
                    candidate_error = statistics.mean(r["metric_error"] for r in candidate)
                    baseline_seconds = statistics.mean(r["fit_seconds"] for r in baseline)
                    candidate_seconds = statistics.mean(r["fit_seconds"] for r in candidate)
                    dataset_comparisons.append({
                        "dataset": dataset,
                        "baseline_error": baseline_error,
                        "candidate_error": candidate_error,
                        "relative_error_improvement": (baseline_error - candidate_error) / max(baseline_error, 1e-12),
                        "baseline_fit_seconds": baseline_seconds,
                        "candidate_fit_seconds": candidate_seconds,
                        "fit_time_ratio": candidate_seconds / max(baseline_seconds, 1e-12),
                        "baseline_deadline_stopped_fits": sum(r["deadline_stopped"] for r in baseline),
                        "candidate_deadline_stopped_fits": sum(r["deadline_stopped"] for r in candidate),
                    })
            stats = {}
            if dataset_comparisons:
                improvements = [d["relative_error_improvement"] for d in dataset_comparisons]
                runtimes = [d["fit_time_ratio"] for d in dataset_comparisons]
                stats = {
                    "median_relative_error_improvement": statistics.median(improvements),
                    "dataset_wins": sum(value > gate["dataset_win_epsilon"] for value in improvements),
                    "dataset_count": len(datasets),
                    "worst_relative_error_increase": max(0.0, -min(improvements)),
                    "geometric_mean_fit_time_ratio": math.exp(statistics.mean(math.log(value) for value in runtimes)),
                    "p90_fit_time_ratio": _percentile_linear(runtimes, 0.90),
                }
                if round(stats["median_relative_error_improvement"], 12) < round(gate["minimum_median_relative_error_improvement"], 12):
                    reasons.append("median_quality_gain_below_threshold")
                if stats["dataset_wins"] <= len(datasets) / 2:
                    reasons.append("no_strict_majority_dataset_wins")
                if round(stats["worst_relative_error_increase"], 12) > round(gate["maximum_any_dataset_relative_error_increase"], 12):
                    reasons.append("worst_dataset_regression_exceeds_threshold")
                if round(stats["geometric_mean_fit_time_ratio"], 12) > round(gate["maximum_geometric_mean_fit_time_ratio"], 12):
                    reasons.append("geometric_mean_runtime_exceeds_threshold")
                if round(stats["p90_fit_time_ratio"], 12) > round(gate["maximum_p90_fit_time_ratio"], 12):
                    reasons.append("p90_runtime_exceeds_threshold")
            if global_errors:
                reasons.append("global_integrity_check_failed")
            variant_reports[variant] = {
                "passed": not reasons,
                "reasons": reasons,
                "metrics": stats,
                "datasets": dataset_comparisons,
            }
        passing = [name for name, report in variant_reports.items() if report["passed"]]
        priority = {name: i for i, name in enumerate(gate["tie_priority"])}
        passing.sort(key=lambda name: (-round(variant_reports[name]["metrics"]["median_relative_error_improvement"], 12), priority[name]))
        winner = passing[0] if passing else "baseline"
        family_reports[family] = {"passed": bool(passing), "winner": winner, "variants": variant_reports}
    return {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "expected_fit_count": len(expected),
        "observed_record_count": len(result_records),
        "expected_job_ids": [job["job_id"] for job in expected],
        "missing_job_ids": missing,
        "duplicate_job_ids": duplicate,
        "status_counts": dict(Counter(str(record.get("status")) for record in result_records if isinstance(record, Mapping))),
        "global_integrity_passed": not global_errors,
        "global_errors": sorted(set(global_errors)),
        "families": family_reports,
        "approved_modes_by_problem_type": {name: report["winner"] for name, report in family_reports.items()},
        "full_hpo_justified": not global_errors and any(report["passed"] for report in family_reports.values()),
        "interpretation": gate["interpretation"],
    }
