"""Decision-gate regressions: completeness, paired provenance and real tradeoffs."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.tabarena.pilot_evaluation import evaluate_pilot, expected_pilot_jobs


@pytest.fixture
def pilot():
    path = Path(__file__).resolve().parents[1] / "benchmarks/tabarena/pilot_0159_v1.json"
    raw = path.read_bytes()
    protocol = json.loads(raw)
    protocol_hash = hashlib.sha256(raw).hexdigest()
    records = []
    for job in expected_pilot_jobs(protocol):
        block_hash = hashlib.sha256(f"{job['dataset']}:{job['inner_fold']}".encode()).hexdigest()
        records.append({
            **job,
            "status": "ok",
            "metric_error": 0.5,
            "fit_seconds": 10.0,
            "peak_rss_bytes": 100_000_000,
            "deadline_stopped": False,
            "protocol_sha256": protocol_hash,
            "split_sha256": block_hash,
            "preprocessing_sha256": block_hash,
            "resource_contract_sha256": "a" * 64,
        })
    return protocol, protocol_hash, records


def evaluate(pilot):
    protocol, protocol_hash, records = pilot
    return evaluate_pilot(protocol, records, protocol_sha256=protocol_hash)


def improve(records, variant="backtracking_3", family="binary", error=0.48, runtime=20):
    for record in records:
        if record["variant"] == variant and record["problem_type"] == family:
            record.update(metric_error=error, fit_seconds=runtime)


def test_grid_and_no_gain_do_not_launch_duplicate_hpo(pilot):
    report = evaluate(pilot)
    assert report["expected_fit_count"] == 88
    assert len(set(report["expected_job_ids"])) == 88
    assert report["status_counts"] == {"ok": 88}
    assert report["global_integrity_passed"]
    assert not report["full_hpo_justified"]
    assert set(report["approved_modes_by_problem_type"].values()) == {"baseline"}


def test_admission_is_per_family_and_does_not_mutate_inputs(pilot):
    improve(pilot[2])
    before = copy.deepcopy(pilot)
    report = evaluate(pilot)
    assert report["full_hpo_justified"]
    assert report["approved_modes_by_problem_type"] == {
        "binary": "backtracking_3", "regression": "baseline", "multiclass": "baseline"
    }
    metrics = report["families"]["binary"]["variants"]["backtracking_3"]["metrics"]
    assert metrics["median_relative_error_improvement"] == pytest.approx(0.04)
    assert metrics["geometric_mean_fit_time_ratio"] == pytest.approx(2)
    assert pilot == before


def test_multiclass_selects_largest_gain_and_declared_tie_priority(pilot):
    for variant, error in (("full_3", 0.49), ("joint", 0.48), ("full_3_joint", 0.47)):
        improve(pilot[2], variant, "multiclass", error)
    assert evaluate(pilot)["approved_modes_by_problem_type"]["multiclass"] == "full_3_joint"
    for variant in ("full_3", "joint", "full_3_joint"):
        improve(pilot[2], variant, "multiclass", 0.48)
    assert evaluate(pilot)["approved_modes_by_problem_type"]["multiclass"] == "full_3"


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "foreign", "hash", "seed", "split", "preprocessing", "resources", "nan", "infinite_time", "job_id", "deadline_flag", "binary_range", "missing_provenance"])
def test_incomplete_or_unpaired_records_block_all_hpo(pilot, mutation):
    records = pilot[2]
    improve(records)
    if mutation == "missing":
        records.pop()
    elif mutation == "duplicate":
        records.append(dict(records[0]))
    elif mutation == "foreign":
        records.append({**records[0], "dataset": "foreign_dataset"})
    elif mutation == "hash":
        records[0]["protocol_sha256"] = "b" * 64
    elif mutation == "seed":
        records[0]["seed"] += 1
    elif mutation == "split":
        records[0]["split_sha256"] = "b" * 64
    elif mutation == "preprocessing":
        records[0]["preprocessing_sha256"] = "b" * 64
    elif mutation == "resources":
        records[0]["resource_contract_sha256"] = "b" * 64
    elif mutation == "nan":
        records[0]["metric_error"] = float("nan")
    elif mutation == "infinite_time":
        records[0]["fit_seconds"] = float("inf")
    elif mutation == "job_id":
        records[0]["job_id"] = "wrong"
    elif mutation == "deadline_flag":
        records[0]["deadline_stopped"] = None
    elif mutation == "binary_range":
        records[0]["metric_error"] = 1.1
    elif mutation == "missing_provenance":
        records[0].pop("split_sha256")
    report = evaluate(pilot)
    assert not report["global_integrity_passed"]
    assert report["global_errors"]
    assert not report["full_hpo_justified"]


def test_runtime_quality_and_large_regression_guards_are_independent(pilot):
    records = pilot[2]
    improve(records, error=0.40)
    binary_datasets = [d["dataset_name"] for d in pilot[0]["datasets"] if d["problem_type"] == "binary"]
    for record in records:
        if record["variant"] == "backtracking_3" and record["dataset"] == binary_datasets[-1]:
            record.update(metric_error=0.60, fit_seconds=80)
    report = evaluate(pilot)["families"]["binary"]["variants"]["backtracking_3"]
    assert report["metrics"]["dataset_wins"] == 2
    assert report["metrics"]["median_relative_error_improvement"] == pytest.approx(0.2)
    assert report["metrics"]["p90_fit_time_ratio"] == pytest.approx(6.8)
    assert set(report["reasons"]) == {
        "worst_dataset_regression_exceeds_threshold",
        "geometric_mean_runtime_exceeds_threshold",
        "p90_runtime_exceeds_threshold",
    }


def test_threshold_boundaries_are_not_rejected_by_float_rounding(pilot):
    improve(pilot[2], error=0.495, runtime=30)
    report = evaluate(pilot)["families"]["binary"]["variants"]["backtracking_3"]
    assert report["passed"]
    assert report["metrics"]["median_relative_error_improvement"] == pytest.approx(0.01)
    assert report["metrics"]["geometric_mean_fit_time_ratio"] == pytest.approx(3.0)


def test_average_errors_before_relative_improvement(pilot):
    for record in pilot[2]:
        if record["problem_type"] == "binary":
            record["metric_error"] = 0.1 if record["inner_fold"] == 0 else 0.9
            if record["variant"] == "backtracking_3" and record["inner_fold"] == 0:
                record["metric_error"] = 0.05
    report = evaluate(pilot)["families"]["binary"]["variants"]["backtracking_3"]
    assert report["metrics"]["median_relative_error_improvement"] == pytest.approx(0.05)


def test_perfect_baseline_candidate_ties_cannot_claim_improvement(pilot):
    for record in pilot[2]:
        record["metric_error"] = 0
    report = evaluate(pilot)
    assert not report["full_hpo_justified"]
    assert report["families"]["binary"]["variants"]["backtracking_3"]["metrics"]["dataset_wins"] == 0


@pytest.mark.parametrize("status", ["failed", "timeout", "resource_limit", "not_applicable"])
def test_terminal_failures_cannot_be_filtered_from_comparison(pilot, status):
    for variant in ("full_3", "full_3_joint"):
        improve(pilot[2], variant, "multiclass")
    record = next(r for r in pilot[2] if r["variant"] == "full_3_joint")
    record.update(status=status, metric_error=None)
    report = evaluate(pilot)
    assert report["global_integrity_passed"]
    assert report["status_counts"][status] == 1
    assert report["approved_modes_by_problem_type"]["multiclass"] == "full_3"
    assert not report["families"]["multiclass"]["variants"]["full_3_joint"]["passed"]


def test_valid_deadline_stop_is_retained_but_overrun_and_memory_fail(pilot):
    improve(pilot[2])
    candidate = next(r for r in pilot[2] if r["variant"] == "backtracking_3")
    candidate["deadline_stopped"] = True
    assert evaluate(pilot)["full_hpo_justified"]
    candidate["fit_seconds"] = 331
    report = evaluate(pilot)["families"]["binary"]["variants"]["backtracking_3"]
    assert "fit_hard_grace_limit_exceeded" in report["reasons"]
    candidate["fit_seconds"] = 20
    candidate["peak_rss_bytes"] = 8 * 1024**3 + 1
    report = evaluate(pilot)["families"]["binary"]["variants"]["backtracking_3"]
    assert "peak_process_memory_limit_exceeded" in report["reasons"]


def test_outer_test_scores_are_never_used(pilot):
    improve(pilot[2])
    before = evaluate(pilot)
    for record in pilot[2]:
        record["test_metric_error"] = 0 if record["variant"] == "baseline" else float("inf")
    assert evaluate(pilot) == before
