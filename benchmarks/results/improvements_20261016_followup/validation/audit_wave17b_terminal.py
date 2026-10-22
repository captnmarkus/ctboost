"""Audit closed wave-17b metadata/hashes; never decode data, arrays or models."""
from __future__ import annotations
import copy
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("selection_audit", HERE / "audit_wave17b_selection.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
ROOT, OUT, read, digest = audit.ROOT, audit.OUT, audit.read, audit.digest


def identities(rows):
    result = {(r["dataset"], r["library"]): r for r in rows}
    assert len(result) == len(rows)
    return result


def time(value):
    return datetime.fromisoformat(value)


def main():
    plan, controls = read(OUT / "plan.json"), read(OUT / "scalar_controls.json")
    frozen, dev = read(OUT / "frozen_selections.json"), read(OUT / "development.json")
    summary = read(OUT / "summary.json")
    reviewed_selection = read(HERE / "wave17b_selection_independent_review.json")
    assert reviewed_selection["audit_source_sha256"] == digest(HERE / "audit_wave17b_selection.py")
    assert reviewed_selection["controls_sha256"] == frozen["scalar_controls_sha256"] == digest(OUT / "scalar_controls.json")
    assert frozen["plan_sha256"] == dev["plan_sha256"] == reviewed_selection["plan_sha256"] == digest(OUT / "plan.json")
    assert dev["frozen_selections_sha256"] == digest(OUT / "frozen_selections.json")
    for path, expected in plan["dependencies"].items():
        if Path(path).suffix == ".py" or Path(path).name == "robustness_protocol_scalar_numerical.json":
            assert digest(path) == expected
    assert digest(ROOT / "benchmarks/fair_accuracy_scalar_numerical_retry.py") == plan["script_sha256"]
    assert frozen["all_fits_successful"] is True and frozen["new_anchor_fits_completed"] == 0
    assert frozen["new_clean_union_fits_completed"] == frozen["new_tree_fits_planned"] == 24
    assert frozen["carried_models"] == 12 and frozen["official_test_scored"] is False
    cases = {c["dataset_name"]: c for c in plan["cases"]}
    original_jobs, jobs = identities(plan["jobs"]), identities(frozen["jobs"])
    selections, bases, choices = identities(frozen["selections"]), identities(plan["base_selections"]), identities(controls["choices"])
    expected = {(dataset, library) for dataset in cases for library in ("ctboost", "xgboost", "lightgbm")}
    assert len(expected) == 36 and jobs.keys() == original_jobs.keys() == selections.keys() == bases.keys() == expected
    assert len(choices) == 24
    transfer_checks, carried, resources, correction_checks = [], [], [], []
    for identity in sorted(expected):
        job, selected = original_jobs[identity], selections[identity]
        model_path = Path(selected["model_path"])
        assert digest(model_path) == selected["model_sha256"] and model_path.stat().st_size == selected["model_bytes"]
        if identity not in choices:
            assert cases[identity[0]]["problem_type"] == "multiclass"
            assert selected == bases[identity] and jobs[identity] == job
            carried.append(identity)
            continue
        choice = choices[identity]
        problem = cases[identity[0]]["problem_type"]
        directory = OUT / "transfer_fits" / identity[0] / identity[1]
        record, clean = read(directory / "job.json"), read(directory / "clean_fit.json")
        assert (record["dataset"], record["library"]) == identity and record["status"] == clean["status"] == "ok"
        assert record["selection"] == selected
        assert time(controls["created_at"]) <= time(record["started_at"]) <= time(record["finished_at"]) <= time(frozen["created_at"])
        clean_controls = {name: job.get(name) for name in ("parameters", "rounds", "temperature")}
        clean_controls.update(leaf_refit=None, binary_refit=None, multiclass_refit=None)
        assert clean["clean_fit_controls"] == clean_controls and clean["old_scalar_corrections_disabled_before_fit"] is True
        assert clean["clean_fit_controls_sha256"] == hashlib.sha256(json.dumps(clean_controls, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        assert clean["fit"]["requested_rounds"] == job["rounds"] and clean["fit"]["early_stopping"] is False
        topology = clean["topology"]
        assert selected["clean_refit_check"] == topology and topology["exact_clean_refit_identity_except_scalar_leaf_outputs"] is True
        assert topology["leaf_outputs_ignored"] is True and topology["old_correction_subtracted"] is False
        updated = copy.deepcopy(job)
        updated.update(leaf_refit=None, binary_refit=None, multiclass_refit=None)
        field = "binary_refit" if problem == "binary" else "leaf_refit"
        updated[field] = choice["correction"]
        assert jobs[identity] == selected["fixed_job"] == updated
        assert selected["base_selection"] == bases[identity] and selected["independent_stop_selection"] == choice
        assert selected["seed"] == 47 and selected["source_wave"] == "17b" and selected["requested_rounds"] == job["rounds"]
        assert selected["temperature"] is selected["multiclass_refit"] is None
        assert selected[field] == choice["correction"]
        details = selected["refit_details"]
        clean_details = {k: v for k, v in details.items() if k not in ("scalar_refresh", "scalar_fitting_data")}
        assert clean_details == clean["fit"]
        if choice["correction"] is None:
            assert "scalar_refresh" not in details and "scalar_fitting_data" not in details
        else:
            correction, fitting = details["scalar_refresh"], details["scalar_fitting_data"]
            assert fitting["stop_excluded"] is False and fitting["row_order"] == "TRAIN, STOP, ascending outside-cap"
            assert fitting["rows"] == sum(fitting[k] for k in ("original_train_rows", "original_stop_rows", "outside_cap_rows"))
            assert correction["independent_sparse_raw_margin_oracle"] is True
            assert set(correction["checks"]) == {"fitting", "original_train", "original_stop"}
            assert all(check["leaf_routing_exact"] is True for check in correction["checks"].values())
            fold = correction["fold"]
            assert fold["topology_sha256_before"] == fold["topology_sha256_after"] == topology["nonleaf_state_sha256"]
            memory, solve = correction["memory"], correction["solve"]
            assert memory["rows"] == fitting["rows"]
            assert memory["refit_array_budget_bytes"] + memory.get("extra_certificate_vector_bytes", 0) <= plan["resources"]["maximum_named_array_bytes"]
            if problem == "regression":
                certificate = audit.certify(solve, choice["correction"]["lambda"], fitting["rows"])
            else:
                assert solve["penalty"] == choice["correction"]["lambda"] and solve["status"] == 0
                assert solve["maximum_absolute_mean_gradient"] <= solve["gradient_gate"] == plan["solvers"]["binary"]["gradient_gate"]
                certificate = {"status": solve["status"], "iterations": solve["iterations"]}
            correction_checks.append({"dataset": identity[0], "library": identity[1], "problem": problem, "solve": certificate})
        resource = read(OUT / "transfer_job_logs" / f"{identity[0]}-{identity[1]}-resource.json")
        assert resource["exit_code"] == 0 and resource["resource_failure"] is None and resource["cleanup_complete"] is True
        assert resource["peak_rss_bytes"] <= plan["resources"]["maximum_job_rss_bytes"]
        resources.append(resource)
        transfer_checks.append({"dataset": identity[0], "library": identity[1], "requested_rounds": job["rounds"],
                                "retained_rounds": details["retained_rounds"], "nonidentity": choice["correction"] is not None})
    assert len(transfer_checks) == 24 and len(carried) == 12 and len(correction_checks) == 15
    assert time(frozen["created_at"]) <= time(dev["created_at"]) and dev["official_test_scored"] is False
    rows = {r["dataset"]: r for r in dev["results"]}
    assert len(rows) == len(dev["results"]) == 12 and rows.keys() == cases.keys()
    artifact_root = ROOT / ".tmp/improvements-20261016/followup/accuracy_wave17b"
    for dataset, row in rows.items():
        assert row["problem_type"] == cases[dataset]["problem_type"]
        assert set(row["models"]) == {"ctboost", "xgboost", "lightgbm"}
        for library, model in row["models"].items():
            assert model["selection"] == selections[dataset, library]
            assert math.isfinite(model["error"]) and model["error"] >= 0
            assert digest(artifact_root / "development" / dataset / f"{library}.npz") == model["predictions_sha256"]
    recomputed = []
    for comparator in ("xgboost", "lightgbm"):
        for family in ("all", "binary", "regression", "multiclass"):
            chosen = [r for r in dev["results"] if family == "all" or r["problem_type"] == family]
            gains = [(r["models"][comparator]["error"] - r["models"]["ctboost"]["error"]) / max(r["models"][comparator]["error"], 1e-12) for r in chosen]
            recomputed.append({"comparator": comparator, "family": family, "datasets": len(chosen),
                "median_relative_error_gain": statistics.median(gains), "wins": sum(g > 1e-12 for g in gains),
                "ties": sum(abs(g) <= 1e-12 for g in gains), "worst_relative_regression": max(0, -min(gains))})
    assert recomputed == dev["summary"]
    passed = all(r["median_relative_error_gain"] >= 0 and r["wins"] + r["ties"] >= 6 for r in recomputed if r["family"] == "all")
    assert passed == dev["development_target_met"] == summary["development_target_met"]
    assert summary["new_clean_union_tree_fits_completed"] == summary["clean_parent_topology_checks_passed"] == 24
    assert summary["total_quality_tree_fits_through_wave17b"] == 2472 and summary["diagnostic_replay_fits_separate"] == 8
    assert summary["union_correction_solves_completed"] == summary["selected_nonzero_corrections"] == 15
    assert summary["all_stages_successful"] is True and summary["resource_failures"] == []
    assert summary["development_summary"] == recomputed
    assert summary["results"] == [{"dataset": r["dataset"], "problem_type": r["problem_type"],
        "errors": {k: v["error"] for k, v in r["models"].items()}} for r in dev["results"]]
    for filename, expected_hash in summary["source_sha256"].items():
        assert digest(OUT / filename) == expected_hash
    assert summary["report_source_sha256"] == digest(ROOT / "benchmarks/summarize_scalar_numerical_retry.py")
    evidence = {"status": "passed_closed_terminal_metadata_audit", "date": "2026-10-16",
        "scope": "Closed JSON metadata, source/model/archive byte hashes, and aggregation from already reported development errors only; no data/model/array decode, fitting, predictions or new scoring.",
        "selection_audit_sha256": digest(HERE / "wave17b_selection_independent_review.json"),
        "plan_sha256": digest(OUT / "plan.json"), "frozen_selections_sha256": digest(OUT / "frozen_selections.json"),
        "development_sha256": digest(OUT / "development.json"), "summary_sha256": digest(OUT / "summary.json"),
        "new_clean_union_model_training_runs": 24, "cumulative_benchmark_model_fits": 2472,
        "cumulative_fits_per_library": 824, "diagnostic_replay_fits_separate": 8,
        "carried_multiclass_exact": carried, "clean_union_checks": transfer_checks, "new_corrections": correction_checks,
        "development_aggregation_reconstructed": recomputed, "development_target_met": passed,
        "resource_failures": 0, "maximum_transfer_worker_seconds": max(r["wall_seconds"] for r in resources),
        "maximum_transfer_peak_rss_bytes": max(r["peak_rss_bytes"] for r in resources),
        "limitations": "Recorded structural/native-fold certificates are verified as metadata, not reproduced by loading models. Development errors are not rescored. Equal candidate counts do not imply equal compute. This development gate is exploratory, not a general ranking.",
        "audit_source_sha256": digest(__file__), "verified_file_sha256": audit.HASHES}
    target = HERE / "wave17b_terminal_independent_review.json"
    assert not target.exists()
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(target)
    print(json.dumps({k: evidence[k] for k in ("status", "new_clean_union_model_training_runs", "cumulative_benchmark_model_fits", "development_target_met")}))


if __name__ == "__main__":
    main()
