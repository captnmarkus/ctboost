"""Report wave 17b saved receipts without fitting, predicting or selecting."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

OUT = Path(__file__).parent / "results/improvements_20261016_followup/accuracy_wave17b"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    plan = read(OUT / "plan.json")
    controls = read(OUT / "scalar_controls.json")
    penalties = [read(p) for p in (OUT / "penalty_fits").glob("*/*/*/job.json")]
    transfers = [read(p) for p in (OUT / "transfer_fits").glob("*/*/job.json")]
    clean = [read(p) for p in (OUT / "transfer_fits").glob("*/*/clean_fit.json")]
    resources = {stage: [read(p) for p in (OUT / directory).glob("*-resource.json")]
                 for stage, directory in (("selection", "penalty_logs"), ("transfer", "transfer_job_logs"))}
    all_resources = [r for rows in resources.values() for r in rows]
    candidates = [c for r in penalties for c in r.get("candidates", [])]
    memories = [r["memory"] for r in penalties if "memory" in r]
    transfer_corrections = [r["selection"]["refit_details"]["scalar_refresh"] for r in transfers
                           if "scalar_refresh" in r.get("selection", {}).get("refit_details", {})]
    memories.extend(r["memory"] for r in transfer_corrections)
    solves = [r["solve"] for r in penalties if "solve" in r]
    sources = [name for name in ("plan.json", "synthetic_validation.json", "scalar_controls.json", "frozen_selections.json", "development.json")
               if (OUT / name).exists()]
    summary = {
        "status": "terminal_selection_failure" if not controls["all_fits_successful"] else "selection_complete",
        "source_sha256": {name: digest(OUT / name) for name in sources},
        "report_source_sha256": digest(__file__),
        "new_train_anchor_fits": 0, "reused_clean_train_anchors": 24,
        "reused_binary_grids": 12, "reused_binary_solves": 48,
        "reused_binary_nonidentity_evaluations": 96, "unique_identity_candidates": 24,
        "new_regression_penalty_jobs_planned": 48,
        "new_regression_penalty_jobs_successful": sum(r["status"] == "ok" for r in penalties),
        "new_regression_penalty_jobs_failed": sum(r["status"] != "ok" for r in penalties),
        "new_regression_certified_solves_completed": sum(r["success"] for r in solves),
        "new_nonidentity_evaluations_planned": 96,
        "new_nonidentity_evaluations_successful": sum(c["status"] == "ok" for c in candidates),
        "new_nonidentity_evaluations_failed": sum(c["status"] != "ok" for c in candidates),
        "repeated_regression_identity_checks": controls["regression_identity_prediction_checks"],
        "failed_penalty_jobs": [{key: r.get(key) for key in ("dataset", "library", "penalty_index", "error", "resource")}
                                for r in penalties if r["status"] != "ok"],
        "new_clean_union_tree_fits_planned": 24, "new_clean_union_tree_fits_completed": len(clean),
        "clean_parent_topology_checks_passed": sum(r["status"] == "ok" for r in clean),
        "selected_nonzero_corrections": sum(c["correction"] is not None for c in controls["choices"]),
        "union_correction_solves_completed": len(transfer_corrections),
        "total_quality_tree_fits_through_wave17b": 2448 + len(clean), "diagnostic_replay_fits_separate": 8,
        "maximum_job_seconds": max((r["wall_seconds"] for r in all_resources), default=0),
        "maximum_process_tree_rss_bytes": max((r["peak_rss_bytes"] for r in all_resources), default=0),
        "sum_stage_job_seconds": {stage: sum(r["wall_seconds"] for r in rows) for stage, rows in resources.items()},
        "maximum_named_array_estimate_with_certificate_bytes": max(
            (r["refit_array_budget_bytes"] + r.get("extra_certificate_vector_bytes", 0) for r in memories), default=0),
        "resource_failures": [r for r in all_resources if r["resource_failure"] is not None],
        "timing_interpretation": "Diagnostic process durations only. A separate CP310 production correctness suite ran on other CPUs during part of the union stage; these are not isolated speed benchmarks or equal-compute comparisons.",
        "selected_controls": controls["choices"], "development_scored": False,
        "development_target_met": None, "conditional_seed569_run": False, "official_test_scored": False,
        "limitations": plan["limitations"],
    }
    if (OUT / "frozen_selections.json").exists():
        frozen = read(OUT / "frozen_selections.json")
        summary.update(all_stages_successful=frozen["all_fits_successful"],
                       exact_carried_multiclass_models=frozen["carried_models"],
                       status="models_frozen" if frozen["all_fits_successful"] else "terminal_transfer_failure")
        summary["libraries"] = {lib: {
            "new_train_fits": 0, "reused_train_anchors": 8,
            "new_union_tree_fits_completed": len(list((OUT / "transfer_fits").glob(f"*/{lib}/clean_fit.json"))),
            "selected_nonzero": sum(c["library"] == lib and c["correction"] is not None for c in controls["choices"]),
            "total_policy_payload_bytes": sum(s["model_bytes"] for s in frozen["selections"] if s["library"] == lib),
        } for lib in ("ctboost", "xgboost", "lightgbm")}
    if (OUT / "development.json").exists():
        dev = read(OUT / "development.json")
        summary.update(status="complete_development_assessment", development_scored=True,
            development_target_met=dev["development_target_met"], development_summary=dev["summary"],
            results=[{"dataset": row["dataset"], "problem_type": row["problem_type"],
                      "errors": {lib: value["error"] for lib, value in row["models"].items()}} for row in dev["results"]])
    target = OUT / "summary.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(target)
    print(json.dumps({key: summary[key] for key in ("status", "new_regression_penalty_jobs_successful",
        "new_clean_union_tree_fits_completed", "total_quality_tree_fits_through_wave17b", "development_target_met")}, indent=2))


if __name__ == "__main__":
    main()
