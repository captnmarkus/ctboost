"""Reporting-only summary of the closed external-interruption recovery."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "accuracy_wave18b_recovery"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    recovery = read(OUT / "recovery_plan.json")
    completion = read(OUT / "recovery_completion.json")
    fold_completion = read(OUT / "fold_completion.json")
    plan = read(OUT / "plan.json")
    folds = [read(path) for path in sorted((OUT / "fold_jobs").glob("*.json"))]
    unions = [read(path) for path in sorted((OUT / "union_jobs").glob("*.json"))]
    first_new = min(recovery["fresh_global_indices"])
    fresh_folds = folds[first_new:]
    native_count = lambda rows: sum(row.get("native_fit_completed") is True for row in rows)
    prior18 = read(ROOT / "accuracy_wave18/summary.json")
    interrupted = read(ROOT / "accuracy_wave18b/summary.json")
    stage_resources = {}
    for stage in ("fold", "union"):
        paths = sorted((OUT / f"{stage}_job_logs").glob("*-resource.json"))
        stage_resources[stage] = [read(path) for path in paths if stage == "union" or int(path.name[:4]) >= first_new]
    resources = [row for rows in stage_resources.values() for row in rows]
    files = [name for name in ("plan.json", "fold_manifest.json", "recovery_plan.json", "recovery_completion.json",
        "fold_completion.json", "selected_cv.json", "frozen_selections.json", "development.json") if (OUT / name).exists()]
    summary = {"status": "folds_complete" if completion["all_fits_successful"] else "terminal_recovery_failure",
        "source_sha256": {name: digest(OUT / name) for name in files}, "report_source_sha256": digest(__file__),
        "scientific_plan_unchanged": digest(OUT / "plan.json") == digest(ROOT / "accuracy_wave18b/plan.json"),
        "reused_verified_fold_jobs": first_new, "new_fold_process_counts": completion["new_process_counts"],
        "scientific_fold_completion_counts": fold_completion["completion_counts"],
        "new_union_fits_completed": native_count(unions),
        "fully_verified_union_jobs": sum(row.get("status") == "ok" for row in unions),
        "extra_original_interrupted_native_fit": 1,
        "total_quality_tree_fits": interrupted["total_quality_tree_fits_through_wave18b"] + native_count(fresh_folds) + native_count(unions),
        "diagnostic_replay_fits_separate": 8,
        "original18_resource_failure": {"native_fits_completed": prior18["fold_completion_counts"]["native_fits_completed"],
            "summary_sha256": digest(ROOT / "accuracy_wave18/summary.json")},
        "original18b_external_interruption": {"native_fits_completed": interrupted["fold_completion_counts"]["native_fits_completed"],
            "fully_verified_jobs": interrupted["fold_completion_counts"]["fully_verified_jobs"],
            "summary_sha256": digest(ROOT / "accuracy_wave18b/summary.json"),
            "incident_sha256": digest(ROOT / "release_preparation/wave18b_external_interruption_incident.json"),
            "cause": "A reviewing agent mistakenly terminated the owned worker. This was not a scientific or watchdog failure."},
        "terminal_failures": [row for row in fresh_folds + unions if row.get("status") != "ok"
            and row.get("failure") != "not_started_after_terminal_failure"],
        "resources": plan["resources"],
        "maximum_new_job_seconds": max((row["wall_seconds"] for row in resources), default=0),
        "maximum_new_process_tree_rss_bytes": max((row["peak_rss_bytes"] for row in resources), default=0),
        "new_job_seconds": {stage: sum(row["wall_seconds"] for row in rows) for stage, rows in stage_resources.items()},
        "new_resource_failures": [row for row in resources if row["resource_failure"] is not None],
        "timing_interpretation": "Diagnostic process costs only. Some workers overlapped separately bounded release correctness/build work on other CPUs; no isolated training or inference-speed claim.",
        "development_scored": False, "development_target_met": None, "conditional_seed677_run": False,
        "official_test_scored": False, "limitations": plan["limitations"]}
    if (OUT / "selected_cv.json").exists():
        choices = read(OUT / "selected_cv.json")["choices"]
        anchors = {(job["dataset"], job["library"]): job for job in plan["jobs"]}
        retained = {(row["dataset"], row["library"]): row["selection"]["best"]["retained_rounds"]
                    for row in unions if row["status"] == "ok"}
        summary["selected_rounds"] = [{"dataset": row["dataset"], "library": row["library"],
            "anchor_requested_rounds": anchors[row["dataset"], row["library"]]["rounds"],
            "selected_requested_rounds": row["best"]["requested_rounds"],
            "selected_profile_index": row["best"]["profile_index"], "profile_scores": row["scores"],
            "native_retained_rounds": retained.get((row["dataset"], row["library"]))} for row in choices]
        summary["round_profiles_by_library"] = {library: {name: sum(row["library"] == library
            and row["best"]["profile_index"] == index for row in choices)
            for index, name in enumerate(("half", "same", "double"))} for library in ("ctboost", "xgboost", "lightgbm")}
    if (OUT / "frozen_selections.json").exists():
        frozen = read(OUT / "frozen_selections.json")
        summary.update(status="models_frozen" if frozen["all_fits_successful"] else "terminal_union_failure",
                       all_stages_successful=frozen["all_fits_successful"])
        summary["policy_payload_bytes"] = {library: sum(row["model_bytes"] for row in frozen["selections"]
            if row["library"] == library) for library in ("ctboost", "xgboost", "lightgbm")}
    if (OUT / "development.json").exists():
        development = read(OUT / "development.json")
        summary.update(status="complete_development_assessment", development_scored=True,
            development_target_met=development["development_target_met"], development_summary=development["summary"],
            results=[{"dataset": row["dataset"], "problem_type": row["problem_type"],
                "errors": {library: value["error"] for library, value in row["models"].items()}}
                for row in development["results"]])
    path = OUT / "summary.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    print(json.dumps({key: summary[key] for key in ("status", "reused_verified_fold_jobs", "new_fold_process_counts",
        "new_union_fits_completed", "total_quality_tree_fits", "development_target_met")}, indent=2))


if __name__ == "__main__":
    main()
