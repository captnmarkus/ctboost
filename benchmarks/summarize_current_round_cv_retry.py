"""Summarize closed wave 18b receipts without fitting, predicting or selection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

OUT = Path(__file__).parent / "results/improvements_20261016_followup/accuracy_wave18b"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    plan = read(OUT / "plan.json")
    folds = read(OUT / "fold_completion.json")
    fold_rows = [read(path) for path in sorted((OUT / "fold_jobs").glob("*.json"))]
    union_rows = [read(path) for path in sorted((OUT / "union_jobs").glob("*.json"))]
    resources = {stage: [read(path) for path in sorted((OUT / f"{stage}_job_logs").glob("*-resource.json"))]
                 for stage in ("fold", "union")}
    headroom = [read(path) for stage in ("fold", "union")
                for path in sorted((OUT / f"{stage}_job_logs").glob("*-headroom.json"))]
    all_resources = [row for rows in resources.values() for row in rows]
    paths = [name for name in ("plan.json", "fold_manifest.json", "fold_completion.json", "selected_cv.json",
                               "frozen_selections.json", "development.json") if (OUT / name).exists()]
    count = lambda rows, key: sum(row.get(key) is True for row in rows)
    failures = [{key: row.get(key) for key in ("dataset", "library", "profile_index", "fold", "failure", "error", "resource")}
                for row in fold_rows + union_rows if row["status"] != "ok" and row.get("failure") != "not_started_after_terminal_failure"]
    summary = {
        "status": "folds_complete" if folds["all_fits_successful"] else "terminal_fold_failure",
        "source_sha256": {name: digest(OUT / name) for name in paths}, "report_source_sha256": digest(__file__),
        "fold_completion_counts": folds["completion_counts"],
        "union_completion_counts": {"planned_jobs": 36, "native_fits_started": count(union_rows, "native_fit_started"),
            "native_fits_completed": count(union_rows, "native_fit_completed"),
            "fully_verified_jobs": sum(row["status"] == "ok" for row in union_rows)},
        "total_quality_tree_fits_through_wave18b": 2530 + count(fold_rows + union_rows, "native_fit_completed"),
        "diagnostic_replay_fits_separate": 8, "terminal_failures": failures,
        "prior_resource_failed_attempt": {"directory": "accuracy_wave18", "native_fits_completed": 58,
            "new_attempt_reuses_fits": False,
            "fold_completion_sha256": digest(OUT.parent / "accuracy_wave18/fold_completion.json"),
            "limits": "All 540 folds are fitted afresh under the new resource plan. The 58 earlier fits remain preserved and are included separately in cumulative compute counts."},
        "not_started_after_terminal_failure": sum(row.get("failure") == "not_started_after_terminal_failure" for row in fold_rows + union_rows),
        "maximum_job_seconds": max((row["wall_seconds"] for row in all_resources), default=0),
        "maximum_process_tree_rss_bytes": max((row["peak_rss_bytes"] for row in all_resources), default=0),
        "sum_stage_job_seconds": {stage: sum(row["wall_seconds"] for row in rows) for stage, rows in resources.items()},
        "resource_failures": [row for row in all_resources if row["resource_failure"] is not None],
        "headroom_failures": [row for row in headroom if not row["sufficient"]],
        "maximum_headroom_wait_seconds": max((row["samples"][-1]["elapsed_seconds"] for row in headroom), default=0),
        "quiet_baseline_memory": plan["quiet_baseline_memory"], "resources": plan["resources"],
        "prospective_profiles_checked": sum(row["profiles_checked"] for row in plan["memory_bounds"]),
        "maximum_prospective_named_array_bytes": max(row["maximum_named_bound_bytes"] for row in plan["memory_bounds"]),
        "timing_interpretation": "Diagnostic process durations including fitting, fixed corrections, native verification and persistence; no inference-speed or equal-compute claim.",
        "concurrent_validation": {
            "disclosure": "Some fold jobs overlapped separately bounded release correctness/build work on CPU0. Accuracy workers remained on CPU4 with their frozen resources; durations are not isolated timing measurements.",
            "jvm_receipt": "release_preparation/security/jvm_validation_summary.json",
            "jvm_receipt_sha256": digest(OUT.parent / "release_preparation/security/jvm_validation_summary.json"),
        },
        "development_scored": False, "development_target_met": None, "conditional_seed677_run": False,
        "official_test_scored": False, "limitations": plan["limitations"],
    }
    if (OUT / "selected_cv.json").exists():
        selected = read(OUT / "selected_cv.json")
        anchors = {(j["dataset"], j["library"]): j for j in plan["jobs"]}
        selections = {(r["dataset"], r["library"]): r.get("selection") for r in union_rows if r["status"] == "ok"}
        choices = []
        for row in selected["choices"]:
            key = row["dataset"], row["library"]
            model = selections.get(key)
            choices.append({"dataset": row["dataset"], "library": row["library"],
                "anchor_requested_rounds": anchors[key]["rounds"], "selected_requested_rounds": row["best"]["requested_rounds"],
                "selected_profile_index": row["best"]["profile_index"], "profile_scores": row["scores"],
                "native_retained_rounds": model["best"]["retained_rounds"] if model else None})
        summary["selected_rounds"] = choices
        summary["round_profiles_by_library"] = {library: {name: sum(row["library"] == library and row["selected_profile_index"] == index
            for row in choices) for index, name in enumerate(("half", "same", "double"))}
            for library in ("ctboost", "xgboost", "lightgbm")}
    if (OUT / "frozen_selections.json").exists():
        frozen = read(OUT / "frozen_selections.json")
        summary.update(status="models_frozen" if frozen["all_fits_successful"] else "terminal_union_failure",
            all_stages_successful=frozen["all_fits_successful"], union_completion_counts=frozen["union_completion_counts"])
        summary["libraries"] = {library: {
            "new_native_fold_fits_completed": count([row for row in fold_rows if row["library"] == library], "native_fit_completed"),
            "new_native_union_fits_completed": count([row for row in union_rows if row["library"] == library], "native_fit_completed"),
            "total_policy_payload_bytes": sum(row["model_bytes"] for row in frozen["selections"] if row["library"] == library)}
            for library in ("ctboost", "xgboost", "lightgbm")}
    if (OUT / "development.json").exists():
        development = read(OUT / "development.json")
        summary.update(status="complete_development_assessment", development_scored=True,
            development_target_met=development["development_target_met"], development_summary=development["summary"],
            results=[{"dataset": row["dataset"], "problem_type": row["problem_type"],
                      "errors": {library: value["error"] for library, value in row["models"].items()}}
                     for row in development["results"]])
    target = OUT / "summary.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(target)
    print(json.dumps({key: summary[key] for key in ("status", "fold_completion_counts", "union_completion_counts",
        "total_quality_tree_fits_through_wave18b", "development_target_met")}, indent=2))


if __name__ == "__main__":
    main()
