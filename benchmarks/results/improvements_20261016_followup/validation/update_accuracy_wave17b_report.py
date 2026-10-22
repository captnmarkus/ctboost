"""Atomic narrative update from closed, independently audited wave-17b JSON."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
ROOT = HERE.parents[3]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    summary_path = REPORT / "accuracy_wave17b/summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    audit_path = HERE / "wave17b_terminal_independent_review.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["summary_sha256"] == sha(summary_path) and audit["status"] == "passed_closed_terminal_metadata_audit"
    path = REPORT / "accuracy.md"
    before = sha(path)
    backup = ROOT / ".tmp/improvements-20261016/accuracy.before_wave17b_terminal.md"
    assert not backup.exists()
    backup.write_bytes(path.read_bytes())
    text = path.read_text(encoding="utf-8")
    text = text.replace("| 15 | Houses training-data expansion |", "| 15b | Houses training-data expansion |")
    marker = "| 16 | Fixed multiclass leaf correction | -0.143% | 6/12 | -1.439% | 6/12 |"
    assert marker in text
    overall = {r["comparator"]: r for r in summary["development_summary"] if r["family"] == "all"}
    row = (f"| 17b | Certified scalar correction refresh | {overall['xgboost']['median_relative_error_gain']*100:+.3f}% | "
           f"{overall['xgboost']['wins']}/12 | {overall['lightgbm']['median_relative_error_gain']*100:+.3f}% | {overall['lightgbm']['wins']}/12 |")
    text = text.replace(marker, marker + "\n" + row)
    text = text.replace("assigned to wave 17. Wave 16 remains the latest complete evaluated policy.",
                        "assigned to wave 17. At that point, wave 16 remained the latest complete evaluated policy.")
    text = text.replace("resource receipts, not isolated speed measurements. Cumulative work is now\n**2,448", 
                        "resource receipts, not isolated speed measurements. At that stage, cumulative work was\n**2,448")
    text = text.replace("These are primary seed-47 errors of the complete wave-16 policy.",
                        "These are primary seed-47 errors of the complete wave-17b policy.")
    text = text.replace("`accuracy_wave16/frozen_selections.json`; original source-wave records remain", 
                        "`accuracy_wave17b/frozen_selections.json`; original source-wave records remain")
    start = text.index("| Dataset | Error metric | CTBoost | XGBoost | LightGBM |")
    end = text.index("\n\nAll prior per-task errors", start)
    lines = ["| Dataset | Error metric | CTBoost | XGBoost | LightGBM |", "|---|---|---:|---:|---:|"]
    metrics = {"binary": "1-AUC", "regression": "RMSE", "multiclass": "Logloss"}
    for result in summary["results"]:
        e = result["errors"]
        lines.append(f"| {result['dataset']} | {metrics[result['problem_type']]} | {e['ctboost']:.9f} | {e['xgboost']:.9f} | {e['lightgbm']:.9f} |")
    text = text[:start] + "\n".join(lines) + text[end:]
    text = text.replace("457 for wave 16, and 569 for wave 17.", "457 for wave 16, and 569 for waves 17/17b.")
    text = text.replace("`fair_accuracy_scalar_refresh.py` (17). All are under `benchmarks/`.",
                        "`fair_accuracy_scalar_refresh.py` (17), and\n`fair_accuracy_scalar_numerical_retry.py` (17b). All are under `benchmarks/`.")
    text = text.replace("[latest full scores](accuracy_wave16/development.json), [latest summary](accuracy_wave16/summary.json)",
                        "[latest full scores](accuracy_wave17b/development.json), [latest summary](accuracy_wave17b/summary.json)")
    start = text.index("## Wave 17b: separately frozen numerical retry (in progress)")
    text = text[:start].rstrip() + "\n"
    section = """### Wave 17b: certified numerical retry and completed scalar refresh

The original wave 17 remains a terminal numerical failure. A separate TRAIN-only
extended diagnostic certified zero-start, unscaled LSQR at 5,541 iterations
(with a prospectively fixed 10,000 cap), relative original-objective gap
1.8320e-14 and normalized original gradient 2.1776e-9. It performed no new
model fit or STOP/DEV score. Its success established numerical feasibility,
not an accuracy gain.

The separately frozen [17b plan](accuracy_wave17b/plan.json) reuses all 24 clean
TRAIN anchors and 12 successful binary grids from the failed wave. All 48 new
regression penalty solves certify, and all 96 new native-folded candidate
evaluations pass. The original 96 binary evaluations remain exact reuses.
All 24 STOP choices freeze before the 24 mandatory clean union refits; 15 select
a nonzero correction and nine select clean identity. Prior scalar corrections
are disabled before every fresh union fit. All 24 recorded nonleaf/schema
checks and final persistence checks pass, followed by 15 fixed correction
solves. Twelve wave-16 multiclass artifacts carry byte-for-byte unchanged.

The [selection audit](validation/wave17b_selection_independent_review.json)
reconstructs every strict choice, including all 108 binary identity/candidate
STOP metrics from saved archives in the exact frozen metric environment.
The [terminal audit](validation/wave17b_terminal_independent_review.json)
verifies sources, closed receipts, model/archive hashes, chronology and all
development aggregates. Regression certificates are checked for recorded
arithmetic/threshold consistency, not rebuilt from the design matrices.

The completed gate still fails: CTBoost's median relative error gain is
**+0.286104% against XGBoost (7/12 wins)** and **-1.358863% against LightGBM
(5/12 wins)**. All errors come from the frozen single seed-47 policies; there
is no DEV-selected fallback. Conditional seed 569 and official test remain
unopened. The same immutable c5 accuracy runtime is retained; production CTR
precision changes are not part of this comparison.

The 24 new union fits bring cumulative work to **2,472 benchmark model fits**,
824 per library, plus eight separate diagnostic replays. Anchor reuse and
coefficient solves are not counted as new model fits. Maximum worker duration
is 445.578 s, peak summed process-tree RSS 2.434 GB, and maximum named-array
estimate 2.264 GB, with no resource failures. Selection and transfer worker
durations sum to 1,327.749 s and 268.909 s. These are diagnostic durations:
a separate production correctness suite overlapped part of the union stage.
Policy payload sizes are CTBoost 83.849 MB, XGBoost 73.149 MB and LightGBM
12.217 MB. Exact controls, costs and all twelve errors are retained in the
[closed summary](accuracy_wave17b/summary.json).

"""
    text = text.replace("## Latest complete task errors and controls", section + "## Latest complete task errors and controls")
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    receipt = {"status": "report_updated_from_closed_audited_metadata", "date": "2026-10-16",
        "accuracy_before_sha256": before, "accuracy_after_sha256": sha(path), "backup": str(backup),
        "summary_sha256": sha(summary_path), "terminal_audit_sha256": sha(audit_path),
        "source_sha256": sha(__file__), "model_training_runs": 2472,
        "all12_latest_error_rows_updated": True, "wave13_all36_historical_count_table_preserved": True,
        "no_fitting_prediction_or_new_scoring": True}
    target = HERE / "accuracy_report_wave17b_review.json"
    assert not target.exists()
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
