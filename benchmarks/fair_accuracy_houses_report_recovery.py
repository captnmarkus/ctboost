"""Recover only missing descriptive report text; preserve original predictions."""
from __future__ import annotations

import argparse
import copy
import importlib.util
from pathlib import Path

MAIN_PATH = Path(__file__).with_name("fair_accuracy_houses_cap_v2.py")
spec = importlib.util.spec_from_file_location("houses_frozen_main", MAIN_PATH)
main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main)
base = main.base


def report_plan(plan, policy):
    if "limitations" in plan:
        raise ValueError("Only the missing descriptive field may be recovered")
    return {**plan, "limitations": policy["limitations"]}


def run(args):
    import numpy as np
    plan = main.checked(args)
    main.prepared(args, plan)
    failure = base.read(args.output / "development_report_failed.json")
    if (failure["error"] != "KeyError('limitations')" or not failure["scoring_completed"]
            or failure["plan_sha256"] != base.digest(args.output / "plan.json")
            or failure["frozen_selections_sha256"] != base.digest(args.output / "frozen_selections.json")):
        raise ValueError("Expected the preserved report-only failure")
    frozen = base.read(args.output / "frozen_selections.json")
    if (not frozen["all_fits_successful"] or frozen["plan_sha256"] != base.digest(args.output / "plan.json")
            or frozen["prepared_data_sha256"] != base.digest(args.output / "prepared_data.json")):
        raise ValueError("All frozen models and prepared data must be intact")
    policy = base.read(main.POLICY)
    recovered = report_plan(plan, policy)
    marker = args.output / "report_recovery_started.json"
    if marker.exists() or (args.output / "development.json").exists():
        raise ValueError("Preserve every assessment attempt")
    originals = {}
    for case in plan["cases"]:
        for library in base.LIBRARIES:
            relative = Path("development") / case["dataset_name"] / (library + ".npz")
            path = args.artifacts / relative
            originals[str(relative)] = {"path": str(path), "sha256": base.digest(path)}
    receipt = {"created_at": base.now(), "script_sha256": base.digest(__file__),
        "main_script_sha256": base.digest(MAIN_PATH), "policy_sha256": base.digest(main.POLICY),
        "plan_sha256": base.digest(args.output / "plan.json"),
        "frozen_selections_sha256": base.digest(args.output / "frozen_selections.json"),
        "failure_receipt_sha256": base.digest(args.output / "development_report_failed.json"),
        "original_error": "KeyError('limitations') after scoring and before report serialization",
        "original_prediction_artifacts": originals,
        "change": "Supply only frozen policy limitations text to the checked in-memory plan; identical scorer/models/metrics",
        "new_fits": 0, "new_selection": False}
    base.write(marker, receipt)
    recovery = copy.copy(args)
    recovery.artifacts = args.artifacts / "report_recovery"
    if recovery.artifacts.exists():
        raise ValueError("Preserve prior recovery artifacts")
    original_write, reports = base.write, []
    def capture(path, value):
        if Path(path) != args.output / "development.json":
            raise ValueError("Unexpected scorer write")
        reports.append(value)
    base.checked = lambda supplied: recovered if supplied is recovery else main.checked(supplied)
    base.predict, base.write = main.predict, capture
    try:
        base.development(recovery)
    finally:
        base.write = original_write
    if len(reports) != 1:
        raise ValueError("Expected one unchanged scorer report")
    for relative, source in originals.items():
        if base.digest(source["path"]) != source["sha256"]:
            raise ValueError("Original failed-assessment predictions changed")
        with np.load(source["path"], allow_pickle=False) as first, np.load(recovery.artifacts / relative, allow_pickle=False) as retry:
            if set(first.files) != set(retry.files):
                raise ValueError("Prediction archive fields changed")
            for key in first.files:
                if first[key].dtype != retry[key].dtype or first[key].shape != retry[key].shape or first[key].tobytes() != retry[key].tobytes():
                    raise ValueError("Reporting recovery changed predictions/labels/IDs")
    completed = {**receipt, "completed_at": base.now(), "all_36_prediction_archives_bit_exact": True,
        "recovery_artifacts": str(recovery.artifacts), "started_receipt_sha256": base.digest(marker)}
    base.write(args.output / "report_recovery_completed.json", completed)
    reports[0]["report_recovery"] = {"script_path": str(Path(__file__).resolve()), "script_sha256": base.digest(__file__),
        "receipt_sha256": base.digest(args.output / "report_recovery_completed.json"), "predictions_bit_exact": True}
    base.write(args.output / "development.json", reports[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "artifacts", "package-root", "dependencies", "previous"):
        parser.add_argument("--" + name, type=lambda value: Path(value).resolve(), required=True)
    run(parser.parse_args())
