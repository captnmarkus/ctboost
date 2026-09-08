"""Explicit, input-dtype-only amendment for the remaining-task comparator.

The original failed XGBoost fit and original gate remain unchanged. All tasks
with plain boolean input columns are enumerated before this amendment fits any
model; only those XGBoost arms receive bool-to-uint8 compatibility conversion.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path


def bootstrap():
    import ctboost
    root = Path(__file__).resolve().parents[1]
    if Path(ctboost.__file__).resolve().is_relative_to(root / "ctboost"):
        raise RuntimeError("Use the public-wheel Python with -I")
    sys.path.insert(0, str(root))
    from benchmarks import accuracy_grouped_scout as base
    base.bootstrap()
    return base


def prepare(output, predecessor):
    base = bootstrap()
    import numpy as np

    if output.exists():
        raise ValueError("Preserve earlier attempts; use a new amendment directory")
    original = base.checked_plan(predecessor)
    affected, evidence = [], []
    for case in original["datasets"]:
        roles = {role: base.load_role(predecessor, case, role) for role in ("train", "stop", "development")}
        columns = [name for name, dtype in roles["train"]["X"].dtypes.items() if dtype == np.dtype(bool)]
        evidence.append({"dataset": case["dataset_name"], "plain_boolean_columns": columns})
        if not columns:
            continue
        previous = base.read(predecessor / "fits" / case["dataset_name"] / "xgboost_ag_default/result.json")
        if previous["status"] not in ("ok", "failed"):
            raise ValueError("Affected original fit must be terminal before correction")
        corrected = {**case, "roles": {}, "boolean_columns_to_uint8": columns}
        for role, payload in roles.items():
            frame = payload["X"].copy()
            original_columns = list(frame.columns)
            for name in columns:
                if frame[name].dtype != np.dtype(bool):
                    raise ValueError("Boolean schema differs across frozen row roles")
                before = frame[name].to_numpy(copy=True)
                frame[name] = frame[name].astype(np.uint8)
                np.testing.assert_array_equal(frame[name].to_numpy().astype(bool), before)
                np.testing.assert_array_equal(frame[name].to_numpy(), before.astype(np.uint8))
            if list(frame.columns) != original_columns:
                raise ValueError("Compatibility conversion changed input features")
            target = output / "data" / case["dataset_name"] / (role + ".pkl")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as stream:
                pickle.dump({**payload, "X": frame}, stream, protocol=5)
            corrected["roles"][role] = {**case["roles"][role], "sha256": base.digest(target),
                                       "original_sha256": case["roles"][role]["sha256"]}
        affected.append(corrected)
    plan = {**original, "protocol_id": "remaining37_xgb_bool_compatibility_20260908_v1",
            "created_at": base.now(), "predecessor": str(predecessor),
            "predecessor_plan_sha256": base.digest(predecessor / "plan.json"),
            "source_sha256": {**original["source_sha256"], "benchmarks/accuracy_xgb_bool_amendment.py": base.digest(__file__)},
            "amendment_runner_sha256": base.digest(__file__), "datasets": affected,
            "all_task_dtype_inventory": evidence, "expected_fits": len(affected), "arms": ["xgboost_ag_default"],
            "compatibility_rule": "For every task with plain numpy bool columns, convert those columns to uint8 (exact0/1) in training, stopping, and prediction frames. Preserve all columns, categories, missingness in other columns, row IDs, targets, resources, seed, and XGBoost defaults.",
            "reason": "AG OheFeatureGenerator sends mixed bool/numeric frames as object arrays to scipy.csr_matrix; it rejects object dtype. This correction is input representation only, not a CTBoost quality change.",
            "selection": "Every plain-bool-affected task among all37, regardless of original score or success. No labels or outcomes determine affected membership.",
            "reuse_policy": "Amended111-fit analysis reuses every unchanged CTBoost arm and every unaffected XGBoost arm by record/model/prediction hashes. All affected XGBoost arms receive exactly one separately recorded corrected attempt, even if an original had succeeded. Original records and failed original all-success gate remain intact.",
            "amended_gate": original["gate"], "original_expected_fits": 111,
            "resources": {**original["resources"], "max_workers": 1},
            "reuse_disclosure": "Comparator compatibility amendment based solely on input dtypes, after original XGBoost preprocessing failure. Original37-task assessment remains a separate failed-gate run. Unaffected model outputs are reused explicitly; old14-task confirmation remains unscored. No Elo claim."}
    base.write(output / "plan.json", plan)
    print(json.dumps({"enumerated_tasks": len(evidence), "affected": [c["dataset_name"] for c in affected],
                      "plan_sha256": base.digest(output / "plan.json")}))


def run(output):
    base = bootstrap()
    import psutil

    plan = base.checked_plan(output)
    env = dict(os.environ, CTBOOST_HIST_THREADS="2", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    for case in plan["datasets"]:
        directory = output / "fits" / case["dataset_name"] / "xgboost_ag_default"
        if directory.exists():
            raise ValueError("Corrected fit already started; no automatic retries")
        command = [sys.executable, "-I", str(Path(base.__file__).resolve()), "fit", "--output", str(output),
                   "--dataset", case["dataset_name"], "--arm", "xgboost_ag_default", "--slot", "0"]
        log = output / "logs" / (case["dataset_name"] + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        peak, reason, started = 0, None, time.monotonic()
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            owner = psutil.Process(process.pid)
            while process.poll() is None:
                try:
                    peak = max(peak, sum(p.memory_info().rss for p in [owner] + owner.children(recursive=True)))
                except psutil.NoSuchProcess:
                    pass
                if peak > plan["resources"]["rss_bytes"]:
                    reason = "memory_limit"
                elif time.monotonic() - started > plan["resources"]["hard_wall_seconds"]:
                    reason = "hard_timeout"
                if reason:
                    for child in owner.children(recursive=True):
                        child.kill()
                    owner.kill()
                    process.wait(timeout=15)
                    break
                time.sleep(.25)
        result_path = directory / "result.json"
        result = base.read(result_path) if result_path.exists() else {"dataset": case["dataset_name"],
                  "arm": "xgboost_ag_default", "status": "failed", "plan_sha256": base.digest(output / "plan.json")}
        result.update(peak_rss_bytes=peak, exit_code=process.returncode, resource_failure=reason)
        if reason or process.returncode != 0:
            result["status"] = "failed"
        if result["status"] == "ok":
            with (directory / "model.pkl").open("rb") as stream:
                model = pickle.load(stream)["model"]
            retained = model._ohe_generator.get_original_feature_names()
            if not set(case["boolean_columns_to_uint8"]).issubset(retained):
                raise ValueError("Corrected boolean features were dropped")
            result.update(boolean_features_retained=True, retained_features=retained,
                          prediction_input_conversion={name: "uint8" for name in case["boolean_columns_to_uint8"]})
        base.write(result_path, result)
        print(json.dumps({"dataset": case["dataset_name"], "status": result["status"], "fit_seconds": result.get("fit_seconds")}))


def report(output):
    base = bootstrap()
    import numpy as np

    amendment = base.checked_plan(output)
    original_root = Path(amendment["predecessor"])
    original = base.checked_plan(original_root)
    if base.digest(original_root / "plan.json") != amendment["predecessor_plan_sha256"]:
        raise ValueError("Original plan changed")
    affected = {case["dataset_name"] for case in amendment["datasets"]}
    rows, ledger, failures, ratios = [], [], [], []
    for case in original["datasets"]:
        errors, paired = {}, {}
        data = base.load_role(original_root, case, "development")
        for arm in base.ARMS:
            corrected = case["dataset_name"] in affected and arm == "xgboost_ag_default"
            source = output if corrected else original_root
            directory = source / "fits" / case["dataset_name"] / arm
            record = base.read(directory / "result.json")
            if record["plan_sha256"] != base.digest(source / "plan.json"):
                raise ValueError("Record plan mismatch")
            paired[arm] = record
            ledger.append({"dataset": case["dataset_name"], "arm": arm, "source": "amendment" if corrected else "original",
                           "record_sha256": base.digest(directory / "result.json"), "model_sha256": record.get("model_sha256"),
                           "prediction_sha256": record.get("development_sha256"), "plan_sha256": record["plan_sha256"]})
            if record["status"] != "ok" or record.get("resource_failure"):
                failures.append(record)
                continue
            if base.digest(directory / "model.pkl") != record["model_sha256"] or base.digest(directory / "development.npz") != record["development_sha256"]:
                raise ValueError("Artifact changed")
            with np.load(directory / "development.npz") as saved:
                np.testing.assert_array_equal(saved["labels"], data["y"])
                np.testing.assert_array_equal(saved["outer_train_positions"], data["outer_train_positions"])
                error = base.score_predictions(saved["labels"], saved["predictions"], case)
                if abs(error - record["development_error"]) > 1e-12:
                    raise ValueError("Metric changed")
            errors[arm] = error
        row = {"dataset": case["dataset_name"], "problem_type": case["problem_type"], "errors": errors}
        if len(errors) == 3:
            row["candidate_relative_gain"] = (errors[base.ARMS[0]] - errors[base.ARMS[1]]) / max(errors[base.ARMS[0]], 1e-12)
            row["candidate_vs_xgboost_gain"] = (errors[base.ARMS[2]] - errors[base.ARMS[1]]) / max(errors[base.ARMS[2]], 1e-12)
            ratios.append(paired[base.ARMS[1]]["fit_seconds"] / max(paired[base.ARMS[0]]["fit_seconds"], 1e-12))
        if case["dataset_name"] in affected:
            old = base.read(original_root / "fits" / case["dataset_name"] / "xgboost_ag_default/result.json")
            row["original_xgboost"] = {"status": old["status"], "error": old.get("development_error")}
        rows.append(row)
    base.write(output / "reuse_ledger.json", {"at": base.now(), "amendment_plan_sha256": base.digest(output / "plan.json"), "records": ledger})
    gains = [row["candidate_relative_gain"] for row in rows if "candidate_relative_gain" in row]
    complete = len(gains) == 37 and len(ledger) == 111 and not failures
    stats = {"median_relative_gain": float(np.median(gains)) if gains else None,
             "wins": sum(g > 1e-12 for g in gains), "worst_relative_regression": max([0.0] + [-g for g in gains]),
             "geometric_fit_ratio": float(np.exp(np.mean(np.log(ratios)))) if ratios else None,
             "p90_fit_ratio": float(np.quantile(ratios, .9)) if ratios else None}
    gate = amendment["amended_gate"]
    passed = complete and stats["median_relative_gain"] >= gate["minimum_median_relative_error_gain"] and stats["wins"] >= gate["minimum_dataset_wins"] and stats["worst_relative_regression"] <= gate["maximum_relative_regression"] and stats["geometric_fit_ratio"] <= gate["maximum_geometric_fit_ratio"] and stats["p90_fit_ratio"] <= gate["maximum_p90_fit_ratio"]
    result = {"stage": "amended37task_assessment", "created_at": base.now(), "plan_sha256": base.digest(output / "plan.json"),
              "reuse_ledger_sha256": base.digest(output / "reuse_ledger.json"), "original_gate_remains_failed": True,
              "complete_success": complete, "gate_passed": bool(passed), "statistics": stats, "datasets": rows,
              "failures": failures, "reuse_disclosure": amendment["reuse_disclosure"]}
    base.write(output / "assessment_report.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "report"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predecessor", type=Path)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.output.resolve(), args.predecessor.resolve())
    elif args.stage == "run":
        run(args.output.resolve())
    else:
        report(args.output.resolve())
