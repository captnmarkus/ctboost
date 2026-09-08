"""Staged temperature-scaling scout on the eight saved multiclass defaults.

Model selection and calibration reuse the stopping rows. Development scoring
and confirmation use separate rows, and historical dataset reuse is disclosed.
This script never changes a model's trees, argmax, or library defaults.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path


def bootstrap():
    import ctboost
    root = Path(__file__).resolve().parents[1]
    if Path(ctboost.__file__).resolve().is_relative_to(root / "ctboost"):
        raise RuntimeError("Use the isolated public-wheel Python with -I")
    sys.path.insert(0, str(root))
    from benchmarks import accuracy_grouped_scout as base
    base.bootstrap()
    return base


def prepare(output, predecessor):
    base = bootstrap()
    if output.exists():
        raise ValueError("Preserve previous studies; use a new output directory")
    previous = base.checked_plan(predecessor)
    cases = [case for case in previous["datasets"] if case["problem_type"] == "multiclass"]
    if len(cases) != 8:
        raise ValueError("Expected all eight preselected multiclass tasks")
    models = {}
    for case in cases:
        directory = predecessor / "fits" / case["dataset_name"] / "ctboost_default"
        record = base.read(directory / "result.json")
        if record["status"] != "ok" or record["model_sha256"] != base.digest(directory / "model.pkl"):
            raise ValueError("A saved default model is missing or changed")
        models[case["dataset_name"]] = record["model_sha256"]
    base.write(output / "plan.json", {
        "protocol_id": "ctboost_default_temperature_development_20260908_v1",
        "created_at": base.now(), "runner_sha256": base.digest(__file__),
        "predecessor": str(predecessor), "predecessor_plan_sha256": base.digest(predecessor / "plan.json"),
        "model_sha256": models, "datasets": cases,
        "hypothesis": "One positive temperature can improve multiclass probability calibration without changing trees or argmax.",
        "fit": "Fit log(T) on stopping fold5 only using scipy.optimize.minimize_scalar(method=bounded), bounds log(.25)..log(4), xatol1e-8,maxiter100; retainT1 if its stopping negative log likelihood is no worse.",
        "scoring": "Use the estimator's float32 softmax on raw margins divided by float32(T), preserving original probabilities exactly atT1. Score sklearn log_loss with all classes, identically to predecessor.",
        "resources": {"cpu_affinity": [0, 8], "threads": 2},
        "gate": {"minimum_median_relative_logloss_gain": 0.01, "minimum_wins": 5,
                 "maximum_any_relative_regression": 0.05, "require_all_eight_finite": True},
        "sequence": "Freeze this plan, fit and freeze all eight temperatures on fold5, then scorefold6. Openfold7 only if temperature development gate passes, with no refitting/reselection; apply identical qualitygate.",
        "reuse_disclosure": "Staged hypothesis after grouped-eight scout. These development scores and datasets have been used previously. Calibration reuses early-stopping/model-selection rows. Fold7 is separate from this study's fitting and development but historical dataset reuse means no independent benchmark/Elo claim.",
    })
    print(base.digest(output / "plan.json"))


def checked(output):
    base = bootstrap()
    plan = base.read(output / "plan.json")
    if base.digest(__file__) != plan["runner_sha256"]:
        raise RuntimeError("Runner changed after predeclaration")
    predecessor = Path(plan["predecessor"])
    base.checked_plan(predecessor)
    if base.digest(predecessor / "plan.json") != plan["predecessor_plan_sha256"]:
        raise ValueError("Predecessor plan changed")
    return base, plan, predecessor


def saved_model(base, plan, predecessor, case):
    path = predecessor / "fits" / case["dataset_name"] / "ctboost_default/model.pkl"
    if base.digest(path) != plan["model_sha256"][case["dataset_name"]]:
        raise ValueError("Saved default model changed")
    with path.open("rb") as stream:
        return pickle.load(stream)["model"]


def margins(model, frame):
    import numpy as np
    # Apply the same fitted preprocessing and float32 raw-score convention as
    # CTBoostClassifier.predict_proba; no transformations are fitted here.
    pool = model._transform_prediction_pool(frame)
    return np.asarray(model.get_booster().predict(pool), dtype=np.float32)


def fit(output):
    base, plan, predecessor = checked(output)
    import numpy as np
    import psutil
    from scipy.optimize import minimize_scalar
    from scipy.special import logsumexp

    psutil.Process().cpu_affinity(plan["resources"]["cpu_affinity"])
    if (output / "frozen_temperatures.json").exists():
        raise ValueError("Calibration is already frozen; no refitting")
    values = {}
    for case in plan["datasets"]:
        model = saved_model(base, plan, predecessor, case)
        stopping = base.load_role(predecessor, case, "stop")
        raw = margins(model, stopping["X"]).astype(np.float64)
        labels = np.asarray(stopping["y"], dtype=int)

        def objective(log_temperature, raw=raw, labels=labels):
            scaled = raw / np.exp(log_temperature)
            return float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(labels)), labels]))

        fitted = minimize_scalar(objective, method="bounded", bounds=(np.log(.25), np.log(4.0)),
                                 options={"xatol": 1e-8, "maxiter": 100})
        if not fitted.success or not np.isfinite(fitted.fun):
            raise RuntimeError(f"Calibration optimizer failed for {case['dataset_name']}")
        uncalibrated = objective(0.0)
        temperature = float(np.exp(fitted.x)) if fitted.fun < uncalibrated else 1.0
        values[case["dataset_name"]] = {"temperature": temperature, "stop_nll_default": uncalibrated,
                                         "stop_nll_calibrated": objective(np.log(temperature)),
                                         "optimizer_evaluations": fitted.nfev}
    base.write(output / "frozen_temperatures.json", {"created_at": base.now(), "plan_sha256": base.digest(output / "plan.json"), "values": values})
    print("All eight temperatures frozen before development scoring")


def evaluate(output, role):
    base, plan, predecessor = checked(output)
    import numpy as np
    import psutil

    psutil.Process().cpu_affinity(plan["resources"]["cpu_affinity"])
    temperatures = base.read(output / "frozen_temperatures.json")
    if temperatures["plan_sha256"] != base.digest(output / "plan.json"):
        raise ValueError("Temperatures do not match the plan")
    if role == "confirmation":
        development = base.read(output / "development_report.json")
        if not development["gate_passed"] or development["temperatures_sha256"] != base.digest(output / "frozen_temperatures.json"):
            raise RuntimeError("Temperature confirmation remains sealed")
        base.write(output / "confirmation_opened.json", {"at": base.now(), "development_report_sha256": base.digest(output / "development_report.json")})
    rows = []
    for case in plan["datasets"]:
        model = saved_model(base, plan, predecessor, case)
        data = base.load_role(predecessor, case, role)
        raw = margins(model, data["X"])
        baseline = model._softmax(raw)
        temperature = temperatures["values"][case["dataset_name"]]["temperature"]
        calibrated = model._softmax(raw / np.float32(temperature))
        if not np.array_equal(baseline.argmax(axis=1), calibrated.argmax(axis=1)):
            raise ValueError("Positive temperature changed argmax")
        if role == "development":
            previous = predecessor / "fits" / case["dataset_name"] / "ctboost_default/development.npz"
            with np.load(previous) as stored:
                np.testing.assert_array_equal(baseline, stored["predictions"])
        baseline_loss = base.score_predictions(data["y"], baseline, case)
        calibrated_loss = base.score_predictions(data["y"], calibrated, case)
        row = {"dataset": case["dataset_name"], "temperature": temperature,
               "baseline_logloss": baseline_loss, "calibrated_logloss": calibrated_loss,
               "relative_gain": (baseline_loss - calibrated_loss) / max(baseline_loss, 1e-12)}
        rows.append(row)
        directory = output / role
        directory.mkdir(exist_ok=True)
        np.savez_compressed(directory / (case["dataset_name"] + ".npz"), predictions=calibrated,
                            baseline=baseline, labels=data["y"], outer_train_positions=data["outer_train_positions"])
    gains = [row["relative_gain"] for row in rows]
    summary = {"median_relative_gain": float(np.median(gains)), "wins": sum(g > 1e-12 for g in gains),
               "worst_relative_regression": max([0.0] + [-g for g in gains])}
    gate = plan["gate"]
    passed = len(rows) == 8 and summary["median_relative_gain"] >= gate["minimum_median_relative_logloss_gain"] and summary["wins"] >= gate["minimum_wins"] and summary["worst_relative_regression"] <= gate["maximum_any_relative_regression"]
    report = {"stage": role, "created_at": base.now(), "plan_sha256": base.digest(output / "plan.json"),
              "temperatures_sha256": base.digest(output / "frozen_temperatures.json"), "gate_passed": bool(passed),
              "summary": summary, "all_argmax_preserved": True, "datasets": rows,
              "reuse_disclosure": plan["reuse_disclosure"]}
    base.write(output / (role + "_report.json"), report)
    import json
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "fit", "development", "confirmation"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predecessor", type=Path)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.output.resolve(), args.predecessor.resolve())
    elif args.stage == "fit":
        fit(args.output.resolve())
    else:
        evaluate(args.output.resolve(), args.stage)
