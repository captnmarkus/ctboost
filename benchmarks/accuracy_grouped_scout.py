"""Predeclared development scout; never reads official outer-test rows.

Compare unchanged CTBoost 0.1.60, its existing grouped-eight feature test, and
AutoGluon's default XGBoost. Train, early stopping, development scoring, and
confirmation use separate rows. Previously used pilot datasets are explicitly
development evidence, not an independent benchmark or an Elo estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import pickle
import random
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".tmp/tabarena-local-0159/pilot/data"
PROTOCOL = ROOT / "benchmarks/tabarena/pilot_0159_v1.json"
ARMS = ("ctboost_default", "ctboost_grouped8", "xgboost_ag_default")
ROLES = ("train", "stop", "development", "confirmation")
AFFINITIES = ((0, 8), (1, 9))
CTBOOST_PARAMS = {
    "iterations": 1000, "learning_rate": 0.05, "max_depth": 6,
    "alpha": 0.05, "lambda_l2": 1.0, "subsample": 0.8,
    "bootstrap_type": "Bernoulli", "ordered_ctr": True,
    "max_cat_threshold": 64, "verbose": False, "random_seed": 47,
}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def bootstrap():
    # Public wheel must load before the source checkout is added for adapters.
    import ctboost
    if Path(ctboost.__file__).resolve().is_relative_to(ROOT / "ctboost"):
        raise RuntimeError("Use an isolated public-wheel Python with -I")
    if ctboost.__version__ != "0.1.60":
        raise RuntimeError("This protocol requires public CTBoost 0.1.60")
    sys.path.insert(0, str(ROOT))
    return ctboost


def checked_plan(output):
    plan = read(output / "plan.json")
    if plan["runner_sha256"] != digest(__file__):
        raise RuntimeError("Runner changed after the plan was frozen")
    for relative, expected in plan["source_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError(f"Frozen dependency changed: {relative}")
    return plan


def prepare(output):
    ctboost = bootstrap()
    import numpy as np
    from autogluon.common.utils.cv_splitter import CVSplitter
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    if output.exists():
        raise ValueError("Use a new output directory; preserve previous attempts")
    protocol = read(PROTOCOL)
    cases = []
    for metadata in protocol["datasets"]:
        source = SOURCE / metadata["dataset_name"] / "outer_train.pkl"
        receipt = read(source.with_name("prepared.json"))
        if receipt["contains_outer_test_rows"] is not False or receipt["data_sha256"] != digest(source):
            raise ValueError("Source must be verified outer-training rows only")
        if receipt["protocol_sha256"] != digest(PROTOCOL):
            raise ValueError("Source does not match the original metadata-selected pilot")
        with source.open("rb") as stream:
            original = pickle.load(stream)
        labels = np.asarray(original["y"])
        classification = metadata["problem_type"] != "regression"
        if classification:
            labels = LabelEncoder().fit_transform(labels)
        selected = np.arange(len(labels))
        if len(selected) > 10000:
            selected, _ = train_test_split(selected, train_size=10000, random_state=47,
                                           stratify=labels if classification else None)
            selected.sort()
        frame = original["X"].iloc[selected].reset_index(drop=True)
        labels = labels[selected]
        splitter = CVSplitter(n_splits=8, n_repeats=1, random_state=47, stratify=classification)
        folds = [validation for _, validation in splitter.split(frame, labels)]
        indices = {"train": np.sort(np.concatenate(folds[:5])), "stop": folds[5],
                   "development": folds[6], "confirmation": folds[7]}
        if len(np.unique(np.concatenate(list(indices.values())))) != len(selected):
            raise ValueError("All four row roles must partition the capped outer training data")
        case = {**metadata, "source_sha256": digest(source), "original_rows": len(original["y"]),
                "capped_rows": len(selected), "roles": {}}
        for role, positions in indices.items():
            target = output / "data" / metadata["dataset_name"] / (role + ".pkl")
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = {"X": frame.iloc[positions].copy(), "y": labels[positions],
                       "outer_train_positions": selected[positions]}
            with target.open("wb") as stream:
                pickle.dump(payload, stream, protocol=5)
            case["roles"][role] = {"rows": len(positions), "sha256": digest(target),
                                   "index_sha256": hashlib.sha256(np.asarray(selected[positions], dtype="<i8").tobytes()).hexdigest()}
        cases.append(case)
    if len(cases) != 14:
        raise ValueError("Expected all 14 original metadata-selected tasks")
    dependencies = ["benchmarks/tabarena/ctboost_model.py", "benchmarks/tabarena/learning_options.py"]
    versions = {name: importlib.metadata.version(name) for name in
                ("ctboost", "autogluon.tabular", "autogluon.core", "xgboost", "numpy", "pandas", "scikit-learn")}
    plan = {
        "protocol_id": "ctboost_grouped8_development_20260908_v1", "created_at": now(),
        "runner_sha256": digest(__file__), "source_sha256": {p: digest(ROOT / p) for p in dependencies},
        "ctboost_native_sha256": digest(ctboost._core.__file__), "versions": versions,
        "selection": "All 14 original metadata-selected pilot tasks, without score-based filtering.",
        "reuse_disclosure": "These outer-training datasets were used in previous development studies. New seed47 disjoint role folds do not make dataset reuse independent confirmation. No official outer-test rows or Elo.",
        "split_policy": "Cap outer-training rows to10000 first (seed47; classification stratified). CVSplitter8 seed47: train folds0-4, early-stop5, development6, sealed confirmation7.",
        "arms": list(ARMS), "ctboost_params": CTBOOST_PARAMS, "candidate_override": {"feature_test": "grouped", "feature_test_bins": 8, "feature_test_adjustment": "none"},
        "xgboost": "AutoGluon XGBoostModel standard defaults, seed47, same early-stopping rows/300s/2CPU; its default adaptive patience and tree cap are retained.",
        "resources": {"fit_seconds": 300, "hard_wall_seconds": 390, "rss_bytes": 8 * 1024 ** 3,
                      "threads": 2, "max_workers": 2, "affinities": AFFINITIES, "ctboost_early_stop": 50},
        "gate": {"minimum_median_relative_error_gain": 0.01, "minimum_dataset_wins": 8,
                 "maximum_relative_regression": 0.10, "maximum_geometric_fit_ratio": 3,
                 "maximum_p90_fit_ratio": 5, "require_all_42_fits_finite_without_hard_timeout_or_oom": True},
        "confirmation_policy": "Only unseal fold7 if the development gate passes. Apply the same quality gate without retraining or choosing another candidate. A passing reused-data scout supports a separate benchmark, not an automatic library-default change.",
        "datasets": cases, "expected_fits": 42,
    }
    write(output / "plan.json", plan)
    print(json.dumps({"prepared": str(output), "fits": 42, "plan_sha256": digest(output / "plan.json")}))


def load_role(output, case, role):
    path = output / "data" / case["dataset_name"] / (role + ".pkl")
    if digest(path) != case["roles"][role]["sha256"]:
        raise ValueError(f"Changed {role} rows")
    with path.open("rb") as stream:
        return pickle.load(stream)


def score_predictions(y, prediction, case):
    import numpy as np
    from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite predictions")
    if case["problem_type"] == "binary":
        error = 1 - roc_auc_score(y, prediction)
    elif case["problem_type"] == "regression":
        error = np.sqrt(mean_squared_error(y, prediction))
    else:
        error = log_loss(y, prediction, labels=np.arange(case["num_classes"]))
    if not np.isfinite(error):
        raise ValueError("Nonfinite metric")
    return float(error)


def predict_saved(payload, frame, case):
    import numpy as np

    from benchmarks.tabarena.ctboost_model import normalize_tabarena_frame
    model = payload["model"]
    frame, _ = normalize_tabarena_frame(frame, categorical_columns=payload["categorical_columns"])
    resources = {"num_cpus": 2} if payload["arm"] == "xgboost_ag_default" else {}
    if case["problem_type"] == "regression":
        return np.asarray(model.predict(frame, **resources))
    prediction = np.asarray(model.predict_proba(frame, **resources))
    if case["problem_type"] == "binary" and prediction.ndim == 2:
        prediction = prediction[:, 1]
    return prediction


def fit(output, dataset, arm, slot):
    import psutil
    psutil.Process().cpu_affinity(list(AFFINITIES[slot]))
    ctboost = bootstrap()
    import numpy as np
    from autogluon.tabular.models.xgboost.xgboost_model import XGBoostModel

    from benchmarks.tabarena.ctboost_model import normalize_tabarena_frame

    plan = checked_plan(output)
    if digest(ctboost._core.__file__) != plan["ctboost_native_sha256"]:
        raise ValueError("CTBoost runtime changed")
    case = next(row for row in plan["datasets"] if row["dataset_name"] == dataset)
    directory = output / "fits" / dataset / arm
    if directory.exists():
        raise ValueError("A started fit is terminal; use a new protocol/run for retries")
    directory.mkdir(parents=True)
    record = {"dataset": dataset, "arm": arm, "started_at": now(), "status": "failed",
              "plan_sha256": digest(output / "plan.json"), "affinity": list(AFFINITIES[slot]),
              "soft_deadline_stopped": False}
    write(directory / "started.json", record)
    try:
        train = load_role(output, case, "train")
        stopping = load_role(output, case, "stop")
        X, categoricals = normalize_tabarena_frame(train["X"])
        X_stop, _ = normalize_tabarena_frame(stopping["X"], categorical_columns=categoricals)
        if case["problem_type"] != "regression" and len(np.unique(train["y"])) != case["num_classes"]:
            raise ValueError("A training role lacks a class; do not silently change the split")
        started = time.perf_counter()
        if arm.startswith("ctboost"):
            params = dict(plan["ctboost_params"])
            if arm == "ctboost_grouped8":
                params.update(plan["candidate_override"])
            params.update(cat_features=categoricals or None,
                          eval_metric={"binary": "AUC", "regression": "RMSE", "multiclass": "MultiClass"}[case["problem_type"]])
            model_type = ctboost.CTBoostRegressor if case["problem_type"] == "regression" else ctboost.CTBoostClassifier
            model = model_type(**params)

            def deadline(env):
                elapsed = time.perf_counter() - started
                stop = elapsed + 2 * elapsed / (env.iteration + 1) >= plan["resources"]["fit_seconds"]
                record["soft_deadline_stopped"] |= stop
                return stop

            model.fit(X, train["y"], eval_set=(X_stop, stopping["y"]), early_stopping_rounds=50, callbacks=[deadline])
            record["rounds"] = model.get_booster().num_iterations_trained
            record["parameters"] = params
        else:
            model = XGBoostModel(path=str(directory / "ag"), name="XGBoost", problem_type=case["problem_type"],
                                 eval_metric=case["eval_metric"], hyperparameters={"seed": 47})
            model.fit(X=X, y=train["y"], X_val=X_stop, y_val=stopping["y"],
                      time_limit=300, num_cpus=2, num_gpus=0, verbosity=0)
            record["rounds"] = int(model.model.get_booster().num_boosted_rounds())
            record["parameters"] = model.params
        record["fit_seconds"] = time.perf_counter() - started
        payload = {"model": model, "arm": arm, "categorical_columns": categoricals}
        with (directory / "model.pkl").open("wb") as stream:
            pickle.dump(payload, stream, protocol=5)
        record["model_sha256"] = digest(directory / "model.pkl")
        development = load_role(output, case, "development")
        prediction = predict_saved(payload, development["X"], case)
        record["development_error"] = score_predictions(development["y"], prediction, case)
        np.savez_compressed(directory / "development.npz", predictions=prediction, labels=development["y"],
                            outer_train_positions=development["outer_train_positions"])
        record.update(status="ok", development_sha256=digest(directory / "development.npz"))
    except Exception:  # noqa: BLE001 - retain every predeclared failed arm as evidence
        record["error"] = traceback.format_exc()
    record["finished_at"] = now()
    write(directory / "result.json", record)
    print(json.dumps({k: record.get(k) for k in ("dataset", "arm", "status", "fit_seconds", "development_error")}))


def run(output, workers):
    plan = checked_plan(output)
    if workers not in (1, 2):
        raise ValueError("Use one or two workers")
    jobs = [(case["dataset_name"], arm) for case in plan["datasets"] for arm in ARMS]
    random.Random(47).shuffle(jobs)
    (output / "logs").mkdir(exist_ok=True)
    env = dict(os.environ, CTBOOST_HIST_THREADS="2", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")

    def worker(slot):
        import psutil
        for dataset, arm in jobs[slot::workers]:
            directory = output / "fits" / dataset / arm
            if directory.exists():
                continue
            command = [sys.executable, "-I", str(Path(__file__).resolve()), "fit", "--output", str(output),
                       "--dataset", dataset, "--arm", arm, "--slot", str(slot)]
            started = time.monotonic()
            peak = 0
            reason = None
            with (output / "logs" / f"{dataset}-{arm}.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env,
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
                        # psutil checks PID reuse in kill(); this Process owns this subprocess.
                        for child in owner.children(recursive=True):
                            child.kill()
                        owner.kill()
                        process.wait(timeout=15)
                        break
                    time.sleep(0.25)
            result_path = directory / "result.json"
            result = read(result_path) if result_path.exists() else {"dataset": dataset, "arm": arm, "status": "failed", "plan_sha256": digest(output / "plan.json")}
            result.update(peak_rss_bytes=peak, exit_code=process.returncode, resource_failure=reason)
            if reason or process.returncode != 0:
                result["status"] = "failed"
            write(result_path, result)
            print(json.dumps({"finished": f"{dataset}/{arm}", "status": result["status"]}), flush=True)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(worker, range(workers)))


def report(output, stage):
    bootstrap()
    import numpy as np
    plan = checked_plan(output)
    if stage == "confirmation":
        development = read(output / "development_report.json")
        if development["plan_sha256"] != digest(output / "plan.json") or not development["gate_passed"]:
            raise RuntimeError("Confirmation remains sealed because development did not pass")
        write(output / "confirmation_opened.json", {"at": now(), "development_report_sha256": digest(output / "development_report.json")})
    records, rows = [], []
    for case in plan["datasets"]:
        per_arm = {}
        for arm in ARMS:
            directory = output / "fits" / case["dataset_name"] / arm
            record = read(directory / "result.json")
            if record["plan_sha256"] != digest(output / "plan.json"):
                raise ValueError("Result belongs to a different plan")
            records.append(record)
            if record["status"] != "ok" or record.get("resource_failure"):
                continue
            if digest(directory / "model.pkl") != record["model_sha256"] or digest(directory / "development.npz") != record["development_sha256"]:
                raise ValueError("Changed fitted artifact")
            if stage == "confirmation":
                heldout = load_role(output, case, "confirmation")
                with (directory / "model.pkl").open("rb") as stream:
                    payload = pickle.load(stream)
                predictions = predict_saved(payload, heldout["X"], case)
                error = score_predictions(heldout["y"], predictions, case)
                np.savez_compressed(directory / "confirmation.npz", predictions=predictions, labels=heldout["y"], outer_train_positions=heldout["outer_train_positions"])
            else:
                error = record["development_error"]
            per_arm[arm] = float(error)
        row = {"dataset": case["dataset_name"], "problem_type": case["problem_type"], "errors": per_arm}
        if len(per_arm) == len(ARMS):
            baseline, candidate = per_arm[ARMS[0]], per_arm[ARMS[1]]
            row["candidate_relative_gain"] = (baseline - candidate) / max(baseline, 1e-12)
            row["candidate_vs_xgboost_gain"] = (per_arm[ARMS[2]] - candidate) / max(per_arm[ARMS[2]], 1e-12)
        rows.append(row)
    gains = [row["candidate_relative_gain"] for row in rows if "candidate_relative_gain" in row]
    successful = len(gains) == 14 and len(records) == 42 and all(r["status"] == "ok" and not r.get("resource_failure") for r in records)
    ratios = []
    for case in plan["datasets"]:
        pair = {r["arm"]: r for r in records if r["dataset"] == case["dataset_name"]}
        if all(pair[a]["status"] == "ok" for a in ARMS[:2]):
            ratios.append(pair[ARMS[1]]["fit_seconds"] / max(pair[ARMS[0]]["fit_seconds"], 1e-12))
    statistics = {"median_relative_gain": float(np.median(gains)) if gains else None,
                  "wins": sum(g > 1e-12 for g in gains), "worst_relative_regression": max([0.0] + [-g for g in gains]),
                  "geometric_fit_ratio": float(np.exp(np.mean(np.log(ratios)))) if ratios else None,
                  "p90_fit_ratio": float(np.quantile(ratios, .9)) if ratios else None}
    gate = plan["gate"]
    passed = successful and statistics["median_relative_gain"] >= gate["minimum_median_relative_error_gain"] and statistics["wins"] >= gate["minimum_dataset_wins"] and statistics["worst_relative_regression"] <= gate["maximum_relative_regression"] and statistics["geometric_fit_ratio"] <= gate["maximum_geometric_fit_ratio"] and statistics["p90_fit_ratio"] <= gate["maximum_p90_fit_ratio"]
    result = {"stage": stage, "created_at": now(), "plan_sha256": digest(output / "plan.json"),
              "gate_passed": bool(passed), "complete_success": successful, "statistics": statistics, "datasets": rows,
              "failures": [r for r in records if r["status"] != "ok"], "reuse_disclosure": plan["reuse_disclosure"]}
    write(output / (stage + "_report.json"), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "fit", "run", "development", "confirmation"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dataset")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--slot", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.stage == "prepare":
        prepare(output)
    elif args.stage == "fit":
        fit(output, args.dataset, args.arm, args.slot)
    elif args.stage == "run":
        run(output, args.workers)
    else:
        report(output, args.stage)
