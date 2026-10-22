"""Equal-candidate-count development search on cached real TabArena tasks.

No official test role or failed-gate confirmation role is loaded by this script.
All model selection uses the stopping role, and every selection is frozen before
the separate development report. These historically reused tasks are exploratory.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import platform
import subprocess
import sys
import time
import traceback

for _variable in ("CTBOOST_HIST_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".tmp/inference-accuracy-20260908"
LIBRARIES = ("ctboost", "xgboost", "lightgbm")


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
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def bootstrap(args):
    sys.meta_path = [finder for finder in sys.meta_path if type(finder).__module__ != "_editable_skbc_ctboost"]
    sys.path[:0] = [str(args.package_root), str(args.dependencies)]
    import ctboost
    import lightgbm
    import xgboost
    if not Path(ctboost.__file__).resolve().is_relative_to(args.package_root) or not Path(ctboost._core.__file__).resolve().is_relative_to(args.package_root):
        raise ValueError("CTBoost must resolve inside the frozen isolated package")
    runtime = {
        "python": sys.executable, "python_version": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "psutil")},
        "ctboost_build": ctboost.build_info(), "ctboost_native_sha256": digest(ctboost._core.__file__),
        "ctboost_python_sha256": {str(p.relative_to(args.package_root)): digest(p) for p in sorted((args.package_root / "ctboost").rglob("*.py"))},
        "xgboost_library_sha256": digest(xgboost.core._LIB._name),
        "lightgbm_library_sha256": digest(lightgbm.basic._LIB._name),
        "imported_packages": {"ctboost": {"path": ctboost.__file__, "version": ctboost.__version__},
                              "xgboost": {"path": xgboost.__file__, "version": xgboost.__version__},
                              "lightgbm": {"path": lightgbm.__file__, "version": lightgbm.__version__}},
    }
    return runtime


def configurations(library, problem, wave):
    if wave != 3:
        raise ValueError("This frozen extension implements wave3 only")
    if library == "ctboost":
        profiles = [
            (4, .1, .05, True, 0, 64, False),
            (4, .1, .05, False, 0, 64, False),
            (4, .1, .05, True, 8, 64, False),
            (4, .1, .05, False, 8, 256, False),
            (6, .05, .05, True, 0, 256, False),
            (6, .05, .05, True, 0, 256, True),
            (8, .1, .2, True, 8, 256, False),
            (8, .1, .2, True, 8, 256, True),
        ]
        rows = [{"learning_rate": lr, "max_depth": depth, "alpha": alpha,
                 "lambda_l2": 3, "min_data_in_leaf": 10, "feature_test": "grouped",
                 "feature_test_bins": 16 if alternate and problem != "multiclass" else 8,
                 "ordered_ctr": ctr, "one_hot_max_size": onehot, "max_cat_threshold": threshold,
                 "multiclass_feature_test": "joint" if alternate and problem == "multiclass" else "single",
                 "leaf_estimation_iterations": 1 if problem == "regression" else 3,
                 "multi_strategy": "multi_output_tree" if problem == "multiclass" else "one_output_per_tree"}
                for depth, lr, alpha, ctr, onehot, threshold, alternate in profiles]
        for row in rows:
            if row["leaf_estimation_iterations"] > 1:
                row["multiclass_leaf_solver" if problem == "multiclass" else "leaf_estimation_backtracking"] = "full" if problem == "multiclass" else True
    elif library == "xgboost":
        rows = [{"eta": lr, "max_depth": depth, "lambda": l2,
                 "max_cat_to_onehot": onehot, "max_cat_threshold": threshold,
                 "multi_strategy": "multi_output_tree" if problem == "multiclass" else "one_output_per_tree"}
                for depth, lr, l2 in ((3, .1, 1), (5, .05, 3))
                for onehot in (1, 8) for threshold in (64, 256)]
    else:
        rows = [{"learning_rate": lr, "num_leaves": leaves, "min_data_in_leaf": minimum,
                 "lambda_l2": l2, "max_cat_to_onehot": onehot, "max_cat_threshold": 256,
                 "cat_l2": cat_penalty, "cat_smooth": cat_penalty}
                for leaves, lr, minimum, l2 in ((7, .1, 10, 0), (31, .05, 5, 3))
                for onehot in (1, 8) for cat_penalty in (0, 10)]
    assert len(rows) == 8
    return rows


def load_role(case, role):
    if role not in ("train", "stop", "development"):
        raise ValueError("This runner never loads confirmation or official outer test")
    record = case["roles"][role]
    if digest(record["path"]) != record["sha256"]:
        raise ValueError("Cached role changed")
    with Path(record["path"]).open("rb") as stream:
        return pickle.load(stream)


def prepare(args):
    import numpy as np
    runtime = bootstrap(args)
    if args.output.exists() or args.artifacts.exists():
        raise ValueError("Use fresh output and artifact paths; preserve previous attempts")
    population, inputs = [], {}
    for study in ("accuracy-grouped-scout-v1", "accuracy-remaining-scout-v1"):
        path = CACHE / study / "plan.json"
        inputs[str(path)] = digest(path)
        for original in read(path)["datasets"]:
            population.append({**original, "source": str(path.parent)})
    selected = []
    for problem in ("binary", "regression", "multiclass"):
        ordered = sorted((c for c in population if c["problem_type"] == problem and c["num_features"] <= 256),
                         key=lambda c: (c["original_rows"], c["dataset_name"]))
        for position in [round(i * (len(ordered) - 1) / 3) for i in range(4)]:
            case = ordered[position]
            case["roles"] = {role: {**case["roles"][role], "path": str(Path(case["source"]) / "data" / case["dataset_name"] / (role + ".pkl"))}
                             for role in ("train", "stop", "development")}
            # Validate training/stopping identities without reading development rows.
            train, stop = load_role(case, "train"), load_role(case, "stop")
            assert not np.intersect1d(train["outer_train_positions"], stop["outer_train_positions"]).size
            if problem != "regression" and len(np.unique(train["y"])) != case["num_classes"]:
                raise ValueError("Predeclared training role lacks a class")
            selected.append(case)
    plan = {
        "protocol": "fair_accuracy_development_20261016", "created_at": now(), "wave": args.wave,
        "script_sha256": digest(__file__), "runtime": runtime, "input_plans_sha256": inputs,
        "selection": "From51original metadata tasks: filter<=256features for bounded resources; within each problem type sort(original outer-training row count,dataset name), select4equally spaced rounded rank positions. No score-based dataset choice.",
        "cases": selected, "seed": 47, "round_cap": 2000, "patience": 50,
        "candidate_grids": {lib: {problem: configurations(lib, problem, args.wave) for problem in ("binary", "regression", "multiclass")} for lib in LIBRARIES},
        "next_wave_predeclared": None,
        "next_wave_policy": "Wave3 grid frozen before wave2 development is scored. Run only if wave2 misses the existing development target. Cumulative stopping-only selection across24 configurations per library; official test stays sealed. Further work requires a new frozen protocol.",
        "target": {"median_relative_error_gain_vs_each_competitor": 0.0, "minimum_task_wins_or_ties_vs_each": 6,
                   "all_fits_successful": True, "description": "Exploratory development target, not general ranking or a claim of statistical equivalence"},
        "model_selection": "Minimum primary error on stopping role across8configs, ties fewer retained rounds then configindex. All36library/task selections frozen before development assessment. Wave3 also considers frozen cumulative parent winners by stopping error.",
        "primary_error": {"binary": "1-ROC-AUC", "regression": "RMSE", "multiclass": "logloss"},
        "preprocessing": "Training-only categorical vocabulary shared by libraries; categorical dtype on all frames, unseen values become missing. Bool becomesuint8; numericfloat32. CTBoost compares existing ordered-CTR and raw-category/one-hot options; XGB/LGB vary existing native categorical controls. No target encoding outside CTBoost's existing ordered training pipeline.",
        "resources": {"threads_per_fit": 1, "workers": 2, "cpus": [4, 6], "configurations_per_library_task": 8,
                      "maximum_job_seconds": 3600, "maximum_job_rss_bytes": 8 * 1024 ** 3, "fit_times": "Diagnostic, not isolated performance benchmark"},
        "limitations": "Historically reused development data, one seed, capped10000outer-training rows, small fixed configuration portfolios. Equal candidatecounts and round caps are not equal walltime or equivalent statistical capacity. No official outertest or original14failedgateconfirmation access; no bagging/Elo/leaderboard claim.",
        "previous": str(args.previous) if args.previous else None,
    }
    if args.previous:
        previous = read(args.previous / "plan.json")
        if previous["wave"] != 2 or args.wave != 3:
            raise ValueError("Wave3 must extend frozen wave2 selections")
        if (args.previous / "development.json").exists():
            raise ValueError("Wave3 must freeze before opening wave2 development")
        if [c["dataset_name"] for c in previous["cases"]] != [c["dataset_name"] for c in selected]:
            raise ValueError("Dataset panel changed between waves")
        plan["previous_frozen_sha256"] = digest(args.previous / "frozen_selections.json")
    args.artifacts.mkdir(parents=True)
    write(args.output / "plan.json", plan)
    print(json.dumps({"datasets": [c["dataset_name"] for c in selected], "fits": 288, "plan_sha256": digest(args.output / "plan.json")}))


def checked(args):
    plan = read(args.output / "plan.json")
    if plan["script_sha256"] != digest(__file__) or bootstrap(args) != plan["runtime"]:
        raise ValueError("Runner or runtime changed after freeze")
    for path, expected in plan["input_plans_sha256"].items():
        if digest(path) != expected:
            raise ValueError("Historical source plan changed")
    if plan["previous"] and digest(Path(plan["previous"]) / "frozen_selections.json") != plan["previous_frozen_sha256"]:
        raise ValueError("Previous wave's frozen selections changed")
    return plan


def normalize(frame, schema=None):
    import numpy as np
    import pandas as pd
    output = frame.copy()
    if schema is None:
        schema = {}
        for column in output.columns:
            if pd.api.types.is_bool_dtype(output[column].dtype):
                continue
            if not pd.api.types.is_numeric_dtype(output[column].dtype):
                schema[column] = sorted(output[column].dropna().unique().tolist(), key=lambda value: (type(value).__name__, str(value)))
    for column in output.columns:
        if column in schema:
            output[column] = pd.Categorical(output[column], categories=schema[column])
        elif pd.api.types.is_bool_dtype(output[column].dtype):
            output[column] = output[column].astype(np.uint8)
        else:
            output[column] = output[column].astype(np.float32)
    return output, schema


def error(raw, labels, problem):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    raw, labels = np.asarray(raw, dtype=np.float64), np.asarray(labels)
    if not np.isfinite(raw).all():
        raise ValueError("Non-finite predictions")
    if problem == "binary":
        return float(1 - roc_auc_score(labels, raw))
    if problem == "regression":
        return float(np.sqrt(np.mean((raw - labels) ** 2)))
    shifted = raw - raw.max(axis=1, keepdims=True)
    return float(np.mean(np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(labels)), labels.astype(int)]))


def predict(payload, frame):
    frame, _ = normalize(frame, payload["schema"])
    if payload["library"] == "ctboost":
        return payload["model"].predict(frame)
    if payload["library"] == "xgboost":
        import xgboost as xgb
        return payload["model"].predict(xgb.DMatrix(frame, enable_categorical=True), output_margin=True)
    return payload["model"].predict(frame, raw_score=True, num_threads=1)


def fit_configuration(library, controls, plan, case, X, y, Xs, ys, schema):
    import ctboost
    import lightgbm as lgb
    import xgboost as xgb
    problem = case["problem_type"]
    rounds, patience, seed = plan["round_cap"], plan["patience"], plan["seed"]
    started = time.perf_counter()
    if library == "ctboost":
        params = {"iterations": rounds, "learning_rate": .05, "max_depth": 6, "lambda_l2": 1,
                  "alpha": .05, "max_bins": 256, "subsample": .8, "bootstrap_type": "Bernoulli",
                  "feature_test_bins": 8, "feature_test_adjustment": "none", "multiclass_feature_test": "single",
                  "ordered_ctr": True, "max_cat_threshold": 64, "random_seed": seed, "verbose": False,
                  "cat_features": list(schema) or None,
                  "eval_metric": {"binary": "AUC", "regression": "RMSE", "multiclass": "MultiClass"}[problem], **controls}
        if problem == "regression":
            estimator = ctboost.CTBoostRegressor(**params)
        else:
            estimator = ctboost.CTBoostClassifier(**params, loss_function="MultiClass" if problem == "multiclass" else "Logloss")
        estimator.fit(X, y, eval_set=(Xs, ys), early_stopping_rounds=patience)
        model = estimator.get_booster()
        retained = model.num_iterations_trained
        # CTBoost prunes histories with the model; do not mistake retained
        # history length for physical rounds including patience.
        trained = None
    elif library == "xgboost":
        params = {"tree_method": "hist", "nthread": 1, "seed": seed, "max_bin": 256, "subsample": .8,
                  "objective": {"binary": "binary:logistic", "regression": "reg:squarederror", "multiclass": "multi:softprob"}[problem],
                  "eval_metric": {"binary": "auc", "regression": "rmse", "multiclass": "mlogloss"}[problem], **controls}
        if problem == "multiclass":
            params["num_class"] = case["num_classes"]
        model = xgb.train(params, xgb.DMatrix(X, label=y, enable_categorical=True), num_boost_round=rounds,
                          evals=[(xgb.DMatrix(Xs, label=ys, enable_categorical=True), "stop")],
                          early_stopping_rounds=patience, verbose_eval=False)
        retained, trained = int(model.best_iteration) + 1, model.num_boosted_rounds()
        model = model[:retained]
    else:
        params = {"num_threads": 1, "seed": seed, "deterministic": True, "force_col_wise": True,
                  "max_bin": 255, "bagging_fraction": .8, "bagging_freq": 1, "verbosity": -1,
                  "zero_as_missing": False,
                  "objective": {"binary": "binary", "regression": "regression", "multiclass": "multiclass"}[problem],
                  "metric": {"binary": "auc", "regression": "rmse", "multiclass": "multi_logloss"}[problem], **controls}
        if problem == "multiclass":
            params["num_class"] = case["num_classes"]
        train = lgb.Dataset(X, label=y, categorical_feature=list(schema))
        stop = lgb.Dataset(Xs, label=ys, reference=train, categorical_feature=list(schema))
        history = {}
        model = lgb.train(params, train, num_boost_round=rounds, valid_sets=[stop], valid_names=["stop"],
                          callbacks=[lgb.early_stopping(patience, first_metric_only=True, verbose=False), lgb.record_evaluation(history)])
        retained = int(model.best_iteration or model.current_iteration())
        trained = len(history["stop"][params["metric"]])
        model = lgb.Booster(model_str=model.model_to_string(num_iteration=retained))
    fit_seconds = time.perf_counter() - started
    payload = {"library": library, "model": model, "schema": schema, "problem": problem}
    score = error(predict(payload, Xs), ys, problem)
    return payload, {"parameters": params, "fit_seconds": fit_seconds, "retained_rounds": retained,
                     "trained_rounds_if_available": trained, "retained_at_round_cap": retained == rounds, "stop_error": score}


def fit_job(args):
    import numpy as np
    import psutil
    plan = checked(args)
    psutil.Process().cpu_affinity([plan["resources"]["cpus"][args.slot]])
    case = next(c for c in plan["cases"] if c["dataset_name"] == args.dataset)
    directory = args.output / "fits" / args.dataset / args.library
    directory.mkdir(parents=True, exist_ok=False)
    train, stop = load_role(case, "train"), load_role(case, "stop")
    X, schema = normalize(train["X"])
    Xs, _ = normalize(stop["X"], schema)
    best = None
    best_path = args.artifacts / "models" / args.dataset / f"{args.library}.pkl"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for index, controls in enumerate(plan["candidate_grids"][args.library][case["problem_type"]]):
        while (args.artifacts / "pause_fits").exists():
            time.sleep(.5)
        record = {"dataset": args.dataset, "library": args.library, "config_index": index,
                  "plan_sha256": digest(args.output / "plan.json"), "started_at": now(), "status": "failed"}
        try:
            model, details = fit_configuration(args.library, controls, plan, case, X, train["y"], Xs, stop["y"], schema)
            record.update(details, status="ok", rss_after_fit_bytes=psutil.Process().memory_info().rss)
            key = (record["stop_error"], record["retained_rounds"], index)
            if best is None or key < best["selection_key"]:
                candidate_path = best_path.with_suffix(".candidate.pkl")
                with candidate_path.open("wb") as stream:
                    pickle.dump(model, stream, protocol=5)
                with candidate_path.open("rb") as stream:
                    restored = pickle.load(stream)
                np.testing.assert_array_equal(predict(restored, Xs), predict(model, Xs))
                candidate_path.replace(best_path)
                best = {"selection_key": key, "config_index": index, **details}
        except Exception:
            record.update(status="failed", error=traceback.format_exc())
        record["finished_at"] = now()
        records.append(record)
        write(directory / f"config_{index}.json", record)
        print(json.dumps({k: record.get(k) for k in ("dataset", "library", "config_index", "status", "stop_error", "fit_seconds")}), flush=True)
    selection = {"dataset": args.dataset, "library": args.library, "plan_sha256": digest(args.output / "plan.json"),
                 "all_eight_successful": all(r["status"] == "ok" for r in records), "best": best,
                 "search_fit_seconds": sum(r.get("fit_seconds", 0) for r in records),
                 "model_path": str(best_path), "model_sha256": digest(best_path) if best is not None else None,
                 "model_bytes": best_path.stat().st_size if best is not None else None,
                 "source_wave": plan["wave"]}
    write(directory / "selection.json", selection)


def run(args):
    import psutil
    plan = checked(args)
    jobs = [(case["dataset_name"], library) for case in plan["cases"] for library in LIBRARIES]
    if (args.output / "fits").exists():
        raise ValueError("A started run is terminal; preserve it and use a new registered study for retries")
    (args.output / "logs").mkdir()

    def slot_worker(slot):
        for dataset, library in jobs[slot::2]:
            while (args.artifacts / "pause_fits").exists():
                time.sleep(.5)
            command = [sys.executable, "-I", str(Path(__file__).resolve()), "fit", "--output", str(args.output),
                       "--artifacts", str(args.artifacts), "--package-root", str(args.package_root), "--dependencies", str(args.dependencies),
                       "--dataset", dataset, "--library", library, "--slot", str(slot)]
            started, peak, failure = time.monotonic(), 0, None
            with (args.output / "logs" / f"{dataset}-{library}.txt").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                owner = psutil.Process(process.pid)
                while process.poll() is None:
                    try:
                        peak = max(peak, owner.memory_info().rss)
                    except psutil.NoSuchProcess:
                        pass
                    if time.monotonic() - started > plan["resources"]["maximum_job_seconds"] or peak > plan["resources"]["maximum_job_rss_bytes"]:
                        failure = "job_resource_limit"
                        owner.kill()
                        process.wait(timeout=15)
                        break
                    time.sleep(.5)
            write(args.output / "logs" / f"{dataset}-{library}-resource.json", {
                "dataset": dataset, "library": library, "exit_code": process.returncode,
                "resource_failure": failure, "peak_rss_bytes": peak, "wall_seconds": time.monotonic() - started})
            print(json.dumps({"completed": dataset, "library": library, "exit_code": process.returncode, "resource_failure": failure}), flush=True)
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(slot_worker, range(2)))
    selections = []
    previous_document = read(Path(plan["previous"]) / "frozen_selections.json") if plan["previous"] else {"selections": [], "all_fits_successful": True}
    previous = previous_document["selections"]
    for dataset, library in jobs:
        current = read(args.output / "fits" / dataset / library / "selection.json")
        choices = [current] + [s for s in previous if s["dataset"] == dataset and s["library"] == library]
        chosen = min((s for s in choices if s["best"] is not None), key=lambda s: tuple(s["best"]["selection_key"]))
        selections.append({**chosen, "current_wave_all_eight_successful": current["all_eight_successful"],
                           "cumulative_search_fit_seconds": current["search_fit_seconds"] + sum(s.get("cumulative_search_fit_seconds", s["search_fit_seconds"]) for s in previous if s["dataset"] == dataset and s["library"] == library)})
    write(args.output / "frozen_selections.json", {"plan_sha256": digest(args.output / "plan.json"),
        "created_at": now(), "development_scored": False, "official_test_scored": False,
        "selections": selections, "all_fits_successful": previous_document["all_fits_successful"] and all(s["current_wave_all_eight_successful"] for s in selections)})


def development(args):
    import numpy as np
    plan = checked(args)
    frozen = read(args.output / "frozen_selections.json")
    if frozen["plan_sha256"] != digest(args.output / "plan.json"):
        raise ValueError("Frozen selections belong to another plan")
    if (args.output / "development.json").exists():
        raise ValueError("Preserve existing assessment")
    results = []
    for case in plan["cases"]:
        data = load_role(case, "development")
        training, stopping = load_role(case, "train"), load_role(case, "stop")
        for other in (training, stopping):
            assert not np.intersect1d(data["outer_train_positions"], other["outer_train_positions"]).size
        row = {"dataset": case["dataset_name"], "problem_type": case["problem_type"], "models": {}}
        for library in LIBRARIES:
            chosen = next(s for s in frozen["selections"] if s["dataset"] == case["dataset_name"] and s["library"] == library)
            if digest(chosen["model_path"]) != chosen["model_sha256"]:
                raise ValueError("Selected model changed")
            with Path(chosen["model_path"]).open("rb") as stream:
                model = pickle.load(stream)
            prediction = predict(model, data["X"])
            path = args.artifacts / "development" / case["dataset_name"] / f"{library}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, predictions=prediction, labels=data["y"], outer_train_positions=data["outer_train_positions"])
            row["models"][library] = {"error": error(prediction, data["y"], case["problem_type"]),
                                     "selection": chosen, "predictions_sha256": digest(path)}
        results.append(row)
    summary = []
    for comparator in ("xgboost", "lightgbm"):
        for family in ("all", "binary", "regression", "multiclass"):
            rows = [r for r in results if family == "all" or r["problem_type"] == family]
            gains = [(r["models"][comparator]["error"] - r["models"]["ctboost"]["error"]) / max(r["models"][comparator]["error"], 1e-12) for r in rows]
            summary.append({"comparator": comparator, "family": family, "datasets": len(rows),
                            "median_relative_error_gain": float(np.median(gains)), "wins": sum(g > 1e-12 for g in gains),
                            "ties": sum(abs(g) <= 1e-12 for g in gains), "worst_relative_regression": max(0, -min(gains))})
    passed = frozen["all_fits_successful"] and all(r["median_relative_error_gain"] >= 0 and r["wins"] + r["ties"] >= 6 for r in summary if r["family"] == "all")
    report = {"plan_sha256": digest(args.output / "plan.json"), "frozen_selections_sha256": digest(args.output / "frozen_selections.json"),
              "created_at": now(), "development_target_met": passed, "official_test_scored": False,
              "results": results, "summary": summary, "limitations": plan["limitations"]}
    write(args.output / "development.json", report)
    print(json.dumps({"development_target_met": passed, "summary": summary}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "fit", "development"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--dependencies", type=Path, required=True)
    parser.add_argument("--wave", type=int, choices=(3,), default=3)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--dataset")
    parser.add_argument("--library", choices=LIBRARIES)
    parser.add_argument("--slot", type=int, choices=(0, 1), default=0)
    arguments = parser.parse_args()
    for name in ("output", "artifacts", "package_root", "dependencies", "previous"):
        value = getattr(arguments, name)
        if value is not None:
            setattr(arguments, name, value.resolve())
    {"prepare": prepare, "run": run, "fit": fit_job, "development": development}[arguments.stage](arguments)
