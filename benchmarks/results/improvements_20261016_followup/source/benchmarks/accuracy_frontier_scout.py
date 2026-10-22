"""Validation-selected accuracy/size scout; no production learning changes.

Run prepare, fit, then assess with the same arguments and fresh output paths.
The fit stage freezes every selection before the assess stage scores any test
row. Small, historically reused sklearn datasets are exploratory evidence only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

for _name in ("CTBOOST_HIST_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


SEEDS = (20261016, 20261017)
CHECKPOINTS = (10, 25, 50, 75, 100, 150, 200, 250, 300)
COMMON = {
    "iterations": 300, "learning_rate": 0.05, "max_depth": 6,
    "alpha": 0.05, "lambda_l2": 1.0, "subsample": 0.8,
    "bootstrap_type": "Bernoulli", "max_bins": 256,
    "feature_test": "quadratic", "multiclass_feature_test": "single",
    "task_type": "CPU", "boost_from_average": True,
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def dataset(name):
    loader = {"breast_cancer": load_breast_cancer, "diabetes": load_diabetes, "digits": load_digits}[name]
    data = loader()
    return np.asarray(data.data, dtype=np.float32), np.asarray(data.target, dtype=np.float32)


def native_runtime(package_root):
    # The local conda interpreter has a checkout-wide scikit-build editable
    # finder that otherwise overrides even an explicitly prepended package.
    sys.meta_path = [finder for finder in sys.meta_path if type(finder).__module__ != "_editable_skbc_ctboost"]
    sys.path.insert(0, str(package_root))
    import ctboost
    if not Path(ctboost.__file__).resolve().is_relative_to(package_root):
        raise RuntimeError("CTBoost resolved outside the frozen isolated package root")
    return ctboost, {
        "python": sys.executable, "python_version": platform.python_version(), "platform": platform.platform(),
        "package_path": ctboost.__file__, "native_path": ctboost._core.__file__,
        "native_sha256": digest(ctboost._core.__file__),
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scikit-learn", "xgboost", "psutil")},
        "ctboost_python_version": ctboost.__version__,
        "native_build_info": ctboost.build_info(),
        "python_sources_sha256": {str(path.relative_to(package_root)): digest(path)
                                   for path in sorted((package_root / "ctboost").rglob("*.py"))},
    }


def variants(problem):
    if problem == "multiclass":
        return {
            "baseline": {},
            "vector": {"multi_strategy": "multi_output_tree"},
            "refined": {"multi_strategy": "multi_output_tree", "multiclass_leaf_solver": "full", "leaf_estimation_iterations": 3},
            "refined_fast": {"multi_strategy": "multi_output_tree", "multiclass_leaf_solver": "full", "leaf_estimation_iterations": 3, "learning_rate": 0.15},
        }
    return {
        "baseline": {},
        "refined": {"leaf_estimation_iterations": 3, "leaf_estimation_backtracking": True},
        "refined_fast": {"leaf_estimation_iterations": 3, "leaf_estimation_backtracking": True, "learning_rate": 0.15},
    }


def loss(raw, y, problem):
    raw, y = np.asarray(raw, dtype=np.float64), np.asarray(y)
    if not np.isfinite(raw).all():
        raise ValueError("Non-finite model predictions")
    if problem == "regression":
        return float(np.sqrt(np.mean((raw - y) ** 2)))
    if problem == "binary":
        return float(np.mean(np.logaddexp(0.0, raw) - y * raw))
    shifted = raw - raw.max(axis=1, keepdims=True)
    return float(np.mean(np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(y)), y.astype(int)]))


def prepare(args):
    if args.output.exists() or args.artifacts.exists():
        raise ValueError("Use fresh output and artifact directories; preserve previous attempts")
    _, runtime = native_runtime(args.package_root)
    cases = []
    for name, problem in (("breast_cancer", "binary"), ("diabetes", "regression"), ("digits", "multiclass")):
        X, y = dataset(name)
        for seed in SEEDS:
            indices = np.arange(len(y))
            train_valid, test = train_test_split(indices, test_size=0.2, random_state=seed,
                                                stratify=None if problem == "regression" else y)
            train, validation = train_test_split(train_valid, test_size=0.25, random_state=seed + 100,
                                                 stratify=None if problem == "regression" else y[train_valid])
            roles = {"train": train, "validation": validation, "test": test}
            assert len(np.unique(np.concatenate(list(roles.values())))) == len(y)
            case = {"id": f"{name}_{seed}", "dataset": name, "problem": problem, "seed": seed,
                    "shape": list(X.shape), "classes": int(len(np.unique(y))) if problem != "regression" else 1,
                    "source_sha256": hashlib.sha256(X.tobytes() + y.tobytes()).hexdigest(), "roles": {}}
            for role, positions in roles.items():
                path = args.artifacts / "data" / case["id"] / f"{role}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, X=X[positions], y=y[positions], indices=positions)
                case["roles"][role] = {"path": str(path), "sha256": digest(path), "rows": len(positions), "indices": positions.tolist()}
            cases.append(case)
    write(args.output / "plan.json", {
        "protocol": "accuracy_frontier_20261016_v1", "script_sha256": digest(__file__),
        "runtime": runtime, "cases": cases, "common_parameters": COMMON,
        "variants": {p: variants(p) for p in ("binary", "regression", "multiclass")},
        "split": "60/20/20 train/validation/test; classification stratified, two predeclared seeds; no test scoring until every choice is frozen",
        "selection": {"patience": 40, "checkpoints": CHECKPOINTS, "relative_validation_tolerance": 0.01,
                      "quality": "Minimum validation loss across all retained/checkpoint CTBoost models; ties prefer fewer physical nodes, then rounds, then variant name",
                      "compact": "Fewest physical nodes among checkpoints within 1% of globally best validation loss; ties prefer lower validation loss, fewer rounds, variant name. This targets topology storage/traversal, not serialized bytes or vector-leaf payload bytes.",
                      "baseline": "Default learner controls and learning rate0.05; native validation early stopping selects rounds"},
        "competitor": {"library": "xgboost", "parameters": {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6, "max_bin": 256,
            "reg_lambda": 1.0, "subsample": 0.8, "colsample_bytree": 1.0, "tree_method": "hist", "n_jobs": 1, "early_stopping_rounds": 40},
            "budget_note": "One fixed XGBoost fit per split; CTBoost searches 3 or4 fits. Equal per-fit caps are not equal total search cost; this is not exhaustive competitor HPO."},
        "resources": {"threads": 1, "logical_cpu": args.cpu, "fit_order": "Sequential; all CTBoost variants and competitor per case; no simultaneous inference benchmarks"},
        "hypothesis": "Leaf optimization and validation-selected shrinkage/round budgets may improve loss per stored node while keeping feature tests before cuts. Vector layout alone should preserve predictions exactly.",
        "limitations": "Exploratory small bundled real datasets; breast_cancer and digits were used in earlier different40-round studies. New splits overlap older data and are not an independent confirmation panel. No TabArena, Elo, general ranking, or default change claim.",
        "expected_fits": 26,
    })
    print(json.dumps({"prepared": len(cases), "plan_sha256": digest(args.output / "plan.json")}))


def checked(args):
    plan = read(args.output / "plan.json")
    if plan["script_sha256"] != digest(__file__):
        raise ValueError("Runner changed after protocol freeze")
    ctboost, runtime = native_runtime(args.package_root)
    if runtime != plan["runtime"]:
        raise ValueError("Frozen runtime changed")
    import psutil
    psutil.Process().cpu_affinity([plan["resources"]["logical_cpu"]])
    return ctboost, plan


def role(case, name):
    metadata = case["roles"][name]
    if digest(metadata["path"]) != metadata["sha256"]:
        raise ValueError("Frozen dataset role changed")
    with np.load(metadata["path"]) as loaded:
        np.testing.assert_array_equal(loaded["indices"], metadata["indices"])
        return loaded["X"].copy(), loaded["y"].copy()


def fit(args):
    ctboost, plan = checked(args)
    import psutil
    import xgboost as xgb
    if (args.output / "fits.json").exists():
        raise ValueError("Preserve started fits; use a new protocol directory for another run")
    records, selections = [], []
    for case in plan["cases"]:
        X, y = role(case, "train")
        Xv, yv = role(case, "validation")
        pool, validation = ctboost.Pool(X, y), ctboost.Pool(Xv, yv)
        state_records, frontier, baseline_raw = [], [], None
        for variant, controls in plan["variants"][case["problem"]].items():
            params = {**plan["common_parameters"], **controls, "random_seed": case["seed"],
                      "objective": {"binary": "Logloss", "regression": "RMSE", "multiclass": "MultiClass"}[case["problem"]],
                      "num_classes": case["classes"]}
            started = time.perf_counter()
            model = ctboost.train(pool, params, eval_set=validation, early_stopping_rounds=plan["selection"]["patience"])
            seconds = time.perf_counter() - started
            path = args.artifacts / "models" / case["id"] / f"{variant}.ctb"
            path.parent.mkdir(parents=True, exist_ok=True)
            model.save_model(path)
            state = model._handle.export_state()
            retained = model.num_iterations_trained
            per_round = 1 if model.multi_strategy == "multi_output_tree" else model.prediction_dimension
            checkpoints = sorted({retained, *[n for n in plan["selection"]["checkpoints"] if n < retained]})
            record = {"case": case["id"], "variant": variant, "params": params, "fit_seconds": seconds,
                      "retained_rounds": retained, "model_path": str(path), "model_sha256": digest(path),
                      "model_bytes": path.stat().st_size, "physical_trees": len(state["trees"]),
                      "physical_nodes": sum(len(t["nodes"]) for t in state["trees"]),
                      "rss_after_fit_bytes": psutil.Process().memory_info().rss, "checkpoints": []}
            for rounds in checkpoints:
                raw = model.predict(Xv, num_iteration=rounds)
                entry = {"variant": variant, "rounds": rounds, "validation_loss": loss(raw, yv, case["problem"]),
                         "physical_nodes": sum(len(t["nodes"]) for t in state["trees"][:rounds * per_round])}
                frontier.append(entry)
                record["checkpoints"].append(entry)
            current_raw = model.predict(Xv)
            if variant == "baseline":
                baseline_raw = current_raw
            if variant == "vector":
                np.testing.assert_array_equal(current_raw, baseline_raw)
                record["vector_validation_predictions_exact"] = True
            state_records.append(record)
            records.append(record)
            write(args.output / "fits.json", {"plan_sha256": digest(args.output / "plan.json"), "records": records})
            print(json.dumps({"case": case["id"], "variant": variant, "fit_seconds": seconds, "rounds": retained}), flush=True)
        best = min(frontier, key=lambda r: (r["validation_loss"], r["physical_nodes"], r["rounds"], r["variant"]))
        acceptable = [r for r in frontier if r["validation_loss"] <= best["validation_loss"] * (1 + plan["selection"]["relative_validation_tolerance"])]
        compact = min(acceptable, key=lambda r: (r["physical_nodes"], r["validation_loss"], r["rounds"], r["variant"]))
        params = {**plan["competitor"]["parameters"], "random_state": case["seed"]}
        if case["problem"] == "regression":
            competitor = xgb.XGBRegressor(**params, objective="reg:squarederror", eval_metric="rmse")
        else:
            competitor = xgb.XGBClassifier(**params, eval_metric="logloss" if case["problem"] == "binary" else "mlogloss")
        started = time.perf_counter()
        competitor.fit(X, y, eval_set=[(Xv, yv)], verbose=False)
        seconds = time.perf_counter() - started
        retained = int(competitor.best_iteration) + 1
        booster = competitor.get_booster()[:retained]
        path = args.artifacts / "models" / case["id"] / "xgboost.ubj"
        booster.save_model(path)
        dump = booster.trees_to_dataframe()
        raw = booster.predict(xgb.DMatrix(Xv), output_margin=True)
        records.append({"case": case["id"], "variant": "xgboost", "fit_seconds": seconds,
                        "params": competitor.get_params(), "retained_rounds": retained,
                        "model_path": str(path), "model_sha256": digest(path), "model_bytes": path.stat().st_size,
                        "physical_trees": int(dump.Tree.nunique()), "physical_nodes": len(dump),
                        "validation_loss": loss(raw, yv, case["problem"]), "rss_after_fit_bytes": psutil.Process().memory_info().rss})
        selections.append({"case": case["id"], "quality": best, "compact": compact,
                           "ctboost_search_fit_seconds": sum(r["fit_seconds"] for r in state_records),
                           "ctboost_search_fits": len(state_records)})
        write(args.output / "fits.json", {"plan_sha256": digest(args.output / "plan.json"), "records": records})
        print(json.dumps({"case": case["id"], "variant": "xgboost", "fit_seconds": seconds, "rounds": retained}), flush=True)
    assert len(records) == plan["expected_fits"]
    write(args.output / "frozen_selections.json", {"plan_sha256": digest(args.output / "plan.json"),
          "fits_sha256": digest(args.output / "fits.json"), "test_scoring_performed": False, "selections": selections})


def assess(args):
    ctboost, plan = checked(args)
    import xgboost as xgb
    from ctboost.training._early_stopping import _prune_booster_to_best_iteration
    frozen = read(args.output / "frozen_selections.json")
    if frozen["plan_sha256"] != digest(args.output / "plan.json") or frozen["fits_sha256"] != digest(args.output / "fits.json"):
        raise ValueError("Frozen selections or fits changed")
    if (args.output / "assessment.json").exists():
        raise ValueError("Preserve assessment results")
    fits = read(args.output / "fits.json")["records"]
    results = []
    for case, selected in zip(plan["cases"], frozen["selections"]):
        assert case["id"] == selected["case"]
        Xt, yt = role(case, "test")
        Xv, yv = role(case, "validation")
        models, predictions = {}, {}
        records = {r["variant"]: r for r in fits if r["case"] == case["id"]}
        policies = {name: {"variant": name, "rounds": record["retained_rounds"]} for name, record in records.items()}
        policies.update({name: selected[name] for name in ("quality", "compact")})
        for policy, selection in policies.items():
            variant, rounds = selection["variant"], selection["rounds"]
            record = records[variant]
            if digest(record["model_path"]) != record["model_sha256"]:
                raise ValueError("Saved model changed")
            if variant == "xgboost":
                model = xgb.Booster()
                model.load_model(record["model_path"])
                raw = model.predict(xgb.DMatrix(Xt), output_margin=True)
                entry = {key: record[key] for key in ("model_bytes", "physical_trees", "physical_nodes", "validation_loss")}
            else:
                if variant not in models:
                    models[variant] = ctboost.Booster.load_model(record["model_path"])
                original = models[variant]
                model = ctboost.Booster(_prune_booster_to_best_iteration(original._handle, rounds - 1))
                raw = model.predict(Xt)
                np.testing.assert_array_equal(raw, original.predict(Xt, num_iteration=rounds))
                path = args.artifacts / "selected" / case["id"] / f"{policy}.ctb"
                path.parent.mkdir(parents=True, exist_ok=True)
                model.save_model(path)
                restored = ctboost.Booster.load_model(path)
                np.testing.assert_array_equal(restored.predict(Xt), raw)
                state = model._handle.export_state()
                entry = {"model_bytes": path.stat().st_size, "physical_trees": len(state["trees"]),
                         "physical_nodes": sum(len(t["nodes"]) for t in state["trees"]),
                         "validation_loss": loss(model.predict(Xv), yv, case["problem"]),
                         "pruning_and_reload_exact": True, "selected_model_sha256": digest(path)}
                if policy in ("quality", "compact"):
                    np.testing.assert_allclose(entry["validation_loss"], selection["validation_loss"], rtol=0, atol=1e-12)
            predictions[policy] = raw
            result = {"case": case["id"], "dataset": case["dataset"], "seed": case["seed"], "problem": case["problem"],
                      "policy": policy, "variant": variant, "rounds": rounds, "test_loss": loss(raw, yt, case["problem"]),
                      "selected_fit_seconds": record["fit_seconds"], **entry}
            if case["problem"] == "binary":
                result["test_auc"] = float(roc_auc_score(yt, raw))
            elif case["problem"] == "multiclass":
                result["test_accuracy"] = float(accuracy_score(yt, np.argmax(raw, axis=1)))
            results.append(result)
        if case["problem"] == "multiclass":
            np.testing.assert_array_equal(predictions["baseline"], predictions["vector"])
        path = args.artifacts / "predictions" / f"{case['id']}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, labels=yt, **predictions)
    comparisons = []
    for case in plan["cases"]:
        rows = {r["policy"]: r for r in results if r["case"] == case["id"]}
        baseline = rows["baseline"]
        for policy, row in rows.items():
            comparisons.append({"case": case["id"], "policy": policy,
                "relative_test_loss_gain_vs_baseline": 1 - row["test_loss"] / baseline["test_loss"],
                "nodes_ratio_vs_baseline": row["physical_nodes"] / baseline["physical_nodes"],
                "bytes_ratio_vs_baseline": row["model_bytes"] / baseline["model_bytes"]})
    summary = []
    for policy in sorted({r["policy"] for r in comparisons}):
        rows = [r for r in comparisons if r["policy"] == policy]
        gains = [r["relative_test_loss_gain_vs_baseline"] for r in rows]
        summary.append({"policy": policy, "splits": len(rows), "wins": sum(g > 1e-12 for g in gains),
                        "median_relative_test_loss_gain": float(np.median(gains)),
                        "worst_relative_test_loss_regression": float(max(0, -min(gains))),
                        "median_nodes_ratio": float(np.median([r["nodes_ratio_vs_baseline"] for r in rows])),
                        "median_bytes_ratio": float(np.median([r["bytes_ratio_vs_baseline"] for r in rows]))})
    dataset_summary = []
    for name in sorted({r["dataset"] for r in results}):
        baseline_loss = np.mean([r["test_loss"] for r in results if r["dataset"] == name and r["policy"] == "baseline"])
        for policy in sorted({r["policy"] for r in results if r["dataset"] == name}):
            rows = [r for r in results if r["dataset"] == name and r["policy"] == policy]
            mean_loss = float(np.mean([r["test_loss"] for r in rows]))
            dataset_summary.append({"dataset": name, "policy": policy, "seeds": len(rows), "mean_test_loss": mean_loss,
                                    "relative_mean_test_loss_gain_vs_baseline": 1 - mean_loss / baseline_loss,
                                    "mean_physical_nodes": float(np.mean([r["physical_nodes"] for r in rows])),
                                    "mean_model_bytes": float(np.mean([r["model_bytes"] for r in rows]))})
    write(args.output / "assessment.json", {"plan_sha256": digest(args.output / "plan.json"),
        "frozen_selections_sha256": digest(args.output / "frozen_selections.json"), "results": results,
        "comparisons": comparisons, "summary": summary, "dataset_summary": dataset_summary,
        "aggregation_note": "Two split seeds per dataset are correlated; split-level medians are descriptive, not six independent tasks. Dataset summaries average both seeds before comparison.",
        "vector_test_predictions_exact": True,
        "limitations": plan["limitations"], "timing_note": "Fit seconds are descriptive single-thread local observations; no inference speed measured in this accuracy scout."})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "fit", "assess"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--package-root", required=True, type=Path)
    parser.add_argument("--cpu", type=int, default=4)
    arguments = parser.parse_args()
    for field in ("output", "artifacts", "package_root"):
        setattr(arguments, field, getattr(arguments, field).resolve())
    {"prepare": prepare, "fit": fit, "assess": assess}[arguments.stage](arguments)
