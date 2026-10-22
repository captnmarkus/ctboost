"""Mandatory TRAIN+STOP refit at selected rounds; no new model selection."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
from pathlib import Path
import pickle
import time
import traceback

BASE_PATH = Path(__file__).with_name("fair_accuracy_scout.py")
REFIT_PATH = Path(__file__).with_name("fair_accuracy_leaf_refit_v2.py")
TEMPERATURE_PATH = Path(__file__).with_name("fixed_temperature.py")
POLICY = Path(__file__).parent / "results/improvements_20261016_followup/robustness_protocol_union.json"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_module("union_base", BASE_PATH)
refit = load_module("union_leaf_refit", REFIT_PATH)
temperature = load_module("union_temperature", TEMPERATURE_PATH)
raw_predict = base.predict


def predict(payload, frame):
    return temperature.apply_temperature(payload, raw_predict(payload, frame))


def union_data(case):
    import numpy as np
    import pandas as pd
    train, stop = base.load_role(case, "train"), base.load_role(case, "stop")
    train_ids, stop_ids = train["outer_train_positions"], stop["outer_train_positions"]
    if np.intersect1d(train_ids, stop_ids).size:
        raise ValueError("TRAIN and STOP rows overlap")
    ids = np.concatenate([train_ids, stop_ids])
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Duplicate union training rows")
    frame = pd.concat([train["X"], stop["X"]], ignore_index=True)
    labels = np.concatenate([train["y"], stop["y"]])
    X, schema = base.normalize(frame)
    return X, labels, schema, train, stop, {
        "train_rows": len(train_ids), "stop_rows": len(stop_ids), "union_rows": len(ids),
        "train_ids_sha256": base.hashlib.sha256(train_ids.astype("<i8").tobytes()).hexdigest(),
        "stop_ids_sha256": base.hashlib.sha256(stop_ids.astype("<i8").tobytes()).hexdigest(),
        "union_ids_sha256": base.hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest(),
        "source_roles": {name: case["roles"][name] for name in ("train", "stop")},
        "union_category_counts": {str(column): len(values) for column, values in schema.items()},
        "vocabulary_fit_on": "TRAIN+STOP union only", "row_order": "original TRAIN followed by original STOP"}


def fit_fixed(job, case, X, y, schema, seed):
    import ctboost
    import lightgbm as lgb
    import xgboost as xgb
    params = dict(job["parameters"])
    rounds, library, problem = job["rounds"], job["library"], case["problem_type"]
    for key in ("early_stopping_rounds", "early_stopping", "callbacks", "eval_set", "evals", "valid_sets", "use_best_model"):
        if params.get(key) not in (None, False, [], ()):
            raise ValueError(f"Inherited validation/stopping control is forbidden: {key}")
    if not isinstance(rounds, int) or rounds <= 0:
        raise ValueError("Expected a positive frozen selected-round count")
    started = time.perf_counter()
    if library == "ctboost":
        params.update(iterations=rounds, random_seed=seed, cat_features=list(schema) or None)
        estimator = (ctboost.CTBoostRegressor(**params) if problem == "regression" else
                     ctboost.CTBoostClassifier(**params, loss_function="MultiClass" if problem == "multiclass" else "Logloss"))
        estimator.fit(X, y)
        model = estimator.get_booster()
        actual = model.num_iterations_trained
    elif library == "xgboost":
        params["seed"] = seed
        model = xgb.train(params, xgb.DMatrix(X, label=y, enable_categorical=True),
                          num_boost_round=rounds, verbose_eval=False)
        actual = model.num_boosted_rounds()
    else:
        params["seed"] = seed
        model = lgb.train(params, lgb.Dataset(X, label=y, categorical_feature=list(schema) or "auto", free_raw_data=False),
                          num_boost_round=rounds)
        actual = model.current_iteration()
    if not 0 < actual <= rounds or (library != "lightgbm" and actual != rounds):
        raise ValueError(f"Unexpected fixed-round fit: requested{rounds}, actual{actual}")
    # LightGBM may stop structurally when no additional split is possible; this
    # is recorded and never chosen by validation or development scoring.
    return {"library": library, "model": model, "schema": schema, "problem": problem}, {
        "parameters": params, "requested_rounds": rounds, "retained_rounds": int(actual),
        "structural_early_termination": actual < rounds, "fit_seconds": time.perf_counter() - started,
        "early_stopping": False}


def fit_policy(job, case, seed):
    import numpy as np
    X, y, schema, train, stop, data_receipt = union_data(case)
    payload, details = fit_fixed(job, case, X, y, schema, seed)
    details["union_data"] = data_receipt
    if job["leaf_refit"]:
        correction = job["leaf_refit"]
        h = refit.helper()
        design = h.prepare_design(payload, X, y)
        coefficients, solve = refit.solve_coefficients(design, y, correction["lambda"], correction["solver"])
        corrected, fold = refit.apply_correction(payload, design, coefficients, correction["shrinkage"])
        deltas = h.corrections_by_tree(design["leaf_ids_by_tree"], coefficients, shrinkage=correction["shrinkage"])
        union_check = refit.plain_receipt(h.verify_fold(payload, corrected, X, deltas, fold,
                                                      training_prepared=design["training_prepared"]))
        expected = design["baseline"] + correction["shrinkage"] * (design["Z"] @ coefficients)
        np.testing.assert_allclose(h.predict(corrected, X, training_prepared=design["training_prepared"]),
                                   expected, rtol=5e-5, atol=1e-5)
        inference_checks = {}
        for name, role in (("original_train", train), ("original_stop", stop)):
            frame, _ = base.normalize(role["X"], schema)
            inference_checks[name] = refit.plain_receipt(h.verify_fold(payload, corrected, frame, deltas, fold))
        payload = corrected
        details["fixed_leaf_refit"] = {"lambda": correction["lambda"], "shrinkage": correction["shrinkage"],
            "solve": solve, "fold": fold, "ordered_union_check": union_check, "inference_checks": inference_checks,
            "preprocessing": design["preprocessing_receipt"]}
    if job["temperature"] is not None:
        payload = {**payload, "temperature": job["temperature"]}
        details["temperature"] = job["temperature"]
    # These rows now belong to fitting. This metric is diagnostic only and
    # cannot select between this mandatory refit and its original incumbent.
    details["apparent_union_error_inference_transform"] = base.error(predict(payload, X), y, case["problem_type"])
    return payload, details, stop["X"]


def prepare(args):
    runtime = base.bootstrap(args)
    if args.output.exists() or args.artifacts.exists():
        raise ValueError("Preserve earlier attempts; use fresh paths")
    previous = base.read(args.previous / "plan.json")
    frozen = base.read(args.previous / "frozen_selections.json")
    development = base.read(args.previous / "development.json")
    if (previous["wave"] != 8 or previous["runtime"] != runtime or not frozen["all_fits_successful"]
            or development["development_target_met"]
            or frozen["plan_sha256"] != base.digest(args.previous / "plan.json")
            or development["plan_sha256"] != base.digest(args.previous / "plan.json")
            or development["frozen_selections_sha256"] != base.digest(args.previous / "frozen_selections.json")):
        raise ValueError("Expected intact failed wave8 gate")
    jobs = []
    for selected in frozen["selections"]:
        if base.digest(selected["model_path"]) != selected["model_sha256"]:
            raise ValueError("Parent model changed")
        jobs.append({"dataset": selected["dataset"], "library": selected["library"],
            "parameters": selected["best"]["parameters"], "rounds": selected["best"]["retained_rounds"],
            "leaf_refit": selected.get("leaf_refit"), "temperature": selected.get("temperature"),
            "original_selection": selected})
    expected = {(case["dataset_name"], library) for case in previous["cases"] for library in base.LIBRARIES}
    if len(previous["cases"]) != 12 or len(expected) != 36 or len(jobs) != 36 or {
            (job["dataset"], job["library"]) for job in jobs} != expected:
        raise ValueError("Expected twelve unique cases and36 unique jobs")
    dependencies = {str(path): base.digest(path) for path in
                    (BASE_PATH, REFIT_PATH, refit.HELPER_PATH, TEMPERATURE_PATH, POLICY)}
    dependencies.update({str(args.previous / name): base.digest(args.previous / name)
                         for name in ("plan.json", "frozen_selections.json", "development.json")})
    plan = {"protocol": "mandatory_train_stop_union_fixed_rounds_20261016", "wave": 9,
        "created_at": base.now(), "script_sha256": base.digest(__file__), "dependencies": dependencies,
        "runtime": runtime, "scipy_version": importlib.metadata.version("scipy"), "previous": str(args.previous),
        "cases": previous["cases"], "jobs": jobs, "seed": 47, "patience": None,
        "resources": {"workers": 1, "threads_per_fit": 1, "cpus": [4],
                      "fit_times": "Diagnostic only; one process and already bounded selected round counts"},
        "target": previous["target"], "official_test_scored": False,
        "selection": "Mandatory one refit perlibrary/task on TRAIN+STOP union at frozen selected rounds/controls. No early stopping, HPO, original-versus-refit choice or DEV-driven fallback. Reapply frozen regression ridge and multiclass temperature without reselection. All36models freeze before DEV.",
        "limitations": "Historically reused adaptive development. Training data increases equally for all libraries; any gain is standard final-refit pipeline evidence, not a CT-specific algorithm improvement. Previously selected ridge penalties and temperatures may be suboptimal after the training set changes. LightGBM structural termination without a split is recorded, never validation-selected. No official-test/confirmation access."}
    args.artifacts.mkdir(parents=True)
    base.write(args.output / "plan.json", plan)
    print({"plan_sha256": base.digest(args.output / "plan.json"), "fits": len(jobs)})


def checked(args):
    plan = base.read(args.output / "plan.json")
    if (plan["script_sha256"] != base.digest(__file__) or plan["runtime"] != base.bootstrap(args)
            or plan["scipy_version"] != importlib.metadata.version("scipy") or args.previous != Path(plan["previous"])):
        raise ValueError("Frozen code/runtime/parent changed")
    for path, expected in plan["dependencies"].items():
        if base.digest(path) != expected:
            raise ValueError("Frozen dependency changed")
    return plan


def run(args):
    import numpy as np
    import psutil
    plan = checked(args)
    psutil.Process().cpu_affinity(plan["resources"]["cpus"])
    if (args.output / "fits").exists():
        raise ValueError("A started attempt is terminal; preserve it")
    (args.output / "fits").mkdir()
    selections, successful = [], True
    for job in plan["jobs"]:
        while (args.artifacts / "pause_fits").exists():
            time.sleep(.5)
        case = next(case for case in plan["cases"] if case["dataset_name"] == job["dataset"])
        record = {"dataset": job["dataset"], "library": job["library"], "status": "failed", "started_at": base.now()}
        try:
            payload, details, verification_frame = fit_policy(job, case, plan["seed"])
            path = args.artifacts / "models" / job["dataset"] / (job["library"] + ".pkl")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".candidate.pkl")
            with temporary.open("wb") as stream:
                pickle.dump(payload, stream, protocol=5)
            with temporary.open("rb") as stream:
                restored = pickle.load(stream)
            np.testing.assert_array_equal(predict(payload, verification_frame), predict(restored, verification_frame))
            temporary.replace(path)
            selected = {**job["original_selection"], "base_selection": job["original_selection"],
                "source_wave": 9, "plan_sha256": base.digest(args.output / "plan.json"), "mandatory_union_refit": True,
                "model_path": str(path), "model_sha256": base.digest(path), "model_bytes": path.stat().st_size,
                "best": {**job["original_selection"]["best"], "parameters": details["parameters"],
                         "retained_rounds": details["retained_rounds"], "stop_error": None, "selection_key": None},
                "refit_details": details}
            selections.append(selected)
            record.update(status="ok", selection=selected)
        except Exception:
            successful = False
            record["error"] = traceback.format_exc()
        record["finished_at"] = base.now()
        base.write(args.output / "fits" / job["dataset"] / (job["library"] + ".json"), record)
        print({"dataset": job["dataset"], "library": job["library"], "status": record["status"]}, flush=True)
    base.write(args.output / "frozen_selections.json", {"plan_sha256": base.digest(args.output / "plan.json"),
        "created_at": base.now(), "selections": selections, "all_fits_successful": successful and len(selections) == 36,
        "development_scored": False, "official_test_scored": False})


def development(args):
    if not base.read(args.output / "frozen_selections.json")["all_fits_successful"]:
        raise ValueError("All36 mandatory refits must succeed before DEV")
    base.checked, base.predict = checked, predict
    base.development(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "development"))
    for name in ("output", "artifacts", "package-root", "dependencies", "previous"):
        parser.add_argument("--" + name, type=lambda value: Path(value).resolve(), required=True)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "development": development}[args.stage](args)
