"""Fixed-selected-configuration seed check; no official test access or new HPO."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import pickle
from pathlib import Path
import time
import traceback

BASE_PATH = Path(__file__).with_name("fair_accuracy_scout.py")
spec = importlib.util.spec_from_file_location("fair_accuracy_base", BASE_PATH)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
POLICY = base.ROOT / "benchmarks/results/improvements_20261016_followup/robustness_protocol_temperature.json"
REFIT_PATH = Path(__file__).with_name("fair_accuracy_leaf_refit_v2.py")
TEMPERATURE_PATH = Path(__file__).with_name("fixed_temperature.py")
temperature_spec = importlib.util.spec_from_file_location("fixed_temperature", TEMPERATURE_PATH)
temperature_helper = importlib.util.module_from_spec(temperature_spec)
temperature_spec.loader.exec_module(temperature_helper)


def predict(payload, frame):
    return temperature_helper.apply_temperature(payload, base.predict(payload, frame))


def refit_module():
    spec = importlib.util.spec_from_file_location("fair_leaf_refit", REFIT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(args):
    runtime = base.bootstrap(args)
    if args.output.exists() or args.artifacts.exists():
        raise ValueError("Preserve previous attempts; use fresh paths")
    previous = base.read(args.previous / "plan.json")
    frozen = base.read(args.previous / "frozen_selections.json")
    development = base.read(args.previous / "development.json")
    if not development["development_target_met"] or not frozen["all_fits_successful"]:
        raise ValueError("The original development gate must pass before checking seed947")
    if (previous["runtime"] != runtime
            or frozen["plan_sha256"] != base.digest(args.previous / "plan.json")
            or development["plan_sha256"] != base.digest(args.previous / "plan.json")
            or development["frozen_selections_sha256"] != base.digest(args.previous / "frozen_selections.json")):
        raise ValueError("Parent evidence or runtime changed")
    source_plans, dependencies = {}, {}
    directory = args.previous
    while directory:
        path = directory / "plan.json"
        plan = base.read(path)
        source_plans[base.digest(path)] = plan
        dependencies[str(path)] = base.digest(path)
        directory = Path(plan["previous"]) if plan["previous"] else None
    for name in ("frozen_selections.json", "development.json"):
        path = args.previous / name
        dependencies[str(path)] = base.digest(path)
    jobs = []
    for selected in frozen["selections"]:
        original_tree_selection = selected
        while original_tree_selection.get("leaf_refit") or "temperature" in original_tree_selection:
            original_tree_selection = original_tree_selection["base_selection"]
        source = source_plans[original_tree_selection["plan_sha256"]]
        case = next(c for c in previous["cases"] if c["dataset_name"] == selected["dataset"])
        jobs.append({"dataset": selected["dataset"], "library": selected["library"],
                     "controls": source["candidate_grids"][selected["library"]][case["problem_type"]][original_tree_selection["best"]["config_index"]],
                     "round_cap": source["round_cap"], "original_selection": selected,
                     "leaf_refit": selected.get("leaf_refit"), "temperature": selected.get("temperature")})
    dependencies[str(TEMPERATURE_PATH)] = base.digest(TEMPERATURE_PATH)
    if any(job["leaf_refit"] for job in jobs):
        for path in (REFIT_PATH, Path(__file__).with_name("fixed_leaf_refit.py")):
            dependencies[str(path)] = base.digest(path)
    expected = {(case["dataset_name"], library) for case in previous["cases"] for library in base.LIBRARIES}
    if (previous["seed"] != 47 or len(jobs) != 36
            or {(job["dataset"], job["library"]) for job in jobs} != expected):
        raise ValueError("Expected exactly36 unique seed47 parent selections")
    plan = {"protocol": "fixed_config_seed947_20261016", "created_at": base.now(),
            "script_sha256": base.digest(__file__), "base_script_sha256": base.digest(BASE_PATH),
            "policy_sha256": base.digest(POLICY), "runtime": runtime, "dependencies": dependencies,
            "scipy_version": importlib.metadata.version("scipy") if any(job["leaf_refit"] for job in jobs) else None,
            "previous": str(args.previous), "seed": 947, "patience": previous["patience"],
            "cases": previous["cases"], "jobs": jobs,
            "resources": {"workers": 1, "threads_per_fit": 1, "cpus": previous["resources"]["cpus"][:1],
                          "fit_times": "Diagnostic only; single process, no isolated performance claim",
                          "hard_limits": "No separate per-fit wall/RSS kill in these already bounded selected-configuration refits"},
            "selection": "Exactly one refit per dataset/library with the previously selected controls. Only learner seed changes. No HPO or replacement based on development.",
            "aggregation": "Mean primary error within each dataset over seeds47and947, then relative gains and median across12datasets. Not24independent tasks.",
            "official_test_scored": False}
    args.artifacts.mkdir(parents=True)
    base.write(args.output / "plan.json", plan)
    print({"plan_sha256": base.digest(args.output / "plan.json"), "fits": len(jobs)})


def checked(args):
    plan = base.read(args.output / "plan.json")
    if (plan["script_sha256"] != base.digest(__file__) or plan["base_script_sha256"] != base.digest(BASE_PATH)
            or plan["policy_sha256"] != base.digest(POLICY) or plan["runtime"] != base.bootstrap(args)):
        raise ValueError("Frozen implementation/runtime changed")
    if plan["scipy_version"] is not None and plan["scipy_version"] != importlib.metadata.version("scipy"):
        raise ValueError("Frozen correction solver changed")
    for path, expected in plan["dependencies"].items():
        if base.digest(path) != expected:
            raise ValueError("Frozen dependency changed")
    return plan


def run(args):
    import numpy as np
    import psutil
    plan = checked(args)
    psutil.Process().cpu_affinity([plan["resources"]["cpus"][0]])
    if (args.output / "fits").exists():
        raise ValueError("A started attempt is terminal; preserve it")
    (args.output / "fits").mkdir()
    selections = []
    for job in plan["jobs"]:
        while (args.artifacts / "pause_fits").exists():
            time.sleep(.5)
        case = next(c for c in plan["cases"] if c["dataset_name"] == job["dataset"])
        record = {"dataset": job["dataset"], "library": job["library"], "seed": plan["seed"],
                  "status": "failed", "started_at": base.now(), "original_selection": job["original_selection"]}
        try:
            train, stop = base.load_role(case, "train"), base.load_role(case, "stop")
            X, schema = base.normalize(train["X"])
            Xs, _ = base.normalize(stop["X"], schema)
            model, details = base.fit_configuration(job["library"], job["controls"],
                {**plan, "round_cap": job["round_cap"]}, case, X, train["y"], Xs, stop["y"], schema)
            if job["leaf_refit"]:
                if case["problem_type"] != "regression":
                    raise ValueError("Fixed leaf correction is regression-only")
                correction = job["leaf_refit"]
                refit = refit_module()
                h = refit.helper()
                design = h.prepare_design(model, X, train["y"])
                old_stop = base.predict(model, Xs)
                old_leaves = h.leaf_indices(model, Xs)
                coefficients, solve = refit.solve_coefficients(design, train["y"], correction["lambda"], correction["solver"])
                corrected, fold = refit.apply_correction(model, design, coefficients, correction["shrinkage"])
                deltas = h.corrections_by_tree(design["leaf_ids_by_tree"], coefficients, shrinkage=correction["shrinkage"])
                verified_train = refit.plain_receipt(h.verify_fold(model, corrected, X, deltas, fold, training_prepared=design["training_prepared"]))
                verified_stop = refit.plain_receipt(h.verify_fold(model, corrected, Xs, deltas, fold))
                np.testing.assert_array_equal(h.leaf_indices(corrected, X, training_prepared=design["training_prepared"]), design["training_leaves"])
                np.testing.assert_array_equal(h.leaf_indices(corrected, Xs), old_leaves)
                prediction_train = refit.training_prediction(corrected, X, design["training_prepared"])
                prediction_stop = base.predict(corrected, Xs)
                expected_train = design["baseline"] + correction["shrinkage"] * (design["Z"] @ coefficients)
                expected_stop = old_stop + correction["shrinkage"] * refit.contribution(old_leaves, design["leaf_ids_by_tree"], coefficients)
                for actual, expected in ((prediction_train, expected_train), (prediction_stop, expected_stop)):
                    np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=1e-5)
                model = corrected
                details.update(stop_error=base.error(prediction_stop, stop["y"], "regression"),
                               fixed_leaf_refit={"lambda": correction["lambda"], "shrinkage": correction["shrinkage"],
                                                "solve": solve, "fold": fold, "preprocessing": design["preprocessing_receipt"]})
                details["fixed_leaf_refit"].update(verification_train=verified_train, verification_stop=verified_stop)
            if job["temperature"] is not None:
                model = {**model, "temperature": job["temperature"]}
                details.update(temperature=job["temperature"], stop_error=base.error(predict(model, Xs), stop["y"], case["problem_type"]))
            path = args.artifacts / "models" / job["dataset"] / (job["library"] + ".pkl")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".candidate.pkl")
            with temporary.open("wb") as stream:
                pickle.dump(model, stream, protocol=5)
            with temporary.open("rb") as stream:
                restored = pickle.load(stream)
            np.testing.assert_array_equal(predict(model, Xs), predict(restored, Xs))
            temporary.replace(path)
            record.update(details, status="ok", model_path=str(path), model_sha256=base.digest(path),
                          model_bytes=path.stat().st_size, rss_after_fit_bytes=psutil.Process().memory_info().rss)
        except Exception:
            record["error"] = traceback.format_exc()
        record["finished_at"] = base.now()
        selections.append(record)
        base.write(args.output / "fits" / job["dataset"] / (job["library"] + ".json"), record)
        print({key: record.get(key) for key in ("dataset", "library", "status", "stop_error", "fit_seconds")}, flush=True)
    base.write(args.output / "frozen_selections.json", {"created_at": base.now(),
        "plan_sha256": base.digest(args.output / "plan.json"), "selections": selections,
        "all_fits_successful": all(s["status"] == "ok" for s in selections),
        "development_scored": False, "official_test_scored": False})


def aggregate(results):
    import numpy as np
    summaries = []
    for seed in ("47", "947", "mean"):
        for comparator in ("xgboost", "lightgbm"):
            for family in ("all", "binary", "regression", "multiclass"):
                rows = [r for r in results if family == "all" or r["problem_type"] == family]
                gains = [(r["models"][comparator]["errors"][seed] - r["models"]["ctboost"]["errors"][seed])
                         / max(r["models"][comparator]["errors"][seed], 1e-12) for r in rows]
                summaries.append({"seed": seed, "comparator": comparator, "family": family, "datasets": len(rows),
                    "median_relative_error_gain": float(np.median(gains)), "wins": sum(g > 1e-12 for g in gains),
                    "ties": sum(abs(g) <= 1e-12 for g in gains), "worst_relative_regression": max(0, -min(gains))})
    passed = all(s["median_relative_error_gain"] >= 0 and s["wins"] + s["ties"] >= 6
                 for s in summaries if s["seed"] == "mean" and s["family"] == "all")
    return summaries, passed


def development(args):
    import numpy as np
    plan = checked(args)
    frozen = base.read(args.output / "frozen_selections.json")
    if frozen["plan_sha256"] != base.digest(args.output / "plan.json") or not frozen["all_fits_successful"]:
        raise ValueError("All36frozen refits must succeed before assessment")
    if (args.output / "development.json").exists():
        raise ValueError("Preserve the existing assessment")
    original = base.read(Path(plan["previous"]) / "development.json")
    results = []
    for case in plan["cases"]:
        data = base.load_role(case, "development")
        original_row = next(r for r in original["results"] if r["dataset"] == case["dataset_name"])
        row = {"dataset": case["dataset_name"], "problem_type": case["problem_type"], "models": {}}
        for library in base.LIBRARIES:
            selected = next(s for s in frozen["selections"] if s["dataset"] == case["dataset_name"] and s["library"] == library)
            if base.digest(selected["model_path"]) != selected["model_sha256"]:
                raise ValueError("Selected model changed")
            with Path(selected["model_path"]).open("rb") as stream:
                model = pickle.load(stream)
            predictions = predict(model, data["X"])
            error947 = base.error(predictions, data["y"], case["problem_type"])
            error47 = original_row["models"][library]["error"]
            path = args.artifacts / "development" / case["dataset_name"] / (library + ".npz")
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, predictions=predictions, labels=data["y"], outer_train_positions=data["outer_train_positions"])
            row["models"][library] = {"errors": {"47": error47, "947": error947, "mean": (error47 + error947) / 2},
                                      "predictions947_sha256": base.digest(path), "selection947": selected}
        results.append(row)
    summary, passed = aggregate(results)
    report = {"created_at": base.now(), "plan_sha256": base.digest(args.output / "plan.json"),
              "frozen_selections_sha256": base.digest(args.output / "frozen_selections.json"),
              "development_target_met": passed, "official_test_scored": False, "results": results, "summary": summary}
    base.write(args.output / "development.json", report)
    print({"development_target_met": passed, "summary": [s for s in summary if s["family"] == "all"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "development"))
    for name in ("output", "artifacts", "package-root", "dependencies", "previous"):
        parser.add_argument("--" + name, type=lambda value: Path(value).resolve(), required=True)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "development": development}[args.stage](args)
