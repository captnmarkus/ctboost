"""Four-arm conditional-test/leaf-threshold search, then mandatory union refit."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
from pathlib import Path
import pickle
import time
import traceback

BINARY_PATH = Path(__file__).with_name("fair_accuracy_binary_refit.py")
ENSEMBLE_PATH = Path(__file__).with_name("fair_accuracy_seed_ensemble.py")
SEED_PATH = Path(__file__).with_name("fair_accuracy_threshold_seed_check.py")
OUTER_PATH = Path(__file__).with_name("fair_accuracy_threshold_outer_test.py")
POLICY = Path(__file__).parent / "results/improvements_20261016_followup/robustness_protocol_threshold.json"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


binary = load_module("threshold_binary", BINARY_PATH)
union, base, predict, fit_policy = binary.union, binary.base, binary.predict, binary.fit_policy


def grid(job):
    params = dict(job["parameters"])
    library = job["library"]
    if library == "ctboost":
        params.pop("iterations", None)
        lr = params["learning_rate"]
        return [{**params, "alpha": alpha, "learning_rate": rate} for alpha in (.5, .8) for rate in (lr, lr / 2)]
    if library == "xgboost":
        lr, minimum = params.get("eta", .3), params.get("min_child_weight", 1.)
        return [{**params, "min_child_weight": value, "eta": rate}
                for value in (minimum, max(.25, minimum / 4)) for rate in (lr, lr / 2)]
    lr, minimum = params.get("learning_rate", .1), params.get("min_data_in_leaf", 20)
    return [{**params, "min_data_in_leaf": value, "learning_rate": rate}
            for value in (minimum, max(1, int(minimum) // 4)) for rate in (lr, lr / 2)]


def canonical(params, library):
    params = dict(params)
    if library == "ctboost":
        params.pop("iterations", None)
    elif library == "xgboost":
        params.setdefault("min_child_weight", 1.)
    else:
        params.setdefault("min_data_in_leaf", 20)
    return params


def correct(payload, job, X, labels, Xs):
    """Fixed TRAIN-only correction of a fresh native candidate."""
    import numpy as np
    if job.get("binary_refit"):
        return binary.corrected(payload, X, labels, job["binary_refit"], {"stop": Xs})
    if not job.get("leaf_refit"):
        return payload, None
    correction, r = job["leaf_refit"], union.refit
    h = r.helper()
    design = h.prepare_design(payload, X, labels)
    coefficients, solve = r.solve_coefficients(design, labels, correction["lambda"], correction["solver"])
    candidate, fold = r.apply_correction(payload, design, coefficients, correction["shrinkage"])
    deltas = h.corrections_by_tree(design["leaf_ids_by_tree"], coefficients, correction["shrinkage"])
    train_check = h.verify_fold(payload, candidate, X, deltas, fold, training_prepared=design["training_prepared"])
    stop_check = h.verify_fold(payload, candidate, Xs, deltas, fold)
    np.testing.assert_allclose(h.predict(candidate, X, design["training_prepared"]),
        design["baseline"] + correction["shrinkage"] * (design["Z"] @ coefficients), rtol=5e-5, atol=1e-5)
    return candidate, r.plain_receipt({"solve": solve, "fold": fold, "training_check": train_check,
                                      "stop_check": stop_check, "preprocessing": design["preprocessing_receipt"]})


def persist(payload, path, frame):
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".candidate.pkl")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=5)
    with temporary.open("rb") as stream:
        restored = pickle.load(stream)
    np.testing.assert_array_equal(predict(payload, frame), predict(restored, frame))
    temporary.replace(path)
    return {"model_path": str(path), "model_sha256": base.digest(path), "model_bytes": path.stat().st_size}


def prepare(args):
    runtime = base.bootstrap(args)
    if args.output.exists() or args.artifacts.exists():
        raise ValueError("Preserve all previous attempts")
    parent = base.read(args.previous / "plan.json")
    frozen = base.read(args.previous / "frozen_selections.json")
    development = base.read(args.previous / "development.json")
    if (parent["wave"] != 11 or parent["phase"] != "initial" or parent["runtime"] != runtime
            or parent["script_sha256"] != base.digest(ENSEMBLE_PATH) or not frozen["all_fits_successful"]
            or development["development_target_met"] or parent["scipy_version"] != importlib.metadata.version("scipy")
            or frozen["plan_sha256"] != base.digest(args.previous / "plan.json")
            or development["plan_sha256"] != base.digest(args.previous / "plan.json")
            or development["frozen_selections_sha256"] != base.digest(args.previous / "frozen_selections.json")):
        raise ValueError("Expected intact failed first ensemble policy")
    singles_path = Path(parent["previous"])
    singles = base.read(singles_path / "plan.json")
    single_frozen = base.read(singles_path / "frozen_selections.json")
    original_path = Path(singles["training_parent"])
    original_plan = base.read(original_path / "plan.json")
    original_frozen = base.read(original_path / "frozen_selections.json")
    if (singles["wave"] != 10 or original_plan["wave"] != 8 or
            single_frozen["plan_sha256"] != base.digest(singles_path / "plan.json") or
            original_frozen["plan_sha256"] != base.digest(original_path / "plan.json") or
            not single_frozen["all_fits_successful"] or not original_frozen["all_fits_successful"]):
        raise ValueError("Expected intact frozen single and original TRAIN policies")
    expected = {(c["dataset_name"], library) for c in parent["cases"] for library in base.LIBRARIES}
    if len(parent["cases"]) != 12 or len(expected) != 36:
        raise ValueError("Expected12 unique tasks")
    for document in (single_frozen, original_frozen):
        if len(document["selections"]) != 36 or {(s["dataset"], s["library"]) for s in document["selections"]} != expected:
            raise ValueError("Expected36 unique parent models")
        for selected in document["selections"]:
            if base.digest(selected["model_path"]) != selected["model_sha256"]:
                raise ValueError("Parent native model changed")
    dependencies = dict(parent["dependencies"])
    for path, expected_hash in dependencies.items():
        if base.digest(path) != expected_hash:
            raise ValueError("Frozen implementation changed")
    for path in (BINARY_PATH, ENSEMBLE_PATH, POLICY, SEED_PATH, OUTER_PATH):
        dependencies[str(path)] = base.digest(path)
    dependencies.update({str(args.previous / name): base.digest(args.previous / name)
                         for name in ("plan.json", "frozen_selections.json", "development.json")})
    jobs, audit = [], []
    for job in parent["jobs"]:
        case = next(c for c in parent["cases"] if c["dataset_name"] == job["dataset"])
        original = next(s for s in original_frozen["selections"] if (s["dataset"], s["library"]) == (job["dataset"], job["library"]))
        if job["parameters"] != original["best"]["parameters"] or job["rounds"] != original["best"]["retained_rounds"]:
            raise ValueError("Original tree controls do not match the fixed carried policy")
        profiles = grid(job) if case["problem_type"] != "multiclass" else []
        jobs.append({**job, "profiles": profiles, "training_selection": original})
        if profiles:
            baseline = canonical(job["parameters"], job["library"])
            audit.append({"dataset": job["dataset"], "library": job["library"],
                "matches_original_declared_controls": [i for i, p in enumerate(profiles) if canonical(p, job["library"]) == baseline],
                "within_grid_unique_declared_controls": len({repr(sorted(canonical(p, job["library"]).items())) for p in profiles})})
    if len(jobs) != 36 or {(j["dataset"], j["library"]) for j in jobs} != expected or sum(len(j["profiles"]) for j in jobs) != 96:
        raise ValueError("Expected96 declared new fits")
    plan = {"protocol": "conditional_threshold_four_arm_union_20261016", "wave": 12, "created_at": base.now(),
        "script_sha256": base.digest(__file__), "dependencies": dependencies, "runtime": runtime,
        "scipy_version": importlib.metadata.version("scipy"), "previous": str(args.previous), "seed": 47,
        "cases": parent["cases"], "jobs": jobs, "single_selections": single_frozen["selections"],
        "binary_choices": base.read(singles_path / "binary_controls.json")["choices"],
        "round_cap": 2000, "patience": 50, "half_rate_round_cap": 4000, "half_rate_patience": 100,
        "resources": {"workers": 1, "threads_per_fit": 1, "cpus": [4]},
        "target": singles["target"], "duplicate_audit": audit, "official_test_scored": False,
        "selection": "Independent original TRAIN/STOP only. Early stopping chooses each uncorrected native tree prefix; then recompute unchanged correction controls on TRAIN and compare final corrected STOP error. Exact ties retain original corrected incumbent; new ties prefer fewer rounds then index. Freeze24choices before mandatory union refit24models. Carry12wave10multiclass singles by this fixed new policy, with no ensemble fallback.",
        "limitations": "Adaptive development continuation, not96unique hypotheses. Base comparator arms intentionally repeat controls. HalfLR receives4000rounds/patience100 versus2000/50 atbaseLR, preserving approximate learning-rate-times-rounds budget equally for all libraries, not equal compute or effective optimization. Alpha.8 retains a nonvacuous feature test but does not imply calibrated ensemble-level error control. No corrected-prefix optimization. Frozen correction controls may be suboptimal for new trees or union refits."}
    args.artifacts.mkdir(parents=True)
    base.write(args.output / "plan.json", plan)
    base.write(args.output / "duplicate_audit.json", audit)
    print({"plan_sha256": base.digest(args.output / "plan.json"), "fits": 96,
           "declared_incumbent_matches": sum(len(a["matches_original_declared_controls"]) for a in audit)})


def checked(args):
    plan = base.read(args.output / "plan.json")
    if (plan["script_sha256"] != base.digest(__file__) or plan["runtime"] != base.bootstrap(args)
            or plan["scipy_version"] != importlib.metadata.version("scipy") or args.previous != Path(plan["previous"])):
        raise ValueError("Frozen implementation/runtime changed")
    for path, expected in plan["dependencies"].items():
        if base.digest(path) != expected:
            raise ValueError("Frozen dependency changed")
    return plan


def search(args):
    import numpy as np
    import psutil
    plan = checked(args)
    psutil.Process().cpu_affinity(plan["resources"]["cpus"])
    if (args.output / "fits").exists():
        raise ValueError("Preserve started search")
    (args.output / "fits").mkdir()
    choices, successful = [], True
    for job in plan["jobs"]:
        if not job["profiles"]:
            continue
        case = next(c for c in plan["cases"] if c["dataset_name"] == job["dataset"])
        directory = args.output / "fits" / job["dataset"] / job["library"]
        records = []
        try:
            train, stop = base.load_role(case, "train"), base.load_role(case, "stop")
            if np.intersect1d(train["outer_train_positions"], stop["outer_train_positions"]).size:
                raise ValueError("TRAIN and STOP overlap")
            X, schema = base.normalize(train["X"])
            Xs, _ = base.normalize(stop["X"], schema)
            original = job["training_selection"]
            if base.digest(original["model_path"]) != original["model_sha256"]:
                raise ValueError("Original TRAIN model changed")
            with Path(original["model_path"]).open("rb") as stream:
                incumbent = pickle.load(stream)
            if schema != incumbent["schema"]:
                raise ValueError("Original TRAIN vocabulary changed")
            receipt = None
            if case["problem_type"] == "binary":
                incumbent, receipt = correct(incumbent, job, X, train["y"], Xs)
                prior = next(c for c in plan["binary_choices"] if (c["dataset"], c["library"]) == (job["dataset"], job["library"]))
                expected_error = prior["stop_error"]
            else:
                # The saved regression incumbent already contains its selected correction.
                expected_error = original["best"]["stop_error"]
            score = base.error(predict(incumbent, Xs), stop["y"], case["problem_type"])
            np.testing.assert_allclose(score, expected_error, rtol=1e-12, atol=1e-12)
            path = args.artifacts / "training_winners" / job["dataset"] / (job["library"] + ".pkl")
            best = {**original["best"], "stop_error": score, "config_index": "incumbent", "correction": receipt}
            saved = persist(incumbent, path, Xs)
            best_key = (score, 0, job["rounds"], -1)
            for index, params in enumerate(job["profiles"]):
                while (args.artifacts / "pause_fits").exists():
                    time.sleep(.5)
                record = {"dataset": job["dataset"], "library": job["library"], "config_index": index,
                          "status": "failed", "started_at": base.now()}
                try:
                    trial_plan = {**plan, "round_cap": plan["half_rate_round_cap"] if index % 2 else plan["round_cap"],
                                  "patience": plan["half_rate_patience"] if index % 2 else plan["patience"]}
                    payload, details = base.fit_configuration(job["library"], params, trial_plan, case, X, train["y"], Xs, stop["y"], schema)
                    payload, correction = correct(payload, job, X, train["y"], Xs)
                    score = base.error(predict(payload, Xs), stop["y"], case["problem_type"])
                    record.update(details, uncorrected_stop_error=details["stop_error"], stop_error=score, correction=correction,
                                  round_cap=trial_plan["round_cap"], patience=trial_plan["patience"],
                                  trained_at_round_cap=(None if details["trained_rounds_if_available"] is None else
                                                        details["trained_rounds_if_available"] == trial_plan["round_cap"]))
                    restored = pickle.loads(pickle.dumps(payload, protocol=5))
                    np.testing.assert_array_equal(predict(payload, Xs), predict(restored, Xs))
                    key = (score, 1, details["retained_rounds"], index)
                    if key < best_key:
                        saved = persist(payload, path, Xs)
                        best_key, best = key, {**details, "stop_error": score, "config_index": index, "correction": correction}
                    record["status"] = "ok"
                except Exception:
                    successful = False
                    record["error"] = traceback.format_exc()
                record["finished_at"] = base.now()
                records.append(record)
                base.write(directory / f"config_{index}.json", union.refit.plain_receipt(record))
            selected_job = {**job, "parameters": best["parameters"], "rounds": best["retained_rounds"]}
            choice = {"dataset": job["dataset"], "library": job["library"], "job": selected_job,
                      "best": best, **saved, "all_four_successful": len(records) == 4 and all(r["status"] == "ok" for r in records)}
            choices.append(choice)
            base.write(directory / "selection.json", union.refit.plain_receipt(choice))
            print({"dataset": job["dataset"], "library": job["library"], "winner": best["config_index"], "stop_error": best["stop_error"]}, flush=True)
        except Exception:
            successful = False
            base.write(directory / "failure.json", {"error": traceback.format_exc()})
    base.write(args.output / "selected_training.json", union.refit.plain_receipt({"plan_sha256": base.digest(args.output / "plan.json"),
        "created_at": base.now(), "choices": choices,
        "all_fits_successful": successful and len(choices) == 24 and all(c["all_four_successful"] for c in choices)}))


def refit(args):
    import psutil
    plan = checked(args)
    psutil.Process().cpu_affinity(plan["resources"]["cpus"])
    training_path = args.output / "selected_training.json"
    training_hash, training = base.digest(training_path), base.read(training_path)
    if training["plan_sha256"] != base.digest(args.output / "plan.json") or not training["all_fits_successful"]:
        raise ValueError("All96 independent TRAIN/STOP trials must pass before union refit")
    expected_choices = {(job["dataset"], job["library"]) for job in plan["jobs"] if job["profiles"]}
    if len(training["choices"]) != 24 or {(c["dataset"], c["library"]) for c in training["choices"]} != expected_choices:
        raise ValueError("Expected24 unique independent TRAIN choices")
    if (args.output / "refits").exists():
        raise ValueError("Preserve started mandatory refit")
    (args.output / "refits").mkdir()
    jobs, selections, successful = [], [], True
    for original_job in plan["jobs"]:
        if not original_job["profiles"]:
            jobs.append(original_job)
            selections.append(next(s for s in plan["single_selections"] if (s["dataset"], s["library"]) == (original_job["dataset"], original_job["library"])))
            continue
        choice = next(c for c in training["choices"] if (c["dataset"], c["library"]) == (original_job["dataset"], original_job["library"]))
        job = choice["job"]
        jobs.append(job)
        case = next(c for c in plan["cases"] if c["dataset_name"] == job["dataset"])
        record = {"dataset": job["dataset"], "library": job["library"], "status": "failed"}
        try:
            payload, details, frame = fit_policy(job, case, 47)
            path = args.artifacts / "models" / job["dataset"] / (job["library"] + ".pkl")
            selected = {"dataset": job["dataset"], "library": job["library"], "source_wave": 12,
                "best": choice["best"], "training_selection": choice, "refit_details": details,
                "leaf_refit": job.get("leaf_refit"), "binary_refit": job.get("binary_refit"), "temperature": job.get("temperature"),
                **persist(payload, path, frame)}
            selections.append(selected)
            record.update(status="ok", selection=selected)
        except Exception:
            successful = False
            record["error"] = traceback.format_exc()
        base.write(args.output / "refits" / job["dataset"] / (job["library"] + ".json"), union.refit.plain_receipt(record))
        print({key: record[key] for key in ("dataset", "library", "status")}, flush=True)
    if base.digest(training_path) != training_hash:
        raise ValueError("Selected training controls changed during refit")
    expected = {(job["dataset"], job["library"]) for job in plan["jobs"]}
    complete = (successful and len(selections) == 36 and len(jobs) == 36 and
                {(s["dataset"], s["library"]) for s in selections} == expected and
                {(j["dataset"], j["library"]) for j in jobs} == expected)
    base.write(args.output / "frozen_selections.json", union.refit.plain_receipt({"plan_sha256": base.digest(args.output / "plan.json"),
        "selected_training_sha256": training_hash, "created_at": base.now(), "jobs": jobs, "selections": selections,
        "all_fits_successful": complete, "development_scored": False, "official_test_scored": False}))


def development(args):
    frozen = base.read(args.output / "frozen_selections.json")
    if not frozen["all_fits_successful"] or frozen["selected_training_sha256"] != base.digest(args.output / "selected_training.json"):
        raise ValueError("All mandatory policies and frozen training choices must be intact")
    base.checked, base.predict = checked, predict
    base.development(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "search", "refit", "development"))
    for name in ("output", "artifacts", "package-root", "dependencies", "previous"):
        parser.add_argument("--" + name, type=lambda value: Path(value).resolve(), required=True)
    args = parser.parse_args()
    {"prepare": prepare, "search": search, "refit": refit, "development": development}[args.stage](args)
