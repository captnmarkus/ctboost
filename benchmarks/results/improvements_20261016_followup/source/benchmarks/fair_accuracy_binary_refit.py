"""Independent-STOP binary leaf correction, then fixed transfer to union models."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
from pathlib import Path
import pickle
import time
import traceback

UNION_PATH = Path(__file__).with_name("fair_accuracy_union_refit.py")
HELPER_PATH = Path(__file__).with_name("fixed_binary_leaf_refit.py")
POLICY = Path(__file__).parent / "results/improvements_20261016_followup/robustness_protocol_binary.json"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


union = load_module("binary_union", UNION_PATH)
base, predict = union.base, union.predict


def helper():
    return load_module("fixed_binary_leaf_refit", HELPER_PATH)


def corrected(payload, X, labels, controls, verification_frames):
    """Recompute one fixed correction; no scoring or selection here."""
    import numpy as np
    h = helper()
    design = h.prepare_design(payload, X, labels)
    coefficients, solve = h.solve_coefficients(design, labels, controls["lambda"], **controls["solver"])
    deltas = h.corrections_by_tree(design["leaf_ids_by_tree"], coefficients, controls["shrinkage"])
    candidate, fold = h.fold_corrections(payload, deltas)
    fold = union.refit.plain_receipt(fold)
    candidate["fixed_leaf_refit"] = fold
    train_check = h.verify_fold(payload, candidate, X, deltas, fold, design["training_prepared"])
    np.testing.assert_allclose(h.predict(candidate, X, design["training_prepared"]),
        design["baseline"] + controls["shrinkage"] * (design["Z"] @ coefficients), rtol=5e-5, atol=1e-5)
    checks = {name: h.verify_fold(payload, candidate, frame, deltas, fold) for name, frame in verification_frames.items()}
    return candidate, union.refit.plain_receipt({"solve": solve, "fold": fold,
        "training_check": train_check, "inference_checks": checks, "preprocessing": design["preprocessing_receipt"]})


def fit_policy(job, case, seed):
    """Seed-check entry point: fixed union trees and fixed post-fit controls."""
    payload, details, frame = union.fit_policy(job, case, seed)
    if job.get("binary_refit"):
        X, labels, schema, train, stop, receipt = union.union_data(case)
        if schema != payload["schema"]:
            raise ValueError("Union vocabulary changed")
        verification = {name: base.normalize(role["X"], schema)[0]
                        for name, role in (("original_train", train), ("original_stop", stop))}
        payload, correction = corrected(payload, X, labels, job["binary_refit"], verification)
        details["fixed_binary_refit"] = correction
        details["apparent_union_error_inference_transform"] = base.error(predict(payload, X), labels, "binary")
    return payload, details, frame


def prepare(args):
    runtime = base.bootstrap(args)
    if args.output.exists() or args.artifacts.exists():
        raise ValueError("Preserve earlier attempts; use fresh paths")
    previous = base.read(args.previous / "plan.json")
    frozen = base.read(args.previous / "frozen_selections.json")
    development = base.read(args.previous / "development.json")
    if (previous["wave"] != 9 or previous["runtime"] != runtime or not frozen["all_fits_successful"]
            or development["development_target_met"] or previous["script_sha256"] != base.digest(UNION_PATH)
            or frozen["plan_sha256"] != base.digest(args.previous / "plan.json")
            or development["plan_sha256"] != base.digest(args.previous / "plan.json")
            or development["frozen_selections_sha256"] != base.digest(args.previous / "frozen_selections.json")):
        raise ValueError("Expected the intact failed mandatory union gate")
    training_parent = Path(previous["previous"])
    training_plan = base.read(training_parent / "plan.json")
    training_frozen = base.read(training_parent / "frozen_selections.json")
    if (training_plan["wave"] != 8 or not training_frozen["all_fits_successful"]
            or training_frozen["plan_sha256"] != base.digest(training_parent / "plan.json")):
        raise ValueError("Independent STOP must use original wave8 TRAIN-fitted models")
    if len(previous["cases"]) != 12 or previous["scipy_version"] != importlib.metadata.version("scipy"):
        raise ValueError("Parent task population or SciPy runtime changed")
    expected = {(case["dataset_name"], library) for case in previous["cases"] for library in base.LIBRARIES}
    for document in (frozen, training_frozen):
        if len(document["selections"]) != 36 or len(expected) != 36 or {
                (s["dataset"], s["library"]) for s in document["selections"]} != expected:
            raise ValueError("Expected36 unique original and union models")
        for selected in document["selections"]:
            if base.digest(selected["model_path"]) != selected["model_sha256"]:
                raise ValueError("Parent model changed")
            params = selected["best"]["parameters"]
            for key in ("monotone_constraints", "interaction_constraints", "sample_weight", "class_weight", "class_weights", "baseline", "base_margin"):
                if params.get(key) not in (None, "", [], (), {}):
                    raise ValueError("Weighted/constrained/externally based models are unsupported")
            if params.get("max_leaf_weight", 0) or params.get("max_delta_step", 0):
                raise ValueError("Constrained leaf outputs are unsupported")
            if params.get("scale_pos_weight", 1) != 1 or params.get("is_unbalance", False) or params.get("sigmoid", 1) != 1:
                raise ValueError("Expected unweighted binary models with unit-scale logits")
    dependencies = dict(previous["dependencies"])
    for path, expected_hash in dependencies.items():
        if base.digest(path) != expected_hash:
            raise ValueError("Union dependency changed")
    for path in (UNION_PATH, HELPER_PATH, POLICY, training_parent / "plan.json", training_parent / "frozen_selections.json"):
        dependencies[str(path)] = base.digest(path)
    dependencies.update({str(args.previous / name): base.digest(args.previous / name)
                         for name in ("plan.json", "frozen_selections.json", "development.json")})
    plan = {"protocol": "independent_stop_binary_correction_union_transfer_20261016", "wave": 10,
        "created_at": base.now(), "script_sha256": base.digest(__file__), "dependencies": dependencies,
        "runtime": runtime, "scipy_version": importlib.metadata.version("scipy"), "previous": str(args.previous),
        "training_parent": str(training_parent), "cases": previous["cases"], "jobs": previous["jobs"],
        "base_selections": frozen["selections"], "training_selections": training_frozen["selections"], "seed": 47,
        "candidate_grid": [{"lambda": penalty, "shrinkage": shrink} for penalty in (1., 10., 100., 1000.) for shrink in (.25, 1.)],
        "solver": {"maxiter": 2000, "ftol": 1e-13, "gtol": 1e-9, "gradient_gate": 1e-6},
        "resources": {"workers": 1, "threads_per_fit": 1, "cpus": [4]}, "target": previous["target"],
        "official_test_scored": False,
        "selection": "Four binary tasks only, eight nonzero candidates plus identity perlibrary. TRAIN-only penalized logistic solve on original wave8 models; independent STOP1-AUC chooses controls. Exact ties prefer identity, then largerlambda, then smaller shrinkage. Freeze all12choices before fixed recomputation on wave9union models. Other8union tasks carry unchanged; freeze all36before DEV.",
        "limitations": "Adaptive reused development. Logistic training objective and AUC selection metric deliberately differ. Positive ridge objective is sumlogisticloss+lambda/2||delta||^2 with fixed original logits. Equal candidate counts are not equal compute/effective regularization. Union transfer may change the best penalty; no reselection or DEV fallback. No new tree fits or inference wrapper cost."}
    args.artifacts.mkdir(parents=True)
    base.write(args.output / "plan.json", plan)
    print({"plan_sha256": base.digest(args.output / "plan.json"), "candidate_evaluations": 96, "selection_solves": 48})


def checked(args):
    plan = base.read(args.output / "plan.json")
    if (plan["script_sha256"] != base.digest(__file__) or plan["runtime"] != base.bootstrap(args)
            or plan["scipy_version"] != importlib.metadata.version("scipy") or args.previous != Path(plan["previous"])):
        raise ValueError("Frozen code/runtime/parent changed")
    for path, expected in plan["dependencies"].items():
        if base.digest(path) != expected:
            raise ValueError("Frozen dependency changed")
    return plan


def select(args):
    import numpy as np
    import psutil
    plan = checked(args)
    psutil.Process().cpu_affinity(plan["resources"]["cpus"])
    if (args.output / "selection_fits").exists():
        raise ValueError("Preserve a started selection attempt")
    (args.output / "selection_fits").mkdir()
    choices, successful = [], True
    for original in plan["training_selections"]:
        case = next(c for c in plan["cases"] if c["dataset_name"] == original["dataset"])
        if case["problem_type"] != "binary":
            continue
        h = helper()
        directory = args.output / "selection_fits" / original["dataset"] / original["library"]
        records, chosen = [], None
        try:
            if base.digest(original["model_path"]) != original["model_sha256"]:
                raise ValueError("Original TRAIN model changed after freeze")
            with Path(original["model_path"]).open("rb") as stream:
                payload = pickle.load(stream)
            train, stop = base.load_role(case, "train"), base.load_role(case, "stop")
            if np.intersect1d(train["outer_train_positions"], stop["outer_train_positions"]).size:
                raise ValueError("Independent TRAIN and STOP overlap")
            X, _ = base.normalize(train["X"], payload["schema"])
            Xs, _ = base.normalize(stop["X"], payload["schema"])
            design = h.prepare_design(payload, X, train["y"])
            identity_error = base.error(predict(payload, Xs), stop["y"], "binary")
            np.testing.assert_allclose(identity_error, original["best"]["stop_error"], rtol=1e-12, atol=1e-12)
            best_key = (identity_error, 0, 0., 0.)
            chosen = {"dataset": original["dataset"], "library": original["library"], "binary_refit": None,
                      "stop_error": identity_error, "original_model_sha256": original["model_sha256"]}
            base.write(directory / "design.json", design["preprocessing_receipt"])
            cached_penalty, coefficients, solve, solver_failure = None, None, None, None
            for index, controls in enumerate(plan["candidate_grid"]):
                while (args.artifacts / "pause_fits").exists():
                    time.sleep(.5)
                record = {"dataset": original["dataset"], "library": original["library"], "config_index": index,
                          "controls": controls, "status": "failed", "started_at": base.now()}
                try:
                    if cached_penalty != controls["lambda"]:
                        cached_penalty, solver_failure = controls["lambda"], None
                        try:
                            coefficients, solve = h.solve_coefficients(design, train["y"], cached_penalty, **plan["solver"])
                        except Exception:
                            solver_failure = traceback.format_exc()
                    if solver_failure:
                        raise ValueError(solver_failure)
                    deltas = h.corrections_by_tree(design["leaf_ids_by_tree"], coefficients, controls["shrinkage"])
                    candidate, fold = h.fold_corrections(payload, deltas)
                    fold = union.refit.plain_receipt(fold)
                    candidate["fixed_leaf_refit"] = fold
                    train_check = h.verify_fold(payload, candidate, X, deltas, fold, design["training_prepared"])
                    stop_check = h.verify_fold(payload, candidate, Xs, deltas, fold)
                    np.testing.assert_allclose(h.predict(candidate, X, design["training_prepared"]),
                        design["baseline"] + controls["shrinkage"] * (design["Z"] @ coefficients), rtol=5e-5, atol=1e-5)
                    score = base.error(predict(candidate, Xs), stop["y"], "binary")
                    key = (score, 1, -controls["lambda"], controls["shrinkage"])
                    # Check persistence before permitting this candidate to win.
                    restored = pickle.loads(pickle.dumps(candidate, protocol=5))
                    np.testing.assert_array_equal(predict(restored, Xs), predict(candidate, Xs))
                    record.update(status="ok", stop_error=score, solve=solve, fold=fold,
                                  training_check=train_check, stop_check=stop_check)
                    if key < best_key:
                        best_key = key
                        chosen = {**chosen, "binary_refit": {**controls, "solver": plan["solver"]}, "stop_error": score}
                except Exception:
                    successful = False
                    record["error"] = traceback.format_exc()
                record["finished_at"] = base.now()
                records.append(record)
                base.write(directory / f"config_{index}.json", union.refit.plain_receipt(record))
        except Exception:
            successful = False
            base.write(directory / "failure.json", {"error": traceback.format_exc()})
        if chosen is not None:
            choices.append(chosen)
            base.write(directory / "selected_controls.json", chosen)
        successful = successful and len(records) == 8 and all(r["status"] == "ok" for r in records)
        print({"dataset": original["dataset"], "library": original["library"], "candidates": len(records),
               "selected": chosen.get("binary_refit") if chosen else None}, flush=True)
    base.write(args.output / "binary_controls.json", {"plan_sha256": base.digest(args.output / "plan.json"),
        "created_at": base.now(), "choices": choices, "all_fits_successful": successful and len(choices) == 12,
        "union_transfer_started": False, "development_scored": False})


def transfer(args):
    import numpy as np
    import psutil
    plan = checked(args)
    psutil.Process().cpu_affinity(plan["resources"]["cpus"])
    controls_path = args.output / "binary_controls.json"
    controls_sha256 = base.digest(controls_path)
    controls = base.read(controls_path)
    if controls["plan_sha256"] != base.digest(args.output / "plan.json") or not controls["all_fits_successful"]:
        raise ValueError("All96 independent-STOP candidates must succeed before union transfer")
    expected_choices = {(case["dataset_name"], library) for case in plan["cases"] if case["problem_type"] == "binary"
                        for library in base.LIBRARIES}
    if len(expected_choices) != 12 or len(controls["choices"]) != 12 or {
            (choice["dataset"], choice["library"]) for choice in controls["choices"]} != expected_choices:
        raise ValueError("Expected12 unique frozen binary choices")
    if (args.output / "transfer_fits").exists():
        raise ValueError("Preserve a started transfer attempt")
    (args.output / "transfer_fits").mkdir()
    selections, successful = [], True
    for original in plan["base_selections"]:
        case = next(c for c in plan["cases"] if c["dataset_name"] == original["dataset"])
        if case["problem_type"] != "binary":
            selections.append(original)
            continue
        chosen = next(c for c in controls["choices"] if c["dataset"] == original["dataset"] and c["library"] == original["library"])
        record = {"dataset": original["dataset"], "library": original["library"], "status": "failed"}
        try:
            if base.digest(original["model_path"]) != original["model_sha256"]:
                raise ValueError("Union model changed")
            selected = {**original, "binary_refit": chosen["binary_refit"], "independent_stop_choice": chosen}
            if chosen["binary_refit"]:
                with Path(original["model_path"]).open("rb") as stream:
                    payload = pickle.load(stream)
                X, labels, schema, train, stop, data_receipt = union.union_data(case)
                if schema != payload["schema"]:
                    raise ValueError("Union vocabulary changed")
                verification = {name: base.normalize(role["X"], schema)[0]
                                for name, role in (("original_train", train), ("original_stop", stop))}
                candidate, receipt = corrected(payload, X, labels, chosen["binary_refit"], verification)
                path = args.artifacts / "models" / original["dataset"] / (original["library"] + ".pkl")
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".candidate.pkl")
                with temporary.open("wb") as stream:
                    pickle.dump(candidate, stream, protocol=5)
                with temporary.open("rb") as stream:
                    restored = pickle.load(stream)
                np.testing.assert_array_equal(predict(candidate, stop["X"]), predict(restored, stop["X"]))
                temporary.replace(path)
                selected.update(source_wave=10, plan_sha256=base.digest(args.output / "plan.json"),
                    base_selection=original, model_path=str(path), model_sha256=base.digest(path),
                    model_bytes=path.stat().st_size, binary_refit_receipt=receipt, union_data=data_receipt)
            selections.append(selected)
            record.update(status="ok", selection=selected)
        except Exception:
            successful = False
            record["error"] = traceback.format_exc()
        base.write(args.output / "transfer_fits" / original["dataset"] / (original["library"] + ".json"), record)
        print({key: record[key] for key in ("dataset", "library", "status")}, flush=True)
    if base.digest(controls_path) != controls_sha256:
        raise ValueError("Selected controls changed during transfer")
    base.write(args.output / "frozen_selections.json", {"plan_sha256": base.digest(args.output / "plan.json"),
        "created_at": base.now(), "binary_controls_sha256": controls_sha256, "selections": selections,
        "all_fits_successful": successful and len(selections) == 36, "development_scored": False, "official_test_scored": False})


def development(args):
    frozen = base.read(args.output / "frozen_selections.json")
    if (not frozen["all_fits_successful"] or frozen["binary_controls_sha256"] != base.digest(args.output / "binary_controls.json")):
        raise ValueError("Frozen independent choices and all36 union policies must be intact")
    base.checked, base.predict = checked, predict
    base.development(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "select", "transfer", "development"))
    for name in ("output", "artifacts", "package-root", "dependencies", "previous"):
        parser.add_argument("--" + name, type=lambda value: Path(value).resolve(), required=True)
    args = parser.parse_args()
    {"prepare": prepare, "select": select, "transfer": transfer, "development": development}[args.stage](args)
