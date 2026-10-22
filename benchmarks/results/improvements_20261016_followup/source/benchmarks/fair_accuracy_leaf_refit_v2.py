"""Equal-budget, fixed-topology regression leaf corrections on development data."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import importlib.util
from pathlib import Path
import pickle
import subprocess
import sys
import time
import traceback

BASE_PATH = Path(__file__).with_name("fair_accuracy_scout.py")
HELPER_PATH = Path(__file__).with_name("fixed_leaf_refit.py")
spec = importlib.util.spec_from_file_location("fair_base", BASE_PATH)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
PREDECLARATION = base.ROOT / "benchmarks/results/improvements_20261016_followup/leaf_refit_predeclaration.json"


def helper():
    spec = importlib.util.spec_from_file_location("fixed_leaf_refit", HELPER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plain_receipt(value):
    """Convert numerical diagnostic scalars without changing any computation."""
    import numpy as np
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: plain_receipt(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_receipt(item) for item in value]
    return value


def prepare(args):
    runtime = base.bootstrap(args)
    if args.output.exists() or args.artifacts.exists():
        raise ValueError("Preserve earlier attempts; use fresh paths")
    previous = base.read(args.previous / "plan.json")
    frozen = base.read(args.previous / "frozen_selections.json")
    development = base.read(args.previous / "development.json")
    declaration = base.read(PREDECLARATION)
    if (previous["wave"] != 5 or previous["runtime"] != runtime or not frozen["all_fits_successful"]
            or development["development_target_met"]
            or frozen["plan_sha256"] != base.digest(args.previous / "plan.json")
            or development["plan_sha256"] != base.digest(args.previous / "plan.json")
            or development["frozen_selections_sha256"] != base.digest(args.previous / "frozen_selections.json")):
        raise ValueError("Expected the intact failed wave5 development gate")
    for selection in frozen["selections"]:
        if base.digest(selection["model_path"]) != selection["model_sha256"]:
            raise ValueError("Base model changed before freeze")
        parameters = selection["best"]["parameters"]
        for key in ("monotone_constraints", "monotonic_cst", "interaction_constraints", "sample_weight", "baseline", "base_margin"):
            if parameters.get(key) not in (None, "", (), [], {}):
                raise ValueError(f"Unsupported constrained/weighted/baseline model: {key}")
        if parameters.get("max_leaf_weight", 0) or parameters.get("max_delta_step", 0):
            raise ValueError("Constrained leaf outputs are unsupported")
    grid = [{"lambda": penalty, "shrinkage": shrink} for penalty in declaration["lambda"] for shrink in declaration["shrinkage"]]
    if len(grid) != 8 or len(previous["cases"]) != 12 or len(frozen["selections"]) != 36:
        raise ValueError("Unexpected study population/budget")
    datasets = {case["dataset_name"] for case in previous["cases"]}
    if len(datasets) != 12 or {(s["dataset"], s["library"]) for s in frozen["selections"]} != {
            (dataset, library) for dataset in datasets for library in base.LIBRARIES}:
        raise ValueError("Expected twelve unique cases and36 unique dataset/library models")
    plan = {"protocol": declaration["protocol"], "created_at": base.now(), "wave": 6,
        "script_sha256": base.digest(__file__), "base_script_sha256": base.digest(BASE_PATH),
        "helper_sha256": base.digest(HELPER_PATH), "predeclaration_sha256": base.digest(PREDECLARATION),
        "runtime": runtime, "scipy_version": importlib.metadata.version("scipy"),
        "dependencies": {str(args.previous / name): base.digest(args.previous / name)
                         for name in ("plan.json", "frozen_selections.json", "development.json")},
        "previous": str(args.previous), "previous_frozen_sha256": base.digest(args.previous / "frozen_selections.json"),
        "cases": previous["cases"], "base_selections": frozen["selections"], "seed": 47,
        "patience": previous["patience"], "candidate_grid": grid, "solver": declaration["solver"],
        "prediction_oracle_tolerance": {"rtol": 5e-5, "atol": 1e-5},
        "eligibility": "Unconstrained scalar squared loss only. Original frozen fit code consumes X/y without sample weights, external baselines or base margins; selected parameter dictionaries are checked for constraints.",
        "resources": {"workers": 2, "threads_per_fit": 1, "cpus": [4, 6],
                      "maximum_job_seconds": 900, "maximum_job_rss_bytes": 8 * 1024 ** 3},
        "target": previous["target"], "limitations": declaration["limitations"],
        "model_selection": declaration["selection"], "official_test_scored": False}
    args.artifacts.mkdir(parents=True)
    base.write(args.output / "plan.json", plan)
    print({"plan_sha256": base.digest(args.output / "plan.json"), "candidate_evaluations": 96, "ridge_solves": 48})


def checked(args):
    plan = base.read(args.output / "plan.json")
    if (plan["script_sha256"] != base.digest(__file__) or plan["base_script_sha256"] != base.digest(BASE_PATH)
            or plan["helper_sha256"] != base.digest(HELPER_PATH)
            or plan["predeclaration_sha256"] != base.digest(PREDECLARATION)
            or plan["runtime"] != base.bootstrap(args)
            or plan["scipy_version"] != importlib.metadata.version("scipy")):
        raise ValueError("Frozen code/runtime changed")
    for path, expected in plan["dependencies"].items():
        if base.digest(path) != expected:
            raise ValueError("Parent evidence changed")
    return plan


def contribution(leaves, ids_by_tree, coefficients):
    import numpy as np
    result, offset = np.zeros(len(leaves), dtype=np.float64), 0
    for tree, known in enumerate(ids_by_tree):
        mapping = {int(leaf): float(coefficients[offset + index]) for index, leaf in enumerate(known)}
        result += np.fromiter((mapping.get(int(leaf), 0.0) for leaf in leaves[:, tree]), dtype=np.float64, count=len(leaves))
        offset += len(known)
    if offset != len(coefficients):
        raise ValueError("Leaf basis and coefficient count differ")
    return result


def training_prediction(payload, frame, prepared):
    return helper().predict(payload, frame, training_prepared=prepared)


def solve_coefficients(design, labels, penalty, solver):
    import numpy as np
    from scipy.sparse.linalg import lsqr
    residual = np.asarray(labels, dtype=np.float64) - np.asarray(design["baseline"], dtype=np.float64)
    result = lsqr(design["Z"], residual, damp=np.sqrt(penalty), atol=solver["atol"],
                  btol=solver["btol"], iter_lim=solver["iter_lim"], show=False)
    receipt = {"istop": int(result[1]), "iterations": int(result[2]), "residual_norm": float(result[3]),
               "regularized_residual_norm": float(result[4]), "matrix_norm": float(result[5]),
               "condition_estimate": float(result[6]), "normal_residual_norm": float(result[7]), "coefficient_norm": float(result[8])}
    if receipt["istop"] not in (0, 1, 2, 4, 5) or not np.isfinite(result[0]).all():
        raise ValueError(f"Ridge solve did not converge: {receipt}")
    return result[0], receipt


def apply_correction(payload, design, coefficients, shrinkage):
    h = helper()
    delta = h.corrections_by_tree(design["leaf_ids_by_tree"], coefficients, shrinkage=shrinkage)
    candidate, receipt = h.fold_corrections(payload, delta)
    receipt = plain_receipt(receipt)
    candidate["fixed_leaf_refit"] = receipt
    return candidate, receipt


def fit_job(args):
    import numpy as np
    import psutil
    plan = checked(args)
    h = helper()
    psutil.Process().cpu_affinity([plan["resources"]["cpus"][args.slot]])
    case = next(c for c in plan["cases"] if c["dataset_name"] == args.dataset)
    if case["problem_type"] != "regression":
        raise ValueError("This correction protocol is regression-only")
    incumbent = next(s for s in plan["base_selections"] if s["dataset"] == args.dataset and s["library"] == args.library)
    directory = args.output / "fits" / args.dataset / args.library
    directory.mkdir(parents=True, exist_ok=False)
    best = {**incumbent, "leaf_refit": None, "base_selection": incumbent}
    best_key = (incumbent["best"]["stop_error"], 0, 0, 0)
    records, total_seconds = [], 0.0
    try:
        if base.digest(incumbent["model_path"]) != incumbent["model_sha256"]:
            raise ValueError("Base model changed")
        with Path(incumbent["model_path"]).open("rb") as stream:
            payload = pickle.load(stream)
        train, stop = base.load_role(case, "train"), base.load_role(case, "stop")
        X, _ = base.normalize(train["X"], payload["schema"])
        Xs, _ = base.normalize(stop["X"], payload["schema"])
        design = h.prepare_design(payload, X, train["y"])
        old_stop = base.predict(payload, Xs)
        old_stop_leaves = h.leaf_indices(payload, Xs)
        np.testing.assert_allclose(base.error(old_stop, stop["y"], "regression"), incumbent["best"]["stop_error"], rtol=1e-12, atol=1e-12)
        base.write(directory / "design.json", {"shape": list(design["Z"].shape), "nonzeros": int(design["Z"].nnz),
            "preprocessing": design["preprocessing_receipt"], "base_train_rmse": base.error(design["baseline"], train["y"], "regression"),
            "base_stop_rmse": best_key[0], "model_sha256": incumbent["model_sha256"]})
    except Exception:
        failure = traceback.format_exc()
        for index, controls in enumerate(plan["candidate_grid"]):
            record = {"dataset": args.dataset, "library": args.library, "config_index": index,
                      "controls": controls, "status": "failed", "error": failure, "failure_stage": "design"}
            records.append(record)
            base.write(directory / f"config_{index}.json", record)
    else:
        cached_penalty, coefficients, solve_receipt, solve_error, solve_seconds = None, None, None, None, 0.0
        for index, controls in enumerate(plan["candidate_grid"]):
            while (args.artifacts / "pause_fits").exists():
                time.sleep(.5)
            record = {"dataset": args.dataset, "library": args.library, "config_index": index,
                      "controls": controls, "status": "failed", "started_at": base.now()}
            started = time.perf_counter()
            try:
                if cached_penalty != controls["lambda"]:
                    cached_penalty, solve_error = controls["lambda"], None
                    solve_started = time.perf_counter()
                    try:
                        coefficients, solve_receipt = solve_coefficients(design, train["y"], cached_penalty, plan["solver"])
                    except Exception:
                        solve_error = traceback.format_exc()
                    solve_seconds = time.perf_counter() - solve_started
                else:
                    solve_seconds = 0.0
                if solve_error:
                    raise ValueError(solve_error)
                candidate, fold_receipt = apply_correction(payload, design, coefficients, controls["shrinkage"])
                deltas = h.corrections_by_tree(design["leaf_ids_by_tree"], coefficients, shrinkage=controls["shrinkage"])
                verified_train = plain_receipt(h.verify_fold(payload, candidate, X, deltas, fold_receipt, training_prepared=design["training_prepared"]))
                verified_stop = plain_receipt(h.verify_fold(payload, candidate, Xs, deltas, fold_receipt))
                np.testing.assert_array_equal(h.leaf_indices(candidate, X, training_prepared=design["training_prepared"]), design["training_leaves"])
                np.testing.assert_array_equal(h.leaf_indices(candidate, Xs), old_stop_leaves)
                train_predictions = training_prediction(candidate, X, design["training_prepared"])
                stop_predictions = base.predict(candidate, Xs)
                expected_train = design["baseline"] + controls["shrinkage"] * (design["Z"] @ coefficients)
                expected_stop = old_stop + controls["shrinkage"] * contribution(old_stop_leaves, design["leaf_ids_by_tree"], coefficients)
                for actual, expected in ((train_predictions, expected_train), (stop_predictions, expected_stop)):
                    np.testing.assert_allclose(actual, expected, **plan["prediction_oracle_tolerance"])
                score = base.error(stop_predictions, stop["y"], "regression")
                record.update(status="ok", stop_error=score,
                    train_error=base.error(train_predictions, train["y"], "regression"),
                    solve=solve_receipt, solve_seconds=solve_seconds, fold=fold_receipt,
                    verification_train=verified_train, verification_stop=verified_stop,
                    native_oracle_maxabs_train=float(np.max(np.abs(train_predictions - expected_train))),
                    native_oracle_maxabs_stop=float(np.max(np.abs(stop_predictions - expected_stop))))
                key = (score, 1, -controls["lambda"], controls["shrinkage"])
                if key < best_key:
                    path = args.artifacts / "models" / args.dataset / (args.library + ".pkl")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_suffix(".candidate.pkl")
                    with temporary.open("wb") as stream:
                        pickle.dump(candidate, stream, protocol=5)
                    with temporary.open("rb") as stream:
                        restored = pickle.load(stream)
                    np.testing.assert_array_equal(base.predict(restored, Xs), stop_predictions)
                    temporary.replace(path)
                    best_key = key
                    best = {**incumbent, "plan_sha256": base.digest(args.output / "plan.json"),
                        "source_wave": 6, "leaf_refit": {**controls, "solver": plan["solver"], "fold": fold_receipt},
                        "base_selection": incumbent, "model_path": str(path), "model_sha256": base.digest(path),
                        "model_bytes": path.stat().st_size,
                        "best": {**incumbent["best"], "stop_error": score, "selection_key": key, "config_index": index}}
            except Exception:
                record.update(status="failed", error=traceback.format_exc())
            record["candidate_seconds"] = time.perf_counter() - started
            record["finished_at"] = base.now()
            total_seconds += record["candidate_seconds"]
            records.append(record)
            base.write(directory / f"config_{index}.json", record)
            print({k: record.get(k) for k in ("dataset", "library", "config_index", "status", "stop_error", "candidate_seconds")}, flush=True)
    best["current_wave_all_eight_successful"] = all(r["status"] == "ok" for r in records)
    best["postfit_candidate_seconds"] = total_seconds
    base.write(directory / "selection.json", best)


def run(args):
    import psutil
    plan = checked(args)
    if (args.output / "fits").exists():
        raise ValueError("A started attempt is terminal; preserve it")
    (args.output / "logs").mkdir()
    jobs = [(c["dataset_name"], library) for c in plan["cases"] if c["problem_type"] == "regression" for library in base.LIBRARIES]

    def worker(slot):
        for dataset, library in jobs[slot::2]:
            while (args.artifacts / "pause_fits").exists():
                time.sleep(.5)
            command = [sys.executable, "-I", str(Path(__file__).resolve()), "fit", "--output", str(args.output),
                "--artifacts", str(args.artifacts), "--package-root", str(args.package_root), "--dependencies", str(args.dependencies),
                "--previous", str(args.previous), "--dataset", dataset, "--library", library, "--slot", str(slot)]
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
            base.write(args.output / "logs" / f"{dataset}-{library}-resource.json", {"exit_code": process.returncode,
                "resource_failure": failure, "peak_rss_bytes": peak, "wall_seconds": time.monotonic() - started})
            print({"completed": dataset, "library": library, "exit_code": process.returncode, "resource_failure": failure}, flush=True)
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(worker, range(2)))
    selections = []
    for incumbent in plan["base_selections"]:
        path = args.output / "fits" / incumbent["dataset"] / incumbent["library"] / "selection.json"
        selections.append(base.read(path) if path.exists() else incumbent)
    completed = [base.read(args.output / "fits" / dataset / library / "selection.json") for dataset, library in jobs]
    base.write(args.output / "frozen_selections.json", {"plan_sha256": base.digest(args.output / "plan.json"),
        "created_at": base.now(), "development_scored": False, "official_test_scored": False,
        "selections": selections, "all_fits_successful": all(s["current_wave_all_eight_successful"] for s in completed)})


def development(args):
    # The shared scorer only predicts frozen native payloads. Bind this new
    # protocol's strict check rather than changing an earlier frozen runner.
    base.checked = checked
    base.development(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "fit", "development"))
    for name in ("output", "artifacts", "package-root", "dependencies", "previous"):
        parser.add_argument("--" + name, type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--library", choices=base.LIBRARIES)
    parser.add_argument("--slot", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "fit": fit_job, "development": development}[args.stage](args)
