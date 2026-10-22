"""TRAIN-only ridge solver diagnosis; no native model mutation or heldout scores."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time
import traceback

PARENT_PATH = Path(__file__).with_name("fair_accuracy_scalar_refresh.py")


def parent_module():
    spec = importlib.util.spec_from_file_location("ridge_diagnostic_parent", PARENT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scaled_operator(Z, penalty):
    import numpy as np
    from scipy.sparse import isspmatrix_csr
    from scipy.sparse.linalg import LinearOperator
    if (not isspmatrix_csr(Z) or Z.dtype != np.float64 or not Z.has_canonical_format
            or not np.all(Z.data == 1)):
        raise ValueError("Expected canonical CSR leaf indicators containing only ones")
    if not np.isfinite(penalty) or penalty <= 0 or min(Z.shape) < 1:
        raise ValueError("Expected positive ridge penalty and nonempty design")
    rows, columns = Z.shape
    # Leaf indicators are binary; sum equals squared column norm without a CSR copy.
    scale = 1 / np.sqrt(np.asarray(Z.sum(axis=0)).ravel() + penalty)
    root = np.sqrt(penalty)
    def matvec(theta):
        delta = scale * np.asarray(theta).reshape(-1)
        return np.concatenate((Z @ delta, root * delta))
    def rmatvec(vector):
        vector = np.asarray(vector).reshape(-1)
        return scale * (Z.T @ vector[:rows] + root * vector[rows:])
    return LinearOperator((rows + columns, columns), matvec=matvec, rmatvec=rmatvec, dtype=np.float64), scale


def certificate(Z, residual_target, delta, penalty):
    import numpy as np
    r = np.asarray(residual_target, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    if (r.shape != (Z.shape[0],) or delta.shape != (Z.shape[1],) or penalty <= 0
            or not np.isfinite(penalty) or not np.all(np.isfinite(r)) or not np.all(np.isfinite(delta))):
        raise ValueError("Invalid ridge certificate inputs")
    u = r - Z @ delta
    ztu = Z.T @ u
    gradient = ztu - penalty * delta
    primal = .5 * (np.dot(u, u) + penalty * np.dot(delta, delta))
    primal_zero = .5 * np.dot(r, r)
    gap = np.dot(gradient, gradient) / (2 * penalty)
    dual = np.dot(r, u) - .5 * np.dot(u, u) - np.dot(ztu, ztu) / (2 * penalty)
    normal_scale = np.linalg.norm(Z.T @ r) + penalty * np.linalg.norm(delta)
    gradient_norm = np.linalg.norm(gradient)
    denominator = max(primal, np.finfo(np.float64).eps * primal_zero)
    relative_gap = gap / denominator if denominator > 0 else (0. if gap == 0 else float("inf"))
    normal_relative = gradient_norm / normal_scale if normal_scale > 0 else (0. if gradient_norm == 0 else float("inf"))
    values = dict(primal=float(primal), primal_zero=float(primal_zero), dual=float(dual),
        gradient_norm=float(gradient_norm), original_gradient_dual_gap=float(gap),
        primal_minus_dual=float(primal - dual), relative_gap=float(relative_gap),
        normalized_original_gradient=float(normal_relative),
        fitting_prediction_rms_error_bound=float(np.sqrt(2 * gap / len(r))),
        coefficient_l2_error_bound=float(np.sqrt(2 * gap / penalty)))
    values["finite"] = all(np.isfinite(value) for value in values.values())
    values["certified"] = bool(values["finite"] and relative_gap <= 1e-10 and normal_relative <= 1e-8)
    return values


def solve(Z, residual_target, penalty, variant, *, iterations=2000):
    import numpy as np
    from scipy.sparse.linalg import lsqr, lsmr
    if variant not in ("original_lsqr", "scaled_lsqr", "scaled_lsmr"):
        raise ValueError("Unknown frozen diagnostic variant")
    if type(iterations) is not int or iterations < 1:
        raise ValueError("Iterations must be a positive integer")
    r = np.asarray(residual_target, dtype=np.float64)
    if r.shape != (Z.shape[0],) or not np.all(np.isfinite(r)):
        raise ValueError("Invalid residual target")
    operator, scale = scaled_operator(Z, penalty)
    started = time.perf_counter()
    if variant == "original_lsqr":
        output = lsqr(Z, r, damp=np.sqrt(penalty), atol=1e-8, btol=1e-8, conlim=1e8, iter_lim=iterations)
        delta = output[0]
        native = {"istop": int(output[1]), "iterations": int(output[2]), "residual_norm": float(output[3]),
                  "regularized_residual_norm": float(output[4]), "matrix_norm": float(output[5]),
                  "condition_estimate": float(output[6]), "normal_residual_norm": float(output[7])}
    else:
        rhs = np.concatenate((r, np.zeros(Z.shape[1])))
        if variant == "scaled_lsqr":
            output = lsqr(operator, rhs, damp=0, atol=1e-12, btol=1e-12, conlim=1e8, iter_lim=iterations)
            native = {"istop": int(output[1]), "iterations": int(output[2]), "residual_norm": float(output[3]),
                      "matrix_norm": float(output[5]), "condition_estimate": float(output[6]), "normal_residual_norm": float(output[7])}
        else:
            output = lsmr(operator, rhs, damp=0, atol=1e-12, btol=1e-12, conlim=1e8, maxiter=iterations)
            native = {"istop": int(output[1]), "iterations": int(output[2]), "residual_norm": float(output[3]),
                      "normal_residual_norm": float(output[4]), "matrix_norm": float(output[5]), "condition_estimate": float(output[6])}
        delta = scale * output[0]
    elapsed = time.perf_counter() - started
    certified = certificate(Z, r, delta, penalty)
    passed = native["istop"] in (0, 1, 2, 4, 5) and certified["certified"]
    return delta, {"variant": variant, "penalty": penalty, "iterations_limit": iterations,
        "atol": 1e-8 if variant == "original_lsqr" else 1e-12,
        "btol": 1e-8 if variant == "original_lsqr" else 1e-12, "conlim": 1e8,
        "solver_seconds_diagnostic_only": elapsed, "native": native, "certificate": certified,
        "success": bool(passed), "no_heldout_score": True}


def prepare(args):
    parent = parent_module()
    base = parent.base
    runtime = base.bootstrap(args)
    if args.output.exists():
        raise ValueError("Preserve earlier numerical attempts")
    prior = base.read(args.previous / "plan.json")
    controls = base.read(args.previous / "scalar_controls.json")
    anchor_path = args.previous / "selection_fits/diamonds/lightgbm/anchor.json"
    anchor = base.read(anchor_path)
    if controls["all_fits_successful"] or prior["script_sha256"] != base.digest(PARENT_PATH):
        raise ValueError("Expected the unchanged failed wave17 selection")
    checked_args = argparse.Namespace(**vars(args))
    checked_args.output, checked_args.previous = args.previous, Path(prior["previous"])
    parent.checked(checked_args)
    job = next(job for job in prior["jobs"] if (job["dataset"], job["library"]) == ("diamonds", "lightgbm"))
    if (anchor["status"] != "ok" or base.digest(anchor["model"]["model_path"]) != anchor["model"]["model_sha256"]
            or base.digest(job["expanded_source"]["path"]) != job["expanded_source"]["sha256"]
            or anchor["fitting_data"]["source_sha256"] != job["expanded_source"]["sha256"]
            or anchor["fit"]["requested_rounds"] != job["rounds"] or not anchor["fitting_data"]["stop_excluded"]):
        raise ValueError("Frozen anchor or expanded source identity differs")
    anchor_rows, source_rows = anchor["fitting_data"], job["expanded_source"]["receipt"]
    if (any(anchor_rows[key] != source_rows[key] for key in ("original_train_rows", "original_stop_rows"))
            or anchor_rows["outside_cap_rows"] != source_rows["added_rows"]
            or anchor_rows["rows"] != source_rows["original_train_rows"] + source_rows["added_rows"]
            or anchor_rows["row_order"] != "TRAIN then ascending outside-cap"):
        raise ValueError("Anchor fitting row manifest differs from inherited source")
    dependencies = dict(prior["dependencies"])
    for path in (PARENT_PATH, anchor_path, args.previous / "plan.json", args.previous / "scalar_controls.json",
                 args.previous / "selection_fits/diamonds/lightgbm/candidate_0.json", Path(anchor["model"]["model_path"]),
                 Path(job["expanded_source"]["path"])):
        dependencies[str(path)] = base.digest(path)
    plan = {"protocol": "train_only_augmented_ridge_numerical_diagnostic_20261016", "created_at": base.now(),
        "script_sha256": base.digest(__file__), "runtime": runtime, "scipy_version": prior["scipy_version"],
        "dependencies": dependencies, "previous": str(args.previous), "anchor": anchor, "job": job,
        "variants": ["original_lsqr", "scaled_lsqr", "scaled_lsmr"], "penalty": 1., "iterations": 2000,
        "certificate": {"maximum_relative_original_dual_gap": 1e-10, "maximum_normalized_original_gradient": 1e-8,
                        "accepted_statuses": [0, 1, 2, 4, 5], "reject_iteration_limit_even_if_certificate_passes": True},
        "resources": {"cpus": [4], "maximum_seconds": 1200, "maximum_rss_bytes": 8 * 1024**3,
                      "maximum_named_array_bytes": 4 * 1024**3,
                      "extra_diagnostic_vector_reserve_bytes_per_row_or_column": 256,
                      "scope": "one complete design plus three sequential solver variants; add 256*(rows+columns) beyond the original LSQR vector reserve"},
        "data_access": "Decode the existing authorized expanded pickle, which contains original STOP rows; select only original TRAIN plus outside-cap rows using frozen counts/order. No STOP role file is decoded; its rows and labels are excluded from the numerical problem and no STOP metric is computed. No DEV, confirmation or official-test values are parsed or scored; inherited dependency integrity checks may rehash previously recorded files as bytes.",
        "selection": "No quality selection or model mutation; retain all numerical successes/failures, no next-policy promotion",
        "limitations": "Training-only numerical diagnosis; differing tolerances prevent speed superiority claims. The original objective and coefficients remain in original coordinates."}
    base.write(args.output / "plan.json", plan)
    print({"plan_sha256": base.digest(args.output / "plan.json"), "variants": plan["variants"]})


def checked(args):
    import importlib.metadata
    parent = parent_module()
    base = parent.base
    plan = base.read(args.output / "plan.json")
    if (plan["script_sha256"] != base.digest(__file__) or plan["runtime"] != base.bootstrap(args)
            or args.previous != Path(plan["previous"])
            or plan["scipy_version"] != importlib.metadata.version("scipy")):
        raise ValueError("Frozen numerical code/runtime changed")
    for path, digest in plan["dependencies"].items():
        if base.digest(path) != digest:
            raise ValueError("Frozen numerical dependency changed")
    return parent, plan


def worker(args):
    import hashlib
    import numpy as np
    import psutil
    parent, plan = checked(args)
    base = parent.base
    psutil.Process().cpu_affinity(plan["resources"]["cpus"])
    directory = args.output / "numerics"
    if directory.exists():
        raise ValueError("Preserve started diagnosis")
    directory.mkdir()
    record = {"status": "failed", "started_at": base.now(), "variants": []}
    base.write(directory / "receipt.json", record)
    try:
        source = plan["job"]["expanded_source"]
        with Path(source["path"]).open("rb") as stream:
            expanded = pickle.load(stream)
        if expanded["receipt"] != source["receipt"]:
            raise ValueError("Expanded source receipt changed")
        receipt = plan["anchor"]["fitting_data"]
        train_count, stop_count = receipt["original_train_rows"], receipt["original_stop_rows"]
        selected = np.r_[np.arange(train_count), np.arange(train_count + stop_count, len(expanded["y"]))]
        row_ids = np.asarray(expanded["outer_train_positions"])[selected]
        if (len(selected) != receipt["rows"] or hashlib.sha256(row_ids.astype("<i8").tobytes()).hexdigest() != receipt["row_ids_sha256"]):
            raise ValueError("Exact frozen TRAIN-only row order differs")
        with Path(plan["anchor"]["model"]["model_path"]).open("rb") as stream:
            payload = pickle.load(stream)
        X, schema = base.normalize(expanded["X"].iloc[selected])
        labels = np.asarray(expanded["y"])[selected]
        if schema != payload["schema"] or schema != receipt["schema"]:
            raise ValueError("TRAIN-only schema differs from frozen anchor")
        memory = parent.checker().memory_preflight(payload, len(X), max_working_bytes=plan["resources"]["maximum_named_array_bytes"], solver_kind="regression")
        extra = plan["resources"]["extra_diagnostic_vector_reserve_bytes_per_row_or_column"] * (len(X) + memory["parameters_upper_bound"])
        memory["extra_diagnostic_vector_reserve_bytes"] = extra
        memory["total_diagnostic_named_array_budget_bytes"] = memory["refit_array_budget_bytes"] + extra
        if memory["total_diagnostic_named_array_budget_bytes"] > plan["resources"]["maximum_named_array_bytes"]:
            raise MemoryError("Frozen diagnostic array reserve exceeded")
        design = parent.helper("regression").prepare_design(payload, X, labels)
        r = labels - design["baseline"]
        base.write(directory / "design.json", {"rows": len(labels), "shape": list(design["Z"].shape), "nnz": design["Z"].nnz,
            "row_ids_sha256": receipt["row_ids_sha256"], "memory": memory, "training_only": True})
        del expanded, X, labels
        for variant in plan["variants"]:
            start = base.now()
            try:
                delta, result = solve(design["Z"], r, plan["penalty"], variant, iterations=plan["iterations"])
                coefficients = directory / (variant + "_coefficients.npy")
                np.save(coefficients, delta, allow_pickle=False)
                result.update(coefficients_sha256=base.digest(coefficients), status="ok" if result["success"] else "failed_certificate_or_status")
            except Exception:
                result = {"variant": variant, "status": "failed_exception", "error": traceback.format_exc()}
            result.update(started_at=start, finished_at=base.now())
            base.write(directory / (variant + ".json"), result)
            record["variants"].append(result)
            print({"variant": variant, "status": result["status"]}, flush=True)
        record["status"] = "completed"
    except Exception:
        record["error"] = traceback.format_exc()
    record["finished_at"] = base.now()
    base.write(directory / "receipt.json", record)


def run(args):
    parent, plan = checked(args)
    if (args.output / "worker.txt").exists():
        raise ValueError("Preserve prior started diagnostic")
    command = [sys.executable, "-I", str(Path(__file__).resolve()), "worker"]
    for name in ("output", "previous", "package_root", "dependencies"):
        command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    with (args.output / "worker.txt").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        resource = parent.cv.watchdog.watch(process, maximum_seconds=plan["resources"]["maximum_seconds"],
                                           maximum_rss_bytes=plan["resources"]["maximum_rss_bytes"])
    parent.base.write(args.output / "resource.json", resource)
    print(resource)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "worker", "run"))
    for name in ("output", "previous", "package-root", "dependencies"):
        parser.add_argument("--" + name, type=lambda value: Path(value).resolve(), required=True)
    arguments = parser.parse_args()
    globals()[arguments.stage](arguments)
