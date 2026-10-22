"""Eight frozen CTBoost union replays for root-stopping diagnosis, never HPO.

Only original TRAIN and STOP roles may be loaded. Native profiler output is
diagnostic: its six-significant-digit p-values are not an exact test API.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import subprocess
import sys
import traceback

UNION_PATH = Path(__file__).with_name("fair_accuracy_union_refit.py")
spec = importlib.util.spec_from_file_location("root_replay_union", UNION_PATH)
union = importlib.util.module_from_spec(spec)
spec.loader.exec_module(union)
base = union.base
_original_load_role = base.load_role


def training_role_only(case, role):
    if role not in ("train", "stop"):
        raise ValueError("Root diagnosis may load only TRAIN and STOP")
    return _original_load_role(case, role)


base.load_role = training_role_only


def canonical_hash(value):
    return base.hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                          allow_nan=False).encode()).hexdigest()


def state_without_logging(model):
    state = copy.deepcopy(model._handle.export_state())
    state.pop("verbose", None)
    return state


def rounded_relation(token, threshold):
    """Compare only when a six-significant-digit rounding interval is clear."""
    value = float(token)
    if not math.isfinite(value):
        return "nonfinite"
    radius = 0.0 if value == 0 else .5 * 10.0 ** (math.floor(math.log10(abs(value))) - 5)
    if value - radius > threshold:
        return "above"
    if value + radius < threshold:
        return "below"
    return "rounded_boundary"


def parse_roots(path):
    roots, pending, search = [], None, None
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("[ctboost] "):
            continue
        words = line[len("[ctboost] "):].split()
        if not words:
            continue
        fields = dict(word.split("=", 1) for word in words[1:] if "=" in word)
        if words[0] == "node_hist" and fields.get("depth") == "0":
            if pending is not None:
                raise ValueError("Unclosed root histogram in native log")
            pending, search = fields, None
        elif words[0] == "node_search" and fields.get("depth") == "0":
            if pending is None or search is not None:
                raise ValueError("Unexpected root search in native log")
            search = fields
        elif words[0] == "tree_build":
            if pending is None:
                raise ValueError("Tree build without root histogram")
            roots.append({"iteration": int(fields["iteration"].split("/")[0]),
                          "histogram": pending, "search": search})
            pending, search = None, None
    if pending is not None:
        raise ValueError("Incomplete root profile")
    return roots


def classify_root(root, is_leaf, alpha):
    search = root["search"]
    if not is_leaf:
        if search is None or search["split_valid"] != "1":
            raise ValueError("Profile contradicts saved split topology")
        return "split_created"
    if search is None:
        return "pre_search_structural_limit"
    if int(search["feature"]) < 0 or int(search["tested_features"]) == 0:
        return "no_eligible_feature"
    if search["split_valid"] == "1":
        if int(search["left_rows"]) and int(search["right_rows"]):
            raise ValueError("Profile claims nonempty split but saved root is a leaf")
        return "empty_partition_after_search"
    relation = rounded_relation(search["stopping_p_value"], alpha)
    if relation == "above":
        return ("test_rejected_p_rounded_one" if float(search["stopping_p_value"]) == 1.
                else "test_rejected")
    if relation == "below":
        return "test_passed_no_feasible_positive_gain"
    return "rounded_p_boundary_unresolved"


def prepare(args):
    runtime = base.bootstrap(args)
    if args.output.exists() or args.artifacts.exists():
        raise ValueError("Preserve earlier attempts; choose fresh paths")
    parent = base.read(args.previous / "plan.json")
    frozen = base.read(args.previous / "frozen_selections.json")
    if (parent["wave"] != 9 or parent["seed"] != 47 or parent["runtime"] != runtime
            or parent["script_sha256"] != base.digest(UNION_PATH)
            or not frozen["all_fits_successful"]
            or frozen["plan_sha256"] != base.digest(args.previous / "plan.json")
            or parent["scipy_version"] != importlib.metadata.version("scipy")):
        raise ValueError("Expected intact frozen wave9/runtime")
    jobs = []
    for job in parent["jobs"]:
        case = next(case for case in parent["cases"] if case["dataset_name"] == job["dataset"])
        if job["library"] != "ctboost" or case["problem_type"] not in ("binary", "regression"):
            continue
        for key in ("feature_weights", "first_feature_use_penalties", "monotone_constraints", "interaction_constraints"):
            if job["parameters"].get(key) not in (None, [], (), {}):
                raise ValueError("This diagnosis requires the strict minimum-p feature path")
        if job["parameters"].get("random_strength", 0) != 0:
            raise ValueError("Randomized ranked feature search is outside this diagnosis")
        selected = next(s for s in frozen["selections"]
                        if s["dataset"] == job["dataset"] and s["library"] == "ctboost")
        if base.digest(selected["model_path"]) != selected["model_sha256"]:
            raise ValueError("Frozen model changed")
        # Remove even the unused development role from the executable plan.
        case = {**case, "roles": {role: case["roles"][role] for role in ("train", "stop")}}
        jobs.append({"job": job, "case": case, "reference": selected})
    if (len(jobs) != 8 or len({item["job"]["dataset"] for item in jobs}) != 8
            or Counter(item["case"]["problem_type"] for item in jobs) != {"binary": 4, "regression": 4}):
        raise ValueError("Expected exactly four binary and four regression CT models")
    dependencies = {str(path): base.digest(path) for path in
                    (UNION_PATH, union.BASE_PATH, union.REFIT_PATH, union.refit.HELPER_PATH,
                     union.TEMPERATURE_PATH, args.previous / "plan.json",
                     args.previous / "frozen_selections.json")}
    plan = {"protocol": "eight_frozen_union_root_replays_20261016", "created_at": base.now(),
            "script_sha256": base.digest(__file__), "runtime": runtime,
            "scipy_version": importlib.metadata.version("scipy"), "dependencies": dependencies,
            "previous": str(args.previous), "package_root": str(args.package_root),
            "dependencies_root": str(args.dependencies), "artifacts": str(args.artifacts),
            "seed": 47, "jobs": jobs, "cpus": [8], "threads": 1,
            "fit_policy": "Exact wave9 requested rounds/parameters and fixed regression correction; no early stopping or model selection",
            "allowed_roles": ["train", "stop"], "development_or_test_access": False,
            "only_behavioral_override": "CTBOOST_PROFILE=1; compare all native state except verbose metadata",
            "required_identity": "Exact final native state except verbose, pipeline state, inference predictions and leaf indices on union; same raw TRAIN/STOP predictions",
            "log_contract": "Root node_search plus tree_build boundaries. P values have six significant digits; report ambiguous threshold equality. No post-test cut infeasibility subtype is logged.",
            "purpose": "Diagnostic replay only, excluded from HPO trial count and performance timing"}
    args.artifacts.mkdir(parents=True)
    base.write(args.output / "plan.json", plan)
    print({"replays": len(jobs), "rounds": sum(item["job"]["rounds"] for item in jobs),
           "plan_sha256": base.digest(args.output / "plan.json")})


def checked(args):
    plan = base.read(args.output / "plan.json")
    if (plan["script_sha256"] != base.digest(__file__) or base.bootstrap(args) != plan["runtime"]
            or plan["scipy_version"] != importlib.metadata.version("scipy")
            or args.previous != Path(plan["previous"]) or args.artifacts != Path(plan["artifacts"])):
        raise ValueError("Frozen source/runtime/paths changed")
    for path, expected in plan["dependencies"].items():
        if base.digest(path) != expected:
            raise ValueError("Frozen dependency changed")
    return plan


def worker(args):
    import numpy as np
    import psutil
    plan = checked(args)
    if os.environ.get("CTBOOST_PROFILE") != "1":
        raise ValueError("Native profiler must be enabled before model construction")
    psutil.Process().cpu_affinity(plan["cpus"])
    item = plan["jobs"][args.index]
    reference = item["reference"]
    if base.digest(reference["model_path"]) != reference["model_sha256"]:
        raise ValueError("Reference model changed")
    with Path(reference["model_path"]).open("rb") as stream:
        saved = pickle.load(stream)
    replay, details, _ = union.fit_policy(item["job"], item["case"], 47)
    X, _, schema, train, stop, role_receipt = union.union_data(item["case"])
    if schema != saved["schema"] or schema != replay["schema"]:
        raise ValueError("Fitted union vocabulary differs")
    before, after = state_without_logging(saved["model"]), state_without_logging(replay["model"])
    if before != after:
        differing = sorted(key for key in set(before) | set(after) if before.get(key) != after.get(key))
        raise ValueError(f"Native model changed beyond verbose: {differing}")
    pipes = [payload["model"]._feature_pipeline for payload in (saved, replay)]
    states = [pipe.to_state() if pipe is not None else None for pipe in pipes]
    if states[0] != states[1]:
        raise ValueError("Pipeline state changed")
    prediction_hashes = {}
    for name, frame in (("union", X), ("original_train", train["X"]), ("original_stop", stop["X"])):
        expected, actual = [np.asarray(union.predict(payload, frame)) for payload in (saved, replay)]
        if expected.dtype != actual.dtype or expected.shape != actual.shape:
            raise ValueError("Prediction shape/dtype changed")
        np.testing.assert_array_equal(np.ascontiguousarray(expected).view(np.uint8),
                                      np.ascontiguousarray(actual).view(np.uint8))
        prediction_hashes[name] = base.hashlib.sha256(np.ascontiguousarray(actual).tobytes()).hexdigest()
    leaves = [np.asarray(payload["model"].predict_leaf_index(X)) for payload in (saved, replay)]
    np.testing.assert_array_equal(leaves[0], leaves[1])
    state_path = args.artifacts / f"{args.index:02d}-replay.pkl"
    with state_path.open("wb") as stream:
        pickle.dump(replay, stream, protocol=5)
    root_leaf = [len(tree["nodes"]) == 1 for tree in after["trees"]]
    base.write(args.artifacts / f"{args.index:02d}-identity.json", {
        "status": "ok", "dataset": item["job"]["dataset"], "rounds": item["job"]["rounds"],
        "reference_model_sha256": reference["model_sha256"], "replay_pickle_sha256": base.digest(state_path),
        "state_without_verbose_sha256": canonical_hash(after), "pipeline_sha256": canonical_hash(states[1]),
        "prediction_sha256": prediction_hashes,
        "leaf_indices_sha256": base.hashlib.sha256(np.ascontiguousarray(leaves[1]).tobytes()).hexdigest(),
        "exact_native_state_except_verbose": True, "exact_pipeline_state": True,
        "exact_prediction_bits": True, "exact_union_leaf_ids": True, "root_leaf": root_leaf,
        "role_receipt": role_receipt, "fixed_correction_reapplied": bool(item["job"]["leaf_refit"]),
        "fit_details": union.refit.plain_receipt(details)})


def run(args):
    import numpy as np
    plan = checked(args)
    if (args.output / "started.json").exists():
        raise ValueError("A started attempt is terminal; preserve it")
    base.write(args.output / "started.json", {"created_at": base.now(), "plan_sha256": base.digest(args.output / "plan.json")})
    results = []
    for index, item in enumerate(plan["jobs"]):
        record = {"dataset": item["job"]["dataset"], "status": "failed"}
        log = args.artifacts / f"{index:02d}-native.log"
        command = [sys.executable, "-I", str(Path(__file__).resolve()), "worker", "--index", str(index)]
        for name in ("output", "artifacts", "package_root", "dependencies", "previous"):
            command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
        env = dict(os.environ, CTBOOST_PROFILE="1")
        try:
            with log.open("wb") as stream:
                completed = subprocess.run(command, stdout=stream, stderr=stream, env=env, check=False)
            record.update(native_log_path=str(log), native_log_sha256=base.digest(log), exit_code=completed.returncode)
            if completed.returncode:
                raise ValueError("Replay subprocess failed; inspect preserved native log")
            identity_path = args.artifacts / f"{index:02d}-identity.json"
            identity = base.read(identity_path)
            roots = parse_roots(log)
            if len(roots) != item["job"]["rounds"] or len(roots) != len(identity["root_leaf"]):
                raise ValueError("Root log count differs from fixed native trees")
            if [root["iteration"] for root in roots] != list(range(1, len(roots) + 1)):
                raise ValueError("Root log iterations are not the expected contiguous sequence")
            alpha = float(item["job"]["parameters"].get("alpha", .05))
            for root, is_leaf in zip(roots, identity["root_leaf"]):
                root.update(is_leaf=is_leaf, reason=classify_root(root, is_leaf, alpha))
            roots_path = args.artifacts / f"{index:02d}-roots.json"
            base.write(roots_path, roots)
            pvalues = [float(root["search"]["stopping_p_value"]) for root in roots if root["search"] is not None]
            record.update(status="ok", alpha=alpha, rounds=len(roots), root_only_trees=sum(identity["root_leaf"]),
                          reason_counts=dict(Counter(root["reason"] for root in roots)),
                          rounded_root_p_quantiles=dict(zip(("min", "q25", "median", "q75", "max"),
                                                           np.quantile(pvalues, [0, .25, .5, .75, 1]).tolist())) if pvalues else {},
                          identity_path=str(identity_path), identity_sha256=base.digest(identity_path),
                          roots_path=str(roots_path), roots_sha256=base.digest(roots_path))
        except Exception:
            record["error"] = traceback.format_exc()
        results.append(record)
        base.write(args.output / "cases" / (item["job"]["dataset"] + ".json"), record)
        print({"dataset": record["dataset"], "status": record["status"], "reasons": record.get("reason_counts")}, flush=True)
    base.write(args.output / "summary.json", {"created_at": base.now(),
        "plan_sha256": base.digest(args.output / "plan.json"), "all_replays_exact": all(r["status"] == "ok" for r in results),
        "diagnostic_fits": 8, "hpo_fits": 0, "development_or_test_access": False, "results": results,
        "limits": "Frozen wave9 union models only; does not retroactively identify wave3 causes. Post-test split failure subtypes and p=1 degeneracy are not individually logged. P-value summaries use rounded native logs."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "worker"))
    parser.add_argument("--index", type=int)
    for name in ("output", "artifacts", "package-root", "dependencies", "previous"):
        parser.add_argument("--" + name, type=lambda value: Path(value).resolve(), required=True)
    arguments = parser.parse_args()
    {"prepare": prepare, "run": run, "worker": worker}[arguments.stage](arguments)
