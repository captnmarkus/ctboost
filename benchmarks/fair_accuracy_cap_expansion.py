"""Mandatory larger-data refit on three verified official-TRAIN-only caches."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time
import traceback

DATA_PATH = Path(__file__).with_name("fair_accuracy_cap_data.py")
SEED_PATH = Path(__file__).with_name("fair_accuracy_cap_seed_check.py")
OUTER_PATH = Path(__file__).with_name("fair_accuracy_cap_outer_test.py")
POLICY = Path(__file__).parent / "results/improvements_20261016_followup/robustness_protocol_cap.json"
_spec = importlib.util.spec_from_file_location("draft_cap_data", DATA_PATH)
data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(data)

DRAFT_ONLY = False
LIBRARIES = ("ctboost", "xgboost", "lightgbm")
INITIAL_SEED = 47
CONDITIONAL_SEED = 239
PILOT_PROTOCOL_SHA256 = "1fa7e1b482a9048a52f75d2b554b015ee027da5eddad78bb43908c213c55eff5"
APPROVED_CACHE_HASHES = {
    "GiveMeSomeCredit": "0c5aa46c3324c6579ce6ac8bb64ef09025ac6c6dc3b88e35f884d211a3183280",
    "diamonds": "c2bc8052f073a196a3ca1c72331eff333c091d822eca1adc5f175f0c116566a0",
    "SDSS17": "f789e02b5e010a071c9f1121f90cc5c454d91478f14583a16990fee3bdabf32d",
}


def require_activated():
    if DRAFT_ONLY:
        raise RuntimeError("Inactive draft: real preparation, fitting, freezing and scoring are disabled")


def modules(root):
    sys.path.insert(0, str(root))
    from benchmarks import fair_accuracy_round_cv as cv
    return cv


cv = modules(Path(__file__).resolve().parents[1])
base, union, predict = cv.base, cv.union, cv.predict


def fit_policy(job, case, seed):
    """Reuse the same expanded source for all libraries and conditional seed239."""
    if case["dataset_name"] not in data.EXPAND_DATASETS:
        if job.get("expanded_source") is not None:
            raise ValueError("Ineligible task must retain its original union fitting rows")
        return cv.fit_policy(job, case, seed)
    source = job["expanded_source"]
    if source["dataset"] != case["dataset_name"] or base.digest(source["path"]) != source["sha256"]:
        raise ValueError("Frozen expanded fitting source changed")
    with Path(source["path"]).open("rb") as stream:
        expanded = pickle.load(stream)
    if expanded["receipt"] != source["receipt"]:
        raise ValueError("Expanded row provenance changed")
    return fit_expansion(job, case, expanded, seed, cv)


def verify_terminal_gate(anchor, anchor_hashes, terminal=None, *, anchor_directory=None, terminal_script_hash=None):
    """The policy anchor is always seed47, regardless of terminal gate branch."""
    plan, frozen, quality = anchor
    if (plan.get("wave") != 13 or plan.get("seed") != 47 or not frozen.get("all_fits_successful")
            or frozen["plan_sha256"] != anchor_hashes["plan.json"]
            or frozen["selected_cv_sha256"] != anchor_hashes["selected_cv.json"]
            or plan["fold_manifest_sha256"] != anchor_hashes["fold_manifest.json"]
            or quality["plan_sha256"] != anchor_hashes["plan.json"]
            or quality["frozen_selections_sha256"] != anchor_hashes["frozen_selections.json"]):
        raise ValueError("Expected intact wave13 seed47 policy and gate")
    if quality.get("development_target_met") is False:
        if terminal is not None:
            raise ValueError("Initial failed gate needs no alternate seed anchor")
        return "wave13_initial_seed47_failure"
    if quality.get("development_target_met") is not True or terminal is None:
        raise ValueError("Cap expansion requires a failed terminal gate")
    other_plan, other_frozen, other_quality, other_hashes = terminal
    if (other_plan.get("protocol") != "round_cv_union_fixed_seed137_20261016"
            or other_plan.get("seed") != 137 or not other_frozen.get("all_fits_successful")
            or other_plan.get("runtime") != plan["runtime"]
            or other_plan.get("scipy_version") != plan["scipy_version"]
            or terminal_script_hash is None or other_plan.get("script_sha256") != terminal_script_hash
            or other_quality.get("development_target_met") is not False
            or other_frozen["plan_sha256"] != other_hashes["plan.json"]
            or other_quality["plan_sha256"] != other_hashes["plan.json"]
            or other_quality["frozen_selections_sha256"] != other_hashes["frozen_selections.json"]
            or Path(other_plan["previous"]).resolve() != Path(anchor_directory).resolve()):
        raise ValueError("Expected failed linked wave13 seed137 mean gate")
    for name, expected in anchor_hashes.items():
        path = str((Path(anchor_directory) / name).resolve())
        if other_plan["dependencies"].get(path) != expected:
            raise ValueError("Terminal seed gate is not bound to the exact seed47 anchor")
    mean = [row for row in other_quality["summary"] if row["seed"] == "mean" and row["family"] == "all"]
    if (len(mean) != 2 or {row["comparator"] for row in mean} != {"xgboost", "lightgbm"}
            or all(row["median_relative_error_gain"] >= 0 and row["wins"] + row["ties"] >= 6 for row in mean)):
        raise ValueError("Terminal report does not record a failed two-seed mean gate")
    return "wave13_linked_seed137_mean_failure"


def partition_policy(plan, frozen):
    """Nine mandatory refits and27 exact carries; never choose using DEV."""
    cases = plan["cases"]
    expected = {(case["dataset_name"], library) for case in cases for library in LIBRARIES}
    if (len(cases) != 12 or len(expected) != 36 or not set(data.EXPAND_DATASETS).issubset({c["dataset_name"] for c in cases})
            or len(frozen["jobs"]) != 36 or len(frozen["selections"]) != 36
            or {(j["dataset"], j["library"]) for j in frozen["jobs"]} != expected
            or {(j["dataset"], j["library"]) for j in frozen["selections"]} != expected):
        raise ValueError("Expected the complete twelve-task/three-library single-model policy")
    selections = {(s["dataset"], s["library"]): s for s in frozen["selections"]}
    refits, carried = [], []
    for job in frozen["jobs"]:
        source = selections[job["dataset"], job["library"]]
        if type(job["rounds"]) is not int or job["rounds"] <= 0:
            raise ValueError("Missing fixed requested rounds")
        if source["cv_selection"]["job"] != job:
            raise ValueError("CV-selected requested controls differ from final anchor job")
        if job["dataset"] in data.EXPAND_DATASETS:
            refits.append(copy.deepcopy(job))
        else:
            carried.append(copy.deepcopy(source))
    if len(refits) != 9 or len(carried) != 27:
        raise ValueError("Expected nine expansions and27 exact carried models")
    return refits, carried


def load_expansion(case, root, base):
    """Future activated preparation only; no general dataset loader exists here."""
    if case["dataset_name"] not in data.EXPAND_DATASETS:
        raise ValueError("No cache extraction permitted for this task")
    source = root / ".tmp/tabarena-local-0159/pilot/data" / case["dataset_name"] / "outer_train.pkl"
    prepared_path = source.with_name("prepared.json")
    protocol_path = root / "benchmarks/tabarena/pilot_0159_v1.json"
    history_path = Path(case["source"]) / "plan.json"
    provenance = {str(path.resolve()): base.digest(path) for path in (source, prepared_path, protocol_path, history_path)}
    source_hash = provenance[str(source.resolve())]
    if source_hash != APPROVED_CACHE_HASHES[case["dataset_name"]] or base.digest(protocol_path) != PILOT_PROTOCOL_SHA256:
        raise ValueError("The specifically reviewed official-TRAIN-only cache changed")
    receipt = base.read(prepared_path)
    source_identity = data.verify_prepared_receipt(case, receipt, source_hash=source_hash,
                                                  protocol_hash=PILOT_PROTOCOL_SHA256)
    history = base.read(history_path)
    historical = next(c for c in history["datasets"] if c["dataset_name"] == case["dataset_name"])
    if historical["source_sha256"] != source_hash:
        raise ValueError("Historical role source differs from approved cache")
    # Only these three role paths are ever opened; confirmation metadata alone
    # comes from the historical plan. No OpenML/test/cache-extraction API is used.
    roles = {name: base.load_role(case, name) for name in ("train", "stop", "development")}
    for name in roles:
        provenance[str(Path(case["roles"][name]["path"]).resolve())] = case["roles"][name]["sha256"]
    with source.open("rb") as stream:
        original = pickle.load(stream)
    result = data.expand_rows(case, original, roles, historical["roles"])
    result["receipt"].update(source_identity, source_prepared_sha256=base.digest(prepared_path),
                             source_data_sha256=source_hash, provenance=provenance)
    return result


def fit_expansion(job, case, expanded, seed, cv):
    """Same fixed policy as wave13, fresh vocabulary on expanded fitting rows."""
    X, schema = cv.base.normalize(expanded["X"])
    verification, _ = cv.base.normalize(expanded["verification_frame"], schema)
    payload, details = cv.union.fit_fixed(job, case, X, expanded["y"], schema, seed)
    payload, correction = cv.threshold.correct(payload, job, X, expanded["y"], verification)
    if job.get("temperature") is not None:
        payload = {**payload, "temperature": job["temperature"]}
    details.update(correction=correction, temperature=job.get("temperature"),
                   expanded_data=expanded["receipt"], vocabulary_fit_on="expanded fitting rows only",
                   expanded_category_schema=schema)
    return payload, details, expanded["verification_frame"]


def prepare(args):
    require_activated()
    cv = modules(args.root)
    base = cv.base
    runtime = base.bootstrap(args)
    if args.output.exists() or args.artifacts.exists():
        raise ValueError("Preserve every previous attempt")
    names = ("plan.json", "frozen_selections.json", "development.json", "selected_cv.json", "fold_manifest.json")
    hashes = {name: base.digest(args.previous / name) for name in names}
    anchor = tuple(base.read(args.previous / name) for name in names[:3])
    plan, frozen, _ = anchor
    if (plan["runtime"] != runtime or plan["script_sha256"] != base.digest(cv.__file__)
            or plan["scipy_version"] != importlib.metadata.version("scipy")):
        raise ValueError("Anchor runtime or implementation changed")
    terminal = None
    dependencies = dict(plan["dependencies"])
    dependencies.update({str((args.previous / name).resolve()): value for name, value in hashes.items()})
    if args.terminal:
        terminal_names = names[:3]
        terminal_hashes = {name: base.digest(args.terminal / name) for name in terminal_names}
        documents = [base.read(args.terminal / name) for name in terminal_names]
        terminal = (*documents, terminal_hashes)
        dependencies.update(documents[0]["dependencies"])
        dependencies.update({str((args.terminal / name).resolve()): value for name, value in terminal_hashes.items()})
    trigger = verify_terminal_gate(anchor, hashes, terminal, anchor_directory=args.previous,
                                   terminal_script_hash=base.digest(cv.SEED_PATH) if terminal else None)
    refits, carried = partition_policy(plan, frozen)
    for path, expected in dependencies.items():
        if base.digest(path) != expected:
            raise ValueError("Parent source or dependency changed")
    for model in frozen["selections"]:
        if base.digest(model["model_path"]) != model["model_sha256"]:
            raise ValueError("Anchor model changed")
        dependencies[str(Path(model["model_path"]).resolve())] = model["model_sha256"]
    for path in (Path(__file__), DATA_PATH, SEED_PATH, OUTER_PATH, POLICY, Path(cv.__file__), Path(cv.WATCHDOG_PATH)):
        dependencies[str(path.resolve())] = base.digest(path)
    expansions = []
    args.artifacts.mkdir(parents=True)
    for case in plan["cases"]:
        if case["dataset_name"] not in data.EXPAND_DATASETS:
            continue
        expanded = load_expansion(case, args.root, base)
        dependencies.update(expanded["receipt"]["provenance"])
        path = args.artifacts / "expanded" / (case["dataset_name"] + ".pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            pickle.dump(expanded, stream, protocol=5)
        expansions.append({"dataset": case["dataset_name"], "path": str(path),
                           "sha256": base.digest(path), "receipt": expanded["receipt"]})
    sources = {item["dataset"]: item for item in expansions}
    jobs = [{**job, "expanded_source": sources[job["dataset"]]} if job["dataset"] in sources else copy.deepcopy(job)
            for job in frozen["jobs"]]
    if len(sources) != 3 or any(len([j for j in jobs if j["dataset"] == name and j["expanded_source"] == source]) != 3
                                for name, source in sources.items()):
        raise ValueError("Each eligible task must share one exact expanded source across all three libraries")
    refits = [j for j in jobs if j["dataset"] in sources]
    output = {"protocol": "mandatory_sample_cap_release_20261016", "wave": 14, "created_at": base.now(),
        "script_sha256": base.digest(__file__), "dependencies": dependencies, "runtime": runtime,
        "scipy_version": importlib.metadata.version("scipy"),
        "previous": str(args.previous), "terminal_gate": str(args.terminal) if args.terminal else None,
        "trigger": trigger, "seed": 47, "conditional_seed": 239, "cases": plan["cases"],
        "jobs": refits, "all_anchor_jobs": jobs, "carry_selections": carried,
        "expanded_data": expansions, "target": plan["target"], "official_test_scored": False,
        "resources": {"workers": 1, "cpus": [4], "threads_per_fit": 1,
                      "maximum_job_seconds": 1200, "maximum_job_rss_bytes": 8 * 1024 ** 3},
        "selection": "None: nine mandatory larger-data fits,27 exact carried models. Same requested rounds, controls, corrections and T. No earlystop, HPO, old/new DEV choice or fallback.",
        "limitations": "Adaptive development; increased training rows equally for all libraries on only three approved cache tasks. Houses remains unchanged. Full official-TRAIN labels reconstruct historical stratification/vocabulary only; excluded rows never enter fitting or normalization."}
    base.write(args.output / "plan.json", output)
    print({"plan_sha256": base.digest(args.output / "plan.json"), "new_fits": 9, "carried_models": 27,
           "expanded_rows": {entry["dataset"]: entry["receipt"]["rows"] for entry in expansions}})


def checked(args):
    require_activated()
    cv = modules(args.root)
    plan = cv.base.read(args.output / "plan.json")
    if (plan["script_sha256"] != cv.base.digest(__file__) or plan["runtime"] != cv.base.bootstrap(args)
            or plan["scipy_version"] != importlib.metadata.version("scipy")
            or Path(plan["previous"]) != args.previous):
        raise ValueError("Frozen implementation/runtime/parent changed")
    for path, expected in plan["dependencies"].items():
        if cv.base.digest(path) != expected:
            raise ValueError("Frozen dependency changed")
    return cv, plan


def fit(args):
    cv, plan = checked(args)
    import psutil
    psutil.Process().cpu_affinity(plan["resources"]["cpus"])
    job = next(j for j in plan["jobs"] if (j["dataset"], j["library"]) == (args.dataset, args.library))
    case = next(c for c in plan["cases"] if c["dataset_name"] == args.dataset)
    source = next(item for item in plan["expanded_data"] if item["dataset"] == args.dataset)
    record = {"dataset": args.dataset, "library": args.library, "status": "failed", "started_at": cv.base.now()}
    path = args.output / "fits" / args.dataset / (args.library + ".json")
    if path.exists():
        raise ValueError("A started fit is terminal")
    cv.base.write(path, record)
    try:
        if job["expanded_source"] != source:
            raise ValueError("Job does not use the task's common frozen expanded source")
        payload, details, frame = fit_policy(job, case, plan["seed"])
        model_path = args.artifacts / "models" / args.dataset / (args.library + ".pkl")
        selection = {"dataset": args.dataset, "library": args.library, "source_wave": 14,
            "seed": plan["seed"], "fixed_job": job, "mandatory_cap_expansion": True,
            "requested_rounds": job["rounds"], "leaf_refit": job.get("leaf_refit"),
            "binary_refit": job.get("binary_refit"), "temperature": job.get("temperature"),
            "best": {"parameters": job["parameters"], "retained_rounds": details["retained_rounds"]},
            "refit_details": details, **cv.threshold.persist(payload, model_path, frame)}
        record.update(status="ok", selection=selection)
    except Exception:
        record["error"] = traceback.format_exc()
    record["finished_at"] = cv.base.now()
    cv.base.write(path, cv.union.refit.plain_receipt(record))


def run(args):
    cv, plan = checked(args)
    directory = args.output / "logs"
    directory.mkdir(exist_ok=False)
    selections = copy.deepcopy(plan["carry_selections"])
    failures = []
    for job in plan["jobs"]:
        while (args.artifacts / "pause_fits").exists():
            time.sleep(.5)
        command = [sys.executable, "-I", str(Path(__file__).resolve()), "fit"]
        for name in ("root", "output", "artifacts", "package_root", "dependencies", "previous"):
            command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
        command.extend(["--dataset", job["dataset"], "--library", job["library"]])
        name = f"{job['dataset']}-{job['library']}"
        with (directory / (name + ".txt")).open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            resource = cv.watchdog.watch(process, maximum_seconds=plan["resources"]["maximum_job_seconds"],
                                        maximum_rss_bytes=plan["resources"]["maximum_job_rss_bytes"])
        cv.base.write(directory / (name + "-resource.json"), resource)
        if not resource["cleanup_complete"]:
            raise RuntimeError("Owned worker cleanup failed; stop dispatch")
        path = args.output / "fits" / job["dataset"] / (job["library"] + ".json")
        result = cv.base.read(path) if path.exists() else {"status": "failed", "error": "not_completed"}
        if result["status"] == "ok" and resource["exit_code"] == 0 and resource["resource_failure"] is None:
            selections.append(result["selection"])
        else:
            failures.append({"dataset": job["dataset"], "library": job["library"], "result": result, "resource": resource})
        print({"dataset": job["dataset"], "library": job["library"], "status": result["status"],
               "resource_failure": resource["resource_failure"]}, flush=True)
    expected = {(case["dataset_name"], library) for case in plan["cases"] for library in LIBRARIES}
    success = not failures and len(selections) == 36 and {(s["dataset"], s["library"]) for s in selections} == expected
    cv.base.write(args.output / "frozen_selections.json", {"plan_sha256": cv.base.digest(args.output / "plan.json"),
        "created_at": cv.base.now(), "selections": selections, "all_fits_successful": success,
        "jobs": plan["all_anchor_jobs"],
        "new_fits": 9, "carried_models": 27, "failures": failures,
        "development_scored": False, "official_test_scored": False})


def development(args):
    cv, plan = checked(args)
    frozen = cv.base.read(args.output / "frozen_selections.json")
    if frozen["plan_sha256"] != cv.base.digest(args.output / "plan.json") or not frozen["all_fits_successful"]:
        raise ValueError("All nine new fits plus27 frozen carries must pass before DEV")
    def gate_checked(inner_args):
        return checked(inner_args)[1]
    cv.base.checked, cv.base.predict = gate_checked, cv.predict
    cv.base.development(args)


if __name__ == "__main__":
    require_activated()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "fit", "development"))
    for name in ("root", "output", "artifacts", "package-root", "dependencies", "previous"):
        parser.add_argument("--" + name, type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--terminal", type=lambda value: Path(value).resolve())
    parser.add_argument("--dataset")
    parser.add_argument("--library", choices=LIBRARIES)
    arguments = parser.parse_args()
    globals()[arguments.stage](arguments)
