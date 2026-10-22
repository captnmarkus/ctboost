"""Three mandatory houses refits from an authorized-row-only ARFF extraction."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
from pathlib import Path
import pickle
import subprocess
import sys
import time
import traceback
import urllib.request
import xml.etree.ElementTree as ET

CAP_PATH = Path(__file__).with_name("fair_accuracy_cap_expansion.py")
PARSER_PATH = Path(__file__).with_name("authorized_houses_arff.py")
SPLIT_PATH = Path(__file__).with_name("fair_accuracy_outer_test.py")
SEED_PATH = Path(__file__).with_name("fair_accuracy_houses_seed_check.py")
OUTER_PATH = Path(__file__).with_name("fair_accuracy_houses_outer_test.py")
POLICY = Path(__file__).parent / "results/improvements_20261016_followup/robustness_protocol_houses.json"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cap, selector, split_protocol = load("houses_cap_parent", CAP_PATH), load("houses_selector", PARSER_PATH), load("houses_metadata", SPLIT_PATH)
base, union, cv, predict = cap.base, cap.union, cap.cv, cap.predict
EXPAND_DATASETS = (*cap.data.EXPAND_DATASETS, "houses")
CACHE = base.ROOT / ".tmp/tabarena-full-hpo25-0160-20260907/local/openml_cache/org/openml/www"
SOURCE_URL = "https://openml.org/data/v1/download/22125245/houses.arff"
NS = "{http://openml.org/openml}"


def fit_policy(job, case, seed):
    if case["dataset_name"] != "houses":
        return cap.fit_policy(job, case, seed)
    source = job["expanded_source"]
    if source["dataset"] != "houses" or base.digest(source["path"]) != source["sha256"]:
        raise ValueError("Frozen houses fitting rows changed")
    with Path(source["path"]).open("rb") as stream:
        expanded = pickle.load(stream)
    if expanded["receipt"] != source["receipt"]:
        raise ValueError("Frozen houses provenance changed")
    return cap.fit_expansion(job, case, expanded, seed, cv)


def authorized_manifest(case, train, test, roles):
    """Reconstruct the regression cap from metadata/known role IDs only."""
    import numpy as np
    from sklearn.model_selection import train_test_split
    if (case["dataset_name"] != "houses" or case["problem_type"] != "regression"
            or case["original_rows"] != len(train) or len(train) + len(test) != selector.HOUSES_ROWS
            or cap.data.index_hash(train) != case["outer_train_index_sha256"]
            or cap.data.index_hash(test) != case["outer_test_index_sha256"]):
        raise ValueError("Official houses split identity changed")
    selected, _ = train_test_split(np.arange(len(train)), train_size=10000, random_state=47)
    selected.sort()
    capped = train[selected]
    ids = {}
    for name in ("train", "stop", "development"):
        ids[name] = cap.data.checked_ids(roles[name]["outer_train_positions"], selector.HOUSES_ROWS, name)
        if (len(ids[name]) != case["roles"][name]["rows"]
                or cap.data.index_hash(ids[name]) != case["roles"][name]["index_sha256"]):
            raise ValueError("Historical role IDs changed")
    joined = np.concatenate(list(ids.values()))
    if len(np.unique(joined)) != 10000 or not np.array_equal(np.sort(joined), np.sort(capped)):
        raise ValueError("TRAIN/STOP/DEV must exhaust the historical regression cap")
    additional = np.setdiff1d(train, capped)
    allowed = np.concatenate([ids["train"], ids["stop"], additional])
    excluded = np.concatenate([test, ids["development"]])
    if (len(np.unique(allowed)) != len(allowed) or np.intersect1d(allowed, excluded).size
            or len(allowed) + len(excluded) != selector.HOUSES_ROWS):
        raise ValueError("Authorized fitting rows intersect held-out roles")
    return {"dataset": "houses", "authorized_global_rows": allowed.tolist(),
        "authorized_rows_sha256": cap.data.index_hash(allowed), "added_rows": len(additional),
        "train_rows": len(ids["train"]), "stop_rows": len(ids["stop"]),
        "development_rows_excluded": len(ids["development"]), "official_test_rows_excluded": len(test),
        "confirmation_rows_excluded": 0, "total_rows": selector.HOUSES_ROWS,
        "cap_positions_sha256": cap.data.index_hash(capped), "cap_seed": 47,
        "role_index_sha256": {name: cap.data.index_hash(value) for name, value in ids.items()},
        "row_order": "original TRAIN, STOP, then outside-cap official TRAIN global IDs ascending",
        "feature_names": list(roles["train"]["X"].columns),
        "historical_feature_dtypes": [str(dtype) for dtype in roles["train"]["X"].dtypes],
        "historical_target_dtype": str(roles["train"]["y"].dtype)}


def prepare(args):
    runtime = base.bootstrap(args)
    if args.output.exists() or args.artifacts.exists():
        raise ValueError("Preserve every previous attempt")
    parent, frozen, quality = [base.read(args.previous / name) for name in ("plan.json", "frozen_selections.json", "development.json")]
    if (parent["wave"] != 14 or parent["seed"] != 47 or parent["runtime"] != runtime
            or parent["script_sha256"] != base.digest(CAP_PATH) or not frozen["all_fits_successful"]
            or parent["scipy_version"] != importlib.metadata.version("scipy") or quality["development_target_met"]
            or frozen["plan_sha256"] != base.digest(args.previous / "plan.json")
            or quality["plan_sha256"] != base.digest(args.previous / "plan.json")
            or quality["frozen_selections_sha256"] != base.digest(args.previous / "frozen_selections.json")
            or frozen["jobs"] != parent["all_anchor_jobs"]):
        raise ValueError("Expected intact failed wave14 policy")
    expected = {(c["dataset_name"], lib) for c in parent["cases"] for lib in base.LIBRARIES}
    if len(parent["cases"]) != 12:
        raise ValueError("Expected exactly12 tasks")
    for items in (frozen["jobs"], frozen["selections"]):
        if len(items) != 36 or {(r["dataset"], r["library"]) for r in items} != expected or len(expected) != 36:
            raise ValueError("Expected36 unique policies")
    dependencies = dict(parent["dependencies"])
    for path, expected_hash in dependencies.items():
        if base.digest(path) != expected_hash:
            raise ValueError("Parent dependency changed")
    for path in (CAP_PATH, PARSER_PATH, SPLIT_PATH, SEED_PATH, OUTER_PATH, POLICY):
        dependencies[str(path)] = base.digest(path)
    for name in ("plan.json", "frozen_selections.json", "development.json"):
        dependencies[str(args.previous / name)] = base.digest(args.previous / name)
    for selection in frozen["selections"]:
        if base.digest(selection["model_path"]) != selection["model_sha256"]:
            raise ValueError("Parent model changed")
        dependencies[selection["model_path"]] = selection["model_sha256"]
    for source in parent["expanded_data"]:
        if base.digest(source["path"]) != source["sha256"]:
            raise ValueError("Inherited expanded data changed")
        dependencies[source["path"]] = source["sha256"]
    case = next(c for c in parent["cases"] if c["dataset_name"] == "houses")
    task_path = CACHE / "tasks" / str(case["task_id"]) / "task.xml"
    description = CACHE / "datasets" / str(case["dataset_id"]) / "description.xml"
    split_path = task_path.with_name("datasplits.arff")
    task, desc = ET.parse(task_path), ET.parse(description)
    if (int(task.find(".//" + NS + "data_set_id").text) != case["dataset_id"]
            or task.find(".//" + NS + "target_feature").text != selector.ATTRIBUTES[-1]
            or desc.find(NS + "url").text != SOURCE_URL
            or desc.find(NS + "md5_checksum").text != selector.HOUSES_MD5
            or int(desc.find(NS + "file_id").text) != selector.HOUSES_FILE_ID):
        raise ValueError("Official source metadata changed")
    for path in (task_path, description, split_path):
        dependencies[str(path)] = base.digest(path)
    train, test = split_protocol.splits(split_path)
    roles = {name: base.load_role(case, name) for name in ("train", "stop", "development")}
    manifest = authorized_manifest(case, train, test, roles)
    if tuple(manifest["feature_names"]) != selector.ATTRIBUTES[:-1]:
        raise ValueError("Historical feature order differs from the narrow parser")
    args.artifacts.mkdir(parents=True)
    base.write(args.output / "authorized_rows.json", manifest)
    plan = {"protocol": "authorized_houses_cap_20261016", "wave": 15, "seed": 47, "conditional_seed": 349,
        "created_at": base.now(), "script_sha256": base.digest(__file__), "dependencies": dependencies,
        "runtime": runtime, "scipy_version": importlib.metadata.version("scipy"), "previous": str(args.previous),
        "cases": parent["cases"], "all_anchor_jobs": frozen["jobs"],
        "carry_selections": [s for s in frozen["selections"] if s["dataset"] != "houses"],
        "inherited_expanded_data": parent["expanded_data"], "target": parent["target"],
        "resources": parent["resources"], "official_test_scored": False,
        "authorized_rows_sha256": base.digest(args.output / "authorized_rows.json"),
        "source_url": SOURCE_URL, "source_md5": selector.HOUSES_MD5, "expected_source_bytes": 1446365,
        "selection": "None:3mandatory houses expansions plus33 exact wave14 carries, unchanged requested counts and fixed postfit controls. No DEV fallback.",
        "source_access": "Freeze IDs before body acquisition; acquire/hash bytes only; persist SHA receipt before authorized-row parser. Excluded TEST/DEV lines receive byte structural checks only, never decoding/tokenization/numeric conversion."}
    base.write(args.output / "plan.json", plan)
    print({"plan_sha256": base.digest(args.output / "plan.json"), "authorized_rows": len(manifest["authorized_global_rows"]),
           "added_rows": manifest["added_rows"], "source_body_acquired": False})


def checked(args):
    plan = base.read(args.output / "plan.json")
    if (plan["script_sha256"] != base.digest(__file__) or plan["runtime"] != base.bootstrap(args)
            or plan["scipy_version"] != importlib.metadata.version("scipy") or Path(plan["previous"]) != args.previous
            or plan["authorized_rows_sha256"] != base.digest(args.output / "authorized_rows.json")):
        raise ValueError("Frozen implementation/runtime/rows changed")
    for path, expected in plan["dependencies"].items():
        if base.digest(path) != expected:
            raise ValueError("Frozen dependency changed")
    return plan


def acquire(args):
    plan = checked(args)
    marker = args.output / "acquisition_started.json"
    if marker.exists():
        raise ValueError("Preserve this acquisition attempt")
    base.write(marker, {"plan_sha256": base.digest(args.output / "plan.json"), "started_at": base.now()})
    with urllib.request.urlopen(plan["source_url"], timeout=120) as response:
        raw = response.read(plan["expected_source_bytes"] + 1)
        final_url = response.geturl()
    if len(raw) != plan["expected_source_bytes"] or hashlib.md5(raw).hexdigest() != plan["source_md5"]:
        raise ValueError("Raw byte count or trusted source MD5 mismatch")
    path = args.artifacts / "houses-source.arff.bin"
    path.write_bytes(raw)
    receipt = {"plan_sha256": base.digest(args.output / "plan.json"), "created_at": base.now(),
        "source_url": plan["source_url"], "final_url": final_url, "path": str(path), "bytes": len(raw),
        "md5": hashlib.md5(raw).hexdigest(), "sha256": hashlib.sha256(raw).hexdigest(),
        "body_decoded": False, "data_rows_parsed": 0}
    base.write(args.output / "acquired_source.json", receipt)
    print({"source_sha256": receipt["sha256"], "bytes": len(raw), "data_rows_parsed": 0})


def materialize(args):
    import numpy as np
    import pandas as pd
    plan = checked(args)
    if (args.output / "prepared_data.json").exists():
        raise ValueError("Preserve frozen prepared data")
    source = base.read(args.output / "acquired_source.json")
    manifest = base.read(args.output / "authorized_rows.json")
    if (source["plan_sha256"] != base.digest(args.output / "plan.json") or source["md5"] != plan["source_md5"]
            or source["sha256"] != base.digest(source["path"]) or source["data_rows_parsed"] != 0):
        raise ValueError("Pre-parse source receipt changed")
    raw = Path(source["path"]).read_bytes()
    rows, parser_receipt = selector.select_authorized_rows(raw, tuple(manifest["authorized_global_rows"]),
        expected_md5=source["md5"], expected_sha256=source["sha256"])
    matrix = np.asarray(rows, dtype=np.float64)
    frame = pd.DataFrame(matrix[:, :-1], columns=selector.ATTRIBUTES[:-1])
    labels = matrix[:, -1]
    case = next(c for c in plan["cases"] if c["dataset_name"] == "houses")
    roles = {name: base.load_role(case, name) for name in ("train", "stop")}
    offset = 0
    for name in ("train", "stop"):
        role = roles[name]
        count = len(role["y"])
        pd.testing.assert_frame_equal(frame.iloc[offset:offset + count].reset_index(drop=True),
            role["X"].reset_index(drop=True), check_dtype=False, check_exact=True)
        np.testing.assert_array_equal(labels[offset:offset + count], role["y"])
        offset += count
    receipt = {**parser_receipt, "dataset": "houses", "rows": len(frame), "added_rows": manifest["added_rows"],
        "authorized_rows_file_sha256": plan["authorized_rows_sha256"],
        "source_receipt_sha256": base.digest(args.output / "acquired_source.json"),
        "original_train_stop_exact_parity": True, "normalization": "common numeric float32 after selected-only float64 parse"}
    expanded = {"X": frame, "y": labels, "outer_train_positions": np.asarray(manifest["authorized_global_rows"], dtype=np.int64),
                "verification_frame": roles["stop"]["X"], "receipt": receipt}
    path = args.artifacts / "expanded" / "houses.pkl"
    path.parent.mkdir(parents=True)
    with path.open("wb") as stream:
        pickle.dump(expanded, stream, protocol=5)
    expansion = {"dataset": "houses", "path": str(path), "sha256": base.digest(path), "receipt": receipt}
    jobs = [{**job, "expanded_source": expansion} if job["dataset"] == "houses" else job for job in plan["all_anchor_jobs"]]
    prepared = {"plan_sha256": base.digest(args.output / "plan.json"), "created_at": base.now(), "jobs": jobs,
        "source_receipt_sha256": base.digest(args.output / "acquired_source.json"),
        "expanded_data": [*plan["inherited_expanded_data"], expansion], "data_rows_parsed": len(frame)}
    base.write(args.output / "prepared_data.json", prepared)
    print({"prepared_data_sha256": base.digest(args.output / "prepared_data.json"), "authorized_rows": len(frame),
           "excluded_rows_decoded": parser_receipt["excluded_rows_decoded"]})


def prepared(args, plan):
    result = base.read(args.output / "prepared_data.json")
    acquisition = base.read(args.output / "acquired_source.json")
    if (result["plan_sha256"] != base.digest(args.output / "plan.json")
            or result["source_receipt_sha256"] != base.digest(args.output / "acquired_source.json")
            or acquisition["sha256"] != base.digest(acquisition["path"])):
        raise ValueError("Prepared source freeze changed")
    sources = {s["dataset"]: s for s in result["expanded_data"]}
    if len(result["expanded_data"]) != 4 or set(sources) != set(EXPAND_DATASETS) or len(result["jobs"]) != 36:
        raise ValueError("Expected four expanded sources and36 policies")
    for source in sources.values():
        if base.digest(source["path"]) != source["sha256"]:
            raise ValueError("Expanded fitting source changed")
    for anchor, job in zip(plan["all_anchor_jobs"], result["jobs"], strict=True):
        expected = {**anchor, "expanded_source": sources["houses"]} if anchor["dataset"] == "houses" else anchor
        if job != expected:
            raise ValueError("Prepared policy differs from frozen controls")
    return result


def fit(args):
    import psutil
    plan = checked(args)
    data = prepared(args, plan)
    psutil.Process().cpu_affinity(plan["resources"]["cpus"])
    job = next(j for j in data["jobs"] if (j["dataset"], j["library"]) == ("houses", args.library))
    case = next(c for c in plan["cases"] if c["dataset_name"] == "houses")
    path = args.output / "fits" / (args.library + ".json")
    if path.exists():
        raise ValueError("Preserve every started fit")
    record = {"dataset": "houses", "library": args.library, "status": "failed", "started_at": base.now()}
    base.write(path, record)
    try:
        payload, details, frame = fit_policy(job, case, 47)
        model_path = args.artifacts / "models" / "houses" / (args.library + ".pkl")
        selection = {"dataset": "houses", "library": args.library, "source_wave": 15, "seed": 47, "fixed_job": job,
            "requested_rounds": job["rounds"], "leaf_refit": job.get("leaf_refit"), "binary_refit": job.get("binary_refit"),
            "temperature": job.get("temperature"), "best": {"parameters": job["parameters"], "retained_rounds": details["retained_rounds"]},
            "refit_details": details, **cv.threshold.persist(payload, model_path, frame)}
        record.update(status="ok", selection=selection)
    except Exception:
        record["error"] = traceback.format_exc()
    record["finished_at"] = base.now()
    base.write(path, union.refit.plain_receipt(record))


def run(args):
    plan = checked(args)
    data = prepared(args, plan)
    data_hash = base.digest(args.output / "prepared_data.json")
    (args.output / "logs").mkdir(exist_ok=False)
    selections, failures = copy.deepcopy(plan["carry_selections"]), []
    for library in base.LIBRARIES:
        command = [sys.executable, "-I", str(Path(__file__).resolve()), "fit", "--library", library]
        for name in ("output", "artifacts", "package_root", "dependencies", "previous"):
            command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
        with (args.output / "logs" / (library + ".txt")).open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            resource = cv.watchdog.watch(process, maximum_seconds=plan["resources"]["maximum_job_seconds"],
                                         maximum_rss_bytes=plan["resources"]["maximum_job_rss_bytes"])
        base.write(args.output / "logs" / (library + "-resource.json"), resource)
        if not resource["cleanup_complete"]:
            raise RuntimeError("Owned worker cleanup failed; stop dispatch")
        path = args.output / "fits" / (library + ".json")
        record = base.read(path) if path.exists() else {"status": "failed", "error": "not_completed"}
        if record["status"] == "ok" and resource["exit_code"] == 0 and resource["resource_failure"] is None:
            selections.append(record["selection"])
        else:
            failures.append({"library": library, "record": record, "resource": resource})
        print({"library": library, "status": record["status"], "resource_failure": resource["resource_failure"]}, flush=True)
    if data_hash != base.digest(args.output / "prepared_data.json"):
        raise ValueError("Prepared policies changed during fitting")
    expected = {(c["dataset_name"], lib) for c in plan["cases"] for lib in base.LIBRARIES}
    success = not failures and len(selections) == 36 and {(s["dataset"], s["library"]) for s in selections} == expected
    base.write(args.output / "frozen_selections.json", {"plan_sha256": base.digest(args.output / "plan.json"),
        "prepared_data_sha256": data_hash, "created_at": base.now(), "jobs": data["jobs"], "selections": selections,
        "all_fits_successful": success, "new_fits": 3, "carried_models": 33, "failures": failures,
        "development_scored": False, "official_test_scored": False})


def development(args):
    frozen = base.read(args.output / "frozen_selections.json")
    if not frozen["all_fits_successful"] or frozen["prepared_data_sha256"] != base.digest(args.output / "prepared_data.json"):
        raise ValueError("All three mandatory fits and33 carries must be intact")
    base.checked, base.predict = checked, predict
    base.development(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "acquire", "materialize", "run", "fit", "development"))
    for name in ("output", "artifacts", "package-root", "dependencies", "previous"):
        parser.add_argument("--" + name, type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--library", choices=base.LIBRARIES)
    arguments = parser.parse_args()
    globals()[arguments.stage](arguments)
