"""One-shot official outer-test assessment after the two-seed development gate.

Preparation reads metadata, indices and file hashes only. Assessment uses the
already frozen models and never fits, selects or modifies a model.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path
import pickle
import traceback
import xml.etree.ElementTree as ET

SEED_PATH = Path(__file__).with_name("fair_accuracy_multiclass_seed_check.py")
spec = importlib.util.spec_from_file_location("fair_seed_check", SEED_PATH)
seed_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seed_check)
base = seed_check.base
NS = "{http://openml.org/openml}"
CACHE = base.ROOT / ".tmp/tabarena-full-hpo25-0160-20260907/local/openml_cache/org/openml/www"


def splits(path):
    import numpy as np
    attributes, train, test = [], [], []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.lower().startswith("@attribute"):
                attributes.append(line.split()[1].strip("'\""))
            if line.strip().lower() == "@data":
                break
        if attributes not in (["type", "rowid", "repeat", "fold"], ["type", "rowid", "repeat", "fold", "sample"]):
            raise ValueError(f"Unexpected split ARFF schema: {attributes}")
        for values in csv.reader(stream):
            if not values or values[0].startswith("%"):
                continue
            row = dict(zip(attributes, values, strict=True))
            if any(int(row.get(key, "0")) != 0 for key in ("repeat", "fold", "sample")):
                continue
            if row["type"] not in ("TRAIN", "TEST"):
                raise ValueError("Unexpected split role")
            (train if row["type"] == "TRAIN" else test).append(int(row["rowid"]))
    train, test = np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64)
    if not len(train) or not len(test) or len(np.unique(np.concatenate([train, test]))) != len(train) + len(test):
        raise ValueError("Official partitions overlap or have duplicates")
    return train, test


def index_hash(indices, length_prefixed=False):
    prefix = len(indices).to_bytes(8, "little") if length_prefixed else b""
    return base.hashlib.sha256(prefix + indices.astype("<i8").tobytes()).hexdigest()


def prepare(args):
    runtime = base.bootstrap(args)
    if args.output.exists() or args.artifacts.exists():
        raise ValueError("Preserve previous attempts; choose fresh paths")
    parent = base.read(args.previous / "plan.json")
    frozen = base.read(args.previous / "frozen_selections.json")
    development = base.read(args.previous / "development.json")
    if (not development["development_target_met"] or not frozen["all_fits_successful"]
            or parent["runtime"] != runtime or parent["seed"] != 457
            or parent["protocol"] != "multiclass_fixed_seed457_20261016"
            or parent["script_sha256"] != base.digest(SEED_PATH)
            or frozen["plan_sha256"] != base.digest(args.previous / "plan.json")
            or development["plan_sha256"] != base.digest(args.previous / "plan.json")
            or development["frozen_selections_sha256"] != base.digest(args.previous / "frozen_selections.json")):
        raise ValueError("The frozen two-seed development gate must pass first")
    if len(parent["cases"]) != 12 or len({c["dataset_name"] for c in parent["cases"]}) != 12:
        raise ValueError("Expected12 unique datasets")
    dependencies = dict(parent["dependencies"])
    for path, expected in dependencies.items():
        if base.digest(path) != expected:
            raise ValueError("Parent implementation dependency changed")
    dependencies.update({str(args.previous / name): base.digest(args.previous / name)
                         for name in ("plan.json", "frozen_selections.json", "development.json")})
    dependencies[str(seed_check.TEMPERATURE_PATH)] = base.digest(seed_check.TEMPERATURE_PATH)
    cache_cases, models = [], []
    for case in parent["cases"]:
        task_path = CACHE / "tasks" / str(case["task_id"]) / "task.xml"
        tree = ET.parse(task_path)
        if int(tree.find(".//" + NS + "task_id").text) != case["task_id"]:
            raise ValueError("Task ID mismatch")
        dataset_id = int(tree.find(".//" + NS + "data_set_id").text)
        target = tree.find(".//" + NS + "target_feature").text
        directory = CACHE / "datasets" / str(dataset_id)
        description = directory / "description.xml"
        if ET.parse(description).find(".//" + NS + "name").text != case["dataset_name"]:
            raise ValueError("Dataset identity mismatch")
        split_path = task_path.with_name("datasplits.arff")
        train, test = splits(split_path)
        local_positions = Path(case["source"]).name == "accuracy-grouped-scout-v1"
        if local_positions:
            receipt_path = base.ROOT / ".tmp/tabarena-local-0159/pilot/data" / case["dataset_name"] / "prepared.json"
            dependencies[str(receipt_path)] = base.digest(receipt_path)
            expected_train_hash = base.read(receipt_path)["outer_train_indices_sha256"]
        else:
            expected_train_hash = case["outer_train_index_sha256"]
            if index_hash(test) != case["outer_test_index_sha256"]:
                raise ValueError("Previously registered outer-test indices differ")
        # The older pilot receipt prefixes each array with its length; the
        # remaining-panel receipt hashes the raw little-endian int64 bytes.
        if len(train) != case["original_rows"] or index_hash(train, local_positions) != expected_train_hash:
            raise ValueError("Official outer-training indices do not match the cached roles")
        parquet = directory / f"dataset_{dataset_id}.pq"
        cache_cases.append({"case": case, "dataset_id": dataset_id, "target": target,
            "parquet": str(parquet), "split": str(split_path), "positions_are_local_to_outer_train": local_positions,
            "outer_train_rows": len(train), "outer_test_rows": len(test),
            "outer_train_index_sha256": index_hash(train), "outer_test_index_sha256": index_hash(test)})
        for path in (task_path, split_path, description, parquet):
            dependencies[str(path)] = base.digest(path)
        for library in base.LIBRARIES:
            selected = next(s for s in frozen["selections"] if s["dataset"] == case["dataset_name"] and s["library"] == library)
            for seed, selection in ((47, selected["original_selection"]), (457, selected)):
                if base.digest(selection["model_path"]) != selection["model_sha256"]:
                    raise ValueError("Selected model changed before final freeze")
                models.append({"dataset": case["dataset_name"], "library": library, "seed": seed,
                    "model_path": selection["model_path"], "model_sha256": selection["model_sha256"]})
    if len(models) != 72 or len({(m["dataset"], m["library"], m["seed"]) for m in models}) != 72:
        raise ValueError("Expected72 unique frozen models")
    plan = {"protocol": "official_outer_test_one_shot_20261016", "created_at": base.now(),
        "script_sha256": base.digest(__file__), "base_script_sha256": base.digest(base.__file__),
        "seed_script_sha256": base.digest(SEED_PATH), "policy_sha256": base.digest(seed_check.POLICY),
        "runtime": runtime, "dependencies": dependencies, "previous": str(args.previous),
        "cases": cache_cases, "models": models, "models_fit_after_freeze": False,
        "assessment": "Score uncapped original repeat0/fold0/sample0 outer-test using existing72models; no refit or post-test tuning.",
        "aggregation": "Average primary error across the2seeds within each dataset, then12-task median relative error gains and wins. Also report each seed separately.",
        "limits": "Partial outer-training models, historically reused adaptive development search,12tasks and2learner seeds. Not a full TabArena run, Elo, universal ranking or significance claim."}
    args.artifacts.mkdir(parents=True)
    base.write(args.output / "plan.json", plan)
    print({"plan_sha256": base.digest(args.output / "plan.json"), "models": len(models),
           "outer_test_rows": sum(c["outer_test_rows"] for c in cache_cases)})


def checked(args):
    plan = base.read(args.output / "plan.json")
    if (plan["script_sha256"] != base.digest(__file__) or plan["base_script_sha256"] != base.digest(base.__file__)
            or plan["seed_script_sha256"] != base.digest(SEED_PATH)
            or plan["policy_sha256"] != base.digest(seed_check.POLICY) or plan["runtime"] != base.bootstrap(args)):
        raise ValueError("Frozen implementation/runtime changed")
    for path, expected in plan["dependencies"].items():
        if base.digest(path) != expected:
            raise ValueError("Frozen cache or parent evidence changed")
    return plan


def assess(args):
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    plan = checked(args)
    if (args.output / "assessment_started.json").exists():
        raise ValueError("One-shot assessment already started; preserve this attempt")
    base.write(args.output / "assessment_started.json", {"created_at": base.now(), "plan_sha256": base.digest(args.output / "plan.json")})
    results = []
    try:
        for entry in plan["cases"]:
            case = entry["case"]
            train_indices, test_indices = splits(entry["split"])
            raw = pd.read_parquet(entry["parquet"])
            if len(raw) != len(train_indices) + len(test_indices):
                raise ValueError("Official split does not partition all source rows")
            training = base.load_role(case, "train")
            columns = list(training["X"].columns)
            if entry["target"] in columns:
                raise ValueError("Target appeared among model inputs")
            X, raw_y = raw[columns], raw[entry["target"]]
            if case["problem_type"] == "regression":
                labels = np.asarray(raw_y)
                classes = None
            else:
                encoder = LabelEncoder().fit(raw_y.iloc[train_indices])
                if len(encoder.classes_) != case["num_classes"]:
                    raise ValueError("Training label vocabulary changed")
                labels, classes = encoder.transform(raw_y), encoder.classes_.tolist()
            # Verify the exact data/encoding provenance using every already-open
            # role. Original14 caches store local outer-train positions; the
            # remaining37 caches store global source-row IDs.
            role_checks = {}
            for role in ("train", "stop", "development"):
                cached = training if role == "train" else base.load_role(case, role)
                ids = np.asarray(cached["outer_train_positions"], dtype=np.int64)
                if entry["positions_are_local_to_outer_train"]:
                    ids = train_indices[ids]
                if np.intersect1d(ids, test_indices).size or not np.isin(ids, train_indices).all():
                    raise ValueError("Development role overlaps official test")
                pd.testing.assert_frame_equal(X.iloc[ids].reset_index(drop=True), cached["X"].reset_index(drop=True),
                                              check_dtype=False, check_categorical=False)
                np.testing.assert_array_equal(labels[ids], cached["y"])
                role_checks[role] = {"rows": len(ids), "global_index_sha256": index_hash(ids)}
            row = {"dataset": case["dataset_name"], "problem_type": case["problem_type"],
                   "official_test_rows": len(test_indices), "class_labels": classes, "verified_roles": role_checks, "models": {}}
            for library in base.LIBRARIES:
                errors, evidence = {}, {}
                for seed in (47, 457):
                    selected = next(m for m in plan["models"] if (m["dataset"], m["library"], m["seed"]) == (case["dataset_name"], library, seed))
                    if base.digest(selected["model_path"]) != selected["model_sha256"]:
                        raise ValueError("Selected model changed after freeze")
                    with Path(selected["model_path"]).open("rb") as stream:
                        model = pickle.load(stream)
                    predictions = seed_check.predict(model, X.iloc[test_indices])
                    errors[str(seed)] = base.error(predictions, labels[test_indices], case["problem_type"])
                    path = args.artifacts / case["dataset_name"] / f"{library}-seed{seed}.npz"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(path, predictions=predictions, labels=labels[test_indices], global_row_indices=test_indices)
                    evidence[str(seed)] = {"predictions_sha256": base.digest(path), "model": selected}
                errors["mean"] = (errors["47"] + errors["457"]) / 2
                row["models"][library] = {"errors": errors, "evidence": evidence}
            results.append(row)
            base.write(args.output / "tasks" / (case["dataset_name"] + ".json"), row)
            print({"assessed": case["dataset_name"], "rows": len(test_indices)}, flush=True)
        summary, passed = seed_check.aggregate(results)
        base.write(args.output / "assessment.json", {"created_at": base.now(),
            "plan_sha256": base.digest(args.output / "plan.json"), "official_test_scored": True,
            "bounded_panel_target_met": passed, "results": results, "summary": summary, "limits": plan["limits"]})
        print({"bounded_panel_target_met": passed, "summary": [s for s in summary if s["family"] == "all"]})
    except Exception:
        base.write(args.output / "assessment_failure.json", {"created_at": base.now(), "completed_tasks": len(results),
            "error": traceback.format_exc(), "instruction": "Preserve this attempt. No tuning or model replacement on any observed test results."})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "assess"))
    for name in ("output", "artifacts", "package-root", "dependencies", "previous"):
        parser.add_argument("--" + name, type=lambda value: Path(value).resolve(), required=True)
    args = parser.parse_args()
    {"prepare": prepare, "assess": assess}[args.stage](args)





