"""New 37-task assessment of the unchanged grouped-eight candidate.

Reuses the original, unchanged fit worker but freezes a separate protocol and
new task population. It neither revises the failed 14-task decision nor reads
that panel's reserved confirmation rows. Only official outer-training rows
enter any fitted or scored role.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import pickle
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE_SOURCE = ROOT / ".tmp/tabarena-full-hpo25-0160-20260907/local/openml_cache"
XML_NS = "{http://openml.org/openml}"


def bootstrap():
    import ctboost
    if Path(ctboost.__file__).resolve().is_relative_to(ROOT / "ctboost"):
        raise RuntimeError("Use the isolated public-wheel Python with -I")
    sys.path.insert(0, str(ROOT))
    from benchmarks import accuracy_grouped_scout as base
    base.bootstrap()
    return base


def snapshot_cache(base, cases, output, cache):
    if cache.exists():
        raise ValueError("Preserve previous cache snapshots; use a new destination")
    source = CACHE_SOURCE / "org/openml/www"
    copied = []
    for case in cases:
        task = source / "tasks" / str(case["task_id"])
        description = ET.parse(task / "task.xml")
        if int(description.find(".//" + XML_NS + "task_id").text) != case["task_id"]:
            raise ValueError("Cached task ID mismatch")
        dataset_id = int(description.find(".//" + XML_NS + "data_set_id").text)
        dataset = source / "datasets" / str(dataset_id)
        metadata = ET.parse(dataset / "description.xml")
        if metadata.find(".//" + XML_NS + "name").text != case["dataset_name"]:
            raise ValueError("Cached dataset identity differs from the population")
        required = [task / "task.xml", task / "datasplits.arff", task / "datasplits.pkl.py3",
                    dataset / "description.xml", dataset / "features.xml",
                    dataset / f"dataset_{dataset_id}.pkl.py3", dataset / f"dataset_{dataset_id}.pq"]
        if not all(path.is_file() for path in required):
            raise ValueError(f"Incomplete official cache for {case['dataset_name']}")
        evidence = {}
        for directory in (task, dataset):
            for path in sorted(directory.iterdir()):
                if not path.is_file():
                    continue
                relative = path.relative_to(CACHE_SOURCE)
                digest = base.digest(path)
                target = cache / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                if base.digest(target) != digest or base.digest(path) != digest:
                    raise ValueError("Cached source changed while being snapshotted")
                evidence[relative.as_posix()] = digest
        copied.append({"task_id": case["task_id"], "dataset_id": dataset_id,
                       "dataset": case["dataset_name"], "files_sha256": evidence})
    base.write(output / "cache_provenance.json", {"at": base.now(), "cache_source": str(CACHE_SOURCE),
               "snapshot": str(cache), "source_cache_modified": False, "tasks": copied})
    return copied


def prepare(output, cache, runtime):
    base = bootstrap()
    import numpy as np
    import openml
    import requests
    from autogluon.common.utils.cv_splitter import CVSplitter
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    import ctboost

    runtime_code = """import ctboost, hashlib, importlib.metadata, json, pathlib, sys
print(json.dumps({'python': str(pathlib.Path(sys.executable).resolve()),
'native_sha256': hashlib.sha256(pathlib.Path(ctboost._core.__file__).read_bytes()).hexdigest(),
'versions': {name: importlib.metadata.version(name) for name in
('ctboost','autogluon.tabular','autogluon.core','xgboost','numpy','pandas','scikit-learn')}}))
"""
    execution = json.loads(subprocess.check_output([str(runtime), "-I", "-c", runtime_code], text=True))
    if execution["versions"]["ctboost"] != "0.1.60":
        raise ValueError("Execution runtime must use public CTBoost 0.1.60")
    if output.exists():
        raise ValueError("Use a new output directory; preserve earlier attempts")
    original = base.read(base.PROTOCOL)
    excluded = {case["dataset_name"] for case in original["datasets"]}
    population = [case for case in original["selection"]["population_metadata"]
                  if case["dataset_name"] not in excluded]
    if len(population) != 37 or {d["problem_type"] for d in population} != {"binary", "regression"}:
        raise ValueError("Expected the complete 37-task complement: 27 binary and 10 regression")
    provenance = snapshot_cache(base, population, output, cache)

    def no_network(*_args, **_kwargs):
        raise RuntimeError("Preparation requires the verified local snapshot; network access is disabled")

    requests.sessions.Session.request = no_network
    openml.config.set_root_cache_directory(str(cache))
    cases = []
    for metadata in population:
        task = openml.tasks.get_task(int(metadata["task_id"]), download_data=True,
                                    download_splits=True, download_qualities=False)
        train, test = task.get_train_test_split_indices(fold=0, repeat=0, sample=0)
        train, test = np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64)
        if len(np.intersect1d(train, test)) or len(np.unique(train)) != len(train):
            raise ValueError("Official outer partitions overlap or contain duplicated training rows")
        X_all, y_all, _, _ = task.get_dataset().get_data(target=task.target_name, dataset_format="dataframe")
        if len(train) + len(test) != len(X_all) or len(np.unique(np.concatenate([train, test]))) != len(X_all):
            raise ValueError("Official outer split does not partition the source dataset")
        outer_test_hash = base.hashlib.sha256(test.astype("<i8").tobytes()).hexdigest()
        outer_test_rows = len(test)
        X = X_all.iloc[train].copy().reset_index(drop=True)
        y = np.asarray(y_all.iloc[train])
        del X_all, y_all, test
        if X.shape[1] != metadata["num_features"]:
            raise ValueError("Feature metadata does not match the cached data")
        classification = metadata["problem_type"] == "binary"
        if classification:
            encoder = LabelEncoder().fit(y)
            if len(encoder.classes_) != metadata["num_classes"]:
                raise ValueError("Class-count metadata changed")
            y = encoder.transform(y)
        if not np.isfinite(np.asarray(y, dtype=np.float64)).all():
            raise ValueError("Outer-training targets must be finite")
        selected = np.arange(len(y))
        if len(selected) > 10000:
            selected, _ = train_test_split(selected, train_size=10000, random_state=47,
                                           stratify=y if classification else None)
            selected.sort()
        frame, labels = X.iloc[selected].reset_index(drop=True), y[selected]
        splitter = CVSplitter(n_splits=8, n_repeats=1, random_state=47, stratify=classification)
        folds = [validation for _, validation in splitter.split(frame, labels)]
        roles = {"train": np.sort(np.concatenate(folds[:5])), "stop": folds[5],
                 "development": np.sort(np.concatenate(folds[6:]))}
        joined = np.concatenate(list(roles.values()))
        if len(joined) != len(selected) or len(np.unique(joined)) != len(selected):
            raise ValueError("Training, stopping, and assessment roles must be disjoint")
        case = {**metadata, "dataset_id": task.dataset_id, "original_rows": len(y),
                "capped_rows": len(selected), "outer_train_index_sha256": base.hashlib.sha256(train.astype("<i8").tobytes()).hexdigest(),
                "outer_test_index_sha256": outer_test_hash, "outer_test_rows": outer_test_rows,
                "contains_outer_test_rows": False, "roles": {}}
        for role, positions in roles.items():
            path = output / "data" / metadata["dataset_name"] / (role + ".pkl")
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"X": frame.iloc[positions].copy(), "y": labels[positions],
                       "outer_train_positions": train[selected[positions]]}
            with path.open("wb") as stream:
                pickle.dump(payload, stream, protocol=5)
            case["roles"][role] = {"rows": len(positions), "sha256": base.digest(path),
                                   "index_sha256": base.hashlib.sha256(payload["outer_train_positions"].astype("<i8").tobytes()).hexdigest()}
        cases.append(case)
        print(json.dumps({"prepared": metadata["dataset_name"], "outer_training_rows": len(y),
                          "assessment_rows": len(roles["development"])}), flush=True)
    dependencies = ["benchmarks/accuracy_remaining_scout.py", "benchmarks/tabarena/ctboost_model.py",
                    "benchmarks/tabarena/learning_options.py"]
    plan = {
        "protocol_id": "ctboost_grouped8_remaining37_20260908_v1", "created_at": base.now(),
        "runner_sha256": base.digest(base.__file__), "orchestrator_sha256": base.digest(__file__),
        "source_sha256": {name: base.digest(ROOT / name) for name in dependencies},
        "ctboost_native_sha256": execution["native_sha256"], "versions": execution["versions"],
        "execution_runtime": execution,
        "preparation_runtime": {"python": sys.executable, "ctboost_native_sha256": base.digest(ctboost._core.__file__),
            "versions": {name: importlib.metadata.version(name) for name in
                         ("ctboost", "autogluon.common", "numpy", "pandas", "scikit-learn", "openml")}},
        "metadata_sha256": base.digest(base.PROTOCOL), "cache_provenance_sha256": base.digest(output / "cache_provenance.json"),
        "cached_tasks": len(provenance), "selection": "All 37 metadata tasks excluded from the original14-task scout; no score-based filtering.",
        "excluded_original_tasks": sorted(excluded), "problem_type_counts": {"binary": 27, "regression": 10},
        "split_policy": "Official outer repeat0/fold0 training rows only. Cap10000 seed47 first, classification stratified. CV8seed47: trainfolds0-4, stopping5, assessment6+7 combined. The worker's development field denotes this assessment role.",
        "source_loading_disclosure": "OpenML reads the full cached source dataset before selecting official training indices. Only official outer-training rows enter prepared roles, fitting, model selection, or scoring; official test outcomes are not scored or used for decisions. Source cache is snapshotted separately and contains the original full datasets by design.",
        "reuse_disclosure": "New task panel after the failed14-task grouped and calibration gates. Those failed decisions remain intact; their reservedfold7 is never read. These37 tasks were not selected using score outcomes. This is a capped single-model development assessment, not official outer-test evaluation, eight-child bagging, or Elo.",
        "arms": list(base.ARMS), "ctboost_params": base.CTBOOST_PARAMS,
        "candidate_override": {"feature_test": "grouped", "feature_test_bins": 8, "feature_test_adjustment": "none"},
        "xgboost": "Unchanged AG XGBoostModel manual default with seed47, same2CPU300s allocation; retain default adaptive patience and tree cap.",
        "resources": {"fit_seconds": 300, "hard_wall_seconds": 390, "rss_bytes": 8 * 1024 ** 3,
                      "threads": 2, "max_workers": 2, "affinities": base.AFFINITIES, "ctboost_early_stop": 50},
        "gate": {"minimum_median_relative_error_gain": .01, "minimum_dataset_wins": 19,
                 "maximum_relative_regression": .10, "maximum_geometric_fit_ratio": 3, "maximum_p90_fit_ratio": 5,
                 "all111_fits_successful_finite": True},
        "retry_policy": "Started fits are terminal, all failures remain visible, no score-based retries or exclusions.",
        "datasets": cases, "expected_fits": 111,
    }
    base.write(output / "plan.json", plan)
    print(json.dumps({"plan_sha256": base.digest(output / "plan.json"), "fits": 111, "ready": True}))


def report(output):
    base = bootstrap()
    import numpy as np
    plan = base.checked_plan(output)
    rows, records, ratios = [], [], []
    for case in plan["datasets"]:
        errors, paired = {}, {}
        assessment = base.load_role(output, case, "development")
        for arm in base.ARMS:
            directory = output / "fits" / case["dataset_name"] / arm
            result = base.read(directory / "result.json")
            if result["plan_sha256"] != base.digest(output / "plan.json"):
                raise ValueError("Result belongs to another plan")
            records.append(result)
            paired[arm] = result
            if result["status"] != "ok" or result.get("resource_failure"):
                continue
            if base.digest(directory / "model.pkl") != result["model_sha256"] or base.digest(directory / "development.npz") != result["development_sha256"]:
                raise ValueError("Changed fitted artifact")
            with np.load(directory / "development.npz") as saved:
                np.testing.assert_array_equal(saved["labels"], assessment["y"])
                np.testing.assert_array_equal(saved["outer_train_positions"], assessment["outer_train_positions"])
                error = base.score_predictions(saved["labels"], saved["predictions"], case)
                if abs(error - result["development_error"]) > 1e-12:
                    raise ValueError("Metric does not match its prediction artifact")
            errors[arm] = error
        row = {"dataset": case["dataset_name"], "problem_type": case["problem_type"], "errors": errors}
        if len(errors) == 3:
            row["candidate_relative_gain"] = (errors[base.ARMS[0]] - errors[base.ARMS[1]]) / max(errors[base.ARMS[0]], 1e-12)
            row["candidate_vs_xgboost_gain"] = (errors[base.ARMS[2]] - errors[base.ARMS[1]]) / max(errors[base.ARMS[2]], 1e-12)
            ratios.append(paired[base.ARMS[1]]["fit_seconds"] / max(paired[base.ARMS[0]]["fit_seconds"], 1e-12))
        rows.append(row)
    gains = [row["candidate_relative_gain"] for row in rows if "candidate_relative_gain" in row]
    complete = len(gains) == 37 and len(records) == 111 and all(r["status"] == "ok" and not r.get("resource_failure") for r in records)
    stats = {"median_relative_gain": float(np.median(gains)) if gains else None,
             "wins": sum(g > 1e-12 for g in gains), "worst_relative_regression": max([0.0] + [-g for g in gains]),
             "geometric_fit_ratio": float(np.exp(np.mean(np.log(ratios)))) if ratios else None,
             "p90_fit_ratio": float(np.quantile(ratios, .9)) if ratios else None}
    gate = plan["gate"]
    passed = complete and stats["median_relative_gain"] >= gate["minimum_median_relative_error_gain"] and stats["wins"] >= gate["minimum_dataset_wins"] and stats["worst_relative_regression"] <= gate["maximum_relative_regression"] and stats["geometric_fit_ratio"] <= gate["maximum_geometric_fit_ratio"] and stats["p90_fit_ratio"] <= gate["maximum_p90_fit_ratio"]
    result = {"created_at": base.now(), "plan_sha256": base.digest(output / "plan.json"), "stage": "new37task_assessment",
              "gate_passed": bool(passed), "complete_success": complete, "statistics": stats, "datasets": rows,
              "failures": [r for r in records if r["status"] != "ok"], "reuse_disclosure": plan["reuse_disclosure"]}
    base.write(output / "assessment_report.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "report"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--runtime", type=Path)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.output.resolve(), args.cache.resolve(), args.runtime.resolve())
    elif args.stage == "run":
        base = bootstrap()
        plan = base.checked_plan(args.output.resolve())
        if str(Path(sys.executable).resolve()) != plan["execution_runtime"]["python"]:
            raise RuntimeError("Launch with the frozen execution Python")
        base.run(args.output.resolve(), 2)
    else:
        report(args.output.resolve())
