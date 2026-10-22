"""Bounded final installed-wheel checks after reviewed test-only repairs."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import psutil

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("validation", HERE / "validate_installed.py")
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)
OUT = v.OUT / "final-sequence"
FIXTURES = HERE / "source-fixtures-final"
PANDAS = v.ROOT / ".tmp/improvements-20261016/release-pandas306"
PATCH = v.ROOT / "benchmarks/results/improvements_20261016_followup/release_preparation/pandas3_fixture_review/application.json"
SELECTED_PANDAS = ["tests/test_native_categorical_codes.py", "tests/test_feature_pipeline_categorical_row.py",
                   "tests/test_categorical_codes_prototype.py"]


def copy_fixtures():
    if FIXTURES.exists():
        assert (OUT / "fixture-source-manifest.json").exists()
        return
    names = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                                    cwd=v.RELEASE, text=True).splitlines()
    copied = []
    for name in sorted(set(names)):
        if name.startswith("ctboost/") or name == "skills.md":
            continue
        source = v.RELEASE / name
        if not source.is_file():
            continue
        destination = FIXTURES / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        copied.append({"path": name, "sha256": v.digest(source)})
    v.write(OUT / "fixture-source-manifest.json", {"source": str(v.RELEASE), "destination": str(FIXTURES),
        "files": copied, "excluded_package": "ctboost/", "pandas_fixture_application_sha256": v.digest(PATCH)})


def pandas_worker():
    sys.path.insert(0, str(PANDAS))
    import pandas
    import ctboost
    import pytest
    assert pandas.__version__ == "3.0.6"
    assert PANDAS.resolve() in Path(pandas.__file__).resolve().parents
    assert v.ENV.resolve() in Path(ctboost.__file__).resolve().parents
    installed = json.loads((v.OUT / "installed-runtime.json").read_text())
    assert v.digest(Path(ctboost._core.__file__)) == installed["native"]["ctboost"]["sha256"]
    v.write(OUT / "pandas3-runtime.json", {"python": sys.version, "executable": sys.executable,
        "pandas_version": pandas.__version__, "pandas_file": pandas.__file__, "ctboost_file": ctboost.__file__,
        "native_sha256": v.digest(Path(ctboost._core.__file__)), "native_build": ctboost.build_info(),
        "numpy_version": __import__("numpy").__version__, "target_install_only": True})
    return pytest.main(SELECTED_PANDAS + ["-q", "--junitxml=" + sys.argv[-1]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["full-pytest", "scientific-pytest", "pandas3-install", "pandas3-pytest"])
    parser.add_argument("--attempt", required=False, default="01")
    parser.add_argument("--pandas-worker", action="store_true")
    parser.add_argument("--junit")
    args = parser.parse_args()
    if args.pandas_worker:
        return pandas_worker()
    if not args.stage or not re.fullmatch(r"[0-9]{2}", args.attempt):
        parser.error("stage and two-digit attempt required")
    OUT.mkdir(parents=True, exist_ok=True)
    stem = args.stage + "-" + args.attempt
    receipt_path = OUT / (stem + ".json")
    assert not receipt_path.exists(), "Preserve prior attempts"
    assert json.loads((v.OUT / "setup.json").read_text())["status"] == "passed"
    assert json.loads((v.OUT / "native-attempt-02/receipt.json").read_text())["status"] == "passed"
    affinity = [0, 2, 8, 10] if args.stage == "full-pytest" else [0, 2] if args.stage.endswith("pytest") else [0]
    psutil.Process().cpu_affinity(affinity)
    available = psutil.virtual_memory().available
    receipt = {"stage": args.stage, "attempt": args.attempt, "status": "preflight", "started_at": v.now(),
        "available_before_bytes": available, "limits": {"minimum_available_bytes": v.MIN_AVAILABLE,
            "maximum_process_tree_rss_bytes": v.MAX_RSS, "maximum_seconds": v.MAX_SECONDS,
            "cpu_affinity": affinity, "native_threads": 1},
        "four_cpu_reason": "Existing two-worker controller tests require two logical CPUs per worker.",
        "runner_sha256": v.digest(Path(__file__)), "environment_helper_sha256": v.digest(HERE / "validate_installed.py"),
        "wheel_sha256": v.digest(v.WHEEL), "installed_runtime_sha256": v.digest(v.OUT / "installed-runtime.json"),
        "test_only_patch_sha256": v.digest(PATCH), "original_environments_modified": False, "no_latency_claim": True}
    v.write(receipt_path, receipt)
    shutil.copyfile(Path(__file__), OUT / (stem + "-runner.py"))
    if available < v.MIN_AVAILABLE:
        receipt.update(status="deferred_insufficient_available_memory", finished_at=v.now())
        v.write(receipt_path, receipt)
        print(json.dumps({"stage": args.stage, "status": receipt["status"], "available": available}), flush=True)
        return 75
    junit = OUT / (stem + ".xml")
    if args.stage == "pandas3-install":
        assert not PANDAS.exists(), "Expected a new target, without changing the pinned environment"
        command = [str(v.PYTHON), "-I", "-m", "pip", "--isolated", "install", "--no-deps", "--only-binary=:all:",
                   "--index-url", "https://pypi.org/simple", "--cache-dir", str(HERE / "pip-cache"),
                   "--target", str(PANDAS), "pandas==3.0.6"]
        cwd = HERE
    else:
        copy_fixtures()
        receipt["fixture_manifest_sha256"] = v.digest(OUT / "fixture-source-manifest.json")
        cwd = FIXTURES
        if args.stage == "pandas3-pytest":
            assert (PANDAS / "pandas").is_dir()
            command = [str(v.PYTHON), "-I", "-X", "faulthandler", str(Path(__file__)), "--pandas-worker", "--junit", str(junit)]
            receipt["selected_paths"] = SELECTED_PANDAS
        else:
            selected = ["tests"]
            if args.stage == "scientific-pytest":
                workflow = (FIXTURES / ".github/workflows/scientific-tests.yml").read_text(encoding="utf-8")
                block = workflow.split("$scientificTests = @(", 1)[1].split("\n            )", 1)[0]
                selected = re.findall(r'"((?:tests|benchmarks)/[^"\n]+\.py)"', block)
                assert len(selected) == 23 and len(set(selected)) == 23
            command = [str(v.PYTHON), "-I", "-X", "faulthandler", "-m", "pytest"] + selected + ["-q", "--junitxml=" + str(junit)]
            receipt["selected_paths"] = selected
    watchdog_path = v.ROOT / ".tmp/improvements-20261016/release-wheel-cp312/release_build_watchdog.py"
    spec = importlib.util.spec_from_file_location("watchdog", watchdog_path)
    watchdog = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(watchdog)
    receipt.update(status="running", command=command, cwd=str(cwd), watchdog_sha256=v.digest(watchdog_path))
    log_path = OUT / (stem + ".log")
    with log_path.open("wb") as log:
        process = subprocess.Popen(command, cwd=cwd, env=v.environment(), stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        receipt.update(owned_process_pid=process.pid, owned_process_create_time=psutil.Process(process.pid).create_time())
        v.write(receipt_path, receipt)
        print(json.dumps({"stage": args.stage, "pid": process.pid, "started_at": receipt["started_at"],
                          "available_before_bytes": available, "affinity": affinity}), flush=True)
        result = watchdog.watch(process, maximum_seconds=v.MAX_SECONDS, maximum_rss_bytes=v.MAX_RSS)
    receipt.update(result, finished_at=v.now(), log_sha256=v.digest(log_path))
    receipt["status"] = "passed" if result["exit_code"] == 0 and not result["resource_failure"] and result["cleanup_complete"] else "failed"
    if junit.exists():
        suites = list(ET.parse(junit).getroot().iter("testsuite"))
        receipt["counts"] = {key: sum(int(suite.attrib.get(key, 0)) for suite in suites) for key in ["tests", "failures", "errors", "skipped"]}
        receipt["junit_sha256"] = v.digest(junit)
        if args.stage != "full-pytest" and receipt["counts"]["skipped"]:
            receipt["status"] = "failed_required_coverage"
    v.write(receipt_path, receipt)
    print(json.dumps({key: receipt.get(key) for key in ["stage", "status", "exit_code", "resource_failure", "peak_rss_bytes", "wall_seconds", "cleanup_complete", "counts"]}), flush=True)
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
