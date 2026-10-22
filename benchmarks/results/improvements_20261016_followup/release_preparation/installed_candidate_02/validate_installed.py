"""Bounded validation of the candidate wheel in a new CP312 environment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
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

ROOT = Path("C:/apps/ctboost")
RELEASE = Path("C:/Users/captn/.codex/worktrees/release-0-1-62/ctboost")
WORK = Path(__file__).resolve().parent
ENV = WORK / "env"
PYTHON = ENV / "Scripts/python.exe"
FIXTURES = WORK / "source-fixtures"
OUT = ROOT / "benchmarks/results/improvements_20261016_followup/release_preparation/installed_candidate_02"
BUILD = ROOT / ".tmp/improvements-20261016/release-wheel-cp312/build-cp312-02"
BUILD_RECEIPT = ROOT / "benchmarks/results/improvements_20261016_followup/release_preparation/wheel_candidate_02/receipt.json"
WHEEL = ROOT / ".tmp/improvements-20261016/release-wheel-cp312/candidate-02/wheels/ctboost-0.1.62-cp312-cp312-win_amd64.whl"
PINS = {"numpy": "2.0.2", "pandas": "2.3.3", "scikit-learn": "1.6.1", "scipy": "1.16.3",
        "xgboost": "3.4.1", "lightgbm": "4.7.0", "psutil": "7.2.2", "pytest": "9.1.1",
        "jsonschema": "4.26.0", "pybind11": "2.13.6"}
MIN_AVAILABLE = 14 * 1024**3
MAX_RSS = 2 * 1024**3
MAX_SECONDS = 600


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def now():
    return datetime.now(timezone.utc).isoformat()


def environment():
    result = os.environ.copy()
    for name in ["PYTHONPATH", "PYTHONHOME", "CFLAGS", "CXXFLAGS", "CL", "_CL_", "CMAKE_ARGS"]:
        result.pop(name, None)
    for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "CTBOOST_HIST_THREADS"]:
        result[name] = "1"
    compiler = Path("C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.50.35717")
    sdk = Path("C:/Program Files (x86)/Windows Kits/10")
    result["PATH"] = os.pathsep.join([str(ENV / "Scripts"),
        "C:/Users/captn/miniconda3/Lib/site-packages/cmake/data/bin", str(compiler / "bin/Hostx64/x64"),
        str(sdk / "bin/10.0.26100.0/x64"), result.get("PATH", "")])
    result["INCLUDE"] = os.pathsep.join([str(compiler / "include")]
        + [str(sdk / ("Include/10.0.26100.0/" + name)) for name in ["ucrt", "shared", "um", "winrt"]])
    result["LIB"] = os.pathsep.join([str(compiler / "lib/x64"), str(sdk / "Lib/10.0.26100.0/ucrt/x64"),
                                    str(sdk / "Lib/10.0.26100.0/um/x64")])
    result["CMAKE_BUILD_PARALLEL_LEVEL"] = "1"
    result["PYTHONDONTWRITEBYTECODE"] = "1"
    return result


def runtime():
    import importlib.metadata as metadata
    import ctboost
    import xgboost
    import lightgbm
    actual = {name: metadata.version(name) for name in PINS}
    assert actual == PINS, (actual, PINS)
    assert sys.version_info[:2] == (3, 12)
    package = Path(ctboost.__file__).resolve()
    assert ENV.resolve() in package.parents, package
    native = Path(ctboost._core.__file__).resolve()
    build = json.loads(BUILD_RECEIPT.read_text(encoding="utf-8"))
    assert digest(native) == build["wheel_native"][0]["sha256"]
    assert ctboost.__version__ == "0.1.62", ctboost.__version__
    libraries = {"ctboost": native, "xgboost": Path(xgboost.core._LIB._name),
                 "lightgbm": Path(lightgbm.basic._LIB._name)}
    return {"python": sys.version, "executable": sys.executable, "versions": actual,
            "package": str(package), "build_info": ctboost.build_info(),
            "native": {name: {"path": str(path), "sha256": digest(path)} for name, path in libraries.items()},
            "affinity": psutil.Process().cpu_affinity()}


def setup():
    if ENV.exists():
        raise RuntimeError("Expected a new validation venv")
    build = json.loads(BUILD_RECEIPT.read_text(encoding="utf-8"))
    assert build["status"] == "passed" and digest(WHEEL) == build["wheels"][0]["sha256"]
    commands = [[sys.executable, "-I", "-m", "venv", str(ENV)],
                [str(PYTHON), "-I", "-m", "pip", "--isolated", "install", "--only-binary=:all:",
                 "--index-url", "https://pypi.org/simple", "--cache-dir", str(WORK / "pip-cache")]
                + [f"{name}=={version}" for name, version in PINS.items()],
                [str(PYTHON), "-I", "-m", "pip", "--isolated", "install", "--no-deps", str(WHEEL)],
                [str(PYTHON), "-I", str(Path(__file__)), "--worker", "runtime"]]
    write(OUT / "setup-commands.json", commands)
    for command in commands:
        print(json.dumps({"command": command}), flush=True)
        subprocess.run(command, env=environment(), cwd=WORK, check=True)


def copy_fixtures():
    if FIXTURES.exists():
        raise RuntimeError("Expected a new fixture tree")
    names = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                                    cwd=RELEASE, text=True).splitlines()
    copied = []
    for name in sorted(set(names)):
        # The package is deliberately absent: all Python processes must use the
        # installed wheel. Git metadata, build output and local data are absent.
        if name.startswith("ctboost/") or name == "skills.md":
            continue
        source = RELEASE / name
        if not source.is_file():
            continue
        destination = FIXTURES / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        copied.append({"path": name, "sha256": digest(source)})
    write(OUT / "fixture-source-manifest.json", {"source": str(RELEASE), "destination": str(FIXTURES),
          "excluded_package": "ctboost/", "files": copied})


def scientific_files():
    workflow = (FIXTURES / ".github/workflows/scientific-tests.yml").read_text(encoding="utf-8")
    block = workflow.split("$scientificTests = @(", 1)[1].split("\n            )", 1)[0]
    files = re.findall(r'"((?:tests|benchmarks)/[^"\n]+\.py)"', block)
    assert len(files) == 23 and len(set(files)) == 23
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["setup", "native", "full-pytest", "scientific-pytest"])
    parser.add_argument("--worker", choices=["setup", "runtime"])
    args = parser.parse_args()
    if args.worker == "setup":
        setup()
        return 0
    if args.worker == "runtime":
        write(OUT / "installed-runtime.json", runtime())
        print(json.dumps(runtime()), flush=True)
        return 0
    if not args.stage:
        parser.error("--stage or --worker required")
    OUT.mkdir(parents=True, exist_ok=True)
    receipt_path = OUT / (args.stage + ".json")
    if receipt_path.exists():
        raise RuntimeError("Stage already recorded; retain its identity")
    affinity = [0, 2] if args.stage.endswith("pytest") else [0]
    psutil.Process().cpu_affinity(affinity)
    available = psutil.virtual_memory().available
    receipt = {"stage": args.stage, "status": "preflight", "started_at": now(),
               "available_before_bytes": available, "limits": {"minimum_available_bytes": MIN_AVAILABLE,
                    "maximum_process_tree_rss_bytes": MAX_RSS, "maximum_seconds": MAX_SECONDS,
                    "cpu_affinity": affinity, "native_threads": 1},
               "affinity_note": "Two slots are authorized only for pytest controller correctness; this is not a performance measurement.",
               "runner_sha256": digest(Path(__file__)), "wheel_sha256": digest(WHEEL),
               "build_receipt_sha256": digest(BUILD_RECEIPT), "pins": PINS,
               "original_environments_modified": False, "no_latency_claim": True}
    write(receipt_path, receipt)
    if available < MIN_AVAILABLE:
        receipt.update(status="deferred_insufficient_available_memory", finished_at=now())
        write(receipt_path, receipt)
        print(json.dumps({"stage": args.stage, "status": receipt["status"], "available_bytes": available}), flush=True)
        return 75
    env = environment()
    if args.stage == "setup":
        command = [sys.executable, "-I", str(Path(__file__)), "--worker", "setup"]
        cwd = WORK
    elif args.stage == "native":
        command = ["C:/Users/captn/miniconda3/Lib/site-packages/cmake/data/bin/ctest.exe",
                   "--test-dir", str(BUILD), "-C", "Release", "--output-on-failure", "-V"]
        env["PYTHONHOME"] = sys.base_prefix
        env["PATH"] = sys.base_prefix + os.pathsep + env["PATH"]
        cwd = WORK
        receipt["python_home_for_embedding"] = sys.base_prefix
    else:
        assert json.loads((OUT / "setup.json").read_text())["status"] == "passed"
        if not FIXTURES.exists():
            copy_fixtures()
        receipt["fixture_manifest_sha256"] = digest(OUT / "fixture-source-manifest.json")
        selected = ["tests"] if args.stage == "full-pytest" else scientific_files()
        receipt["selected_paths"] = selected
        command = [str(PYTHON), "-I", "-X", "faulthandler", "-m", "pytest"] + selected + ["-q", "--junitxml=" + str(OUT / (args.stage + ".xml"))]
        cwd = FIXTURES
    watchdog_path = ROOT / ".tmp/improvements-20261016/release-wheel-cp312/release_build_watchdog.py"
    spec = importlib.util.spec_from_file_location("release_validation_watchdog", watchdog_path)
    watchdog = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(watchdog)
    receipt.update(status="running", command=command, cwd=str(cwd),
                   watchdog_sha256=digest(watchdog_path),
                   watchdog_note="Only this launcher and observed descendants are owned; after completion only observed MSVC PDB/telemetry daemons may be closed.")
    write(receipt_path, receipt)
    shutil.copyfile(Path(__file__), OUT / "validate_installed.py")
    shutil.copyfile(watchdog_path, OUT / "release_validation_watchdog.py")
    log_path = OUT / (args.stage + ".log")
    with log_path.open("wb") as log:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        receipt.update(owned_process_pid=process.pid, owned_process_create_time=psutil.Process(process.pid).create_time())
        write(receipt_path, receipt)
        print(json.dumps({"stage": args.stage, "pid": process.pid, "started_at": receipt["started_at"],
                          "available_before_bytes": available, "affinity": affinity}), flush=True)
        result = watchdog.watch(process, maximum_seconds=MAX_SECONDS, maximum_rss_bytes=MAX_RSS)
    receipt.update(result, finished_at=now(), log_sha256=digest(log_path))
    receipt["status"] = "passed" if result["exit_code"] == 0 and not result["resource_failure"] and result["cleanup_complete"] else "failed"
    if args.stage == "native":
        match = re.search(r"checks=(\d+) failures=(\d+)", log_path.read_text(encoding="utf-8", errors="replace"))
        receipt["native_checks"] = {"checks": int(match[1]), "failures": int(match[2])} if match else None
        if not match or int(match[1]) != 230 or int(match[2]) != 0:
            receipt["status"] = "failed_native_count"
    if args.stage.endswith("pytest"):
        xml = OUT / (args.stage + ".xml")
        if xml.is_file():
            suites = list(ET.parse(xml).getroot().iter("testsuite"))
            counts = {key: sum(int(suite.attrib.get(key, 0)) for suite in suites) for key in ["tests", "failures", "errors", "skipped"]}
            receipt.update(counts=counts, junit_sha256=digest(xml))
            if args.stage == "scientific-pytest" and counts["skipped"]:
                receipt["status"] = "failed_scientific_coverage"
    if args.stage == "setup" and receipt["status"] == "passed":
        receipt["installed_runtime_sha256"] = digest(OUT / "installed-runtime.json")
    write(receipt_path, receipt)
    print(json.dumps({key: receipt.get(key) for key in ["stage", "status", "exit_code", "resource_failure",
                     "peak_rss_bytes", "wall_seconds", "cleanup_complete", "native_checks", "counts"]}), flush=True)
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
