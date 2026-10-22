"""One bounded isolated wheel build; no installation or test execution."""
from __future__ import annotations

from datetime import datetime, timezone
from email.parser import Parser
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import psutil

ROOT = Path("C:/apps/ctboost")
RELEASE = Path("C:/Users/captn/.codex/worktrees/release-0-1-62/ctboost")
WORK = Path(__file__).resolve().parent
CANDIDATE = WORK / "candidate-01"
BUILD = WORK / "build-cp312"
OUT = ROOT / "benchmarks/results/improvements_20261016_followup/release_preparation/wheel_candidate_01"
MINIMUM_AVAILABLE = 14 * 1024**3
MAXIMUM_RSS = 2 * 1024**3
MAXIMUM_SECONDS = 1200


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def source_hashes():
    paths = [RELEASE / name for name in ["CMakeLists.txt", "pyproject.toml", "README.md", "LICENSE"]]
    for folder in ["ctboost", "benchmarks", "src", "include", "cuda"]:
        for path in (RELEASE / folder).rglob("*"):
            if (path.is_file() and "__pycache__" not in path.parts
                    and "results" not in path.relative_to(RELEASE).parts
                    and path.suffix.lower() in [".py", ".hpp", ".cpp", ".h", ".cu", ".cuh", ".json", ".md"]):
                paths.append(path)
    return {path.relative_to(RELEASE).as_posix(): sha(path) for path in sorted(set(paths))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    receipt_path = OUT / "receipt.json"
    if receipt_path.exists() or CANDIDATE.exists():
        raise RuntimeError("Candidate identity already exists; preserve it and use a new attempt")
    psutil.Process().cpu_affinity([0])
    available = psutil.virtual_memory().available
    receipt = {"status": "preflight", "started_at": now(),
               "release_checkout": str(RELEASE), "candidate": str(CANDIDATE),
               "persistent_build_directory": str(BUILD),
               "available_before_bytes": available,
               "limits": {"cpu_affinity": [0], "compiler_jobs": 1,
                          "maximum_process_tree_rss_bytes": MAXIMUM_RSS,
                          "maximum_seconds": MAXIMUM_SECONDS,
                          "minimum_available_bytes": MINIMUM_AVAILABLE},
               "predeclared_wall_limit_note": "1200 seconds approved before the one-worker full compile; no post-failure relaxation",
               "python": sys.executable, "python_version": sys.version,
               "pip_version": importlib.metadata.version("pip"),
               "psutil_version": psutil.__version__,
               "runner_sha256": sha(Path(__file__)),
               "source_hashes_before": source_hashes(),
               "purpose": "Pre-final-docs validation wheel; preserve this identity and rebuild final release metadata later",
               "installation_performed": False, "tests_executed": False,
               "no_latency_claim": True}
    write(receipt_path, receipt)
    if available < MINIMUM_AVAILABLE:
        receipt.update(status="deferred_insufficient_available_memory", finished_at=now())
        write(receipt_path, receipt)
        print(json.dumps({"status": receipt["status"], "available_bytes": available}), flush=True)
        return 75
    (CANDIDATE / "wheels").mkdir(parents=True)
    (CANDIDATE / "tmp").mkdir()
    constraints = CANDIDATE / "build-constraints.txt"
    constraints.write_text("pybind11==2.13.6\nscikit-build-core==0.12.2\n", encoding="utf-8")
    environment = os.environ.copy()
    compiler_root = Path("C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.50.35717")
    sdk = Path("C:/Program Files (x86)/Windows Kits/10")
    environment.update({
        "PATH": str(compiler_root / "bin/Hostx64/x64") + os.pathsep
                + str(sdk / "bin/10.0.26100.0/x64") + os.pathsep
                + "C:/Users/captn/miniconda3/Scripts" + os.pathsep + environment.get("PATH", ""),
        "INCLUDE": os.pathsep.join([str(compiler_root / "include")]
                    + [str(sdk / ("Include/10.0.26100.0/" + part)) for part in ["ucrt", "shared", "um", "winrt"]]),
        "LIB": os.pathsep.join([str(compiler_root / "lib/x64"), str(sdk / "Lib/10.0.26100.0/ucrt/x64"),
                                str(sdk / "Lib/10.0.26100.0/um/x64")]),
        "CMAKE_GENERATOR": "Ninja", "CMAKE_BUILD_PARALLEL_LEVEL": "1",
        "TMP": str(CANDIDATE / "tmp"), "TEMP": str(CANDIDATE / "tmp"),
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    })
    for key in ["CFLAGS", "CXXFLAGS", "CL", "_CL_", "CMAKE_ARGS"]:
        environment.pop(key, None)
    command = [sys.executable, "-I", "-m", "pip", "--isolated", "wheel", "--no-deps", "--no-clean",
               "--verbose", "--index-url", "https://pypi.org/simple",
               "--build-constraint", str(constraints), "--wheel-dir", str(CANDIDATE / "wheels"),
               "-Cbuild-dir=" + str(BUILD), "-Ccmake.build-type=Release",
               "-Ccmake.define.CTBOOST_ENABLE_CUDA=OFF", "-Ccmake.define.CTBOOST_BUILD_NATIVE_TESTS=ON",
               "-Cbuild.tool-args=-j1", str(RELEASE)]
    watchdog_path = ROOT / "benchmarks/owned_process_watchdog.py"
    spec = importlib.util.spec_from_file_location("release_wheel_watchdog", watchdog_path)
    watchdog = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(watchdog)
    receipt.update(command=command, status="running", constraints_sha256=sha(constraints),
                   environment_overrides={key: environment[key] for key in
                       ["PATH", "INCLUDE", "LIB", "CMAKE_GENERATOR", "CMAKE_BUILD_PARALLEL_LEVEL", "TMP", "TEMP",
                        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]},
                   watchdog_sha256=sha(watchdog_path),
                   tool_versions={"cmake": "4.3.1", "ninja": "1.13.0", "msvc": "14.50.35717", "windows_sdk": "10.0.26100.0"},
                   tool_hashes={str(path): sha(path) for path in [compiler_root / "bin/Hostx64/x64/cl.exe",
                       Path("C:/Users/captn/miniconda3/Scripts/cmake.exe"), Path("C:/Users/captn/miniconda3/Scripts/ninja.exe")]})
    write(receipt_path, receipt)
    shutil.copyfile(Path(__file__), OUT / "build_candidate.py")
    shutil.copyfile(watchdog_path, OUT / "owned_process_watchdog.py")
    shutil.copyfile(constraints, OUT / "build-constraints.txt")
    log_path = OUT / "build.log"
    with log_path.open("wb") as log:
        process = subprocess.Popen(command, cwd=RELEASE, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        receipt["owned_process_pid"] = process.pid
        receipt["owned_process_create_time"] = psutil.Process(process.pid).create_time()
        write(receipt_path, receipt)
        print(json.dumps({"pid": process.pid, "started_at": receipt["started_at"],
                          "available_before_bytes": available, "maximum_seconds": MAXIMUM_SECONDS}), flush=True)
        result = watchdog.watch(process, maximum_seconds=MAXIMUM_SECONDS, maximum_rss_bytes=MAXIMUM_RSS)
    receipt.update(result, finished_at=now(), source_hashes_after=source_hashes(),
                   log_sha256=sha(log_path), available_after_bytes=psutil.virtual_memory().available)
    dependencies = []
    for metadata_path in sorted((CANDIDATE / "tmp").rglob("*.dist-info/METADATA")):
        metadata = Parser().parsestr(metadata_path.read_text(encoding="utf-8"))
        dependencies.append({"name": metadata["Name"], "version": metadata["Version"],
                             "metadata_path": str(metadata_path), "metadata_sha256": sha(metadata_path)})
    receipt["isolated_build_dependencies"] = dependencies
    receipt["status"] = "passed" if (result["exit_code"] == 0 and not result["resource_failure"]
        and result["cleanup_complete"] and receipt["source_hashes_before"] == receipt["source_hashes_after"]) else "failed"
    wheels = sorted((CANDIDATE / "wheels").glob("*.whl"))
    receipt["wheels"] = [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)} for path in wheels]
    if receipt["status"] == "passed":
        if len(wheels) != 1 or not wheels[0].name.startswith("ctboost-0.1.62-cp312-cp312-win_amd64"):
            receipt["status"] = "failed_wheel_identity"
        else:
            with zipfile.ZipFile(wheels[0]) as archive:
                names = archive.namelist()
                receipt["wheel_members"] = names
                receipt["wheel_native"] = [{"path": name, "sha256": hashlib.sha256(archive.read(name)).hexdigest()}
                                            for name in names if name.endswith(".pyd")]
                receipt["categorical_helper_present"] = "ctboost/_categorical_codes.py" in names
                receipt["excluded_research_archives_absent"] = not any(name.startswith("benchmarks/results/improvements_20261016") for name in names)
            if not receipt["categorical_helper_present"] or not receipt["excluded_research_archives_absent"]:
                receipt["status"] = "failed_wheel_membership"
    for name in ["CMakeCache.txt", "build.ninja"]:
        path = BUILD / name
        if path.is_file():
            shutil.copyfile(path, OUT / (name + ".txt"))
    write(receipt_path, receipt)
    print(json.dumps({key: receipt.get(key) for key in ["status", "exit_code", "resource_failure",
                      "peak_rss_bytes", "wall_seconds", "cleanup_complete", "wheels"]}), flush=True)
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
