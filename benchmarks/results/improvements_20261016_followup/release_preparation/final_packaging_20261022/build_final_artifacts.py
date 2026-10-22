"""Repackage final release sources using the validated, unchanged native cache.

Execution requires the parent to freeze release documentation and explicitly pass
--execute. Every attempt has a fresh identity. The backend build remains a normal
incremental build; a process-local pre-build guard rejects pending native work
after CMake configuration, before the original Builder.build method is called.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from email.parser import BytesParser
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import runpy
import shutil
import subprocess
import sys
import tarfile
import tomllib
import zipfile

ROOT = Path("C:/apps/ctboost")
RELEASE = Path("C:/Users/captn/.codex/worktrees/release-0-1-62/ctboost")
PREP = ROOT / "benchmarks/results/improvements_20261016_followup/release_preparation"
BUILD_WORK = ROOT / ".tmp/improvements-20261016/release-wheel-cp312"
BUILD = BUILD_WORK / "build-cp312-02"
BASE_RECEIPT = PREP / "wheel_candidate_02/receipt.json"
BASE_RECEIPT_SHA = "f59e5194bca1c31bd8534a1f6c62af41e31291f7990edb48ffd28c423f5e4d7d"
PYTHON = ROOT / ".tmp/inference-accuracy-20260908/baseline-env/Scripts/python.exe"
ISOLATION = BUILD_WORK / "candidate-02/tmp/pip-build-env-x8ux36ju"
NINJA = Path("C:/Users/captn/miniconda3/Scripts/ninja.exe")
WATCHDOG = BUILD_WORK / "release_build_watchdog.py"
VERSION = "0.1.62"
NATIVE = "ctboost/_core.cp312-win_amd64.pyd"
EXTENSIONS = {".py", ".hpp", ".cpp", ".h", ".cu", ".cuh", ".json", ".md"}
POLICY_FIXTURES = {
    "benchmarks/fair_accuracy_current_round_cv_policy.json",
    "benchmarks/results/improvements_20261016_followup/robustness_protocol_multiclass.json",
    "benchmarks/results/improvements_20261016_followup/robustness_protocol_scalar_refresh.json",
    "benchmarks/results/improvements_20261016_followup/robustness_protocol_scalar_numerical.json",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def now():
    return datetime.now(timezone.utc).isoformat()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def native_sources():
    paths = {RELEASE / "CMakeLists.txt"}
    for folder in ["ctboost", "src", "include", "cuda"]:
        paths.update(p for p in (RELEASE / folder).rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts and p.suffix.lower() in EXTENSIONS)
    return {p.relative_to(RELEASE).as_posix(): digest(p) for p in sorted(paths)}


def expected_sdist():
    project = tomllib.loads((RELEASE / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["version"] == VERSION
    settings = project["tool"]["scikit-build"]
    assert settings["sdist"]["exclude"] == ["*"]
    assert not settings["sdist"].get("cmake", False)
    assert not settings.get("generate")
    paths = set()
    for pattern in settings["sdist"]["include"]:
        assert pattern.startswith("/") and ".." not in PurePosixPath(pattern).parts
        paths.update(p for p in RELEASE.glob(pattern[1:]) if p.is_file())
    assert all(not p.is_symlink() and RELEASE in p.resolve().parents for p in paths)
    result = {p.relative_to(RELEASE).as_posix(): digest(p) for p in sorted(paths)}
    assert POLICY_FIXTURES <= set(result)
    assert "docs/release-0.1.62.md" in result and "ctboost/_categorical_codes.py" in result
    for name in result:
        assert not name.endswith((".npz", ".npy", ".ctb", ".pyd", ".dll", ".log", ".pem", ".key")), name
        assert name != "skills.md" and ".tmp" not in PurePosixPath(name).parts, name
        if name.startswith("benchmarks/results/improvements_20261016"):
            assert name in POLICY_FIXTURES, name
    return result


def dry_run(path):
    result = subprocess.run([str(NINJA), "-C", str(BUILD), "-n"],
                            capture_output=True, text=True, timeout=30, check=False)
    write(path, {"command": result.args, "exit_code": result.returncode,
                 "stdout": result.stdout, "stderr": result.stderr, "at": now()})
    lines = result.stdout.strip().splitlines()
    assert result.returncode == 0 and not result.stderr.strip(), result
    assert lines and lines[-1] == "ninja: no work to do.", "Unexpected native build scheduled; preserve and review"
    assert len(lines) <= 2, "Unexpected dry-run commands; preserve and review"


def check_metadata(data):
    metadata = BytesParser().parsebytes(data)
    assert metadata["Name"] == "ctboost" and metadata["Version"] == VERSION
    body = metadata.get_payload().replace("\r\n", "\n").rstrip("\n")
    expected = (RELEASE / "README.md").read_text(encoding="utf-8").rstrip("\n")
    assert body == expected, "Artifact metadata must contain the final README"


def worker(attempt):
    out = PREP / "final_packaging_20261022" / ("attempt-" + attempt)
    work = ROOT / ".tmp/improvements-20261016/release-final-artifacts-20261022" / ("attempt-" + attempt)
    before = json.loads((out / "sources_before.json").read_text())
    assert native_sources() == before["production"]
    assert expected_sdist() == before["sdist"]
    baseline = json.loads(BASE_RECEIPT.read_text())
    assert digest(BASE_RECEIPT) == BASE_RECEIPT_SHA
    baseline_wheel = Path(baseline["wheels"][0]["path"])
    assert digest(baseline_wheel) == baseline["wheels"][0]["sha256"]
    native_sha = baseline["wheel_native"][0]["sha256"]
    assert digest(BUILD / Path(NATIVE).name) == native_sha

    # Reuse the exact retained isolated backend/pybind11 paths used by candidate02.
    # This does not install or mutate any package in the original environments.
    runpy.run_path(str(ISOLATION / "site/sitecustomize.py"))
    import scikit_build_core
    import pybind11
    from scikit_build_core import build as backend
    from scikit_build_core.builder.builder import Builder
    assert scikit_build_core.__version__ == "0.12.2" and pybind11.__version__ == "2.13.6"
    assert ISOLATION in Path(scikit_build_core.__file__).resolve().parents
    original_build = Builder.build
    calls = []

    def guarded_build(self, build_args):
        assert Path(self.config.build_dir).resolve() == BUILD.resolve()
        assert not calls
        dry_run(out / "after_configure_ninja_dry_run.json")
        calls.append("normal_incremental_build")
        return original_build(self, build_args)

    Builder.build = guarded_build
    config = {"build-dir": str(BUILD), "cmake.build-type": "Release",
              "cmake.define.CTBOOST_ENABLE_CUDA": "OFF",
              "cmake.define.CTBOOST_BUILD_NATIVE_TESTS": "ON", "build.tool-args": "-j1"}
    wheel = work / "dist" / backend.build_wheel(str(work / "dist"), config)
    assert calls == ["normal_incremental_build"]
    assert digest(BUILD / Path(NATIVE).name) == native_sha
    sdist = work / "dist" / backend.build_sdist(str(work / "dist"))
    assert native_sources() == before["production"] and expected_sdist() == before["sdist"]

    wheel_members = {}
    with zipfile.ZipFile(baseline_wheel) as previous, zipfile.ZipFile(wheel) as current:
        names = current.namelist()
        assert current.testzip() is None and len(names) == len(set(names))
        assert set(names) == set(previous.namelist()), "Review changed wheel membership"
        for name in names:
            payload = current.read(name)
            wheel_members[name] = hashlib.sha256(payload).hexdigest()
            if name.startswith("ctboost/"):
                assert payload == previous.read(name), "Tested runtime payload changed: " + name
                if name != NATIVE:
                    assert payload == (RELEASE / name).read_bytes(), name
            elif name.startswith("benchmarks/"):
                assert payload == (RELEASE / name).read_bytes(), name
            assert not name.startswith("benchmarks/results/improvements_20261016"), name
        check_metadata(current.read("ctboost-0.1.62.dist-info/METADATA"))
        assert wheel_members[NATIVE] == native_sha
    sdist_members = {}
    with tarfile.open(sdist, "r:gz") as archive:
        members = archive.getmembers()
        assert all(p.isfile() for p in members), "Unexpected non-file source member"
        names = [str(PurePosixPath(p.name).relative_to("ctboost-0.1.62")) for p in members]
        assert len(names) == len(set(names))
        assert set(names) == set(before["sdist"]) | {"PKG-INFO"}, "Source allowlist membership mismatch"
        for member, name in zip(members, names):
            payload = archive.extractfile(member).read()
            sdist_members[name] = hashlib.sha256(payload).hexdigest()
            if name == "PKG-INFO":
                check_metadata(payload)
            else:
                assert sdist_members[name] == before["sdist"][name] and payload == (RELEASE / name).read_bytes(), name
    sys.path.insert(0, str(RELEASE / "scripts"))
    validator = load_module("final_release_artifact_validator", RELEASE / "scripts/validate_release_artifacts.py")
    assert not validator._sdist_errors(sdist, expected_version=VERSION)
    assert not validator._wheel_errors(wheel, validator.parse_wheel_identity(wheel.name),
                                      expected_version=VERSION, expected_cuda=False,
                                      expected_license_sha256=validator.CUDA_RUNTIME_LICENSE_SHA256)
    write(out / "worker_result.json", {"status": "passed", "finished_at": now(),
          "backend": scikit_build_core.__version__, "pybind11": pybind11.__version__,
          "native_sha256": native_sha, "all_ctboost_payloads_equal_tested_candidate": True,
          "final_readme_metadata_verified": True, "native_build_calls": calls,
          "wheel_members_sha256": wheel_members, "sdist_members_sha256": sdist_members,
          "artifacts": [{"path": str(p), "bytes": p.stat().st_size, "sha256": digest(p)} for p in [wheel, sdist]]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    assert re.fullmatch(r"[0-9]{2}", args.attempt)
    if args.worker:
        worker(args.attempt)
        return 0
    if not args.execute:
        parser.error("Wait for root to freeze final docs; execution requires --execute")
    import psutil
    out = PREP / "final_packaging_20261022" / ("attempt-" + args.attempt)
    work = ROOT / ".tmp/improvements-20261016/release-final-artifacts-20261022" / ("attempt-" + args.attempt)
    assert not out.exists() and not work.exists(), "Preserve previous attempts"
    assert Path(sys.executable).resolve() == PYTHON.resolve()
    assert digest(BASE_RECEIPT) == BASE_RECEIPT_SHA
    base = json.loads(BASE_RECEIPT.read_text())
    expected_prod = {p: x for p, x in base["source_hashes_after"].items()
                     if p.startswith(("ctboost/", "src/", "include/", "cuda/")) or p == "CMakeLists.txt"}
    assert native_sources() == expected_prod, "Production sources changed; do not reuse validated cache"
    out.mkdir(parents=True)
    for part in ["dist", "tmp"]:
        (work / part).mkdir(parents=True, exist_ok=True)
    psutil.Process().cpu_affinity([0])
    available = psutil.virtual_memory().available
    receipt = {"status": "preflight", "started_at": now(), "available_before_bytes": available,
               "runner_sha256": digest(Path(__file__)), "baseline_receipt_sha256": BASE_RECEIPT_SHA,
               "release_checkout": str(RELEASE), "limits": {"minimum_available_bytes": 14 * 1024**3,
                    "maximum_process_tree_rss_bytes": 2 * 1024**3, "maximum_seconds": 1200,
                    "cpu_affinity": [0], "compiler_jobs": 1},
               "purpose": "Final metadata and source artifact validation; no new native compilation expected",
               "no_latency_claim": True, "original_environments_modified": False}
    write(out / "receipt.json", receipt)
    if available < 14 * 1024**3:
        receipt.update(status="deferred_insufficient_available_memory", finished_at=now())
        write(out / "receipt.json", receipt)
        print(json.dumps(receipt), flush=True)
        return 75
    write(out / "sources_before.json", {"production": expected_prod, "sdist": expected_sdist()})
    dry_run(out / "before_configure_ninja_dry_run.json")
    environment = os.environ.copy()
    for key in ["PYTHONPATH", "PYTHONHOME", "CFLAGS", "CXXFLAGS", "CL", "_CL_", "CMAKE_ARGS"]:
        environment.pop(key, None)
    environment.update(base["environment_overrides"])
    environment.update(TMP=str(work / "tmp"), TEMP=str(work / "tmp"), CTBOOST_HIST_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    command = [str(PYTHON), "-I", str(Path(__file__).resolve()), "--attempt", args.attempt, "--worker"]
    watchdog = load_module("final_artifact_watchdog", WATCHDOG)
    receipt.update(status="running", command=command, cwd=str(RELEASE), watchdog_sha256=digest(WATCHDOG),
                   source_manifest_sha256=digest(out / "sources_before.json"))
    write(out / "receipt.json", receipt)
    with (out / "build.log").open("wb") as log:
        process = subprocess.Popen(command, cwd=RELEASE, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        receipt.update(owned_process_pid=process.pid, owned_process_create_time=psutil.Process(process.pid).create_time())
        write(out / "receipt.json", receipt)
        print(json.dumps({"status": "running", "pid": process.pid, "started_at": receipt["started_at"]}), flush=True)
        result = watchdog.watch(process, maximum_seconds=1200, maximum_rss_bytes=2 * 1024**3)
    receipt.update(result, finished_at=now(), log_sha256=digest(out / "build.log"))
    success = result["exit_code"] == 0 and not result["resource_failure"] and result["cleanup_complete"]
    receipt["status"] = "passed" if success else "failed_preserved_for_review"
    if success:
        receipt["worker_result_sha256"] = digest(out / "worker_result.json")
        receipt["artifacts"] = json.loads((out / "worker_result.json").read_text())["artifacts"]
    write(out / "receipt.json", receipt)
    print(json.dumps({k: receipt.get(k) for k in ["status", "exit_code", "resource_failure", "wall_seconds",
                         "peak_rss_bytes", "cleanup_complete", "artifacts"]}), flush=True)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
