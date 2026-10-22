"""Bounded, isolated release-worktree JVM validation; one stage per invocation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import psutil

ROOT = Path("C:/apps/ctboost")
RELEASE = Path("C:/Users/captn/.codex/worktrees/release-0-1-62/ctboost")
TOOLS = ROOT / ".tmp/improvements-20261016/release-jvm-tools"
JAVA_HOME = TOOLS / "Temurin17/jdk-17.0.20.1+1"
MAVEN = TOOLS / "Apache-Maven/apache-maven-3.9.16/bin/mvn.cmd"
OUT = ROOT / "benchmarks/results/improvements_20261016_followup/release_preparation/security/jvm_validation_02"
WORK = Path(__file__).resolve().parent
MINIMUM_AVAILABLE = 14 * 1024**3
MAXIMUM_RSS = 2 * 1024**3
MAXIMUM_SECONDS = 600
JAVA_OPTIONS = "-Xms32m -Xmx256m -XX:ActiveProcessorCount=1 -XX:+UseSerialGC"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sources():
    files = [RELEASE / "bindings/jvm/pom.xml"]
    files += sorted((RELEASE / "bindings/jvm/src").rglob("*.java"))
    files += sorted((RELEASE / "tests/export_conformance").glob("*"))
    return {path.relative_to(RELEASE).as_posix(): sha(path)
            for path in files if path.is_file()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["dependency-tree", "conformance-tests"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    receipt_path = OUT / (args.stage + ".json")
    if receipt_path.exists():
        raise RuntimeError("Stage already recorded; preserve it and coordinate a fresh attempt")
    psutil.Process().cpu_affinity([0])
    available = psutil.virtual_memory().available
    receipt = {"stage": args.stage, "started_at": now(), "status": "preflight",
               "working_directory": str(RELEASE / "bindings/jvm"),
               "available_before_bytes": available,
               "limits": {"minimum_available_bytes": MINIMUM_AVAILABLE,
                          "maximum_process_tree_rss_bytes": MAXIMUM_RSS,
                          "maximum_seconds": MAXIMUM_SECONDS, "cpu_affinity": [0],
                          "maven_threads": 1, "surefire_forks": 1,
                          "maven_heap_mib": 256, "surefire_heap_mib": 256},
               "source_hashes_before": sources(), "runner_sha256": sha(Path(__file__)),
               "python": sys.executable, "psutil_version": psutil.__version__,
               "java_executable_sha256": sha(JAVA_HOME / "bin/java.exe"),
               "maven_cmd_sha256": sha(MAVEN), "no_global_install": True,
               "no_latency_claim": True}
    write(receipt_path, receipt)
    if available < MINIMUM_AVAILABLE:
        receipt.update(status="deferred_insufficient_available_memory", finished_at=now())
        write(receipt_path, receipt)
        print(json.dumps({"stage": args.stage, "status": receipt["status"],
                          "available_bytes": available}), flush=True)
        return 75
    pom = ET.parse(RELEASE / "bindings/jvm/pom.xml").getroot()
    ns = {"m": "http://maven.apache.org/POM/4.0.0"}
    assert pom.findtext("m:version", namespaces=ns) == "0.1.62"
    assert pom.findtext("m:properties/m:jackson.version", namespaces=ns) == "2.21.7"
    settings = WORK / "settings.xml"
    settings.write_text('<settings xmlns="http://maven.apache.org/SETTINGS/1.2.0"/>\n', encoding="utf-8")
    common = ["--batch-mode", "--no-transfer-progress", "--settings", str(settings),
              "--global-settings", str(settings), "-T", "1",
              "-Dmaven.repo.local=" + str(TOOLS / "repository")]
    if args.stage == "dependency-tree":
        arguments = common + ["org.apache.maven.plugins:maven-dependency-plugin:3.8.1:tree",
                              "-DoutputType=json", "-DoutputFile=" + str(OUT / "dependency-tree-output.json")]
    else:
        prior = json.loads((OUT / "dependency-tree.json").read_text(encoding="utf-8"))
        assert prior["status"] == "passed" and prior["source_hashes_after"] == receipt["source_hashes_before"]
        arguments = common + ["test", "-DforkCount=1", "-DreuseForks=true", "-DargLine=" + JAVA_OPTIONS]
    environment = os.environ.copy()
    for key in ["_JAVA_OPTIONS", "JDK_JAVA_OPTIONS", "JAVA_TOOL_OPTIONS", "MAVEN_ARGS"]:
        environment.pop(key, None)
    environment.update(JAVA_HOME=str(JAVA_HOME), MAVEN_OPTS=JAVA_OPTIONS, MAVEN_SKIP_RC="true",
                       PATH=str(JAVA_HOME / "bin") + os.pathsep + environment.get("PATH", ""))
    # Popen must receive the complete cmd.exe command line as a string: passing
    # its /c payload through list2cmdline again escapes quotes for the C runtime,
    # while cmd.exe uses its own quote rules.
    command = (subprocess.list2cmdline([environment.get("COMSPEC", "C:/Windows/System32/cmd.exe")])
               + ' /d /s /c "' + subprocess.list2cmdline([str(MAVEN)] + arguments) + '"')
    receipt.update(status="running", command=command, maven_arguments=arguments,
                   environment_overrides={"JAVA_HOME": str(JAVA_HOME), "MAVEN_OPTS": JAVA_OPTIONS,
                                          "MAVEN_SKIP_RC": "true"},
                   settings_sha256=sha(settings))
    log_path = OUT / (args.stage + ".log")
    watchdog_path = ROOT / "benchmarks/owned_process_watchdog.py"
    spec = importlib.util.spec_from_file_location("release_jvm_watchdog", watchdog_path)
    watchdog = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(watchdog)
    receipt["watchdog_sha256"] = sha(watchdog_path)
    with log_path.open("wb") as log:
        process = subprocess.Popen(command, cwd=RELEASE / "bindings/jvm", env=environment,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        receipt["owned_process_pid"] = process.pid
        receipt["owned_process_create_time"] = psutil.Process(process.pid).create_time()
        write(receipt_path, receipt)
        print(json.dumps({"stage": args.stage, "pid": process.pid,
                          "available_before_bytes": available, "started_at": receipt["started_at"]}), flush=True)
        result = watchdog.watch(process, maximum_seconds=MAXIMUM_SECONDS,
                                maximum_rss_bytes=MAXIMUM_RSS)
    receipt.update(result, finished_at=now(), log_sha256=sha(log_path),
                   source_hashes_after=sources(), available_after_bytes=psutil.virtual_memory().available)
    receipt["status"] = "passed" if (result["exit_code"] == 0 and not result["resource_failure"]
                                       and result["cleanup_complete"]
                                       and receipt["source_hashes_before"] == receipt["source_hashes_after"]) else "failed"
    if receipt["status"] == "passed":
        if args.stage == "dependency-tree":
            path = OUT / "dependency-tree-output.json"
            tree = json.loads(path.read_text(encoding="utf-8"))
            dependencies = {}

            def visit(node):
                if node.get("groupId") == "com.fasterxml.jackson.core":
                    dependencies.setdefault(node["artifactId"], set()).add(node["version"])
                for child in node.get("children", []):
                    visit(child)

            visit(tree)
            actual = {key: sorted(value) for key, value in dependencies.items()}
            expected = {"jackson-core": ["2.21.7"], "jackson-databind": ["2.21.7"], "jackson-annotations": ["2.21"]}
            receipt.update(resolved_jackson=actual, expected_jackson=expected,
                           dependency_tree_sha256=sha(path))
            if actual != expected:
                receipt["status"] = "failed_dependency_versions"
        else:
            reports = sorted((RELEASE / "bindings/jvm/target/surefire-reports").glob("TEST-*.xml"))
            counts = dict(tests=0, failures=0, errors=0, skipped=0)
            cases = []
            for path in reports:
                suite = ET.parse(path).getroot()
                for key in counts:
                    counts[key] += int(suite.attrib.get(key, 0))
                cases.extend(case.attrib for case in suite.findall("testcase"))
                shutil.copyfile(path, OUT / path.name)
            receipt.update(counts=counts, test_cases=cases,
                           report_hashes={path.name: sha(path) for path in reports})
            if counts != dict(tests=9, failures=0, errors=0, skipped=0):
                receipt["status"] = "failed_conformance_counts"
    write(receipt_path, receipt)
    shutil.copyfile(Path(__file__), OUT / "run_validation.py")
    shutil.copyfile(watchdog_path, OUT / "owned_process_watchdog.py")
    print(json.dumps({key: receipt.get(key) for key in
                      ["stage", "status", "exit_code", "resource_failure", "peak_rss_bytes",
                       "wall_seconds", "cleanup_complete", "resolved_jackson", "counts"]}), flush=True)
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
