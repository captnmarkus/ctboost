from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import marshal
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCOUT_IMPORT_ROOT = REPO_ROOT / "benchmarks" / "split_research" / "g8s1_harness"
if str(SCOUT_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCOUT_IMPORT_ROOT))

import g8s1_scout.summary as scout_summary
from g8s1_scout import identity, loader, schedule
from g8s1_scout.constants import (
    BOOTSTRAP_RELATIVE,
    EXPECTED_ARTIFACTS,
    EXPECTED_CHUNKS,
    EXPECTED_SCHEDULE_SHA256,
    FORBIDDEN_BASE_FIELDS,
    NUM_CONFIGS,
    P50_SHA256,
    P200_RANDOM_SHA256,
    P201_SHA256,
    RUNBOOK_LF_NORMALIZED_SHA256,
    RUNBOOK_RELATIVE,
    RUNTIME_MODULE_FILES,
    TASKS,
    experiment_name,
    harness_package_root,
    source_root,
)
from g8s1_scout.models import (
    CTBoostGrouped8ScoutV1Model,
    CTBoostQuadraticScoutV1Model,
    _validate_pairs,
    base_p50,
    base_p201,
    base_random_p200,
    canonical_json_bytes,
    generate_configs_ctboost,
    paired_configs,
)
from g8s1_scout.summary import (
    _assert_no_absolute_paths,
    _comparison_outcome,
    _evaluate_decision_gates,
    _expected_raw_paths,
    namespace_inventory,
    validate_namespace,
    write_failure_summary,
)

identity.reset_and_seal_stdlib_statistics()


def _sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def test_frozen_portfolio_hashes_prefixes_and_released_default() -> None:
    random_p200 = base_random_p200()
    p201 = base_p201()
    p50 = base_p50()

    assert len(random_p200) == 200
    assert len(p201) == 201
    assert len(p50) == NUM_CONFIGS == 51
    assert _sha256(random_p200) == P200_RANDOM_SHA256
    assert _sha256(p201) == P201_SHA256
    assert _sha256(p50) == P50_SHA256
    assert p201[0] == {}
    assert p50 == p201[:NUM_CONFIGS]
    for count in (1, 8, 50, 200):
        assert generate_configs_ctboost(count) == random_p200[:count]

    for config in p201:
        assert not FORBIDDEN_BASE_FIELDS.intersection(config)
        assert "random_seed" not in config

    tree_header = (REPO_ROOT / "include" / "ctboost" / "tree.hpp").read_text(
        encoding="utf-8"
    )
    assert "FeatureTest feature_test{FeatureTest::Quadratic};" in tree_header
    assert (
        "FeatureTestAdjustment feature_test_adjustment{FeatureTestAdjustment::None};"
        in tree_header
    )


def test_tracked_import_root_preserves_subprocess_model_identity() -> None:
    assert source_root() == REPO_ROOT
    assert harness_package_root() == SCOUT_IMPORT_ROOT / "g8s1_scout"
    assert CTBoostQuadraticScoutV1Model.__module__ == "g8s1_scout.models"
    assert CTBoostGrouped8ScoutV1Model.__module__ == "g8s1_scout.models"
    assert CTBoostQuadraticScoutV1Model.ag_key == "CTBQS1"
    assert CTBoostGrouped8ScoutV1Model.ag_key == "CTBG8S1"


def _copy_bootstrap_repo(destination: Path) -> Path:
    fresh_repo = destination / "repo"
    copied_import_root = fresh_repo / "benchmarks/split_research/g8s1_harness"
    shutil.copytree(
        SCOUT_IMPORT_ROOT / "g8s1_scout",
        copied_import_root / "g8s1_scout",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for relative in (
        Path("benchmarks/__init__.py"),
        Path("benchmarks/tabarena/__init__.py"),
        Path("benchmarks/tabarena/ctboost_model.py"),
        Path("benchmarks/split_research/G8S1_SCOUT_MANIFEST.json"),
        Path(RUNBOOK_RELATIVE),
        Path("benchmarks/split_research/TABARENA_GROUPED_SCOUT.md"),
        Path("benchmarks/split_research/g8s1_scout_bootstrap.py"),
    ):
        destination_file = fresh_repo / relative
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / relative, destination_file)
    return fresh_repo


def test_canonical_no_bytecode_invocation_stays_clean_across_mocked_phases(
    tmp_path: Path,
) -> None:
    fresh_repo = _copy_bootstrap_repo(tmp_path)
    fresh_repo = tmp_path / "repo"
    copied_import_root = fresh_repo / "benchmarks/split_research/g8s1_harness"
    bootstrap = fresh_repo / "benchmarks/split_research/g8s1_scout_bootstrap.py"
    environment = {
        **os.environ,
        "PYTHONPATH": str(tmp_path / "untrusted-pythonpath"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    help_run = subprocess.run(
        [sys.executable, "-I", "-B", str(bootstrap), "--help"],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert help_run.returncode == 0, help_run.stderr
    normalized_help = " ".join(help_run.stdout.split())
    assert "pip install --no-compile" in normalized_help
    assert "G8S1_SCOUT_RUNBOOK.md" in normalized_help

    phase_script = r"""
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[2])
from g8s1_scout.summary import _expected_raw_paths, validate_namespace

assert sys.dont_write_bytecode
root = Path(sys.argv[1])
raw = root / "raw"
report = root / "report"
assert validate_namespace(raw_dir=raw, report_dir=report, phase="preflight")["complete"]

def write(relative_root, relative):
    path = relative_root / Path(relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"sealed")

write(report, "scout_provenance.json")
assert validate_namespace(raw_dir=raw, report_dir=report, phase="run_input")["complete"]
for relative in _expected_raw_paths():
    write(raw, relative)
write(report, "run_manifest.shard-00000-of-00001.json")
assert validate_namespace(raw_dir=raw, report_dir=report, phase="run_output")["complete"]
assert validate_namespace(raw_dir=raw, report_dir=report, phase="summarize_input")["complete"]
for relative in (
    "sanitized/scout_summary.json",
    "sanitized/paired_configs.json",
    "sanitized/config_results.csv",
    "sanitized/endpoint_results.csv",
):
    write(report, relative)
assert validate_namespace(raw_dir=raw, report_dir=report, phase="summary_success")["complete"]
"""
    phase_run = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            phase_script,
            str(tmp_path / "namespace"),
            str(copied_import_root),
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert phase_run.returncode == 0, phase_run.stderr
    assert not list(fresh_repo.rglob("*.pyc"))
    assert not list(fresh_repo.rglob("__pycache__"))


@pytest.mark.parametrize("module_name", ["__main__", "models"])
def test_external_bootstrap_rejects_effective_forged_runtime_cache_before_execution(
    tmp_path: Path, module_name: str
) -> None:
    fresh_repo = _copy_bootstrap_repo(tmp_path)
    package = fresh_repo / "benchmarks/split_research/g8s1_harness/g8s1_scout"
    source = package / f"{module_name}.py"
    sentinel = tmp_path / f"{module_name}-executed.txt"
    cache = Path(importlib.util.cache_from_source(str(source)))
    cache.parent.mkdir(parents=True)
    malicious = compile(
        "from pathlib import Path\n"
        f"Path({str(sentinel)!r}).write_text('executed', encoding='utf-8')\n",
        str(source.resolve()),
        "exec",
    )
    source_stat = source.stat()
    header = (
        importlib.util.MAGIC_NUMBER
        + b"\0" * 4
        + (int(source_stat.st_mtime) & 0xFFFFFFFF).to_bytes(4, "little")
        + (source_stat.st_size & 0xFFFFFFFF).to_bytes(4, "little")
    )
    cache.write_bytes(header + marshal.dumps(malicious))
    bootstrap = fresh_repo / "benchmarks/split_research/g8s1_scout_bootstrap.py"
    completed = subprocess.run(
        [sys.executable, "-I", "-B", str(bootstrap), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode != 0
    assert "exact source-only runtime" in completed.stderr
    assert not sentinel.exists()


def test_provenance_rejects_noncanonical_bytecode_writing_invocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    with pytest.raises(RuntimeError, match=r"python -I -B"):
        identity.collect_provenance(
            ctboost_root=tmp_path,
            tabarena_root=tmp_path,
            expected_ctboost_commit="a" * 40,
            expected_native_sha256="b" * 64,
        )


def test_paired_configs_are_deep_copied_and_differ_only_by_treatment() -> None:
    before = base_p50()
    paired = paired_configs()
    assert before == base_p50()

    for index, (quadratic, grouped) in enumerate(
        zip(paired["quadratic"], paired["grouped"], strict=True)
    ):
        differences = {
            key
            for key in set(quadratic).union(grouped)
            if quadratic.get(key) != grouped.get(key)
        }
        assert differences == {"feature_test"}, index
        assert quadratic["feature_test"] == "quadratic"
        assert grouped["feature_test"] == "grouped"
        assert quadratic["feature_test_bins"] == grouped["feature_test_bins"] == 8
        assert (
            quadratic["feature_test_adjustment"]
            == grouped["feature_test_adjustment"]
            == "none"
        )

    paired["grouped"][3]["max_depth"] = -1
    assert base_p50() == before


def test_pair_validation_fails_closed_on_non_treatment_change() -> None:
    paired = paired_configs()
    paired["grouped"][7]["feature_test_bins"] = 16
    with pytest.raises(RuntimeError, match="not only feature_test"):
        _validate_pairs(paired)


class _TaskMetadataCollection:
    @staticmethod
    def dataset_to_tid() -> dict[str, int]:
        return {dataset: task_id for task_id, dataset, _problem, _metric in TASKS}


def _frozen_chunks() -> tuple[SimpleNamespace, list[list[SimpleNamespace]]]:
    context = SimpleNamespace(task_metadata_collection=_TaskMetadataCollection())
    chunks: list[list[SimpleNamespace]] = []
    for _task_id, dataset, _problem, _metric in TASKS:
        jobs = []
        for treatment in ("quadratic", "grouped"):
            for index in range(NUM_CONFIGS):
                jobs.append(
                    SimpleNamespace(
                        experiment=SimpleNamespace(
                            name=experiment_name(treatment, index)
                        ),
                        task=SimpleNamespace(
                            dataset=dataset, repeat=0, fold=0, sample=0
                        ),
                    )
                )
        chunks.append(jobs)
    return context, chunks


def test_mocked_schedule_matches_every_frozen_identity() -> None:
    context, chunks = _frozen_chunks()
    observed = schedule.validate_job_chunks(context, chunks)
    assert observed == {
        "jobs": EXPECTED_ARTIFACTS,
        "schedule_sha256": EXPECTED_SCHEDULE_SHA256,
        "chunks": list(EXPECTED_CHUNKS),
    }


@pytest.mark.parametrize("field", ["repeat", "fold", "sample"])
def test_mocked_schedule_rejects_nonzero_split_coordinates(field: str) -> None:
    context, chunks = _frozen_chunks()
    setattr(chunks[0][0].task, field, 1)
    with pytest.raises(RuntimeError, match="nonzero repeat/fold/sample"):
        schedule.validate_job_chunks(context, chunks)


def test_mocked_schedule_rejects_identity_and_chunk_tampering() -> None:
    context, chunks = _frozen_chunks()
    wrong_method = copy.deepcopy(chunks)
    wrong_method[0][0].experiment.name = "CTBoostPostHoc"
    with pytest.raises(RuntimeError, match="identities differ"):
        schedule.validate_job_chunks(context, wrong_method)

    duplicate = copy.deepcopy(chunks)
    duplicate[0][-1] = copy.deepcopy(duplicate[0][0])
    with pytest.raises(RuntimeError, match="identities differ"):
        schedule.validate_job_chunks(context, duplicate)

    flattened = [[job for chunk in chunks for job in chunk]]
    with pytest.raises(RuntimeError, match="frozen job chunks"):
        schedule.validate_job_chunks(context, flattened)


def test_mocked_schedule_rejects_runtime_hash_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, chunks = _frozen_chunks()
    monkeypatch.setattr(schedule, "EXPECTED_SCHEDULE_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="schedule hash drift"):
        schedule.validate_job_chunks(context, chunks)


def _copy_harness_identity_tree(destination: Path) -> Path:
    relative_files = [
        Path("benchmarks/split_research/G8S1_SCOUT_MANIFEST.json"),
        Path(RUNBOOK_RELATIVE),
        Path("benchmarks/split_research/TABARENA_GROUPED_SCOUT.md"),
        Path(BOOTSTRAP_RELATIVE),
        *[
            Path("benchmarks/split_research/g8s1_harness/g8s1_scout") / name
            for name in RUNTIME_MODULE_FILES
        ],
    ]
    for relative in relative_files:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / relative, target)
    return destination


def test_tracked_harness_manifest_binds_exact_runtime_files_and_import() -> None:
    observed = identity._validate_harness_source(
        REPO_ROOT,
        require_tracked=False,
        verify_import_location=True,
    )
    manifest = json.loads(
        (REPO_ROOT / "benchmarks/split_research/G8S1_SCOUT_MANIFEST.json").read_text(
            encoding="utf-8"
        )
    )
    assert observed["runtime_files"] == dict(sorted(manifest["runtime_files"].items()))
    assert len(observed["runtime_files"]) == len(RUNTIME_MODULE_FILES) + 1 == 9
    assert (
        observed["runbook"]
        == manifest["runbook"]
        == {
            "path": RUNBOOK_RELATIVE,
            "lf_normalized_sha256": RUNBOOK_LF_NORMALIZED_SHA256,
        }
    )
    assert (
        identity.sha256_lf_normalized_file(REPO_ROOT / RUNBOOK_RELATIVE)
        == RUNBOOK_LF_NORMALIZED_SHA256
    )
    assert observed["imported_from_tracked_package"] is True


def test_harness_validation_requires_every_identity_file_to_be_tracked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_git(_root: Path, *arguments: str, check: bool = True) -> str:
        assert check
        calls.append(arguments)
        assert arguments[:3] == ("ls-files", "--error-unmatch", "--")
        return "\n".join(arguments[3:])

    monkeypatch.setattr(identity, "_git", fake_git)
    observed = identity._validate_harness_source(
        REPO_ROOT,
        require_tracked=True,
        verify_import_location=True,
    )
    assert observed["tracked"] is True
    assert len(calls) == 1
    assert len(calls[0][3:]) == len(RUNTIME_MODULE_FILES) + 4


def test_harness_validation_rejects_untracked_identity_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_untracked(_root: Path, *_arguments: str, check: bool = True) -> str:
        assert check
        raise RuntimeError("untracked")

    monkeypatch.setattr(identity, "_git", reject_untracked)
    with pytest.raises(RuntimeError, match="untracked"):
        identity._validate_harness_source(
            REPO_ROOT,
            require_tracked=True,
            verify_import_location=False,
        )


def test_harness_validation_rejects_modified_missing_and_extra_files(
    tmp_path: Path,
) -> None:
    modified = _copy_harness_identity_tree(tmp_path / "modified")
    identity_file = (
        modified / "benchmarks/split_research/g8s1_harness/g8s1_scout/identity.py"
    )
    identity_file.write_text(
        identity_file.read_text(encoding="utf-8") + "# tamper\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="hash drifted"):
        identity._validate_harness_source(
            modified,
            require_tracked=False,
            verify_import_location=False,
        )

    missing = _copy_harness_identity_tree(tmp_path / "missing")
    (missing / "benchmarks/split_research/g8s1_harness/g8s1_scout/loader.py").unlink()
    with pytest.raises(RuntimeError, match="file set drifted"):
        identity._validate_harness_source(
            missing,
            require_tracked=False,
            verify_import_location=False,
        )

    extra = _copy_harness_identity_tree(tmp_path / "extra")
    (
        extra / "benchmarks/split_research/g8s1_harness/g8s1_scout/post_hoc.py"
    ).write_text("# forbidden\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="file set drifted"):
        identity._validate_harness_source(
            extra,
            require_tracked=False,
            verify_import_location=False,
        )


def test_harness_validation_rejects_tampered_sealed_runbook(tmp_path: Path) -> None:
    copied = _copy_harness_identity_tree(tmp_path / "runbook-tamper")
    runbook = copied / RUNBOOK_RELATIVE
    runbook.write_text(
        runbook.read_text(encoding="utf-8") + "\npost-hoc instruction\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="runbook hash drifted"):
        identity._validate_harness_source(
            copied,
            require_tracked=False,
            verify_import_location=False,
        )


def test_harness_runtime_hashes_are_stable_across_git_line_endings(
    tmp_path: Path,
) -> None:
    copied = _copy_harness_identity_tree(tmp_path / "crlf")
    runtime_file = (
        copied / "benchmarks/split_research/g8s1_harness/g8s1_scout/models.py"
    )
    payload = runtime_file.read_bytes().replace(b"\r\n", b"\n")
    runtime_file.write_bytes(payload.replace(b"\n", b"\r\n"))
    protocol_file = copied / "benchmarks/split_research/TABARENA_GROUPED_SCOUT.md"
    protocol_file.write_bytes(protocol_file.read_bytes().replace(b"\r\n", b"\n"))
    identity._validate_harness_source(
        copied,
        require_tracked=False,
        verify_import_location=False,
    )


def test_harness_validation_rejects_binary_module_shadow(tmp_path: Path) -> None:
    copied = _copy_harness_identity_tree(tmp_path / "binary-shadow")
    shadow = copied / "benchmarks/split_research/g8s1_harness/g8s1_scout/models.pyd"
    shadow.write_bytes(b"not a real extension")
    with pytest.raises(RuntimeError, match="unexpected importable"):
        identity._validate_harness_source(
            copied,
            require_tracked=False,
            verify_import_location=False,
        )


@pytest.mark.parametrize(
    "relative",
    [
        "statistics.py",
        "statistics.pyc",
        "rogue.pyd",
        "nested/rogue.so",
        "nested/rogue.py",
    ],
)
def test_harness_validation_closes_entire_import_root(
    tmp_path: Path, relative: str
) -> None:
    copied = _copy_harness_identity_tree(tmp_path / "import-root-shadow")
    shadow = copied / "benchmarks/split_research/g8s1_harness" / relative
    shadow.parent.mkdir(parents=True, exist_ok=True)
    shadow.write_bytes(b"shadow")
    with pytest.raises(RuntimeError, match="import root.*unexpected importable"):
        identity._validate_harness_source(
            copied,
            require_tracked=False,
            verify_import_location=False,
        )


def test_harness_validation_rejects_tampered_current_runtime_cache(
    tmp_path: Path,
) -> None:
    copied = _copy_harness_identity_tree(tmp_path / "tampered-cache")
    source = copied / "benchmarks/split_research/g8s1_harness/g8s1_scout/models.py"
    cache = Path(identity.importlib.util.cache_from_source(str(source)))
    cache.parent.mkdir(parents=True)
    source_stat = source.stat()
    matching_header = (
        identity.importlib.util.MAGIC_NUMBER
        + b"\0" * 4
        + (int(source_stat.st_mtime) & 0xFFFFFFFF).to_bytes(4, "little")
        + (source_stat.st_size & 0xFFFFFFFF).to_bytes(4, "little")
    )
    cache.write_bytes(matching_header + b"not valid marshal data")
    with pytest.raises(RuntimeError, match="import root.*unexpected importable"):
        identity._validate_harness_source(
            copied,
            require_tracked=False,
            verify_import_location=False,
        )


def test_harness_validation_rejects_preloaded_nonstdlib_statistics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    del tmp_path
    sealed = sys.modules["statistics"]
    expected = Path(sealed.__file__).resolve()
    fake = ModuleType("statistics")
    fake_loader = identity.importlib.machinery.SourceFileLoader(
        "statistics", str(expected)
    )
    fake.__file__ = str(expected)
    fake.__loader__ = fake_loader
    fake.__spec__ = identity.importlib.util.spec_from_loader(
        "statistics", fake_loader, origin=str(expected)
    )
    monkeypatch.setitem(sys.modules, "statistics", fake)
    with pytest.raises(RuntimeError, match="bootstrap-sealed stdlib source"):
        identity._validate_stdlib_statistics()


def test_loader_requires_exact_package_path_and_file(tmp_path: Path) -> None:
    package_root = tmp_path / "benchmarks"
    package_root.mkdir()
    expected_file = package_root / "__init__.py"
    expected_file.write_text("", encoding="utf-8")
    name = "g8s1_adversarial_benchmarks"
    sys.modules[name] = SimpleNamespace(
        __path__=[str(package_root), str(tmp_path / "shadow")],
        __file__=str(expected_file),
    )
    try:
        with pytest.raises(RuntimeError, match="preloaded benchmark package"):
            loader._ensure_package(name, package_root)
    finally:
        sys.modules.pop(name, None)


def test_benchmark_child_validation_rejects_spoofed_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "benchmarks.tabarena.ctboost_model"
    expected_module = loader._LOADED_MODULES[name]
    fake = ModuleType(name)
    fake.__file__ = expected_module.__file__
    fake.__loader__ = expected_module.__loader__
    fake.__spec__ = expected_module.__spec__
    monkeypatch.setitem(sys.modules, name, fake)
    with pytest.raises(RuntimeError, match="invalid origin"):
        loader.validate_loaded_benchmark_modules(REPO_ROOT / "benchmarks/tabarena")


def test_benchmark_child_validation_rejects_unexpected_preloaded_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules, "benchmarks.tabarena.stale_child", ModuleType("stale_child")
    )
    with pytest.raises(RuntimeError, match="unexpected or preloaded"):
        loader.validate_loaded_benchmark_modules(REPO_ROOT / "benchmarks/tabarena")


def test_tabarena_validation_rejects_extended_namespace_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "tabarena"
    (checkout / ".git").mkdir(parents=True)
    package_root = checkout / "packages/tabarena/src/tabarena"
    package_root.mkdir(parents=True)
    init_file = package_root / "__init__.py"
    init_file.write_text("", encoding="utf-8")
    fake = SimpleNamespace(
        __file__=str(init_file),
        __path__=[str(package_root), str(tmp_path / "shadow")],
    )
    monkeypatch.setitem(sys.modules, "tabarena", fake)
    monkeypatch.setattr(
        identity, "_full_commit", lambda _root: identity.PROTOCOL_TABARENA_COMMIT
    )
    monkeypatch.setattr(identity, "_git", lambda *_args, **_kwargs: "")
    with pytest.raises(RuntimeError, match="does not come from the pinned checkout"):
        identity._validate_tabarena_source(checkout)


def test_harness_validation_rejects_symlinked_runtime_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copied = _copy_harness_identity_tree(tmp_path / "symlink")
    loader = copied / "benchmarks/split_research/g8s1_harness/g8s1_scout/loader.py"
    original_is_symlink = Path.is_symlink

    def report_loader_as_symlink(path: Path) -> bool:
        return path == loader or original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", report_loader_as_symlink)
    with pytest.raises(RuntimeError, match="symlink"):
        identity._validate_harness_source(
            copied,
            require_tracked=False,
            verify_import_location=False,
        )


def test_import_location_validation_rejects_shadow_package(tmp_path: Path) -> None:
    expected = REPO_ROOT / "benchmarks/split_research/g8s1_harness/g8s1_scout"
    with pytest.raises(RuntimeError, match="does not come from"):
        identity._validate_import_location(
            expected_package_root=expected,
            package_file=tmp_path / "g8s1_scout/__init__.py",
            package_paths=[expected],
            identity_file=expected / "identity.py",
        )


def _write_mock_python_package(root: Path, marker: str) -> None:
    (root / "core").mkdir(parents=True)
    (root / "__init__.py").write_text(f"IDENTITY = {marker!r}\n", encoding="utf-8")
    (root / "core/__init__.py").write_text("", encoding="utf-8")


def test_source_package_seal_binds_installed_package_to_commit(tmp_path: Path) -> None:
    source_package = tmp_path / "source/ctboost"
    installed_package = tmp_path / "site-packages/ctboost"
    _write_mock_python_package(source_package, "expected")
    shutil.copytree(source_package, installed_package)
    commit = "a" * 40

    binding = identity._bind_installed_package_to_source(
        source_package_root=source_package,
        installed_package_root=installed_package,
        source_commit=commit,
    )
    assert binding["source_commit"] == commit
    assert binding["python_file_count"] == 2
    assert len(binding["source_package_seal_sha256"]) == 64


def test_source_package_seal_rejects_unrelated_installed_package_even_if_external_identity_matches(
    tmp_path: Path,
) -> None:
    source_package = tmp_path / "source/ctboost"
    installed_package = tmp_path / "site-packages/ctboost"
    _write_mock_python_package(source_package, "expected")
    _write_mock_python_package(installed_package, "unrelated")

    with pytest.raises(RuntimeError, match="does not match the expected source commit"):
        identity._bind_installed_package_to_source(
            source_package_root=source_package,
            installed_package_root=installed_package,
            source_commit="b" * 40,
        )


def test_installed_identity_rejects_unrelated_package_with_matching_version_api_and_native(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    source_package = source_root / "ctboost"
    installed_package = tmp_path / "site-packages/ctboost"
    _write_mock_python_package(source_package, "expected")
    _write_mock_python_package(installed_package, "unrelated")
    native_payload = b"matching caller-provided native identity"
    (installed_package / "_core.pyd").write_bytes(native_payload)

    class MatchingAPI:
        def __init__(
            self,
            *,
            feature_test: str | None = None,
            feature_test_bins: int | None = None,
            feature_test_adjustment: str | None = None,
        ) -> None:
            del feature_test, feature_test_bins, feature_test_adjustment

    fake_ctboost = SimpleNamespace(
        __file__=str(installed_package / "__init__.py"),
        CTBoostClassifier=MatchingAPI,
        CTBoostRegressor=MatchingAPI,
        build_info=lambda: {"cuda_enabled": False},
    )
    monkeypatch.setitem(sys.modules, "ctboost", fake_ctboost)
    monkeypatch.setattr(
        identity.importlib.metadata,
        "version",
        lambda name: "0.1.54" if name == "ctboost" else "unexpected",
    )
    monkeypatch.setattr(identity, "_PREVALIDATED_INSTALLED", None)
    with pytest.raises(RuntimeError, match="not sealed before package import"):
        identity._installed_ctboost_identity(
            hashlib.sha256(native_payload).hexdigest(),
            source_root,
            "c" * 40,
        )


def _installed_record_hashes(package_root: Path) -> dict[str, str]:
    return {
        path.relative_to(package_root).as_posix(): identity.sha256_file(path)
        for path in package_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".py", ".pyc", ".pyd", ".so"}
    }


@pytest.mark.parametrize(
    "extra_relative",
    [
        "__pycache__/__init__.cpython-312.pyc",
        "__init__.pyd",
        "nested/rogue.so",
    ],
)
def test_installed_source_seal_rejects_extra_importables_before_import(
    tmp_path: Path, extra_relative: str
) -> None:
    source_package = tmp_path / "source/ctboost"
    installed_package = tmp_path / "site-packages/ctboost"
    _write_mock_python_package(source_package, "expected")
    shutil.copytree(source_package, installed_package)
    native = installed_package / "_core.pyd"
    native.write_bytes(b"sealed native")
    extra = installed_package / Path(extra_relative)
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_bytes(b"must never execute")
    with pytest.raises(
        RuntimeError,
        match="forbidden cached bytecode|exactly one top-level _core extension",
    ):
        identity._installed_importable_files(
            source_package_root=source_package,
            installed_package_root=installed_package,
            source_commit="d" * 40,
            expected_native_sha256=identity.sha256_file(native),
            record_sha256=_installed_record_hashes(installed_package),
        )


def test_installed_source_seal_requires_exact_wheel_record(tmp_path: Path) -> None:
    source_package = tmp_path / "source/ctboost"
    installed_package = tmp_path / "site-packages/ctboost"
    _write_mock_python_package(source_package, "expected")
    shutil.copytree(source_package, installed_package)
    native = installed_package / "_core.pyd"
    native.write_bytes(b"sealed native")
    record = _installed_record_hashes(installed_package)
    record["__init__.py"] = "0" * 64
    with pytest.raises(RuntimeError, match="wheel RECORD"):
        identity._installed_importable_files(
            source_package_root=source_package,
            installed_package_root=installed_package,
            source_commit="e" * 40,
            expected_native_sha256=identity.sha256_file(native),
            record_sha256=record,
        )


def test_installed_source_seal_accepts_exact_source_native_and_record(
    tmp_path: Path,
) -> None:
    source_package = tmp_path / "source/ctboost"
    installed_package = tmp_path / "site-packages/ctboost"
    _write_mock_python_package(source_package, "expected")
    shutil.copytree(source_package, installed_package)
    native = installed_package / "_core.pyd"
    native.write_bytes(b"sealed native")
    sealed = identity._installed_importable_files(
        source_package_root=source_package,
        installed_package_root=installed_package,
        source_commit="f" * 40,
        expected_native_sha256=identity.sha256_file(native),
        record_sha256=_installed_record_hashes(installed_package),
    )
    assert sealed["native_path"] == native.resolve()
    assert len(sealed["source_binding"]["wheel_record_sha256"]) == 64


BOUNDARY_GATES = {
    "primary_wins": 2,
    "primary_macro_median": 0.0025,
    "primary_worst": -0.01,
    "tuned_macro_median": 0.0,
    "tuned_worst": -0.02,
    "median_paired_training_ratio": 1.15,
}


def test_decision_gates_pass_at_every_frozen_boundary() -> None:
    assert all(_evaluate_decision_gates(**BOUNDARY_GATES).values())
    assert _comparison_outcome(0.000999) == "tie"
    assert _comparison_outcome(-0.000999) == "tie"
    assert _comparison_outcome(0.001) == "win"
    assert _comparison_outcome(-0.001) == "loss"


@pytest.mark.parametrize(
    ("field", "failing_value"),
    [
        ("primary_wins", 1),
        ("primary_macro_median", 0.002499999),
        ("primary_worst", -0.010000001),
        ("tuned_macro_median", -0.000000001),
        ("tuned_worst", -0.020000001),
        ("median_paired_training_ratio", 1.150000001),
    ],
)
def test_each_decision_gate_fails_immediately_outside_boundary(
    field: str, failing_value: float
) -> None:
    values = {**BOUNDARY_GATES, field: failing_value}
    gates = _evaluate_decision_gates(**values)
    assert sum(not passed for passed in gates.values()) == 1


def _write_relative_files(
    root: Path, relative_paths: set[str] | frozenset[str]
) -> None:
    for relative in relative_paths:
        path = root / Path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"sealed")


def test_namespace_inventory_enforces_exact_files_at_every_phase(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    report_dir = tmp_path / "report"
    _write_relative_files(report_dir, {"scout_provenance.json"})
    expected_raw = _expected_raw_paths()
    first = min(expected_raw)
    _write_relative_files(raw_dir, {first})
    run_input = validate_namespace(
        raw_dir=raw_dir, report_dir=report_dir, phase="run_input"
    )
    assert run_input["observed_outer_artifacts"] == 1

    _write_relative_files(raw_dir, expected_raw - {first})
    _write_relative_files(report_dir, {"run_manifest.shard-00000-of-00001.json"})
    assert validate_namespace(
        raw_dir=raw_dir, report_dir=report_dir, phase="run_output"
    )["complete"]
    assert validate_namespace(
        raw_dir=raw_dir, report_dir=report_dir, phase="summarize_input"
    )["complete"]

    success_outputs = {
        "sanitized/scout_summary.json",
        "sanitized/paired_configs.json",
        "sanitized/config_results.csv",
        "sanitized/endpoint_results.csv",
    }
    _write_relative_files(report_dir, success_outputs)
    assert validate_namespace(
        raw_dir=raw_dir, report_dir=report_dir, phase="summary_success"
    )["complete"]

    failure_report = tmp_path / "failure-report"
    _write_relative_files(
        failure_report,
        {
            "scout_provenance.json",
            "run_manifest.shard-00000-of-00001.json",
            "sanitized/scout_failure.json",
        },
    )
    assert validate_namespace(
        raw_dir=raw_dir, report_dir=failure_report, phase="summary_failure"
    )["complete"]


def test_existing_sealed_namespace_counts_and_rejects_every_stale_file(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    report_dir = tmp_path / "report"
    _write_relative_files(
        raw_dir,
        {
            "data/Unscheduled/363614/0_0/results.pkl",
            "data/Unscheduled/363614/0_0/debug.txt",
        },
    )
    _write_relative_files(
        report_dir,
        {"scout_provenance.json", "sanitized/stale.json"},
    )
    inventory = namespace_inventory(
        raw_dir=raw_dir, report_dir=report_dir, phase="run_input"
    )
    assert inventory["stale_or_unexpected"] == 3
    with pytest.raises(RuntimeError, match="stale, unexpected, or linked"):
        validate_namespace(raw_dir=raw_dir, report_dir=report_dir, phase="run_input")


def test_summary_coverage_uses_observed_namespace_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        scout_summary,
        "load_and_validate_artifacts",
        lambda _raw_dir: (
            [],
            [],
            "0" * 64,
            {"stale_or_unexpected": 3, "complete": False},
        ),
    )
    monkeypatch.setattr(scout_summary, "_endpoint_rows", lambda _loaded: ([], {}))
    monkeypatch.setattr(
        scout_summary,
        "_decorate_endpoint_selections",
        lambda _rows, _records, _selected: None,
    )
    monkeypatch.setattr(scout_summary, "_decision_summary", lambda *_args: {})
    monkeypatch.setattr(scout_summary, "_resource_summary", lambda _records: {})
    summary = scout_summary.summarize(
        raw_dir=tmp_path / "raw",
        output_dir=tmp_path / "report/sanitized",
        provenance={"identity": "sealed"},
        schedule={"schedule_sha256": EXPECTED_SCHEDULE_SHA256},
        namespace_state={"stale_or_unexpected": 7, "complete": False},
    )
    assert summary["coverage"]["stale_or_unexpected"] == 7
    assert summary["coverage"]["complete"] is False


def test_partial_success_staging_is_removed_before_failure_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        scout_summary,
        "load_and_validate_artifacts",
        lambda _raw_dir: (
            [],
            [],
            "0" * 64,
            {"stale_or_unexpected": 0, "complete": True},
        ),
    )
    monkeypatch.setattr(scout_summary, "_endpoint_rows", lambda _loaded: ([], {}))
    monkeypatch.setattr(
        scout_summary,
        "_decorate_endpoint_selections",
        lambda _rows, _records, _selected: None,
    )
    monkeypatch.setattr(scout_summary, "_decision_summary", lambda *_args: {})
    monkeypatch.setattr(scout_summary, "_resource_summary", lambda _records: {})
    original_write_json = scout_summary._write_json
    writes = 0

    def fail_second_json(path: Path, value: object) -> None:
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("simulated partial success write")
        original_write_json(path, value)

    monkeypatch.setattr(scout_summary, "_write_json", fail_second_json)
    output_dir = tmp_path / "report/sanitized"
    with pytest.raises(OSError, match="partial success"):
        scout_summary.summarize(
            raw_dir=tmp_path / "raw",
            output_dir=output_dir,
            provenance={"identity": "sealed"},
            schedule={"schedule_sha256": EXPECTED_SCHEDULE_SHA256},
            namespace_state={"stale_or_unexpected": 0, "complete": True},
        )
    assert not output_dir.exists()
    assert not list(output_dir.parent.glob(".g8s1-success-*"))

    monkeypatch.setattr(scout_summary, "_write_json", original_write_json)
    write_failure_summary(
        output_dir=output_dir,
        provenance={"identity": "sealed"},
        schedule={"schedule_sha256": EXPECTED_SCHEDULE_SHA256},
        error=RuntimeError("simulated integration failure"),
    )
    assert {path.name for path in output_dir.iterdir()} == {"scout_failure.json"}


@pytest.mark.parametrize(
    "interruption",
    [KeyboardInterrupt(), SystemExit(17)],
    ids=("keyboard-interrupt", "system-exit"),
)
def test_summary_staging_is_removed_for_base_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interruption: BaseException,
) -> None:
    monkeypatch.setattr(
        scout_summary,
        "load_and_validate_artifacts",
        lambda _raw_dir: (
            [],
            [],
            "0" * 64,
            {"stale_or_unexpected": 0, "complete": True},
        ),
    )
    monkeypatch.setattr(scout_summary, "_endpoint_rows", lambda _loaded: ([], {}))
    monkeypatch.setattr(
        scout_summary,
        "_decorate_endpoint_selections",
        lambda _rows, _records, _selected: None,
    )
    monkeypatch.setattr(scout_summary, "_decision_summary", lambda *_args: {})
    monkeypatch.setattr(scout_summary, "_resource_summary", lambda _records: {})
    original_write_json = scout_summary._write_json
    writes = 0

    def interrupt_second_json(path: Path, value: object) -> None:
        nonlocal writes
        writes += 1
        if writes == 2:
            raise interruption
        original_write_json(path, value)

    monkeypatch.setattr(scout_summary, "_write_json", interrupt_second_json)
    output_dir = tmp_path / "report/sanitized"
    with pytest.raises(type(interruption)):
        scout_summary.summarize(
            raw_dir=tmp_path / "raw",
            output_dir=output_dir,
            provenance={"identity": "sealed"},
            schedule={"schedule_sha256": EXPECTED_SCHEDULE_SHA256},
            namespace_state={"stale_or_unexpected": 0, "complete": True},
        )
    assert not output_dir.exists()
    assert not list(output_dir.parent.glob(".g8s1-success-*"))


@pytest.mark.parametrize(
    "interruption",
    [KeyboardInterrupt(), SystemExit(23)],
    ids=("keyboard-interrupt", "system-exit"),
)
def test_failure_staging_is_removed_for_base_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interruption: BaseException,
) -> None:
    original_write_json = scout_summary._write_json

    def interrupt_after_partial_write(path: Path, value: object) -> None:
        original_write_json(path, value)
        raise interruption

    monkeypatch.setattr(scout_summary, "_write_json", interrupt_after_partial_write)
    output_dir = tmp_path / "report/sanitized"
    with pytest.raises(type(interruption)):
        write_failure_summary(
            output_dir=output_dir,
            provenance={"identity": "sealed"},
            schedule={"schedule_sha256": EXPECTED_SCHEDULE_SHA256},
            error=RuntimeError("simulated integration failure"),
        )
    assert not output_dir.exists()
    assert not list(output_dir.parent.glob(".g8s1-failure-*"))


def test_failure_writer_refuses_existing_partial_success_directory(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "sanitized"
    output_dir.mkdir()
    (output_dir / "scout_summary.json").write_text("partial", encoding="utf-8")
    with pytest.raises(RuntimeError, match="clean sanitized directory"):
        write_failure_summary(
            output_dir=output_dir,
            provenance={"identity": "sealed"},
            schedule={"schedule_sha256": EXPECTED_SCHEDULE_SHA256},
            error=RuntimeError("later failure"),
        )
    assert not (output_dir / "scout_failure.json").exists()


@pytest.mark.parametrize(
    "absolute_path",
    [
        "/etc/passwd",
        "/root/private/result.pkl",
        "/opt/ctboost/result.pkl",
        "embedded=/srv/private/result.pkl",
        r"C:\private\artifact.pkl",
        "D:/private/artifact.pkl",
        r"\private\artifact.pkl",
        r"\\server\share\artifact.pkl",
        "//server/share/artifact.pkl",
        "file:///etc/passwd",
        "FILE://server/share/artifact.pkl",
    ],
)
def test_sanitizer_rejects_all_absolute_path_forms(absolute_path: str) -> None:
    with pytest.raises(RuntimeError, match="absolute path"):
        _assert_no_absolute_paths({"nested": [{"bad": f"failure: {absolute_path}"}]})


def test_sanitizer_allows_relative_paths_and_https_urls() -> None:
    _assert_no_absolute_paths(
        {
            "safe": "raw/data/model/363614/0_0/results.pkl",
            "source": "https://example.invalid/docs/path",
        }
    )


@pytest.mark.parametrize(
    "absolute_path",
    [
        "/etc/passwd",
        r"C:\private\artifact.pkl",
        r"\private\artifact.pkl",
        r"\\server\share\artifact.pkl",
        "file:///root/private",
    ],
)
def test_failure_summary_scrubs_embedded_absolute_paths(
    tmp_path: Path, absolute_path: str
) -> None:

    output_dir = tmp_path / "sanitized"
    write_failure_summary(
        output_dir=output_dir,
        provenance={"identity": "sealed"},
        schedule={"schedule_sha256": EXPECTED_SCHEDULE_SHA256},
        error=RuntimeError(f"validation failed at {absolute_path} inside artifact"),
    )
    failure = json.loads(
        (output_dir / "scout_failure.json").read_text(encoding="utf-8")
    )
    assert failure["status"] == "integration failure"
    assert (
        failure["error"]
        == "validation failed with a path-bearing diagnostic; inspect private logs"
    )
    _assert_no_absolute_paths(failure)
