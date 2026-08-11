"""Fail-closed source, package, and protocol identity checks."""

from __future__ import annotations

import base64
import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import importlib.util
import inspect
import json
import re
import subprocess
import sys
import sysconfig
from pathlib import Path
from types import ModuleType
from typing import Any

from .constants import (
    BOOTSTRAP_RELATIVE,
    EXPECTED_ARTIFACTS,
    EXPECTED_CHILD_FITS,
    EXPECTED_CHUNKS,
    EXPECTED_SCHEDULE_SHA256,
    HISTOGRAM_THREADS,
    JOB_BATCH_SIZE,
    MEMORY_LIMIT_GB,
    NUM_CPUS,
    NUM_GPUS,
    P50_SHA256,
    P200_RANDOM_SHA256,
    P201_SHA256,
    PROTOCOL_LF_NORMALIZED_SHA256,
    PROTOCOL_SHA256,
    PROTOCOL_TABARENA_COMMIT,
    RUNBOOK_LF_NORMALIZED_SHA256,
    RUNBOOK_RELATIVE,
    RUNTIME_MODULE_FILES,
    TIME_LIMIT_SECONDS,
    bootstrap_path,
    harness_package_root,
    manifest_path,
    protocol_path,
)

_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SEALED_STATISTICS: ModuleType | None = None
_SEALED_STATISTICS_SHA256: str | None = None
_EXTERNAL_BOOTSTRAP: tuple[Path, Path, Path] | None = None
_PREVALIDATED_INSTALLED: dict[str, Any] | None = None


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_lf_normalized_file(path: Path) -> str:
    """Hash tracked text reproducibly across Git LF/CRLF checkout policies."""
    payload = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def _git(root: Path, *arguments: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if check and completed.returncode != 0:
        detail = (
            completed.stderr.strip() or completed.stdout.strip() or "git command failed"
        )
        raise RuntimeError(detail)
    return completed.stdout.strip()


def _full_commit(root: Path) -> str:
    commit = _git(root, "rev-parse", "HEAD").lower()
    if not _SHA1.fullmatch(commit):
        raise RuntimeError("source checkout did not resolve to a full Git commit")
    return commit


def _expected_manifest_contract() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "name": "ctboost-grouped-statistic-tabarena-scout-harness-v1",
        "runtime_hash_mode": "sha256-lf-normalized-bytes",
        "protocol": {
            "name": "ctboost-grouped-statistic-tabarena-scout-v1",
            "path": "benchmarks/split_research/TABARENA_GROUPED_SCOUT.md",
            "crlf_sha256": PROTOCOL_SHA256,
            "lf_normalized_sha256": PROTOCOL_LF_NORMALIZED_SHA256,
        },
        "runbook": {
            "path": RUNBOOK_RELATIVE,
            "lf_normalized_sha256": RUNBOOK_LF_NORMALIZED_SHA256,
        },
        "tabarena_commit": PROTOCOL_TABARENA_COMMIT,
        "portfolio": {
            "p50_with_default_sha256": P50_SHA256,
            "random_p200_sha256": P200_RANDOM_SHA256,
            "p201_with_default_sha256": P201_SHA256,
        },
        "schedule": {
            "experiments": 102,
            "outer_artifacts": EXPECTED_ARTIFACTS,
            "bagged_child_fits": EXPECTED_CHILD_FITS,
            "chunks": list(EXPECTED_CHUNKS),
            "sha256": EXPECTED_SCHEDULE_SHA256,
        },
        "resources": {
            "device": "cpu",
            "num_cpus": NUM_CPUS,
            "num_gpus": NUM_GPUS,
            "memory_limit_gb": MEMORY_LIMIT_GB,
            "time_limit_seconds": TIME_LIMIT_SECONDS,
            "histogram_threads": int(HISTOGRAM_THREADS),
            "ray": False,
            "shard_count": 1,
            "job_batch_size": JOB_BATCH_SIZE,
        },
    }


def _validate_import_location(
    *,
    expected_package_root: Path,
    package_file: Path,
    package_paths: list[Path],
    identity_file: Path,
) -> None:
    expected_package_root = expected_package_root.resolve()
    if package_file.resolve() != expected_package_root / "__init__.py":
        raise RuntimeError(
            "imported g8s1_scout package does not come from the tracked harness"
        )
    if [path.resolve() for path in package_paths] != [expected_package_root]:
        raise RuntimeError(
            "imported g8s1_scout package path differs from the tracked harness"
        )
    if identity_file.resolve() != expected_package_root / "identity.py":
        raise RuntimeError(
            "imported g8s1_scout identity module is not the tracked file"
        )


def _validate_current_import_location(expected_package_root: Path) -> None:
    package = sys.modules.get("g8s1_scout")
    if package is None or not getattr(package, "__file__", None):
        raise RuntimeError(
            "tracked harness must be imported as top-level package g8s1_scout"
        )
    package_paths = [Path(value) for value in getattr(package, "__path__", ())]
    _validate_import_location(
        expected_package_root=expected_package_root,
        package_file=Path(package.__file__),
        package_paths=package_paths,
        identity_file=Path(__file__),
    )


def _module_origin(module: Any) -> Path | None:
    spec = getattr(module, "__spec__", None)
    origin = getattr(spec, "origin", None) or getattr(module, "__file__", None)
    if not isinstance(origin, str) or origin in {"built-in", "frozen"}:
        return None
    return Path(origin).resolve()


def reset_and_seal_stdlib_statistics() -> ModuleType:
    global _SEALED_STATISTICS, _SEALED_STATISTICS_SHA256
    expected = (Path(sysconfig.get_path("stdlib")) / "statistics.py").resolve()
    if not expected.is_file() or expected.is_symlink():
        raise RuntimeError("standard-library statistics source is missing or linked")
    source_sha256 = sha256_file(expected)
    loader = importlib.machinery.SourceFileLoader("statistics", str(expected))
    spec = importlib.util.spec_from_loader("statistics", loader, origin=str(expected))
    if spec is None:
        raise RuntimeError("could not construct the sealed statistics module")
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(expected)
    sys.modules.pop("statistics", None)
    sys.modules["statistics"] = module
    try:
        code = compile(expected.read_bytes(), str(expected), "exec", dont_inherit=True)
        exec(code, module.__dict__)  # noqa: S102 - execute exact stdlib source
    except Exception:
        sys.modules.pop("statistics", None)
        raise
    _SEALED_STATISTICS = module
    _SEALED_STATISTICS_SHA256 = source_sha256
    _validate_stdlib_statistics()
    return module


def _validate_stdlib_statistics() -> None:
    expected = (Path(sysconfig.get_path("stdlib")) / "statistics.py").resolve()
    loaded = sys.modules.get("statistics")
    if (
        _SEALED_STATISTICS is None
        or loaded is not _SEALED_STATISTICS
        or not isinstance(loaded, ModuleType)
        or not isinstance(getattr(loaded, "__file__", None), str)
        or Path(loaded.__file__).resolve() != expected
        or _module_origin(loaded) != expected
        or not isinstance(loaded.__loader__, importlib.machinery.SourceFileLoader)
        or Path(loaded.__loader__.path).resolve() != expected
        or loaded.__spec__ is None
        or loaded.__spec__.loader is not loaded.__loader__
        or _SEALED_STATISTICS_SHA256 != sha256_file(expected)
    ):
        raise RuntimeError(
            "statistics module is not the bootstrap-sealed stdlib source"
        )
    for name in ("mean", "median"):
        function = getattr(loaded, name, None)
        code = getattr(function, "__code__", None)
        if code is None or Path(code.co_filename).resolve() != expected:
            raise RuntimeError("statistics implementation does not match stdlib source")


def mark_external_bootstrap_validated(
    *, source_root: Path, import_root: Path, bootstrap_file: Path
) -> None:
    global _EXTERNAL_BOOTSTRAP
    expected_source = harness_package_root().resolve().parents[3]
    expected_import = harness_package_root().resolve().parent
    expected_bootstrap = bootstrap_path().resolve()
    observed = (
        source_root.resolve(),
        import_root.resolve(),
        bootstrap_file.resolve(),
    )
    expected = (expected_source, expected_import, expected_bootstrap)
    if (
        observed != expected
        or not sys.flags.isolated
        or not sys.dont_write_bytecode
        or Path(__file__).resolve().parent != harness_package_root().resolve()
    ):
        raise RuntimeError("external isolated scout bootstrap identity is invalid")
    _EXTERNAL_BOOTSTRAP = observed


def _require_external_bootstrap(source_root: Path) -> None:
    expected = (
        source_root.resolve(),
        harness_package_root().resolve().parent,
        bootstrap_path().resolve(),
    )
    if (
        _EXTERNAL_BOOTSTRAP != expected
        or not sys.flags.isolated
        or not sys.dont_write_bytecode
    ):
        raise RuntimeError(
            "invoke the scout through python -I -B "
            "benchmarks/split_research/g8s1_scout_bootstrap.py"
        )


def _validate_harness_source(
    root: Path,
    *,
    require_tracked: bool = True,
    verify_import_location: bool = True,
) -> dict[str, Any]:
    root = root.resolve()
    package_relative = (
        Path("benchmarks") / "split_research" / "g8s1_harness" / "g8s1_scout"
    )
    package_root = root / package_relative
    import_root = package_root.parent
    manifest_relative = (
        Path("benchmarks") / "split_research" / "G8S1_SCOUT_MANIFEST.json"
    )
    protocol_relative = (
        Path("benchmarks") / "split_research" / "TABARENA_GROUPED_SCOUT.md"
    )
    runbook_relative = Path(RUNBOOK_RELATIVE)
    manifest_file = root / manifest_relative
    protocol_file = root / protocol_relative
    runbook_file = root / runbook_relative
    if not manifest_file.is_file():
        raise RuntimeError("tracked grouped-scout harness manifest is missing")
    if (
        manifest_file.is_symlink()
        or protocol_file.is_symlink()
        or runbook_file.is_symlink()
    ):
        raise RuntimeError(
            "grouped-scout manifest/protocol/runbook must not be a symlink"
        )

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    expected_contract = _expected_manifest_contract()
    if set(manifest) != {*expected_contract, "runtime_files"}:
        raise RuntimeError("grouped-scout manifest top-level fields drifted")
    for key, expected in expected_contract.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"grouped-scout manifest field {key!r} drifted")
    if (
        not protocol_file.is_file()
        or sha256_lf_normalized_file(protocol_file) != PROTOCOL_LF_NORMALIZED_SHA256
    ):
        raise RuntimeError("tracked grouped-scout protocol hash drifted")
    if (
        not runbook_file.is_file()
        or sha256_lf_normalized_file(runbook_file) != RUNBOOK_LF_NORMALIZED_SHA256
    ):
        raise RuntimeError("tracked grouped-scout runbook hash drifted")

    prefix = package_relative.as_posix()
    expected_package_paths = {f"{prefix}/{name}" for name in RUNTIME_MODULE_FILES}
    expected_paths = expected_package_paths | {BOOTSTRAP_RELATIVE}
    runtime_files = dict(manifest.get("runtime_files", {}))
    if set(runtime_files) != expected_paths:
        raise RuntimeError("grouped-scout manifest runtime file set drifted")
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in package_root.rglob("*.py")
        if path.is_file() or path.is_symlink()
    }
    if actual_paths != expected_package_paths:
        raise RuntimeError("tracked grouped-scout runtime file set drifted")
    if {path.name for path in import_root.iterdir()} != {package_root.name}:
        raise RuntimeError(
            "tracked grouped-scout import root contains an unexpected importable file, entry, or symlink"
        )
    if {path.name for path in package_root.iterdir()} != set(RUNTIME_MODULE_FILES):
        raise RuntimeError(
            "tracked grouped-scout import root contains an unexpected importable file, entry, or symlink"
        )
    allowed_sources = {
        (root / Path(relative)).resolve() for relative in expected_package_paths
    }
    unexpected_importables = set()
    for path in import_root.rglob("*"):
        if not (path.is_file() or path.is_symlink()):
            continue
        if path.is_symlink():
            unexpected_importables.add(path.relative_to(root).as_posix())
            continue
        suffix = path.suffix.lower()
        if (
            suffix == ".py"
            and path.resolve() not in allowed_sources
            or suffix
            in {
                ".pyc",
                ".pyd",
                ".so",
            }
        ):
            unexpected_importables.add(path.relative_to(root).as_posix())
    if unexpected_importables:
        raise RuntimeError(
            "tracked grouped-scout import root contains an unexpected importable file or symlink"
        )

    for relative in sorted(expected_paths):
        path = root / Path(relative)
        expected_sha256 = runtime_files[relative]
        if not isinstance(expected_sha256, str) or not _SHA256.fullmatch(
            expected_sha256
        ):
            raise RuntimeError(
                f"grouped-scout manifest has invalid hash for {relative}"
            )
        if path.is_symlink():
            raise RuntimeError(
                f"tracked grouped-scout runtime file is a symlink: {relative}"
            )
        if not path.is_file() or sha256_lf_normalized_file(path) != expected_sha256:
            raise RuntimeError(
                f"tracked grouped-scout runtime file hash drifted: {relative}"
            )

    tracked_paths = sorted(
        {
            *expected_paths,
            manifest_relative.as_posix(),
            protocol_relative.as_posix(),
            runbook_relative.as_posix(),
        }
    )
    if require_tracked:
        output = _git(root, "ls-files", "--error-unmatch", "--", *tracked_paths)
        observed_tracked = {
            line.replace("\\", "/") for line in output.splitlines() if line
        }
        if observed_tracked != set(tracked_paths):
            raise RuntimeError(
                "grouped-scout runtime, manifest, and protocol must all be tracked"
            )
    if verify_import_location:
        _validate_current_import_location(package_root)
    _validate_stdlib_statistics()

    if verify_import_location:
        from .loader import validate_loaded_benchmark_modules

        validate_loaded_benchmark_modules(root / "benchmarks" / "tabarena")

    canonical_runtime_files = dict(sorted(runtime_files.items()))
    return {
        "manifest_sha256": sha256_file(manifest_file),
        "source_tree_sha256": hashlib.sha256(
            canonical_json_bytes(canonical_runtime_files)
        ).hexdigest(),
        "runtime_files": canonical_runtime_files,
        "runbook": {
            "path": runbook_relative.as_posix(),
            "lf_normalized_sha256": RUNBOOK_LF_NORMALIZED_SHA256,
        },
        "tracked": require_tracked,
        "imported_from_tracked_package": verify_import_location,
    }


def _validate_ctboost_source(root: Path, expected_commit: str) -> dict[str, Any]:
    if not (root / ".git").exists():
        raise RuntimeError("CTBoost source root is not a Git worktree")
    commit = _full_commit(root)
    if commit != expected_commit:
        raise RuntimeError(
            f"CTBoost source commit mismatch: expected {expected_commit}, observed {commit}"
        )
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise RuntimeError("CTBoost source worktree is not clean")
    merged = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", commit, "master"],
        check=False,
        capture_output=True,
    )
    if merged.returncode != 0:
        raise RuntimeError(
            "CTBoost scout source commit is not merged into local master"
        )
    return {"commit": commit, "clean": True, "merged_into_master": True}


def _is_link_or_junction(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(callable(is_junction) and is_junction())


def _validate_tabarena_import_root(root: Path, package_source: Path) -> Path:
    package_root = package_source / "tabarena"
    initializer = package_root / "__init__.py"
    if (
        _is_link_or_junction(root)
        or _is_link_or_junction(root / "packages")
        or _is_link_or_junction(root / "packages" / "tabarena")
        or _is_link_or_junction(package_source)
        or not package_source.is_dir()
        or _is_link_or_junction(package_root)
        or not package_root.is_dir()
        or _is_link_or_junction(initializer)
        or not initializer.is_file()
    ):
        raise RuntimeError("pinned TabArena source package is missing or linked")
    if {path.name for path in package_source.iterdir()} != {"tabarena"}:
        raise RuntimeError("pinned TabArena source import root has an unexpected entry")
    forbidden: set[str] = set()
    actual_entries: set[str] = set()
    pending = [package_source]
    while pending:
        directory = pending.pop()
        for path in directory.iterdir():
            relative = path.relative_to(package_source).as_posix()
            actual_entries.add(relative)
            linked = _is_link_or_junction(path)
            if (
                linked
                or (path.is_dir() and path.name == "__pycache__")
                or (path.is_file() and path.suffix.lower() in {".pyc", ".pyd", ".so"})
            ):
                forbidden.add(relative)
            elif path.is_dir():
                pending.append(path)
    if forbidden:
        raise RuntimeError(
            "pinned TabArena source import root contains bytecode, a native shadow, or a symlink"
        )
    tracked_prefix = "packages/tabarena/src/"
    tracked = _git(
        root,
        "ls-tree",
        "-r",
        "--name-only",
        PROTOCOL_TABARENA_COMMIT,
        "--",
        "packages/tabarena/src/tabarena",
    ).splitlines()
    tracked_files = {
        value.removeprefix(tracked_prefix)
        for value in tracked
        if value.startswith(f"{tracked_prefix}tabarena/")
    }
    tracked_directories: set[str] = {"tabarena"}
    for relative in tracked_files:
        parent = Path(relative).parent
        while parent != Path("."):
            tracked_directories.add(parent.as_posix())
            parent = parent.parent
    if not tracked_files or actual_entries != tracked_files | tracked_directories:
        raise RuntimeError("pinned TabArena source tree differs from its exact commit")
    return package_root.resolve()


def _validate_loaded_tabarena_modules(package_root: Path) -> None:
    package_root = package_root.resolve()
    observed = {
        name: module
        for name, module in sys.modules.items()
        if name == "tabarena" or name.startswith("tabarena.")
    }
    if "tabarena" not in observed:
        raise RuntimeError("pinned TabArena package was not imported")
    for name, module in observed.items():
        if not isinstance(module, ModuleType):
            raise TypeError("loaded TabArena child is not a source module")
        spec = getattr(module, "__spec__", None)
        loader = getattr(module, "__loader__", None)
        if (
            getattr(module, "__name__", None) != name
            or spec is None
            or spec.name != name
            or type(loader) is not importlib.machinery.SourceFileLoader
            or spec.loader is not loader
            or loader.name != name
        ):
            raise RuntimeError("loaded TabArena module lacks its exact source loader")
        relative_parts = name.split(".")[1:]
        candidate = package_root.joinpath(*relative_parts)
        is_package = spec.submodule_search_locations is not None
        if is_package:
            expected_package_root = candidate.resolve()
            expected_file = expected_package_root / "__init__.py"
            module_paths = [
                Path(value).resolve() for value in getattr(module, "__path__", ())
            ]
            spec_paths = [
                Path(value).resolve() for value in spec.submodule_search_locations or ()
            ]
            if (
                module_paths != [expected_package_root]
                or spec_paths != [expected_package_root]
                or getattr(module, "__package__", None) != name
            ):
                raise RuntimeError("loaded TabArena package path is not exact")
        else:
            expected_file = candidate.with_suffix(".py").resolve()
            expected_package = name.rpartition(".")[0]
            if (
                hasattr(module, "__path__")
                or getattr(module, "__package__", None) != expected_package
            ):
                raise RuntimeError("loaded TabArena child package identity is invalid")
        module_file = getattr(module, "__file__", None)
        if (
            _is_link_or_junction(expected_file)
            or not expected_file.is_file()
            or not isinstance(module_file, str)
            or Path(module_file).resolve() != expected_file
            or not isinstance(spec.origin, str)
            or Path(spec.origin).resolve() != expected_file
            or Path(loader.path).resolve() != expected_file
        ):
            raise RuntimeError("loaded TabArena module source origin is not exact")


def _validate_tabarena_source(root: Path) -> dict[str, Any]:
    if _is_link_or_junction(root):
        raise RuntimeError("TabArena source root must not be linked")
    root = root.absolute()
    if not (root / ".git").exists():
        raise RuntimeError("TabArena source root is not a Git worktree")
    commit = _full_commit(root)
    if commit != PROTOCOL_TABARENA_COMMIT:
        raise RuntimeError(
            "TabArena source commit mismatch: "
            f"expected {PROTOCOL_TABARENA_COMMIT}, observed {commit}"
        )
    status_lines = _git(
        root, "status", "--porcelain=v1", "--untracked-files=all"
    ).splitlines()
    allowed_cache = [line for line in status_lines if line.startswith("?? datasets/")]
    rejected = [line for line in status_lines if line not in allowed_cache]
    if rejected:
        raise RuntimeError("TabArena source worktree has non-cache changes")

    package_source = root / "packages" / "tabarena" / "src"
    package_root = _validate_tabarena_import_root(root, package_source)
    package_source = package_source.resolve()
    preloaded = {
        name
        for name in sys.modules
        if name == "tabarena" or name.startswith("tabarena.")
    }
    if preloaded:
        raise RuntimeError("TabArena was imported before pinned-source validation")
    for entry in sys.path:
        if entry and Path(entry).resolve() == package_source:
            raise RuntimeError(
                "pinned TabArena source was importable before physical validation"
            )
    package_spec = importlib.machinery.PathFinder.find_spec(
        "tabarena", [str(package_source)]
    )
    expected_initializer = package_root / "__init__.py"
    if (
        package_spec is None
        or package_spec.name != "tabarena"
        or type(package_spec.loader) is not importlib.machinery.SourceFileLoader
        or package_spec.loader.name != "tabarena"
        or Path(package_spec.loader.path).resolve() != expected_initializer
        or package_spec.origin is None
        or Path(package_spec.origin).resolve() != expected_initializer
        or [
            Path(value).resolve()
            for value in package_spec.submodule_search_locations or ()
        ]
        != [package_root]
    ):
        raise RuntimeError("pinned TabArena import spec is not exact source")
    try:
        sys.path.insert(0, str(package_source))
        importlib.invalidate_caches()
        package = importlib.util.module_from_spec(package_spec)
        sys.modules["tabarena"] = package
        package_spec.loader.exec_module(package)
        _validate_loaded_tabarena_modules(package_root)
    except BaseException:
        sys.path[:] = [
            entry
            for entry in sys.path
            if not entry or Path(entry).resolve() != package_source
        ]
        for name in tuple(sys.modules):
            if name == "tabarena" or name.startswith("tabarena."):
                sys.modules.pop(name, None)
        raise
    return {
        "commit": commit,
        "clean_source": True,
        "source_import_root_clean": True,
        "allowed_public_dataset_cache_entries": len(allowed_cache),
    }


def validate_loaded_tabarena_modules(tabarena_root: Path) -> None:
    """Revalidate every TabArena module loaded after provenance collection."""

    package_root = (
        tabarena_root.absolute() / "packages" / "tabarena" / "src" / "tabarena"
    )
    _validate_loaded_tabarena_modules(package_root)


def _python_package_identity(package_root: Path) -> dict[str, Any]:
    package_root = package_root.resolve()
    if not package_root.is_dir():
        raise RuntimeError("CTBoost Python package root is missing")
    paths = sorted(package_root.rglob("*.py"), key=lambda path: path.as_posix())
    if not paths or any(path.is_symlink() or not path.is_file() for path in paths):
        raise RuntimeError(
            "CTBoost Python package contains a missing or linked source file"
        )
    files = {
        path.relative_to(package_root).as_posix(): sha256_lf_normalized_file(path)
        for path in paths
    }
    return {
        "python_files": dict(sorted(files.items())),
        "python_files_sha256": hashlib.sha256(canonical_json_bytes(files)).hexdigest(),
        "python_file_count": len(files),
    }


def _bind_installed_package_to_source(
    *, source_package_root: Path, installed_package_root: Path, source_commit: str
) -> dict[str, Any]:
    source_identity = _python_package_identity(source_package_root)
    installed_identity = _python_package_identity(installed_package_root)
    if installed_identity != source_identity:
        raise RuntimeError(
            "installed CTBoost Python package does not match the expected source commit"
        )
    seal_payload = {
        "source_commit": source_commit,
        "python_files_sha256": source_identity["python_files_sha256"],
        "python_file_count": source_identity["python_file_count"],
    }
    return {
        **seal_payload,
        "source_package_seal_sha256": hashlib.sha256(
            canonical_json_bytes(seal_payload)
        ).hexdigest(),
    }


def _installed_importable_files(
    *,
    source_package_root: Path,
    installed_package_root: Path,
    source_commit: str,
    expected_native_sha256: str,
    record_sha256: dict[str, str],
) -> dict[str, Any]:
    installed_package_root = installed_package_root.resolve()
    source_binding = _bind_installed_package_to_source(
        source_package_root=source_package_root,
        installed_package_root=installed_package_root,
        source_commit=source_commit,
    )
    importables: dict[str, str] = {}
    native_candidates: list[Path] = []
    for path in installed_package_root.rglob("*"):
        if not (path.is_file() or path.is_symlink()):
            continue
        suffix = path.suffix.lower()
        if suffix not in {".py", ".pyc", ".pyd", ".so"}:
            continue
        relative = path.relative_to(installed_package_root).as_posix()
        if path.is_symlink():
            raise RuntimeError("installed CTBoost contains a linked importable file")
        if suffix == ".pyc":
            raise RuntimeError("installed CTBoost contains forbidden cached bytecode")
        if suffix in {".pyd", ".so"}:
            native_candidates.append(path)
        importables[relative] = sha256_file(path)

    if (
        len(native_candidates) != 1
        or not native_candidates[0].name.startswith("_core.")
        or native_candidates[0].parent.resolve() != installed_package_root
    ):
        raise RuntimeError(
            "installed CTBoost must contain exactly one top-level _core extension"
        )
    native_path = native_candidates[0]
    native_relative = native_path.relative_to(installed_package_root).as_posix()
    native_sha256 = importables[native_relative]
    if native_sha256 != expected_native_sha256:
        raise RuntimeError(
            "installed CTBoost native extension mismatch: "
            f"expected {expected_native_sha256}, observed {native_sha256}"
        )

    expected_importables = set(
        _python_package_identity(source_package_root)["python_files"]
    )
    expected_importables.add(native_relative)
    if set(importables) != expected_importables:
        raise RuntimeError(
            "installed CTBoost importable file set differs from release source"
        )
    if record_sha256 != importables:
        raise RuntimeError("installed CTBoost importables differ from the wheel RECORD")
    record_seal = hashlib.sha256(canonical_json_bytes(record_sha256)).hexdigest()
    return {
        "package_root": installed_package_root,
        "native_path": native_path.resolve(),
        "native_extension_sha256": native_sha256,
        "source_binding": {
            **source_binding,
            "wheel_record_sha256": record_seal,
        },
        "install_sha256": hashlib.sha256(canonical_json_bytes(importables)).hexdigest(),
    }


def _ctboost_distribution_identity() -> tuple[Any, Path, dict[str, str]]:
    if any(name == "ctboost" or name.startswith("ctboost.") for name in sys.modules):
        raise RuntimeError("CTBoost was imported before installed-package validation")
    distribution = importlib.metadata.distribution("ctboost")
    entries = list(distribution.files or ())
    init_entries = [
        entry for entry in entries if entry.as_posix() == "ctboost/__init__.py"
    ]
    if len(init_entries) != 1:
        raise RuntimeError("CTBoost wheel RECORD lacks one exact package initializer")
    init_path = Path(distribution.locate_file(init_entries[0]))
    raw_package_root = init_path.parent
    if init_path.is_symlink() or raw_package_root.is_symlink():
        raise RuntimeError("installed CTBoost package root is linked")
    package_root = raw_package_root.resolve()
    if not package_root.is_dir():
        raise RuntimeError("installed CTBoost package root is missing or linked")

    record_sha256: dict[str, str] = {}
    for entry in entries:
        normalized = entry.as_posix()
        if not normalized.startswith("ctboost/"):
            continue
        relative = normalized.removeprefix("ctboost/")
        suffix = Path(relative).suffix.lower()
        if suffix not in {".py", ".pyc", ".pyd", ".so"}:
            continue
        file_hash = entry.hash
        if file_hash is None or file_hash.mode != "sha256":
            raise RuntimeError("CTBoost wheel RECORD has an unhashed importable file")
        raw_path = Path(distribution.locate_file(entry))
        if raw_path.is_symlink():
            raise RuntimeError(
                "CTBoost wheel RECORD points to a missing or linked file"
            )
        path = raw_path.resolve()
        if not path.is_file():
            raise RuntimeError(
                "CTBoost wheel RECORD points to a missing or linked file"
            )
        if (
            not path.is_relative_to(package_root)
            or path.relative_to(package_root).as_posix() != relative
            or relative in record_sha256
        ):
            raise RuntimeError("CTBoost wheel RECORD has a duplicate or escaped path")
        observed_value = (
            base64.urlsafe_b64encode(bytes.fromhex(sha256_file(path)))
            .rstrip(b"=")
            .decode("ascii")
        )
        if observed_value != file_hash.value:
            raise RuntimeError("CTBoost wheel RECORD hash validation failed")
        record_sha256[relative] = sha256_file(path)
    return distribution, package_root, dict(sorted(record_sha256.items()))


def _validate_unloaded_ctboost_specs(package_root: Path, native_path: Path) -> None:
    package_root = package_root.resolve()
    package_spec = importlib.util.find_spec("ctboost")
    if (
        package_spec is None
        or not isinstance(package_spec.loader, importlib.machinery.SourceFileLoader)
        or package_spec.origin is None
        or Path(package_spec.origin).resolve() != package_root / "__init__.py"
        or [
            Path(value).resolve()
            for value in package_spec.submodule_search_locations or ()
        ]
        != [package_root]
    ):
        raise RuntimeError(
            "CTBoost import spec does not resolve to the sealed wheel package"
        )
    native_spec = importlib.machinery.PathFinder.find_spec(
        "ctboost._core", [str(package_root)]
    )
    if (
        native_spec is None
        or not isinstance(native_spec.loader, importlib.machinery.ExtensionFileLoader)
        or native_spec.origin is None
        or Path(native_spec.origin).resolve() != native_path.resolve()
    ):
        raise RuntimeError(
            "CTBoost native import spec does not resolve to sealed _core"
        )


def validate_installed_ctboost_before_import(
    *,
    source_root: Path,
    expected_source_commit: str,
    expected_native_sha256: str,
) -> dict[str, Any]:
    global _PREVALIDATED_INSTALLED
    source_root = source_root.resolve()
    if not _SHA1.fullmatch(expected_source_commit):
        raise RuntimeError(
            "expected CTBoost source commit must be a full lowercase SHA-1"
        )
    if not _SHA256.fullmatch(expected_native_sha256):
        raise RuntimeError("expected native hash must be a lowercase SHA-256")
    _validate_ctboost_source(source_root, expected_source_commit)
    distribution, package_root, record_sha256 = _ctboost_distribution_identity()
    if package_root.is_relative_to(source_root):
        raise RuntimeError(
            "CTBoost resolved from source instead of the installed wheel"
        )
    sealed = _installed_importable_files(
        source_package_root=source_root / "ctboost",
        installed_package_root=package_root,
        source_commit=expected_source_commit,
        expected_native_sha256=expected_native_sha256,
        record_sha256=record_sha256,
    )
    _validate_unloaded_ctboost_specs(package_root, sealed["native_path"])
    _PREVALIDATED_INSTALLED = {
        **sealed,
        "source_root": source_root,
        "source_commit": expected_source_commit,
        "version": distribution.version,
    }
    return dict(_PREVALIDATED_INSTALLED)


def _installed_ctboost_identity(
    expected_native_sha256: str, source_root: Path, expected_source_commit: str
) -> dict[str, Any]:
    expected_package_root = (
        None
        if _PREVALIDATED_INSTALLED is None
        else Path(_PREVALIDATED_INSTALLED["package_root"]).resolve()
    )
    if (
        _PREVALIDATED_INSTALLED is None
        or Path(_PREVALIDATED_INSTALLED["source_root"]).resolve()
        != source_root.resolve()
        or _PREVALIDATED_INSTALLED["source_commit"] != expected_source_commit
        or _PREVALIDATED_INSTALLED["native_extension_sha256"] != expected_native_sha256
        or expected_package_root is None
    ):
        raise RuntimeError("installed CTBoost was not sealed before package import")

    import ctboost
    from ctboost import CTBoostClassifier, CTBoostRegressor

    package_root = Path(ctboost.__file__).resolve().parent
    core = sys.modules.get("ctboost._core")
    native_path = Path(_PREVALIDATED_INSTALLED["native_path"]).resolve()
    if (
        package_root != expected_package_root
        or not isinstance(ctboost.__loader__, importlib.machinery.SourceFileLoader)
        or ctboost.__spec__ is None
        or ctboost.__spec__.loader is not ctboost.__loader__
        or _module_origin(ctboost) != expected_package_root / "__init__.py"
        or [Path(value).resolve() for value in ctboost.__path__]
        != [expected_package_root]
        or not isinstance(core, ModuleType)
        or not isinstance(core.__loader__, importlib.machinery.ExtensionFileLoader)
        or core.__spec__ is None
        or core.__spec__.loader is not core.__loader__
        or _module_origin(core) != native_path
    ):
        raise RuntimeError(
            "loaded CTBoost package/native origins differ from sealed wheel"
        )

    required = {"feature_test", "feature_test_bins", "feature_test_adjustment"}
    for estimator in (CTBoostClassifier, CTBoostRegressor):
        missing = required.difference(inspect.signature(estimator).parameters)
        if missing:
            raise RuntimeError(
                f"installed {estimator.__name__} lacks grouped-scout parameters: {sorted(missing)}"
            )

    build = dict(ctboost.build_info())
    if bool(build.get("cuda_enabled", False)):
        raise RuntimeError("CPU scout requires a CPU-only CTBoost build")
    return {
        "version": _PREVALIDATED_INSTALLED["version"],
        "install_sha256": _PREVALIDATED_INSTALLED["install_sha256"],
        "native_extension_sha256": expected_native_sha256,
        "source_binding": _PREVALIDATED_INSTALLED["source_binding"],
        "build": build,
    }


def collect_provenance(
    *,
    ctboost_root: Path,
    tabarena_root: Path,
    expected_ctboost_commit: str,
    expected_native_sha256: str,
) -> dict[str, Any]:
    ctboost_root = ctboost_root.resolve()
    _require_external_bootstrap(ctboost_root)
    expected_ctboost_commit = expected_ctboost_commit.lower()
    expected_native_sha256 = expected_native_sha256.lower()
    if not _SHA1.fullmatch(expected_ctboost_commit):
        raise RuntimeError(
            "--expected-ctboost-commit must be a full lowercase 40-character SHA-1"
        )
    if not _SHA256.fullmatch(expected_native_sha256):
        raise RuntimeError(
            "--expected-native-sha256 must be a lowercase 64-character SHA-256"
        )

    protocol = protocol_path()
    if not protocol.is_file():
        raise RuntimeError("frozen grouped-scout protocol is missing")
    expected_package = (
        ctboost_root / "benchmarks" / "split_research" / "g8s1_harness" / "g8s1_scout"
    )
    expected_manifest = (
        ctboost_root / "benchmarks" / "split_research" / "G8S1_SCOUT_MANIFEST.json"
    )
    if (
        harness_package_root().resolve() != expected_package
        or manifest_path().resolve() != expected_manifest
    ):
        raise RuntimeError(
            "imported grouped-scout harness does not belong to --expected-ctboost-commit"
        )
    ctboost_source = _validate_ctboost_source(ctboost_root, expected_ctboost_commit)
    harness_source = _validate_harness_source(ctboost_root)
    tabarena_source = _validate_tabarena_source(tabarena_root)
    installed = _installed_ctboost_identity(
        expected_native_sha256, ctboost_root, expected_ctboost_commit
    )

    versions = {}
    for name in ("tabarena", "autogluon.tabular", "numpy", "pandas", "scikit-learn"):
        versions[name] = importlib.metadata.version(name)
    provenance: dict[str, Any] = {
        "schema_version": 1,
        "protocol": {
            "name": "ctboost-grouped-statistic-tabarena-scout-v1",
            "crlf_sha256": PROTOCOL_SHA256,
            "lf_normalized_sha256": PROTOCOL_LF_NORMALIZED_SHA256,
            "working_tree_sha256": sha256_file(protocol),
            "p50_sha256": P50_SHA256,
            "random_p200_sha256": P200_RANDOM_SHA256,
            "p201_sha256": P201_SHA256,
        },
        "ctboost_source": ctboost_source,
        "harness_source": harness_source,
        "tabarena_source": tabarena_source,
        "installed_ctboost": installed,
        "runtime": {
            "python": ".".join(str(part) for part in sys.version_info[:3]),
            "package_versions": versions,
        },
        "resources": {
            "device": "cpu",
            "num_cpus": NUM_CPUS,
            "num_gpus": NUM_GPUS,
            "memory_limit_gb": MEMORY_LIMIT_GB,
            "time_limit_seconds": TIME_LIMIT_SECONDS,
            "histogram_threads": int(HISTOGRAM_THREADS),
            "ray": False,
            "shard_count": 1,
            "job_batch_size": JOB_BATCH_SIZE,
        },
    }
    provenance["identity_sha256"] = hashlib.sha256(
        canonical_json_bytes(provenance)
    ).hexdigest()
    return provenance


def write_or_validate_provenance(path: Path, provenance: dict[str, Any]) -> None:
    encoded = json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != provenance:
            raise RuntimeError(
                "sealed scout provenance differs from the current runtime identity"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8", newline="\n")
