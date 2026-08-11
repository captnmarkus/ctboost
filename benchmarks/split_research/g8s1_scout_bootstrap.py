"""Isolated, source-only bootstrap for the frozen grouped-statistic scout.

Set ``PYTHONDONTWRITEBYTECODE=1`` and invoke this file directly with
``python -I -B``. It validates the complete harness import root before making
that root importable, so cached bytecode and native-module shadows cannot
execute ahead of the package's own checks. Candidate wheels must be installed
pristine with ``pip install --no-compile``; see ``G8S1_SCOUT_RUNBOOK.md``.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import NoReturn

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
_IMPORT_ROOT = (
    _SOURCE_ROOT / "benchmarks" / "split_research" / "g8s1_harness"
).resolve()
_PACKAGE_ROOT = (_IMPORT_ROOT / "g8s1_scout").resolve()
_BOOTSTRAP_RELATIVE = "benchmarks/split_research/g8s1_scout_bootstrap.py"
_RUNBOOK_RELATIVE = "benchmarks/split_research/G8S1_SCOUT_RUNBOOK.md"
_MANIFEST = _SOURCE_ROOT / "benchmarks" / "split_research" / "G8S1_SCOUT_MANIFEST.json"
_RUNBOOK = _SOURCE_ROOT / _RUNBOOK_RELATIVE
_ADAPTER = _SOURCE_ROOT / "benchmarks" / "tabarena" / "ctboost_model.py"

# Updated only with reviewed source changes.  These constants are the trust
# anchor used before any g8s1_scout module is imported.
_EXPECTED_RUNTIME_HASHES = {
    "__init__.py": "52c4a1b7167d24027e1b301423f84d7d5656290c79b447e020cfce9b5c6b34a1",
    "__main__.py": "01f5b72fa7c1e25c30001eaf3725aa2bb4f4d6a522da656ca4464f6b52d4e780",
    "constants.py": "b9fa89ada11842870e939cdd0e5b2a86cbf70c81d6500065f3ad05f9c8b81757",
    "identity.py": "082fb28a353c8b0b59444ec4c889b80ba2e045c953a5e5eceac9a6d782b02d6f",
    "loader.py": "c00e18a318ef3f3ff6765fe003dedcd6013c464c76d65581c62a279689567efd",
    "models.py": "3a1f0833cda78e526cd9526e50034c44d14f7a2562db33d81d285610d6f2111d",
    "schedule.py": "f3128dff07e825c81d4abbb50e52cbe1b03f726e5c06600b50244aab61bc3a97",
    "summary.py": "84bcfb3d33e06b4409a702d93b4f6628bfe33c37c8ffc44a02f1fee394b9a985",
}
_EXPECTED_ADAPTER_SHA256 = (
    "c5e1edccd155f70cad52e9fd514e01233f074c4db4e982363363c317c4db90d8"
)
_EXPECTED_RUNBOOK_SHA256 = (
    "e393765efa18cf031a0bdc60ebf6475cd3e79a93b363757cb3679dae89d26e45"
)


def _abort(message: str) -> NoReturn:
    raise RuntimeError(message)


def _lf_sha256(path: Path) -> str:
    payload = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def _validate_invocation() -> None:
    if not sys.flags.isolated or not sys.dont_write_bytecode:
        _abort(
            "invoke the scout as: python -I -B "
            "benchmarks/split_research/g8s1_scout_bootstrap.py"
        )
    forbidden_roots = {_SOURCE_ROOT.resolve(), _IMPORT_ROOT}
    for entry in sys.path:
        if not entry:
            _abort("isolated scout bootstrap received an empty sys.path entry")
        resolved = Path(entry).resolve()
        if any(
            resolved == root or resolved.is_relative_to(root)
            for root in forbidden_roots
        ):
            _abort("source checkout was importable before bootstrap validation")
    protected_roots = ("g8s1_scout", "ctboost", "benchmarks", "tabarena")
    preloaded = {
        name
        for name in sys.modules
        if any(name == root or name.startswith(f"{root}.") for root in protected_roots)
    }
    if preloaded:
        _abort("scout, CTBoost, or benchmark modules were preloaded before validation")


def _validate_source_only_import_root() -> None:
    if (
        Path(__file__).is_symlink()
        or _MANIFEST.is_symlink()
        or not _MANIFEST.is_file()
        or _RUNBOOK.is_symlink()
        or not _RUNBOOK.is_file()
    ):
        _abort("grouped-scout bootstrap/manifest/runbook is missing or linked")
    if (
        _IMPORT_ROOT.is_symlink()
        or not _IMPORT_ROOT.is_dir()
        or _PACKAGE_ROOT.is_symlink()
        or not _PACKAGE_ROOT.is_dir()
    ):
        _abort("grouped-scout import root is missing, linked, or not a directory")
    expected_sources = {
        (_PACKAGE_ROOT / name).resolve() for name in _EXPECTED_RUNTIME_HASHES
    }
    if {path.name for path in _IMPORT_ROOT.iterdir()} != {_PACKAGE_ROOT.name}:
        _abort("grouped-scout import root is not the exact source-only runtime")
    if {path.name for path in _PACKAGE_ROOT.iterdir()} != set(_EXPECTED_RUNTIME_HASHES):
        _abort("grouped-scout package is not the exact source-only runtime")
    actual_sources: set[Path] = set()
    forbidden_importables: set[Path] = set()
    for path in _IMPORT_ROOT.rglob("*"):
        if path.is_symlink():
            forbidden_importables.add(path)
            continue
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".py":
            actual_sources.add(path.resolve())
        elif suffix in {".pyc", ".pyd", ".so"}:
            forbidden_importables.add(path)
    if actual_sources != expected_sources or forbidden_importables:
        _abort("grouped-scout import root is not the exact source-only runtime")
    for name, expected in _EXPECTED_RUNTIME_HASHES.items():
        if _lf_sha256(_PACKAGE_ROOT / name) != expected:
            _abort(f"grouped-scout bootstrap source hash drifted: {name}")

    if (
        _ADAPTER.is_symlink()
        or not _ADAPTER.is_file()
        or _lf_sha256(_ADAPTER) != _EXPECTED_ADAPTER_SHA256
    ):
        _abort("tracked CTBoost TabArena adapter hash drifted")
    if _lf_sha256(_RUNBOOK) != _EXPECTED_RUNBOOK_SHA256:
        _abort("tracked grouped-scout runbook hash drifted")

    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("runbook") != {
        "path": _RUNBOOK_RELATIVE,
        "lf_normalized_sha256": _EXPECTED_RUNBOOK_SHA256,
    }:
        _abort("grouped-scout manifest runbook identity drifted")
    runtime_files = dict(manifest.get("runtime_files", {}))
    prefix = "benchmarks/split_research/g8s1_harness/g8s1_scout"
    expected_manifest_paths = {
        f"{prefix}/{name}" for name in _EXPECTED_RUNTIME_HASHES
    } | {_BOOTSTRAP_RELATIVE}
    if set(runtime_files) != expected_manifest_paths:
        _abort("grouped-scout manifest does not enumerate the exact bootstrap runtime")
    for name, expected in _EXPECTED_RUNTIME_HASHES.items():
        if runtime_files[f"{prefix}/{name}"] != expected:
            _abort(f"grouped-scout manifest hash differs from bootstrap: {name}")
    if runtime_files[_BOOTSTRAP_RELATIVE] != _lf_sha256(Path(__file__).resolve()):
        _abort("grouped-scout bootstrap hash differs from its manifest")


def _argument(name: str) -> str:
    try:
        index = sys.argv.index(name)
        value = sys.argv[index + 1]
    except (IndexError, ValueError) as error:
        raise RuntimeError(f"canonical scout invocation requires {name}") from error
    if value.startswith("--"):
        raise RuntimeError(f"canonical scout invocation has no value for {name}")
    return value


def main() -> int:
    _validate_invocation()
    _validate_source_only_import_root()
    sys.path.insert(0, str(_IMPORT_ROOT))

    from g8s1_scout import identity

    identity.reset_and_seal_stdlib_statistics()
    identity.mark_external_bootstrap_validated(
        source_root=_SOURCE_ROOT,
        import_root=_IMPORT_ROOT,
        bootstrap_file=Path(__file__).resolve(),
    )
    if "--help" not in sys.argv and "-h" not in sys.argv:
        expected_commit = _argument("--expected-ctboost-commit").lower()
        expected_native = _argument("--expected-native-sha256").lower()
        identity.validate_installed_ctboost_before_import(
            source_root=_SOURCE_ROOT,
            expected_source_commit=expected_commit,
            expected_native_sha256=expected_native,
        )

    from g8s1_scout.__main__ import main as scout_main

    return int(scout_main())


if __name__ == "__main__":
    raise SystemExit(main())
