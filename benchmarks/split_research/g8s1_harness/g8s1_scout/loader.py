"""Load the tracked benchmark adapter without putting its checkout on sys.path.

Keeping the CTBoost checkout off ``sys.path`` is important in the Python 3.12
TabArena environment: fits must import the installed candidate wheel, not the
source tree (which may contain a native extension for another Python ABI).
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

from .constants import source_root

_OWNED_PACKAGES: dict[str, types.ModuleType] = {}
_LOADED_MODULES: dict[str, types.ModuleType] = {}


def _ensure_package(name: str, path: Path) -> None:
    expected_path = path.resolve()
    expected_file = expected_path / "__init__.py"
    existing = sys.modules.get(name)
    if existing is not None:
        if existing is not _OWNED_PACKAGES.get(name):
            raise RuntimeError(f"refusing preloaded benchmark package {name!r}")
        paths = [Path(item).resolve() for item in getattr(existing, "__path__", ())]
        package_file = getattr(existing, "__file__", None)
        if (
            paths != [expected_path]
            or not isinstance(package_file, str)
            or Path(package_file).resolve() != expected_file
        ):
            raise RuntimeError(
                f"refusing conflicting imported package {name!r}; expected package path {path.name!r}"
            )
        return
    package = types.ModuleType(name)
    package.__package__ = name
    package.__path__ = [str(expected_path)]  # type: ignore[attr-defined]
    package.__file__ = str(expected_file)
    sys.modules[name] = package
    _OWNED_PACKAGES[name] = package


def _validate_loaded_module(name: str, module: Any, expected_file: Path) -> None:
    expected_file = expected_file.resolve()
    spec = getattr(module, "__spec__", None)
    loader = getattr(module, "__loader__", None)
    if (
        not isinstance(module, types.ModuleType)
        or module is not _LOADED_MODULES.get(name)
        or sys.modules.get(name) is not module
        or not isinstance(getattr(module, "__file__", None), str)
        or Path(module.__file__).resolve() != expected_file
        or spec is None
        or not isinstance(spec.origin, str)
        or Path(spec.origin).resolve() != expected_file
        or not isinstance(loader, importlib.machinery.SourceFileLoader)
        or spec.loader is not loader
        or Path(loader.path).resolve() != expected_file
        or hasattr(module, "__path__")
    ):
        raise RuntimeError(f"benchmark child module {name!r} has an invalid origin")


def validate_loaded_benchmark_modules(tabarena_root: Path) -> None:
    expected_root = (source_root() / "benchmarks" / "tabarena").resolve()
    if tabarena_root.resolve() != expected_root:
        raise RuntimeError("benchmark validation root differs from tracked source")
    observed = {
        name
        for name in sys.modules
        if name == "benchmarks" or name.startswith("benchmarks.")
    }
    expected = set(_OWNED_PACKAGES) | set(_LOADED_MODULES)
    if observed != expected:
        raise RuntimeError("unexpected or preloaded benchmark child module detected")
    for name, module in _OWNED_PACKAGES.items():
        expected_path = (
            source_root() / "benchmarks" if name == "benchmarks" else expected_root
        ).resolve()
        if (
            sys.modules.get(name) is not module
            or [Path(value).resolve() for value in module.__path__] != [expected_path]
            or Path(module.__file__).resolve() != expected_path / "__init__.py"
        ):
            raise RuntimeError("owned benchmark package identity was replaced")
    for name, module in _LOADED_MODULES.items():
        leaf = name.rsplit(".", 1)[-1]
        _validate_loaded_module(name, module, expected_root / f"{leaf}.py")


def load_benchmark_module(module_name: str) -> Any:
    root = source_root()
    benchmark_root = root / "benchmarks"
    tabarena_root = benchmark_root / "tabarena"
    expected_file = tabarena_root / f"{module_name}.py"
    if not expected_file.is_file() or expected_file.is_symlink():
        raise RuntimeError(f"tracked benchmark module is missing: {module_name}")
    full_name = f"benchmarks.tabarena.{module_name}"
    if full_name in sys.modules or full_name in _LOADED_MODULES:
        raise RuntimeError(f"refusing preloaded benchmark child module {full_name!r}")
    validate_loaded_benchmark_modules(tabarena_root)
    _ensure_package("benchmarks", benchmark_root)
    _ensure_package("benchmarks.tabarena", tabarena_root)
    loader = importlib.machinery.SourceFileLoader(
        full_name, str(expected_file.resolve())
    )
    spec = importlib.util.spec_from_loader(
        full_name, loader, origin=str(expected_file.resolve())
    )
    if spec is None:
        raise RuntimeError(f"could not create benchmark module spec: {module_name}")
    module = importlib.util.module_from_spec(spec)
    module.__loader__ = loader
    module.__file__ = str(expected_file.resolve())
    sys.modules[full_name] = module
    _LOADED_MODULES[full_name] = module
    try:
        code = compile(
            expected_file.read_bytes(),
            str(expected_file.resolve()),
            "exec",
            dont_inherit=True,
        )
        exec(code, module.__dict__)  # noqa: S102 - execute hash-sealed tracked source
        _validate_loaded_module(full_name, module, expected_file)
    except Exception:
        sys.modules.pop(full_name, None)
        _LOADED_MODULES.pop(full_name, None)
        raise
    return module
