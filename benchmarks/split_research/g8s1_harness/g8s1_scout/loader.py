"""Load sealed benchmark sources in a private, cache-free module namespace.

The repository's ordinary ``benchmarks.tabarena`` modules may already be
imported by an unrelated test or tool.  The scout therefore never owns or
mutates that public namespace.  It compiles the exact tracked adapter sources
under a private synthetic package, which also prevents Python from consulting
an adjacent bytecode cache.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

from .constants import source_root

_PRIVATE_PACKAGE = "_g8s1_scout_sealed_benchmark"
_ALLOWED_MODULES = frozenset({"ctboost_model", "run"})
_OWNED_PACKAGE: types.ModuleType | None = None
_LOADED_MODULES: dict[str, types.ModuleType] = {}


def _private_module_name(module_name: str) -> str:
    if module_name not in _ALLOWED_MODULES:
        raise RuntimeError(
            f"benchmark module is outside the sealed allowlist: {module_name}"
        )
    return f"{_PRIVATE_PACKAGE}.{module_name}"


def _validate_private_package() -> None:
    package = _OWNED_PACKAGE
    spec = None if package is None else getattr(package, "__spec__", None)
    if (
        package is None
        or sys.modules.get(_PRIVATE_PACKAGE) is not package
        or not isinstance(package, types.ModuleType)
        or getattr(package, "__name__", None) != _PRIVATE_PACKAGE
        or getattr(package, "__package__", None) != _PRIVATE_PACKAGE
        or not hasattr(package, "__path__")
        or list(package.__path__) != []
        or getattr(package, "__loader__", None) is not None
        or spec is None
        or spec.name != _PRIVATE_PACKAGE
        or spec.loader is not None
        or spec.submodule_search_locations is None
        or list(spec.submodule_search_locations) != []
    ):
        raise RuntimeError("sealed benchmark private package identity was replaced")
    for leaf in _ALLOWED_MODULES:
        expected = _LOADED_MODULES.get(f"{_PRIVATE_PACKAGE}.{leaf}")
        if expected is None:
            if hasattr(package, leaf):
                raise RuntimeError(
                    "sealed benchmark private package has an unexpected child attribute"
                )
        elif getattr(package, leaf, None) is not expected:
            raise RuntimeError(
                "sealed benchmark private package child identity was replaced"
            )


def _ensure_private_package() -> types.ModuleType:
    global _OWNED_PACKAGE
    existing = sys.modules.get(_PRIVATE_PACKAGE)
    if _OWNED_PACKAGE is not None:
        _validate_private_package()
        return _OWNED_PACKAGE
    if existing is not None:
        raise RuntimeError("refusing a preloaded sealed benchmark private package")
    package = types.ModuleType(_PRIVATE_PACKAGE)
    package.__package__ = _PRIVATE_PACKAGE
    package.__path__ = []  # type: ignore[attr-defined]
    package.__loader__ = None
    spec = importlib.machinery.ModuleSpec(
        _PRIVATE_PACKAGE,
        loader=None,
        is_package=True,
    )
    spec.submodule_search_locations = []
    package.__spec__ = spec
    sys.modules[_PRIVATE_PACKAGE] = package
    _OWNED_PACKAGE = package
    _validate_private_package()
    return package


def _validate_loaded_module(name: str, module: Any, expected_file: Path) -> None:
    expected_file = expected_file.resolve()
    spec = getattr(module, "__spec__", None)
    loader = getattr(module, "__loader__", None)
    if (
        not isinstance(module, types.ModuleType)
        or module is not _LOADED_MODULES.get(name)
        or sys.modules.get(name) is not module
        or getattr(module, "__name__", None) != name
        or getattr(module, "__package__", None) != _PRIVATE_PACKAGE
        or not isinstance(getattr(module, "__file__", None), str)
        or Path(module.__file__).resolve() != expected_file
        or spec is None
        or spec.name != name
        or not isinstance(spec.origin, str)
        or Path(spec.origin).resolve() != expected_file
        or type(loader) is not importlib.machinery.SourceFileLoader
        or spec.loader is not loader
        or loader.name != name
        or Path(loader.path).resolve() != expected_file
        or hasattr(module, "__path__")
    ):
        raise RuntimeError(f"sealed benchmark module {name!r} has an invalid identity")


def validate_loaded_benchmark_modules(tabarena_root: Path) -> None:
    expected_root = (source_root() / "benchmarks" / "tabarena").resolve()
    if tabarena_root.resolve() != expected_root:
        raise RuntimeError("benchmark validation root differs from tracked source")
    observed = {
        name
        for name in sys.modules
        if name == _PRIVATE_PACKAGE or name.startswith(f"{_PRIVATE_PACKAGE}.")
    }
    expected = set(_LOADED_MODULES)
    if _OWNED_PACKAGE is not None:
        expected.add(_PRIVATE_PACKAGE)
    if observed != expected:
        raise RuntimeError("unexpected or preloaded sealed benchmark module detected")
    if _OWNED_PACKAGE is not None:
        _validate_private_package()
    for name, module in _LOADED_MODULES.items():
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in _ALLOWED_MODULES or name != _private_module_name(leaf):
            raise RuntimeError(
                "loaded benchmark module is outside the sealed allowlist"
            )
        _validate_loaded_module(name, module, expected_root / f"{leaf}.py")


def load_benchmark_module(module_name: str) -> Any:
    full_name = _private_module_name(module_name)
    root = source_root()
    tabarena_root = root / "benchmarks" / "tabarena"
    expected_file = tabarena_root / f"{module_name}.py"
    if not expected_file.is_file() or expected_file.is_symlink():
        raise RuntimeError(f"tracked benchmark module is missing: {module_name}")
    if full_name in _LOADED_MODULES:
        validate_loaded_benchmark_modules(tabarena_root)
        return _LOADED_MODULES[full_name]
    if full_name in sys.modules:
        raise RuntimeError(f"refusing preloaded sealed benchmark module {full_name!r}")
    validate_loaded_benchmark_modules(tabarena_root)
    package = _ensure_private_package()
    if module_name == "run":
        load_benchmark_module("ctboost_model")

    resolved_file = expected_file.resolve()
    source_loader = importlib.machinery.SourceFileLoader(full_name, str(resolved_file))
    spec = importlib.util.spec_from_loader(
        full_name,
        source_loader,
        origin=str(resolved_file),
    )
    if spec is None:
        raise RuntimeError(f"could not create benchmark module spec: {module_name}")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = _PRIVATE_PACKAGE
    module.__loader__ = source_loader
    module.__file__ = str(resolved_file)
    sys.modules[full_name] = module
    _LOADED_MODULES[full_name] = module
    setattr(package, module_name, module)
    try:
        code = compile(
            expected_file.read_bytes(),
            str(resolved_file),
            "exec",
            dont_inherit=True,
        )
        exec(code, module.__dict__)  # noqa: S102 - execute hash-sealed tracked source
        _validate_loaded_module(full_name, module, expected_file)
        _validate_private_package()
    except BaseException:
        sys.modules.pop(full_name, None)
        _LOADED_MODULES.pop(full_name, None)
        if getattr(package, module_name, None) is module:
            delattr(package, module_name)
        raise
    return module
