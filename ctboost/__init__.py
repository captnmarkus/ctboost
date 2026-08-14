"""Public Python API for CTBoost."""

import importlib.machinery
import importlib.util
import pathlib
import sys
from typing import Any, Dict


def _load_colocated_native_extension() -> None:
    module_name = f"{__name__}._core"
    if module_name in sys.modules:
        return

    package_dir = pathlib.Path(__file__).resolve().parent
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        candidate = package_dir / f"_core{suffix}"
        if not candidate.is_file():
            continue
        spec = importlib.util.spec_from_file_location(module_name, candidate)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        return


_load_colocated_native_extension()
from ._core import build_info as _native_build_info
from ._version import __version__

del _load_colocated_native_extension


def _ensure_split_package(name: str) -> None:
    full_name = f"{__name__}.{name}"
    if full_name in sys.modules:
        return
    package_dir = pathlib.Path(__file__).with_name(name)
    init_py = package_dir / "__init__.py"
    if not init_py.is_file():
        return
    spec = importlib.util.spec_from_file_location(
        full_name,
        init_py,
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)


# Keep the sklearn layer lazy so plain `import ctboost` works without the
# optional scikit-learn stack installed.
for _split_package in ("core", "distributed", "training"):
    _ensure_split_package(_split_package)
del _split_package

_CORE_EXPORT_NAMES = {
    "Booster",
    "EvalMetricSpec",
    "ObjectiveSpec",
    "FeaturePipeline",
    "Pool",
    "PoolBatch",
    "PreparedTrainingData",
    "TrainingCallbackEnv",
    "checkpoint_callback",
    "prepare_pool",
    "pool_from_batches",
    "prepare_training_data",
    "cv",
    "log_evaluation",
    "load_exported_predictor",
    "load_inference_manifest",
    "load_model",
    "make_eval_metric",
    "make_objective",
    "train",
    "validate_inference_manifest",
}

_SKLEARN_EXPORT_NAMES = {
    "CBoostClassifier",
    "CBoostRanker",
    "CBoostRegressor",
    "CTBoostClassifier",
    "CTBoostAFTRegressor",
    "CTBoostAFTSurvivalRegressor",
    "CTBoostMultiLabelClassifier",
    "CTBoostMultiOutputRegressor",
    "CTBoostRanker",
    "CTBoostRegressor",
    "compare_estimators",
}


def _load_core_exports() -> Dict[str, Any]:
    from ._export import load_exported_predictor
    from .core import Pool
    from .feature_pipeline import FeaturePipeline
    from .inference_manifest import load_inference_manifest, validate_inference_manifest
    from .prepared_data import prepare_pool
    from .streaming import PoolBatch, pool_from_batches
    from .training import (
        Booster,
        EvalMetricSpec,
        ObjectiveSpec,
        PreparedTrainingData,
        TrainingCallbackEnv,
        checkpoint_callback,
        cv,
        load_model,
        log_evaluation,
        make_eval_metric,
        make_objective,
        prepare_training_data,
        train,
    )

    exports = {
        "Booster": Booster,
        "EvalMetricSpec": EvalMetricSpec,
        "ObjectiveSpec": ObjectiveSpec,
        "FeaturePipeline": FeaturePipeline,
        "Pool": Pool,
        "PoolBatch": PoolBatch,
        "PreparedTrainingData": PreparedTrainingData,
        "TrainingCallbackEnv": TrainingCallbackEnv,
        "checkpoint_callback": checkpoint_callback,
        "prepare_pool": prepare_pool,
        "pool_from_batches": pool_from_batches,
        "prepare_training_data": prepare_training_data,
        "cv": cv,
        "load_exported_predictor": load_exported_predictor,
        "load_inference_manifest": load_inference_manifest,
        "log_evaluation": log_evaluation,
        "load_model": load_model,
        "make_eval_metric": make_eval_metric,
        "make_objective": make_objective,
        "train": train,
        "validate_inference_manifest": validate_inference_manifest,
    }
    globals().update(exports)
    return exports


def _load_sklearn_exports() -> Dict[str, Any]:
    try:
        _ensure_split_package("sklearn")
        from .sklearn import (
            CBoostClassifier,
            CBoostRanker,
            CBoostRegressor,
            CTBoostClassifier,
            CTBoostAFTRegressor,
            CTBoostAFTSurvivalRegressor,
            CTBoostMultiLabelClassifier,
            CTBoostMultiOutputRegressor,
            CTBoostRanker,
            CTBoostRegressor,
            compare_estimators,
        )
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("sklearn"):
            raise ImportError(
                "ctboost scikit-learn estimators require scikit-learn>=1.3. "
                "Install 'ctboost[sklearn]' or add scikit-learn to your environment."
            ) from exc
        raise

    exports = {
        "CBoostClassifier": CBoostClassifier,
        "CBoostRanker": CBoostRanker,
        "CBoostRegressor": CBoostRegressor,
        "CTBoostClassifier": CTBoostClassifier,
        "CTBoostAFTRegressor": CTBoostAFTRegressor,
        "CTBoostAFTSurvivalRegressor": CTBoostAFTSurvivalRegressor,
        "CTBoostMultiLabelClassifier": CTBoostMultiLabelClassifier,
        "CTBoostMultiOutputRegressor": CTBoostMultiOutputRegressor,
        "CTBoostRanker": CTBoostRanker,
        "CTBoostRegressor": CTBoostRegressor,
        "compare_estimators": compare_estimators,
    }
    globals().update(exports)
    return exports


def build_info() -> Dict[str, Any]:
    info = dict(_native_build_info())
    info["package_version"] = __version__
    return info


def __getattr__(name: str) -> Any:
    if name == "_core":
        module = __import__(f"{__name__}._core", fromlist=["_core"])
        globals()["_core"] = module
        return module
    if name in _CORE_EXPORT_NAMES:
        return _load_core_exports()[name]
    if name in _SKLEARN_EXPORT_NAMES:
        return _load_sklearn_exports()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> Any:
    return sorted(set(globals()) | _CORE_EXPORT_NAMES | _SKLEARN_EXPORT_NAMES | {"_core"})

__all__ = [
    "__version__",
    "Booster",
    "CBoostClassifier",
    "CBoostRanker",
    "CBoostRegressor",
    "CTBoostClassifier",
    "CTBoostAFTRegressor",
    "CTBoostAFTSurvivalRegressor",
    "CTBoostMultiLabelClassifier",
    "CTBoostMultiOutputRegressor",
    "CTBoostRanker",
    "CTBoostRegressor",
    "compare_estimators",
    "EvalMetricSpec",
    "ObjectiveSpec",
    "FeaturePipeline",
    "Pool",
    "PoolBatch",
    "PreparedTrainingData",
    "TrainingCallbackEnv",
    "prepare_pool",
    "pool_from_batches",
    "prepare_training_data",
    "build_info",
    "checkpoint_callback",
    "cv",
    "load_exported_predictor",
    "load_inference_manifest",
    "load_model",
    "log_evaluation",
    "make_eval_metric",
    "make_objective",
    "train",
    "validate_inference_manifest",
]
