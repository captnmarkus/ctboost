"""Public Python API for CTBoost."""

from typing import Any, Dict

from ._core import build_info as _native_build_info
from ._version import __version__
from .core import Pool
from .feature_pipeline import FeaturePipeline
from .prepared_data import prepare_pool
from .training import Booster, cv, load_model, train

_SKLEARN_EXPORT_NAMES = {
    "CBoostClassifier",
    "CBoostRanker",
    "CBoostRegressor",
    "CTBoostClassifier",
    "CTBoostRanker",
    "CTBoostRegressor",
}


def _load_sklearn_exports() -> Dict[str, Any]:
    try:
        from .sklearn import (
            CBoostClassifier,
            CBoostRanker,
            CBoostRegressor,
            CTBoostClassifier,
            CTBoostRanker,
            CTBoostRegressor,
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
        "CTBoostRanker": CTBoostRanker,
        "CTBoostRegressor": CTBoostRegressor,
    }
    globals().update(exports)
    return exports


def build_info() -> Dict[str, Any]:
    info = dict(_native_build_info())
    info["version"] = __version__
    return info


def __getattr__(name: str) -> Any:
    if name in _SKLEARN_EXPORT_NAMES:
        return _load_sklearn_exports()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> Any:
    return sorted(set(globals()) | _SKLEARN_EXPORT_NAMES)

__all__ = [
    "__version__",
    "Booster",
    "CBoostClassifier",
    "CBoostRanker",
    "CBoostRegressor",
    "CTBoostClassifier",
    "CTBoostRanker",
    "CTBoostRegressor",
    "FeaturePipeline",
    "Pool",
    "prepare_pool",
    "build_info",
    "cv",
    "load_model",
    "train",
]
