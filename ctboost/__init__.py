"""Public Python API for CTBoost."""

from typing import Any, Dict

from ._core import build_info as _native_build_info
from ._version import __version__
from .core import Pool
from .sklearn import CBoostClassifier, CBoostRegressor, CTBoostClassifier, CTBoostRegressor
from .training import Booster, cv, load_model, train


def build_info() -> Dict[str, Any]:
    info = dict(_native_build_info())
    info["version"] = __version__
    return info

__all__ = [
    "__version__",
    "Booster",
    "CBoostClassifier",
    "CBoostRegressor",
    "CTBoostClassifier",
    "CTBoostRegressor",
    "Pool",
    "build_info",
    "cv",
    "load_model",
    "train",
]
