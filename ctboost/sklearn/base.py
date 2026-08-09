"""Base estimator shell for ctboost.sklearn."""

from sklearn.base import BaseEstimator

from ._fit import _BaseFitMixin
from ._params import _BaseInitMixin
from ._predict import _BasePredictionMixin
from .model_selection import _ModelSelectionMixin
from .serialization import _SerializationMixin


class _BaseCTBoost(
    _BaseInitMixin,
    _BaseFitMixin,
    _BasePredictionMixin,
    _ModelSelectionMixin,
    _SerializationMixin,
    BaseEstimator,
):
    pass
