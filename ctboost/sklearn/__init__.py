"""scikit-learn compatible estimator shells for CTBoost."""

from .classifier import CTBoostClassifier
from .ranker import CTBoostRanker
from .regressor import CTBoostRegressor

CBoostClassifier = CTBoostClassifier
CBoostRegressor = CTBoostRegressor
CBoostRanker = CTBoostRanker

__all__ = [
    "CBoostClassifier",
    "CBoostRanker",
    "CBoostRegressor",
    "CTBoostClassifier",
    "CTBoostRanker",
    "CTBoostRegressor",
]
