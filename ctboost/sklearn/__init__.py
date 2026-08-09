"""scikit-learn compatible estimator shells for CTBoost."""

from .classifier import CTBoostClassifier
from .model_selection import compare_estimators
from .aft import CTBoostAFTRegressor, CTBoostAFTSurvivalRegressor
from .multioutput import CTBoostMultiLabelClassifier, CTBoostMultiOutputRegressor
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
    "CTBoostAFTRegressor",
    "CTBoostAFTSurvivalRegressor",
    "CTBoostMultiLabelClassifier",
    "CTBoostMultiOutputRegressor",
    "CTBoostRanker",
    "CTBoostRegressor",
    "compare_estimators",
]
