import os
from pathlib import Path
import socket
import subprocess
import sys
import textwrap
import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
import ctboost

from tests.helpers import make_classification_data as _make_classification_data

def test_classifier_predict_proba_and_feature_importances():
    X, y = _make_classification_data()

    clf = ctboost.CTBoostClassifier(
        iterations=18,
        learning_rate=0.2,
        max_depth=2,
        alpha=1.0,
        lambda_l2=1.0,
    )
    clf.fit(X, y)

    probabilities = clf.predict_proba(X)
    predictions = clf.predict(X)
    importances = clf.feature_importances_

    assert probabilities.shape == (X.shape[0], 2)
    assert predictions.shape == (X.shape[0],)
    assert importances.shape == (X.shape[1],)
    assert np.all(importances >= 0.0)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(importances.sum(), 100.0, rtol=1e-5, atol=1e-5)

def test_regressor_feature_importances_match_feature_count():
    X, y = make_regression(
        n_samples=128,
        n_features=6,
        n_informative=4,
        noise=0.1,
        random_state=21,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    reg = ctboost.CTBoostRegressor(
        iterations=12,
        learning_rate=0.2,
        max_depth=2,
        alpha=1.0,
        lambda_l2=1.0,
    )
    reg.fit(X, y)

    assert reg.predict(X).shape == (X.shape[0],)
    assert reg.feature_importances_.shape == (X.shape[1],)

def test_regressor_accepts_learning_rate_schedule():
    X, y = make_regression(
        n_samples=128,
        n_features=6,
        n_informative=4,
        noise=0.1,
        random_state=27,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    schedule = [0.2, 0.2, 0.1, 0.1, 0.05, 0.05]

    reg = ctboost.CTBoostRegressor(
        iterations=len(schedule),
        learning_rate=schedule[0],
        max_depth=2,
        alpha=1.0,
        lambda_l2=1.0,
        random_seed=17,
    )
    reg.fit(X, y, learning_rate_schedule=schedule)

    assert reg._booster.learning_rate_history == pytest.approx(schedule)
    assert reg.predict(X).shape == (X.shape[0],)

def test_regressor_early_stopping_uses_eval_set():
    X, y = make_regression(
        n_samples=256,
        n_features=8,
        n_informative=5,
        noise=20.0,
        random_state=29,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    X_train, X_val = X[:96], X[96:]
    y_train = y[:96]
    y_val = -y[96:]

    reg = ctboost.CTBoostRegressor(
        iterations=500,
        learning_rate=0.3,
        max_depth=4,
        alpha=1.0,
        lambda_l2=0.5,
    )
    reg.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=15,
    )

    predictions = reg.predict(X_val)

    assert reg.best_iteration_ + 1 < 500
    assert len(reg._booster.loss_history) == reg.best_iteration_ + 1
    assert predictions.shape == (X_val.shape[0],)
