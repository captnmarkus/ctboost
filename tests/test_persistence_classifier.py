import importlib.util
import json
from pathlib import Path
import os
import socket
import subprocess
import sys
import threading
import time
import textwrap
import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
import ctboost
import ctboost._core as _core
from ctboost.distributed import (
    DistributedCollectiveServer,
    distributed_tcp_request,
)

def test_classifier_save_load_and_staged_predict_proba(tmp_path: Path):
    X, y = make_classification(
        n_samples=192,
        n_features=8,
        n_informative=5,
        n_redundant=0,
        random_state=23,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    clf = ctboost.CTBoostClassifier(
        iterations=14,
        learning_rate=0.15,
        max_depth=2,
        alpha=1.0,
        lambda_l2=1.0,
    )
    clf.fit(X, y)

    staged_probabilities = list(clf.staged_predict_proba(X))

    assert len(staged_probabilities) == clf._booster.num_iterations_trained
    np.testing.assert_allclose(staged_probabilities[-1], clf.predict_proba(X), rtol=1e-6, atol=1e-6)

    model_path = tmp_path / "classifier.pkl"
    clf.save_model(model_path)
    restored = ctboost.CTBoostClassifier.load_model(model_path)

    np.testing.assert_array_equal(restored.classes_, clf.classes_)
    np.testing.assert_allclose(restored.predict_proba(X), clf.predict_proba(X), rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(restored.predict(X), clf.predict(X))

def test_classifier_json_save_load_round_trip(tmp_path: Path):
    X, y = make_classification(
        n_samples=128,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=31,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    clf = ctboost.CTBoostClassifier(
        iterations=10,
        learning_rate=0.2,
        max_depth=2,
        alpha=1.0,
        lambda_l2=1.0,
    )
    clf.fit(X, y)

    model_path = tmp_path / "classifier.json"
    clf.save_model(model_path)
    restored = ctboost.CTBoostClassifier.load_model(model_path)

    np.testing.assert_array_equal(restored.classes_, clf.classes_)
    np.testing.assert_allclose(restored.predict_proba(X), clf.predict_proba(X), rtol=1e-6, atol=1e-6)
