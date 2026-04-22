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

def test_cv_returns_fold_aggregates():
    X, y = make_classification(
        n_samples=180,
        n_features=8,
        n_informative=5,
        n_redundant=0,
        random_state=19,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    results = ctboost.cv(
        ctboost.Pool(X, y),
        {
            "objective": "Logloss",
            "learning_rate": 0.2,
            "max_depth": 2,
            "alpha": 1.0,
            "lambda_l2": 1.0,
        },
        num_boost_round=10,
        nfold=3,
        random_state=7,
    )

    expected_keys = {
        "iterations",
        "train_loss_mean",
        "train_loss_std",
        "valid_loss_mean",
        "valid_loss_std",
        "best_iteration_mean",
        "best_iteration_std",
    }
    assert expected_keys == set(results)
    assert results["iterations"].shape == results["train_loss_mean"].shape
    assert results["iterations"].shape == results["valid_loss_mean"].shape
    assert np.all(np.isfinite(results["train_loss_mean"]))
    assert np.all(np.isfinite(results["valid_loss_mean"]))
    assert np.isfinite(results["best_iteration_mean"])
