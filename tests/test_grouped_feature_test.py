import ctypes
import ctypes.util
import hashlib
import json
import os
import pickle
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import ctboost._core as _core
import numpy as np
import pytest
from scipy.stats import chi2
from sklearn.base import clone

import ctboost
from ctboost.cli import main as cli_main
from tests.helpers import authenticated_tcp_root as _authenticated_tcp_root
from tests.helpers import find_free_tcp_port as _find_free_tcp_port


def _root(booster):
    return booster._handle.export_state()["trees"][0]["nodes"][0]


def _require_cuda_device_for_hardware_test():
    package_parent = Path(ctboost.__file__).resolve().parent.parent
    pattern = "cudart64_*.dll" if os.name == "nt" else "libcudart.so*"
    candidates = list(package_parent.glob(f"*/{pattern}"))
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        runtime_directory = Path(cuda_path) / (
            "bin" if os.name == "nt" else "lib64"
        )
        candidates.extend(sorted(runtime_directory.glob(pattern), reverse=True))
    if os.name == "nt":
        for path_entry in os.environ.get("PATH", "").split(os.pathsep):
            if path_entry:
                candidates.extend(
                    sorted(Path(path_entry).glob("cudart64_*.dll"), reverse=True)
                )
    else:
        maps_path = Path("/proc/self/maps")
        if maps_path.is_file():
            for line in maps_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines():
                mapped_path = line.rsplit(maxsplit=1)[-1]
                if "libcudart.so" in mapped_path:
                    candidates.append(mapped_path)
        candidates.extend(
            ("libcudart.so", "libcudart.so.12", "/usr/local/cuda/lib64/libcudart.so")
        )
    discovered = ctypes.util.find_library("cudart")
    if discovered:
        candidates.append(discovered)
    errors = []
    seen = set()
    for candidate in candidates:
        candidate = str(candidate)
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            runtime = ctypes.CDLL(candidate)
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
            continue
        runtime.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        runtime.cudaGetDeviceCount.restype = ctypes.c_int
        count = ctypes.c_int()
        status = runtime.cudaGetDeviceCount(ctypes.byref(count))
        if status == 0 and count.value > 0:
            return
        errors.append(f"cudaGetDeviceCount returned {status} / {count.value}")
    pytest.skip("CUDA device unavailable before hardware test: " + "; ".join(errors))


def _python_grouped_reference(scores, bins, weights, groups, missing_bin=-1):
    """Independent executable reference matching the audited midpoint rule."""

    scores = np.asarray(scores, dtype=np.float32).astype(np.float64)
    bins = np.asarray(bins, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float32).astype(np.float64)
    keep = weights > 0.0
    complete = keep & (bins != missing_bin)
    levels = np.unique(bins[complete])
    level_weights = np.asarray(
        [weights[complete & (bins == level)].sum() for level in levels],
        dtype=np.float64,
    )
    level_gradients = np.asarray(
        [(weights[complete & (bins == level)] * scores[complete & (bins == level)]).sum()
         for level in levels],
        dtype=np.float64,
    )
    if levels.size:
        cumulative_before = np.cumsum(level_weights) - level_weights
        midpoints = cumulative_before + 0.5 * level_weights
        raw_groups = np.floor(groups * midpoints / level_weights.sum()).astype(np.int64)
        raw_groups = np.clip(raw_groups, 0, groups - 1)
        _, compact = np.unique(raw_groups, return_inverse=True)
        grouped_weights = np.bincount(compact, weights=level_weights).astype(np.float64)
        grouped_gradients = np.bincount(compact, weights=level_gradients).astype(np.float64)
    else:
        grouped_weights = np.empty(0, dtype=np.float64)
        grouped_gradients = np.empty(0, dtype=np.float64)

    missing = keep & (bins == missing_bin)
    if missing_bin >= 0 and np.any(missing):
        grouped_weights = np.append(grouped_weights, weights[missing].sum())
        grouped_gradients = np.append(
            grouped_gradients, (weights[missing] * scores[missing]).sum()
        )

    total_weight = weights.sum()
    total_gradient = np.dot(weights, scores)
    mean = total_gradient / total_weight
    variance = np.dot(weights, np.square(scores - mean)) / total_weight
    degrees_of_freedom = max(0, grouped_weights.size - 1)
    if total_weight <= 1.0 or degrees_of_freedom == 0 or variance <= np.finfo(float).eps:
        statistic = 0.0
        p_value = 1.0
    else:
        reduced_weights = grouped_weights[:degrees_of_freedom]
        difference = grouped_gradients[:degrees_of_freedom] - reduced_weights * mean
        diagonal_scale = total_weight / (total_weight - 1.0) * variance
        outer_scale = variance / (total_weight - 1.0)
        covariance = -outer_scale * np.outer(reduced_weights, reduced_weights)
        covariance[np.diag_indices_from(covariance)] += (
            diagonal_scale * reduced_weights + 1e-7
        )
        statistic = float(difference @ np.linalg.solve(covariance, difference))
        p_value = float(chi2.sf(statistic, degrees_of_freedom))
    return {
        "weight_sums": grouped_weights,
        "gradient_sums": grouped_gradients,
        "degrees_of_freedom": degrees_of_freedom,
        "chi_square": statistic,
        "p_value": p_value,
    }


@pytest.mark.parametrize("missing_position", ["none", "min", "max"])
def test_native_grouping_matches_midpoint_python_reference_with_unequal_weights(
    missing_position,
):
    nonmissing_bins = np.arange(7, dtype=np.int64)
    weights = np.asarray([91.0, 1.0, 2.0, 17.0, 3.0, 1.0, 29.0], dtype=np.float32)
    gradients = np.asarray([-1.7, 0.4, 2.2, -0.3, 1.1, 3.0, -2.0], dtype=np.float32)
    if missing_position == "min":
        bins = np.concatenate(([0], nonmissing_bins + 1))
        weights = np.concatenate(([11.0], weights)).astype(np.float32)
        gradients = np.concatenate(([4.5], gradients)).astype(np.float32)
        missing_bin = 0
    elif missing_position == "max":
        bins = np.concatenate((nonmissing_bins, [7]))
        weights = np.concatenate((weights, [11.0])).astype(np.float32)
        gradients = np.concatenate((gradients, [4.5])).astype(np.float32)
        missing_bin = 7
    else:
        bins = nonmissing_bins
        missing_bin = -1

    expected = _python_grouped_reference(
        gradients, bins, weights, groups=4, missing_bin=missing_bin
    )
    actual = _core._debug_compute_grouped_pvalue(
        gradients,
        bins,
        weights,
        requested_groups=4,
        missing_bin=missing_bin,
    )

    np.testing.assert_allclose(actual["weight_sums"], expected["weight_sums"], rtol=0, atol=0)
    np.testing.assert_allclose(
        actual["gradient_sums"], expected["gradient_sums"], rtol=1e-7, atol=1e-7
    )
    assert actual["degrees_of_freedom"] == expected["degrees_of_freedom"]
    assert actual["chi_square"] == pytest.approx(expected["chi_square"], rel=1e-10, abs=1e-10)
    assert actual["p_value"] == pytest.approx(expected["p_value"], rel=1e-9, abs=1e-12)


def test_implicit_and_explicit_quadratic_defaults_are_byte_identical():
    rng = np.random.default_rng(901)
    X = rng.normal(size=(384, 5)).astype(np.float32)
    y = (1.1 * X[:, 0] - 0.7 * X[:, 2] + 0.2 * rng.normal(size=X.shape[0])).astype(
        np.float32
    )
    params = {
        "objective": "RMSE",
        "learning_rate": 0.12,
        "max_depth": 3,
        "alpha": 0.2,
        "random_seed": 29,
        "max_bins": 255,
    }
    implicit = ctboost.train(X, params, label=y, num_boost_round=7)
    explicit = ctboost.train(
        X,
        {
            **params,
            "feature_test": "quadratic",
            "feature_test_bins": 8,
            "feature_test_adjustment": "none",
        },
        label=y,
        num_boost_round=7,
    )

    np.testing.assert_array_equal(implicit.predict(X), explicit.predict(X))
    implicit_state = dict(implicit._handle.export_state())
    explicit_state = dict(explicit._handle.export_state())
    assert implicit_state["trees"] == explicit_state["trees"]
    assert hashlib.sha256(pickle.dumps(implicit_state, protocol=5)).digest() == hashlib.sha256(
        pickle.dumps(explicit_state, protocol=5)
    ).digest()


@pytest.mark.parametrize(
    "shape",
    [
        lambda x: 1.8 * x,
        lambda x: np.where(x > 0.15, 1.0, -1.0),
        lambda x: 2.5 * np.square(x),
    ],
    ids=["smooth", "step", "u_shape"],
)
def test_grouped_high_cardinality_test_detects_numeric_signal(shape):
    rng = np.random.default_rng(902)
    signal = np.linspace(-1.0, 1.0, 2048, dtype=np.float32)
    noise_feature = rng.normal(size=signal.size).astype(np.float32)
    y = (shape(signal) + 0.06 * rng.normal(size=signal.size)).astype(np.float32)
    X = np.column_stack((noise_feature, signal)).astype(np.float32)

    model = ctboost.train(
        X,
        {
            "objective": "RMSE",
            "learning_rate": 1.0,
            "max_depth": 1,
            "alpha": 0.01,
            "max_bins": 255,
            "feature_test": "grouped",
            "feature_test_bins": 8,
        },
        label=y,
        num_boost_round=1,
    )
    root = _root(model)
    assert root["is_leaf"] is False
    assert root["split_feature_id"] == 1


def test_grouped_null_behavior_is_not_driven_by_raw_cardinality():
    rng = np.random.default_rng(13)
    gradients = rng.normal(size=4096).astype(np.float32)
    weights = np.ones(gradients.size, dtype=np.float32)
    high_cardinality = rng.integers(0, 255, size=gradients.size, dtype=np.int64)
    low_cardinality = rng.integers(0, 8, size=gradients.size, dtype=np.int64)

    high = _core._debug_compute_grouped_pvalue(
        gradients, high_cardinality, weights, requested_groups=8
    )
    low = _core._debug_compute_grouped_pvalue(
        gradients, low_cardinality, weights, requested_groups=8
    )
    assert high["degrees_of_freedom"] == low["degrees_of_freedom"] == 7
    assert high["p_value"] > 0.05
    assert low["p_value"] > 0.05


def test_grouping_only_affects_feature_test_and_raw_bins_remain_available_for_cut():
    x = np.linspace(0.0, 1.0, 2048, dtype=np.float32)
    y = np.where(x > 0.82, 3.0, -0.4).astype(np.float32)
    model = ctboost.train(
        x.reshape(-1, 1),
        {
            "objective": "RMSE",
            "max_depth": 1,
            "alpha": 0.01,
            "max_bins": 255,
            "feature_test": "grouped",
            "feature_test_bins": 8,
        },
        label=y,
        num_boost_round=1,
    )
    root = _root(model)
    assert root["is_leaf"] is False
    assert root["split_bin_index"] > 8


@pytest.mark.parametrize("nan_mode", ["Min", "Max"])
def test_grouped_test_keeps_missing_bin_separate_for_both_nan_modes(nan_mode):
    rng = np.random.default_rng(903)
    X = rng.normal(size=(800, 2)).astype(np.float32)
    missing = np.arange(X.shape[0]) % 5 == 0
    X[missing, 0] = np.nan
    y = (4.0 * missing + 0.05 * rng.normal(size=X.shape[0])).astype(np.float32)
    model = ctboost.train(
        X,
        {
            "objective": "RMSE",
            "max_depth": 1,
            "alpha": 0.01,
            "max_bins": 64,
            "nan_mode": nan_mode,
            "learning_rate": 1.0,
            "feature_test": "grouped",
            "feature_test_bins": 8,
        },
        label=y,
        num_boost_round=1,
    )
    root = _root(model)
    assert root["is_leaf"] is False
    assert root["split_feature_id"] == 0
    predictions = model.predict(X)
    assert abs(predictions[missing].mean() - predictions[~missing].mean()) > 0.5


def test_grouped_test_uses_node_weights():
    rng = np.random.default_rng(904)
    x = np.linspace(-1.0, 1.0, 1200, dtype=np.float32)
    X = np.column_stack((rng.normal(size=x.size), x)).astype(np.float32)
    y = (x * x + 0.05 * rng.normal(size=x.size)).astype(np.float32)
    weights = np.where(x < -0.5, 7.0, np.where(x > 0.6, 3.0, 0.5)).astype(np.float32)
    model = ctboost.train(
        ctboost.Pool(X, y, weight=weights),
        {
            "objective": "RMSE",
            "max_depth": 1,
            "alpha": 0.01,
            "max_bins": 255,
            "feature_test": "grouped",
            "feature_test_bins": 8,
        },
        num_boost_round=1,
    )
    assert _root(model)["split_feature_id"] == 1


def test_categorical_feature_statistic_is_unchanged_in_grouped_mode():
    categories = np.tile(np.arange(5, dtype=np.float32), 120)
    y = np.isin(categories, [1.0, 4.0]).astype(np.float32)
    pool = ctboost.Pool(categories.reshape(-1, 1), y, cat_features=[0])
    params = {"objective": "Logloss", "max_depth": 2, "alpha": 1.0, "random_seed": 17}
    quadratic = ctboost.train(pool, params, num_boost_round=3)
    grouped = ctboost.train(
        pool, {**params, "feature_test": "grouped", "feature_test_bins": 3}, num_boost_round=3
    )
    np.testing.assert_array_equal(quadratic.predict(pool), grouped.predict(pool))
    assert quadratic._handle.export_state()["trees"] == grouped._handle.export_state()["trees"]


def test_bonferroni_changes_only_global_stopping_decision():
    rng = np.random.default_rng(2)
    x = rng.normal(size=256).astype(np.float32)
    y = (0.05 * x + rng.normal(size=x.size)).astype(np.float32)
    X = np.repeat(x[:, None], 8, axis=1)
    params = {
        "objective": "RMSE",
        "max_depth": 1,
        "alpha": 0.05,
        "max_bins": 32,
        "feature_test": "grouped",
        "feature_test_bins": 8,
    }
    unadjusted = ctboost.train(X, params, label=y, num_boost_round=1)
    adjusted = ctboost.train(
        X,
        {**params, "feature_test_adjustment": "bonferroni"},
        label=y,
        num_boost_round=1,
    )
    assert _root(unadjusted)["is_leaf"] is False
    assert _root(adjusted)["is_leaf"] is True


def test_leafwise_bonferroni_gates_both_child_candidates():
    rng = np.random.default_rng(921)
    row_count = 4000
    root_category = rng.integers(0, 2, row_count).astype(np.float32)
    left_signal = rng.normal(size=row_count).astype(np.float32)
    right_signal = rng.normal(size=row_count).astype(np.float32)
    noise_features = rng.normal(size=(row_count, 5)).astype(np.float32)
    noise = rng.normal(size=row_count).astype(np.float32)
    X = np.column_stack(
        (root_category, left_signal, right_signal, noise_features)
    ).astype(np.float32)
    conditional_signal = np.where(root_category == 0, left_signal, right_signal)
    y = (
        3.0 * np.where(root_category > 0, 1.0, -1.0)
        + 0.06 * conditional_signal
        + noise
    ).astype(np.float32)
    pool = ctboost.Pool(X, y, cat_features=[0])
    params = {
        "objective": "RMSE",
        "learning_rate": 1.0,
        "max_depth": 2,
        "grow_policy": "LeafWise",
        "max_leaves": 4,
        "alpha": 0.05,
        "max_bins": 64,
        "feature_test": "grouped",
        "feature_test_bins": 8,
        "random_seed": 1,
    }
    unadjusted = ctboost.train(pool, params, num_boost_round=1)
    adjusted = ctboost.train(
        pool,
        {**params, "feature_test_adjustment": "bonferroni"},
        num_boost_round=1,
    )
    unadjusted_nodes = unadjusted._handle.export_state()["trees"][0]["nodes"]
    adjusted_nodes = adjusted._handle.export_state()["trees"][0]["nodes"]
    assert [node["split_feature_id"] for node in unadjusted_nodes if not node["is_leaf"]] == [
        0,
        1,
        2,
    ]
    assert [node["split_feature_id"] for node in adjusted_nodes if not node["is_leaf"]] == [0]
    assert sum(node["is_leaf"] for node in adjusted_nodes) == 2


@pytest.mark.parametrize(
    "ranked_trigger",
    [
        {"monotone_constraints": [0] * 8},
        {"random_strength": 1e-12},
    ],
    ids=["constraints", "random_strength"],
)
def test_ranked_candidate_paths_keep_raw_significance_and_bonferroni_gate(
    ranked_trigger,
):
    rng = np.random.default_rng(2)
    x = rng.normal(size=256).astype(np.float32)
    y = (0.05 * x + rng.normal(size=x.size)).astype(np.float32)
    X = np.repeat(x[:, None], 8, axis=1)
    params = {
        "objective": "RMSE",
        "max_depth": 1,
        "alpha": 0.05,
        "max_bins": 32,
        "feature_test": "grouped",
        "feature_test_bins": 8,
        **ranked_trigger,
    }
    unadjusted = ctboost.train(X, params, label=y, num_boost_round=1)
    assert _root(unadjusted)["is_leaf"] is False
    adjusted = ctboost.train(
        X,
        {
            **params,
            "feature_test_adjustment": "bonferroni",
            "verbose": True,
        },
        label=y,
        num_boost_round=1,
    )
    assert _root(adjusted)["is_leaf"] is True


def test_profiler_distinguishes_raw_and_adjusted_stopping_p_values():
    script = textwrap.dedent(
        """
        import numpy as np
        import ctboost
        rng = np.random.default_rng(2)
        x = rng.normal(size=256).astype(np.float32)
        y = (0.05 * x + rng.normal(size=x.size)).astype(np.float32)
        X = np.repeat(x[:, None], 8, axis=1)
        ctboost.train(
            X,
            {
                "objective": "RMSE",
                "max_depth": 1,
                "alpha": 0.05,
                "max_bins": 32,
                "feature_test": "grouped",
                "feature_test_bins": 8,
                "feature_test_adjustment": "bonferroni",
                "monotone_constraints": [0] * 8,
                "verbose": True,
            },
            label=y,
            num_boost_round=1,
        )
        """
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path.cwd()) + os.pathsep + environment.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(Path.cwd()),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    profile = completed.stderr
    match = re.search(
        r"\bp_value=([0-9.e+-]+) stopping_p_value=([0-9.e+-]+) tested_features=(\d+)",
        profile,
    )
    assert match is not None
    raw_p_value = float(match.group(1))
    stopping_p_value = float(match.group(2))
    assert int(match.group(3)) == 8
    assert stopping_p_value == pytest.approx(min(1.0, 8.0 * raw_p_value), rel=2e-5)
    assert raw_p_value < 0.05 < stopping_p_value


@pytest.mark.parametrize("feature_test", ["linear", "maxstat", "unknown"])
def test_unknown_feature_tests_are_rejected(feature_test):
    with pytest.raises(ValueError, match="feature_test must be one of"):
        _core.GradientBooster(feature_test=feature_test)


@pytest.mark.parametrize("feature_test_bins", [1, 65])
def test_grouped_bin_count_is_bounded(feature_test_bins):
    with pytest.raises(ValueError, match=r"feature_test_bins must be in \[2, 64\]"):
        _core.GradientBooster(feature_test="grouped", feature_test_bins=feature_test_bins)


def test_unknown_feature_test_adjustment_is_rejected():
    with pytest.raises(ValueError, match="feature_test_adjustment must be one of"):
        _core.GradientBooster(feature_test_adjustment="holm")


def test_grouped_parameters_round_trip_persist_clone_resume_and_warm_start(tmp_path):
    rng = np.random.default_rng(905)
    X = rng.normal(size=(320, 4)).astype(np.float32)
    y = (X[:, 0] * X[:, 0] - 0.5 * X[:, 1] + 0.05 * rng.normal(size=X.shape[0])).astype(
        np.float32
    )
    params = {
        "objective": "RMSE",
        "learning_rate": 0.15,
        "max_depth": 2,
        "alpha": 0.2,
        "random_seed": 31,
        "feature_test": "grouped",
        "feature_test_bins": 12,
        "feature_test_adjustment": "bonferroni",
    }
    reference = ctboost.train(X, params, label=y, num_boost_round=6)
    assert reference.feature_test == "grouped"
    assert reference.feature_test_bins == 12
    assert reference.feature_test_adjustment == "bonferroni"
    state = dict(reference._handle.export_state())
    assert state["feature_test"] == "grouped"
    assert state["feature_test_bins"] == 12
    assert state["feature_test_adjustment"] == "bonferroni"

    path = tmp_path / "grouped.ctb"
    reference.save_model(path)
    restored = ctboost.load_model(path)
    np.testing.assert_array_equal(reference.predict(X), restored.predict(X))
    assert restored.feature_test == "grouped"

    snapshot = tmp_path / "grouped-snapshot.ctb"
    ctboost.train(X, params, label=y, num_boost_round=2, snapshot_path=snapshot)
    resumed = ctboost.train(
        X,
        params,
        label=y,
        num_boost_round=6,
        snapshot_path=snapshot,
        resume_from_snapshot=True,
    )
    np.testing.assert_array_equal(reference.predict(X), resumed.predict(X))
    assert reference._handle.export_state()["trees"] == resumed._handle.export_state()["trees"]
    with pytest.raises(ValueError, match="Use init_model"):
        ctboost.train(
            X,
            {**params, "feature_test_bins": 8},
            label=y,
            num_boost_round=7,
            snapshot_path=snapshot,
            resume_from_snapshot=True,
        )

    mixed_case_snapshot = tmp_path / "grouped-mixed-case.ctb"
    mixed_case_params = {
        **params,
        "feature_test": " Grouped ",
        "feature_test_adjustment": "BONFERRONI",
    }
    ctboost.train(
        X,
        mixed_case_params,
        label=y,
        num_boost_round=2,
        snapshot_path=mixed_case_snapshot,
    )
    mixed_case_resumed = ctboost.train(
        X,
        mixed_case_params,
        label=y,
        num_boost_round=4,
        snapshot_path=mixed_case_snapshot,
        resume_from_snapshot=True,
    )
    assert mixed_case_resumed.feature_test == "grouped"
    assert mixed_case_resumed.feature_test_adjustment == "bonferroni"

    empty_alias_params = {
        **params,
        "feature_test": "",
        "feature_test_adjustment": "",
    }
    empty_alias_reference = ctboost.train(
        X,
        empty_alias_params,
        label=y,
        num_boost_round=4,
    )
    empty_alias_snapshot = tmp_path / "quadratic-empty-alias.ctb"
    ctboost.train(
        X,
        empty_alias_params,
        label=y,
        num_boost_round=2,
        snapshot_path=empty_alias_snapshot,
    )
    empty_alias_resumed = ctboost.train(
        X,
        empty_alias_params,
        label=y,
        num_boost_round=4,
        snapshot_path=empty_alias_snapshot,
        resume_from_snapshot=True,
    )
    assert empty_alias_resumed.feature_test == "quadratic"
    assert empty_alias_resumed.feature_test_adjustment == "none"
    np.testing.assert_array_equal(
        empty_alias_reference.predict(X), empty_alias_resumed.predict(X)
    )
    assert (
        empty_alias_reference._handle.export_state()["trees"]
        == empty_alias_resumed._handle.export_state()["trees"]
    )

    inherited = ctboost.train(
        X,
        {"objective": "RMSE"},
        label=y,
        num_boost_round=1,
        init_model=reference,
    )
    assert inherited.feature_test == "grouped"
    assert inherited.feature_test_bins == 12
    assert inherited.feature_test_adjustment == "bonferroni"

    estimator = ctboost.CTBoostRegressor(
        iterations=2,
        feature_test="grouped",
        feature_test_bins=10,
        feature_test_adjustment="bonferroni",
    )
    cloned = clone(estimator)
    assert cloned.get_params()["feature_test"] == "grouped"
    assert cloned.get_params()["feature_test_bins"] == 10
    assert cloned.get_params()["feature_test_adjustment"] == "bonferroni"
    cloned.fit(X, y)
    assert cloned.get_booster().feature_test == "grouped"


def test_legacy_state_without_grouped_keys_defaults_to_quadratic():
    X = np.arange(48, dtype=np.float32).reshape(16, 3)
    y = X[:, 0] - X[:, 2]
    model = ctboost.train(X, {"objective": "RMSE", "alpha": 1.0}, label=y, num_boost_round=2)
    state = dict(model._handle.export_state())
    state.pop("feature_test")
    state.pop("feature_test_bins")
    state.pop("feature_test_adjustment")
    restored = ctboost.Booster(_core.GradientBooster.from_state(state))
    assert restored.feature_test == "quadratic"
    assert restored.feature_test_bins == 8
    assert restored.feature_test_adjustment == "none"
    np.testing.assert_array_equal(model.predict(X), restored.predict(X))


def test_cli_grouped_overrides_reach_persisted_model(tmp_path):
    x = np.linspace(-1.0, 1.0, 160, dtype=np.float32)
    X = np.column_stack((x, x * x)).astype(np.float32)
    y = (x * x).astype(np.float32)
    input_path = tmp_path / "grouped-input.npz"
    model_path = tmp_path / "grouped-cli.ctb"
    np.savez(input_path, X=X, target=y)
    result = cli_main(
        [
            "train",
            "--task",
            "regression",
            "--input",
            str(input_path),
            "--array-key",
            "X",
            "--target",
            "target",
            "--model",
            str(model_path),
            "--iterations",
            "1",
            "--feature-test",
            "grouped",
            "--feature-test-bins",
            "6",
            "--feature-test-adjustment",
            "bonferroni",
        ]
    )
    assert result == 0
    restored = ctboost.CTBoostRegressor.load_model(model_path)
    params = restored.get_params()
    assert params["feature_test"] == "grouped"
    assert params["feature_test_bins"] == 6
    assert params["feature_test_adjustment"] == "bonferroni"


def test_gpu_grouped_statistics_reach_the_cuda_backend_boundary():
    if ctboost.build_info()["cuda_enabled"]:
        booster = _core.GradientBooster(
            task_type="GPU",
            feature_test="grouped",
            feature_test_adjustment="bonferroni",
        )
        assert booster.feature_test() == "grouped"
        assert booster.feature_test_adjustment() == "bonferroni"
    else:
        with pytest.raises(RuntimeError, match="compiled without CUDA support"):
            _core.GradientBooster(
                task_type="GPU",
                feature_test="grouped",
                feature_test_adjustment="bonferroni",
            )


def test_gpu_grouped_bonferroni_matches_cpu_root_selection_when_available():
    if not ctboost.build_info()["cuda_enabled"]:
        pytest.skip("CUDA support is not compiled into this build")
    _require_cuda_device_for_hardware_test()

    rng = np.random.default_rng(15521)
    signal = np.linspace(-1.0, 1.0, 4096, dtype=np.float32)
    X = np.column_stack(
        (
            rng.normal(size=signal.size),
            signal,
            rng.normal(size=signal.size),
            rng.normal(size=signal.size),
        )
    ).astype(np.float32)
    y = (2.4 * np.square(signal) + 0.04 * rng.normal(size=signal.size)).astype(
        np.float32
    )
    params = {
        "objective": "RMSE",
        "learning_rate": 1.0,
        "max_depth": 1,
        "alpha": 0.01,
        "max_bins": 255,
        "feature_test": "grouped",
        "feature_test_bins": 8,
        "feature_test_adjustment": "bonferroni",
        "random_seed": 15521,
    }
    cpu = ctboost.train(X, params, label=y, num_boost_round=1)
    gpu = ctboost.train(
        X,
        {**params, "task_type": "GPU", "devices": "0"},
        label=y,
        num_boost_round=1,
    )

    cpu_root = _root(cpu)
    gpu_root = _root(gpu)
    assert cpu_root["is_leaf"] is False
    assert gpu_root["is_leaf"] is False
    assert cpu_root["split_feature_id"] == gpu_root["split_feature_id"] == 1
    assert cpu_root["split_bin_index"] == gpu_root["split_bin_index"]
    assert gpu_root["split_bin_index"] > params["feature_test_bins"]


def test_distributed_gpu_zeroes_an_empty_local_child_histogram_when_available(
    tmp_path: Path,
):
    if not ctboost.build_info()["cuda_enabled"]:
        pytest.skip("CUDA support is not compiled into this build")
    _require_cuda_device_for_hardware_test()

    rows_per_rank = 256
    within_rank = np.linspace(-1.0, 1.0, rows_per_rank, dtype=np.float32)
    rng = np.random.default_rng(15529)
    X = np.column_stack(
        (
            np.concatenate(
                (
                    np.full(rows_per_rank, -2.0, dtype=np.float32),
                    np.full(rows_per_rank, 2.0, dtype=np.float32),
                )
            ),
            np.tile(within_rank, 2),
            rng.normal(size=2 * rows_per_rank).astype(np.float32),
        )
    ).astype(np.float32)
    y = (
        5.0 * np.sign(X[:, 0])
        + 1.5 * (X[:, 1] > 0.0)
        + 0.01 * rng.normal(size=X.shape[0])
    ).astype(np.float32)
    np.save(tmp_path / "global_X.npy", X)
    for rank in range(2):
        begin = rank * rows_per_rank
        end = begin + rows_per_rank
        np.save(tmp_path / f"shard_X_{rank}.npy", X[begin:end])
        np.save(tmp_path / f"shard_y_{rank}.npy", y[begin:end])

    worker = tmp_path / "distributed_gpu_empty_child.py"
    worker.write_text(
        textwrap.dedent(
            """
            import json
            from pathlib import Path
            import sys

            import numpy as np
            import ctboost

            rank = int(sys.argv[1])
            root = Path(sys.argv[2])
            distributed_root = sys.argv[3]
            X = np.load(root / f"shard_X_{rank}.npy")
            y = np.load(root / f"shard_y_{rank}.npy")
            global_X = np.load(root / "global_X.npy")
            model = ctboost.train(
                X,
                {
                    "objective": "RMSE",
                    "task_type": "GPU",
                    "devices": "0",
                    "learning_rate": 0.5,
                    "max_depth": 2,
                    "alpha": 1.0,
                    "max_bins": 64,
                    "feature_test": "grouped",
                    "feature_test_bins": 8,
                    "feature_test_adjustment": "bonferroni",
                    "distributed_world_size": 2,
                    "distributed_rank": rank,
                    "distributed_root": distributed_root,
                    "distributed_run_id": "gpu-empty-local-child",
                    "distributed_timeout": 120.0,
                },
                label=y,
                num_boost_round=1,
            )
            np.save(root / f"distributed_gpu_pred_{rank}.npy", model.predict(global_X))
            with (root / f"distributed_gpu_tree_{rank}.json").open(
                "w", encoding="utf-8"
            ) as stream:
                json.dump(model._handle.export_state()["trees"], stream, sort_keys=True)
            """
        ),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        str(Path.cwd()) + os.pathsep + environment.get("PYTHONPATH", "")
    )
    distributed_root = _authenticated_tcp_root(_find_free_tcp_port())
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                str(worker),
                str(rank),
                str(tmp_path),
                distributed_root,
            ],
            env=environment,
        )
        for rank in range(2)
    ]
    assert [process.wait(timeout=180) for process in processes] == [0, 0]

    trees = [
        json.loads(
            (tmp_path / f"distributed_gpu_tree_{rank}.json").read_text(
                encoding="utf-8"
            )
        )
        for rank in range(2)
    ]
    assert trees[0] == trees[1]
    predictions = [
        np.load(tmp_path / f"distributed_gpu_pred_{rank}.npy") for rank in range(2)
    ]
    np.testing.assert_array_equal(predictions[0], predictions[1])
    assert np.isfinite(predictions[0]).all()
    split_features = [
        node["split_feature_id"]
        for node in trees[0][0]["nodes"]
        if not node["is_leaf"]
    ]
    assert split_features[0] == 0
    assert 1 in split_features[1:]


@pytest.mark.parametrize(
    ("grouped", "bonferroni"),
    [(True, False), (False, True), (True, True)],
)
def test_public_tree_build_boundary_accepts_gpu_feature_tests_before_workspace_validation(
    grouped, bonferroni
):
    with pytest.raises(ValueError, match="GPU histogram workspace must be provided"):
        _core._debug_tree_build_options_boundary(
            use_gpu=True,
            grouped=grouped,
            bonferroni=bonferroni,
        )


def test_public_tree_build_boundary_keeps_legacy_gpu_dispatch():
    with pytest.raises(ValueError, match="GPU histogram workspace must be provided"):
        _core._debug_tree_build_options_boundary(use_gpu=True)


@pytest.mark.parametrize("feature_test_bins", [1, 65])
def test_public_tree_build_boundary_validates_grouped_bin_count(feature_test_bins):
    with pytest.raises(ValueError, match=r"feature_test_bins must be in \[2, 64\]"):
        _core._debug_tree_build_options_boundary(
            use_gpu=False,
            grouped=True,
            feature_test_bins=feature_test_bins,
        )


@pytest.mark.parametrize("nan_mode", ["Min", "Max"])
def test_distributed_grouping_uses_global_missing_schema_when_only_one_shard_has_nans(
    tmp_path: Path, nan_mode
):
    row_count = 256
    X = np.column_stack(
        (
            np.linspace(-2.0, 2.0, row_count, dtype=np.float32),
            np.arange(row_count, dtype=np.float32) % 2,
        )
    )
    y = np.empty(row_count, dtype=np.float32)
    y[: row_count // 2] = np.where(np.arange(row_count // 2) % 2 == 0, 1.0, -1.0)
    second_shard_rows = np.arange(row_count // 2, row_count)
    missing_second = second_shard_rows % 2 == 0
    X[second_shard_rows[missing_second], 0] = np.nan
    y[second_shard_rows] = np.where(missing_second, 3.0, -3.0)
    np.save(tmp_path / "global_X.npy", X)
    for rank, rows in enumerate((np.arange(0, 128), np.arange(128, 256))):
        np.save(tmp_path / ("shard_X_%d.npy" % rank), X[rows])
        np.save(tmp_path / ("shard_y_%d.npy" % rank), y[rows])

    worker = tmp_path / ("missing_schema_%s.py" % nan_mode.lower())
    worker.write_text(
        textwrap.dedent(
            """
            import json
            from pathlib import Path
            import sys
            import numpy as np
            import ctboost

            rank = int(sys.argv[1])
            root = Path(sys.argv[2])
            nan_mode = sys.argv[3]
            X = np.load(root / ("shard_X_%d.npy" % rank))
            y = np.load(root / ("shard_y_%d.npy" % rank))
            global_X = np.load(root / "global_X.npy")
            model = ctboost.train(
                X,
                {
                    "objective": "RMSE",
                    "learning_rate": 1.0,
                    "max_depth": 1,
                    "alpha": 0.01,
                    "max_bins": 64,
                    "nan_mode": nan_mode,
                    "feature_test": "grouped",
                    "feature_test_bins": 8,
                    "distributed_world_size": 2,
                    "distributed_rank": rank,
                    "distributed_root": str(root / ("dist_" + nan_mode.lower())),
                    "distributed_run_id": "one-shard-missing-" + nan_mode.lower(),
                    "distributed_timeout": 120.0,
                },
                label=y,
                num_boost_round=1,
            )
            np.save(root / ("distributed_pred_%s_%d.npy" % (nan_mode.lower(), rank)), model.predict(global_X))
            with (root / ("distributed_tree_%s_%d.json" % (nan_mode.lower(), rank))).open("w", encoding="utf-8") as stream:
                json.dump(model._handle.export_state()["trees"], stream, sort_keys=True)
            """
        ),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path.cwd()) + os.pathsep + environment.get("PYTHONPATH", "")
    processes = [
        subprocess.Popen(
            [sys.executable, str(worker), str(rank), str(tmp_path), nan_mode],
            env=environment,
        )
        for rank in range(2)
    ]
    assert [process.wait(timeout=180) for process in processes] == [0, 0]

    central = ctboost.train(
        X,
        {
            "objective": "RMSE",
            "learning_rate": 1.0,
            "max_depth": 1,
            "alpha": 0.01,
            "max_bins": 64,
            "nan_mode": nan_mode,
            "feature_test": "grouped",
            "feature_test_bins": 8,
        },
        label=y,
        num_boost_round=1,
    )
    central_predictions = central.predict(X)
    central_trees = central._handle.export_state()["trees"]
    for rank in range(2):
        np.testing.assert_array_equal(
            np.load(tmp_path / ("distributed_pred_%s_%d.npy" % (nan_mode.lower(), rank))),
            central_predictions,
        )
        with (tmp_path / ("distributed_tree_%s_%d.json" % (nan_mode.lower(), rank))).open(
            "r", encoding="utf-8"
        ) as stream:
            assert json.load(stream) == central_trees
