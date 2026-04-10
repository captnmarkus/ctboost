import numpy as np
import pytest

import ctboost
import ctboost._core as _core


def test_pool_reports_dimensions_and_column_major_storage():
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    label = np.array([0.0, 1.0], dtype=np.float32)

    pool = ctboost.Pool(data, label, cat_features=[1])

    assert pool.num_rows == 2
    assert pool.num_cols == 3
    assert pool.cat_features == [1]
    np.testing.assert_array_equal(
        pool.feature_data,
        np.array([1.0, 4.0, 2.0, 5.0, 3.0, 6.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(pool.label, label)
    np.testing.assert_array_equal(pool.data, data)


def test_pool_rejects_mismatched_label_length():
    data = np.random.default_rng(0).normal(size=(4, 3)).astype(np.float32)
    label = np.random.default_rng(1).normal(size=(3,)).astype(np.float32)

    with pytest.raises(ValueError, match="label size must match"):
        ctboost.Pool(data, label)


def test_logloss_gradients_match_numpy():
    preds = np.array([-2.0, -0.5, 0.0, 1.75], dtype=np.float32)
    labels = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)

    gradients, hessians = _core._debug_compute_objective("Logloss", preds, labels)

    probabilities = 1.0 / (1.0 + np.exp(-preds))
    np.testing.assert_allclose(gradients, probabilities - labels, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        hessians,
        probabilities * (1.0 - probabilities),
        rtol=1e-6,
        atol=1e-6,
    )


def test_squared_error_gradients_match_numpy():
    preds = np.array([1.5, -0.5, 3.0], dtype=np.float32)
    labels = np.array([1.0, 0.0, 2.5], dtype=np.float32)

    gradients, hessians = _core._debug_compute_objective("RMSE", preds, labels)

    np.testing.assert_allclose(gradients, preds - labels, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(hessians, np.ones_like(preds), rtol=1e-6, atol=1e-6)


def test_softmax_loss_gradients_match_numpy():
    logits = np.array(
        [
            0.2,
            -0.1,
            1.1,
            -0.3,
            0.4,
            0.8,
            1.2,
            -0.7,
            0.1,
        ],
        dtype=np.float32,
    )
    labels = np.array([2.0, 1.0, 0.0], dtype=np.float32)

    gradients, hessians = _core._debug_compute_objective(
        "MultiClass",
        logits,
        labels,
        num_classes=3,
    )

    logits_2d = logits.reshape(3, 3)
    shifted = logits_2d - logits_2d.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    expected_gradients = probabilities.copy()
    expected_gradients[np.arange(labels.size), labels.astype(np.int64)] -= 1.0
    expected_hessians = probabilities * (1.0 - probabilities)

    np.testing.assert_allclose(
        gradients.reshape(3, 3),
        expected_gradients,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        hessians.reshape(3, 3),
        expected_hessians,
        rtol=1e-6,
        atol=1e-6,
    )
