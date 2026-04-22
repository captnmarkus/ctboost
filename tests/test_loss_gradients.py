import numpy as np
import pytest
import ctboost
import ctboost.core as ctcore
import ctboost._core as _core

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

def test_poisson_gradients_match_numpy():
    preds = np.array([-0.2, 0.0, 0.7], dtype=np.float32)
    labels = np.array([0.0, 1.0, 3.0], dtype=np.float32)

    gradients, hessians = _core._debug_compute_objective("Poisson", preds, labels)

    expected_mean = np.exp(preds)
    np.testing.assert_allclose(gradients, expected_mean - labels, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(hessians, expected_mean, rtol=1e-6, atol=1e-6)

def test_tweedie_gradients_match_numpy():
    preds = np.array([-0.1, 0.3, 0.8], dtype=np.float32)
    labels = np.array([0.0, 0.5, 2.0], dtype=np.float32)
    variance_power = 1.4

    gradients, hessians = _core._debug_compute_objective(
        "Tweedie",
        preds,
        labels,
        tweedie_variance_power=variance_power,
    )

    one_minus_power = 1.0 - variance_power
    two_minus_power = 2.0 - variance_power
    expected_gradients = -labels * np.exp(one_minus_power * preds) + np.exp(two_minus_power * preds)
    expected_hessians = (
        labels * (variance_power - 1.0) * np.exp(one_minus_power * preds)
        + two_minus_power * np.exp(two_minus_power * preds)
    )
    np.testing.assert_allclose(gradients, expected_gradients, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(hessians, expected_hessians, rtol=1e-6, atol=1e-6)

def test_survival_exponential_gradients_match_numpy():
    preds = np.array([-0.2, 0.1, 0.5], dtype=np.float32)
    labels = np.array([2.0, -1.5, 3.0], dtype=np.float32)

    gradients, hessians = _core._debug_compute_objective("SurvivalExponential", preds, labels)

    expected_time = np.abs(labels)
    expected_event = (labels > 0.0).astype(np.float32)
    expected_hazard = np.exp(preds)
    np.testing.assert_allclose(
        gradients,
        expected_hazard * expected_time - expected_event,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        hessians,
        expected_hazard * expected_time,
        rtol=1e-6,
        atol=1e-6,
    )

def test_cox_gradients_are_finite_for_signed_survival_labels():
    preds = np.array([0.2, -0.1, 0.4, -0.3], dtype=np.float32)
    labels = np.array([4.0, -3.0, 2.0, 1.0], dtype=np.float32)

    gradients, hessians = _core._debug_compute_objective("Cox", preds, labels)

    assert gradients.shape == labels.shape
    assert hessians.shape == labels.shape
    assert np.all(np.isfinite(gradients))
    assert np.all(np.isfinite(hessians))
    assert np.all(hessians > 0.0)

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
