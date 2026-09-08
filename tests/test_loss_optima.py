"""Objective optima checked against analytic, model-independent references."""

import numpy as np
import pytest

import ctboost


def _constant_feature_huber(labels, weights, delta=1.0):
    labels = np.asarray(labels, dtype=np.float32)
    return ctboost.train(
        ctboost.Pool(
            np.zeros((len(labels), 1), dtype=np.float32), labels,
            weight=np.asarray(weights, dtype=np.float32),
        ),
        {"objective": "Huber", "huber_delta": delta, "max_depth": 0},
        num_boost_round=1,
    )


@pytest.mark.parametrize("outlier", [1e38, -1e38])
@pytest.mark.parametrize("outlier_weight", [0.0, 0.1])
def test_huber_initializer_resolves_finite_extreme_outliers(outlier, outlier_weight):
    # The fifteen central observations remain in Huber's quadratic region;
    # the outlier contributes a clipped gradient of +/- its effective weight.
    labels = [7.0] * 15 + [outlier]
    weights = [1.0] * 15 + [outlier_weight]
    effective_weight = float(np.float32(outlier_weight))
    expected = 7.0 + np.sign(outlier - 7.0) * effective_weight / 15.0
    model = _constant_feature_huber(labels, weights)

    assert model.base_score[0] == pytest.approx(expected, rel=0.0, abs=2e-14)
    residual = model.base_score[0] - np.asarray(labels, dtype=np.float32).astype(np.float64)
    gradient = np.dot(
        np.asarray(weights, dtype=np.float32).astype(np.float64), np.clip(residual, -1.0, 1.0)
    )
    assert abs(gradient) < 1e-12


def test_huber_zero_weight_rows_do_not_change_constant_optimum():
    labels = [7.0] * 16
    maximum = np.finfo(np.float32).max
    reference = _constant_feature_huber(labels, np.ones(16))
    padded = _constant_feature_huber(labels + [-maximum, maximum], [1.0] * 16 + [0.0, 0.0])
    assert reference.base_score == padded.base_score == [7.0]
    probe = np.zeros((3, 1), dtype=np.float32)
    np.testing.assert_array_equal(reference.predict(probe), np.full(3, 7.0, dtype=np.float32))
    np.testing.assert_array_equal(padded.predict(probe), reference.predict(probe))


@pytest.mark.parametrize(
    "labels,weights,delta,expected",
    [
        ([0.0, 1.0, 2.0, 10.0], [1.0, 2.0, 3.0, 0.5], 10.0, 2.0),
        ([0.0, 0.0, 0.0, 10.0], [1.0, 2.0, 3.0, 0.5], 1.0, 1.0 / 12.0),
        ([0.0, 0.0, 0.0, 10.0], [1.0, 1.0, 1.0, 1.0], 1.5, 0.5),
        # Preserve the existing upper-edge choice on a flat optimum interval.
        ([0.0, 10.0], [1.0, 1.0], 1.0, 9.0),
    ],
)
def test_huber_initializer_preserves_ordinary_weighted_optima(labels, weights, delta, expected):
    model = _constant_feature_huber(labels, weights, delta)
    assert model.base_score[0] == pytest.approx(expected, rel=0.0, abs=2e-14)


@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9])
def test_quantile_gradient_is_zero_at_exact_fit_and_has_correct_signs(alpha):
    prediction = np.asarray([2.0, 2.0, 2.0, 0.0], dtype=np.float32)
    labels = np.asarray([1.0, 2.0, 3.0, 0.0], dtype=np.float32)
    gradient, hessian = ctboost._core._debug_compute_objective(
        "Quantile", prediction, labels, quantile_alpha=alpha
    )
    # Pinball loss is minimized at each exact prediction. Away from a tie,
    # its derivative is 1-alpha above the target and -alpha below the target.
    expected = np.asarray([1.0 - alpha, 0.0, -alpha, 0.0], dtype=np.float32)
    np.testing.assert_array_equal(gradient, expected)
    np.testing.assert_array_equal(hessian, np.ones_like(prediction))


@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("weighted", [False, True])
def test_quantile_keeps_a_zero_loss_constant_fit_at_its_optimum(alpha, weighted):
    X = np.zeros((16, 1), dtype=np.float32)
    labels = np.full(16, 7.0, dtype=np.float32)
    weights = np.arange(1, 17, dtype=np.float32) if weighted else np.ones(16, dtype=np.float32)
    model = ctboost.train(
        ctboost.Pool(X, labels, weight=weights),
        {"objective": "Quantile", "quantile_alpha": alpha, "max_depth": 0,
         "learning_rate": 0.1, "lambda_l2": 1.0},
        num_boost_round=1,
    )
    prediction = model.predict(X)
    residual = labels.astype(np.float64) - prediction.astype(np.float64)
    pinball = np.maximum(alpha * residual, (alpha - 1.0) * residual)
    assert float(np.average(pinball, weights=weights)) == 0.0
    np.testing.assert_array_equal(prediction, labels)
