"""Prediction-contract checks for the benchmark-only calibration helper."""
import numpy as np
import pytest

from benchmarks.fixed_temperature import apply_temperature


def test_absent_temperature_preserves_original_array_and_dtype():
    margins = np.array([-.2, .3], dtype=np.float32)
    assert apply_temperature({"problem": "binary"}, margins) is margins


@pytest.mark.parametrize("temperature", [.25, .5, .75, 1., 1.25, 1.5, 2., 4.])
def test_positive_temperature_preserves_argmax_and_uses_declared_float64_rule(temperature):
    margins = np.array([[1., 1., -.5], [-1e5, 1e5, 0.], [.125, .375, .25]], dtype=np.float32)
    original = margins.copy()
    actual = apply_temperature({"problem": "multiclass", "temperature": temperature}, margins)
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, margins.astype(np.float64) / temperature)
    np.testing.assert_array_equal(np.argmax(actual, axis=1), np.argmax(margins, axis=1))
    np.testing.assert_array_equal(margins, original)


@pytest.mark.parametrize("temperature", [0., -1., np.nan, np.inf, -np.inf])
def test_invalid_temperature_is_rejected(temperature):
    with pytest.raises(ValueError, match="finite and positive"):
        apply_temperature({"problem": "multiclass", "temperature": temperature}, [[0., 1.]])


@pytest.mark.parametrize("margins", [[0., 1.], [[np.nan, 1.]], [[np.inf, 1.]]])
def test_invalid_margin_shape_or_values_are_rejected(margins):
    with pytest.raises(ValueError, match="two-dimensional"):
        apply_temperature({"problem": "multiclass", "temperature": 1.}, margins)


def test_temperature_cannot_silently_change_regression_or_binary_predictions():
    for problem in ("regression", "binary"):
        with pytest.raises(ValueError, match="restricted to multiclass"):
            apply_temperature({"problem": problem, "temperature": .5}, [[0., 1.]])
