import numpy as np

import ctboost._core as _core


def test_perfectly_aligned_feature_has_tiny_pvalue():
    gradients = np.repeat(np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float32), 64)
    binned_feature = np.repeat(np.arange(4, dtype=np.int64), 64)

    result = _core._debug_compute_pvalue(gradients, binned_feature)

    assert result["degrees_of_freedom"] == 3
    assert result["chi_square"] > 100.0
    assert result["p_value"] < 1e-10


def test_random_feature_fails_to_reject_null():
    rng = np.random.default_rng(1)
    gradients = rng.normal(size=512).astype(np.float32)
    binned_feature = rng.integers(0, 8, size=512, dtype=np.int64)

    result = _core._debug_compute_pvalue(gradients, binned_feature)

    assert result["degrees_of_freedom"] == 7
    assert result["p_value"] > 0.05
