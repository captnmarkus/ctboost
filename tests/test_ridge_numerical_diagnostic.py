"""Independent small dense oracles for the unchanged augmented ridge objective."""
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr

from benchmarks import ridge_numerical_diagnostic as ridge


def design():
    rows = np.arange(80)
    groups = np.eye(4)[rows % 4]
    rare = np.column_stack((rows == 0, rows != 0)).astype(float)
    # Duplicate tree blocks, dependent partition indicators, rare leaves.
    return csr_matrix(np.column_stack((groups, groups, rare)))


@pytest.mark.parametrize("penalty", [1., 10., 100., 1000.])
@pytest.mark.parametrize("target_scale", [1e-6, 1., 1e6])
@pytest.mark.parametrize("variant", ["original_lsqr", "scaled_lsqr", "scaled_lsmr"])
def test_original_objective_dense_solution_and_certificate(penalty, target_scale, variant):
    Z = design()
    dense = Z.toarray()
    r = target_scale * np.random.default_rng(813).normal(size=Z.shape[0])
    expected = np.linalg.solve(dense.T @ dense + penalty * np.eye(Z.shape[1]), dense.T @ r)
    delta, receipt = ridge.solve(Z, r, penalty, variant)
    np.testing.assert_allclose(delta, expected, rtol=1e-7, atol=target_scale * 1e-10)
    if variant != "original_lsqr":
        assert receipt["success"], receipt
    c = receipt["certificate"]
    assert c["finite"] and c["original_gradient_dual_gap"] >= 0
    np.testing.assert_allclose(c["primal_minus_dual"], c["original_gradient_dual_gap"],
        rtol=1e-5, atol=np.finfo(float).eps * c["primal_zero"] * 20)
    actual_rms = np.linalg.norm(Z @ (delta - expected)) / np.sqrt(Z.shape[0])
    assert actual_rms <= c["fitting_prediction_rms_error_bound"] + target_scale * 1e-13


def test_operator_adjoint_and_explicit_augmented_dense_matrix():
    Z = design()
    penalty = 3.
    B, scale = ridge.scaled_operator(Z, penalty)
    dense = np.vstack((Z.toarray(), np.sqrt(penalty) * np.eye(Z.shape[1]))) * scale
    np.testing.assert_allclose(B @ np.eye(Z.shape[1]), dense, rtol=0, atol=0)
    rng = np.random.default_rng(41)
    theta, u = rng.normal(size=Z.shape[1]), rng.normal(size=B.shape[0])
    np.testing.assert_allclose(B @ theta, dense @ theta)
    np.testing.assert_allclose(B.T @ u, dense.T @ u)
    np.testing.assert_allclose(np.dot(B @ theta, u), np.dot(theta, B.T @ u), atol=1e-13)


def test_isotropic_damping_after_column_scaling_changes_the_objective():
    Z, penalty = design(), 10.
    r = np.linspace(-2, 3, Z.shape[0])
    _, scale = ridge.scaled_operator(Z, penalty)
    wrong = scale * lsqr(Z.multiply(scale), r, damp=np.sqrt(penalty), atol=1e-14, btol=1e-14)[0]
    right, receipt = ridge.solve(Z, r, penalty, "scaled_lsmr")
    assert receipt["success"]
    assert not np.allclose(wrong, right, rtol=1e-3, atol=1e-3)
    assert not ridge.certificate(Z, r, wrong, penalty)["certified"]


@pytest.mark.parametrize("variant", ["original_lsqr", "scaled_lsqr", "scaled_lsmr"])
def test_zero_rhs_is_exact_and_finite(variant):
    Z = design()
    delta, receipt = ridge.solve(Z, np.zeros(Z.shape[0]), 1., variant)
    np.testing.assert_array_equal(delta, np.zeros(Z.shape[1]))
    assert receipt["success"] and receipt["certificate"]["relative_gap"] == 0
    assert receipt["certificate"]["normalized_original_gradient"] == 0


def test_noncanonical_or_nonindicator_design_rejected():
    duplicate = csr_matrix((np.ones(2), np.array([0, 0]), np.array([0, 2])), shape=(1, 1))
    assert not duplicate.has_canonical_format
    with pytest.raises(ValueError, match="canonical CSR"):
        ridge.scaled_operator(duplicate, 1.)
    invalid = design()
    invalid.data[0] = 2
    with pytest.raises(ValueError, match="canonical CSR"):
        ridge.scaled_operator(invalid, 1.)
    with pytest.raises(ValueError, match="canonical CSR"):
        ridge.scaled_operator(design().astype(np.float32), 1.)


def test_iteration_limit_remains_failure_and_bad_certificate_rejected():
    Z = design()
    r = np.random.default_rng(119).normal(size=Z.shape[0])
    _, receipt = ridge.solve(Z, r, 1., "scaled_lsmr", iterations=1)
    assert receipt["native"]["istop"] == 7 and not receipt["success"]
    assert not ridge.certificate(Z, r, np.full(Z.shape[1], 1e6), 1.)["certified"]
    with pytest.raises(ValueError, match="Invalid ridge certificate"):
        ridge.certificate(Z, r, np.full(Z.shape[1], np.nan), 1.)
    with pytest.raises(ValueError, match="positive integer"):
        ridge.solve(Z, r, 1., "scaled_lsmr", iterations=0)
