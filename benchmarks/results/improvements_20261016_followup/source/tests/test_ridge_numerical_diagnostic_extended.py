import numpy as np
import pytest
from scipy.sparse import csr_matrix

from benchmarks import ridge_numerical_diagnostic as original
from benchmarks import ridge_numerical_diagnostic_extended as extended


@pytest.mark.parametrize("penalty", [1., 10., 100., 1000.])
@pytest.mark.parametrize("target_scale", [1e-6, 1., 1e6])
def test_extended_original_solver_dense_solution_same_certificate(penalty, target_scale):
    rows = np.arange(80)
    groups = np.eye(4)[rows % 4]
    Z = csr_matrix(np.column_stack((groups, groups, rows == 0, rows != 0)).astype(float))
    r = target_scale * np.random.default_rng(121).normal(size=len(rows))
    expected = np.linalg.solve((Z.T @ Z).toarray() + penalty * np.eye(Z.shape[1]), Z.T @ r)
    delta, receipt = extended.solve(Z, r, penalty)
    assert receipt["success"], receipt
    assert receipt["iterations_limit"] == 10000 and receipt["atol"] == receipt["btol"] == 1e-12
    np.testing.assert_allclose(delta, expected, rtol=1e-7, atol=target_scale * 1e-10)
    assert receipt["certificate"] == original.certificate(Z, r, delta, penalty)


def test_unscaled_matrix_damped_at_zero_with_exact_new_controls(monkeypatch):
    import scipy.sparse.linalg
    Z = csr_matrix(np.eye(3))
    called = []
    actual = scipy.sparse.linalg.lsqr
    def inspect(matrix, rhs, **kwargs):
        assert matrix is Z
        called.append(kwargs)
        return actual(matrix, rhs, **kwargs)
    monkeypatch.setattr(scipy.sparse.linalg, "lsqr", inspect)
    _, receipt = extended.solve(Z, np.array([1., -2., 3.]), 10.)
    assert receipt["success"]
    assert called == [{"damp": np.sqrt(10.), "atol": 1e-12, "btol": 1e-12, "conlim": 1e8, "iter_lim": 10000}]
    assert receipt["initial_coefficients"] == "zero"


def test_iteration_limit_and_zero_rhs_keep_same_gates():
    Z = csr_matrix(np.eye(3))
    delta, receipt = extended.solve(Z, np.zeros(3), 1.)
    assert receipt["success"] and receipt["certificate"]["relative_gap"] == 0
    np.testing.assert_array_equal(delta, np.zeros(3))
    rng = np.random.default_rng(501)
    Z = csr_matrix((rng.uniform(size=(80, 30)) > .8).astype(float))
    _, receipt = extended.solve(Z, rng.normal(size=80), 1., iterations=1)
    assert receipt["native"]["istop"] == 7 and not receipt["success"]
