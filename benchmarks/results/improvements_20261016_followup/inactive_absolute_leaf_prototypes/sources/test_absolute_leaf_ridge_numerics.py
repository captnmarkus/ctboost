"""Synthetic-only numerical checks; no datasets, models or native learners."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy import sparse

SPEC = importlib.util.spec_from_file_location("absolute_leaf_ridge_numerics_synthetic", Path(__file__).with_name("absolute_leaf_ridge_numerics.py"))
ridge = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ridge
SPEC.loader.exec_module(ridge)

SETTINGS = ridge.LSQRSettings(iterations=10000, atol=1e-12, btol=1e-12, conlim=1e8)
LIMITS = ridge.CertificateLimits(relative_gap=1e-10, normalized_gradient=1e-8,
    normalized_intercept=1e-8, normalized_dual_feasibility=1e-12)


def design():
    # Two partitions, duplicated column, constant column, unseen zero column,
    # and a canceling column. No random real-world data or learner is involved.
    assignments = np.array([[0,0], [0,1], [1,1], [1,0], [2,1], [2,0], [2,0], [1,1]])
    dense = np.zeros((8, 9), dtype=np.float64)
    dense[np.arange(8), assignments[:,0]] = 1
    dense[np.arange(8), 3+assignments[:,1]] = 1
    dense[:,5] = dense[:,0]
    dense[:,6] = 1
    dense[:,8] = -dense[:,1]
    return sparse.csr_matrix(dense)


def dense_oracle(Z, y, penalty):
    dense = Z.toarray()
    mean_y, means = np.mean(y), np.mean(dense, axis=0)
    centered, target = dense-means, y-mean_y
    weights = np.linalg.solve(centered.T @ centered + penalty*np.eye(Z.shape[1]), centered.T @ target)
    return weights, mean_y-means@weights


@pytest.mark.parametrize("penalty", [0.1, 1.0, 10.0, 1000.0])
@pytest.mark.parametrize("scale", [1e-4, 1.0, 1e4])
def test_matches_independent_dense_normal_equations(penalty, scale):
    Z = design()
    y = np.array([2.,-1.,3.,7.,0.,4.,-2.,5.])*scale
    before = Z.data.copy(), y.copy()
    w, b, receipt = ridge.solve_absolute_ridge(Z,y,penalty,settings=SETTINGS,limits=LIMITS)
    expected_w, expected_b = dense_oracle(Z,y,penalty)
    assert receipt["success"]
    np.testing.assert_allclose(w, expected_w, rtol=2e-8, atol=2e-10*scale)
    np.testing.assert_allclose(b, expected_b, rtol=2e-8, atol=2e-10*scale)
    np.testing.assert_array_equal(Z.data,before[0]); np.testing.assert_array_equal(y,before[1])
    assert w[7] == 0  # unseen column is zero at the absolute ridge optimum
    np.testing.assert_allclose(w[0],w[5],atol=2e-10*scale)  # duplicate columns
    np.testing.assert_allclose(w[1],-w[8],atol=2e-10*scale)  # canceling columns
    np.testing.assert_allclose(w[6],0,atol=2e-10*scale)  # free intercept absorbs constant
    assert not receipt["certificate"]["rigorous_interval_bound"]
    json.dumps(receipt,allow_nan=False)


def test_centered_operator_matches_dense_and_is_adjoint():
    Z = design()
    A,p = ridge.centered_operator(Z)
    dense=Z.toarray(); centered=dense-dense.mean(axis=0)
    rng=np.random.default_rng(18018)
    w=rng.normal(size=Z.shape[1]);v=rng.normal(size=Z.shape[0])
    np.testing.assert_allclose(p,dense.mean(axis=0),atol=1e-15)
    np.testing.assert_allclose(A@w,centered@w,atol=1e-14)
    np.testing.assert_allclose(A.T@v,centered.T@v,atol=1e-14)
    np.testing.assert_allclose(v@(A@w),w@(A.T@v),atol=1e-13)


@pytest.mark.parametrize("shift", [-1e6, 0., 1e6])
def test_target_shift_changes_only_intercept(shift):
    Z=design();y=np.array([2.,-1.,3.,7.,0.,4.,-2.,5.])
    w,b,r=ridge.solve_absolute_ridge(Z,y,10.,settings=SETTINGS,limits=LIMITS)
    shifted_w,shifted_b,s=ridge.solve_absolute_ridge(Z,y+shift,10.,settings=SETTINGS,limits=LIMITS)
    assert r["success"] and s["success"]
    np.testing.assert_allclose(shifted_w,w,rtol=1e-9,atol=1e-10)
    np.testing.assert_allclose(shifted_b-shift,b,rtol=1e-9,atol=2e-10)


@pytest.mark.parametrize("rows", [1,3,7,8,13])
@pytest.mark.parametrize("target", [-7.,0.,0.1,1e8])
def test_single_row_or_constant_target(rows,target):
    Z=sparse.csr_matrix(np.ones((rows,3),dtype=np.float64))
    y=np.full(rows,target)
    w,b,r=ridge.solve_absolute_ridge(Z,y,1.,settings=SETTINGS,limits=LIMITS)
    assert r["success"] and b == target
    np.testing.assert_array_equal(w,np.zeros(3))
    assert r["certificate"]["stable_gap_terms"] == 0


@pytest.mark.parametrize("rows", [3,7,13])
def test_non_power_length_constant_columns_and_adjoint(rows):
    rng=np.random.default_rng(9123+rows)
    dense=np.column_stack([np.ones(rows),np.full(rows,.1),rng.normal(size=rows),np.zeros(rows)])
    Z=sparse.csr_matrix(dense)
    A,_=ridge.centered_operator(Z)
    np.testing.assert_array_equal(A@np.array([1.,0.,0.,0.]),np.zeros(rows))
    np.testing.assert_array_equal(A@np.array([0.,1.,0.,0.]),np.zeros(rows))
    w=rng.normal(size=4);v=rng.normal(size=rows)
    np.testing.assert_allclose(v@(A@w),w@(A.T@v),rtol=1e-12,atol=1e-12)
    y=rng.normal(size=rows)
    solved,b,receipt=ridge.solve_absolute_ridge(Z,y,1.,settings=SETTINGS,limits=LIMITS)
    assert receipt["success"]
    np.testing.assert_allclose(solved[[0,1,3]],0,atol=1e-12)


def test_large_common_shift_and_design_dynamic_range():
    rng=np.random.default_rng(761)
    dense=rng.normal(size=(13,4))*np.array([1e-3,1.,1e3,1.])
    dense[:,3]=1
    Z=sparse.csr_matrix(dense)
    y=rng.integers(-30,30,size=13).astype(np.float64)
    w,b,r=ridge.solve_absolute_ridge(Z,y,10.,settings=SETTINGS,limits=LIMITS)
    shifted_w,shifted_b,s=ridge.solve_absolute_ridge(Z,y+1e8,10.,settings=SETTINGS,limits=LIMITS)
    assert r["success"] and s["success"]
    np.testing.assert_allclose(shifted_w,w,rtol=2e-7,atol=2e-8)
    np.testing.assert_allclose(shifted_b-1e8,b,rtol=2e-7,atol=2e-8)
    assert abs(s["certificate"]["dual_vector_sum"]) < 1e-10


def test_wrong_intercept_rejected_even_if_centered_gradient_zero():
    Z=sparse.csr_matrix(np.ones((8,2),dtype=np.float64));y=np.full(8,3.)
    c=ridge.certify(Z,y,np.zeros(2),2.,1.,limits=LIMITS)
    assert c["coefficient_gradient_norm"] == 0
    assert c["residual_mean"] == 1 and c["analytic_intercept_gap_term"] == 4
    assert not c["certified"] and not c["gates"]["normalized_intercept"]
    json.dumps(c,allow_nan=False)


def test_original_objective_dual_identity_and_stationarity():
    Z=design();y=np.array([2.,-1.,3.,7.,0.,4.,-2.,5.])
    w=np.array([.1,.2,-.3,.4,-.1,.2,.8,-.5,-.7]);b=1.234
    c=ridge.certify(Z,y,w,b,3.,limits=LIMITS)
    e=y-b-Z@w;u=e-e.mean()
    primal=.5*(e@e)+1.5*(w@w)
    dual=y@u-.5*(u@u)-(Z.T@u)@(Z.T@u)/6
    np.testing.assert_allclose(c["primal"],primal,atol=1e-12)
    np.testing.assert_allclose(c["dual_direct"],dual,atol=1e-12)
    assert abs(c["direct_gap_minus_reconstructed_gap"]) <= 64*c["gap_cancellation_roundoff_scale"]
    assert not c["certified"]


def test_one_iteration_and_native_status_are_not_silently_accepted():
    Z=design();y=np.array([2.,-1.,3.,7.,0.,4.,-2.,5.])
    short=ridge.LSQRSettings(iterations=1,atol=1e-12,btol=1e-12,conlim=1e8)
    _,_,receipt=ridge.solve_absolute_ridge(Z,y,.1,settings=short,limits=LIMITS)
    assert receipt["native"]["status"] == 7 and not receipt["success"]


def test_identity_preserves_bits_and_shrinkage_moves_both_parts():
    original=np.array([-0.,1.,-2.]);optimized=np.array([0.,3.,-4.])
    identity,b=ridge.interpolate_policy(original,-0.,optimized,4.,0.)
    np.testing.assert_array_equal(identity.view(np.uint64),original.view(np.uint64))
    assert np.signbit(b)
    w,b=ridge.interpolate_policy(original,2.,optimized,4.,.25)
    np.testing.assert_array_equal(w,.75*original+.25*optimized)
    assert b == 2.5
    w,b=ridge.interpolate_policy(original,2.,optimized,4.,1.)
    np.testing.assert_array_equal(w,optimized);assert b == 4.


@pytest.mark.parametrize("penalty", [0.,-1.,np.nan,np.inf])
def test_rejects_invalid_penalty(penalty):
    with pytest.raises(ValueError):
        ridge.solve_absolute_ridge(design(),np.arange(8.),penalty,settings=SETTINGS,limits=LIMITS)


def test_rejects_noncanonical_or_nonfinite_design_and_shapes():
    with pytest.raises(TypeError):
        ridge.centered_operator(sparse.csr_matrix(np.ones((3,2),dtype=np.float32)))
    duplicated=sparse.csr_matrix((np.array([1.,1.]),np.array([0,0]),np.array([0,2])),shape=(1,2))
    with pytest.raises(ValueError):ridge.centered_operator(duplicated)
    bad=design();bad.data[0]=np.nan
    with pytest.raises(ValueError):ridge.centered_operator(bad)
    with pytest.raises(ValueError):ridge.solve_absolute_ridge(design(),np.arange(7.),1.,settings=SETTINGS,limits=LIMITS)
    with pytest.raises(ValueError):ridge.solve_absolute_ridge(design(),np.full(8,np.inf),1.,settings=SETTINGS,limits=LIMITS)
    with pytest.raises(ValueError):ridge.interpolate_policy(np.zeros(2),0.,np.ones(2),1.,1.1)
