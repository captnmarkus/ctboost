"""Inactive numerical prototype: absolute ridge with a free intercept.

Minimize .5*||y - b*1 - Z*w||^2 + .5*penalty*||w||^2 on a fixed CSR design.
This module has no dataset/model loading, native folding, protocol or CLI.
It does not modify the existing offset-ridge helpers. Numerical certificates
are floating-point diagnostics, not rigorous interval-arithmetic bounds.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, lsqr


@dataclass(frozen=True)
class LSQRSettings:
    """Every solver setting is explicit; no data-dependent budget changes."""

    iterations: int
    atol: float
    btol: float
    conlim: float

    def validate(self):
        if type(self.iterations) is not int or self.iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if any(not math.isfinite(v) or v <= 0 for v in (self.atol, self.btol, self.conlim)):
            raise ValueError("solver tolerances and conlim must be finite and positive")


@dataclass(frozen=True)
class CertificateLimits:
    """Prospectively supplied limits, separate from native solver termination."""

    relative_gap: float
    normalized_gradient: float
    normalized_intercept: float
    normalized_dual_feasibility: float

    def validate(self):
        if any(not math.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("certificate limits must be finite and positive")


def _design(Z):
    if not sparse.isspmatrix_csr(Z) or Z.dtype != np.dtype(np.float64):
        raise TypeError("Z must be a float64 scipy CSR matrix")
    if min(Z.shape) <= 0 or not Z.has_canonical_format or not Z.has_sorted_indices:
        raise ValueError("Z must have positive dimensions and canonical sorted indices")
    if not np.isfinite(Z.data).all():
        raise ValueError("Z contains nonfinite values")
    return Z


def _vector(values, length, name):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size != length or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite vector with the required length")
    return array


def _positive_penalty(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("penalty must be finite and strictly positive")
    return value


def _checked(values, name):
    if not np.isfinite(values).all():
        raise FloatingPointError(f"nonfinite {name}")
    return values


def _mean(values):
    # A reference offset makes an exactly constant floating vector stay exact,
    # including non-power-of-two lengths. Reject overflowing differences rather
    # than silently lose a finite-precision stationarity check.
    reference = float(values[0])
    differences = _checked(values - reference, "mean reference differences")
    offset = _checked(np.sum(differences / values.size, dtype=np.float64), "mean offset")
    return float(_checked(reference + offset, "mean"))


def centered_operator(Z):
    """Return A=PZ and p=Z.T*1/n without a dense centered matrix or Gram.

    Matvec and rmatvec are mathematical adjoints. Floating-point centering and
    reductions are separately checked by synthetic adjoint/certificate tests.
    """
    Z = _design(Z)
    n, p = Z.shape
    means = _checked(np.asarray(Z.T @ np.full(n, 1.0 / n), dtype=np.float64).reshape(p), "column means")

    def matvec(weights):
        weights = np.asarray(weights, dtype=np.float64).reshape(p)
        raw = _checked(np.asarray(Z @ weights).reshape(n), "centered matvec input")
        return _checked(raw - _mean(raw), "centered matvec")

    def rmatvec(values):
        values = np.asarray(values, dtype=np.float64).reshape(n)
        centered = _checked(values - _mean(values), "centered transpose input")
        return _checked(np.asarray(Z.T @ centered).reshape(p), "centered rmatvec")

    return LinearOperator((n, p), matvec=matvec, rmatvec=rmatvec, dtype=np.float64), means


def _ratio(numerator, denominator):
    if denominator == 0:
        return 0.0 if numerator == 0 else math.inf
    return numerator / denominator


def _number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def certify(Z, y, weights, bias, penalty, *, limits):
    """Check original joint objective, centered gradient and intercept residual.

    The exact dual constraint is 1.T*u=0. Rounded u=e-mean(e) may not satisfy
    it exactly. We expose that residual and its bias-amplified contribution.
    stable_gap_terms is a nonnegative floating-point convergence diagnostic;
    it is not represented as a formal dual upper bound or interval proof.
    """
    Z = _design(Z)
    y, weights = _vector(y, Z.shape[0], "y"), _vector(weights, Z.shape[1], "weights")
    bias, penalty = float(bias), _positive_penalty(penalty)
    if not math.isfinite(bias) or not isinstance(limits, CertificateLimits):
        raise ValueError("finite bias and explicit CertificateLimits required")
    limits.validate()
    n = Z.shape[0]
    raw = _checked(np.asarray(Z @ weights), "raw prediction")
    # Subtract the intercept first to avoid adding a large constant to each raw leaf sum.
    residual = _checked((y - bias) - raw, "original residual")
    residual_mean = _mean(residual)
    dual_vector = _checked(residual - residual_mean, "centered dual vector")
    ztu = _checked(np.asarray(Z.T @ dual_vector), "dual transpose product")
    gradient = _checked(ztu - penalty * weights, "centered coefficient gradient")
    mean_y = _mean(y)
    centered_y = _checked(y - mean_y, "centered target")
    centered_raw = _checked(raw - _mean(raw), "centered prediction")
    rhs = _checked(np.asarray(Z.T @ centered_y), "centered right hand side")
    gradient_norm = float(np.linalg.norm(gradient))
    coefficient_term = gradient_norm**2 / (2 * penalty)
    projection_term = float(np.dot(residual - dual_vector, residual - dual_vector)) / 2
    intercept_term = n * residual_mean**2 / 2
    stable_gap = coefficient_term + projection_term
    primal = (float(np.dot(residual, residual)) + penalty * float(np.dot(weights, weights))) / 2
    primal_zero = float(np.dot(centered_y, centered_y)) / 2
    # fsum exposes the sum of the stored floating values more accurately than
    # the reduction used by an arbitrary BLAS dot product.
    dual_sum = math.fsum(float(value) for value in dual_vector)
    ytu = float(np.dot(y, dual_vector))
    dual_residual_term = float(np.dot(dual_vector, dual_vector)) / 2
    dual_coefficient_term = float(np.dot(ztu, ztu)) / (2 * penalty)
    dual = ytu - dual_residual_term - dual_coefficient_term
    centered_dual = float(np.dot(centered_y, dual_vector)) - dual_residual_term - dual_coefficient_term
    shift_term = bias * dual_sum
    direct_gap = primal - dual
    reconstructed_gap = stable_gap - shift_term
    roundoff_scale = np.finfo(np.float64).eps * (
        abs(primal) + abs(ytu) + abs(dual_residual_term) + abs(dual_coefficient_term)
        + abs(stable_gap) + abs(shift_term))
    intercept_scale = max(float(np.linalg.norm(centered_y)), float(np.linalg.norm(centered_raw))) / math.sqrt(n)
    ratios = {
        "relative_gap": _ratio(stable_gap, max(primal, np.finfo(np.float64).eps * primal_zero)),
        "normalized_gradient": _ratio(gradient_norm, float(np.linalg.norm(rhs)) + penalty * float(np.linalg.norm(weights))),
        "normalized_intercept": _ratio(abs(residual_mean), intercept_scale),
        "normalized_dual_feasibility": _ratio(abs(dual_sum), math.sqrt(n) * float(np.linalg.norm(dual_vector))),
    }
    scalars = {
        "primal": primal, "primal_zero_leaves_fitted_intercept": primal_zero,
        "dual_direct": dual, "dual_using_centered_target": centered_dual,
        "direct_primal_minus_dual": direct_gap, "stable_gap_terms": stable_gap,
        "coefficient_gradient_norm": gradient_norm, "coefficient_gradient_gap_term": coefficient_term,
        "projection_gap_term": projection_term, "analytic_intercept_gap_term": intercept_term,
        "residual_mean": residual_mean, "intercept_gradient_sum": math.fsum(float(v) for v in residual),
        "dual_vector_sum": dual_sum, "dual_feasibility_bias_product": shift_term,
        "gap_reconstructed_with_dual_constraint_residual": reconstructed_gap,
        "direct_gap_minus_reconstructed_gap": direct_gap - reconstructed_gap,
        "gap_cancellation_roundoff_scale": roundoff_scale,
    }
    finite = all(math.isfinite(float(v)) for v in (*scalars.values(), *ratios.values()))
    gates = {name: math.isfinite(value) and value <= getattr(limits, name) for name, value in ratios.items()}
    return {**{k: _number(v) for k, v in scalars.items()}, **{k: _number(v) for k, v in ratios.items()},
        "finite": finite, "gates": gates, "certified": finite and all(gates.values()),
        "limits": asdict(limits), "arithmetic": "float64; rounded dual feasibility explicitly reported",
        "rigorous_interval_bound": False, "no_dense_centered_design_or_gram": True}


def solve_absolute_ridge(Z, y, penalty, *, settings, limits):
    """Cold, unscaled LSQR with damping sqrt(penalty) and a free intercept."""
    Z = _design(Z)
    y, penalty = _vector(y, Z.shape[0], "y"), _positive_penalty(penalty)
    if not isinstance(settings, LSQRSettings) or not isinstance(limits, CertificateLimits):
        raise TypeError("explicit LSQRSettings and CertificateLimits required")
    settings.validate()
    limits.validate()
    operator, means = centered_operator(Z)
    mean_y = _mean(y)
    centered_y = _checked(y - mean_y, "centered target")
    result = lsqr(operator, centered_y, damp=math.sqrt(penalty), atol=settings.atol, btol=settings.btol,
                  conlim=settings.conlim, iter_lim=settings.iterations, show=False, calc_var=False, x0=None)
    weights = _checked(np.asarray(result[0], dtype=np.float64), "solved coefficients")
    raw = _checked(np.asarray(Z @ weights), "solved raw prediction")
    bias = _mean(_checked(y - raw, "intercept target residual"))
    certificate = certify(Z, y, weights, bias, penalty, limits=limits)
    status, iterations = int(result[1]), int(result[2])
    native_values = {"residual_norm": result[3], "regularized_residual_norm": result[4],
        "matrix_norm": result[5], "condition_estimate": result[6], "normal_residual_norm": result[7], "solution_norm": result[8]}
    native_finite = all(math.isfinite(float(v)) for v in native_values.values())
    success = native_finite and status in (0, 1, 2, 4, 5) and certificate["certified"]
    receipt = {"success": bool(success), "objective": "0.5*sum((y-b-Zw)^2)+0.5*penalty*sum(w^2)",
        "penalty": penalty, "rows": Z.shape[0], "columns": Z.shape[1], "nonzeros": Z.nnz,
        "method": "cold unscaled LSQR on PZ with damping sqrt(penalty)", "initial_coefficients": "zero",
        "settings": asdict(settings), "native": {"status": status, "iterations": iterations,
            "finite": native_finite, **{k: _number(v) for k,v in native_values.items()}},
        "certificate": certificate, "centered_operator": "matvec=Zw-mean(Zw); rmatvec=Z.T@(v-mean(v))",
        "intercept_arithmetic": "mean(y-Zw), equivalent to mean(y)-column_means.T*w in exact arithmetic",
        "prototype_only": True, "native_model_fold_performed": False,
        "limit": "Success is floating-point numerical evidence, not an interval proof or a generalization claim"}
    return weights, bias, receipt


def interpolate_policy(original_weights, original_bias, optimized_weights, optimized_bias, shrinkage):
    """Interpolate both intercept and weights; zero preserves exact float bits.

    This is coefficient arithmetic only. Native q=0 serving must bypass any
    export/folding path; no claim of native-model identity is made here.
    """
    original = np.asarray(original_weights, dtype=np.float64)
    if original.ndim != 1:
        raise ValueError("original weights must be a vector")
    original = _vector(original, original.size, "original weights")
    optimized = _vector(optimized_weights, original.size, "optimized weights")
    original_bias, optimized_bias, shrinkage = float(original_bias), float(optimized_bias), float(shrinkage)
    if not all(math.isfinite(v) for v in (original_bias, optimized_bias, shrinkage)) or not 0 <= shrinkage <= 1:
        raise ValueError("finite intercepts and shrinkage in [0,1] required")
    if shrinkage == 0:
        return original.copy(), original_bias
    if shrinkage == 1:
        return optimized.copy(), optimized_bias
    weights = _checked((1-shrinkage)*original + shrinkage*optimized, "interpolated weights")
    bias = float(_checked((1-shrinkage)*original_bias + shrinkage*optimized_bias, "interpolated intercept"))
    return weights, bias
