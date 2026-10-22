"""Independent tiny dense math oracles; no real datasets or model fits."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")
from benchmarks import fixed_multiclass_leaf_refit as helper


def fixture(vector):
    labels = np.array([0, 1, 2, 1, 0, 2, 1], dtype=np.int64)
    baseline = np.random.default_rng(934).normal(scale=.6, size=(7, 3))
    leaves = np.array([[3, 2], [3, 5], [7, 5], [7, 8], [11, 8], [3, 2], [11, 5]], dtype=np.int64)
    if vector:
        tree_classes = (-1, -1)
    else:
        leaves = np.column_stack([leaves[:, 0], leaves[:, 1], leaves[:, 0],
                                  leaves[:, 1], leaves[:, 1], leaves[:, 0]])
        tree_classes = (0, 1, 2, 0, 1, 2)
    return leaves, baseline, labels, tree_classes


def independent_basis(leaves, tree_classes, classes):
    """Build the documented class/tree/sorted-leaf basis without helper output."""
    columns, offsets = [], [0]
    for output in range(classes):
        for tree, assigned in enumerate(tree_classes):
            if assigned not in (-1, output):
                continue
            for leaf in sorted(set(leaves[:, tree].tolist())):
                column = np.zeros((len(leaves), classes), dtype=np.float64)
                column[:, output] = leaves[:, tree] == leaf
                columns.append(column)
        offsets.append(len(columns))
    return np.stack(columns, axis=2), np.asarray(offsets, dtype=np.int64)


def dense_oracle(coefficients, basis, baseline, labels, penalty, temperature):
    """Different representation: explicit n x K x P tensor and dense Hessian."""
    rows, _, dimensions = basis.shape
    logits = (baseline + np.einsum("nkp,p->nk", basis, coefficients)) / temperature
    maximum = logits.max(axis=1, keepdims=True)
    exponent = np.exp(logits - maximum)
    probabilities = exponent / exponent.sum(axis=1, keepdims=True)
    log_normalizer = maximum[:, 0] + np.log(exponent.sum(axis=1))
    value = np.mean(log_normalizer - logits[np.arange(rows), labels])
    value += penalty * coefficients.dot(coefficients) / (2 * rows)
    residual = probabilities.copy()
    residual[np.arange(rows), labels] -= 1
    gradient = np.einsum("nkp,nk->p", basis, residual) / (rows * temperature)
    gradient += penalty * coefficients / rows
    mean_basis = np.einsum("nkp,nk->np", basis, probabilities)
    hessian = (np.einsum("nkp,nk,nkq->pq", basis, probabilities, basis)
               - mean_basis.T @ mean_basis) / (rows * temperature * temperature)
    hessian += np.eye(dimensions) * (penalty / rows)
    return float(value), gradient, hessian


def dense_newton(basis, baseline, labels, penalty, temperature):
    coefficients = np.zeros(basis.shape[2], dtype=np.float64)
    for _ in range(60):
        value, gradient, hessian = dense_oracle(coefficients, basis, baseline, labels, penalty, temperature)
        if np.linalg.norm(gradient, ord=np.inf) < 1e-10:
            return coefficients
        step = np.linalg.solve(hessian, gradient)
        scale = 1.
        for _ in range(40):
            candidate = coefficients - scale * step
            candidate_value = dense_oracle(candidate, basis, baseline, labels, penalty, temperature)[0]
            if candidate_value <= value - 1e-4 * scale * float(gradient @ step):
                coefficients = candidate
                break
            scale *= .5
        else:
            raise AssertionError("Independent Newton line search failed")
    raise AssertionError("Independent Newton did not converge")


@pytest.mark.parametrize("vector", [False, True])
@pytest.mark.parametrize("temperature", [.45, 1., 2.3])
def test_objective_gradient_and_basis_against_independent_dense_oracle(vector, temperature):
    leaves, baseline, labels, tree_classes = fixture(vector)
    design = helper.make_design(leaves, baseline, tree_classes, temperature=temperature)
    basis, offsets = independent_basis(leaves, tree_classes, 3)
    np.testing.assert_array_equal(design["offsets"], offsets)
    for output in range(3):
        np.testing.assert_array_equal(design["matrices"][output].toarray(),
                                      basis[:, output, offsets[output]:offsets[output + 1]])
    coefficients = np.random.default_rng(531).normal(scale=.3, size=basis.shape[2])
    penalty = .7
    value, gradient = helper.softmax_objective(coefficients, design, labels, penalty)
    expected_value, expected_gradient, hessian = dense_oracle(coefficients, basis, baseline, labels, penalty, temperature)
    np.testing.assert_allclose(value, expected_value, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-11, atol=1e-12)
    assert np.linalg.eigvalsh(hessian).min() >= penalty / len(labels) - 1e-12
    epsilon = 2e-6
    finite_difference = np.empty_like(coefficients)
    for index in range(len(coefficients)):
        step = np.zeros_like(coefficients)
        step[index] = epsilon
        plus = helper.softmax_objective(coefficients + step, design, labels, penalty)[0]
        minus = helper.softmax_objective(coefficients - step, design, labels, penalty)[0]
        finite_difference[index] = (plus - minus) / (2 * epsilon)
    np.testing.assert_allclose(gradient, finite_difference, rtol=2e-6, atol=2e-9)


@pytest.mark.parametrize("vector", [False, True])
def test_solver_matches_independent_dense_newton_with_fixed_temperature(vector):
    leaves, baseline, labels, tree_classes = fixture(vector)
    temperature, penalty = .65, 1.3
    design = helper.make_design(leaves, baseline, tree_classes, temperature=temperature)
    basis, _ = independent_basis(leaves, tree_classes, 3)
    expected = dense_newton(basis, baseline, labels, penalty, temperature)
    coefficients, _ = helper.solve_coefficients(design, labels, penalty)
    actual_value, gradient, _ = dense_oracle(coefficients, basis, baseline, labels, penalty, temperature)
    expected_value = dense_oracle(expected, basis, baseline, labels, penalty, temperature)[0]
    assert np.linalg.norm(gradient, ord=np.inf) < 2e-6
    np.testing.assert_allclose(coefficients, expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(actual_value, expected_value, rtol=0, atol=2e-10)


@pytest.mark.parametrize("vector", [False, True])
def test_common_margin_shift_invariance_and_native_correction_support(vector):
    leaves, baseline, labels, tree_classes = fixture(vector)
    design = helper.make_design(leaves, baseline, tree_classes, temperature=1.7)
    shifted = helper.make_design(leaves, baseline + np.arange(len(labels))[:, None] * .25,
                                 tree_classes, temperature=1.7)
    basis, _ = independent_basis(leaves, tree_classes, 3)
    coefficients = np.random.default_rng(301).normal(size=basis.shape[2])
    first = helper.softmax_objective(coefficients, design, labels, 2.)
    second = helper.softmax_objective(coefficients, shifted, labels, 2.)
    np.testing.assert_allclose(first[0], second[0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(first[1], second[1], rtol=1e-12, atol=1e-12)
    updates = helper.corrections_by_tree(design, coefficients, shrinkage=.25)
    actual = helper.correction_values(leaves, updates, 3)
    expected = .25 * np.einsum("nkp,p->nk", basis, coefficients)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    for assigned, update in zip(tree_classes, updates):
        if assigned >= 0:
            for vector_update in update.values():
                np.testing.assert_array_equal(np.delete(vector_update, assigned), np.zeros(2))
