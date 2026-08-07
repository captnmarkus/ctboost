import numpy as np
import pytest

import ctboost
from ctboost.training._eval_sets import _slice_pool

scipy_sparse = pytest.importorskip("scipy.sparse")


def test_pool_canonicalizes_unsorted_duplicate_csc_without_mutating_input():
    # Column 0 contains unsorted rows and a duplicate row 2.  Column 1 is
    # already canonical.  Duplicate values should be summed like SciPy's
    # normal sparse arithmetic.
    matrix = scipy_sparse.csc_matrix(
        (
            np.array([3.0, 1.0, 4.0, 5.0], dtype=np.float32),
            np.array([2, 0, 2, 1], dtype=np.int32),
            np.array([0, 3, 4], dtype=np.int32),
        ),
        shape=(3, 2),
    )
    original_indices = matrix.indices.copy()
    original_data = matrix.data.copy()

    pool = ctboost.Pool(matrix, np.array([0.0, 1.0, 0.0], dtype=np.float32))

    np.testing.assert_allclose(
        pool.data,
        np.array([[1.0, 0.0], [0.0, 5.0], [7.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(matrix.indices, original_indices)
    np.testing.assert_array_equal(matrix.data, original_data)


def test_native_pool_rejects_noncanonical_csc_components():
    with pytest.raises(ValueError, match="sorted and unique"):
        ctboost._core.Pool.from_csc(
            np.array([3.0, 1.0], dtype=np.float32),
            np.array([1, 0], dtype=np.int64),
            np.array([0, 2], dtype=np.int64),
            2,
            1,
            np.array([0.0, 1.0], dtype=np.float32),
        )


def test_cv_pool_slicing_preserves_sparse_storage():
    matrix = scipy_sparse.random(
        20,
        5,
        density=0.2,
        format="csr",
        random_state=7,
        dtype=np.float32,
    )
    labels = np.arange(20, dtype=np.float32)

    sliced = _slice_pool(ctboost.Pool(matrix, labels), np.array([1, 4, 8, 13]))

    assert sliced._handle.is_sparse()
    np.testing.assert_allclose(sliced.data, matrix[[1, 4, 8, 13]].toarray())
    np.testing.assert_array_equal(sliced.label, labels[[1, 4, 8, 13]])
