import numpy as np
import pytest

from ctboost import Pool
from ctboost.training._eval_sets import _resolve_eval_pool


@pytest.mark.parametrize("grouped", [False, True])
def test_eval_pool_helper_constructs_tuple_inputs(grouped):
    X = np.arange(12, dtype=np.float32).reshape(6, 2)
    y = np.arange(6, dtype=np.float32)
    groups = np.repeat(np.arange(3), 2)
    pool = _resolve_eval_pool((X, y, groups) if grouped else (X, y))
    assert isinstance(pool, Pool)
    np.testing.assert_array_equal(pool.label, y)
    if grouped:
        np.testing.assert_array_equal(pool.group_id, groups)
    assert _resolve_eval_pool(pool) is pool
    assert _resolve_eval_pool(None) is None
