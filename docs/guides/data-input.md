# Data input

## Eager matrices and columnar frames

`Pool` accepts NumPy and CPU DLPack arrays, pandas DataFrames, SciPy sparse
matrices, PyArrow tables, Polars DataFrames, and cuDF DataFrames. Arrow and
Polars support can be installed together with:

```bash
python -m pip install "ctboost[dataframes]"
```

Pass feature names and training metadata when constructing the pool. The same
schema is then available to training, persistence, and deployment tooling.

## Streaming numeric batches

For a one-pass source, build a disk-backed numeric pool without retaining every
source batch in memory:

```python
from ctboost import PoolBatch, pool_from_batches, train


def batches():
    for features, target in source:
        yield PoolBatch(features, target)


pool = pool_from_batches(batches(), directory="./ctboost-spill")
model = train(pool, {"iterations": 200, "depth": 7})
```

Batch items may also be `(data, label)` tuples, mappings, existing pools, or
feature matrices. The column count, feature names, categorical indices, and
baseline width must agree across batches. Labels, weights, group metadata, and
baselines must each be present in every batch or absent from every batch.

The streaming bridge is numeric. Apply categorical, text, or embedding
preprocessing before yielding batches. `Pool.from_batches(...)` is the
equivalent convenience constructor.

## Schema metadata

Attach stable names, roles, and application metadata directly to a pool:

```python
import ctboost

pool = ctboost.Pool(
    X,
    y,
    feature_names=["score", "ratio", "city_code"],
    column_roles=["numeric", "numeric", "categorical"],
    feature_metadata={"score": {"description": "normalized score"}},
    categorical_schema={
        "city_code": {"categories": ["berlin", "paris", "rome"]}
    },
)

booster = ctboost.train(pool, {"objective": "RMSE"}, num_boost_round=16)
print(booster.data_schema)
```

The scikit-learn estimators expose the persisted schema through
`data_schema_`. Schema metadata also participates in snapshot/resume validation
and is included in inference manifests.
