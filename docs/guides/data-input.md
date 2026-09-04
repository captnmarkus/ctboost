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

Canonical SciPy CSC columns are scanned linearly during quantization. Implicit
zeros and explicitly stored zeros/NaNs retain the same bin semantics as dense
input, while the sparse path avoids a per-row binary search and per-feature dense
scratch buffer. Training still materializes a dense feature-major bin matrix; this
is a quantization/ingestion optimization, not a sparse tree-storage engine.

CPU node histograms use deterministic feature-parallel workers for sufficiently
large nodes. `CTBOOST_NODE_HIST_THREADS` sets that worker budget (falling back to
`CTBOOST_HIST_THREADS`); `CTBOOST_NODE_HIST_MIN_PARALLEL_VALUES` and
`CTBOOST_NODE_HIST_MIN_VALUES_PER_WORKER` tune the work thresholds. Each feature
keeps its existing row accumulation order, so changing the worker count does not
change the model. Disk-backed external bins remain single-threaded because their
one-column cache is not thread-safe.

## Explicit pre-quantized CUDA input

`Pool.from_cuda_quantized(...)` is an expert, schema-bound path for keeping an
already-quantized CUDA matrix off the host:

```python
import cupy as cp
from ctboost import Pool, train

# `schema` comes from a compatible model's get_quantization_schema().
# `host_bins` was quantized according to that exact schema.
device_bins = cp.asarray(host_bins)  # uint8, or uint16 when a feature has >256 bins
pool = Pool.from_cuda_quantized(device_bins, schema, label=y)

model = train(
    pool,
    {"objective": "RMSE", "task_type": "GPU", "devices": "0"},
    num_boost_round=100,
)
```

The input must expose CUDA Array Interface version 3, be a two-dimensional C- or
Fortran-contiguous little-endian `uint8`/`uint16` array, reside on the single selected
training device, and contain only bin indices valid for the supplied schema. CTBoost
synchronizes the legacy-default or named producer stream, validates the declared span
against the CUDA allocation, and makes an owned device-to-device, feature-major copy;
feature values are never copied to host. The per-thread default stream marker is rejected
because copying is deferred until fit. Labels and row metadata remain host-side.

This first path deliberately rejects raw floating-point device quantization, DLPack,
multiple GPUs, distributed/external-memory training, evaluation sets, DART, callbacks,
warm starts, and device prediction. Ordinary `Pool(CuPy/cuDF)` retains its compatible
host-materialization behavior instead of silently opting into this narrow API.

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
