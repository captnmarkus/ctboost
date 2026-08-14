# Distributed training

CTBoost's native distributed coordinator uses authenticated per-run TCP roots. Dask
and Ray adapters distribute data shards into that training path. Tokens are ephemeral
and removed from persisted model state.

Install only the integration you need:

```bash
python -m pip install "ctboost[dask]"
python -m pip install "ctboost[ray]"
python -m pip install "ctboost[spark]"
```

## Dask

```python
from ctboost.dask import train

model = train(client, dask_frame, label="target", params=params)
```

Dask training requires process workers. Thread-only workers are rejected because the
native fit occupies the Python call until a shard completes.

## Ray

```python
from ctboost.ray import train

model = train(dataset, label="target", params=params)
```

Ray can keep prediction partitioned through `map_batches`. Validate the exact Ray and
cluster version used in production; Windows Ray support remains upstream beta.

## Spark

`mode="auto"` uses native distributed training when both the DataFrame and Spark's
estimated barrier-task capacity permit at least two workers. It caps the worker count by
that estimate and otherwise fails closed; it never silently collects rows to the driver.
Spark remains authoritative when task CPU/GPU resources or dynamic allocation make the
estimate optimistic. Each barrier task owns one DataFrame partition and one CTBoost TCP
rank; only the root model returns to the driver. Prediction remains partitioned through
pandas UDFs.

```python
from ctboost.spark import train

model = train(
    frame.repartition(4),
    {"objective": "RMSE", "iterations": 200},
    label_col="target",
    feature_cols=["x0", "x1", "x2"],
    mode="distributed",
    num_workers=4,
)
predicted = model.transform(frame, prediction_col="ctboost_prediction")
```

Barrier mode requires at least two process tasks and matching partitions/workers. A
ranking group is remapped to an opaque, globally unique integer ID, then repartitioned
round-robin with an exact worker partitioner and sorted so it cannot straddle native
ranks. Ranking requires at least one distinct query group per requested worker; otherwise
training fails closed and asks for fewer workers or explicit collection. Spark GPU
resources are mapped to each task's visible device when the cluster supplies them.
Distributed evaluation sets are not implemented yet and fail closed.

Use `mode="collect"` only as an explicit driver-memory fallback for small data. CTBoost
does not yet claim mature Spark/JVM training, Kubernetes, TLS, elasticity, coordinator
fault recovery, or an NCCL/GPU-direct multi-node collective stack. The current native
coordinator remains a trusted-network TCP reference path.
