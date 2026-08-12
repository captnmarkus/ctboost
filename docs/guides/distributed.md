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

Spark fitting is explicitly collect-to-driver and must be requested as such. Prediction
uses partitioned pandas UDFs. CTBoost does not currently claim native distributed Spark,
JVM, Kubernetes, TLS, or coordinator fault-tolerance parity with mature incumbents.
