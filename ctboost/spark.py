"""PySpark DataFrame convenience adapter for CTBoost.

Training is deliberately an explicit driver-memory operation: Spark does not
offer a stable Python gang-scheduling contract for CTBoost's native TCP ranks.
Inference remains partitioned through a pandas UDF.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from ._integration_utils import prediction_columns, split_feature_frame


def _require_pyspark() -> Any:
    try:
        import pyspark
    except ImportError as exc:
        raise ImportError(
            "CTBoost's Spark adapter requires PySpark and Arrow. Install 'ctboost[spark]'."
        ) from exc
    return pyspark


def _vector_matrix(values: Any) -> np.ndarray:
    rows = []
    for value in values:
        if value is None:
            raise ValueError("Spark vector features cannot contain null values")
        to_array = getattr(value, "toArray", None)
        rows.append(np.asarray(to_array() if callable(to_array) else value, dtype=np.float32))
    if not rows:
        return np.empty((0, 0), dtype=np.float32)
    width = int(rows[0].size)
    if any(int(row.size) != width for row in rows):
        raise ValueError("Spark vector features must have a fixed width")
    return np.ascontiguousarray(np.vstack(rows), dtype=np.float32)


def _collect_training_frame(
    dataframe: Any,
    *,
    label_col: str,
    feature_cols: Optional[Sequence[str]],
    features_col: Optional[str],
    metadata_columns: Mapping[str, str],
) -> tuple[Any, Any, Dict[str, Any], Optional[Sequence[str]]]:
    if feature_cols is not None and features_col is not None:
        raise ValueError("pass either feature_cols or features_col, not both")
    selected = [label_col, *metadata_columns.values()]
    if features_col is not None:
        selected.append(features_col)
    elif feature_cols is not None:
        selected.extend(feature_cols)
    else:
        selected.extend(
            column
            for column in dataframe.columns
            if column not in {label_col, *metadata_columns.values()}
        )
    selected = list(dict.fromkeys(selected))
    frame = dataframe.select(*selected).toPandas()
    if features_col is not None:
        features = _vector_matrix(frame[features_col])
        labels = frame[label_col]
        metadata = {
            argument: frame[column]
            for argument, column in metadata_columns.items()
        }
        return features, labels, metadata, None
    features, labels, by_column = split_feature_frame(
        frame,
        label=label_col,
        feature_columns=feature_cols,
        metadata_columns=list(metadata_columns.values()),
    )
    metadata = {
        argument: by_column[column]
        for argument, column in metadata_columns.items()
        if column in by_column
    }
    return features, labels, metadata, list(features.columns)


@dataclass
class SparkCTBoostModel:
    """A fitted CTBoost booster with Spark DataFrame transform metadata."""

    booster: Any
    feature_cols: Optional[Sequence[str]] = None
    features_col: Optional[str] = None

    def transform(
        self,
        dataframe: Any,
        *,
        prediction_col: str = "prediction",
    ) -> Any:
        """Append raw CTBoost predictions using a partitioned Arrow pandas UDF."""

        _require_pyspark()
        try:
            import pandas as pd
            from pyspark.sql import functions as F
            from pyspark.sql.types import ArrayType, DoubleType
        except ImportError as exc:
            raise ImportError(
                "Spark CTBoost inference requires pandas and pyarrow on every executor"
            ) from exc
        columns = prediction_columns(self.booster.prediction_dimension, prediction_col)
        return_type = DoubleType() if len(columns) == 1 else ArrayType(DoubleType())
        payload = pickle.dumps(self.booster, protocol=pickle.HIGHEST_PROTOCOL)
        feature_cols = None if self.feature_cols is None else list(self.feature_cols)
        use_vector = self.features_col is not None

        @F.pandas_udf(return_type)
        def predict_batch(*series: Any) -> Any:
            model = pickle.loads(payload)
            if use_vector:
                features = _vector_matrix(series[0])
            else:
                features = pd.concat(
                    [column.reset_index(drop=True) for column in series],
                    axis=1,
                )
                features.columns = feature_cols
            values = np.asarray(model.predict(features), dtype=np.float64)
            if values.ndim == 1:
                return pd.Series(values)
            return pd.Series([row.tolist() for row in values])

        if use_vector:
            return dataframe.withColumn(prediction_col, predict_batch(F.col(str(self.features_col))))
        if not feature_cols:
            raise ValueError("feature_cols are required for non-vector Spark inference")
        return dataframe.withColumn(
            prediction_col,
            predict_batch(*(F.col(column) for column in feature_cols)),
        )

    def save_model(self, path: Any, *, model_format: Optional[str] = None) -> None:
        self.booster.save_model(path, model_format=model_format)


def train(
    dataframe: Any,
    params: Mapping[str, Any],
    *,
    label_col: str,
    feature_cols: Optional[Sequence[str]] = None,
    features_col: Optional[str] = None,
    num_boost_round: Optional[int] = None,
    mode: str = "collect",
    weight_col: Optional[str] = None,
    group_id_col: Optional[str] = None,
    group_weight_col: Optional[str] = None,
    subgroup_id_col: Optional[str] = None,
    baseline_col: Optional[str] = None,
    eval_set: Any = None,
    **train_kwargs: Any,
) -> SparkCTBoostModel:
    """Collect a Spark DataFrame on the driver and fit CTBoost.

    The required ``mode='collect'`` spelling makes the driver-memory boundary
    visible at the call site.  The returned model performs distributed Spark
    inference through :meth:`SparkCTBoostModel.transform`.
    """

    _require_pyspark()
    if str(mode).lower() != "collect":
        raise ValueError(
            "Spark training currently supports only mode='collect'; use Dask or Ray "
            "for native multi-worker CTBoost TCP training"
        )
    metadata_columns = {
        name: column
        for name, column in (
            ("weight", weight_col),
            ("group_id", group_id_col),
            ("group_weight", group_weight_col),
            ("subgroup_id", subgroup_id_col),
            ("baseline", baseline_col),
        )
        if column is not None
    }
    features, labels, metadata, resolved_feature_cols = _collect_training_frame(
        dataframe,
        label_col=str(label_col),
        feature_cols=feature_cols,
        features_col=features_col,
        metadata_columns=metadata_columns,
    )
    eager_eval_set = None
    if eval_set is not None:
        eval_features, eval_labels, _, _ = _collect_training_frame(
            eval_set,
            label_col=str(label_col),
            feature_cols=feature_cols,
            features_col=features_col,
            # Exclude training metadata columns from automatically selected
            # evaluation features when they are present in both frames.
            metadata_columns={
                name: column
                for name, column in metadata_columns.items()
                if column in eval_set.columns
            },
        )
        eager_eval_set = (eval_features, eval_labels)

    from .training import train as local_train

    booster = local_train(
        features,
        params,
        label=labels,
        num_boost_round=num_boost_round,
        eval_set=eager_eval_set,
        **metadata,
        **train_kwargs,
    )
    return SparkCTBoostModel(
        booster=booster,
        feature_cols=resolved_feature_cols,
        features_col=features_col,
    )


def predict(
    model: SparkCTBoostModel,
    dataframe: Any,
    *,
    prediction_col: str = "prediction",
) -> Any:
    return model.transform(dataframe, prediction_col=prediction_col)


__all__ = ["SparkCTBoostModel", "predict", "train"]
