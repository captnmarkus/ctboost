"""Feature preprocessing wrappers around the native CTBoost pipeline."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import _core
from .core import Pool

try:
    import pandas as pd
except ImportError:  # pragma: no cover - pandas is optional at runtime
    pd = None


def _is_pandas_dataframe(value: Any) -> bool:
    return pd is not None and isinstance(value, pd.DataFrame)


def _embedding_stat_names(stats: Sequence[str]) -> Tuple[str, ...]:
    supported = {"mean", "std", "min", "max", "l2", "sum", "dim"}
    resolved = tuple(str(stat).lower() for stat in stats)
    for stat in resolved:
        if stat not in supported:
            raise ValueError(f"unsupported embedding stat: {stat}")
    return resolved


class FeaturePipeline:
    """Native-backed preprocessing for categorical, text, and embedding features."""

    def __init__(
        self,
        *,
        cat_features: Optional[Sequence[Any]] = None,
        ordered_ctr: bool = False,
        categorical_combinations: Optional[Sequence[Sequence[Any]]] = None,
        pairwise_categorical_combinations: bool = False,
        text_features: Optional[Sequence[Any]] = None,
        text_hash_dim: int = 64,
        embedding_features: Optional[Sequence[Any]] = None,
        embedding_stats: Sequence[str] = ("mean", "std", "min", "max", "l2"),
        ctr_prior_strength: float = 1.0,
        random_seed: int = 0,
    ) -> None:
        self.cat_features = None if cat_features is None else list(cat_features)
        self.ordered_ctr = bool(ordered_ctr)
        self.categorical_combinations = (
            None if categorical_combinations is None else [list(values) for values in categorical_combinations]
        )
        self.pairwise_categorical_combinations = bool(pairwise_categorical_combinations)
        self.text_features = None if text_features is None else list(text_features)
        self.text_hash_dim = int(text_hash_dim)
        self.embedding_features = None if embedding_features is None else list(embedding_features)
        self.embedding_stats = _embedding_stat_names(embedding_stats)
        self.ctr_prior_strength = float(ctr_prior_strength)
        self.random_seed = int(random_seed)

        self.feature_names_in_: Optional[List[str]] = None
        self.n_features_in_: Optional[int] = None
        self.cat_feature_indices_: List[int] = []
        self.output_feature_names_: List[str] = []

        self._native = _core.NativeFeaturePipeline(
            cat_features=self.cat_features,
            ordered_ctr=self.ordered_ctr,
            categorical_combinations=self.categorical_combinations,
            pairwise_categorical_combinations=self.pairwise_categorical_combinations,
            text_features=self.text_features,
            text_hash_dim=self.text_hash_dim,
            embedding_features=self.embedding_features,
            embedding_stats=list(self.embedding_stats),
            ctr_prior_strength=self.ctr_prior_strength,
            random_seed=self.random_seed,
        )

    @staticmethod
    def _extract_frame(data: Any) -> Tuple[np.ndarray, Optional[List[str]]]:
        if _is_pandas_dataframe(data):
            return data.to_numpy(dtype=object, copy=False), [str(name) for name in data.columns]
        array = np.asarray(data, dtype=object)
        if array.ndim != 2:
            raise ValueError("feature pipelines expect a 2D array-like input")
        return array, None

    def _refresh_metadata(self) -> None:
        state = dict(self._native.to_state())
        self.feature_names_in_ = state.get("feature_names_in_")
        self.n_features_in_ = state.get("n_features_in_")
        self.cat_feature_indices_ = list(state.get("cat_feature_indices_", []))
        self.output_feature_names_ = list(state.get("output_feature_names_", []))

    def fit(self, data: Any, label: Any) -> "FeaturePipeline":
        raw_matrix, feature_names = self._extract_frame(data)
        labels = np.asarray(label, dtype=np.float32).reshape(-1)
        self._native.fit_array(raw_matrix, labels, feature_names)
        self._refresh_metadata()
        return self

    def transform_array(self, data: Any) -> Tuple[np.ndarray, List[int], List[str]]:
        raw_matrix, feature_names = self._extract_frame(data)
        transformed, cat_features, output_feature_names = self._native.transform_array(
            raw_matrix,
            feature_names,
        )
        self._refresh_metadata()
        return transformed, list(cat_features), list(output_feature_names)

    def fit_transform_array(self, data: Any, label: Any) -> Tuple[np.ndarray, List[int], List[str]]:
        raw_matrix, feature_names = self._extract_frame(data)
        labels = np.asarray(label, dtype=np.float32).reshape(-1)
        transformed, cat_features, output_feature_names = self._native.fit_transform_array(
            raw_matrix,
            labels,
            feature_names,
        )
        self._refresh_metadata()
        return transformed, list(cat_features), list(output_feature_names)

    def transform_pool(self, data: Any, label: Any, *, weight: Any = None, group_id: Any = None) -> Pool:
        transformed, cat_features, feature_names = self.transform_array(data)
        return Pool(
            data=transformed,
            label=label,
            cat_features=cat_features,
            weight=weight,
            group_id=group_id,
            feature_names=feature_names,
            _releasable_feature_storage=True,
        )

    def to_state(self) -> Dict[str, Any]:
        return dict(self._native.to_state())

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "FeaturePipeline":
        pipeline = cls(
            cat_features=state.get("cat_features"),
            ordered_ctr=state.get("ordered_ctr", False),
            categorical_combinations=state.get("categorical_combinations"),
            pairwise_categorical_combinations=state.get("pairwise_categorical_combinations", False),
            text_features=state.get("text_features"),
            text_hash_dim=state.get("text_hash_dim", 64),
            embedding_features=state.get("embedding_features"),
            embedding_stats=state.get("embedding_stats", ("mean", "std", "min", "max", "l2")),
            ctr_prior_strength=state.get("ctr_prior_strength", 1.0),
            random_seed=state.get("random_seed", 0),
        )
        pipeline._native = _core.NativeFeaturePipeline.from_state(dict(state))
        pipeline._refresh_metadata()
        return pipeline
