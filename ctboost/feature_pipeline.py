"""Feature preprocessing wrappers around the native CTBoost pipeline."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import _core
from .core import Pool
from .core.columnar import (
    _array_protocol_to_numpy,
    _columnar_frame_metadata,
    _columnar_frame_to_numpy,
    _columnar_vector_to_numpy,
    _is_columnar_frame,
    _is_polars_lazy_frame,
)

try:
    import pandas as pd
except ImportError:  # pragma: no cover - pandas is optional at runtime
    pd = None


_FEATURE_PIPELINE_PICKLE_MAGIC = b"CTBOOST_FEATURE_PIPELINE_PICKLE\x00"
_FEATURE_PIPELINE_PICKLE_VERSION = 1


def _is_pandas_dataframe(value: Any) -> bool:
    return pd is not None and isinstance(value, pd.DataFrame)


def _embedding_stat_names(stats: Sequence[str]) -> Tuple[str, ...]:
    supported = {"mean", "std", "min", "max", "l2", "sum", "dim"}
    resolved = tuple(str(stat).lower() for stat in stats)
    for stat in resolved:
        if stat not in supported:
            raise ValueError(f"unsupported embedding stat: {stat}")
    return resolved


def _text_ngram_bounds(values: Sequence[int]) -> Tuple[int, int]:
    resolved = tuple(int(value) for value in values)
    if len(resolved) != 2 or resolved[0] <= 0 or resolved[1] < resolved[0]:
        raise ValueError(
            "text_ngram_range must be a (minimum, maximum) pair of positive integers"
        )
    return resolved


def _choice(value: str, *, name: str, choices: Sequence[str]) -> str:
    resolved = str(value).lower()
    if resolved not in choices:
        raise ValueError(f"{name} must be one of: {', '.join(choices)}")
    return resolved


def _feature_pipelines_equivalent(expected: Any, actual: Any) -> bool:
    """Return whether two fitted pipelines encode the same feature space."""
    if expected is actual:
        return True
    if expected is None or actual is None:
        return False
    try:
        return bool(expected.to_state() == actual.to_state())
    except (AttributeError, TypeError, ValueError):
        return False


class FeaturePipeline:
    """Native-backed preprocessing for categorical, text, and embedding features."""

    def __init__(
        self,
        *,
        cat_features: Optional[Sequence[Any]] = None,
        ordered_ctr: bool = False,
        one_hot_max_size: int = 0,
        max_cat_threshold: int = 0,
        categorical_combinations: Optional[Sequence[Sequence[Any]]] = None,
        pairwise_categorical_combinations: bool = False,
        simple_ctr: Optional[Sequence[str]] = None,
        combinations_ctr: Optional[Sequence[str]] = None,
        per_feature_ctr: Optional[Mapping[Any, Sequence[str]]] = None,
        text_features: Optional[Sequence[Any]] = None,
        text_hash_dim: int = 64,
        text_tokenizer: str = "word",
        text_ngram_range: Sequence[int] = (1, 1),
        text_lowercase: bool = True,
        text_min_token_count: int = 1,
        text_max_dictionary_size: int = 0,
        text_feature_calcer: str = "count",
        embedding_features: Optional[Sequence[Any]] = None,
        embedding_stats: Sequence[str] = ("mean", "std", "min", "max", "l2"),
        embedding_target_features: bool = False,
        embedding_target_regularization: float = 1.0,
        embedding_target_mode: str = "auto",
        ctr_prior_strength: float = 1.0,
        random_seed: int = 0,
    ) -> None:
        self.cat_features = None if cat_features is None else list(cat_features)
        self.ordered_ctr = bool(ordered_ctr)
        self.one_hot_max_size = int(one_hot_max_size)
        self.max_cat_threshold = int(max_cat_threshold)
        self.categorical_combinations = (
            None if categorical_combinations is None else [list(values) for values in categorical_combinations]
        )
        self.pairwise_categorical_combinations = bool(pairwise_categorical_combinations)
        self.simple_ctr = None if simple_ctr is None else [str(value) for value in simple_ctr]
        self.combinations_ctr = (
            None if combinations_ctr is None else [str(value) for value in combinations_ctr]
        )
        self.per_feature_ctr = (
            None
            if per_feature_ctr is None
            else {key: [str(value) for value in values] for key, values in per_feature_ctr.items()}
        )
        self.text_features = None if text_features is None else list(text_features)
        self.text_hash_dim = int(text_hash_dim)
        if self.text_hash_dim <= 0:
            raise ValueError("text_hash_dim must be positive")
        self.text_tokenizer = _choice(
            text_tokenizer,
            name="text_tokenizer",
            choices=("word", "whitespace", "character"),
        )
        self.text_ngram_range = _text_ngram_bounds(text_ngram_range)
        self.text_lowercase = bool(text_lowercase)
        self.text_min_token_count = int(text_min_token_count)
        if self.text_min_token_count <= 0:
            raise ValueError("text_min_token_count must be positive")
        self.text_max_dictionary_size = int(text_max_dictionary_size)
        if self.text_max_dictionary_size < 0:
            raise ValueError("text_max_dictionary_size must be non-negative")
        self.text_feature_calcer = _choice(
            text_feature_calcer,
            name="text_feature_calcer",
            choices=("count", "binary", "tfidf"),
        )
        self.embedding_features = None if embedding_features is None else list(embedding_features)
        self.embedding_stats = _embedding_stat_names(embedding_stats)
        self.embedding_target_features = bool(embedding_target_features)
        self.embedding_target_regularization = float(embedding_target_regularization)
        if self.embedding_target_regularization < 0.0:
            raise ValueError("embedding_target_regularization must be non-negative")
        self.embedding_target_mode = _choice(
            embedding_target_mode,
            name="embedding_target_mode",
            choices=("auto", "regression", "classification"),
        )
        self.ctr_prior_strength = float(ctr_prior_strength)
        self.random_seed = int(random_seed)

        self.feature_names_in_: Optional[List[str]] = None
        self.n_features_in_: Optional[int] = None
        self.cat_feature_indices_: List[int] = []
        self.output_feature_names_: List[str] = []

        self._native = _core.NativeFeaturePipeline(
            cat_features=self.cat_features,
            ordered_ctr=self.ordered_ctr,
            one_hot_max_size=self.one_hot_max_size,
            max_cat_threshold=self.max_cat_threshold,
            categorical_combinations=self.categorical_combinations,
            pairwise_categorical_combinations=self.pairwise_categorical_combinations,
            simple_ctr=self.simple_ctr,
            combinations_ctr=self.combinations_ctr,
            per_feature_ctr=self.per_feature_ctr,
            text_features=self.text_features,
            text_hash_dim=self.text_hash_dim,
            text_tokenizer=self.text_tokenizer,
            text_ngram_range=self.text_ngram_range,
            text_lowercase=self.text_lowercase,
            text_min_token_count=self.text_min_token_count,
            text_max_dictionary_size=self.text_max_dictionary_size,
            text_feature_calcer=self.text_feature_calcer,
            embedding_features=self.embedding_features,
            embedding_stats=list(self.embedding_stats),
            embedding_target_features=self.embedding_target_features,
            embedding_target_regularization=self.embedding_target_regularization,
            embedding_target_mode=self.embedding_target_mode,
            ctr_prior_strength=self.ctr_prior_strength,
            random_seed=self.random_seed,
        )

    @staticmethod
    def _extract_frame(
        data: Any,
        feature_names: Optional[Sequence[str]] = None,
    ) -> Tuple[np.ndarray, Optional[List[str]]]:
        if _is_pandas_dataframe(data):
            return data.to_numpy(dtype=object, copy=False), [str(name) for name in data.columns]
        if _is_columnar_frame(data):
            metadata = _columnar_frame_metadata(data)
            if metadata is None:  # pragma: no cover - guarded by the predicate
                raise TypeError("unsupported columnar frame")
            resolved_feature_names = (
                metadata[2] if feature_names is None else [str(name) for name in feature_names]
            )
            return (
                _columnar_frame_to_numpy(data, dtype=object),
                resolved_feature_names,
            )
        if _is_polars_lazy_frame(data):
            raise TypeError(
                "Polars LazyFrame input is not eager; call collect() before passing it to CTBoost"
            )
        array = np.asarray(_array_protocol_to_numpy(data), dtype=object)
        if array.ndim != 2:
            raise ValueError("feature pipelines expect a 2D array-like input")
        resolved_feature_names = None if feature_names is None else [str(name) for name in feature_names]
        return array, resolved_feature_names

    def _refresh_metadata(self) -> None:
        state = dict(self._native.to_state())
        self.feature_names_in_ = state.get("feature_names_in_")
        self.n_features_in_ = state.get("n_features_in_")
        self.cat_feature_indices_ = list(state.get("cat_feature_indices_", []))
        self.output_feature_names_ = list(state.get("output_feature_names_", []))

    def fit(
        self,
        data: Any,
        label: Any,
        *,
        feature_names: Optional[Sequence[str]] = None,
    ) -> "FeaturePipeline":
        raw_matrix, resolved_feature_names = self._extract_frame(data, feature_names)
        labels = np.asarray(
            _columnar_vector_to_numpy(label), dtype=np.float32
        ).reshape(-1)
        self._native.fit_array(raw_matrix, labels, resolved_feature_names)
        self._refresh_metadata()
        return self

    def transform_array(
        self,
        data: Any,
        *,
        feature_names: Optional[Sequence[str]] = None,
    ) -> Tuple[np.ndarray, List[int], List[str]]:
        raw_matrix, resolved_feature_names = self._extract_frame(data, feature_names)
        transformed, cat_features, output_feature_names = self._native.transform_array(
            raw_matrix,
            resolved_feature_names,
        )
        self._refresh_metadata()
        return transformed, list(cat_features), list(output_feature_names)

    def fit_transform_array(
        self,
        data: Any,
        label: Any,
        *,
        feature_names: Optional[Sequence[str]] = None,
    ) -> Tuple[np.ndarray, List[int], List[str]]:
        raw_matrix, resolved_feature_names = self._extract_frame(data, feature_names)
        labels = np.asarray(
            _columnar_vector_to_numpy(label), dtype=np.float32
        ).reshape(-1)
        transformed, cat_features, output_feature_names = self._native.fit_transform_array(
            raw_matrix,
            labels,
            resolved_feature_names,
        )
        self._refresh_metadata()
        return transformed, list(cat_features), list(output_feature_names)

    def transform_pool(
        self,
        data: Any,
        label: Any = None,
        *,
        weight: Any = None,
        group_id: Any = None,
        group_weight: Any = None,
        subgroup_id: Any = None,
        baseline: Any = None,
        pairs: Any = None,
        pairs_weight: Any = None,
        feature_names: Optional[Sequence[str]] = None,
        column_roles: Any = None,
        feature_metadata: Optional[Mapping[str, Any]] = None,
        categorical_schema: Optional[Mapping[str, Any]] = None,
    ) -> Pool:
        transformed, cat_features, output_feature_names = self.transform_array(
            data,
            feature_names=feature_names,
        )
        pool = Pool(
            data=transformed,
            label=label,
            cat_features=cat_features,
            weight=weight,
            group_id=group_id,
            group_weight=group_weight,
            subgroup_id=subgroup_id,
            baseline=baseline,
            pairs=pairs,
            pairs_weight=pairs_weight,
            feature_names=output_feature_names,
            column_roles=column_roles,
            feature_metadata=feature_metadata,
            categorical_schema=categorical_schema,
            _releasable_feature_storage=True,
        )
        pool._feature_pipeline = self
        return pool

    def to_state(self) -> Dict[str, Any]:
        return dict(self._native.to_state())

    def __getstate__(self) -> Tuple[bytes, int, Dict[str, Any]]:
        """Serialize the native pipeline through its stable state document.

        pybind11 extension objects are not pickleable by default.  Keeping the
        Python pickle contract on this wrapper also makes fitted sklearn models
        work with joblib, AutoGluon bagging, and multiprocessing spawn.
        """
        return (
            _FEATURE_PIPELINE_PICKLE_MAGIC,
            _FEATURE_PIPELINE_PICKLE_VERSION,
            self.to_state(),
        )

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, Mapping):
            # Historical CTBoost pickles stored either the native mapping
            # directly or wrapped it under ``native_state``.
            native_state = state.get("native_state", state)
        else:
            if not isinstance(state, tuple) or len(state) != 3:
                raise TypeError("invalid CTBoost FeaturePipeline pickle envelope")
            magic, version, native_state = state
            if magic != _FEATURE_PIPELINE_PICKLE_MAGIC:
                raise ValueError("invalid CTBoost FeaturePipeline pickle magic")
            if version != _FEATURE_PIPELINE_PICKLE_VERSION:
                raise ValueError(
                    "unsupported CTBoost FeaturePipeline pickle version: "
                    f"{version!r}"
                )
            if not isinstance(native_state, Mapping):
                raise TypeError("CTBoost FeaturePipeline pickle payload must be a mapping")
        restored = type(self).from_state(native_state)
        self.__dict__.update(restored.__dict__)

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "FeaturePipeline":
        pipeline = cls(
            cat_features=state.get("cat_features"),
            ordered_ctr=state.get("ordered_ctr", False),
            one_hot_max_size=state.get("one_hot_max_size", 0),
            max_cat_threshold=state.get("max_cat_threshold", 0),
            categorical_combinations=state.get("categorical_combinations"),
            pairwise_categorical_combinations=state.get("pairwise_categorical_combinations", False),
            simple_ctr=state.get("simple_ctr"),
            combinations_ctr=state.get("combinations_ctr"),
            per_feature_ctr=state.get("per_feature_ctr"),
            text_features=state.get("text_features"),
            text_hash_dim=state.get("text_hash_dim", 64),
            text_tokenizer=state.get("text_tokenizer", "word"),
            text_ngram_range=state.get("text_ngram_range", (1, 1)),
            text_lowercase=state.get("text_lowercase", True),
            text_min_token_count=state.get("text_min_token_count", 1),
            text_max_dictionary_size=state.get("text_max_dictionary_size", 0),
            text_feature_calcer=state.get("text_feature_calcer", "count"),
            embedding_features=state.get("embedding_features"),
            embedding_stats=state.get("embedding_stats", ("mean", "std", "min", "max", "l2")),
            embedding_target_features=state.get("embedding_target_features", False),
            embedding_target_regularization=state.get(
                "embedding_target_regularization", 1.0
            ),
            embedding_target_mode=state.get("embedding_target_mode", "auto"),
            ctr_prior_strength=state.get("ctr_prior_strength", 1.0),
            random_seed=state.get("random_seed", 0),
        )
        pipeline._native = _core.NativeFeaturePipeline.from_state(dict(state))
        pipeline._refresh_metadata()
        return pipeline
