"""One-pass, disk-backed construction of :class:`ctboost.Pool` objects.

The native trainer consumes a random-access ``Pool``.  This module bridges
streaming data sources to that contract without retaining every input batch in
RAM: batches are validated one at a time, spooled to a private directory, and
assembled into NumPy memory maps owned by the resulting pool.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from ._prepared_data_support import ExternalMemoryBacking, _open_memmap
from .core import Pool


@dataclass(frozen=True)
class PoolBatch:
    """A feature batch and its optional row-aligned training metadata."""

    data: Any
    label: Any = None
    weight: Any = None
    group_id: Any = None
    group_weight: Any = None
    subgroup_id: Any = None
    baseline: Any = None


_BATCH_FIELDS = (
    "label",
    "weight",
    "group_id",
    "group_weight",
    "subgroup_id",
    "baseline",
)


def _coerce_batch(value: Any) -> PoolBatch:
    if isinstance(value, PoolBatch):
        return value
    if isinstance(value, Pool):
        return PoolBatch(
            data=value.data,
            label=value.label if value.label.size else None,
            weight=value.weight,
            group_id=value.group_id,
            group_weight=value.group_weight,
            subgroup_id=value.subgroup_id,
            baseline=value.baseline,
        )
    if isinstance(value, Mapping):
        if "data" not in value:
            raise ValueError("streaming batch mappings must contain a 'data' entry")
        unknown = set(value).difference({"data", *_BATCH_FIELDS})
        if unknown:
            names = ", ".join(sorted(str(name) for name in unknown))
            raise ValueError(f"unsupported streaming batch entries: {names}")
        return PoolBatch(**{name: value.get(name) for name in ("data", *_BATCH_FIELDS)})
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError("streaming batch tuples must be (data, label)")
        return PoolBatch(data=value[0], label=value[1])
    return PoolBatch(data=value)


def _metadata_presence(pool: Pool) -> Tuple[bool, ...]:
    return (
        bool(pool.label.size),
        pool.weight is not None,
        pool.group_id is not None,
        pool.group_weight is not None,
        pool.subgroup_id is not None,
        pool.baseline is not None,
    )


def _save_batch(path: Path, pool: Pool) -> None:
    values = {
        "data": np.asarray(pool.data, dtype=np.float32, order="F"),
        "label": pool.label if pool.label.size else np.empty(0, dtype=np.float32),
        "weight": pool.weight if pool.weight is not None else np.empty(0, dtype=np.float32),
        "group_id": pool.group_id if pool.group_id is not None else np.empty(0, dtype=np.int64),
        "group_weight": (
            pool.group_weight if pool.group_weight is not None else np.empty(0, dtype=np.float32)
        ),
        "subgroup_id": (
            pool.subgroup_id if pool.subgroup_id is not None else np.empty(0, dtype=np.int64)
        ),
        "baseline": pool.baseline if pool.baseline is not None else np.empty(0, dtype=np.float32),
    }
    np.savez(path, **values)


def _allocate_metadata(root: Path, total_rows: int, first: Mapping[str, np.ndarray]) -> dict[str, np.memmap]:
    arrays: dict[str, np.memmap] = {}
    specifications = {
        "label": (np.float32, (total_rows,)),
        "weight": (np.float32, (total_rows,)),
        "group_id": (np.int64, (total_rows,)),
        "group_weight": (np.float32, (total_rows,)),
        "subgroup_id": (np.int64, (total_rows,)),
    }
    for name, (dtype, shape) in specifications.items():
        if first[name].size:
            arrays[name] = _open_memmap(root / f"{name}.npy", shape, dtype=dtype)
    if first["baseline"].size:
        baseline = first["baseline"]
        shape = (total_rows,) if baseline.ndim == 1 else (total_rows, baseline.shape[1])
        arrays["baseline"] = _open_memmap(root / "baseline.npy", shape, dtype=np.float32)
    return arrays


def pool_from_batches(
    batches: Iterable[Any],
    *,
    directory: Optional[Any] = None,
    cat_features: Optional[Sequence[int]] = None,
    feature_names: Optional[Sequence[str]] = None,
    column_roles: Any = None,
    feature_metadata: Optional[Mapping[str, Any]] = None,
    categorical_schema: Optional[Mapping[str, Any]] = None,
    pairs: Any = None,
    pairs_weight: Any = None,
) -> Pool:
    """Build a disk-backed pool from a one-pass iterable of numeric batches.

    Each item may be a :class:`PoolBatch`, a ``{"data": ..., "label": ...}``
    mapping, a ``(data, label)`` tuple, an existing :class:`Pool`, or a feature
    matrix.  Optional row metadata must be either present in every batch or
    absent from every batch.  Batch schemas and baseline widths must match.

    The function is intentionally numeric.  Fit text/embedding/categorical
    preprocessing before streaming, or pass already prepared numeric batches.
    The returned pool keeps its assembled ``.npy`` files alive through an
    ``ExternalMemoryBacking`` object and exposes their directory as
    ``pool.streaming_directory``.
    """

    base = None if directory is None else Path(directory)
    if base is not None:
        base.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="ctboost-stream-", dir=base))
    staging = root / "staging"
    staging.mkdir()
    infer_pool_categoricals = cat_features is None
    infer_pool_feature_names = feature_names is None
    resolved_cat_features = [] if cat_features is None else [int(index) for index in cat_features]
    chunk_paths: list[Path] = []
    row_counts: list[int] = []
    resolved_feature_names = None if feature_names is None else [str(name) for name in feature_names]
    expected_columns: Optional[int] = None
    expected_presence: Optional[Tuple[bool, ...]] = None
    expected_baseline_dimension: Optional[int] = None
    data_map: Optional[np.memmap] = None
    metadata_maps: dict[str, np.memmap] = {}

    try:
        for batch_index, raw_batch in enumerate(batches):
            if isinstance(raw_batch, Pool):
                if batch_index == 0:
                    if infer_pool_categoricals:
                        resolved_cat_features = list(raw_batch.cat_features)
                    if infer_pool_feature_names and raw_batch.feature_names is not None:
                        resolved_feature_names = list(raw_batch.feature_names)
                else:
                    if (
                        infer_pool_categoricals
                        and list(raw_batch.cat_features) != resolved_cat_features
                    ):
                        raise ValueError(
                            "categorical feature indices must match across streaming Pool batches"
                        )
                    if (
                        infer_pool_feature_names
                        and raw_batch.feature_names is not None
                        and list(raw_batch.feature_names) != resolved_feature_names
                    ):
                        raise ValueError(
                            "feature names must match across streaming Pool batches"
                        )
            batch = _coerce_batch(raw_batch)
            pool = Pool(
                batch.data,
                batch.label,
                cat_features=resolved_cat_features,
                weight=batch.weight,
                group_id=batch.group_id,
                group_weight=batch.group_weight,
                subgroup_id=batch.subgroup_id,
                baseline=batch.baseline,
                feature_names=resolved_feature_names,
                column_roles=column_roles,
                feature_metadata=feature_metadata,
                categorical_schema=categorical_schema,
            )
            if pool.num_rows == 0:
                raise ValueError(f"streaming batch {batch_index} is empty")
            if expected_columns is None:
                expected_columns = pool.num_cols
                expected_presence = _metadata_presence(pool)
                if resolved_feature_names is None and pool.feature_names is not None:
                    resolved_feature_names = list(pool.feature_names)
                expected_baseline_dimension = (
                    None
                    if pool.baseline is None
                    else (1 if pool.baseline.ndim == 1 else int(pool.baseline.shape[1]))
                )
            elif pool.num_cols != expected_columns:
                raise ValueError(
                    f"streaming batch {batch_index} has {pool.num_cols} columns; "
                    f"expected {expected_columns}"
                )
            if _metadata_presence(pool) != expected_presence:
                raise ValueError(
                    "label, weight, group_id, group_weight, subgroup_id, and baseline "
                    "must each be present in every batch or absent from every batch"
                )
            baseline_dimension = (
                None
                if pool.baseline is None
                else (1 if pool.baseline.ndim == 1 else int(pool.baseline.shape[1]))
            )
            if baseline_dimension != expected_baseline_dimension:
                raise ValueError("baseline width must be identical in every streaming batch")
            path = staging / f"batch-{batch_index:08d}.npz"
            _save_batch(path, pool)
            chunk_paths.append(path)
            row_counts.append(pool.num_rows)

        if not chunk_paths or expected_columns is None:
            raise ValueError("batches must yield at least one non-empty batch")

        total_rows = int(sum(row_counts))
        data_map = _open_memmap(
            root / "data.npy",
            (total_rows, expected_columns),
            dtype=np.float32,
            fortran_order=True,
        )
        with np.load(chunk_paths[0], allow_pickle=False) as first_chunk:
            metadata_maps = _allocate_metadata(root, total_rows, first_chunk)

        offset = 0
        for path, row_count in zip(chunk_paths, row_counts):
            with np.load(path, allow_pickle=False) as chunk:
                row_slice = slice(offset, offset + row_count)
                data_map[row_slice, :] = chunk["data"]
                for name, destination in metadata_maps.items():
                    destination[row_slice, ...] = chunk[name]
            offset += row_count
        data_map.flush()
        for array in metadata_maps.values():
            array.flush()

        for path in chunk_paths:
            path.unlink()
        staging.rmdir()

        label = metadata_maps.get("label")
        result = Pool(
            np.load(root / "data.npy", mmap_mode="r"),
            None if label is None else np.load(root / "label.npy", mmap_mode="r"),
            cat_features=resolved_cat_features,
            weight=None if "weight" not in metadata_maps else np.load(root / "weight.npy", mmap_mode="r"),
            group_id=(
                None if "group_id" not in metadata_maps else np.load(root / "group_id.npy", mmap_mode="r")
            ),
            group_weight=(
                None
                if "group_weight" not in metadata_maps
                else np.load(root / "group_weight.npy", mmap_mode="r")
            ),
            subgroup_id=(
                None
                if "subgroup_id" not in metadata_maps
                else np.load(root / "subgroup_id.npy", mmap_mode="r")
            ),
            baseline=(
                None if "baseline" not in metadata_maps else np.load(root / "baseline.npy", mmap_mode="r")
            ),
            pairs=pairs,
            pairs_weight=pairs_weight,
            feature_names=resolved_feature_names,
            column_roles=column_roles,
            feature_metadata=feature_metadata,
            categorical_schema=categorical_schema,
            _releasable_feature_storage=True,
        )
        result._external_memory_backing = ExternalMemoryBacking(
            root, [data_map, *metadata_maps.values()]
        )
        result.streaming_directory = root
        result.streaming_batch_count = len(chunk_paths)
        return result
    except Exception:
        for array in [data_map, *metadata_maps.values()]:
            if array is None:
                continue
            flush = getattr(array, "flush", None)
            if callable(flush):
                flush()
            memory_map = getattr(array, "_mmap", None)
            close = getattr(memory_map, "close", None)
            if callable(close):
                close()
        shutil.rmtree(root, ignore_errors=True)
        raise


__all__ = ["PoolBatch", "pool_from_batches"]
