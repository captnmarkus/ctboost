"""Optional Arrow and Polars input adapters.

The adapters deliberately identify objects by their defining module and public
protocol instead of importing either dependency.  This keeps both libraries
optional and avoids adding import-time cost to ordinary NumPy/pandas users.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np


def _module_root(value: Any) -> str:
    return type(value).__module__.partition(".")[0]


def _is_arrow_frame(value: Any) -> bool:
    return (
        _module_root(value) == "pyarrow"
        and type(value).__name__ in {"Table", "RecordBatch"}
        and hasattr(value, "column_names")
        and hasattr(value, "num_rows")
        and hasattr(value, "num_columns")
    )


def _is_polars_frame(value: Any) -> bool:
    return (
        _module_root(value) == "polars"
        and type(value).__name__ == "DataFrame"
        and hasattr(value, "columns")
        and hasattr(value, "height")
        and hasattr(value, "width")
    )


def _is_polars_lazy_frame(value: Any) -> bool:
    return _module_root(value) == "polars" and type(value).__name__ == "LazyFrame"


def _is_cudf_frame(value: Any) -> bool:
    return (
        _module_root(value) == "cudf"
        and type(value).__name__ == "DataFrame"
        and hasattr(value, "columns")
        and hasattr(value, "shape")
    )


def _is_arrow_vector(value: Any) -> bool:
    return (
        _module_root(value) == "pyarrow"
        and type(value).__name__ in {"Array", "ChunkedArray"}
        and hasattr(value, "to_pylist")
    )


def _is_polars_series(value: Any) -> bool:
    return (
        _module_root(value) == "polars"
        and type(value).__name__ == "Series"
        and hasattr(value, "to_numpy")
    )


def _is_cudf_vector(value: Any) -> bool:
    return (
        _module_root(value) == "cudf"
        and type(value).__name__ in {"Series", "Index"}
        and (hasattr(value, "to_numpy") or hasattr(value, "to_pandas"))
    )


def _is_columnar_frame(value: Any) -> bool:
    return _is_arrow_frame(value) or _is_polars_frame(value) or _is_cudf_frame(value)


def _array_protocol_to_numpy(value: Any, *, dtype: Any = None) -> Any:
    """Copy supported array protocols to host NumPy without eager imports.

    CPU DLPack producers can be zero-copy. CUDA array producers (CuPy,
    PyTorch CUDA tensors, and similar objects) are copied explicitly because
    the native CTBoost Pool currently owns host memory before GPU training.
    """
    if value is None or isinstance(value, np.ndarray):
        return value

    if hasattr(value, "__cuda_array_interface__"):
        getter = getattr(value, "get", None)
        if callable(getter):
            return np.asarray(getter(), dtype=dtype)
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            host_value = cpu()
            detach = getattr(host_value, "detach", None)
            if callable(detach):
                host_value = detach()
            to_numpy = getattr(host_value, "numpy", None)
            if callable(to_numpy):
                return np.asarray(to_numpy(), dtype=dtype)
        raise TypeError(
            "CUDA array input must provide get() or cpu().numpy() for host transfer"
        )

    if hasattr(value, "__dlpack__"):
        try:
            return np.asarray(np.from_dlpack(value), dtype=dtype)
        except (BufferError, RuntimeError, TypeError):
            # Some tensor libraries expose DLPack only for their current
            # device. Fall through to their explicit host conversion path.
            cpu = getattr(value, "cpu", None)
            if callable(cpu):
                host_value = cpu()
                detach = getattr(host_value, "detach", None)
                if callable(detach):
                    host_value = detach()
                to_numpy = getattr(host_value, "numpy", None)
                if callable(to_numpy):
                    return np.asarray(to_numpy(), dtype=dtype)
            raise TypeError("DLPack input could not be transferred to host NumPy")

    return value


def _columnar_frame_metadata(value: Any) -> Optional[Tuple[int, int, List[str]]]:
    """Return ``(rows, columns, names)`` for a supported eager frame."""
    if _is_arrow_frame(value):
        return (
            int(value.num_rows),
            int(value.num_columns),
            [str(name) for name in value.column_names],
        )
    if _is_polars_frame(value):
        return (
            int(value.height),
            int(value.width),
            [str(name) for name in value.columns],
        )
    if _is_cudf_frame(value):
        return (
            int(value.shape[0]),
            int(value.shape[1]),
            [str(name) for name in value.columns],
        )
    return None


def _column_values(value: Any, column_index: int) -> Any:
    if _is_arrow_frame(value):
        column = value.column(column_index)
        try:
            return column.to_numpy(zero_copy_only=False)
        except (TypeError, ValueError, NotImplementedError):
            return column.to_pylist()
    if _is_cudf_frame(value):
        column = value[value.columns[column_index]]
        try:
            return column.to_numpy()
        except (TypeError, ValueError, NotImplementedError):
            return column.to_pandas().to_numpy(copy=False)
    column = value.get_column(value.columns[column_index])
    try:
        return column.to_numpy()
    except (TypeError, ValueError, NotImplementedError):
        return column.to_list()


def _columnar_frame_to_numpy(
    value: Any,
    *,
    dtype: Any = None,
    order: str = "C",
) -> np.ndarray:
    """Materialize a supported frame while preserving nested object cells."""
    if _is_polars_lazy_frame(value):
        raise TypeError(
            "Polars LazyFrame input is not eager; call collect() before passing it to CTBoost"
        )
    metadata = _columnar_frame_metadata(value)
    if metadata is None:
        raise TypeError("value is not a supported Arrow or Polars frame")
    row_count, column_count, _ = metadata
    resolved_dtype = np.dtype(object if dtype is None else dtype)
    matrix = np.empty((row_count, column_count), dtype=resolved_dtype, order=order)
    for column_index in range(column_count):
        values = _column_values(value, column_index)
        if resolved_dtype == np.dtype(object):
            # Assignment from equally-sized list/embedding cells can otherwise
            # be interpreted as a second array dimension by NumPy.
            values = list(values)
            if len(values) != row_count:
                raise ValueError("column length does not match the tabular row count")
            for row_index, cell in enumerate(values):
                matrix[row_index, column_index] = cell
        else:
            column = np.asarray(values, dtype=resolved_dtype)
            if column.ndim != 1 or column.shape[0] != row_count:
                raise ValueError(
                    "Arrow and Polars Pool columns must contain scalar numeric values; "
                    "use text_features or embedding_features for nested values"
                )
            matrix[:, column_index] = column
    return matrix


def _columnar_vector_to_numpy(value: Any, *, dtype: Any = None) -> Any:
    """Convert an optional columnar vector without changing unrelated inputs."""
    if _is_arrow_vector(value):
        try:
            array = value.to_numpy(zero_copy_only=False)
        except (TypeError, ValueError, NotImplementedError):
            array = value.to_pylist()
        return np.asarray(array, dtype=dtype)
    if _is_polars_series(value):
        return np.asarray(value.to_numpy(), dtype=dtype)
    if _is_cudf_vector(value):
        try:
            return np.asarray(value.to_numpy(), dtype=dtype)
        except (TypeError, ValueError, NotImplementedError):
            return np.asarray(value.to_pandas().to_numpy(copy=False), dtype=dtype)
    metadata = _columnar_frame_metadata(value)
    if metadata is not None:
        if metadata[1] != 1:
            raise ValueError("columnar vector inputs must contain exactly one column")
        return _columnar_frame_to_numpy(value, dtype=dtype)[:, 0]
    return _array_protocol_to_numpy(value, dtype=dtype)


__all__ = [
    "_columnar_frame_metadata",
    "_columnar_frame_to_numpy",
    "_columnar_vector_to_numpy",
    "_array_protocol_to_numpy",
    "_is_arrow_frame",
    "_is_arrow_vector",
    "_is_columnar_frame",
    "_is_cudf_frame",
    "_is_cudf_vector",
    "_is_polars_frame",
    "_is_polars_lazy_frame",
    "_is_polars_series",
]
