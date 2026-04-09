"""Python-side data container shells used by the public API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Pool:
    """Lightweight placeholder for the native training matrix.

    The C++/CUDA-backed storage is introduced in Phase 2. For Phase 1 this class keeps the
    public constructor shape stable while the build system and package layout are established.
    """

    data: Any
    label: Any = None
    cat_features: list[int] = field(default_factory=list)
    weight: Any = None
    feature_names: list[str] | None = None

    def __post_init__(self) -> None:
        self.cat_features = list(self.cat_features)
        if self.feature_names is not None:
            self.feature_names = list(self.feature_names)
