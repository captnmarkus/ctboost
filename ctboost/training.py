"""Training entry points for the Python API."""

from __future__ import annotations

from typing import Any


def train(*args: Any, **kwargs: Any) -> Any:
    """Placeholder training API.

    The native boosting loop will be implemented in later phases. The symbol exists now so the
    public package surface is stable from the first scaffold onward.
    """

    raise NotImplementedError(
        "CTBoost training is not implemented yet. Phase 1 only scaffolds the package and build "
        "system; core data structures and training arrive in later phases."
    )
