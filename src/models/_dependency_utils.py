"""Helpers for robust optional dependency imports."""

from __future__ import annotations


_BINARY_INCOMPATIBILITY_MARKERS = (
    "compiled using NumPy 1.x",
    "numpy.dtype size changed",
    "numpy.core.multiarray failed to import",
    "may indicate binary incompatibility",
    "module was compiled against",
)


def is_binary_incompatibility_error(exc: BaseException) -> bool:
    """Detect common NumPy ABI mismatch errors across chained exceptions."""
    visited: set[int] = set()
    current: BaseException | None = exc

    while current is not None and id(current) not in visited:
        visited.add(id(current))
        message = str(current)
        if any(marker in message for marker in _BINARY_INCOMPATIBILITY_MARKERS):
            return True
        current = current.__cause__ or current.__context__

    return False
