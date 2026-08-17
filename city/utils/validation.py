"""
city/utils/validation.py

Shared validation helpers enforcing the framework's "standardized map"
contract: every spatial layer, regardless of where it came from, is a
NumPy array of a known dtype and shape (WORLD_SIZE x WORLD_SIZE).

Keeping this logic in one place means every class in `city/maps/` gets
the same guarantees without duplicating shape/dtype checks.
"""

from __future__ import annotations

import numpy as np


def ensure_standard_map(
    array: np.ndarray,
    world_size: int,
    name: str,
    dtype: type = np.float32,
) -> np.ndarray:
    """Validate shape and coerce dtype for a standardized spatial layer.

    Args:
        array: The raw array to validate.
        world_size: Expected width/height of the square grid.
        name: Human-readable name used in error messages (e.g. "elevation").
        dtype: The NumPy dtype the array should be coerced to.

    Returns:
        The array, cast to `dtype`.

    Raises:
        ValueError: if the array's shape does not match
            (world_size, world_size).
    """
    array = np.asarray(array)
    expected_shape = (world_size, world_size)
    if array.shape != expected_shape:
        raise ValueError(
            f"{name} map must have shape {expected_shape}, got {array.shape}"
        )
    return array.astype(dtype)


def is_normalized(array: np.ndarray, tolerance: float = 1e-4) -> bool:
    """Check whether all values in `array` fall within [0.0, 1.0].

    A small tolerance absorbs floating point noise from resizing or
    dtype conversion without allowing genuinely out-of-range data through.
    """
    return bool(array.min() >= -tolerance and array.max() <= 1.0 + tolerance)
