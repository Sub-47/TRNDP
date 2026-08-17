"""
city/enums/terrain_type.py

Defines the discrete terrain vocabulary used throughout the framework,
plus the single pure function that turns a normalized elevation array
into terrain classes.

Classification is elevation-in, terrain-out: it has no idea whether the
elevation came from Perlin noise, a raster PNG, or a hybrid mix of both.
Both `TerrainGenerator` (the procedural pipeline) and `TerrainMap` (the
source-agnostic map layer) call `classify_elevation` so this logic lives
in exactly one place.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np


class TerrainType(IntEnum):
    """Discrete terrain classes assigned to each grid cell.

    Stored as an IntEnum so terrain class maps can be compact integer
    NumPy arrays rather than arrays of Python objects.
    """

    OCEAN = 0
    BEACH = 1
    PLAINS = 2
    HILLS = 3
    MOUNTAINS = 4


def classify_elevation(
    elevation: np.ndarray,
    sea_level: float,
    thresholds: dict[str, float],
) -> np.ndarray:
    """Classify a normalized elevation array into TerrainType codes.

    Cells below `sea_level` are always OCEAN. Above sea level, a cell is
    assigned the first threshold (in ascending order) whose elevation it
    does not exceed; anything above the highest threshold becomes
    MOUNTAINS.

    Args:
        elevation: Normalized (0.0-1.0) elevation array of any shape.
        sea_level: Elevation below which a cell is OCEAN.
        thresholds: Ascending upper-bound thresholds for BEACH, PLAINS,
            and HILLS (see config.TERRAIN_THRESHOLDS).

    Returns:
        An int8 array of TerrainType values, same shape as `elevation`.
    """
    classes = np.full(elevation.shape, TerrainType.MOUNTAINS, dtype=np.int8)

    ordered_types = [
        (TerrainType.BEACH, thresholds["BEACH"]),
        (TerrainType.PLAINS, thresholds["PLAINS"]),
        (TerrainType.HILLS, thresholds["HILLS"]),
    ]

    assigned = np.zeros(elevation.shape, dtype=bool)
    for terrain_type, upper_bound in ordered_types:
        mask = (~assigned) & (elevation <= upper_bound)
        classes[mask] = terrain_type
        assigned |= mask

    water_mask = elevation < sea_level
    classes[water_mask] = TerrainType.OCEAN

    return classes
