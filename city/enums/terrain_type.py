from enum import IntEnum

import numpy as np


class TerrainType(IntEnum):
    OCEAN = 0
    BEACH = 1
    PLAINS = 2
    HILLS = 3
    MOUNTAINS = 4


def classify_elevation(elevation, sea_level, thresholds):
    classes = np.full(elevation.shape, TerrainType.MOUNTAINS, dtype=np.int8)
    assigned = np.zeros(elevation.shape, dtype=bool)

    for terrain_type, upper_bound in (
        (TerrainType.BEACH, thresholds["BEACH"]),
        (TerrainType.PLAINS, thresholds["PLAINS"]),
        (TerrainType.HILLS, thresholds["HILLS"]),
    ):
        mask = (~assigned) & (elevation <= upper_bound)
        classes[mask] = terrain_type
        assigned |= mask

    classes[elevation < sea_level] = TerrainType.OCEAN
    return classes
