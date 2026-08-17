"""
city/maps/terrain.py

Standardized terrain classification layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import config
from city.enums.terrain_type import TerrainType, classify_elevation
from city.maps.elevation import ElevationMap
from city.utils.validation import ensure_standard_map


@dataclass
class TerrainMap:
    """Discrete terrain classification for every cell in the world grid.

    An int8 array of shape (world_size, world_size) whose values are
    `TerrainType` members. Terrain classification is a pure function of
    elevation (see `classify_elevation`), so a TerrainMap can always be
    derived from any ElevationMap regardless of that elevation's origin.
    """

    data: np.ndarray
    world_size: int = field(default=config.WORLD_SIZE)

    def __post_init__(self) -> None:
        self.data = ensure_standard_map(
            self.data, self.world_size, "terrain", dtype=np.int8
        )

    @classmethod
    def from_elevation(
        cls,
        elevation: ElevationMap,
        sea_level: float = config.SEA_LEVEL,
        thresholds: dict[str, float] = config.TERRAIN_THRESHOLDS,
    ) -> "TerrainMap":
        """Derive a TerrainMap from an ElevationMap.

        Args:
            elevation: The elevation layer to classify.
            sea_level: Elevation below which a cell is OCEAN.
            thresholds: Ascending upper-bound thresholds for BEACH,
                PLAINS, and HILLS.

        Returns:
            A new TerrainMap.
        """
        classes = classify_elevation(elevation.data, sea_level, thresholds)
        return cls(classes, world_size=elevation.world_size)
