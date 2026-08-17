"""
city/maps/obstacle.py

Standardized traversability layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import config
from city.enums.terrain_type import TerrainType
from city.maps.terrain import TerrainMap
from city.utils.validation import ensure_standard_map


@dataclass
class ObstacleMap:
    """Boolean traversability mask for every cell in the world grid.

    True marks a cell that movement-dependent future modules (road
    generation, transit routing) should treat as non-traversable. This
    milestone derives obstacles from terrain classification alone, but
    the type says nothing about *how* a cell became an obstacle - a
    future module could mark cells from ownership boundaries or GIS data
    without any downstream code changing.
    """

    data: np.ndarray
    world_size: int = field(default=config.WORLD_SIZE)

    def __post_init__(self) -> None:
        self.data = ensure_standard_map(
            self.data, self.world_size, "obstacle", dtype=bool
        )

    @classmethod
    def from_terrain(
        cls,
        terrain: TerrainMap,
        obstacle_types: list[str] = config.OBSTACLE_TERRAIN_TYPES,
    ) -> "ObstacleMap":
        """Derive an ObstacleMap by flagging specific terrain classes.

        Args:
            terrain: The terrain layer to derive obstacles from.
            obstacle_types: TerrainType member names treated as
                non-traversable (see config.OBSTACLE_TERRAIN_TYPES).

        Returns:
            A new ObstacleMap.
        """
        obstacle_values = [TerrainType[name].value for name in obstacle_types]
        mask = np.isin(terrain.data, obstacle_values)
        return cls(mask, world_size=terrain.world_size)
