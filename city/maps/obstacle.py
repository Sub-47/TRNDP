from dataclasses import dataclass, field

import numpy as np

import config
from city.enums.terrain_type import TerrainType
from city.maps.terrain import TerrainMap
from city.utils.validation import ensure_standard_map


@dataclass
class ObstacleMap:
    data: np.ndarray
    world_size: int = field(default=config.WORLD_SIZE)

    def __post_init__(self):
        self.data = ensure_standard_map(self.data, self.world_size, "obstacle", bool)

    @classmethod
    def from_terrain(cls, terrain, obstacle_types=config.OBSTACLE_TERRAIN_TYPES):
        values = [TerrainType[name].value for name in obstacle_types]
        return cls(np.isin(terrain.data, values), terrain.world_size)
