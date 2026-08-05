from dataclasses import dataclass, field

import numpy as np

import config
from city.enums.terrain_type import classify_elevation
from city.utils.validation import ensure_standard_map


@dataclass
class TerrainMap:
    data: np.ndarray
    world_size: int = field(default=config.WORLD_SIZE)

    def __post_init__(self):
        self.data = ensure_standard_map(self.data, self.world_size, "terrain", np.int8)

    @classmethod
    def from_elevation(cls, elevation, sea_level=config.SEA_LEVEL, thresholds=config.TERRAIN_THRESHOLDS):
        return cls(classify_elevation(elevation.data, sea_level, thresholds), elevation.world_size)
