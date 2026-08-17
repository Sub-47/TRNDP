"""
city/maps/population.py

Standardized population density layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import config
from city.utils.validation import ensure_standard_map, is_normalized


@dataclass
class PopulationMap:
    """Normalized population density for every cell in the world grid.

    A float32 array of shape (world_size, world_size) with values in
    [0.0, 1.0]. In this milestone this is either loaded from a raster
    PNG or stands in as a procedurally-generated placeholder; a real
    population model is future work and can replace either source
    without this type, or anything that consumes it, changing.
    """

    data: np.ndarray
    world_size: int = field(default=config.WORLD_SIZE)

    def __post_init__(self) -> None:
        self.data = ensure_standard_map(self.data, self.world_size, "population")
        if not is_normalized(self.data):
            raise ValueError(
                "PopulationMap values must be normalized to [0.0, 1.0]; "
                f"got range [{self.data.min()}, {self.data.max()}]"
            )
