"""
city/maps/elevation.py

Standardized elevation layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import config
from city.utils.validation import ensure_standard_map, is_normalized


@dataclass
class ElevationMap:
    """Normalized elevation for every cell in the world grid.

    A float32 array of shape (world_size, world_size) with values in
    [0.0, 1.0]. Downstream code never needs to know whether this came
    from procedural noise, a raster PNG, or a hybrid combination - the
    contract is identical either way.
    """

    data: np.ndarray
    world_size: int = field(default=config.WORLD_SIZE)

    def __post_init__(self) -> None:
        self.data = ensure_standard_map(self.data, self.world_size, "elevation")
        if not is_normalized(self.data):
            raise ValueError(
                "ElevationMap values must be normalized to [0.0, 1.0]; "
                f"got range [{self.data.min()}, {self.data.max()}]"
            )
