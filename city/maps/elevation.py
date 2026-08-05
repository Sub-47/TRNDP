from dataclasses import dataclass, field

import numpy as np

import config
from city.utils.validation import ensure_standard_map, is_normalized


@dataclass
class ElevationMap:
    data: np.ndarray
    world_size: int = field(default=config.WORLD_SIZE)

    def __post_init__(self):
        self.data = ensure_standard_map(self.data, self.world_size, "elevation")
        if not is_normalized(self.data):
            raise ValueError(
                f"ElevationMap values must be normalized to [0.0, 1.0]; "
                f"got range [{self.data.min()}, {self.data.max()}]"
            )
