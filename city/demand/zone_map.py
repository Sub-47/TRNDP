"""
city/demand/zone_map.py

Aggregates the cell-level PopulationMap into discrete zones for demand
modelling (e.g. gravity model trip generation).

A "zone" is a zone_size x zone_size block of grid cells. Each zone gets
one population mass (population summed over its cells) and a center
point in grid coordinates, which distance_matrix.py later uses to
compute zone-to-zone distances.

Cells flagged as obstacles (ocean, mountains) are treated as
uninhabited when computing zone mass. PopulationMap in this milestone
is a placeholder noise field with no relationship to terrain - without
this masking, a zone that's mostly ocean would still report nonzero
population.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import config
from city.maps.obstacle import ObstacleMap
from city.maps.population import PopulationMap


@dataclass
class ZoneMap:
    """Grid of zones, each holding an aggregated population mass.

    Attributes:
        population: float32 array of shape (zones_per_side, zones_per_side),
            total (obstacle-masked) population summed over each zone's cells.
        centers_row: float32 array, same shape, row-coordinate (in grid
            cells) of each zone's center.
        centers_col: float32 array, same shape, column-coordinate of each
            zone's center.
        zone_size: number of grid cells along each side of one zone.
        world_size: size of the underlying cell grid.
    """

    population: np.ndarray
    centers_row: np.ndarray
    centers_col: np.ndarray
    zone_size: int = field(default=config.ZONE_SIZE)
    world_size: int = field(default=config.WORLD_SIZE)

    def __post_init__(self) -> None:
        if self.world_size % self.zone_size != 0:
            raise ValueError(
                f"world_size ({self.world_size}) must be divisible by "
                f"zone_size ({self.zone_size})"
            )
        expected_shape = (self.zones_per_side, self.zones_per_side)
        for name, array in (
            ("population", self.population),
            ("centers_row", self.centers_row),
            ("centers_col", self.centers_col),
        ):
            if array.shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}, got {array.shape}"
                )

    @property
    def zones_per_side(self) -> int:
        """Number of zones along one edge of the world."""
        return self.world_size // self.zone_size

    @property
    def num_zones(self) -> int:
        """Total number of zones in the world."""
        return self.zones_per_side ** 2

    def flatten(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Flatten the 2D zone grid into 1D arrays, one entry per zone.

        Returns:
            (population, centers_row, centers_col), each a 1D array of
            length num_zones - the shape distance_matrix.py and
            gravity_model.py will want to consume.
        """
        return (
            self.population.flatten(),
            self.centers_row.flatten(),
            self.centers_col.flatten(),
        )

    @classmethod
    def from_maps(
        cls,
        population: PopulationMap,
        obstacle: ObstacleMap,
        zone_size: int = config.ZONE_SIZE,
        mask_obstacles: bool = config.MASK_POPULATION_ON_OBSTACLES,
    ) -> "ZoneMap":
        """Aggregate cell-level population into zones.

        Args:
            population: Cell-level population density (0.0-1.0 per cell).
            obstacle: Cell-level traversability mask; cells flagged True
                are treated as uninhabited when mask_obstacles is True.
            zone_size: Number of grid cells along each side of one zone.
                world_size must be evenly divisible by this.
            mask_obstacles: If True, zero out population on obstacle
                cells (ocean, mountains) before aggregating.

        Returns:
            A new ZoneMap.
        """
        world_size = population.world_size
        if world_size % zone_size != 0:
            raise ValueError(
                f"world_size ({world_size}) must be divisible by "
                f"zone_size ({zone_size})"
            )

        pop_data = population.data.copy()
        if mask_obstacles:
            pop_data[obstacle.data] = 0.0

        zones_per_side = world_size // zone_size

        # (world_size, world_size) -> (zones_per_side, zone_size,
        # zones_per_side, zone_size), grouping each zone's cells together,
        # then sum over the two zone_size axes to get one total per zone.
        reshaped = pop_data.reshape(
            zones_per_side, zone_size, zones_per_side, zone_size
        )
        zone_population = reshaped.sum(axis=(1, 3)).astype(np.float32)

        # Zone centers in grid-cell coordinates, for distance_matrix.py.
        zone_indices = np.arange(zones_per_side)
        center_offset = (zone_size - 1) / 2.0
        centers_1d = zone_indices * zone_size + center_offset
        centers_row, centers_col = np.meshgrid(centers_1d, centers_1d, indexing="ij")

        return cls(
            population=zone_population,
            centers_row=centers_row.astype(np.float32),
            centers_col=centers_col.astype(np.float32),
            zone_size=zone_size,
            world_size=world_size,
        )