"""
city/demand/gravity_model.py

Computes the zone-to-zone gravity demand matrix using zone populations and
pairwise distances.

The raw gravity score (population_i * population_j / distance**beta) is an
attractiveness score, not a trip count - nothing forces it to sum to any
particular number of actual trips, so redrawing zone boundaries (changing
ZONE_SIZE) changes the total even though the same people live in the same
city. This module is singly-constrained: each origin zone's row is rescaled
so it produces exactly `population_i * TRIPS_PER_PERSON` trips, which is a
property of the population, not of how the zones happen to be drawn.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import config
from city.demand.distance_matrix import DistanceMatrix
from city.demand.zone_map import ZoneMap


@dataclass
class GravityModel:
    """Predicted demand between every pair of zones.

    Attributes:
        data: float32 array of shape (zones_per_side, zones_per_side)
            containing predicted trip counts between zones.
        zone_size: Number of grid cells along each side of one zone.
        world_size: Size of the underlying cell grid.
    """

    data: np.ndarray
    zone_size: int = field(default=config.ZONE_SIZE)
    world_size: int = field(default=config.WORLD_SIZE)

    def __post_init__(self) -> None:
        expected_shape = (self.num_zones, self.num_zones)
        if self.data.shape != expected_shape:
            raise ValueError(
                f"data must have shape {expected_shape}, got {self.data.shape}"
            )

    @property
    def zones_per_side(self) -> int:
        return self.world_size // self.zone_size

    @property
    def num_zones(self) -> int:
        return self.zones_per_side ** 2

    @classmethod
    def from_zones_and_distance(
        cls,
        zones: ZoneMap,
        dist: DistanceMatrix,
    ) -> "GravityModel":
        """Build the singly-constrained gravity demand matrix from zones
        and distances: each origin zone's row sums to exactly
        ``population_i * TRIPS_PER_PERSON`` trips, so the total is a
        property of the population, not of ZONE_SIZE. Zones with zero
        population, or with no reachable destination (row sum of the raw
        score is zero - e.g. an unreachable zone, whose distances are
        all `np.inf`), produce zero trips rather than NaN.
        """
        if zones.zone_size != dist.zone_size:
            raise ValueError(
                f"zone_size mismatch: {zones.zone_size} != {dist.zone_size}"
            )
        if zones.world_size != dist.world_size:
            raise ValueError(
                f"world_size mismatch: {zones.world_size} != {dist.world_size}"
            )

        population = zones.population.flatten().astype(np.float32)
        origin_mass = population[:, np.newaxis]
        destination_mass = population[np.newaxis, :]

        denominator = np.power(dist.data, config.GRAVITY_BETA, dtype=np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = config.GRAVITY_K * origin_mass * destination_mass / denominator

        # 1 / inf**beta is correctly 0 (unreachable pairs get no demand),
        # but 0/0 (e.g. a zero-population, zero-distance diagonal entry)
        # produces NaN, and dividing by an exact-zero distance produces
        # inf; both are scrubbed to 0 rather than propagated.
        raw[~np.isfinite(raw)] = 0.0
        np.fill_diagonal(raw, 0.0)

        row_sums = raw.sum(axis=1)
        target_trips = population * config.TRIPS_PER_PERSON
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(row_sums > 0, target_trips / row_sums, 0.0)

        demand = raw * scale[:, np.newaxis]
        demand[~np.isfinite(demand)] = 0.0

        assert not np.isnan(demand).any(), "gravity demand matrix contains NaN"

        return cls(
            data=demand.astype(np.float32),
            zone_size=zones.zone_size,
            world_size=zones.world_size,
        )

    @classmethod
    def for_period(
        cls,
        zones: ZoneMap,
        dist: DistanceMatrix,
        period: str,
    ) -> "GravityModel":
        """Builds the singly-constrained matrix and scales it by the
        given time-of-day's diurnal factor (see config.DIURNAL_FACTORS).

        Raises:
            ValueError: if `period` isn't a known key of DIURNAL_FACTORS.
        """
        if period not in config.DIURNAL_FACTORS:
            valid = ", ".join(sorted(config.DIURNAL_FACTORS))
            raise ValueError(f"Unknown period {period!r}; valid options: {valid}")

        base = cls.from_zones_and_distance(zones, dist)
        factor = config.DIURNAL_FACTORS[period]
        scaled = (base.data * factor).astype(np.float32)

        return cls(
            data=scaled,
            zone_size=base.zone_size,
            world_size=base.world_size,
        )
