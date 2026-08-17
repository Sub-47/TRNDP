"""
city/generators/population_generator.py

Procedural population density generation.

Models density as a small number of city centres (one primary CBD plus
sub-centres) whose influence decays with distance, rather than raw
noise. This gives the field real spatial structure - a downtown core
and secondary nodes - for a later DBSCAN clustering phase to find.
"""

from __future__ import annotations

import numpy as np

import config
from city.enums.terrain_type import TerrainType


class PopulationGenerator:
    """Generates a population density field shaped by city centres.

    Responsibilities are split into small, composable steps
    (`place_centres`, `compute_density`, `apply_suitability`,
    `apply_obstacles`, `normalize`), mirroring `TerrainGenerator`, so
    callers can reuse individual pieces without depending on the whole
    pipeline.
    """

    def __init__(
        self,
        terrain_classes: np.ndarray,
        obstacle_mask: np.ndarray,
        world_size: int = config.WORLD_SIZE,
        seed: int = config.SEED,
        num_centres: int = config.NUM_CITY_CENTRES,
        min_separation: int = config.CENTRE_MIN_SEPARATION,
        max_subcentre_distance: float = config.SUBCENTRE_MAX_DISTANCE,
        decay_length: float = config.CENTRE_DECAY_LENGTH,
        subcentre_weight: float = config.SUBCENTRE_WEIGHT,
        terrain_suitability: dict[str, float] = config.TERRAIN_SUITABILITY,
    ) -> None:
        self.terrain_classes = terrain_classes
        self.obstacle_mask = obstacle_mask
        self.world_size = world_size
        self.seed = seed
        self.num_centres = num_centres
        self.min_separation = min_separation
        self.max_subcentre_distance = max_subcentre_distance
        self.decay_length = decay_length
        self.subcentre_weight = subcentre_weight
        self.terrain_suitability = terrain_suitability

        self.centres: list[tuple[int, int]] | None = None
        self.density: np.ndarray | None = None
        self.population: np.ndarray | None = None

    def place_centres(self) -> list[tuple[int, int]]:
        """Choose city-centre locations deterministically from the seed.

        The first centre chosen is the primary CBD; the rest are
        sub-centres, drawn only from cells within `max_subcentre_distance`
        of the CBD so they read as one polycentric city rather than
        several disconnected settlements (an unbounded pool routinely
        placed sub-centres 100+ cells from the CBD on a 256-cell world,
        which no decay length could bridge without flattening the whole
        density field). If that bounded pool is exhausted before
        `min_separation` can be satisfied, the bound is relaxed for that
        pick so generation still succeeds on small or heavily-obstructed
        worlds.

        Candidates are restricted to habitable cells (not flagged in
        `obstacle_mask`) and, once a centre is placed, every cell within
        `min_separation` of it is removed from the candidate pool before
        the next pick - fully vectorized, no per-candidate Python loop.

        Returns:
            A list of (row, col) centre coordinates, primary first.

        Raises:
            RuntimeError: if fewer than `num_centres` separated
                habitable cells are available.
        """
        rng = np.random.default_rng(self.seed + config.POPULATION_SEED_OFFSET)
        rows, cols = np.indices((self.world_size, self.world_size))

        available = ~self.obstacle_mask
        centres: list[tuple[int, int]] = []

        for index in range(self.num_centres):
            pool = available
            if index > 0:
                primary_row, primary_col = centres[0]
                near_primary = (
                    np.hypot(rows - primary_row, cols - primary_col)
                    <= self.max_subcentre_distance
                )
                if np.any(available & near_primary):
                    pool = available & near_primary

            candidate_rows, candidate_cols = np.nonzero(pool)
            if candidate_rows.size == 0:
                break

            choice = rng.integers(candidate_rows.size)
            centre_row, centre_col = int(candidate_rows[choice]), int(candidate_cols[choice])
            centres.append((centre_row, centre_col))

            dist = np.hypot(rows - centre_row, cols - centre_col)
            available &= dist >= self.min_separation

        if len(centres) < self.num_centres:
            raise RuntimeError(
                f"Could only place {len(centres)}/{self.num_centres} city "
                f"centres at least {self.min_separation} cells apart on "
                "habitable terrain; reduce NUM_CITY_CENTRES or "
                "CENTRE_MIN_SEPARATION."
            )

        self.centres = centres
        return centres

    def compute_density(self) -> np.ndarray:
        """Sum weighted exponential decay from every centre, per cell.

        density = sum over centres of weight_c * exp(-dist(cell, c) /
        decay_length), where weight is 1.0 for the primary centre and
        `subcentre_weight` for every other centre. Fully vectorized
        over the grid with NumPy broadcasting - no per-cell loop.

        # TODO: this uses straight-line (Euclidean) distance as a
        # deliberate simplification. Real density decays by travel
        # distance along the road network, which does not exist yet;
        # once road generation lands, this should switch to graph
        # distance.

        Returns:
            A float64 density array, not yet scaled or normalized.

        Raises:
            RuntimeError: if called before `place_centres()`.
        """
        if self.centres is None:
            raise RuntimeError("compute_density() called before place_centres()")

        rows, cols = np.indices((self.world_size, self.world_size))
        density = np.zeros((self.world_size, self.world_size), dtype=np.float64)

        for index, (centre_row, centre_col) in enumerate(self.centres):
            weight = 1.0 if index == 0 else self.subcentre_weight
            dist = np.hypot(rows - centre_row, cols - centre_col)
            density += weight * np.exp(-dist / self.decay_length)

        self.density = density
        return density

    def apply_suitability(self) -> np.ndarray:
        """Scale density by each cell's terrain-suitability factor.

        Returns:
            The suitability-weighted density array.

        Raises:
            RuntimeError: if called before `compute_density()`.
        """
        if self.density is None:
            raise RuntimeError("apply_suitability() called before compute_density()")

        suitability_lut = np.zeros(len(TerrainType), dtype=np.float64)
        for name, factor in self.terrain_suitability.items():
            suitability_lut[TerrainType[name].value] = factor

        self.density = self.density * suitability_lut[self.terrain_classes]
        return self.density

    def apply_obstacles(self) -> np.ndarray:
        """Force density to exactly 0.0 on every obstacle cell.

        Returns:
            The obstacle-masked density array.

        Raises:
            RuntimeError: if called before `compute_density()`.
        """
        if self.density is None:
            raise RuntimeError("apply_obstacles() called before compute_density()")

        self.density[self.obstacle_mask] = 0.0
        return self.density

    def normalize(self) -> np.ndarray:
        """Min-max normalize density into [0.0, 1.0] as float32.

        Returns:
            The normalized population density array.

        Raises:
            RuntimeError: if called before `compute_density()`.
        """
        if self.density is None:
            raise RuntimeError("normalize() called before compute_density()")

        min_val = self.density.min()
        max_val = self.density.max()
        value_range = max_val - min_val

        if value_range == 0:
            # Degenerate flat field (e.g. no habitable land); avoid
            # division by zero.
            normalized = np.zeros_like(self.density)
        else:
            normalized = (self.density - min_val) / value_range

        self.population = normalized.astype(np.float32)
        return self.population

    def run(self) -> np.ndarray:
        """Execute the full generation pipeline in order.

        Returns:
            The final normalized population density array (float32,
            shape (world_size, world_size)).
        """
        self.place_centres()
        self.compute_density()
        self.apply_suitability()
        self.apply_obstacles()
        return self.normalize()
