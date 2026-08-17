"""
city/generators/terrain_generator.py

Procedural (Perlin noise) terrain generation.

This is the same generation pipeline from Milestone 1, relocated into
the `generators` package. Its public API (`generate`, `normalize`,
`apply_sea_level`, `classify`, `run`) is unchanged; `classify()` now
delegates to the shared `classify_elevation` function so classification
logic lives in exactly one place, shared with `TerrainMap`.
"""

from __future__ import annotations

import numpy as np
from noise import pnoise2

import config
from city.enums.terrain_type import TerrainType, classify_elevation


class TerrainGenerator:
    """Generates and classifies a heightmap for a single world.

    Responsibilities are split into small, composable steps (`generate`,
    `normalize`, `apply_sea_level`, `classify`) so callers can reuse
    individual pieces without depending on the whole pipeline.
    """

    def __init__(
        self,
        world_size: int = config.WORLD_SIZE,
        seed: int = config.SEED,
        scale: float = config.NOISE_SCALE,
        octaves: int = config.OCTAVES,
        persistence: float = config.PERSISTENCE,
        lacunarity: float = config.LACUNARITY,
        sea_level: float = config.SEA_LEVEL,
        thresholds: dict[str, float] = config.TERRAIN_THRESHOLDS,
    ) -> None:
        self.world_size = world_size
        self.seed = seed
        self.scale = scale
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.sea_level = sea_level
        self.thresholds = thresholds

        self._raw_elevation: np.ndarray | None = None
        self.elevation: np.ndarray | None = None
        self.water_mask: np.ndarray | None = None
        self.terrain_classes: np.ndarray | None = None

    def generate(self) -> np.ndarray:
        """Generate raw Perlin noise elevation values for every cell.

        Two seed-derived offsets are applied to the noise coordinates so
        that different seeds produce visibly different terrain (the
        `noise` library's `pnoise2` does not accept an integer seed
        directly, only a `base` offset per octave layer).

        Returns:
            The raw (un-normalized) elevation array.
        """
        rng = np.random.default_rng(self.seed)
        offset_x, offset_y = rng.uniform(0, 10_000, size=2)

        raw = np.empty((self.world_size, self.world_size), dtype=np.float64)
        for row in range(self.world_size):
            for col in range(self.world_size):
                raw[row, col] = pnoise2(
                    (row + offset_x) / self.scale,
                    (col + offset_y) / self.scale,
                    octaves=self.octaves,
                    persistence=self.persistence,
                    lacunarity=self.lacunarity,
                    repeatx=self.world_size,
                    repeaty=self.world_size,
                    base=self.seed % 1024,
                )

        self._raw_elevation = raw
        return raw

    def normalize(self) -> np.ndarray:
        """Min-max normalize the raw elevation map into [0.0, 1.0].

        Returns:
            The normalized elevation array.

        Raises:
            RuntimeError: if called before `generate()`.
        """
        if self._raw_elevation is None:
            raise RuntimeError("normalize() called before generate()")

        raw = self._raw_elevation
        min_val = raw.min()
        max_val = raw.max()
        value_range = max_val - min_val

        if value_range == 0:
            # Degenerate flat noise field; avoid division by zero.
            normalized = np.zeros_like(raw)
        else:
            normalized = (raw - min_val) / value_range

        self.elevation = normalized
        return normalized

    def apply_sea_level(self) -> np.ndarray:
        """Build a boolean water mask from the normalized elevation map.

        Returns:
            A boolean array, True where the cell is below sea level.

        Raises:
            RuntimeError: if called before `normalize()`.
        """
        if self.elevation is None:
            raise RuntimeError("apply_sea_level() called before normalize()")

        self.water_mask = self.elevation < self.sea_level
        return self.water_mask

    def classify(self) -> np.ndarray:
        """Assign a TerrainType to every cell based on elevation.

        Delegates to the shared `classify_elevation` function so this
        logic stays in sync with `TerrainMap.from_elevation`.

        Returns:
            An integer array of TerrainType values.

        Raises:
            RuntimeError: if called before `apply_sea_level()`.
        """
        if self.elevation is None or self.water_mask is None:
            raise RuntimeError("classify() called before apply_sea_level()")

        self.terrain_classes = classify_elevation(
            self.elevation, self.sea_level, self.thresholds
        )
        return self.terrain_classes

    def run(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Execute the full generation pipeline in order.

        Returns:
            A tuple of (elevation, water_mask, terrain_classes).
        """
        self.generate()
        self.normalize()
        self.apply_sea_level()
        self.classify()
        return self.elevation, self.water_mask, self.terrain_classes
