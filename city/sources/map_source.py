"""
city/sources/map_source.py

Abstract interface for anything that can supply standardized spatial
layers to the framework.

Design notes:
    - Concrete subclasses only ever need to implement how to *produce*
      elevation and population (`_compute_elevation`, `_compute_population`).
    - `get_terrain()` and `get_obstacle()` are generic, source-agnostic
      derivations built on top of elevation/terrain, so they are
      implemented once here and inherited for free by every subclass -
      including ones that don't exist yet (GeoTIFFSource,
      SatelliteSource, ...).
    - `get_elevation()` / `get_population()` cache their result, so
      calling them repeatedly (e.g. once directly, once transitively via
      `get_terrain()`) never re-runs expensive generation or I/O twice.
    - `provenance` is populated as maps are computed, and is later read
      by `MapManager` to build reproducibility metadata.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import config
from city.maps.elevation import ElevationMap
from city.maps.obstacle import ObstacleMap
from city.maps.population import PopulationMap
from city.maps.terrain import TerrainMap


class MapSource(ABC):
    """Base class for all spatial data providers (procedural, raster, ...)."""

    def __init__(
        self,
        world_size: int = config.WORLD_SIZE,
        seed: int = config.SEED,
        input_dir: str = config.ASSET_INPUT_DIR,
    ) -> None:
        self.world_size = world_size
        self.seed = seed
        self.input_dir = input_dir

        self.provenance: dict[str, object] = {"input_files": []}

        self._elevation: ElevationMap | None = None
        self._population: PopulationMap | None = None
        self._terrain: TerrainMap | None = None
        self._obstacle: ObstacleMap | None = None

    # -- Required: subclasses implement how to produce raw layers -------

    @abstractmethod
    def _compute_elevation(self) -> ElevationMap:
        """Produce this source's elevation layer."""

    @abstractmethod
    def _compute_population(self) -> PopulationMap:
        """Produce this source's population layer."""

    # -- Public interface (required by the framework) -------------------

    def get_elevation(self) -> ElevationMap:
        """Return the standardized elevation layer, computing it once."""
        if self._elevation is None:
            self._elevation = self._compute_elevation()
        return self._elevation

    def get_population(self) -> PopulationMap:
        """Return the standardized population layer, computing it once."""
        if self._population is None:
            self._population = self._compute_population()
        return self._population

    def get_terrain(self) -> TerrainMap:
        """Return terrain classification, derived from elevation.

        This is intentionally not abstract: terrain classification is a
        pure function of elevation and has nothing to do with where that
        elevation came from, so every subclass gets it for free.
        """
        if self._terrain is None:
            self._terrain = TerrainMap.from_elevation(self.get_elevation())
        return self._terrain

    def get_obstacle(self) -> ObstacleMap:
        """Return the traversability mask, derived from terrain.

        Like `get_terrain()`, this has a generic default so subclasses
        don't need to implement it - though they remain free to override
        it (e.g. a future source with an explicit obstacle raster).
        """
        if self._obstacle is None:
            self._obstacle = ObstacleMap.from_terrain(self.get_terrain())
        return self._obstacle
