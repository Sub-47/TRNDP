"""
city/sources/hybrid_source.py

Supplies standardized maps by combining raster and procedural sources,
choosing per-map based on which raster files are actually present:

    - both elevation.png and population.png exist -> use both rasters
    - only elevation.png exists -> raster elevation, procedural population
    - only population.png exists -> procedural elevation, raster population
    - neither exists -> both procedural
"""

from __future__ import annotations

import os

import config
from city.maps.elevation import ElevationMap
from city.maps.population import PopulationMap
from city.sources.map_source import MapSource
from city.sources.procedural_source import ProceduralSource
from city.sources.raster_source import RasterSource
from city.sources.registry import register_source


@register_source("HYBRID")
class HybridSource(MapSource):
    """Prefers raster input per-map, falling back to procedural generation."""

    def __init__(
        self,
        world_size: int = config.WORLD_SIZE,
        seed: int = config.SEED,
        input_dir: str = config.ASSET_INPUT_DIR,
    ) -> None:
        super().__init__(world_size=world_size, seed=seed, input_dir=input_dir)
        self._procedural = ProceduralSource(
            world_size=world_size, seed=seed, input_dir=input_dir
        )
        self._raster = RasterSource(
            world_size=world_size, seed=seed, input_dir=input_dir
        )

    def _has_raster(self, filename: str) -> bool:
        return os.path.isfile(os.path.join(self.input_dir, filename))

    def _compute_elevation(self) -> ElevationMap:
        if self._has_raster(RasterSource.ELEVATION_FILENAME):
            elevation = self._raster.get_elevation()
            self.provenance["elevation_source"] = "raster"
            self.provenance["input_files"].extend(
                self._raster.provenance["input_files"]
            )
        else:
            elevation = self._procedural.get_elevation()
            self.provenance["elevation_source"] = "procedural"
        return elevation

    def _compute_population(self) -> PopulationMap:
        if self._has_raster(RasterSource.POPULATION_FILENAME):
            population = self._raster.get_population()
            self.provenance["population_source"] = "raster"
            self.provenance["input_files"].extend(
                self._raster.provenance["input_files"]
            )
        else:
            population = self._procedural.get_population()
            self.provenance["population_source"] = "procedural"
        return population
