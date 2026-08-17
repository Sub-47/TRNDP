"""
city/sources/procedural_source.py

Supplies standardized maps generated entirely from procedural
generators - Perlin noise for elevation, a city-centre density model
for population. This class only adapts generator output into the
framework's standardized map types.
"""

from __future__ import annotations

import numpy as np

from city.generators.population_generator import PopulationGenerator
from city.generators.terrain_generator import TerrainGenerator
from city.maps.elevation import ElevationMap
from city.maps.population import PopulationMap
from city.sources.map_source import MapSource
from city.sources.registry import register_source


@register_source("PROCEDURAL")
class ProceduralSource(MapSource):
    """Generates elevation and population entirely from Perlin noise."""

    def _compute_elevation(self) -> ElevationMap:
        print("Generating elevation procedurally...")
        generator = TerrainGenerator(world_size=self.world_size, seed=self.seed)
        generator.generate()
        generator.normalize()
        self.provenance["elevation_source"] = "procedural"
        return ElevationMap(generator.elevation.astype(np.float32), self.world_size)

    def _compute_population(self) -> PopulationMap:
        print("Generating population procedurally (city-centre model)...")
        terrain = self.get_terrain()
        obstacle = self.get_obstacle()
        generator = PopulationGenerator(
            terrain_classes=terrain.data,
            obstacle_mask=obstacle.data,
            world_size=self.world_size,
            seed=self.seed,
        )
        population = generator.run()
        self.provenance["population_source"] = "procedural"
        return PopulationMap(population, self.world_size)
