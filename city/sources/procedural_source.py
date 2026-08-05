import numpy as np

import config
from city.generators.terrain_generator import TerrainGenerator
from city.maps.elevation import ElevationMap
from city.maps.population import PopulationMap
from city.sources.map_source import MapSource
from city.sources.registry import register_source


@register_source("PROCEDURAL")
class ProceduralSource(MapSource):
    def _compute_elevation(self):
        print("Generating elevation procedurally...")
        generator = TerrainGenerator(self.world_size, self.seed)
        generator.generate()
        generator.normalize()
        self.provenance["elevation_source"] = "procedural"
        return ElevationMap(generator.elevation.astype(np.float32), self.world_size)

    def _compute_population(self):
        print("Generating population procedurally (placeholder)...")
        generator = TerrainGenerator(self.world_size, self.seed + config.POPULATION_SEED_OFFSET)
        generator.generate()
        generator.normalize()
        self.provenance["population_source"] = "procedural"
        return PopulationMap(generator.elevation.astype(np.float32), self.world_size)
