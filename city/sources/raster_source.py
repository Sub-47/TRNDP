import os

import numpy as np
from PIL import Image

from city.maps.elevation import ElevationMap
from city.maps.population import PopulationMap
from city.sources.map_source import MapSource
from city.sources.registry import register_source


@register_source("RASTER")
class RasterSource(MapSource):
    ELEVATION_FILENAME = "elevation.png"
    POPULATION_FILENAME = "population.png"

    def _compute_elevation(self):
        self.provenance["elevation_source"] = "raster"
        return ElevationMap(self._load(self.ELEVATION_FILENAME), self.world_size)

    def _compute_population(self):
        self.provenance["population_source"] = "raster"
        return PopulationMap(self._load(self.POPULATION_FILENAME), self.world_size)

    def _load(self, filename):
        path = os.path.join(self.input_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"RasterSource requires '{path}', but it does not exist. "
                "Use MAP_SOURCE = 'HYBRID' in config.py to fall back to procedural generation for missing rasters."
            )
        print(f"Loading raster '{path}'...")
        image = Image.open(path).convert("L").resize((self.world_size, self.world_size), Image.BILINEAR)
        self.provenance["input_files"].append(path)
        return np.asarray(image, dtype=np.float32) / 255
