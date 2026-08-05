import os

import config
from city.sources.map_source import MapSource
from city.sources.procedural_source import ProceduralSource
from city.sources.raster_source import RasterSource
from city.sources.registry import register_source


@register_source("HYBRID")
class HybridSource(MapSource):
    def __init__(self, world_size=config.WORLD_SIZE, seed=config.SEED, input_dir=config.ASSET_INPUT_DIR):
        super().__init__(world_size, seed, input_dir)
        self._procedural = ProceduralSource(world_size, seed, input_dir)
        self._raster = RasterSource(world_size, seed, input_dir)

    def _has_raster(self, filename):
        return os.path.isfile(os.path.join(self.input_dir, filename))

    def _compute_elevation(self):
        if self._has_raster(RasterSource.ELEVATION_FILENAME):
            data = self._raster.get_elevation()
            self.provenance["elevation_source"] = "raster"
            self.provenance["input_files"] += self._raster.provenance["input_files"]
        else:
            data = self._procedural.get_elevation()
            self.provenance["elevation_source"] = "procedural"
        return data

    def _compute_population(self):
        if self._has_raster(RasterSource.POPULATION_FILENAME):
            data = self._raster.get_population()
            self.provenance["population_source"] = "raster"
            self.provenance["input_files"] += self._raster.provenance["input_files"]
        else:
            data = self._procedural.get_population()
            self.provenance["population_source"] = "procedural"
        return data
