"""
city/sources/raster_source.py

Supplies standardized maps loaded from user-provided grayscale PNG
rasters (assets/input/elevation.png, assets/input/population.png).
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image

from city.maps.elevation import ElevationMap
from city.maps.population import PopulationMap
from city.sources.map_source import MapSource
from city.sources.registry import register_source


@register_source("RASTER")
class RasterSource(MapSource):
    """Loads elevation and population from grayscale PNGs.

    RasterSource always requires its input files to exist - it has no
    fallback logic of its own. `HybridSource` is what decides when to
    fall back to procedural generation for a missing raster.
    """

    ELEVATION_FILENAME = "elevation.png"
    POPULATION_FILENAME = "population.png"

    def _compute_elevation(self) -> ElevationMap:
        array = self._load_grayscale_png(self.ELEVATION_FILENAME)
        self.provenance["elevation_source"] = "raster"
        return ElevationMap(array, self.world_size)

    def _compute_population(self) -> PopulationMap:
        array = self._load_grayscale_png(self.POPULATION_FILENAME)
        self.provenance["population_source"] = "raster"
        return PopulationMap(array, self.world_size)

    def _load_grayscale_png(self, filename: str) -> np.ndarray:
        """Load, resize, grayscale-convert, and normalize a PNG raster.

        Args:
            filename: Filename within `self.input_dir`.

        Returns:
            A float32 array of shape (world_size, world_size), values
            in [0.0, 1.0].

        Raises:
            FileNotFoundError: if the file does not exist. RasterSource
                does not silently substitute anything for missing input;
                use MAP_SOURCE = "HYBRID" for automatic fallback.
        """
        path = os.path.join(self.input_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"RasterSource requires '{path}', but it does not exist. "
                "Use MAP_SOURCE = 'HYBRID' in config.py to fall back to "
                "procedural generation for missing rasters."
            )

        print(f"Loading raster '{path}'...")
        image = Image.open(path).convert("L")
        image = image.resize((self.world_size, self.world_size), Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0

        self.provenance["input_files"].append(path)
        return array
