import numpy as np
from noise import pnoise2

import config
from city.enums.terrain_type import classify_elevation


class TerrainGenerator:
    def __init__(
        self,
        world_size=config.WORLD_SIZE,
        seed=config.SEED,
        scale=config.NOISE_SCALE,
        octaves=config.OCTAVES,
        persistence=config.PERSISTENCE,
        lacunarity=config.LACUNARITY,
        sea_level=config.SEA_LEVEL,
        thresholds=config.TERRAIN_THRESHOLDS,
    ):
        self.world_size = world_size
        self.seed = seed
        self.scale = scale
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.sea_level = sea_level
        self.thresholds = thresholds
        self._raw_elevation = None
        self.elevation = None
        self.water_mask = None
        self.terrain_classes = None

    def generate(self):
        rng = np.random.default_rng(self.seed)
        offset_x, offset_y = rng.uniform(0, 10000, 2)
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

    def normalize(self):
        if self._raw_elevation is None:
            raise RuntimeError("normalize() called before generate()")
        raw = self._raw_elevation
        spread = raw.max() - raw.min()
        self.elevation = np.zeros_like(raw) if spread == 0 else (raw - raw.min()) / spread
        return self.elevation

    def apply_sea_level(self):
        if self.elevation is None:
            raise RuntimeError("apply_sea_level() called before normalize()")
        self.water_mask = self.elevation < self.sea_level
        return self.water_mask

    def classify(self):
        if self.elevation is None or self.water_mask is None:
            raise RuntimeError("classify() called before apply_sea_level()")
        self.terrain_classes = classify_elevation(self.elevation, self.sea_level, self.thresholds)
        return self.terrain_classes

    def run(self):
        self.generate()
        self.normalize()
        self.apply_sea_level()
        self.classify()
        return self.elevation, self.water_mask, self.terrain_classes
