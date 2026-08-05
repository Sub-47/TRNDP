import json
import os

import config
from city.enums.terrain_type import TerrainType
from city.renderers.terrain_renderer import render_terrain_map


class World:
    def __init__(self, map_manager):
        self.map_manager = map_manager
        self.elevation = self.terrain = self.obstacle = self.population = None
        self.metadata = None

    def generate(self):
        maps = self.map_manager.get_maps()
        self.elevation = maps.elevation
        self.terrain = maps.terrain
        self.obstacle = maps.obstacle
        self.population = maps.population
        self.metadata = maps.metadata

    def visualize(self, show=True):
        if self.terrain is None:
            raise RuntimeError("visualize() called before generate()")
        render_terrain_map(self.terrain.data, os.path.join(config.OUTPUT_DIR, config.OUTPUT_FILENAME), show=show)

    def save(self):
        if self.terrain is None:
            raise RuntimeError("save() called before generate()")
        path = os.path.join(config.OUTPUT_DIR, config.OUTPUT_FILENAME)
        render_terrain_map(self.terrain.data, path, show=False)
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(config.OUTPUT_DIR, config.METADATA_FILENAME), "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)
        return path

    def terrain_type_at(self, row, col):
        if self.terrain is None:
            raise RuntimeError("terrain_type_at() called before generate()")
        return TerrainType(self.terrain.data[row, col])
