from abc import ABC, abstractmethod

import config


class MapSource(ABC):
    def __init__(self, world_size=config.WORLD_SIZE, seed=config.SEED, input_dir=config.ASSET_INPUT_DIR):
        self.world_size = world_size
        self.seed = seed
        self.input_dir = input_dir
        self.provenance = {"input_files": []}
        self._elevation = self._population = self._terrain = self._obstacle = None

    @abstractmethod
    def _compute_elevation(self): ...

    @abstractmethod
    def _compute_population(self): ...

    def get_elevation(self):
        if self._elevation is None:
            self._elevation = self._compute_elevation()
        return self._elevation

    def get_population(self):
        if self._population is None:
            self._population = self._compute_population()
        return self._population

    def get_terrain(self):
        if self._terrain is None:
            from city.maps.terrain import TerrainMap

            self._terrain = TerrainMap.from_elevation(self.get_elevation())
        return self._terrain

    def get_obstacle(self):
        if self._obstacle is None:
            from city.maps.obstacle import ObstacleMap

            self._obstacle = ObstacleMap.from_terrain(self.get_terrain())
        return self._obstacle
