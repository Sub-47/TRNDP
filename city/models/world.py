"""
city/models/world.py

World no longer generates terrain itself. It requests standardized maps
from a MapManager and stores references to them - it has no idea
whether those maps were generated procedurally, loaded from rasters,
or some hybrid of both, and it never needs to.
"""

from __future__ import annotations

import json
import os

import config
from city.enums.terrain_type import TerrainType
from city.managers.map_manager import MapManager
from city.maps.elevation import ElevationMap
from city.maps.obstacle import ObstacleMap
from city.maps.population import PopulationMap
from city.maps.terrain import TerrainMap
from city.renderers.terrain_renderer import render_terrain_map


class World:
    """Represents a single generated synthetic city world.

    Attributes:
        map_manager: The MapManager this world requests layers from.
        elevation: Normalized elevation layer, set after generate().
        terrain: Terrain classification layer, set after generate().
        obstacle: Traversability mask, set after generate().
        population: Population density layer, set after generate().
        metadata: Reproducibility metadata for this generation run.
    """

    def __init__(self, map_manager: MapManager) -> None:
        self.map_manager = map_manager

        self.elevation: ElevationMap | None = None
        self.terrain: TerrainMap | None = None
        self.obstacle: ObstacleMap | None = None
        self.population: PopulationMap | None = None
        self.metadata: dict | None = None

    def generate(self) -> None:
        """Request all standardized layers from the MapManager."""
        print(f"Requesting maps from MapManager (source={self.map_manager.map_source_name})...")
        maps = self.map_manager.get_maps()

        self.elevation = maps.elevation
        self.terrain = maps.terrain
        self.obstacle = maps.obstacle
        self.population = maps.population
        self.metadata = maps.metadata
        print("Maps received.")

    def visualize(self, show: bool = True) -> None:
        """Render the terrain classification map.

        Args:
            show: Whether to display the matplotlib figure interactively.

        Raises:
            RuntimeError: if called before `generate()`.
        """
        if self.terrain is None:
            raise RuntimeError("visualize() called before generate()")

        output_path = os.path.join(config.OUTPUT_DIR, config.OUTPUT_FILENAME)
        render_terrain_map(self.terrain.data, output_path, show=show)

    def save(self) -> str:
        """Save the terrain map image and generation metadata to disk.

        Returns:
            The path the terrain image was saved to.

        Raises:
            RuntimeError: if called before `generate()`.
        """
        if self.terrain is None:
            raise RuntimeError("save() called before generate()")

        print("Saving terrain...")
        output_path = os.path.join(config.OUTPUT_DIR, config.OUTPUT_FILENAME)
        render_terrain_map(self.terrain.data, output_path, show=False)

        self._save_metadata()
        return output_path

    def _save_metadata(self) -> None:
        """Write reproducibility metadata alongside the terrain image."""
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        metadata_path = os.path.join(config.OUTPUT_DIR, config.METADATA_FILENAME)
        with open(metadata_path, "w", encoding="utf-8") as metadata_file:
            json.dump(self.metadata, metadata_file, indent=2)

    def terrain_type_at(self, row: int, col: int) -> TerrainType:
        """Look up the terrain type of a single cell.

        Args:
            row: Row index into the world grid.
            col: Column index into the world grid.

        Returns:
            The TerrainType at the given cell.

        Raises:
            RuntimeError: if called before `generate()`.
        """
        if self.terrain is None:
            raise RuntimeError("terrain_type_at() called before generate()")

        return TerrainType(self.terrain.data[row, col])
