"""
city/managers/map_manager.py

MapManager is the only piece of the framework that knows `config.MAP_SOURCE`
exists. It resolves that string to a concrete MapSource via the registry,
asks that source for all four standardized layers, and packages them
together with reproducibility metadata. `World` talks only to MapManager
and never touches a MapSource directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import config

# Importing city.sources triggers @register_source registration for every
# built-in MapSource (Procedural, Raster, Hybrid).
import city.sources  # noqa: F401
from city.maps.elevation import ElevationMap
from city.maps.obstacle import ObstacleMap
from city.maps.population import PopulationMap
from city.maps.terrain import TerrainMap
from city.sources.registry import get_registered_source


@dataclass
class MapBundle:
    """All standardized layers for one generated world, plus metadata."""

    elevation: ElevationMap
    terrain: TerrainMap
    obstacle: ObstacleMap
    population: PopulationMap
    metadata: dict


class MapManager:
    """Instantiates the configured MapSource and hides it from the rest
    of the framework.

    Changing `config.MAP_SOURCE` from "PROCEDURAL" to "RASTER" or
    "HYBRID" is the only change needed to switch how spatial data is
    obtained - no other code, including `World`, needs to change.
    """

    def __init__(
        self,
        map_source_name: str = config.MAP_SOURCE,
        world_size: int = config.WORLD_SIZE,
        seed: int = config.SEED,
    ) -> None:
        self.map_source_name = map_source_name.upper()
        source_cls = get_registered_source(self.map_source_name)
        self._source = source_cls(world_size=world_size, seed=seed)

    def get_maps(self) -> MapBundle:
        """Request all standardized layers from the configured source.

        Returns:
            A MapBundle containing elevation, terrain, obstacle,
            population, and reproducibility metadata.
        """
        elevation = self._source.get_elevation()
        terrain = self._source.get_terrain()
        obstacle = self._source.get_obstacle()
        population = self._source.get_population()
        metadata = self._build_metadata()

        return MapBundle(
            elevation=elevation,
            terrain=terrain,
            obstacle=obstacle,
            population=population,
            metadata=metadata,
        )

    def _build_metadata(self) -> dict:
        """Assemble reproducibility metadata describing this generation run."""
        provenance = self._source.provenance
        elevation_source = provenance.get("elevation_source", "unknown")
        population_source = provenance.get("population_source", "unknown")

        return {
            "map_source": self.map_source_name,
            "generation_mode": f"elevation={elevation_source}, population={population_source}",
            "input_files_used": list(provenance.get("input_files", [])),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "generator_version": config.GENERATOR_VERSION,
        }
