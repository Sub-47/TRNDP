from dataclasses import dataclass
from datetime import datetime, timezone

import config
import city.sources  # noqa: F401  (registers PROCEDURAL/RASTER/HYBRID sources)
from city.sources.registry import get_registered_source


@dataclass
class MapBundle:
    elevation: object
    terrain: object
    obstacle: object
    population: object
    metadata: dict


class MapManager:
    def __init__(self, map_source_name=config.MAP_SOURCE, world_size=config.WORLD_SIZE, seed=config.SEED):
        self.map_source_name = map_source_name.upper()
        self._source = get_registered_source(self.map_source_name)(world_size=world_size, seed=seed)

    def get_maps(self):
        elevation = self._source.get_elevation()
        terrain = self._source.get_terrain()
        obstacle = self._source.get_obstacle()
        population = self._source.get_population()
        return MapBundle(elevation, terrain, obstacle, population, self._build_metadata())

    def _build_metadata(self):
        provenance = self._source.provenance
        return {
            "map_source": self.map_source_name,
            "generation_mode": (
                f"elevation={provenance.get('elevation_source', 'unknown')}, "
                f"population={provenance.get('population_source', 'unknown')}"
            ),
            "input_files_used": list(provenance.get("input_files", [])),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "generator_version": config.GENERATOR_VERSION,
        }
