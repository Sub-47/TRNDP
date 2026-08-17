"""
city/maps package

Standardized spatial layers (elevation, terrain, obstacle, population).
Every layer is a validated NumPy array of a fixed shape and dtype; none
of them know or care which MapSource produced the data underneath them.
"""

from city.maps.elevation import ElevationMap
from city.maps.obstacle import ObstacleMap
from city.maps.population import PopulationMap
from city.maps.terrain import TerrainMap

__all__ = ["ElevationMap", "ObstacleMap", "PopulationMap", "TerrainMap"]
