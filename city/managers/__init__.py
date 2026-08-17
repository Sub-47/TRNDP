"""
city/managers package

MapManager: the sole point of contact between `World` and `MapSource`
implementations.
"""

from city.managers.map_manager import MapBundle, MapManager

__all__ = ["MapManager", "MapBundle"]
