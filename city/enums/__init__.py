"""
city/enums package

Terrain vocabulary and other framework-wide discrete classifications.
"""

from city.enums.terrain_type import TerrainType, classify_elevation

__all__ = ["TerrainType", "classify_elevation"]
