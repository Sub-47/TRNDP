"""
city package

Spatial Urban Modeling Framework (SUMF) core package.

Layout:
    city.models     - World and future domain models
    city.managers   - MapManager (World's only point of contact with data sources)
    city.sources    - MapSource and its concrete implementations
    city.generators - Concrete generation algorithms (e.g. TerrainGenerator)
    city.renderers  - Rendering utilities
    city.maps       - Standardized spatial layer types
    city.enums      - Shared discrete classifications (e.g. TerrainType)
    city.utils      - Generation-agnostic shared helpers
"""

from city.enums.terrain_type import TerrainType
from city.managers.map_manager import MapManager
from city.models.world import World

__all__ = ["World", "MapManager", "TerrainType"]
