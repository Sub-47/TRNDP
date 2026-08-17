"""
city/sources package

MapSource and its concrete implementations. Importing this package is
what triggers `@register_source` registration for every built-in
source, so `MapManager` can resolve `config.MAP_SOURCE` by name.

Adding a new source (e.g. GeoTIFFSource) only requires creating a new
module here that subclasses MapSource and decorates itself with
`@register_source(...)`; nothing else in this file needs to change as
long as the new module is imported below.
"""

from city.sources.hybrid_source import HybridSource
from city.sources.map_source import MapSource
from city.sources.procedural_source import ProceduralSource
from city.sources.raster_source import RasterSource

__all__ = ["MapSource", "ProceduralSource", "RasterSource", "HybridSource"]
