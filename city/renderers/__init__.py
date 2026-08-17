"""
city/renderers package

Rendering utilities. Renderers only ever consume standardized map
arrays and never know or care where the underlying data originated.
"""

from city.renderers.terrain_renderer import render_terrain_map

__all__ = ["render_terrain_map"]
