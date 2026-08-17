"""
city/sources/registry.py

A small registry that maps a MAP_SOURCE config string to a concrete
MapSource subclass.

This is what makes the architecture genuinely open/closed: a future
`GeoTIFFSource` or `SatelliteSource` registers itself with
`@register_source("GEOTIFF")` in its own file. Nothing in
`MapManager`, or any existing source, needs to change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from city.sources.map_source import MapSource

_SOURCE_REGISTRY: dict[str, "Type[MapSource]"] = {}


def register_source(name: str):
    """Class decorator that registers a MapSource under a config name.

    Args:
        name: The MAP_SOURCE value (case-insensitive) this class
            handles, e.g. "PROCEDURAL".
    """

    def decorator(cls: "Type[MapSource]") -> "Type[MapSource]":
        _SOURCE_REGISTRY[name.upper()] = cls
        return cls

    return decorator


def get_registered_source(name: str) -> "Type[MapSource]":
    """Look up the MapSource subclass registered under `name`.

    Args:
        name: The MAP_SOURCE value to resolve.

    Returns:
        The registered MapSource subclass.

    Raises:
        ValueError: if no source is registered under that name.
    """
    try:
        return _SOURCE_REGISTRY[name.upper()]
    except KeyError as exc:
        available = ", ".join(sorted(_SOURCE_REGISTRY)) or "none"
        raise ValueError(
            f"Unknown MAP_SOURCE '{name}'. Registered sources: {available}"
        ) from exc
