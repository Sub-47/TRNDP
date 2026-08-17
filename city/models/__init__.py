"""
city/models package

Domain models. Currently just `World`, which coordinates map retrieval
via MapManager but holds no generation logic of its own.
"""

from city.models.world import World

__all__ = ["World"]
