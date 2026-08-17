"""
city/generators package

Concrete generation algorithms: procedural (Perlin noise) terrain
generation and city-centre-driven population generation. Future
milestones may add road-network or lot generators here.
"""

from city.generators.population_generator import PopulationGenerator
from city.generators.terrain_generator import TerrainGenerator

__all__ = ["TerrainGenerator", "PopulationGenerator"]
