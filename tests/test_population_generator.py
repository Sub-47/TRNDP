"""
tests/test_population_generator.py

Tests for the city-centre population density model.
"""

from __future__ import annotations

import numpy as np
import pytest

import config
from city.enums.terrain_type import TerrainType, classify_elevation
from city.generators.population_generator import PopulationGenerator
from city.generators.terrain_generator import TerrainGenerator


def _terrain_and_obstacle(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a real (elevation-derived) terrain + obstacle pair for a seed."""
    generator = TerrainGenerator(seed=seed)
    generator.generate()
    generator.normalize()
    generator.apply_sea_level()
    terrain_classes = generator.classify()

    obstacle_values = [TerrainType[name].value for name in config.OBSTACLE_TERRAIN_TYPES]
    obstacle_mask = np.isin(terrain_classes, obstacle_values)
    return terrain_classes, obstacle_mask


def _make_generator(seed: int = config.SEED) -> PopulationGenerator:
    terrain_classes, obstacle_mask = _terrain_and_obstacle(seed)
    return PopulationGenerator(
        terrain_classes=terrain_classes,
        obstacle_mask=obstacle_mask,
        seed=seed,
    )


def test_shape_and_dtype() -> None:
    population = _make_generator().run()
    assert population.shape == (config.WORLD_SIZE, config.WORLD_SIZE)
    assert population.dtype == np.float32


def test_values_within_unit_range() -> None:
    population = _make_generator().run()
    assert population.min() >= 0.0
    assert population.max() <= 1.0


def test_zero_on_obstacle_cells() -> None:
    generator = _make_generator()
    population = generator.run()
    assert np.all(population[generator.obstacle_mask] == 0.0)


def test_same_seed_is_deterministic() -> None:
    population_a = _make_generator(seed=123).run()
    population_b = _make_generator(seed=123).run()
    np.testing.assert_array_equal(population_a, population_b)


def test_different_seeds_differ() -> None:
    population_a = _make_generator(seed=1).run()
    population_b = _make_generator(seed=2).run()
    assert not np.array_equal(population_a, population_b)


def test_centre_count_habitability_and_separation() -> None:
    generator = _make_generator()
    centres = generator.place_centres()

    assert len(centres) == config.NUM_CITY_CENTRES

    for row, col in centres:
        assert not generator.obstacle_mask[row, col]

    for i in range(len(centres)):
        for j in range(i + 1, len(centres)):
            row_i, col_i = centres[i]
            row_j, col_j = centres[j]
            dist = float(np.hypot(row_i - row_j, col_i - col_j))
            assert dist >= config.CENTRE_MIN_SEPARATION
