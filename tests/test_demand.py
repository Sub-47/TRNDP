"""
tests/test_demand.py

Exercises the graph-based DistanceMatrix and the singly-constrained,
diurnal-aware GravityModel. Uses a small hand-built zone grid + road graph
(never the real generators) for the correctness checks that need an exact,
known topology - including one deliberately unreachable zone - and the
real World/MapManager/road pipeline only for the two checks that are
inherently about the real network (scale invariance, diurnal ratio).
"""

from __future__ import annotations

import functools
import math

import networkx as nx
import numpy as np
import pytest

import config
from city.demand.distance_matrix import DistanceMatrix
from city.demand.gravity_model import GravityModel
from city.demand.zone_map import ZoneMap
from city.generators.population_generator import PopulationGenerator
from city.managers.map_manager import MapManager
from city.models.world import World
from city.roads.connector import connect_components
from city.roads.graph_builder import RoadGraphBuilder
from city.roads.loop_closer import close_loops
from city.roads.segment_grower import RoadNetworkGrower


def _build_test_scenario() -> tuple[ZoneMap, nx.Graph]:
    """A 4x4 zone grid (world_size=16, zone_size=4) whose centroids are
    also road-graph nodes, connected in a 4-connected mesh (no
    diagonals) - so cross-mesh pairs get a genuine graph detour over
    Euclidean. Zone 0's node is added but left edge-free, an isolated
    singleton component distinct from the 15-node mesh, making zone 0
    deliberately unreachable.
    """
    centers_1d = np.array([1.5, 5.5, 9.5, 13.5], dtype=np.float32)
    centers_row, centers_col = np.meshgrid(centers_1d, centers_1d, indexing="ij")
    population = np.full((4, 4), 10.0, dtype=np.float32)
    zones = ZoneMap(
        population=population,
        centers_row=centers_row,
        centers_col=centers_col,
        zone_size=4,
        world_size=16,
    )

    coords = {
        (i, j): (float(centers_1d[j]), float(centers_1d[i])) for i in range(4) for j in range(4)
    }

    graph = nx.Graph()
    for i in range(4):
        for j in range(4):
            if (i, j) == (0, 0):
                continue
            p = coords[(i, j)]
            if i + 1 < 4 and (i + 1, j) != (0, 0):
                q = coords[(i + 1, j)]
                graph.add_edge(p, q, weight=math.dist(p, q))
            if j + 1 < 4 and (i, j + 1) != (0, 0):
                q = coords[(i, j + 1)]
                graph.add_edge(p, q, weight=math.dist(p, q))

    graph.add_node(coords[(0, 0)])  # isolated singleton -> zone 0 unreachable

    return zones, graph


@functools.lru_cache(maxsize=1)
def _real_world_and_graph() -> tuple[World, nx.Graph]:
    """Real world + final road network (grow, connect, close loops),
    built once and cached: only the scale-invariance and diurnal-ratio
    checks need the real pipeline, everything else uses the synthetic
    scenario above for a known, exact topology."""
    map_manager = MapManager()
    world = World(map_manager)
    world.generate()

    population = world.population.data
    obstacle = world.obstacle.data

    population_generator = PopulationGenerator(
        terrain_classes=world.terrain.data,
        obstacle_mask=obstacle,
        world_size=population.shape[0],
        seed=config.SEED,
    )
    population_generator.run()
    starts = [(float(col), float(row)) for row, col in population_generator.centres]

    segments = RoadNetworkGrower(population, obstacle, starts).grow()
    graph = RoadGraphBuilder(segments).build()

    connectors = connect_components(graph, obstacle)
    graph = RoadGraphBuilder(segments + connectors).build()

    loop_edges = close_loops(graph, obstacle)
    graph = RoadGraphBuilder(segments + connectors + loop_edges).build()

    return world, graph


def _total_trips(world: World, graph: nx.Graph, zone_size: int) -> float:
    zones = ZoneMap.from_maps(world.population, world.obstacle, zone_size=zone_size)
    dist = DistanceMatrix.from_graph(zones, graph)
    demand = GravityModel.from_zones_and_distance(zones, dist)
    return float(demand.data.sum())


def test_graph_distance_never_shorter_than_euclidean():
    zones, graph = _build_test_scenario()
    euclidean = DistanceMatrix.from_zone_map(zones)
    graph_dist = DistanceMatrix.from_graph(zones, graph)
    unreachable = set(graph_dist.unreachable_zones)

    checked = 0
    for i in range(zones.num_zones):
        if i in unreachable:
            continue
        for j in range(zones.num_zones):
            if j in unreachable:
                continue
            assert graph_dist.data[i, j] >= euclidean.data[i, j] - 1e-4
            checked += 1
    assert checked > 0


def test_unreachable_zone_gets_inf_distance():
    zones, graph = _build_test_scenario()
    graph_dist = DistanceMatrix.from_graph(zones, graph)

    assert graph_dist.unreachable_zones == [0]
    assert np.isinf(graph_dist.data[0]).all()
    assert np.isinf(graph_dist.data[:, 0]).all()


def test_row_sums_match_population_times_trips_per_person():
    zones, graph = _build_test_scenario()
    graph_dist = DistanceMatrix.from_graph(zones, graph)
    demand = GravityModel.from_zones_and_distance(zones, graph_dist)

    population_flat = zones.population.flatten()
    unreachable = set(graph_dist.unreachable_zones)
    checked = 0
    for i in range(zones.num_zones):
        if i in unreachable or population_flat[i] <= 0:
            continue
        expected = population_flat[i] * config.TRIPS_PER_PERSON
        actual = demand.data[i].sum()
        # float32 throughout (matching this codebase's established
        # convention for ZoneMap/DistanceMatrix/GravityModel storage):
        # relative epsilon at this magnitude is ~2e-6, so a bare
        # absolute 1e-6 is tighter than float32 itself can represent.
        # pytest.approx's default relative tolerance (1e-6) is the
        # numerically sane reading of "within 1e-6" here.
        assert actual == pytest.approx(expected)
        checked += 1
    assert checked > 0


def test_scale_invariance_zone_size_8_vs_16():
    world, graph = _real_world_and_graph()

    total_8 = _total_trips(world, graph, zone_size=8)
    total_16 = _total_trips(world, graph, zone_size=16)

    larger = max(total_8, total_16)
    assert abs(total_8 - total_16) / larger <= 0.05


def test_diurnal_morning_over_off_peak_ratio():
    world, graph = _real_world_and_graph()
    zones = ZoneMap.from_maps(world.population, world.obstacle, zone_size=config.ZONE_SIZE)
    dist = DistanceMatrix.from_graph(zones, graph)

    morning = GravityModel.for_period(zones, dist, "MORNING_PEAK")
    off_peak = GravityModel.for_period(zones, dist, "OFF_PEAK")

    actual_ratio = morning.data.sum() / off_peak.data.sum()
    expected_ratio = config.DIURNAL_FACTORS["MORNING_PEAK"] / config.DIURNAL_FACTORS["OFF_PEAK"]
    assert actual_ratio == pytest.approx(expected_ratio, abs=1e-6)


def test_unknown_period_raises_value_error():
    zones, graph = _build_test_scenario()
    dist = DistanceMatrix.from_graph(zones, graph)

    with pytest.raises(ValueError):
        GravityModel.for_period(zones, dist, "NIGHT_OWL")


def test_no_nan_in_output():
    zones, graph = _build_test_scenario()
    dist = DistanceMatrix.from_graph(zones, graph)
    demand = GravityModel.from_zones_and_distance(zones, dist)

    assert not np.isnan(demand.data).any()


def test_determinism():
    zones, graph = _build_test_scenario()

    dist_a = DistanceMatrix.from_graph(zones, graph)
    dist_b = DistanceMatrix.from_graph(zones, graph)
    assert np.array_equal(dist_a.data, dist_b.data)
    assert dist_a.unreachable_zones == dist_b.unreachable_zones

    demand_a = GravityModel.from_zones_and_distance(zones, dist_a)
    demand_b = GravityModel.from_zones_and_distance(zones, dist_b)
    assert np.array_equal(demand_a.data, demand_b.data)
