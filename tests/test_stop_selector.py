"""
tests/test_stop_selector.py

Exercises StopSelector against small synthetic graphs/population arrays
(never the real road-growth pipeline): target count reached when the
network permits it, every stop is a real graph node, spacing is honoured,
stops come only from the largest component, an impossible spacing shorts
gracefully instead of hanging, spread (not clustering) under uniform
population, and determinism.
"""

from __future__ import annotations

import logging
import math

import networkx as nx
import numpy as np
import pytest

from city.roads.stop_selector import StopSelector


def _grid_graph(size: int, spacing: float = 4.0) -> nx.Graph:
    """A size x size mesh of nodes `spacing` cells apart, 4-connected."""
    graph = nx.Graph()
    for x in range(size):
        for y in range(size):
            if x + 1 < size:
                a = (float(x) * spacing, float(y) * spacing)
                b = (float(x + 1) * spacing, float(y) * spacing)
                graph.add_edge(a, b, weight=math.dist(a, b))
            if y + 1 < size:
                a = (float(x) * spacing, float(y) * spacing)
                b = (float(x) * spacing, float(y + 1) * spacing)
                graph.add_edge(a, b, weight=math.dist(a, b))
    return graph


def _uniform_population(size: int = 200) -> np.ndarray:
    return np.ones((size, size), dtype=np.float32)


def test_reaches_target_count_when_feasible():
    graph = _grid_graph(10, spacing=4.0)  # 100 nodes, plenty for 20 @ spacing 6
    population = _uniform_population()

    selector = StopSelector(graph, population, target_count=20, min_spacing=6.0)
    stops = selector.select()

    assert len(stops) == 20
    assert selector.shortfall == 0


def test_every_stop_is_a_graph_node():
    graph = _grid_graph(10, spacing=4.0)
    population = _uniform_population()

    selector = StopSelector(graph, population, target_count=20, min_spacing=6.0)
    stops = selector.select()

    assert stops
    for stop in stops:
        assert stop in graph.nodes


def test_minimum_spacing_respected():
    graph = _grid_graph(10, spacing=4.0)
    population = _uniform_population()

    selector = StopSelector(graph, population, target_count=20, min_spacing=6.0)
    stops = selector.select()

    assert stops
    for i, a in enumerate(stops):
        for b in stops[i + 1 :]:
            assert math.dist(a, b) >= 6.0


def test_stops_come_only_from_largest_component():
    graph = _grid_graph(8, spacing=4.0)  # 64-node main cluster
    # A small, physically distant, disconnected cluster - never touches
    # the main graph, so it never gets a shared node either.
    small = nx.Graph()
    for x in range(3):
        for y in range(3):
            if x + 1 < 3:
                a, b = (1000.0 + x * 4.0, y * 4.0), (1000.0 + (x + 1) * 4.0, y * 4.0)
                small.add_edge(a, b, weight=math.dist(a, b))
            if y + 1 < 3:
                a, b = (1000.0 + x * 4.0, y * 4.0), (1000.0 + x * 4.0, (y + 1) * 4.0)
                small.add_edge(a, b, weight=math.dist(a, b))
    graph = nx.compose(graph, small)
    assert nx.number_connected_components(graph) == 2

    population = _uniform_population(size=1100)  # big enough to cover both clusters
    # Make the small cluster's catchment score enormous so it would win
    # on population alone if component-restriction weren't enforced.
    population[0:20, 1000:1020] = 1000.0

    selector = StopSelector(graph, population, target_count=40, min_spacing=6.0)
    stops = selector.select()

    small_nodes = set(small.nodes)
    assert stops
    assert all(stop not in small_nodes for stop in stops)


def test_impossible_spacing_shorts_gracefully(caplog):
    graph = _grid_graph(6, spacing=4.0)  # 36 nodes, world only ~20 cells wide
    population = _uniform_population()

    selector = StopSelector(graph, population, target_count=40, min_spacing=500.0)
    with caplog.at_level(logging.WARNING, logger="city.roads.stop_selector"):
        stops = selector.select()

    assert len(stops) < 40
    assert selector.shortfall == 40 - len(stops)
    assert selector.shortfall > 0
    assert any("shortfall" in record.message for record in caplog.records)


def test_uniform_population_gives_spread_not_a_cluster():
    graph = _grid_graph(10, spacing=4.0)
    population = _uniform_population()

    selector = StopSelector(graph, population, target_count=20, min_spacing=6.0)
    stops = selector.select()

    xs = [x for x, _y in stops]
    ys = [y for _x, y in stops]
    # A cluster-in-the-corner failure mode would have a tiny bounding
    # box; real spread should cover most of the 0-36 node-grid extent.
    assert max(xs) - min(xs) > 20.0
    assert max(ys) - min(ys) > 20.0


def test_determinism():
    graph = _grid_graph(10, spacing=4.0)
    population = _uniform_population()

    stops_a = StopSelector(graph, population, target_count=20, min_spacing=6.0).select()
    stops_b = StopSelector(graph, population, target_count=20, min_spacing=6.0).select()

    assert stops_a == stops_b
