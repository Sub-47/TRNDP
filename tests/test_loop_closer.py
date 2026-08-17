"""
test_loop_closer.py

Exercises close_loops against small synthetic graphs (never the real
grower/terrain pipeline): a graph already rich in cycles stays roughly as
is, a long thin path with adjacent ends gets closed into a loop, an
obstacle wall blocks a would-be loop edge outright, the new-edge cap is
honoured, and output is deterministic.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

import config
from city.roads.loop_closer import close_loops


def _walk(p1, p2, step=0.25):
    length = math.dist(p1, p2)
    steps = max(int(math.ceil(length / step)), 1)
    for i in range(steps + 1):
        t = i / steps
        yield (p1[0] + (p2[0] - p1[0]) * t, p1[1] + (p2[1] - p1[1]) * t)


def _grid_graph(size: int) -> nx.Graph:
    """A size x size mesh of unit-spaced nodes, 4-connected - already
    dense with cycles, so no pair should be topologically "far"."""
    graph = nx.Graph()
    for x in range(size):
        for y in range(size):
            if x + 1 < size:
                graph.add_edge((float(x), float(y)), (float(x + 1), float(y)), weight=1.0)
            if y + 1 < size:
                graph.add_edge((float(x), float(y)), (float(x), float(y + 1)), weight=1.0)
    return graph


def _path_graph(points: list[tuple[float, float]]) -> nx.Graph:
    graph = nx.Graph()
    for a, b in zip(points, points[1:]):
        graph.add_edge(a, b, weight=math.dist(a, b))
    return graph


def test_dense_mesh_gains_few_or_no_edges():
    graph = _grid_graph(6)
    obstacle = np.zeros((20, 20), dtype=bool)

    edges = close_loops(graph, obstacle)

    # A 6x6 grid's worst-case detour ratio (opposite corners) is well
    # under LOOP_MIN_DETOUR: Manhattan distance / Euclidean distance for
    # a 5x5-cell diagonal is 10 / (5*sqrt(2)) ~= 1.41.
    assert len(edges) == 0


def test_long_thin_loop_gains_a_closing_edge():
    points = [(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0), (1.0, 0.0)]
    graph = _path_graph(points)
    obstacle = np.zeros((20, 20), dtype=bool)

    components_before = nx.number_connected_components(graph)
    cycles_before = graph.number_of_edges() - graph.number_of_nodes() + components_before
    assert cycles_before == 0

    edges = close_loops(graph, obstacle)
    assert edges

    rebuilt = graph.copy()
    for u, v in edges:
        rebuilt.add_edge(u, v, weight=math.dist(u, v))

    components_after = nx.number_connected_components(rebuilt)
    cycles_after = rebuilt.number_of_edges() - rebuilt.number_of_nodes() + components_after
    assert cycles_after >= cycles_before + 1


def test_obstacle_wall_blocks_loop_edge():
    points = [(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0), (1.0, 0.0)]
    graph = _path_graph(points)

    obstacle = np.zeros((20, 20), dtype=bool)
    obstacle[0, 1] = True  # sits on the straight line between (0,0) and (1,0)

    edges = close_loops(graph, obstacle)

    assert edges == []
    for p1, p2 in edges:
        for x, y in _walk(p1, p2):
            row, col = round(y), round(x)
            assert not obstacle[row, col]


def test_max_new_edges_honoured():
    # Many independent "V" shapes, each with one strong loop candidate
    # (two arms 10 units long meeting a gap of 1 unit), spaced far apart
    # (100 units) so they can never qualify against each other.
    graph = nx.Graph()
    count = config.LOOP_MAX_NEW_EDGES + 10
    for i in range(count):
        cx = 100.0 * i
        a = (cx, 0.0)
        b = (cx, 10.0)
        c = (cx + 1.0, 0.0)
        graph.add_edge(a, b, weight=math.dist(a, b))
        graph.add_edge(b, c, weight=math.dist(b, c))

    # Sized to actually cover every V's coordinates (cx runs up to
    # 100*(count-1)); an undersized array would make every chord bounds
    # check fail and trivially return no edges, which wouldn't exercise
    # the cap at all.
    obstacle = np.zeros((20, int(100 * count) + 20), dtype=bool)

    edges = close_loops(graph, obstacle)

    assert len(edges) <= config.LOOP_MAX_NEW_EDGES


def test_determinism():
    graph = nx.Graph()
    count = 15
    for i in range(count):
        cx = 100.0 * i
        a = (cx, 0.0)
        b = (cx, 10.0)
        c = (cx + 1.0, 0.0)
        graph.add_edge(a, b, weight=math.dist(a, b))
        graph.add_edge(b, c, weight=math.dist(b, c))

    obstacle = np.zeros((20, int(100 * count) + 20), dtype=bool)

    edges_a = close_loops(graph, obstacle)
    edges_b = close_loops(graph, obstacle)

    assert edges_a == edges_b
