"""
test_connector.py

Exercises connect_components against small synthetic graphs/obstacle
arrays (never the real terrain/population generators): no-op on an
already-connected graph, bridging two separated fragments, refusing to
cross an impassable wall, refusing connectors over the length cap, and
(the important one) never emitting a connector that cuts through an
obstacle cell.
"""

from __future__ import annotations

import logging
import math

import networkx as nx
import numpy as np
import pytest

import config
from city.roads.connector import connect_components
from city.roads.graph_builder import RoadGraphBuilder

LOGGER_NAME = "city.roads.connector"


def _walk_segment(p1, p2, step=0.25):
    length = math.dist(p1, p2)
    steps = max(int(math.ceil(length / step)), 1)
    for i in range(steps + 1):
        t = i / steps
        yield (p1[0] + (p2[0] - p1[0]) * t, p1[1] + (p2[1] - p1[1]) * t)


def test_already_connected_returns_empty():
    graph = nx.Graph()
    graph.add_edge((0.0, 0.0), (5.0, 0.0))
    graph.add_edge((5.0, 0.0), (5.0, 5.0))
    obstacle = np.zeros((10, 10), dtype=bool)

    assert connect_components(graph, obstacle) == []


def test_bridges_two_separated_fragments():
    obstacle = np.zeros((100, 100), dtype=bool)
    segments = [
        ((10.0, 10.0), (15.0, 10.0)),
        ((15.0, 10.0), (15.0, 15.0)),
        ((80.0, 10.0), (85.0, 10.0)),
        ((85.0, 10.0), (85.0, 15.0)),
    ]
    graph = RoadGraphBuilder(segments).build()
    assert nx.number_connected_components(graph) == 2

    connectors = connect_components(graph, obstacle)
    assert connectors

    combined = segments + connectors
    final_graph = RoadGraphBuilder(combined).build()
    assert nx.number_connected_components(final_graph) == 1


def test_full_width_wall_blocks_connection(caplog):
    obstacle = np.zeros((100, 100), dtype=bool)
    obstacle[:, 40] = True  # full-height wall: nothing can cross column 40

    segments = [
        ((10.0, 50.0), (15.0, 50.0)),
        ((80.0, 50.0), (85.0, 50.0)),
    ]
    graph = RoadGraphBuilder(segments).build()
    assert nx.number_connected_components(graph) == 2

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        connectors = connect_components(graph, obstacle)

    assert connectors == []
    assert any("unreachable" in record.message for record in caplog.records)

    for p1, p2 in connectors:
        for x, y in _walk_segment(p1, p2):
            row, col = round(y), round(x)
            assert not obstacle[row, col]


def test_connector_over_max_length_is_skipped(monkeypatch, caplog):
    monkeypatch.setattr(config, "ROAD_CONNECTOR_MAX_LENGTH", 5.0)
    obstacle = np.zeros((100, 100), dtype=bool)
    segments = [
        ((10.0, 50.0), (15.0, 50.0)),
        ((80.0, 50.0), (85.0, 50.0)),
    ]
    graph = RoadGraphBuilder(segments).build()

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        connectors = connect_components(graph, obstacle)

    assert connectors == []
    assert any("ROAD_CONNECTOR_MAX_LENGTH" in record.message for record in caplog.records)


def test_connector_never_crosses_an_obstacle_cell():
    obstacle = np.zeros((100, 100), dtype=bool)
    obstacle[40:60, 45:55] = True  # a block sitting between the two fragments

    segments = [
        ((30.0, 50.0), (35.0, 50.0)),
        ((65.0, 50.0), (70.0, 50.0)),
    ]
    graph = RoadGraphBuilder(segments).build()
    assert nx.number_connected_components(graph) == 2

    connectors = connect_components(graph, obstacle)
    assert connectors

    for p1, p2 in connectors:
        for x, y in _walk_segment(p1, p2):
            row, col = round(y), round(x)
            assert 0 <= row < obstacle.shape[0]
            assert 0 <= col < obstacle.shape[1]
            assert not obstacle[row, col]
