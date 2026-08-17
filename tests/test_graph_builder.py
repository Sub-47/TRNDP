"""
test_graph_builder.py

Locks down RoadGraphBuilder's intersection-splitting and node-merging
behavior against exact node/edge counts.
"""

from __future__ import annotations

import networkx as nx
import pytest

from city.roads.graph_builder import RoadGraphBuilder


def test_x_crossing():
    segments = [((0.0, 0.0), (10.0, 0.0)), ((5.0, -5.0), (5.0, 5.0))]
    graph = RoadGraphBuilder(segments).build()

    assert graph.number_of_nodes() == 5
    assert graph.number_of_edges() == 4
    assert graph.degree[(5.0, 0.0)] == 4


def test_t_junction():
    segments = [((0.0, 0.0), (10.0, 0.0)), ((5.0, 0.0), (5.0, 5.0))]
    graph = RoadGraphBuilder(segments).build()

    assert graph.number_of_nodes() == 4
    assert graph.number_of_edges() == 3
    assert graph.degree[(5.0, 0.0)] == 3


def test_shared_endpoint():
    segments = [((0.0, 0.0), (5.0, 0.0)), ((5.0, 0.0), (5.0, 5.0))]
    graph = RoadGraphBuilder(segments).build()

    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 2
    assert graph.degree[(5.0, 0.0)] == 2
    assert nx.is_connected(graph)


def test_near_miss_merges_within_tolerance():
    segments = [((0.0, 0.0), (5.0, 0.0)), ((5.1, 0.0), (10.0, 0.0))]
    graph = RoadGraphBuilder(segments, snap_tolerance=0.5).build()

    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 2
    assert nx.number_connected_components(graph) == 1


def test_gap_stays_split_with_tight_tolerance():
    segments = [((0.0, 0.0), (5.0, 0.0)), ((5.1, 0.0), (10.0, 0.0))]
    graph = RoadGraphBuilder(segments, snap_tolerance=0.01).build()

    assert graph.number_of_nodes() == 4
    assert graph.number_of_edges() == 2
    assert nx.number_connected_components(graph) == 2


def test_edge_weight_is_euclidean_distance():
    segments = [((0.0, 0.0), (3.0, 4.0))]
    graph = RoadGraphBuilder(segments).build()

    assert graph.number_of_edges() == 1
    weight = graph[(0.0, 0.0)][(3.0, 4.0)]["weight"]
    assert weight == pytest.approx(5.0)


def test_build_is_deterministic():
    segments = [
        ((0.0, 0.0), (10.0, 0.0)),
        ((5.0, -5.0), (5.0, 5.0)),
        ((5.0, 0.0), (5.0, 5.0)),
    ]

    graph_a = RoadGraphBuilder(segments).build()
    graph_b = RoadGraphBuilder(segments).build()

    assert set(graph_a.nodes) == set(graph_b.nodes)
    assert set(graph_a.edges) == set(graph_b.edges)


def test_three_way_crossing():
    segments = [
        ((0.0, 5.0), (10.0, 5.0)),
        ((5.0, 0.0), (5.0, 10.0)),
        ((0.0, 0.0), (10.0, 10.0)),
    ]
    graph = RoadGraphBuilder(segments).build()

    assert graph.number_of_nodes() == 7
    assert graph.number_of_edges() == 6
    assert graph.degree[(5.0, 5.0)] == 6
    assert nx.number_connected_components(graph) == 1


def test_sub_tolerance_segment_is_dropped_and_reported():
    # Endpoint offset must be within roughly HALF the snap tolerance (not
    # the full tolerance) for both ends to round to the same grid point:
    # 0.2 / 0.5 = 0.4 rounds to 0, whereas 0.3 / 0.5 = 0.6 rounds to 1.
    segments = [((0.0, 0.0), (0.2, 0.2))]
    builder = RoadGraphBuilder(segments, snap_tolerance=0.5)
    graph = builder.build()

    assert graph.number_of_edges() == 0
    assert builder.dropped_segments == 1
