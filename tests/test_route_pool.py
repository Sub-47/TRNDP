"""
tests/test_route_pool.py

Exercises DemandClusterer and RoutePool against small, hand-built
scenarios (never the real generation pipeline): clustering determinism,
anchor selection, noise promotion, sample_weight actually changing the
clustering outcome, route validity, deduplication, and the minimum-stops
floor.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

from city.routes.cluster import Cluster, DemandClusterer
from city.routes.route_pool import RoutePool


def _distance_matrix(points: list[tuple[float, float]]) -> np.ndarray:
    n = len(points)
    data = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            data[i, j] = math.dist(points[i], points[j])
    return data


def _tight_group_with_outlier():
    """Three mutually-close stops (0, 1, 2) plus one far outlier (3),
    with descending demand within the group so the anchor isn't just
    "whichever comes first"."""
    stops = [(0.0, 0.0), (3.0, 0.0), (6.0, 0.0), (1000.0, 1000.0)]
    stop_distances = _distance_matrix(stops)
    stop_demand = np.array([15.0, 5.0, 2.0, 1.0])
    return stops, stop_distances, stop_demand


def test_clustering_is_deterministic():
    stops, stop_distances, stop_demand = _tight_group_with_outlier()

    clusters_a = DemandClusterer(
        stops, stop_distances, stop_demand, eps=10.0, min_samples=3
    ).cluster()
    clusters_b = DemandClusterer(
        stops, stop_distances, stop_demand, eps=10.0, min_samples=3
    ).cluster()

    key = lambda cs: sorted((tuple(c.members), c.anchor, c.demand) for c in cs)
    assert key(clusters_a) == key(clusters_b)


def test_anchor_is_highest_demand_stop_in_its_cluster():
    stops, stop_distances, stop_demand = _tight_group_with_outlier()

    clusters = DemandClusterer(
        stops, stop_distances, stop_demand, eps=10.0, min_samples=3
    ).cluster()

    for cluster in clusters:
        best_in_cluster = max(cluster.members, key=lambda i: stop_demand[i])
        assert cluster.anchor == best_in_cluster

    # The 3-stop group's anchor should specifically be stop 0 (demand 15),
    # not just internally consistent.
    group_cluster = next(c for c in clusters if len(c.members) == 3)
    assert group_cluster.anchor == 0


def test_noise_points_become_single_stop_clusters_none_dropped():
    stops, stop_distances, stop_demand = _tight_group_with_outlier()

    clusterer = DemandClusterer(stops, stop_distances, stop_demand, eps=10.0, min_samples=3)
    clusters = clusterer.cluster()

    all_members = sorted(i for cluster in clusters for i in cluster.members)
    assert all_members == [0, 1, 2, 3]  # every stop accounted for, none dropped

    assert clusterer.noise_promoted == 1
    outlier_clusters = [c for c in clusters if c.members == [3]]
    assert len(outlier_clusters) == 1
    assert outlier_clusters[0].anchor == 3


def test_sample_weight_changes_clustering_outcome():
    # Two stops, far enough apart that neither is ever within eps of the
    # other - each stop's own weight is the only thing that can qualify
    # it as a DBSCAN core point.
    stops = [(0.0, 0.0), (1000.0, 0.0)]
    stop_distances = _distance_matrix(stops)
    eps, min_samples = 10.0, 2

    uniform_weights = np.array([1.0, 1.0])  # both below min_samples -> both noise
    real_weights = np.array([5.0, 0.5])  # stop 0 alone clears min_samples -> core

    uniform_clusterer = DemandClusterer(
        stops, stop_distances, uniform_weights, eps=eps, min_samples=min_samples
    )
    uniform_clusterer.cluster()

    real_clusterer = DemandClusterer(
        stops, stop_distances, real_weights, eps=eps, min_samples=min_samples
    )
    real_clusterer.cluster()

    assert uniform_clusterer.noise_promoted == 2
    assert real_clusterer.noise_promoted == 1
    assert uniform_clusterer.noise_promoted != real_clusterer.noise_promoted


# --- RoutePool -------------------------------------------------------------
#
# A short chain graph n0-n1-n2-n3-n4, plus a second, node-disjoint detour
# of the same length through x1/x2 that also passes through n2. Every
# simple path from n0 to n4 in this graph must pass through n2 (the only
# cut vertex), so every one of them reduces to the same filtered stop
# sequence [n0, n2, n4] - a clean, deterministic way to exercise both
# duplicate rejection and the minimum-stops floor without depending on
# Yen's internal tie-breaking order.


def _chain_graph_with_detour() -> nx.Graph:
    n0, n1, n2, n3, n4 = (0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)
    x1, x2 = (1.0, 1.0), (3.0, 1.0)

    graph = nx.Graph()
    for a, b in [(n0, n1), (n1, n2), (n2, n3), (n3, n4), (n0, x1), (x1, n2), (n2, x2), (x2, n4)]:
        graph.add_edge(a, b, weight=math.dist(a, b))
    return graph


def _chain_stops_and_clusters():
    n0, n2, n4 = (0.0, 0.0), (2.0, 0.0), (4.0, 0.0)
    stops = [n0, n2, n4]  # indices 0, 1, 2
    clusters = [
        Cluster(members=[0], anchor=0, demand=1.0),
        Cluster(members=[2], anchor=2, demand=1.0),
    ]
    return stops, clusters


def test_every_route_is_a_valid_path_in_the_graph():
    graph = _chain_graph_with_detour()
    stops, clusters = _chain_stops_and_clusters()

    pool = RoutePool(graph, clusters, stops, routes_per_pair=5, min_stops=3, max_length=200.0)
    routes = pool.generate()

    assert routes
    for route in routes:
        for a, b in zip(route.stops, route.stops[1:]):
            assert nx.has_path(graph, stops[a], stops[b])


def test_no_duplicate_stop_sequences_in_pool():
    graph = _chain_graph_with_detour()
    stops, clusters = _chain_stops_and_clusters()

    pool = RoutePool(graph, clusters, stops, routes_per_pair=5, min_stops=3, max_length=200.0)
    routes = pool.generate()

    sequences = [tuple(route.stops) for route in routes]
    assert len(sequences) == len(set(sequences))
    # Every simple path here funnels through n2, so every one of the
    # (up to 5) shortest paths collapses to the same [0, 1, 2] sequence -
    # dedup must actually reject the repeats, not just coincidentally
    # avoid producing any.
    assert len(routes) == 1
    assert pool.rejected_duplicate >= 1


def test_every_route_has_at_least_min_stops():
    graph = _chain_graph_with_detour()
    stops, clusters = _chain_stops_and_clusters()

    pool = RoutePool(graph, clusters, stops, routes_per_pair=5, min_stops=3, max_length=200.0)
    routes = pool.generate()

    assert routes
    for route in routes:
        assert len(route.stops) >= 3
