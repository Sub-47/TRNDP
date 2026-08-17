"""
city/roads/loop_closer.py

Post-processing pass that adds redundancy to an otherwise tree-shaped road
network. RoadNetworkGrower's discard-on-redundant behaviour means two
branches growing toward each other never actually meet, so the grown (plus
connected) network naturally comes out as close to a tree - zero or few
independent cycles. That starves Yen's k-shortest-paths of alternate
routes. This module finds node pairs that are physically close but
topologically far (a large detour ratio through the existing graph) and
emits a direct segment between them - exactly where a real city would have
a connecting street - or leaves the pair alone if the straight line would
cross an obstacle.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np

import config

Point = tuple[float, float]
Segment = tuple[Point, Point]


def close_loops(graph: nx.Graph, obstacle: np.ndarray) -> list[Segment]:
    """Returns extra segments joining node pairs that are near in space
    but far in graph distance, adding cycles to the network.

    Args:
        graph: The routing graph built by RoadGraphBuilder (ideally after
            connect_components, so distances are measured through one
            connected network rather than across separate components).
        obstacle: (N, N) bool array, True = impassable, indexed
            ``[row=y, col=x]`` (same convention as RoadNetworkGrower).

    Returns:
        Flat list of ``((x1, y1), (x2, y2))`` loop-closing segments, in
        the same format RoadGraphBuilder consumes.
    """
    candidates = _find_candidates(graph)
    candidates.sort(key=lambda candidate: candidate[2], reverse=True)

    working = graph.copy()
    new_edges: list[Segment] = []

    for u, v, _ratio in candidates:
        if len(new_edges) >= config.LOOP_MAX_NEW_EDGES:
            break

        # Closing an earlier loop can shorten the graph distance between
        # u and v (or make them adjacent outright), so a candidate that
        # looked valuable at sort time may no longer clear the detour
        # threshold - recheck against the working graph, not the
        # original.
        if not _still_qualifies(working, u, v):
            continue

        if not _chord_is_safe(u, v, obstacle):
            continue

        new_edges.append((u, v))
        working.add_edge(u, v, weight=math.dist(u, v))

    return new_edges


def _find_candidates(graph: nx.Graph) -> list[tuple[Point, Point, float]]:
    """(u, v, detour_ratio) for every node pair within LOOP_MAX_GAP whose
    graph-distance / Euclidean-distance ratio clears LOOP_MIN_DETOUR.

    One single-source Dijkstra run per node covers that node's distance
    to every other node at once, rather than a Dijkstra call per pair.
    """
    nodes = list(graph.nodes)
    candidates: list[tuple[Point, Point, float]] = []

    for i, u in enumerate(nodes):
        distances = nx.single_source_dijkstra_path_length(graph, u, weight="weight")
        for v in nodes[i + 1 :]:
            euclidean = math.dist(u, v)
            if euclidean <= 0.0 or euclidean >= config.LOOP_MAX_GAP:
                continue
            graph_distance = distances.get(v, math.inf)
            ratio = graph_distance / euclidean
            if ratio >= config.LOOP_MIN_DETOUR:
                candidates.append((u, v, ratio))

    return candidates


def _still_qualifies(working: nx.Graph, u: Point, v: Point) -> bool:
    euclidean = math.dist(u, v)
    if euclidean <= 0.0:
        return False
    try:
        graph_distance = nx.shortest_path_length(working, u, v, weight="weight")
    except nx.NetworkXNoPath:
        return True  # still unreachable -> effectively infinite ratio
    return graph_distance / euclidean >= config.LOOP_MIN_DETOUR


def _chord_is_safe(p1: Point, p2: Point, obstacle: np.ndarray) -> bool:
    """Whether the straight line between p1 and p2 stays off every
    obstacle cell, walked at a finer resolution (0.25 units) than a
    single grid step."""
    rows, cols = obstacle.shape
    length = math.dist(p1, p2)
    steps = max(int(math.ceil(length / 0.25)), 1)
    for i in range(steps + 1):
        t = i / steps
        x = p1[0] + (p2[0] - p1[0]) * t
        y = p1[1] + (p2[1] - p1[1]) * t
        row, col = round(y), round(x)
        if not (0 <= row < rows and 0 <= col < cols) or obstacle[row, col]:
            return False
    return True
