"""
city/roads/connector.py

Post-processing pass that stitches disconnected road-network components
into one. RoadNetworkGrower runs each population centre as an independent
BFS growth source with its own segment budget, so a centre far from its
neighbours can end up with a network that never touches the others. This
module finds the shortest obstacle-free path between each orphaned
component and the trunk (largest component) and emits extra segments
covering it, or leaves the fragment alone and logs why if no plausible
path exists.
"""

from __future__ import annotations

import logging
import math

import networkx as nx
import numpy as np

import config

Point = tuple[float, float]
Segment = tuple[Point, Point]
Cell = tuple[int, int]

logger = logging.getLogger(__name__)


def connect_components(graph: nx.Graph, obstacle: np.ndarray) -> list[Segment]:
    """Returns extra segments joining every disconnected component to the
    largest one, or an empty list if the graph is already connected.

    Args:
        graph: The routing graph built by RoadGraphBuilder.
        obstacle: (N, N) bool array, True = impassable, indexed
            ``[row=y, col=x]`` (same convention as RoadNetworkGrower).

    Returns:
        Flat list of ``((x1, y1), (x2, y2))`` connector segments, in the
        same format RoadGraphBuilder consumes.
    """
    components = list(nx.connected_components(graph))
    if len(components) <= 1:
        return []

    components.sort(key=len, reverse=True)
    trunk = set(components[0])
    remaining = components[1:]
    grid_graph = _build_grid_graph(obstacle)

    connectors: list[Segment] = []
    for component in remaining:
        pair = _closest_pair(trunk, component)
        if pair is None:
            continue
        trunk_node, comp_node = pair

        path = _astar_path(grid_graph, trunk_node, comp_node)
        if path is None:
            logger.warning(
                "No obstacle-free path connects component containing %s to "
                "the trunk network; treating it as genuinely unreachable "
                "and leaving it disconnected.",
                comp_node,
            )
            continue

        length = _path_length(path)
        if length > config.ROAD_CONNECTOR_MAX_LENGTH:
            logger.warning(
                "Shortest path to component containing %s is %.1f cells, "
                "over ROAD_CONNECTOR_MAX_LENGTH=%.1f; treating it as "
                "genuinely unreachable rather than building an implausible "
                "connector road.",
                comp_node,
                length,
                config.ROAD_CONNECTOR_MAX_LENGTH,
            )
            continue

        connectors.extend(_path_to_segments(path, obstacle, trunk_node, comp_node))
        trunk |= component

    return connectors


def _build_grid_graph(obstacle: np.ndarray) -> nx.Graph:
    """4-connected grid graph over every non-obstacle cell, keyed by
    ``(row, col)``."""
    rows, cols = obstacle.shape
    grid = nx.grid_2d_graph(rows, cols)
    obstacle_cells = [tuple(cell) for cell in np.argwhere(obstacle)]
    grid.remove_nodes_from(obstacle_cells)
    return grid


def _closest_pair(nodes_a: set[Point], nodes_b: set[Point]) -> tuple[Point, Point] | None:
    """The (a, b) pair with the smallest Euclidean separation, or None if
    either set is empty."""
    best: tuple[Point, Point] | None = None
    best_dist = float("inf")
    for a in nodes_a:
        for b in nodes_b:
            dist = math.dist(a, b)
            if dist < best_dist:
                best_dist = dist
                best = (a, b)
    return best


def _astar_path(grid_graph: nx.Graph, source_xy: Point, target_xy: Point) -> list[Cell] | None:
    """A* path over the obstacle-free grid between two (x, y) graph nodes,
    or None if either endpoint is an obstacle cell or no path exists."""
    source_cell: Cell = (round(source_xy[1]), round(source_xy[0]))
    target_cell: Cell = (round(target_xy[1]), round(target_xy[0]))
    if source_cell not in grid_graph or target_cell not in grid_graph:
        return None
    try:
        return nx.astar_path(grid_graph, source_cell, target_cell, heuristic=math.dist)
    except nx.NetworkXNoPath:
        return None


def _path_length(path: list[Cell]) -> float:
    """Total Euclidean length, in grid cells, of a connected cell path."""
    return sum(math.dist(a, b) for a, b in zip(path, path[1:]))


def _chord_is_safe(p1: Point, p2: Point, obstacle: np.ndarray) -> bool:
    """Whether the straight line between p1 and p2 stays off every
    obstacle cell, checked at a finer resolution than a single grid step
    so a chord can't cut across the corner of a block the A* path itself
    walked around."""
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


def _path_to_segments(
    path: list[Cell], obstacle: np.ndarray, source_xy: Point, target_xy: Point
) -> list[Segment]:
    """Converts a dense unit-step cell path into segments roughly
    ROAD_SEGMENT_LENGTH long, by skipping ahead as many path cells as
    possible per segment - matching the granularity RoadNetworkGrower
    itself produces. A skip is only taken if the straight chord between
    its endpoints is itself obstacle-free; the A* path is guaranteed
    obstacle-free cell by cell, but a chord across several of its cells
    can still cut through a corner the path walked around, so every
    candidate chord is verified and shortened until it's safe.

    The path's own first/last cells are integer grid coordinates, but
    the real graph nodes being bridged can sit at a half-cell offset
    (ROAD_SNAP_TOLERANCE snaps to 0.5). Using the rounded cell instead
    of the exact node would land the connector 0.5 units short - exactly
    ROAD_SNAP_TOLERANCE - so it snaps to a *new* node instead of
    reconnecting to the one it was meant to bridge. source_xy/target_xy
    are substituted in to guarantee the connector actually touches them.
    """
    points: list[Point] = [(float(col), float(row)) for row, col in path]
    if len(points) < 2:
        return []
    points[0] = source_xy
    points[-1] = target_xy

    max_step = max(int(round(config.ROAD_SEGMENT_LENGTH)), 1)
    segments: list[Segment] = []
    start_idx = 0
    last_idx = len(points) - 1
    while start_idx < last_idx:
        end_idx = min(start_idx + max_step, last_idx)
        while end_idx > start_idx + 1 and not _chord_is_safe(
            points[start_idx], points[end_idx], obstacle
        ):
            end_idx -= 1
        segments.append((points[start_idx], points[end_idx]))
        start_idx = end_idx

    return segments
