"""
city/roads/stop_selector.py

Selects bus stops from a built road graph: greedy by population catchment,
with a minimum-spacing exclusion so selection spreads across the city
instead of piling every stop into the CBD (which is always the highest-
scoring spot on its own). Restricted to the graph's largest connected
component - a stop on a smaller, genuinely unreachable fragment could
never be served by any route, so it isn't a candidate at all, not just a
low-scoring one.
"""

from __future__ import annotations

import logging
import math

import networkx as nx
import numpy as np

import config

Point = tuple[float, float]
Cell = tuple[int, int]

logger = logging.getLogger(__name__)


def _disk_offsets(radius: float) -> list[Cell]:
    """Integer (drow, dcol) offsets of every cell within ``radius`` of a
    center cell, used to sum population over a catchment area."""
    r = int(math.ceil(radius))
    return [
        (dr, dc)
        for dr in range(-r, r + 1)
        for dc in range(-r, r + 1)
        if dr * dr + dc * dc <= radius * radius
    ]


class StopSelector:
    """Greedy population-catchment stop selection with spacing exclusion.

    Args:
        graph: Routing graph built by RoadGraphBuilder - nodes are (x, y)
            points, edges carry a ``weight`` attribute (unused here;
            selection only looks at node positions and population).
        population: (N, N) float array, indexed ``[row=y, col=x]`` (same
            convention as RoadNetworkGrower/DistanceMatrix).
        target_count: Number of stops to select, at most.
        min_spacing: Minimum distance, in cells, enforced between any two
            selected stops.

    Attributes:
        shortfall: Set by `select()` to `target_count - len(result)` if
            fewer than `target_count` stops could be placed (0 otherwise).
    """

    def __init__(
        self,
        graph: nx.Graph,
        population: np.ndarray,
        target_count: int = config.TARGET_STOP_COUNT,
        min_spacing: float = config.STOP_MIN_SPACING,
    ) -> None:
        self.graph = graph
        self.population = population
        self.target_count = target_count
        self.min_spacing = min_spacing
        self.shortfall = 0
        self._catchment_offsets = _disk_offsets(config.STOP_CATCHMENT_RADIUS)

    def select(self) -> list[Point]:
        """Returns up to `target_count` stops, all from the graph's
        largest connected component, no two closer than `min_spacing`."""
        if self.graph.number_of_nodes() == 0:
            self.shortfall = self.target_count
            return []

        largest = max(nx.connected_components(self.graph), key=len)
        nodes = sorted(largest)

        scores = {node: self._catchment_score(node) for node in nodes}
        # Highest score first; ties broken by node coordinate so the
        # result is deterministic regardless of set/dict iteration order.
        ordered = sorted(nodes, key=lambda node: (-scores[node], node))

        excluded: set[Point] = set()
        stops: list[Point] = []

        for node in ordered:
            if len(stops) >= self.target_count:
                break
            if node in excluded:
                continue

            stops.append(node)
            for other in nodes:
                if other == node or other in excluded:
                    continue
                if math.dist(other, node) < self.min_spacing:
                    excluded.add(other)

        self.shortfall = max(self.target_count - len(stops), 0)
        if self.shortfall > 0:
            logger.warning(
                "Only placed %d/%d stops (shortfall=%d) at min_spacing=%s; "
                "the largest component doesn't have enough eligible nodes "
                "at this spacing.",
                len(stops),
                self.target_count,
                self.shortfall,
                self.min_spacing,
            )

        return stops

    def _catchment_score(self, node: Point) -> float:
        x, y = node
        row, col = round(y), round(x)
        rows, cols = self.population.shape
        total = 0.0
        for dr, dc in self._catchment_offsets:
            r, c = row + dr, col + dc
            if 0 <= r < rows and 0 <= c < cols:
                total += float(self.population[r, c])
        return total
