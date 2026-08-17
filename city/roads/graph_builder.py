"""
graph_builder.py

Converts flat road line segments into a routable NetworkX graph, splitting
segments at their pairwise crossings so shortest-path search can turn
through intersections rather than skating over them.
"""

from __future__ import annotations

import logging
import math

import networkx as nx

import config

Point = tuple[float, float]
Segment = tuple[Point, Point]

logger = logging.getLogger(__name__)


class RoadGraphBuilder:
    """Builds an intersection-aware routing graph from raw road segments.

    Args:
        segments: Flat list of ``((x1, y1), (x2, y2))`` road segments.
        snap_tolerance: Grid size coordinates are snapped to so that
            near-identical points collapse into one node.
    """

    def __init__(
        self,
        segments: list[Segment],
        snap_tolerance: float = config.ROAD_SNAP_TOLERANCE,
    ) -> None:
        self.segments = segments
        self.snap_tolerance = snap_tolerance
        self.dropped_segments = 0

    def build(self) -> nx.Graph:
        """Builds the routing graph.

        Returns:
            An undirected graph whose nodes are snapped ``(x, y)`` points
            and whose edges carry a ``weight`` attribute equal to the
            Euclidean distance between their endpoints.
        """
        graph = nx.Graph()
        crossings = self._find_crossings()

        for index, segment in enumerate(self.segments):
            p1, p2 = segment
            points_with_t: list[tuple[float, Point]] = [(0.0, p1), (1.0, p2)]
            for t, point in crossings.get(index, []):
                points_with_t.append((t, point))
            points_with_t.sort(key=lambda item: item[0])

            snapped_points = [self._snap(point) for _, point in points_with_t]
            for a, b in zip(snapped_points, snapped_points[1:]):
                if a == b:
                    self.dropped_segments += 1
                    continue
                weight = math.dist(a, b)
                graph.add_edge(a, b, weight=weight)

        if self.dropped_segments > 0:
            logger.warning(
                "Dropped %d zero-length sub-edge(s) at snap_tolerance=%s",
                self.dropped_segments,
                self.snap_tolerance,
            )

        return graph

    def _find_crossings(self) -> dict[int, list[tuple[float, Point]]]:
        """Maps each segment index to its ``(t, point)`` interior crossings."""
        # TODO: replace the O(n^2) pair scan with an STRtree spatial index
        # once segment counts grow beyond prototype scale.
        crossings: dict[int, list[tuple[float, Point]]] = {}
        n = len(self.segments)
        for i in range(n):
            for j in range(i + 1, n):
                hit = self._intersect(self.segments[i], self.segments[j])
                if hit is None:
                    continue
                t, u, point = hit
                crossings.setdefault(i, []).append((t, point))
                crossings.setdefault(j, []).append((u, point))
        return crossings

    @staticmethod
    def _intersect(
        seg_a: Segment, seg_b: Segment
    ) -> tuple[float, float, Point] | None:
        """Returns ``(t, u, point)`` where seg_a/seg_b cross, else None.

        Uses the 2D cross-product parametrization. Parallel segments
        (|d| < 1e-9) are skipped; prototype scope does not handle
        collinear overlap.
        """
        p1, p2 = seg_a
        q1, q2 = seg_b
        rx, ry = p2[0] - p1[0], p2[1] - p1[1]
        sx, sy = q2[0] - q1[0], q2[1] - q1[1]

        d = rx * sy - ry * sx
        if abs(d) < 1e-9:
            return None

        qpx, qpy = q1[0] - p1[0], q1[1] - p1[1]
        t = (qpx * sy - qpy * sx) / d
        u = (qpx * ry - qpy * rx) / d

        if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
            point = (p1[0] + t * rx, p1[1] + t * ry)
            return t, u, point
        return None

    def _snap(self, point: Point) -> Point:
        """Rounds a point to the nearest multiple of ``snap_tolerance``."""
        tol = self.snap_tolerance
        return (round(point[0] / tol) * tol, round(point[1] / tol) * tol)
