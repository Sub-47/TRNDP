"""
city/demand/distance_matrix.py

Computes pairwise distances between zone centers - either Euclidean
(straight-line, kept for comparison) or over the actual road graph.

Euclidean distance treats a lake or a mountain range as no obstacle at
all: a zone pair separated by water gets the same distance as an equally
far-apart pair on flat, connected land, even though nobody can actually
travel that line. Since gravity demand goes as 1/distance**beta, the
resulting under-estimate is squared - exactly the error this project
exists to point out, so `from_graph` walks the real road network instead.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
from dataclasses import dataclass, field

import config
from city.demand.zone_map import ZoneMap


@dataclass
class DistanceMatrix:
    """Pairwise distances between all zones.

    Attributes:
        data: float32 array of shape (zones_per_side, zones_per_side)
            containing the distance between each pair of zones. Entries
            for pairs involving an unreachable zone are ``np.inf``.
        zone_size: Number of grid cells along each side of one zone.
        world_size: Size of the underlying cell grid.
        unreachable_zones: Flat indices of zones whose access node isn't
            in the road graph's largest connected component. Empty for
            `from_zone_map`, which has no notion of reachability.
    """

    data: np.ndarray
    zone_size: int = field(default=config.ZONE_SIZE)
    world_size: int = field(default=config.WORLD_SIZE)
    unreachable_zones: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        expected_shape = (self.num_zones, self.num_zones)
        if self.data.shape != expected_shape:
            raise ValueError(
                f"data must have shape {expected_shape}, got {self.data.shape}"
            )

    @property
    def zones_per_side(self) -> int:
        return self.world_size // self.zone_size

    @property
    def num_zones(self) -> int:
        return self.zones_per_side ** 2

    @classmethod
    def from_zone_map(cls, zones: ZoneMap) -> "DistanceMatrix":
        """Build a Euclidean (straight-line) distance matrix from zone
        center coordinates. Kept for comparison against `from_graph` -
        not what demand modelling should actually use."""
        _, centers_row, centers_col = zones.flatten()
        row_diff = centers_row[:, np.newaxis] - centers_row[np.newaxis, :]
        col_diff = centers_col[:, np.newaxis] - centers_col[np.newaxis, :]
        distances = np.hypot(row_diff, col_diff).astype(np.float32)
        return cls(
            data=distances,
            zone_size=zones.zone_size,
            world_size=zones.world_size,
        )

    @classmethod
    def from_graph(cls, zones: ZoneMap, graph: nx.Graph) -> "DistanceMatrix":
        """Build a distance matrix from shortest paths over the road
        graph, not straight lines.

        Each zone centroid is assigned an "access point" - its nearest
        node in the road graph. Zone-to-zone distance is the walk from
        the centroid to its access node, plus the graph shortest-path
        distance between the two access nodes, plus the walk from the
        other access node to its centroid. Zones whose access node falls
        outside the graph's largest connected component are unreachable
        (distance ``np.inf`` to and from everything, including each
        other) rather than silently given some finite fallback distance -
        the underlying network genuinely does not reach them.

        Args:
            zones: Zone population/center grid.
            graph: Routing graph built by RoadGraphBuilder - nodes are
                (x, y) points, edges carry a ``weight`` attribute.

        Returns:
            A new DistanceMatrix, with `unreachable_zones` populated.
        """
        _, centers_row, centers_col = zones.flatten()
        num_zones = zones.num_zones

        nodes = list(graph.nodes)
        if not nodes:
            raise ValueError("graph has no nodes")
        largest_component = max(nx.connected_components(graph), key=len)

        access_nodes: list[tuple[float, float]] = []
        access_walk: list[float] = []
        for i in range(num_zones):
            centroid = (float(centers_col[i]), float(centers_row[i]))  # graph is (x, y)
            nearest = min(nodes, key=lambda node: math.dist(node, centroid))
            access_nodes.append(nearest)
            access_walk.append(math.dist(nearest, centroid))

        unreachable_zones = [
            i for i in range(num_zones) if access_nodes[i] not in largest_component
        ]
        unreachable_set = set(unreachable_zones)

        # One Dijkstra run per distinct access node (zones can share one),
        # never per zone pair.
        unique_access_nodes = {
            access_nodes[i] for i in range(num_zones) if i not in unreachable_set
        }
        distances_from = {
            node: nx.single_source_dijkstra_path_length(graph, node, weight="weight")
            for node in unique_access_nodes
        }

        data = np.full((num_zones, num_zones), np.inf, dtype=np.float32)
        for i in range(num_zones):
            if i in unreachable_set:
                continue
            distances_from_i = distances_from[access_nodes[i]]
            for j in range(num_zones):
                if j in unreachable_set:
                    continue
                graph_distance = distances_from_i.get(access_nodes[j], np.inf)
                data[i, j] = access_walk[i] + graph_distance + access_walk[j]

        return cls(
            data=data,
            zone_size=zones.zone_size,
            world_size=zones.world_size,
            unreachable_zones=unreachable_zones,
        )
