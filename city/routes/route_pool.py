"""
city/routes/route_pool.py

Generates the pool of candidate bus routes the GA selects from and
recombines - it never invents a route itself. For every pair of cluster
anchors, takes the first `routes_per_pair` shortest simple paths
(`networkx.shortest_simple_paths`, i.e. Yen's algorithm) over the road
graph and converts each into the ordered sequence of STOPS it passes
through, discarding a road-node path's non-stop intersections.
"""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass

import networkx as nx

import config
from city.routes.cluster import Cluster, Point

logger = logging.getLogger(__name__)


@dataclass
class Route:
    """One candidate route.

    Attributes:
        stops: Ordered indices into the `stop_nodes` list this route was
            built from - a route is a sequence of stops, not road nodes.
        length: Total length, in cells, along the road graph (i.e. over
            every intersection the route passes through, not just the
            stops).
    """

    stops: list[int]
    length: float


class RoutePool:
    """Builds candidate routes between every pair of cluster anchors.

    Args:
        graph: Routing graph built by RoadGraphBuilder.
        clusters: Clusters from DemandClusterer.cluster(); routes run
            between every unordered pair of cluster anchors.
        stop_nodes: Stop coordinates, indexed the same way as the stop
            indices used by `clusters` (i.e. `Cluster.anchor`/`members`).
        routes_per_pair: Number of Yen's shortest paths taken per anchor
            pair.
        min_stops: Routes with fewer stops than this are rejected.
        max_length: Routes longer than this, in cells, are rejected.

    Attributes:
        rejected_too_short: Routes discarded for having fewer than
            `min_stops` stops.
        rejected_too_long: Routes discarded for exceeding `max_length`.
        rejected_duplicate: Routes discarded for repeating an already
            accepted route's stop sequence.
    """

    def __init__(
        self,
        graph: nx.Graph,
        clusters: list[Cluster],
        stop_nodes: list[Point],
        routes_per_pair: int = config.ROUTES_PER_PAIR,
        min_stops: int = config.ROUTE_MIN_STOPS,
        max_length: float = config.ROUTE_MAX_LENGTH,
    ) -> None:
        self.graph = graph
        self.clusters = clusters
        self.stop_nodes = stop_nodes
        self.routes_per_pair = routes_per_pair
        self.min_stops = min_stops
        self.max_length = max_length

        self.rejected_too_short = 0
        self.rejected_too_long = 0
        self.rejected_duplicate = 0

    def generate(self) -> list[Route]:
        """Returns the candidate route pool.

        `shortest_simple_paths` is a generator that can in principle
        enumerate every simple path between two nodes; only the first
        `routes_per_pair` are ever pulled from it.
        """
        self.rejected_too_short = 0
        self.rejected_too_long = 0
        self.rejected_duplicate = 0

        node_to_stop = {node: index for index, node in enumerate(self.stop_nodes)}
        anchors = sorted({cluster.anchor for cluster in self.clusters})

        seen_sequences: set[tuple[int, ...]] = set()
        routes: list[Route] = []

        for anchor_a, anchor_b in itertools.combinations(anchors, 2):
            source = self.stop_nodes[anchor_a]
            target = self.stop_nodes[anchor_b]
            try:
                paths = nx.shortest_simple_paths(self.graph, source, target, weight="weight")
            except nx.NetworkXNoPath:
                continue

            for node_path in itertools.islice(paths, self.routes_per_pair):
                length = sum(
                    math.dist(node_path[i], node_path[i + 1])
                    for i in range(len(node_path) - 1)
                )
                stop_sequence = [
                    node_to_stop[node] for node in node_path if node in node_to_stop
                ]

                if len(stop_sequence) < self.min_stops:
                    self.rejected_too_short += 1
                    continue
                if length > self.max_length:
                    self.rejected_too_long += 1
                    continue

                key = tuple(stop_sequence)
                if key in seen_sequences:
                    self.rejected_duplicate += 1
                    continue

                seen_sequences.add(key)
                routes.append(Route(stops=stop_sequence, length=length))

        return routes
