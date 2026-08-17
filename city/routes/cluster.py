"""
city/routes/cluster.py

Groups the selected bus stops into clusters that route_pool.py draws
candidate routes between. Clustering runs on graph distance (not
coordinates) so that stops on either side of an obstacle never merge
just because they look close in a straight line, and is weighted by each
stop's trip generation via DBSCAN's `sample_weight` - a high-demand stop
can single-handedly qualify as a core point, which is what makes the
clustering demand-sensitive rather than merely spatial.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import networkx as nx
from sklearn.cluster import DBSCAN

import config
from city.demand.gravity_model import GravityModel
from city.demand.zone_map import ZoneMap

Point = tuple[float, float]

logger = logging.getLogger(__name__)


@dataclass
class Cluster:
    """One DBSCAN cluster of stops.

    Attributes:
        members: Indices into the `stops` list of every stop in this
            cluster (including the anchor).
        anchor: Index of the highest-demand stop in this cluster - route
            generation runs between anchors, not centroids, since a
            centroid may not be a stop at all.
        demand: Sum of `stop_demand` over every member.
    """

    members: list[int]
    anchor: int
    demand: float


def build_stop_distance_matrix(graph: nx.Graph, stops: list[Point]) -> np.ndarray:
    """All-pairs shortest-path distance between stops, over the road graph.

    One Dijkstra run per stop, never per pair - the same approach
    distance_matrix.py uses for zone access nodes.

    Args:
        graph: Routing graph built by RoadGraphBuilder; every stop must
            be one of its nodes.
        stops: Stop coordinates, in the order that defines stop indices
            everywhere else in this package.

    Returns:
        (len(stops), len(stops)) float64 array. DBSCAN's precomputed
        metric requires a real distance matrix, so entries are always
        finite: StopSelector only selects stops from the graph's largest
        connected component, so every stop can reach every other.
    """
    n = len(stops)
    data = np.full((n, n), np.inf, dtype=np.float64)
    for i, stop in enumerate(stops):
        lengths = nx.single_source_dijkstra_path_length(graph, stop, weight="weight")
        for j, other in enumerate(stops):
            data[i, j] = lengths.get(other, np.inf)
    np.fill_diagonal(data, 0.0)
    return data


def assign_zones_to_stops(zones: ZoneMap, stops: list[Point]) -> np.ndarray:
    """Assigns each zone to its nearest stop by straight-line (walking)
    distance - the zone's access link, same convention
    DistanceMatrix.from_graph uses for zone-to-road-node access.

    Returns:
        int array of length `zones.num_zones`, the index into `stops` of
        each zone's nearest stop.
    """
    _, centers_row, centers_col = zones.flatten()
    assignment = np.zeros(zones.num_zones, dtype=np.int64)
    for i in range(zones.num_zones):
        centroid = (float(centers_col[i]), float(centers_row[i]))
        assignment[i] = min(
            range(len(stops)), key=lambda s: math.dist(stops[s], centroid)
        )
    return assignment


def aggregate_stop_demand(
    zones: ZoneMap, demand: GravityModel, stops: list[Point]
) -> np.ndarray:
    """Aggregates the zone-level demand matrix onto stops.

    Each zone's total trips (as both origin and destination) are added
    to its nearest stop's total.

    Returns:
        float64 array of length len(stops).
    """
    assignment = assign_zones_to_stops(zones, stops)
    trips_per_zone = demand.data.sum(axis=1) + demand.data.sum(axis=0)

    stop_demand = np.zeros(len(stops), dtype=np.float64)
    for zone_index, stop_index in enumerate(assignment):
        stop_demand[stop_index] += trips_per_zone[zone_index]
    return stop_demand


class DemandClusterer:
    """Demand-weighted DBSCAN clustering of stops over graph distance.

    Args:
        stops: Stop coordinates; defines stop indices used everywhere.
        stop_distances: (len(stops), len(stops)) precomputed graph
            distance matrix, e.g. from `build_stop_distance_matrix`.
        stop_demand: Total trips generated at each stop, e.g. from
            `aggregate_stop_demand`. Passed to DBSCAN as `sample_weight`.
        eps: Max graph distance between stops in the same cluster.
        min_samples: DBSCAN's min_samples, in units of total sample
            weight within eps.

    Attributes:
        noise_promoted: Set by `cluster()` to the number of DBSCAN noise
            points (label -1) that were promoted to their own
            single-stop cluster.
    """

    def __init__(
        self,
        stops: list[Point],
        stop_distances: np.ndarray,
        stop_demand: np.ndarray,
        eps: float = config.DBSCAN_EPS,
        min_samples: int = config.DBSCAN_MIN_SAMPLES,
    ) -> None:
        if stop_distances.shape != (len(stops), len(stops)):
            raise ValueError(
                f"stop_distances must have shape {(len(stops), len(stops))}, "
                f"got {stop_distances.shape}"
            )
        if len(stop_demand) != len(stops):
            raise ValueError(
                f"stop_demand must have length {len(stops)}, got {len(stop_demand)}"
            )

        self.stops = stops
        self.stop_distances = stop_distances
        self.stop_demand = stop_demand
        self.eps = eps
        self.min_samples = min_samples
        self.noise_promoted = 0

    def cluster(self) -> list[Cluster]:
        """Runs demand-weighted DBSCAN and returns one Cluster per label,
        with every noise point promoted to its own single-stop cluster -
        a high-demand outlier still deserves service, not silent
        discard."""
        self.noise_promoted = 0
        n = len(self.stops)
        if n == 0:
            return []

        model = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="precomputed")
        labels = model.fit_predict(self.stop_distances, sample_weight=self.stop_demand)

        groups: dict[int, list[int]] = {}
        next_label = int(labels.max()) + 1 if labels.size and labels.max() >= 0 else 0
        for index, label in enumerate(labels):
            label = int(label)
            if label == -1:
                groups[next_label] = [index]
                next_label += 1
                self.noise_promoted += 1
            else:
                groups.setdefault(label, []).append(index)

        clusters = []
        for members in groups.values():
            members = sorted(members)
            anchor = max(members, key=lambda i: self.stop_demand[i])
            total_demand = float(sum(self.stop_demand[i] for i in members))
            clusters.append(Cluster(members=members, anchor=anchor, demand=total_demand))

        if self.noise_promoted:
            logger.info(
                "Promoted %d noise point(s) to single-stop clusters (out of %d stops)",
                self.noise_promoted,
                n,
            )

        return clusters
