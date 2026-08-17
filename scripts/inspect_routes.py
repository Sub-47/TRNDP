"""
scripts/inspect_routes.py

Standalone diagnostic instrument for the candidate route pool: builds the
real world/road/stop/demand pipeline, clusters stops with DemandClusterer,
generates routes with RoutePool, and prints a diagnostic table. Asserts
nothing.

Run: python scripts/inspect_routes.py
"""

from __future__ import annotations

import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config
from city.demand.distance_matrix import DistanceMatrix
from city.demand.gravity_model import GravityModel
from city.demand.zone_map import ZoneMap
from city.generators.population_generator import PopulationGenerator
from city.managers.map_manager import MapManager
from city.models.world import World
from city.roads.connector import connect_components
from city.roads.graph_builder import RoadGraphBuilder
from city.roads.loop_closer import close_loops
from city.roads.segment_grower import RoadNetworkGrower
from city.roads.stop_selector import StopSelector
from city.routes.cluster import (
    DemandClusterer,
    aggregate_stop_demand,
    assign_zones_to_stops,
    build_stop_distance_matrix,
)
from city.routes.route_pool import RoutePool


def build_pipeline():
    """Regenerates the full pipeline up through demand, the same way
    inspect_roads.py and test_demand.py do."""
    map_manager = MapManager()
    world = World(map_manager)
    world.generate()

    population = world.population.data
    obstacle = world.obstacle.data
    world_size = population.shape[0]

    population_generator = PopulationGenerator(
        terrain_classes=world.terrain.data,
        obstacle_mask=obstacle,
        world_size=world_size,
        seed=config.SEED,
    )
    population_generator.run()
    starts = [(float(col), float(row)) for row, col in population_generator.centres]

    segments = RoadNetworkGrower(population, obstacle, starts).grow()
    graph = RoadGraphBuilder(segments).build()

    connectors = connect_components(graph, obstacle)
    graph = RoadGraphBuilder(segments + connectors).build()

    loop_edges = close_loops(graph, obstacle)
    graph = RoadGraphBuilder(segments + connectors + loop_edges).build()

    stops = StopSelector(graph, population).select()

    zones = ZoneMap.from_maps(world.population, world.obstacle)
    dist = DistanceMatrix.from_graph(zones, graph)
    demand = GravityModel.from_zones_and_distance(zones, dist)

    return graph, stops, zones, demand


def _distribution(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return (0.0, 0.0, 0.0)
    return (min(values), statistics.median(values), max(values))


EPS_SWEEP = (6.0, 8.0, 10.0, 12.0, 15.0)


def _coverage_pct(
    routes, zones: ZoneMap, demand: GravityModel, zone_assignment: np.ndarray
) -> float:
    covered_stops = {i for route in routes for i in route.stops}
    covered_zone_mask = np.array(
        [zone_assignment[z] in covered_stops for z in range(zones.num_zones)]
    )
    total_trips = float(demand.data.sum())
    if total_trips <= 0 or not covered_zone_mask.any():
        return 0.0
    covered_trips = float(demand.data[np.ix_(covered_zone_mask, covered_zone_mask)].sum())
    return 100.0 * covered_trips / total_trips


def run_eps_sweep(
    graph,
    stops,
    stop_distances: np.ndarray,
    stop_demand: np.ndarray,
    zones: ZoneMap,
    demand: GravityModel,
    zone_assignment: np.ndarray,
) -> None:
    """Diagnostic only - reports how cluster count, largest cluster size,
    route count, and demand coverage move across DBSCAN_EPS. Not a search
    for the "best" value; config.DBSCAN_EPS is left untouched."""
    print("-" * 60)
    print("DBSCAN_EPS sensitivity sweep (diagnostic, not tuning):")
    header = f"  {'eps':>6} | {'clusters':>8} | {'largest':>7} | {'routes':>7} | {'coverage':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for eps in EPS_SWEEP:
        clusterer = DemandClusterer(stops, stop_distances, stop_demand, eps=eps)
        clusters = clusterer.cluster()
        pool = RoutePool(graph, clusters, stops)
        routes = pool.generate()
        largest = max((len(c.members) for c in clusters), default=0)
        coverage = _coverage_pct(routes, zones, demand, zone_assignment)
        print(
            f"  {eps:>6.1f} | {len(clusters):>8} | {largest:>7} | "
            f"{len(routes):>7} | {coverage:>8.2f}%"
        )


def main() -> None:
    graph, stops, zones, demand = build_pipeline()

    print("=" * 60)
    print("ROUTE POOL DIAGNOSTIC")
    print("=" * 60)
    print(f"stops available                 : {len(stops)}")

    # --- Clustering ---------------------------------------------------
    t0 = time.perf_counter()
    stop_distances = build_stop_distance_matrix(graph, stops)
    stop_demand = aggregate_stop_demand(zones, demand, stops)
    zone_assignment = assign_zones_to_stops(zones, stops)

    clusterer = DemandClusterer(stops, stop_distances, stop_demand)
    clusters = clusterer.cluster()
    clustering_time = time.perf_counter() - t0

    print("-" * 60)
    print(f"clusters found                   : {len(clusters)}")
    print(f"noise points promoted            : {clusterer.noise_promoted}")
    print("-" * 60)
    print("clusters (member count, anchor stop index, total demand):")
    for i, cluster in enumerate(sorted(clusters, key=lambda c: -c.demand)):
        print(
            f"  cluster {i:>2}: {len(cluster.members):>2} members, "
            f"anchor=stop {cluster.anchor}, demand={cluster.demand:.2f}"
        )

    sizes = [len(c.members) for c in clusters]
    print("-" * 60)
    print("stops-per-cluster distribution:")
    print(f"  min / median / max members     : {_distribution(sizes)}")
    if sizes and max(sizes) >= 0.8 * len(stops):
        print("  WARNING: one cluster holds nearly all stops - DBSCAN_EPS is likely too large")
    if sizes and all(s == 1 for s in sizes):
        print("  WARNING: every stop is its own cluster - DBSCAN_EPS is likely too small")

    # --- Route generation ----------------------------------------------
    t0 = time.perf_counter()
    pool = RoutePool(graph, clusters, stops)
    routes = pool.generate()
    yens_time = time.perf_counter() - t0

    print("-" * 60)
    print(f"routes generated                 : {len(routes)}")
    print("rejections:")
    print(f"  too short (< {config.ROUTE_MIN_STOPS} stops)        : {pool.rejected_too_short}")
    print(f"  too long (> {config.ROUTE_MAX_LENGTH} cells)      : {pool.rejected_too_long}")
    print(f"  duplicate stop sequence         : {pool.rejected_duplicate}")

    stop_lengths = [len(r.stops) for r in routes]
    cell_lengths = [r.length for r in routes]
    print("-" * 60)
    print("route length distribution:")
    print(f"  in stops (min / median / max)   : {_distribution(stop_lengths)}")
    print(f"  in cells (min / median / max)   : {_distribution(cell_lengths)}")

    # --- Demand coverage --------------------------------------------------
    covered_stops = {i for route in routes for i in route.stops}
    covered_zone_mask = np.array(
        [zone_assignment[z] in covered_stops for z in range(zones.num_zones)]
    )
    total_trips = float(demand.data.sum())
    covered_trips = float(
        demand.data[np.ix_(covered_zone_mask, covered_zone_mask)].sum()
        if covered_zone_mask.any()
        else 0.0
    )
    coverage_pct = 100.0 * covered_trips / total_trips if total_trips > 0 else 0.0

    print("-" * 60)
    print("demand coverage (origin AND destination stops both in pool):")
    print(f"  stops appearing in the pool     : {len(covered_stops)} / {len(stops)}")
    print(f"  total trips                     : {total_trips:.2f}")
    print(f"  covered trips                   : {covered_trips:.2f}")
    print(f"  coverage                        : {coverage_pct:.2f}%")

    print("-" * 60)
    print(f"clustering time                  : {clustering_time * 1000:.2f} ms")
    print(f"Yen's generation time             : {yens_time * 1000:.2f} ms")

    run_eps_sweep(graph, stops, stop_distances, stop_demand, zones, demand, zone_assignment)
    print("=" * 60)


if __name__ == "__main__":
    main()
