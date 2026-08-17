"""
scripts/inspect_ga.py

Standalone diagnostic instrument for NSGA2: builds the real
world/road/stop/demand/route-pool pipeline (same as inspect_sim.py),
runs the GA, and reports convergence evidence plus a comparison against
the greedy baseline route selection. Asserts nothing.

Run: python scripts/inspect_ga.py
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
from city.ga.nsga2 import NSGA2
from city.generators.population_generator import PopulationGenerator
from city.managers.map_manager import MapManager
from city.models.world import World
from city.roads.connector import connect_components
from city.roads.graph_builder import RoadGraphBuilder
from city.roads.loop_closer import close_loops
from city.roads.segment_grower import RoadNetworkGrower
from city.roads.stop_selector import StopSelector
from city.routes.cluster import DemandClusterer, aggregate_stop_demand, assign_zones_to_stops, build_stop_distance_matrix
from city.routes.route_pool import RoutePool
from city.sim.simulator import TransitSimulator

HYPERVOLUME_STRIDE = 10  # print every Nth generation


def build_pipeline():
    """Regenerates the full pipeline through the candidate route pool,
    the same way inspect_sim.py does."""
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

    stop_distances = build_stop_distance_matrix(graph, stops)
    stop_demand = aggregate_stop_demand(zones, demand, stops)
    clusters = DemandClusterer(stops, stop_distances, stop_demand).cluster()
    routes = RoutePool(graph, clusters, stops).generate()

    return stops, zones, demand, stop_distances, routes


def build_stop_od_matrix(zones: ZoneMap, demand: GravityModel, stops: list) -> np.ndarray:
    """Aggregates the zone-level O-D demand matrix onto stops (same as
    inspect_sim.py's helper of the same name)."""
    assignment = assign_zones_to_stops(zones, stops)
    n_stops = len(stops)
    flat_index = assignment[:, np.newaxis] * n_stops + assignment[np.newaxis, :]
    stop_od = np.bincount(flat_index.ravel(), weights=demand.data.ravel(), minlength=n_stops * n_stops)
    return stop_od.reshape(n_stops, n_stops)


def select_routes_greedily(routes: list, stop_od: np.ndarray, count: int) -> list:
    """Picks `count` routes ranked by the O-D demand each route could
    directly serve on its own (same method inspect_sim.py uses for its
    diagnostic route set) - the baseline the GA is compared against."""

    def score(route) -> float:
        idx = np.array(route.stops)
        return float(stop_od[np.ix_(idx, idx)].sum())

    ranked = sorted(routes, key=score, reverse=True)
    return ranked[:count]


def evaluate_route_set(routes: list, stop_distances: np.ndarray, stop_od: np.ndarray) -> tuple[float, float, float]:
    """Same three objectives NSGA2._evaluate computes, for a route set
    that didn't come from the GA (the greedy baseline)."""
    frequencies = [config.GA_FIXED_FREQUENCY] * len(routes)
    result = TransitSimulator(routes, stop_distances, stop_od, frequencies).run()
    user_cost = (
        result.total_wait_time
        + result.total_travel_time
        + result.total_transfers * config.GA_TRANSFER_PENALTY_MINUTES
    )
    return user_cost, result.total_bus_distance, result.unserved_passengers


def main() -> None:
    stops, zones, demand, stop_distances, pool_routes = build_pipeline()
    stop_od = build_stop_od_matrix(zones, demand, stops)

    print("=" * 60)
    print("GENETIC ALGORITHM (NSGA-II) DIAGNOSTIC")
    print("=" * 60)
    print(f"stops                            : {len(stops)}")
    print(f"routes in pool                   : {len(pool_routes)}")
    print(f"GA_POPULATION                    : {config.GA_POPULATION}")
    print(f"GA_GENERATIONS                   : {config.GA_GENERATIONS}")
    print(f"GA_ROUTES_PER_SOLUTION           : {config.GA_ROUTES_PER_SOLUTION}")
    print(f"GA_MUTATION_RATE                 : {config.GA_MUTATION_RATE}")
    print(f"GA_FIXED_FREQUENCY               : {config.GA_FIXED_FREQUENCY} buses/hour")
    print(f"GA_TRANSFER_PENALTY_MINUTES      : {config.GA_TRANSFER_PENALTY_MINUTES}")

    ga = NSGA2(pool_routes, stop_distances, stop_od, seed=config.SEED)
    result = ga.run()

    total_lookups = result.evaluations + result.cache_hits
    hit_rate = 100.0 * result.cache_hits / total_lookups if total_lookups else 0.0

    print("-" * 60)
    print("Run summary:")
    print(f"  evaluations (simulations run)  : {result.evaluations}")
    print(f"  cache hits                     : {result.cache_hits}")
    print(f"  cache hit rate                 : {hit_rate:.2f}%")
    print(f"  wall-clock time                : {result.wall_clock_seconds:.2f} s")
    print(f"  reference point (user/op/unserved): {tuple(round(v, 2) for v in result.reference_point)}")

    print("-" * 60)
    print(f"Hypervolume per generation (every {HYPERVOLUME_STRIDE}th, index 0 = initial population):")
    for gen, hv in enumerate(result.hypervolume_history):
        if gen % HYPERVOLUME_STRIDE == 0 or gen == len(result.hypervolume_history) - 1:
            print(f"  gen {gen:>4}: {hv:,.2f}")

    tail = result.hypervolume_history[-min(HYPERVOLUME_STRIDE, len(result.hypervolume_history)):]
    if len(tail) >= 2 and tail[0] > 0:
        pct_change = 100.0 * (tail[-1] - tail[0]) / tail[0]
        still_rising = pct_change > 1.0
        print(
            f"  change over last {len(tail) - 1} generations: {pct_change:+.2f}% - "
            f"{'STILL RISING (not converged, more generations may help)' if still_rising else 'flat/converged'}"
        )
    else:
        print("  not enough history to judge convergence")

    print("-" * 60)
    front = result.pareto_front
    print(f"Final Pareto front size          : {len(front)}")
    objectives = np.array([obj for _, obj in front])
    names = ["user cost (min)", "operator cost (cells)", "unserved passengers"]
    for i, name in enumerate(names):
        col = objectives[:, i]
        print(f"  {name:<24}: min={col.min():,.2f}  median={statistics.median(col):,.2f}  max={col.max():,.2f}")

    print("-" * 60)
    print("Extreme solutions on the final front:")
    best_user = min(front, key=lambda pair: pair[1][0])
    best_operator = min(front, key=lambda pair: pair[1][1])
    best_coverage = min(front, key=lambda pair: pair[1][2])
    for label, (chromosome, obj) in (
        ("best user cost", best_user),
        ("best operator cost", best_operator),
        ("best coverage (least unserved)", best_coverage),
    ):
        print(
            f"  {label:<32}: user={obj[0]:,.2f}  operator={obj[1]:,.2f}  "
            f"unserved={obj[2]:,.2f}  routes={len(chromosome)}"
        )

    print("-" * 60)
    print(f"Baseline: greedy {config.GA_ROUTES_PER_SOLUTION}-route selection (inspect_sim.py's method):")
    baseline_routes = select_routes_greedily(pool_routes, stop_od, config.GA_ROUTES_PER_SOLUTION)
    baseline_objectives = evaluate_route_set(baseline_routes, stop_distances, stop_od)
    print(
        f"  greedy   : user={baseline_objectives[0]:,.2f}  operator={baseline_objectives[1]:,.2f}  "
        f"unserved={baseline_objectives[2]:,.2f}"
    )

    beats_on = []
    for i, name in enumerate(("user cost", "operator cost", "unserved")):
        best_ga_value = objectives[:, i].min()
        if best_ga_value < baseline_objectives[i]:
            beats_on.append(name)
    dominates_baseline = any(
        all(obj[i] <= baseline_objectives[i] for i in range(3)) and any(obj[i] < baseline_objectives[i] for i in range(3))
        for _, obj in front
    )

    print("-" * 60)
    if dominates_baseline:
        print("RESULT: at least one GA solution dominates the greedy baseline on all three objectives.")
    elif beats_on:
        print(f"RESULT: no GA solution dominates greedy outright, but the GA front beats it on: {', '.join(beats_on)}.")
    else:
        print("RESULT: the GA did NOT beat the greedy baseline on any objective. Reporting as-is.")
    print("=" * 60)


if __name__ == "__main__":
    main()
