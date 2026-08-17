"""
scripts/inspect_demand.py

Diagnostic for the demand model, like inspect_roads.py. Builds the world
and the final road network (grow + connect + close_loops) once, then
reports the headline evidence for both fixed defects:

  - how far graph distance diverges from Euclidean distance (Defect 1)
  - how close the total trip count stays across different ZONE_SIZE
    values, and that diurnal/row-sum properties hold (Defect 2)

Asserts nothing.

Run: python scripts/inspect_demand.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
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

SCALE_ZONE_SIZES = (4, 8, 16)


def build_road_graph(world: World) -> nx.Graph:
    """Same pipeline as inspect_roads.py: grow, connect, close loops."""
    population = world.population.data
    obstacle = world.obstacle.data

    population_generator = PopulationGenerator(
        terrain_classes=world.terrain.data,
        obstacle_mask=world.obstacle.data,
        world_size=population.shape[0],
        seed=config.SEED,
    )
    population_generator.run()
    starts = [(float(col), float(row)) for row, col in population_generator.centres]

    segments = RoadNetworkGrower(population, obstacle, starts).grow()
    graph = RoadGraphBuilder(segments).build()

    connectors = connect_components(graph, obstacle)
    graph = RoadGraphBuilder(segments + connectors).build()

    loop_edges = close_loops(graph, obstacle)
    return RoadGraphBuilder(segments + connectors + loop_edges).build()


def main() -> None:
    map_manager = MapManager()
    world = World(map_manager)
    world.generate()

    graph = build_road_graph(world)

    print("=" * 70)
    print("DEMAND MODEL DIAGNOSTIC")
    print("=" * 70)

    zones = ZoneMap.from_maps(world.population, world.obstacle, zone_size=config.ZONE_SIZE)
    euclidean = DistanceMatrix.from_zone_map(zones)

    t0 = time.perf_counter()
    graph_dist = DistanceMatrix.from_graph(zones, graph)
    graph_dist_time = time.perf_counter() - t0

    n = zones.num_zones
    unreachable = set(graph_dist.unreachable_zones)
    reachable = [i for i in range(n) if i not in unreachable]

    print(f"ZONE_SIZE                : {config.ZONE_SIZE}")
    print(f"zones total               : {n}")
    print(f"zones reachable           : {len(reachable)}")
    print(f"zones unreachable         : {len(unreachable)}  {sorted(unreachable)}")
    print(f"(graph distance matrix built in {graph_dist_time:.3f}s)")

    # --- Euclidean vs graph distance: the headline claim -----------------
    ratios = []
    worst = None
    for i in reachable:
        for j in reachable:
            if i == j:
                continue
            euclidean_d = float(euclidean.data[i, j])
            graph_d = float(graph_dist.data[i, j])
            if euclidean_d <= 0 or not np.isfinite(graph_d):
                continue
            ratio = graph_d / euclidean_d
            ratios.append(ratio)
            if worst is None or ratio > worst[0]:
                worst = (ratio, i, j, euclidean_d, graph_d)

    ratios_arr = np.array(ratios)
    print("-" * 70)
    print("EUCLIDEAN vs GRAPH DISTANCE (headline evidence for Defect 1)")
    print(f"  reachable pairs compared        : {len(ratios_arr)}")
    print(f"  median  graph/euclidean ratio    : {np.median(ratios_arr):.3f}")
    print(f"  90th pct graph/euclidean ratio   : {np.percentile(ratios_arr, 90):.3f}")
    if worst is not None:
        ratio, i, j, euclidean_d, graph_d = worst
        print(f"  worst ratio                      : {ratio:.3f}")
        print(f"    zone {i} -> zone {j}: euclidean={euclidean_d:.2f} cells, graph={graph_d:.2f} cells")

    # --- MORNING_PEAK trips + row-sum confirmation (Defect 2) ------------
    print("-" * 70)
    print("TRIP GENERATION (headline evidence for Defect 2)")
    morning = GravityModel.for_period(zones, graph_dist, "MORNING_PEAK")
    print(f"  total trips MORNING_PEAK         : {morning.data.sum():.1f}")

    population_flat = zones.population.flatten()
    factor = config.DIURNAL_FACTORS["MORNING_PEAK"]
    max_row_sum_error = 0.0
    checked = 0
    for i in range(n):
        if i in unreachable or population_flat[i] <= 0:
            continue
        expected = population_flat[i] * config.TRIPS_PER_PERSON * factor
        actual = morning.data[i].sum()
        max_row_sum_error = max(max_row_sum_error, abs(float(actual) - float(expected)))
        checked += 1
    print(
        f"  row-sum check (population*TRIPS_PER_PERSON*factor), "
        f"{checked} populated reachable zones: max error = {max_row_sum_error:.6f}"
    )

    # --- scale invariance --------------------------------------------------
    print("-" * 70)
    print("SCALE INVARIANCE (total trips should stay ~constant across ZONE_SIZE)")
    scale_totals = {}
    for zone_size in SCALE_ZONE_SIZES:
        z = ZoneMap.from_maps(world.population, world.obstacle, zone_size=zone_size)
        d = DistanceMatrix.from_graph(z, graph)
        g = GravityModel.from_zones_and_distance(z, d)
        total = float(g.data.sum())
        scale_totals[zone_size] = total
        print(f"  ZONE_SIZE={zone_size:>2} ({z.num_zones:>4} zones): total trips = {total:,.1f}")
    totals_list = list(scale_totals.values())
    spread = (max(totals_list) - min(totals_list)) / max(totals_list) * 100.0
    print(f"  spread across ZONE_SIZE values   : {spread:.2f}% of the largest total")

    # --- diurnal -------------------------------------------------------------
    print("-" * 70)
    print("DIURNAL VARIATION")
    period_totals = {}
    for period in config.DIURNAL_FACTORS:
        m = GravityModel.for_period(zones, graph_dist, period)
        period_totals[period] = float(m.data.sum())
        print(f"  {period:<14}: total trips = {period_totals[period]:,.1f}")

    actual_ratio = period_totals["MORNING_PEAK"] / period_totals["OFF_PEAK"]
    expected_ratio = config.DIURNAL_FACTORS["MORNING_PEAK"] / config.DIURNAL_FACTORS["OFF_PEAK"]
    print(
        f"  MORNING_PEAK / OFF_PEAK total    : {actual_ratio:.6f}  "
        f"(expected {expected_ratio:.6f})"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
