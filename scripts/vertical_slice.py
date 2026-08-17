"""
scripts/vertical_slice.py

THROWAWAY vertical slice. The only deliverable is one number: seconds per
GA fitness evaluation. Every stage below is deliberately the crudest thing
that runs - no weighting, no validity checking, no real optimisation
algorithms - because the point is to measure evaluation cost before
committing to the real design. Do not reuse any of this code.

Run: python scripts/vertical_slice.py
"""

from __future__ import annotations

import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
import numpy as np

import config
from city.generators.population_generator import PopulationGenerator
from city.managers.map_manager import MapManager
from city.models.world import World
from city.roads.connector import connect_components
from city.roads.graph_builder import RoadGraphBuilder
from city.roads.loop_closer import close_loops
from city.roads.segment_grower import RoadNetworkGrower

# Crude vertical-slice knobs. Not config.py material: none of this survives
# past this throwaway script.
SLICE_STOP_COUNT = 40
SLICE_ROUTE_LENGTH = 8
SLICE_ROUTES_PER_SOLUTION = 6
SLICE_TIME_STEPS = 60
SLICE_POPULATION = 10
SLICE_GENERATIONS = 5

RNG_SEED = 0


def build_road_network() -> tuple[nx.Graph, float]:
    """Same pipeline as inspect_roads.py: grow, connect, close loops."""
    t0 = time.perf_counter()

    map_manager = MapManager()
    world = World(map_manager)
    world.generate()
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
    graph = RoadGraphBuilder(segments + connectors + loop_edges).build()

    elapsed = time.perf_counter() - t0
    return graph, elapsed


def pick_stops(graph: nx.Graph, rng: random.Random) -> list[tuple[float, float]]:
    """Stage 1 (crude): largest component only - the smaller one is
    genuinely unreachable, not just inconvenient - then every Nth node,
    sorted for a deterministic order since set iteration isn't stable.
    No weighting, no spacing: real stop selection comes later.
    """
    largest = max(nx.connected_components(graph), key=len)
    nodes = sorted(largest)
    step = max(len(nodes) // SLICE_STOP_COUNT, 1)
    return nodes[::step][:SLICE_STOP_COUNT]


def build_distance_matrix(graph: nx.Graph, stops: list[tuple[float, float]]) -> np.ndarray:
    """Stage 2: all-pairs shortest path over the road graph, computed
    ONCE. Every stop is in the same (largest) component, so no pair is
    unreachable."""
    n = len(stops)
    matrix = np.zeros((n, n), dtype=np.float64)
    for i, source in enumerate(stops):
        lengths = nx.single_source_dijkstra_path_length(graph, source, weight="weight")
        for j, target in enumerate(stops):
            matrix[i, j] = lengths[target]
    return matrix


def build_demand_matrix(n: int) -> np.ndarray:
    """Stage 3 (crude): uniform demand, no gravity model, no population."""
    return np.ones((n, n), dtype=np.float64) - np.eye(n, dtype=np.float64)


def random_route(n_stops: int, rng: random.Random) -> list[int]:
    """Stage 4 (crude): SLICE_ROUTE_LENGTH random stop indices, no
    validity checking (may repeat a stop, may not be geographically
    sensible - the real route generator comes later)."""
    return [rng.randrange(n_stops) for _ in range(SLICE_ROUTE_LENGTH)]


def random_solution(n_stops: int, rng: random.Random) -> list[list[int]]:
    return [random_route(n_stops, rng) for _ in range(SLICE_ROUTES_PER_SOLUTION)]


def fitness(
    solution: list[list[int]], distance_matrix: np.ndarray, demand_matrix: np.ndarray
) -> float:
    """Stage 5: the part being timed. The nested-loop shape (time steps x
    routes x stop pairs, each a matrix lookup) is what matters - the
    accumulated scalar is meaningless by construction."""
    total_time = 0.0
    total_passengers = 0.0
    for _time_step in range(SLICE_TIME_STEPS):
        for route in solution:
            for a, b in zip(route, route[1:]):
                total_time += distance_matrix[a, b]
                total_passengers += demand_matrix[a, b]
    return total_time - total_passengers


def mutate(solution: list[list[int]], n_stops: int, rng: random.Random) -> list[list[int]]:
    """Stage 6 (crude): replace one random stop in one random route. No
    crossover."""
    child = [route[:] for route in solution]
    route = rng.choice(child)
    route[rng.randrange(len(route))] = rng.randrange(n_stops)
    return child


def main() -> None:
    rng = random.Random(RNG_SEED)

    graph, world_time = build_road_network()
    print(f"world + road generation      : {world_time:.4f} s")

    stops = pick_stops(graph, rng)
    n_stops = len(stops)
    print(f"(stops selected: {n_stops})")

    t0 = time.perf_counter()
    distance_matrix = build_distance_matrix(graph, stops)
    distance_time = time.perf_counter() - t0
    print(f"distance matrix (once)       : {distance_time:.4f} s")

    demand_matrix = build_demand_matrix(n_stops)

    population = [random_solution(n_stops, rng) for _ in range(SLICE_POPULATION)]

    total_fitness_evals = 0
    total_fitness_time = 0.0
    ga_t0 = time.perf_counter()

    for _generation in range(SLICE_GENERATIONS):
        scored = []
        for solution in population:
            t0 = time.perf_counter()
            cost = fitness(solution, distance_matrix, demand_matrix)
            total_fitness_time += time.perf_counter() - t0
            total_fitness_evals += 1
            scored.append((cost, solution))

        scored.sort(key=lambda item: item[0])
        survivors = [solution for _cost, solution in scored[: SLICE_POPULATION // 2]]

        children = [
            mutate(rng.choice(survivors), n_stops, rng)
            for _ in range(SLICE_POPULATION - len(survivors))
        ]
        population = survivors + children

    ga_time = time.perf_counter() - ga_t0
    seconds_per_eval = total_fitness_time / total_fitness_evals

    print(f"total fitness evaluations    : {total_fitness_evals}")
    print(f"total time in fitness        : {total_fitness_time:.4f} s")
    print(f"SECONDS PER FITNESS EVAL     : {seconds_per_eval:.6f}")
    print(f"total GA time                 : {ga_time:.4f} s")

    for label, pop_size, generations in [
        ("population 50 x 100 generations", 50, 100),
        ("population 100 x 200 generations", 100, 200),
    ]:
        evals = pop_size * generations
        projected_seconds = evals * seconds_per_eval
        projected_hours = projected_seconds / 3600.0
        print(
            f"projected time for {label} ({evals:,} evals): "
            f"{projected_seconds:.2f} s  ({projected_hours:.4f} h)"
        )


if __name__ == "__main__":
    main()
