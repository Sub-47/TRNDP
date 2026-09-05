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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config
from city.ga.nsga2 import NSGA2
from city.sim.simulator import TransitSimulator

HYPERVOLUME_STRIDE = 10  # print every Nth generation


from city.pipeline import build_pipeline as complete_pipeline, build_stop_od_matrix
from city.experiments import choose_greedy
from city.objectives import objective_values


def build_pipeline():
    p=complete_pipeline()
    return p.stops,p.zones,p.demand,p.stop_distances,p.routes


def select_routes_greedily(routes: list, stop_od: np.ndarray, count: int) -> list:
    return [routes[i] for i in sorted(choose_greedy(routes,stop_od,count))]


def evaluate_route_set(routes: list, stop_distances: np.ndarray, stop_od: np.ndarray) -> tuple[float, float, float]:
    """Same three objectives NSGA2._evaluate computes, for a route set
    that didn't come from the GA (the greedy baseline)."""
    frequencies = [config.GA_FIXED_FREQUENCY] * len(routes)
    result = TransitSimulator(routes, stop_distances, stop_od, frequencies).run()
    return objective_values(result)


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
    print(
        f"Archive hypervolume per generation (fraction of the normalised unit "
        f"cube, every {HYPERVOLUME_STRIDE}th, index 0 = initial population):"
    )
    for gen, hv in enumerate(result.hypervolume_history):
        if gen % HYPERVOLUME_STRIDE == 0 or gen == len(result.hypervolume_history) - 1:
            print(f"  gen {gen:>4}: {hv:.6f}")

    tail = result.hypervolume_history[-min(HYPERVOLUME_STRIDE, len(result.hypervolume_history)):]
    if len(tail) >= 2 and tail[0] > 0:
        pct_change = 100.0 * (tail[-1] - tail[0]) / tail[0]
        still_rising = pct_change > 1.0
        print(
            f"  change over last {len(tail) - 1} generations: {pct_change:+.2f}% - "
            f"{'STILL RISING (not converged, more generations may help)' if still_rising else 'plateau or decline; convergence not established'}"
        )
    else:
        print("  not enough history to judge convergence")

    print("-" * 60)
    front = result.pareto_front
    print(f"Cumulative nondominated archive size          : {len(front)}")
    objectives = np.array([obj for _, obj in front])
    names = ["user cost (min)", "operator cost (cells)", "incomplete transit passengers"]
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
        ("highest completion (least incomplete)", best_coverage),
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
