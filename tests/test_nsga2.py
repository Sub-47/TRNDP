"""
tests/test_nsga2.py

Exercises the NSGA-II building blocks against hand-worked examples (sort,
crowding distance), the chromosome-repair invariant, and a tiny end-to-end
run against a hand-built route pool (config.GA_* monkeypatched down to a
size that runs in milliseconds). Elitism (test 5) is the one that matters
most - if a non-dominated solution can be lost between generations, the
run silently degrades without ever raising an error.
"""

from __future__ import annotations

import numpy as np
import pytest

import config
from city.ga.nsga2 import (
    NSGA2,
    crossover,
    crowding_distance,
    fast_non_dominated_sort,
    mutate,
    repair,
    tournament_select,
)
from city.routes.route_pool import Route


def test_fast_non_dominated_sort_known_front():
    # (1,5) and (2,3) and (4,1) are mutually non-dominated (a classic
    # trade-off curve); (3,4) is dominated only by (2,3); (5,5) is
    # dominated by everything, including (3,4).
    objectives = np.array(
        [
            [1.0, 5.0],  # 0
            [2.0, 3.0],  # 1
            [4.0, 1.0],  # 2
            [3.0, 4.0],  # 3
            [5.0, 5.0],  # 4
        ]
    )

    fronts = fast_non_dominated_sort(objectives)

    assert set(fronts[0]) == {0, 1, 2}
    assert set(fronts[1]) == {3}
    assert set(fronts[2]) == {4}


def test_crowding_distance_boundaries_inf_and_dense_scores_lower():
    # A Pareto curve; index 2 is tightly bracketed by its neighbours in
    # both dimensions, index 3 is far more isolated.
    objectives = np.array(
        [
            [1.0, 9.0],  # 0 - boundary
            [2.0, 5.0],  # 1
            [3.0, 4.0],  # 2 - dense neighbourhood
            [4.0, 3.0],  # 3 - sparse neighbourhood
            [9.0, 1.0],  # 4 - boundary
        ]
    )
    front = [0, 1, 2, 3, 4]

    distance = crowding_distance(objectives, front)

    assert distance[0] == np.inf
    assert distance[4] == np.inf
    assert np.isfinite(distance[1])
    assert np.isfinite(distance[2])
    assert np.isfinite(distance[3])
    assert distance[2] < distance[3]


def test_repair_enforces_exact_distinct_count():
    rng = np.random.default_rng(0)
    pool_size, target = 10, 5

    # Undersized (crossover of near-identical parents can yield this).
    small = frozenset({1, 2, 3})
    fixed = repair(small, pool_size, target, rng)
    assert len(fixed) == target
    assert small <= fixed

    # Already correct.
    exact = frozenset({0, 1, 2, 3, 4})
    assert repair(exact, pool_size, target, rng) == exact

    # Oversized (shouldn't arise from crossover as implemented, but the
    # repair step must handle it per the spec regardless).
    large = frozenset(range(8))
    fixed_large = repair(large, pool_size, target, rng)
    assert len(fixed_large) == target
    assert fixed_large <= large


def test_crossover_then_mutate_then_repair_always_valid():
    rng = np.random.default_rng(1)
    pool_size, target = 12, 5
    parent_a = frozenset({0, 1, 2, 3, 4})
    parent_b = frozenset({3, 4, 5, 6, 7})

    for _ in range(200):
        child = crossover(parent_a, parent_b, target, rng)
        child = mutate(child, pool_size, mutation_rate=1.0, rng=rng)  # always mutate
        child = repair(child, pool_size, target, rng)
        assert len(child) == target
        assert len(set(child)) == target
        assert all(0 <= i < pool_size for i in child)


def _linear_stops(n: int, spacing: float) -> np.ndarray:
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distances[i, j] = abs(i - j) * spacing
    return distances


def _tiny_pool(n_stops: int = 8, n_routes: int = 10) -> list[Route]:
    """Overlapping 3-stop routes over a small linear stop network, so
    chromosomes actually differ in which O-D pairs they cover."""
    routes = []
    for k in range(n_routes):
        start = k % (n_stops - 2)
        stops = [start, start + 1, start + 2]
        routes.append(Route(stops=stops, length=2 * 5.0))
    return routes


def _tiny_demand(n_stops: int = 8) -> np.ndarray:
    demand = np.zeros((n_stops, n_stops))
    rng = np.random.default_rng(42)
    for i in range(n_stops):
        for j in range(n_stops):
            if i != j:
                demand[i, j] = rng.uniform(0, 20)
    return demand


@pytest.fixture
def tiny_ga_config(monkeypatch):
    """Shrinks the GA to a handful of individuals/generations/routes,
    and the simulation each evaluation runs to a short window, so a
    full run takes a fraction of a second rather than tens of them."""
    monkeypatch.setattr(config, "GA_POPULATION", 6)
    monkeypatch.setattr(config, "GA_GENERATIONS", 3)
    monkeypatch.setattr(config, "GA_ROUTES_PER_SOLUTION", 3)
    monkeypatch.setattr(config, "GA_MUTATION_RATE", 0.3)
    monkeypatch.setattr(config, "GA_FIXED_FREQUENCY", 6.0)
    monkeypatch.setattr(config, "SIM_DURATION_MINUTES", 20.0)


def test_determinism(tiny_ga_config):
    pool = _tiny_pool()
    stop_distances = _linear_stops(8, 5.0)
    demand = _tiny_demand()

    result_a = NSGA2(pool, stop_distances, demand, seed=7).run()
    result_b = NSGA2(pool, stop_distances, demand, seed=7).run()

    front_a = sorted(result_a.pareto_front, key=lambda pair: sorted(pair[0]))
    front_b = sorted(result_b.pareto_front, key=lambda pair: sorted(pair[0]))
    assert front_a == front_b


def test_elitism_best_objectives_never_worsen(tiny_ga_config):
    pool = _tiny_pool()
    stop_distances = _linear_stops(8, 5.0)
    demand = _tiny_demand()

    result = NSGA2(pool, stop_distances, demand, seed=3).run()

    best = np.array(result.best_per_generation)
    # Each objective's running-best must be non-increasing generation
    # over generation (all three objectives are minimised).
    for obj_index in range(3):
        deltas = np.diff(best[:, obj_index])
        assert np.all(deltas <= 1e-9), (
            f"objective {obj_index} worsened at some generation: {best[:, obj_index]}"
        )


def test_cache_hit_avoids_second_simulation_run(tiny_ga_config):
    pool = _tiny_pool()
    stop_distances = _linear_stops(8, 5.0)
    demand = _tiny_demand()

    ga = NSGA2(pool, stop_distances, demand, seed=5)
    chromosome = frozenset({0, 1, 2})

    first = ga._evaluate(chromosome)
    second = ga._evaluate(chromosome)

    assert first == second
    assert ga._evaluations == 1
    assert ga._cache_hits == 1


def test_tiny_end_to_end_run_completes_with_nonempty_front(tiny_ga_config):
    pool = _tiny_pool()
    stop_distances = _linear_stops(8, 5.0)
    demand = _tiny_demand()

    result = NSGA2(pool, stop_distances, demand, seed=11).run()

    assert len(result.pareto_front) > 0
    for chromosome, objectives in result.pareto_front:
        assert len(chromosome) == config.GA_ROUTES_PER_SOLUTION
        assert len(set(chromosome)) == config.GA_ROUTES_PER_SOLUTION
        assert all(np.isfinite(objectives))
        assert all(v >= 0.0 for v in objectives)
    assert len(result.hypervolume_history) == config.GA_GENERATIONS + 1
    assert result.evaluations > 0


class _FixedPairRNG:
    """Stub exposing only the `.integers` call tournament_select makes,
    so the two competitors can be forced deterministically instead of
    relying on a real RNG to eventually draw the pair under test."""

    def __init__(self, pair: tuple[int, int]) -> None:
        self._pair = pair

    def integers(self, low: int, high: int, size: int = 2):
        return np.array(self._pair)


def test_tournament_select_lower_rank_wins_despite_lower_crowding():
    rank = np.array([0, 2])
    crowding = np.array([0.0, 100.0])

    winner = tournament_select(rank, crowding, _FixedPairRNG((0, 1)))

    assert winner == 0


def test_tournament_select_ties_broken_by_higher_crowding():
    rank = np.array([1, 1])
    crowding = np.array([0.5, 5.0])

    winner = tournament_select(rank, crowding, _FixedPairRNG((0, 1)))

    assert winner == 1
