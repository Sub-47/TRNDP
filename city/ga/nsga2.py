"""
city/ga/nsga2.py

NSGA-II route selection: a chromosome is a set of GA_ROUTES_PER_SOLUTION
distinct indices into the candidate route pool, evaluated on three
minimised objectives (user cost, operator cost, unserved demand) via
TransitSimulator - the project's actual contribution, since the fitness
function is a behavioural simulation rather than a closed-form formula.

Chromosomes are frozensets of pool indices so they can be dict keys:
fitness is cached by chromosome, since a ~300ms simulation run is the
dominant cost and crossover/mutation frequently reproduce a parent or an
already-seen combination.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

import config
from city.routes.route_pool import Route
from city.sim.simulator import TransitSimulator

Chromosome = frozenset[int]


@dataclass
class GAResult:
    """Outcome of one NSGA2.run() call.

    pareto_front: (chromosome, objectives) pairs from the final rank-0
        front. hypervolume_history/best_per_generation: one entry per
        generation plus entry 0 for the initial random population, so
        index g means "after g generations". best_per_generation (min
        user/operator/unserved cost seen each generation) isn't in the
        spec's field list, but is exposed so elitism - no objective's
        best value ever worsens - can be checked from outside a run.
        reference_point is the worst-per-objective point from the
        initial population, fixed as the hypervolume reference throughout.
    """

    pareto_front: list[tuple[Chromosome, tuple[float, float, float]]]
    hypervolume_history: list[float]
    best_per_generation: list[tuple[float, float, float]]
    evaluations: int
    cache_hits: int
    wall_clock_seconds: float
    reference_point: tuple[float, float, float]


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """True if `a` dominates `b` under minimisation: no worse in every
    objective, and strictly better in at least one."""
    return bool(np.all(a <= b) and np.any(a < b))


def fast_non_dominated_sort(objectives) -> list[list[int]]:
    """Returns fronts (lists of indices into `objectives`), best first."""
    objectives = np.asarray(objectives)
    n = len(objectives)
    dominates_list: list[list[int]] = [[] for _ in range(n)]
    domination_count = np.zeros(n, dtype=int)
    fronts: list[list[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _dominates(objectives[p], objectives[q]):
                dominates_list[p].append(q)
            elif _dominates(objectives[q], objectives[p]):
                domination_count[p] += 1
        if domination_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominates_list[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    fronts.pop()  # trailing empty front left by the loop's exit check
    return fronts


def crowding_distance(objectives, front: list[int]) -> np.ndarray:
    """Crowding distance for each member of `front` (same order as
    `front`, not the full population). Boundary points on any objective
    get inf, so they always survive truncation."""
    objectives = np.asarray(objectives)
    n = len(front)
    distance = np.zeros(n)
    if n == 0:
        return distance

    front_objectives = objectives[front]
    num_objectives = front_objectives.shape[1]
    for obj_index in range(num_objectives):
        order = np.argsort(front_objectives[:, obj_index])
        distance[order[0]] = np.inf
        distance[order[-1]] = np.inf
        span = front_objectives[order[-1], obj_index] - front_objectives[order[0], obj_index]
        if span <= 0:
            continue
        for k in range(1, n - 1):
            prev_val = front_objectives[order[k - 1], obj_index]
            next_val = front_objectives[order[k + 1], obj_index]
            distance[order[k]] += (next_val - prev_val) / span
    return distance


def tournament_select(rank: np.ndarray, crowding: np.ndarray, rng: np.random.Generator) -> int:
    """Binary tournament over the whole population: lower rank wins,
    higher crowding distance breaks ties."""
    n = len(rank)
    i, j = rng.integers(0, n, size=2)
    if i == j:
        j = (j + 1) % n
    if rank[i] != rank[j]:
        return int(i) if rank[i] < rank[j] else int(j)
    return int(i) if crowding[i] >= crowding[j] else int(j)


def crossover(
    parent_a: Chromosome, parent_b: Chromosome, routes_per_solution: int, rng: np.random.Generator
) -> Chromosome:
    """Union of both parents, then sample routes_per_solution from it
    without replacement. Undersized unions (identical parents) are
    topped up by repair(), which always runs after this."""
    union = list(parent_a | parent_b)
    size = min(len(union), routes_per_solution)
    chosen = rng.choice(union, size=size, replace=False)
    return frozenset(int(x) for x in chosen)


def mutate(
    chromosome: Chromosome, pool_size: int, mutation_rate: float, rng: np.random.Generator
) -> Chromosome:
    """With probability mutation_rate, swaps one random member for a
    random pool route not already in the chromosome."""
    if rng.random() >= mutation_rate:
        return chromosome
    candidates = [i for i in range(pool_size) if i not in chromosome]
    if not candidates:
        return chromosome
    members = list(chromosome)
    remove = members[rng.integers(len(members))]
    add = candidates[rng.integers(len(candidates))]
    return frozenset((chromosome - {remove}) | {add})


def repair(
    chromosome: Chromosome, pool_size: int, routes_per_solution: int, rng: np.random.Generator
) -> Chromosome:
    """Enforces exactly routes_per_solution distinct indices. Required
    after every crossover/mutation, not just as a safety net: duplicate
    routes silently shrink the effective network size a chromosome
    represents."""
    members = list(chromosome)
    if len(members) > routes_per_solution:
        kept = rng.choice(members, size=routes_per_solution, replace=False)
        return frozenset(int(x) for x in kept)
    if len(members) < routes_per_solution:
        candidates = [i for i in range(pool_size) if i not in chromosome]
        missing = routes_per_solution - len(members)
        extra = rng.choice(candidates, size=missing, replace=False)
        return frozenset(members) | frozenset(int(x) for x in extra)
    return chromosome


class NSGA2:
    """NSGA-II search over route-pool subsets.

    Args:
        pool: Candidate routes (e.g. from RoutePool.generate()); a
            chromosome is a set of indices into this list.
        stop_distances: (n_stops, n_stops) graph distance matrix, passed
            straight through to TransitSimulator.
        demand: (n_stops, n_stops) stop-to-stop trip matrix for one
            period, passed straight through to TransitSimulator.
        seed: Drives every random choice in this run - initial
            population, tournament selection, crossover sampling,
            mutation - via a single np.random.Generator, so a run is
            fully reproducible.
    """

    def __init__(
        self,
        pool: list[Route],
        stop_distances: np.ndarray,
        demand: np.ndarray,
        seed: int = config.SEED,
    ) -> None:
        if len(pool) < config.GA_ROUTES_PER_SOLUTION:
            raise ValueError(
                f"pool has {len(pool)} routes, fewer than "
                f"GA_ROUTES_PER_SOLUTION={config.GA_ROUTES_PER_SOLUTION}"
            )

        self.pool = pool
        self.stop_distances = stop_distances
        self.demand = demand
        self.rng = np.random.default_rng(seed)
        self._hv_rng = np.random.default_rng(seed + 1)

        self.population_size = config.GA_POPULATION
        self.generations = config.GA_GENERATIONS
        self.routes_per_solution = config.GA_ROUTES_PER_SOLUTION
        self.mutation_rate = config.GA_MUTATION_RATE
        self.frequency = config.GA_FIXED_FREQUENCY
        self.transfer_penalty = config.GA_TRANSFER_PENALTY_MINUTES

        self._cache: dict[Chromosome, tuple[float, float, float]] = {}
        self._evaluations = 0
        self._cache_hits = 0
        self.reference_point: np.ndarray | None = None

    def _random_chromosome(self) -> Chromosome:
        idx = self.rng.choice(len(self.pool), size=self.routes_per_solution, replace=False)
        return frozenset(int(i) for i in idx)

    def _evaluate(self, chromosome: Chromosome) -> tuple[float, float, float]:
        if chromosome in self._cache:
            self._cache_hits += 1
            return self._cache[chromosome]

        routes = [self.pool[i] for i in chromosome]
        frequencies = [self.frequency] * len(routes)
        result = TransitSimulator(routes, self.stop_distances, self.demand, frequencies).run()
        objectives = (
            result.total_wait_time
            + result.total_travel_time
            + result.total_transfers * self.transfer_penalty,
            result.total_bus_distance,
            result.unserved_passengers,
        )
        self._cache[chromosome] = objectives
        self._evaluations += 1
        return objectives

    def _evaluate_all(self, chromosomes: list[Chromosome]) -> np.ndarray:
        return np.array([self._evaluate(c) for c in chromosomes])

    def _hypervolume(self, front_objectives: np.ndarray, n_samples: int = 10_000) -> float:
        """Monte Carlo estimate of the 3D hypervolume dominated by
        `front_objectives` relative to self.reference_point: fraction of
        points sampled uniformly in the box [0, reference_point] that
        some front point dominates, times the box volume. Simple and
        exact in expectation; chosen over an exact slicing algorithm
        since the spec allows an approximation and this is a few lines
        instead of a recursive dimension-sweep."""
        ref = self.reference_point
        if front_objectives.size == 0 or ref is None:
            return 0.0
        box_volume = float(np.prod(ref))
        if box_volume <= 0:
            return 0.0
        samples = self._hv_rng.uniform(low=0.0, high=ref, size=(n_samples, len(ref)))
        dominated = np.any(
            np.all(front_objectives[:, np.newaxis, :] <= samples[np.newaxis, :, :], axis=2),
            axis=0,
        )
        return box_volume * float(dominated.mean())

    def _select_next_generation(
        self, population: list[Chromosome], objectives: np.ndarray
    ) -> tuple[list[Chromosome], np.ndarray]:
        """Elitist truncation: whole fronts are kept in rank order until
        the next one would overflow the population, then that front is
        filled by descending crowding distance. Parents and offspring
        were already merged by the caller, so a non-dominated solution
        can only be dropped by losing a crowding-distance tie-break,
        never by generational replacement alone."""
        fronts = fast_non_dominated_sort(objectives)
        new_population: list[Chromosome] = []
        new_indices: list[int] = []
        for front in fronts:
            if len(new_population) + len(front) <= self.population_size:
                new_population.extend(population[i] for i in front)
                new_indices.extend(front)
            else:
                remaining = self.population_size - len(new_population)
                cd = crowding_distance(objectives, front)
                order = np.argsort(-cd)
                chosen = [front[k] for k in order[:remaining]]
                new_population.extend(population[i] for i in chosen)
                new_indices.extend(chosen)
                break
        return new_population, objectives[new_indices]

    def run(self) -> GAResult:
        start = time.perf_counter()

        population = [self._random_chromosome() for _ in range(self.population_size)]
        objectives = self._evaluate_all(population)
        self.reference_point = objectives.max(axis=0)

        hypervolume_history = [self._hypervolume(objectives[fast_non_dominated_sort(objectives)[0]])]
        best_per_generation = [tuple(objectives.min(axis=0))]

        for _ in range(self.generations):
            fronts = fast_non_dominated_sort(objectives)
            rank = np.empty(len(population), dtype=int)
            crowding = np.empty(len(population), dtype=float)
            for front_rank, front in enumerate(fronts):
                for i in front:
                    rank[i] = front_rank
                cd = crowding_distance(objectives, front)
                for position, i in enumerate(front):
                    crowding[i] = cd[position]

            offspring: list[Chromosome] = []
            while len(offspring) < self.population_size:
                a = tournament_select(rank, crowding, self.rng)
                b = tournament_select(rank, crowding, self.rng)
                child = crossover(population[a], population[b], self.routes_per_solution, self.rng)
                child = mutate(child, len(self.pool), self.mutation_rate, self.rng)
                child = repair(child, len(self.pool), self.routes_per_solution, self.rng)
                offspring.append(child)

            offspring_objectives = self._evaluate_all(offspring)
            merged_population = population + offspring
            merged_objectives = np.vstack([objectives, offspring_objectives])

            population, objectives = self._select_next_generation(merged_population, merged_objectives)

            hypervolume_history.append(
                self._hypervolume(objectives[fast_non_dominated_sort(objectives)[0]])
            )
            best_per_generation.append(tuple(objectives.min(axis=0)))

        final_front = fast_non_dominated_sort(objectives)[0]
        pareto_front = [(population[i], tuple(objectives[i])) for i in final_front]

        return GAResult(
            pareto_front=pareto_front,
            hypervolume_history=hypervolume_history,
            best_per_generation=best_per_generation,
            evaluations=self._evaluations,
            cache_hits=self._cache_hits,
            wall_clock_seconds=time.perf_counter() - start,
            reference_point=tuple(self.reference_point),
        )
