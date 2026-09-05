"""
city/ga/nsga2.py

NSGA-II route selection: a chromosome is a set of GA_ROUTES_PER_SOLUTION
distinct indices into the candidate route pool, evaluated on three
minimised objectives (user cost, operator cost, unserved demand) via
TransitSimulator - the project's actual contribution, since the fitness
function is a behavioural simulation rather than a closed-form formula.

Chromosomes are frozensets of pool indices so they can be dict keys:
fitness is cached by chromosome, since simulation is the dominant cost and crossover/mutation frequently reproduce a parent or an
already-seen combination.
"""

from __future__ import annotations

import time
import math
import itertools
from dataclasses import dataclass, field

import numpy as np

import config
from city.routes.route_pool import Route
from city.sim.simulator import TransitSimulator
from city.objectives import objective_values, reference_point

Chromosome = frozenset[int]


@dataclass
class GAResult:
    """Archive and population evidence from one run.

    pareto_front contains unique nondominated chromosomes from every evaluated
    candidate, including offspring lost during population truncation.
    hypervolume_history is archival; population_hypervolume_history is separate.
    best_per_generation is the archive's running objective minima, not a claim
    that current-population minima always survive. The positive reference is
    fixed from instance bounds or explicitly provided by the experiment runner.
    """

    pareto_front: list[tuple[Chromosome, tuple[float, float, float]]]
    hypervolume_history: list[float]
    best_per_generation: list[tuple[float, float, float]]
    evaluations: int
    cache_hits: int
    wall_clock_seconds: float
    reference_point: tuple[float, float, float]
    population_hypervolume_history: list[float] = field(default_factory=list)
    population_front: list = field(default_factory=list)
    evaluation_history: list[int] = field(default_factory=list)


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

    if n == 0:
        return []
    if objectives.ndim != 2 or not np.all(np.isfinite(objectives)):
        raise ValueError("objectives must be a finite matrix")
    dominates = np.all(objectives[:, None, :] <= objectives[None, :, :], axis=2) & np.any(
        objectives[:, None, :] < objectives[None, :, :], axis=2)
    domination_count = dominates.sum(axis=0)
    dominates_list = [np.flatnonzero(row).tolist() for row in dominates]
    fronts[0] = np.flatnonzero(domination_count == 0).tolist()

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
    get inf on nonconstant objectives to prioritize them during truncation."""
    objectives = np.asarray(objectives)
    n = len(front)
    distance = np.zeros(n)
    if n == 0:
        return distance

    front_objectives = objectives[front]
    num_objectives = front_objectives.shape[1]
    for obj_index in range(num_objectives):
        order = np.argsort(front_objectives[:, obj_index])
        span = front_objectives[order[-1], obj_index] - front_objectives[order[0], obj_index]
        if span <= 0:
            continue
        distance[order[0]] = np.inf
        distance[order[-1]] = np.inf
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
    union = sorted(parent_a | parent_b)
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
    members = sorted(chromosome)
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
    members = sorted(chromosome)
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
        seed: int | None = None,
        reference: np.ndarray | None = None,
        evaluator=None,
    ) -> None:
        if len(pool) < config.GA_ROUTES_PER_SOLUTION:
            raise ValueError(
                f"pool has {len(pool)} routes, fewer than "
                f"GA_ROUTES_PER_SOLUTION={config.GA_ROUTES_PER_SOLUTION}"
            )

        self.pool = pool
        self.stop_distances = stop_distances
        self.demand = demand
        self.rng = np.random.default_rng(config.SEED if seed is None else seed)

        if not isinstance(config.GA_ROUTES_PER_SOLUTION,int) or config.GA_ROUTES_PER_SOLUTION < 1:
            raise ValueError("GA_ROUTES_PER_SOLUTION must be positive")
        if not isinstance(config.GA_POPULATION,int) or config.GA_POPULATION < 2:
            raise ValueError("GA_POPULATION must be at least two")
        if not isinstance(config.GA_GENERATIONS,int) or config.GA_GENERATIONS < 0:
            raise ValueError("GA_GENERATIONS must be nonnegative")
        if not 0 <= config.GA_MUTATION_RATE <= 1:
            raise ValueError("mutation rate must be in [0,1]")
        self.population_size = min(config.GA_POPULATION, math.comb(len(pool),config.GA_ROUTES_PER_SOLUTION))
        self.generations = config.GA_GENERATIONS
        self.routes_per_solution = config.GA_ROUTES_PER_SOLUTION
        self.mutation_rate = config.GA_MUTATION_RATE
        self.frequency = config.GA_FIXED_FREQUENCY
        self.transfer_penalty = config.GA_TRANSFER_PENALTY_MINUTES

        self._cache: dict[Chromosome, tuple[float, float, float]] = {}
        self._evaluations = 0
        self._cache_hits = 0
        self.reference_point = np.asarray(reference if reference is not None else
                                          reference_point(demand,self.routes_per_solution,self.frequency),dtype=float)
        if self.reference_point.shape != (3,) or not np.all(np.isfinite(self.reference_point)) or np.any(self.reference_point <= 0):
            raise ValueError("reference must have three finite positive coordinates")
        self._evaluator = evaluator

    def _random_chromosome(self) -> Chromosome:
        idx = self.rng.choice(len(self.pool), size=self.routes_per_solution, replace=False)
        return frozenset(int(i) for i in idx)

    def _evaluate(self, chromosome: Chromosome) -> tuple[float, float, float]:
        if chromosome in self._cache:
            self._cache_hits += 1
            return self._cache[chromosome]

        if len(chromosome) != self.routes_per_solution or any(i < 0 or i >= len(self.pool) for i in chromosome):
            raise ValueError("invalid chromosome")
        if self._evaluator is not None:
            objectives = tuple(self._evaluator(chromosome))
        else:
            routes = [self.pool[i] for i in sorted(chromosome)]
            result = TransitSimulator(routes, self.stop_distances, self.demand,
                                      [self.frequency]*len(routes)).run()
            objectives = objective_values(result,self.transfer_penalty)
        if len(objectives) != 3 or not np.all(np.isfinite(objectives)) or min(objectives) < 0:
            raise ValueError("fitness must contain three finite nonnegative objectives")
        self._cache[chromosome] = objectives
        self._evaluations += 1
        return objectives

    def _evaluate_all(self, chromosomes: list[Chromosome]) -> np.ndarray:
        return np.array([self._evaluate(c) for c in chromosomes])

    def _hypervolume(self, front_objectives: np.ndarray) -> float:
        """Exact 3D hypervolume of `front_objectives` dominated relative
        to self.reference_point, as a fraction of the unit cube.
        Objectives are first normalised by the reference point (removing
        the ~10^6/10^4/10^3 scale disparity between user cost, operator
        cost and unserved demand), then swept by the standard slicing
        method: sort by the third normalised objective, and at each
        slice accumulate the 2D area dominated in the first two
        objectives so far. Exact rather than sampled. Archive fronts may exceed population size;
        this slicing implementation targets the project's prototype scale."""
        front_objectives = np.asarray(front_objectives, dtype=float)
        ref = self.reference_point
        if front_objectives.size == 0 or ref is None or np.any(ref <= 0):
            return 0.0
        if not np.all(np.isfinite(front_objectives)) or np.any(front_objectives < 0):
            raise ValueError("hypervolume expects finite nonnegative objectives")
        normalised = front_objectives[np.all(front_objectives < ref,axis=1)] / ref
        if len(normalised) == 0:
            return 0.0

        def area_2d(points: list[tuple[float, float]]) -> float:
            """Area of the union of [x, 1] x [y, 1] boxes over `points`."""
            skyline: list[tuple[float, float]] = []
            min_y = np.inf
            for x, y in sorted(points):
                if y < min_y:
                    skyline.append((x, y))
                    min_y = y
            area = 0.0
            for i, (x, y) in enumerate(skyline):
                next_x = skyline[i + 1][0] if i + 1 < len(skyline) else 1.0
                area += (next_x - x) * (1.0 - y)
            return area

        order = np.argsort(normalised[:, 2])
        sorted_points = normalised[order]

        volume = 0.0
        active_xy: list[tuple[float, float]] = []
        for i, (x, y, z) in enumerate(sorted_points):
            active_xy.append((float(x), float(y)))
            next_z = sorted_points[i + 1, 2] if i + 1 < len(sorted_points) else 1.0
            width = next_z - z
            if width > 0:
                volume += width * area_2d(active_xy)
        return volume

    def _select_next_generation(
        self, population: list[Chromosome], objectives: np.ndarray
    ) -> tuple[list[Chromosome], np.ndarray]:
        """Elitist truncation: whole fronts are kept in rank order until
        the next one would overflow the population, then that front is
        filled by descending crowding distance. Parents and offspring
        were already merged by the caller, so a non-dominated solution
        can only be dropped by losing a crowding-distance tie-break,
        never by generational replacement alone."""
        unique = list(dict.fromkeys(population))
        indices = {c:i for i,c in enumerate(population)}
        objectives = objectives[[indices[c] for c in unique]]
        population = unique
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

    def run(self, on_generation=None) -> GAResult:
        """Return a cumulative archive of every evaluated nondominated solution.

        Population HV is separately recorded and may decrease. The archive is
        updated before truncation so even discarded offspring remain evidence.
        on_generation receives an immutable summary suitable for incremental export.
        """
        start = time.perf_counter()
        self._cache.clear(); self._evaluations=0; self._cache_hits=0
        population=[]
        space=math.comb(len(self.pool),self.routes_per_solution)
        if space <= self.population_size:
            population=[frozenset(c) for c in itertools.combinations(range(len(self.pool)),self.routes_per_solution)]
        else:
            while len(population)<self.population_size:
                c=self._random_chromosome()
                if c not in population: population.append(c)
        objectives=self._evaluate_all(population)
        archive={}
        history=[]; pop_history=[]; best=[]; evaluations=[]
        def record(generation, candidates, values):
            archive.update(zip(candidates,map(tuple,values)))
            keys=list(archive)
            front=fast_non_dominated_sort(list(archive.values()))[0]
            retained={keys[i]:archive[keys[i]] for i in front}
            archive.clear();archive.update(retained)
            archive_values=np.array(list(archive.values()))
            history.append(self._hypervolume(archive_values))
            pop_history.append(self._hypervolume(objectives[fast_non_dominated_sort(objectives)[0]]))
            best.append(tuple(archive_values.min(axis=0)))
            evaluations.append(self._evaluations)
            if on_generation:
                on_generation(dict(generation=generation,archive_hypervolume=history[-1],
                                   population_hypervolume=pop_history[-1],evaluations=self._evaluations,
                                   archive_size=len(archive)))
        record(0,population,objectives)
        for generation in range(1,self.generations+1):
            rank=np.empty(len(population),dtype=int); crowding=np.empty(len(population))
            for fr,front in enumerate(fast_non_dominated_sort(objectives)):
                rank[front]=fr;crowding[front]=crowding_distance(objectives,front)
            offspring=[]
            attempts=0
            while len(offspring)<self.population_size:
                attempts+=1
                a=tournament_select(rank,crowding,self.rng); b=tournament_select(rank,crowding,self.rng)
                child=repair(mutate(crossover(population[a],population[b],self.routes_per_solution,self.rng),
                                    len(self.pool),self.mutation_rate,self.rng),len(self.pool),self.routes_per_solution,self.rng)
                if attempts>20*self.population_size: child=self._random_chromosome()
                if child not in offspring: offspring.append(child)
            child_values=self._evaluate_all(offspring)
            population,objectives=self._select_next_generation(population+offspring,np.vstack([objectives,child_values]))
            record(generation,offspring,child_values)
        front=fast_non_dominated_sort(objectives)[0]
        return GAResult(pareto_front=list(archive.items()),hypervolume_history=history,
                        best_per_generation=best,evaluations=self._evaluations,cache_hits=self._cache_hits,
                        wall_clock_seconds=time.perf_counter()-start,reference_point=tuple(self.reference_point),
                        population_hypervolume_history=pop_history,
                        population_front=[(population[i],tuple(objectives[i])) for i in front],
                        evaluation_history=evaluations)
