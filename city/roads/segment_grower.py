"""
segment_grower.py

Grows a road network outward from a start point following Parish & Muller's
CityEngine approach: each proposed "ideal successor" is shaped by global
goals (steer toward population, snapped to a street pattern) and then
filtered by local constraints (stay off obstacles, retrying rotated
candidates so roads hug the edges of water/mountains instead of stopping).

Output is a flat list of segments consumed unchanged by RoadGraphBuilder.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence

import numpy as np

import config

assert config.ROAD_MIN_JUNCTION_LENGTH > config.ROAD_SNAP_TOLERANCE, (
    f"ROAD_MIN_JUNCTION_LENGTH ({config.ROAD_MIN_JUNCTION_LENGTH}) must exceed "
    f"ROAD_SNAP_TOLERANCE ({config.ROAD_SNAP_TOLERANCE}), or a connecting "
    "segment this short collapses to zero length once its endpoints are "
    "snapped together in the graph builder."
)

Point = tuple[float, float]
Segment = tuple[Point, Point]
Cell = tuple[int, int]

# Step size used to walk a segment when marking/testing road proximity.
# Independent of ROAD_SEGMENT_LENGTH: this is a sampling resolution, not
# a road-network parameter, so it does not belong in config.py.
_PROXIMITY_STEP = 0.5

# A candidate this close to 180 degrees from the incoming direction would
# retrace the parent segment's own path almost exactly. parent_cells
# exempts a segment's *entire* footprint regardless of direction, so
# without this guard a road end can "U-turn" straight back onto its own
# parent, find every cell it touches ancestry-exempt, and treat the
# retrace as open ground - ping-ponging between two points indefinitely
# instead of ever reaching a real junction. Comfortably above the ~105
# degree max deviation a legitimate (non-reversal) candidate can reach
# (60 degree ray spread + up to 45 degrees of rotation retries).
_U_TURN_THRESHOLD_DEG = 150.0


def _disk_offsets(radius: float) -> list[Cell]:
    """Integer (drow, dcol) offsets of every cell within ``radius`` of a
    center cell, used to expand a single occupied cell into a clearance
    disk."""
    r = int(math.ceil(radius))
    return [
        (dr, dc)
        for dr in range(-r, r + 1)
        for dc in range(-r, r + 1)
        if dr * dr + dc * dc <= radius * radius
    ]


def _normalize_starts(starts: Point | Sequence[Point]) -> list[Point]:
    """Wraps a single (x, y) start into a one-element list; passes a
    sequence of (x, y) points through unchanged (as a list of tuples)."""
    if (
        len(starts) == 2
        and isinstance(starts[0], (int, float))
        and isinstance(starts[1], (int, float))
    ):
        return [(float(starts[0]), float(starts[1]))]
    return [(float(x), float(y)) for x, y in starts]


def _angle_diff(a: float, b: float) -> float:
    """Smallest angle, in [0, 180], between two directions in degrees."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _nearest_heading(angle_deg: float, line_deg: float) -> float:
    """Snaps ``angle_deg`` to whichever heading of the line (line or
    line+180) is closer, preserving the original direction of travel."""
    opposite = line_deg + 180.0
    if _angle_diff(angle_deg, line_deg) <= _angle_diff(angle_deg, opposite):
        return line_deg
    return opposite


def _snap_grid(angle_deg: float, position: Point, start: Point) -> float:
    del position, start
    relative = angle_deg - config.GRID_BASE_ANGLE_DEG
    snapped_relative = round(relative / 90.0) * 90.0
    return config.GRID_BASE_ANGLE_DEG + snapped_relative


def _snap_radial(angle_deg: float, position: Point, start: Point) -> float:
    dx = position[0] - start[0]
    dy = position[1] - start[1]
    if dx == 0.0 and dy == 0.0:
        # At the start point no spoke bearing is defined yet; let the
        # initial evenly-spaced branch directions stand as-is.
        return angle_deg

    bearing = math.degrees(math.atan2(dy, dx))
    ring = bearing + 90.0

    spoke_dist = min(_angle_diff(angle_deg, bearing), _angle_diff(angle_deg, bearing + 180.0))
    spoke_dist -= config.RADIAL_SPOKE_BIAS
    ring_dist = min(_angle_diff(angle_deg, ring), _angle_diff(angle_deg, ring + 180.0))

    line = bearing if spoke_dist <= ring_dist else ring
    return _nearest_heading(angle_deg, line)


PATTERNS = {
    "GRID": _snap_grid,
    "RADIAL": _snap_radial,
}


class RoadNetworkGrower:
    """Grows a flat list of road segments from one or more seed points.

    Args:
        population: (N, N) float array, 0..1, indexed ``[row=y, col=x]``.
        obstacle: (N, N) bool array, True = impassable, same indexing.
        starts: a single (x, y) seed coordinate, or a sequence of them.
            Each start grows independently and gets its own share of
            ``ROAD_MAX_SEGMENTS``; RADIAL bearings are measured from
            whichever start a given road end descended from.
        pattern: name of the street pattern snapping function to use.
        seed: seeds this instance's own RNG; never touches global random
            state, so multiple growers can run side by side deterministically.
    """

    def __init__(
        self,
        population: np.ndarray,
        obstacle: np.ndarray,
        starts: Point | Sequence[Point],
        pattern: str = config.STREET_PATTERN,
        seed: int = config.SEED,
    ) -> None:
        if pattern not in PATTERNS:
            valid = ", ".join(sorted(PATTERNS))
            raise ValueError(f"Unknown street pattern {pattern!r}; valid options: {valid}")

        self.population = population
        self.obstacle = obstacle
        self.starts: list[Point] = _normalize_starts(starts)
        self.pattern = pattern
        self.snap = PATTERNS[pattern]
        self.rows, self.cols = population.shape
        self.rng = np.random.default_rng(seed)

        # Cells already claimed by emitted road, expanded by
        # ROAD_MIN_SEPARATION, so later segments steer away from roads
        # that already cover the same ground instead of tracing them.
        self.occupied: set[Cell] = set()
        self.redundant_discards: int = 0
        self._separation_offsets = _disk_offsets(config.ROAD_MIN_SEPARATION)

    def grow(self) -> list[Segment]:
        segments: list[Segment] = []
        budget = config.ROAD_MAX_SEGMENTS // len(self.starts)
        counts = [0] * len(self.starts)

        # (position, incoming angle, depth, owning start index, cells the
        # direct parent segment stamped - exempt from this road end's own
        # proximity check, since it legitimately begins there)
        empty_cells: frozenset[Cell] = frozenset()
        queue: deque[tuple[Point, float, int, int, frozenset[Cell]]] = deque()
        for start_index, start in enumerate(self.starts):
            for i in range(config.ROAD_INITIAL_BRANCHES):
                angle = i * 360.0 / config.ROAD_INITIAL_BRANCHES
                queue.append((start, angle, 0, start_index, empty_cells))

        while queue and len(segments) < config.ROAD_MAX_SEGMENTS:
            position, incoming_angle, depth, start_index, parent_cells = queue.popleft()
            if depth >= config.ROAD_MAX_DEPTH or counts[start_index] >= budget:
                continue

            start = self.starts[start_index]
            best_angle, best_score = self._choose_direction(position, incoming_angle, start)
            if best_score < config.ROAD_MIN_POPULATION_SCORE:
                continue

            final_angle = self._resolve_valid_angle(
                position, incoming_angle, best_angle, start, parent_cells
            )
            if final_angle is None:
                continue

            new_position = self._step(position, final_angle, config.ROAD_SEGMENT_LENGTH)
            segments.append((position, new_position))
            stamped_cells = self._mark_road(position, final_angle)
            counts[start_index] += 1
            queue.append((new_position, final_angle, depth + 1, start_index, stamped_cells))

            if self.rng.random() < config.ROAD_BRANCH_PROBABILITY:
                turn = 90.0 if self.rng.random() < 0.5 else -90.0
                queue.append(
                    (new_position, final_angle + turn, depth + 1, start_index, stamped_cells)
                )

        return segments

    def _choose_direction(
        self, position: Point, incoming_angle: float, start: Point
    ) -> tuple[float, float]:
        offsets = np.linspace(
            -config.ROAD_RAY_SPREAD_DEG, config.ROAD_RAY_SPREAD_DEG, config.ROAD_RAY_COUNT
        )
        best_angle = incoming_angle
        best_score = -1.0
        for offset in offsets:
            candidate = self.snap(incoming_angle + float(offset), position, start)
            score = self._score_ray(position, candidate)
            if score > best_score:
                best_score = score
                best_angle = candidate
        return best_angle, best_score

    def _score_ray(self, position: Point, angle_deg: float) -> float:
        rad = math.radians(angle_deg)
        dx, dy = math.cos(rad), math.sin(rad)
        distances = np.linspace(0.0, config.ROAD_SEGMENT_LENGTH, config.ROAD_RAY_SAMPLES)
        score = 0.0
        for dist in distances:
            px = position[0] + dx * dist
            py = position[1] + dy * dist
            score += self._sample_population(px, py) / (1.0 + dist)
        return score

    def _sample_population(self, x: float, y: float) -> float:
        row, col = round(y), round(x)
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return float(self.population[row, col])
        return 0.0

    def _resolve_valid_angle(
        self,
        position: Point,
        incoming_angle: float,
        angle_deg: float,
        start: Point,
        parent_cells: frozenset[Cell],
    ) -> float | None:
        """Finds an angle whose segment is both obstacle-legal and not
        redundant with existing road, retrying rotated candidates exactly
        like the obstacle check does. If every attempt is legal but keeps
        running into existing road, this road end has reached a crossing
        and stops growing (Parish & Muller section 3.3.1); redundant_discards
        is incremented so that specific outcome is distinguishable from a
        plain obstacle-blocked discard. Loop-closing (connecting roads
        that meet) is a separate, dedicated post-processing pass - see
        loop_closer.py - not something the grower does inline.
        """
        saw_redundant = False
        candidate = angle_deg

        for attempt in range(0, config.ROAD_ROTATION_ATTEMPTS + 1):
            if attempt > 0:
                sign = 1.0 if attempt % 2 == 1 else -1.0
                magnitude = math.ceil(attempt / 2) * config.ROAD_ROTATION_STEP_DEG
                candidate = self.snap(angle_deg + sign * magnitude, position, start)

            if not self._is_legal(position, candidate):
                continue

            # A near-U-turn would retrace the parent's own path, which
            # ancestry exemption would then wave through as open ground.
            is_u_turn = _angle_diff(candidate, incoming_angle) > _U_TURN_THRESHOLD_DEG
            effective_parent_cells = frozenset() if is_u_turn else parent_cells

            if self._is_redundant(position, candidate, effective_parent_cells):
                saw_redundant = True
                continue
            return candidate

        if saw_redundant:
            self.redundant_discards += 1
        return None

    def _is_legal(self, position: Point, angle_deg: float) -> bool:
        length = config.ROAD_SEGMENT_LENGTH
        distances = list(np.arange(1.0, length, 1.0)) + [length]
        rad = math.radians(angle_deg)
        dx, dy = math.cos(rad), math.sin(rad)
        for dist in distances:
            px = position[0] + dx * dist
            py = position[1] + dy * dist
            row, col = round(py), round(px)
            if not (0 <= row < self.rows and 0 <= col < self.cols):
                return False
            if self.obstacle[row, col]:
                return False
        return True

    def _is_redundant(
        self, position: Point, angle_deg: float, parent_cells: frozenset[Cell]
    ) -> bool:
        """Whether a proposed segment runs into road already laid.

        Two independent exemptions apply, since every segment legitimately
        begins on an existing road end:
        - ancestry: cells stamped by the direct parent segment (the one
          this road end continues or branches from) never count, no
          matter how far along the new segment they're touched. This is
          direction-aware, unlike a fixed radius, so a segment that turns
          sharply back doesn't falsely collide with the road it just grew
          from (and is itself withheld for a near-U-turn candidate - see
          the caller - so a reversal can't retrace the parent forever).
        - distance: the first ROAD_PROXIMITY_IGNORE units of the new
          segment are skipped outright, as a backstop for road ends with
          no parent (the initial branches) and general safety margin.

        Collisions with any OTHER segment's cells - even ones the parent
        also happens to touch - still make the proposal redundant.
        """
        length = config.ROAD_SEGMENT_LENGTH
        rad = math.radians(angle_deg)
        dx, dy = math.cos(rad), math.sin(rad)
        for dist in np.arange(0.0, length + 1e-9, _PROXIMITY_STEP):
            if dist < config.ROAD_PROXIMITY_IGNORE:
                continue
            px = position[0] + dx * dist
            py = position[1] + dy * dist
            cell = (round(py), round(px))
            if cell in self.occupied and cell not in parent_cells:
                return True
        return False

    def _mark_road(self, position: Point, angle_deg: float) -> frozenset[Cell]:
        """Claims every cell an emitted segment passes through, expanded
        by ROAD_MIN_SEPARATION, so later segments steer away from it.
        Returns that segment's own stamped cells, so the road ends it
        spawns can carry them along as their ancestry exemption."""
        length = config.ROAD_SEGMENT_LENGTH
        rad = math.radians(angle_deg)
        dx, dy = math.cos(rad), math.sin(rad)
        stamped: set[Cell] = set()
        for dist in np.arange(0.0, length + 1e-9, _PROXIMITY_STEP):
            px = position[0] + dx * dist
            py = position[1] + dy * dist
            row, col = round(py), round(px)
            for dr, dc in self._separation_offsets:
                stamped.add((row + dr, col + dc))
        self.occupied |= stamped
        return frozenset(stamped)

    @staticmethod
    def _step(position: Point, angle_deg: float, distance: float) -> Point:
        rad = math.radians(angle_deg)
        return (
            position[0] + math.cos(rad) * distance,
            position[1] + math.sin(rad) * distance,
        )
