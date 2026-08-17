"""
city/sim/simulator.py

Discrete-time simulation of one period's bus operation over a fixed route
set - the GA's fitness function (behavioural simulation, not a closed-form
analytical formula). Steps time forward in fixed increments: passengers
arrive, buses move, passengers alight (at their destination, or at the
closest interchange stop this route offers if it never reaches it) and
board.

Passengers are a continuous fluid quantity (fractional counts), not
discrete individuals - the model tracks how many people are in a given
state, not who they are. There is no randomness anywhere in this model,
so a run is exactly reproducible from its inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config
from city.routes.route_pool import Route


@dataclass
class SimResult:
    """Aggregate outcomes of one simulation run.

    total_wait_time/total_travel_time are passenger-minutes; unserved_passengers
    are those never picked up by the end, or dropped for exceeding
    MAX_TRANSFERS; total_bus_distance is the operator cost proxy;
    peak_load_factor is the busiest bus's peak onboard count / BUS_CAPACITY.

    total_created/total_served/still_onboard aren't in the spec's result
    list, but are exposed so created == served + unserved + still_onboard
    (the conservation invariant) can be checked from outside, and so %
    demand served can be reported. still_onboard is whoever is riding a
    bus, not yet at a stop, when the run ends - neither served nor
    unserved, just cut off by SIM_DURATION_MINUTES.

    total_wait_time mixes wait accrued by passengers who eventually
    boarded with wait accrued by ones who never did, so dividing it by
    total_served alone overstates served passengers' actual experience.
    served_wait_time/unserved_wait_time/still_onboard_wait_time split it
    by the same three outcomes as served/unserved_passengers/still_onboard,
    and always sum to total_wait_time exactly.
    """

    total_wait_time: float
    total_travel_time: float
    total_transfers: float
    unserved_passengers: float
    total_bus_distance: float
    peak_load_factor: float
    total_created: float
    total_served: float
    still_onboard: float
    served_wait_time: float
    unserved_wait_time: float
    still_onboard_wait_time: float


class _Bus:
    """Mutable per-bus state. `position` is the index (into its route's
    stop list) of the stop the bus is travelling *toward*; it has
    arrived once `distance_to_next` reaches zero."""

    __slots__ = (
        "route_index",
        "direction",
        "position",
        "distance_to_next",
        "dwell_remaining",
        "manifest",
        "manifest_wait",
    )

    def __init__(self, route_index: int, n_stops: int, max_transfers: int) -> None:
        self.route_index = route_index
        self.direction = 1
        self.position = 0
        self.distance_to_next = 0.0
        self.dwell_remaining = 0.0
        # (destination, transfer_count) -> onboard count.
        self.manifest = np.zeros((n_stops, max_transfers + 1))
        # Same shape: wait-minutes already accrued (pre-boarding) by
        # whoever currently occupies each manifest slot, carried over
        # from queue_wait at boarding time.
        self.manifest_wait = np.zeros((n_stops, max_transfers + 1))


class TransitSimulator:
    """Discrete-time simulation of bus operation over a fixed route set.

    Args:
        routes: The GA-selected subset of the candidate pool to operate.
        stop_distances: (n_stops, n_stops) graph distance between every
            pair of stops, in cells (e.g. from
            city.routes.cluster.build_stop_distance_matrix).
        demand: (n_stops, n_stops) stop-to-stop trip matrix for one
            period.
        frequencies: Buses dispatched per hour, one per route.
    """

    def __init__(
        self,
        routes: list[Route],
        stop_distances: np.ndarray,
        demand: np.ndarray,
        frequencies: list[float],
    ) -> None:
        if len(frequencies) != len(routes):
            raise ValueError("frequencies must have one entry per route")

        self.routes = routes
        self.stop_distances = stop_distances
        self.demand = demand
        self.frequencies = frequencies
        self.n_stops = stop_distances.shape[0]
        self.max_transfers = config.MAX_TRANSFERS

        # Per-route lookup tables, built once - never recomputed per
        # passenger or per step.
        self._stop_index_in_route = [
            {stop: idx for idx, stop in enumerate(route.stops)} for route in routes
        ]
        self._closest_on_route = [
            self._build_closest_on_route(np.array(route.stops)) for route in routes
        ]

    def _build_closest_on_route(self, route_stops: np.ndarray) -> np.ndarray:
        """closest_on_route[d] = the stop in `route_stops` nearest (by
        graph distance) to destination stop d - where a passenger bound
        for d alights this route, whether that's d itself or the best
        interchange this route offers."""
        distances = self.stop_distances[route_stops, :]  # (len(route), n_stops)
        return route_stops[np.argmin(distances, axis=0)]

    def run(self, on_step=None) -> SimResult:
        """Runs the simulation.

        Args:
            on_step: Optional `(step_index, buses) -> None` called after
                each step's movement is resolved - a diagnostic hook
                (e.g. for tracing per-bus occupancy over time) with no
                effect on the returned SimResult.
        """
        step = config.SIM_TIME_STEP_MINUTES
        duration = config.SIM_DURATION_MINUTES
        num_steps = int(round(duration / step))
        arrival_fraction = step / duration

        queue = np.zeros((self.n_stops, self.n_stops, self.max_transfers + 1))
        # Wait-minutes already accrued by whoever currently occupies each
        # queue bucket - a well-mixed/proportional approximation, since
        # passengers are a continuous quantity with no individual arrival
        # timestamps: when part of a bucket boards, it takes the same
        # fraction of that bucket's accrued wait with it.
        queue_wait = np.zeros_like(queue)
        buses: list[_Bus] = []
        dispatch_schedules = [self._dispatch_times(f, duration) for f in self.frequencies]

        total_created = total_served = total_unserved = 0.0
        total_transfers = total_travel_time = 0.0
        total_bus_distance = peak_load = 0.0
        # Mutated directly by _arrive_at_stop, reset per run() call so a
        # reused instance stays stateless between runs.
        self._served_wait_time = 0.0
        self._unserved_wait_time = 0.0

        for step_index in range(num_steps):
            now = step_index * step

            # 1. Passenger arrival.
            queue[:, :, 0] += self.demand * arrival_fraction
            total_created += self.demand.sum() * arrival_fraction

            # Dispatch buses due this step - each appears already at its
            # route's first stop and immediately boards waiting
            # passengers (a same-step idealisation), then gets this
            # step's full movement budget below like every other bus.
            for route_index, schedule in enumerate(dispatch_schedules):
                for dispatch_time in schedule:
                    if now <= dispatch_time < now + step:
                        bus = _Bus(route_index, self.n_stops, self.max_transfers)
                        s, u, t = self._arrive_at_stop(bus, queue, queue_wait)
                        total_served += s
                        total_unserved += u
                        total_transfers += t
                        buses.append(bus)

            # 2-4. Move every active bus; alight/board at each stop it
            # reaches within this step's time budget.
            for bus in buses:
                distance, s, u, t = self._advance_bus(bus, queue, queue_wait, step)
                total_bus_distance += distance
                total_served += s
                total_unserved += u
                total_transfers += t
                peak_load = max(peak_load, bus.manifest.sum())

            if on_step is not None:
                on_step(step_index, buses)

            # Time accrual: charge whoever is queued/onboard at the end
            # of this step this step's minutes (vs. start-of-step, an
            # unspecified ordering choice that differs by at most one
            # step_minutes per passenger - negligible at step=1).
            queue_wait += queue * step
            total_travel_time += sum(bus.manifest.sum() for bus in buses) * step

        # Anyone still queued at the end never boarded a single bus -
        # unserved by definition, and their accrued wait becomes
        # unserved wait. Anyone mid-ride is neither served nor unserved,
        # just cut off by SIM_DURATION_MINUTES.
        total_unserved += queue.sum()
        self._unserved_wait_time += queue_wait.sum()
        still_onboard = sum(bus.manifest.sum() for bus in buses)
        still_onboard_wait_time = sum(bus.manifest_wait.sum() for bus in buses)
        total_wait_time = self._served_wait_time + self._unserved_wait_time + still_onboard_wait_time

        peak_load_factor = peak_load / config.BUS_CAPACITY if config.BUS_CAPACITY else 0.0

        return SimResult(
            total_wait_time=total_wait_time,
            total_travel_time=total_travel_time,
            total_transfers=total_transfers,
            unserved_passengers=total_unserved,
            total_bus_distance=total_bus_distance,
            peak_load_factor=peak_load_factor,
            total_created=total_created,
            total_served=total_served,
            still_onboard=still_onboard,
            served_wait_time=self._served_wait_time,
            unserved_wait_time=self._unserved_wait_time,
            still_onboard_wait_time=still_onboard_wait_time,
        )

    @staticmethod
    def _dispatch_times(frequency: float, duration: float) -> np.ndarray:
        if frequency <= 0:
            return np.array([])
        interval = 60.0 / frequency
        return np.arange(0.0, duration, interval)

    def _advance_bus(self, bus: _Bus, queue: np.ndarray, queue_wait: np.ndarray, time_budget: float):
        distance_moved = 0.0
        served = unserved = transferred = 0.0
        speed = config.BUS_SPEED_CELLS_PER_MINUTE

        while time_budget > 1e-9:
            if bus.dwell_remaining > 0:
                spend = min(bus.dwell_remaining, time_budget)
                bus.dwell_remaining -= spend
                time_budget -= spend
                continue

            cells_available = speed * time_budget
            if bus.distance_to_next <= cells_available:
                time_budget -= bus.distance_to_next / speed
                distance_moved += bus.distance_to_next
                bus.distance_to_next = 0.0
                s, u, t = self._arrive_at_stop(bus, queue, queue_wait)
                served += s
                unserved += u
                transferred += t
            else:
                bus.distance_to_next -= cells_available
                distance_moved += cells_available
                time_budget = 0.0

        return distance_moved, served, unserved, transferred

    def _arrive_at_stop(self, bus: _Bus, queue: np.ndarray, queue_wait: np.ndarray):
        route = self.routes[bus.route_index]
        index_in_route = self._stop_index_in_route[bus.route_index]
        closest_on_route = self._closest_on_route[bus.route_index]

        # A terminus reverses direction before boarding, since that's
        # the direction the bus actually departs in next.
        if bus.position == 0:
            bus.direction = 1
        elif bus.position == len(route.stops) - 1:
            bus.direction = -1

        stop = route.stops[bus.position]
        served = unserved = transferred = 0.0

        # Alight: everyone whose nearest stop on this route is here,
        # whether that's their true destination or an interchange point.
        for destination in range(self.n_stops):
            if closest_on_route[destination] != stop:
                continue
            for t in range(self.max_transfers + 1):
                count = bus.manifest[destination, t]
                if count <= 0:
                    continue
                wait = bus.manifest_wait[destination, t]
                bus.manifest[destination, t] = 0.0
                bus.manifest_wait[destination, t] = 0.0
                if destination == stop:
                    served += count
                    self._served_wait_time += wait
                elif t >= self.max_transfers:
                    unserved += count
                    self._unserved_wait_time += wait
                else:
                    queue[stop, destination, t + 1] += count
                    queue_wait[stop, destination, t + 1] += wait
                    transferred += count

        # Board: waiting passengers whose interchange/destination stop
        # is downstream of here in the bus's current direction.
        capacity = config.BUS_CAPACITY - bus.manifest.sum()
        for destination in range(self.n_stops):
            if capacity <= 0:
                break
            target = closest_on_route[destination]
            if target == stop:
                continue
            target_index = index_in_route[target]
            downstream = (
                target_index > bus.position if bus.direction == 1 else target_index < bus.position
            )
            if not downstream:
                continue
            for t in range(self.max_transfers + 1):
                if capacity <= 0:
                    break
                waiting = queue[stop, destination, t]
                if waiting <= 0:
                    continue
                boarding = min(waiting, capacity)
                # Well-mixed approximation: boarding takes the same
                # fraction of this bucket's accrued wait with it, since
                # passengers here have no individual arrival timestamps.
                wait_taken = queue_wait[stop, destination, t] * (boarding / waiting)
                queue[stop, destination, t] -= boarding
                queue_wait[stop, destination, t] -= wait_taken
                bus.manifest[destination, t] += boarding
                bus.manifest_wait[destination, t] += wait_taken
                capacity -= boarding

        # Set up the next leg.
        next_index = bus.position + bus.direction
        bus.dwell_remaining = config.DWELL_MINUTES_PER_STOP
        bus.distance_to_next = self.stop_distances[stop, route.stops[next_index]]
        bus.position = next_index

        return served, unserved, transferred
