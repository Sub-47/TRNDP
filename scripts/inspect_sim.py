"""
scripts/inspect_sim.py

Standalone diagnostic instrument for TransitSimulator: builds the real
world/road/stop/demand/route-pool pipeline, picks a plausible route set,
runs one simulation, and prints every result field plus derived summary
numbers. Asserts nothing.

Run: python scripts/inspect_sim.py
"""

from __future__ import annotations

import os
import sys
import time
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config
from city.sim.simulator import TransitSimulator

# Diagnostic-only choices (not generation parameters, so not in config.py):
# a plausible-sized route set and a uniform frequency to exercise the
# simulator with.
DIAGNOSTIC_ROUTE_COUNT = 8
DIAGNOSTIC_FREQUENCY = 4.0  # buses/hour per route
FREQUENCY_SWEEP = (4.0, 8.0, 16.0, 32.0)  # buses/hour per route
REACHABILITY_ROUTE_COUNTS = (8, 20, 40)
OCCUPANCY_TRACE_FREQUENCY = 32.0  # buses/hour per route
OCCUPANCY_TRACE_STRIDE = 20  # print every Nth step


from city.pipeline import build_pipeline as complete_pipeline, build_stop_od_matrix
from city.experiments import choose_greedy


def build_pipeline():
    p=complete_pipeline()
    return p.stops,p.zones,p.demand,p.stop_distances,p.routes


def select_routes_greedily(routes: list, stop_od: np.ndarray, count: int) -> list:
    return [routes[i] for i in sorted(choose_greedy(routes,stop_od,count))]


def reachability_audit(routes: list, stop_od: np.ndarray, max_transfers: int) -> dict:
    """Structural O-D reachability of a route set, independent of the
    simulation entirely - no frequency, capacity, dwell, or direction of
    travel involved, just "does a path through these routes' stop-lists
    exist at all, and how many route-changes does it need". This is a
    ceiling: it answers whether 84% unserved at 32 buses/hour is even
    theoretically fixable by adding more buses to these routes, or
    whether the routes themselves don't cover the demand.

    Two stops are 0 transfers apart if some route serves both. They are
    N transfers apart if the shortest path between "a route serving the
    origin" and "a route serving the destination", in a graph where two
    routes are adjacent iff they share a stop, has length N.

    Returns:
        {"transfers": {0: pct, 1: pct, ..., max_transfers: pct},
         "unreachable": pct} - percentages of total O-D demand mass.
    """
    stop_od = stop_od.copy()
    np.fill_diagonal(stop_od,0.)
    n_routes = len(routes)
    n_stops = stop_od.shape[0]
    route_stops = [set(route.stops) for route in routes]

    stop_to_routes: list[list[int]] = [[] for _ in range(n_stops)]
    for route_index, stops_on_route in enumerate(route_stops):
        for stop in stops_on_route:
            stop_to_routes[stop].append(route_index)

    adjacency: list[list[int]] = [[] for _ in range(n_routes)]
    for i in range(n_routes):
        for j in range(i + 1, n_routes):
            if route_stops[i] & route_stops[j]:
                adjacency[i].append(j)
                adjacency[j].append(i)

    def route_hops_from_stop(stop: int) -> dict[int, int]:
        """BFS distance (route-changes) from any route serving `stop` to
        every route reachable from it."""
        dist = {r: 0 for r in stop_to_routes[stop]}
        frontier = deque(stop_to_routes[stop])
        while frontier:
            current = frontier.popleft()
            for neighbour in adjacency[current]:
                if neighbour not in dist:
                    dist[neighbour] = dist[current] + 1
                    frontier.append(neighbour)
        return dist

    stop_hops = [route_hops_from_stop(s) for s in range(n_stops)]

    transfers = {t: 0.0 for t in range(max_transfers + 1)}
    unreachable = 0.0
    total = 0.0
    for i in range(n_stops):
        for j in range(n_stops):
            mass = stop_od[i, j]
            if mass <= 0:
                continue
            total += mass
            candidates = [
                stop_hops[i][r] for r in stop_to_routes[j] if r in stop_hops[i]
            ]
            best = min(candidates) if candidates else None
            if best is None or best > max_transfers:
                unreachable += mass
            else:
                transfers[best] += mass

    if total <= 0:
        return {"transfers": {t: 0.0 for t in transfers}, "unreachable": 0.0}
    return {
        "transfers": {t: 100.0 * m / total for t, m in transfers.items()},
        "unreachable": 100.0 * unreachable / total,
    }


def run_reachability_audits(pool_routes: list, stop_od: np.ndarray) -> None:
    """Diagnostic only - the audit that actually answers whether the
    frequency sweep's flat served% is a coverage limit (this table shows
    most demand unreachable regardless of route count) or a bug
    elsewhere (this table shows most demand reachable in 0-2 transfers,
    contradicting what the simulation delivers)."""
    print("-" * 60)
    print("Reachability audit (structural, no simulation):")
    header = f"  {'routes':>6} | {'0 xfer':>7} | {'1 xfer':>7} | {'2 xfer':>7} | {'unreachable':>11}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for count in REACHABILITY_ROUTE_COUNTS:
        selected = select_routes_greedily(pool_routes, stop_od, count)
        audit = reachability_audit(selected, stop_od, config.MAX_TRANSFERS)
        t = audit["transfers"]
        print(
            f"  {count:>6} | {t[0]:>6.2f}% | {t[1]:>6.2f}% | {t[2]:>6.2f}% | "
            f"{audit['unreachable']:>10.2f}%"
        )


def run_occupancy_trace(selected: list, stop_distances: np.ndarray, stop_od: np.ndarray) -> None:
    """Diagnostic only - mean onboard passengers per active bus, sampled
    every OCCUPANCY_TRACE_STRIDE steps at OCCUPANCY_TRACE_FREQUENCY. A
    trace that keeps climbing and never comes back down would mean buses
    are absorbing passengers they never deliver (a boarding/alighting
    bug); one that rises then plateaus/falls is buses genuinely filling
    up and cycling passengers through as normal."""
    frequencies = [OCCUPANCY_TRACE_FREQUENCY] * len(selected)
    trace: list[tuple[int, float, int]] = []

    def on_step(step_index: int, buses: list) -> None:
        if step_index % OCCUPANCY_TRACE_STRIDE != 0:
            return
        if not buses:
            trace.append((step_index, 0.0, 0))
            return
        occupancy = [bus.manifest.sum() for bus in buses]
        trace.append((step_index, sum(occupancy) / len(occupancy), len(buses)))

    TransitSimulator(selected, stop_distances, stop_od, frequencies).run(on_step=on_step)

    print("-" * 60)
    print(f"Per-bus occupancy trace at {OCCUPANCY_TRACE_FREQUENCY} buses/hour, {len(selected)} routes:")
    print(f"  {'step':>6} | {'mean onboard':>12} | {'active buses':>12}")
    for step_index, mean_occupancy, active_buses in trace:
        print(f"  {step_index:>6} | {mean_occupancy:>12.2f} | {active_buses:>12}")


def run_frequency_sweep(selected: list, stop_distances: np.ndarray, stop_od: np.ndarray) -> None:
    """Diagnostic only - reports how % demand served, mean wait per
    created passenger, and peak_load_factor move across route frequency,
    holding the same 8 routes fixed. Does not pick a value; at
    DIAGNOSTIC_FREQUENCY=4.0, peak_load_factor pins at 1.000 and only
    ~16% of demand is served, meaning supply (not route quality) is the
    binding constraint there - this sweep exists to show whether that's
    still true further up the range."""
    print("-" * 60)
    print("Frequency sensitivity sweep (diagnostic, not tuning):")
    header = f"  {'freq':>6} | {'served %':>8} | {'wait/created':>12} | {'peak_load':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for freq in FREQUENCY_SWEEP:
        frequencies = [freq] * len(selected)
        result = TransitSimulator(selected, stop_distances, stop_od, frequencies).run()
        pct_served = 100.0 * result.total_served / result.total_created if result.total_created > 0 else 0.0
        mean_wait_per_created = (
            result.total_wait_time / result.total_created if result.total_created > 0 else 0.0
        )
        assert mean_wait_per_created <= config.SIM_DURATION_MINUTES, (
            f"freq={freq}: mean wait per created passenger ({mean_wait_per_created:.2f} min) "
            f"exceeds SIM_DURATION_MINUTES ({config.SIM_DURATION_MINUTES})"
        )
        print(
            f"  {freq:>6.1f} | {pct_served:>7.2f}% | {mean_wait_per_created:>12.2f} | "
            f"{result.peak_load_factor:>9.3f}"
        )


def main() -> None:
    stops, zones, demand, stop_distances, pool_routes = build_pipeline()
    stop_od = build_stop_od_matrix(zones, demand, stops)

    selected = select_routes_greedily(pool_routes, stop_od, DIAGNOSTIC_ROUTE_COUNT)
    frequencies = [DIAGNOSTIC_FREQUENCY] * len(selected)

    print("=" * 60)
    print("TRANSIT SIMULATION DIAGNOSTIC")
    print("=" * 60)
    print(f"stops                            : {len(stops)}")
    print(f"routes in pool                   : {len(pool_routes)}")
    print(f"routes selected                  : {len(selected)}")
    print(f"frequency per route (buses/hour) : {DIAGNOSTIC_FREQUENCY}")
    for i, route in enumerate(selected):
        print(f"  route {i}: {len(route.stops)} stops, {route.length:.2f} cells")

    t0 = time.perf_counter()
    result = TransitSimulator(selected, stop_distances, stop_od, frequencies).run()
    runtime = time.perf_counter() - t0

    print("-" * 60)
    print("SimResult (transit counts exclude local/walking demand):")
    print(f"  local/walking demand            : {result.local_or_walking_demand:.2f}")
    print(f"  incomplete transit passengers   : {result.incomplete_passengers:.2f}")
    print(f"  total_wait_time                : {result.total_wait_time:.2f} passenger-min")
    print(f"  total_travel_time              : {result.total_travel_time:.2f} passenger-min")
    print(f"  total_transfers                : {result.total_transfers:.2f}")
    print(f"  unserved_passengers            : {result.unserved_passengers:.2f}")
    print(f"  total_bus_distance              : {result.total_bus_distance:.2f} cells")
    print(f"  peak_load_factor                : {result.peak_load_factor:.3f}")
    print(f"  total_created                    : {result.total_created:.2f}")
    print(f"  total_served                    : {result.total_served:.2f}")
    print(f"  still_onboard                    : {result.still_onboard:.2f}")
    print(f"  served_wait_time                : {result.served_wait_time:.2f} passenger-min")
    print(f"  unserved_wait_time              : {result.unserved_wait_time:.2f} passenger-min")
    print(f"  still_onboard_wait_time         : {result.still_onboard_wait_time:.2f} passenger-min")

    print("-" * 60)
    served = result.total_served
    created = result.total_created
    # total_wait_time mixes wait accrued by unserved passengers (who,
    # by definition, keep waiting right up to the end of the run) with
    # wait accrued by served ones - dividing it by served alone was
    # overstating a served passenger's actual experience. served_wait_time
    # (tracked separately in simulator.py) fixes that.
    mean_wait_per_served = result.served_wait_time / served if served > 0 else 0.0
    mean_wait_per_created = result.total_wait_time / created if created > 0 else 0.0
    mean_transfers_per_served = result.served_transfers / served if served > 0 else 0.0

    print(f"  mean wait per SERVED passenger   : {mean_wait_per_served:.2f} min")
    print(f"  mean wait per CREATED passenger  : {mean_wait_per_created:.2f} min")
    print(f"  mean transfers per served pax    : {mean_transfers_per_served:.3f}")

    pct_served = 100.0 * served / created if created > 0 else 0.0
    print(f"  percentage of transit demand served      : {pct_served:.2f}%")

    # Sanity check: no passenger-mass, however it ends up, can have
    # waited longer than the simulation actually ran. A violation here
    # means the served/unserved wait bookkeeping in simulator.py has a
    # bug - it would otherwise be invisible in the aggregate numbers.
    assert mean_wait_per_served <= config.SIM_DURATION_MINUTES, (
        f"mean wait per served passenger ({mean_wait_per_served:.2f} min) exceeds "
        f"SIM_DURATION_MINUTES ({config.SIM_DURATION_MINUTES})"
    )
    assert mean_wait_per_created <= config.SIM_DURATION_MINUTES, (
        f"mean wait per created passenger ({mean_wait_per_created:.2f} min) exceeds "
        f"SIM_DURATION_MINUTES ({config.SIM_DURATION_MINUTES})"
    )

    print("-" * 60)
    print(f"wall-clock runtime for one simulation run: {runtime * 1000:.2f} ms")
    print("(this is the GA's per-fitness-evaluation budget)")

    run_frequency_sweep(selected, stop_distances, stop_od)
    run_reachability_audits(pool_routes, stop_od)
    run_occupancy_trace(selected, stop_distances, stop_od)
    print("=" * 60)


if __name__ == "__main__":
    main()
