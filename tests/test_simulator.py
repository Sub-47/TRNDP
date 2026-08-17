"""
tests/test_simulator.py

Exercises TransitSimulator against small, hand-built scenarios (never the
real pipeline): a single 5-stop straight-line route with known
stop-to-stop distances, so wait/travel/transfer/capacity behavior is easy
to reason about by hand. Conservation (test 1) is the one that matters
most - everything else can look reasonable while still silently losing
or duplicating passengers.
"""

from __future__ import annotations

import numpy as np
import pytest

import config
from city.routes.route_pool import Route
from city.sim.simulator import TransitSimulator


def _linear_stops(n: int, spacing: float) -> np.ndarray:
    """Stop-to-stop distance matrix for n stops evenly spaced on a line -
    a simple, exactly-known topology (no shortest-path ambiguity)."""
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distances[i, j] = abs(i - j) * spacing
    return distances


def _single_route(n: int = 5, spacing: float = 5.0) -> tuple[np.ndarray, Route]:
    stop_distances = _linear_stops(n, spacing)
    route = Route(stops=list(range(n)), length=(n - 1) * spacing)
    return stop_distances, route


def test_conservation_created_equals_served_plus_unserved_plus_onboard():
    stop_distances, route = _single_route()
    demand = np.zeros((5, 5))
    demand[0, 4] = 300.0

    result = TransitSimulator([route], stop_distances, demand, [6.0]).run()

    assert result.total_created == pytest.approx(demand.sum())
    assert result.total_created == pytest.approx(
        result.total_served + result.unserved_passengers + result.still_onboard
    )


def test_wait_time_splits_conserve_and_stay_within_duration():
    # Heavy demand on a low frequency so all three outcomes (served,
    # unserved, still onboard) are non-trivially populated, exercising
    # the served/unserved wait split's proportional-withdrawal
    # bookkeeping rather than a scenario where one bucket is just empty.
    stop_distances, route = _single_route()
    demand = np.zeros((5, 5))
    demand[0, 4] = 5000.0

    result = TransitSimulator([route], stop_distances, demand, [1.0]).run()

    assert result.total_served > 0.0
    assert result.unserved_passengers > 0.0
    assert result.still_onboard > 0.0

    assert result.total_wait_time == pytest.approx(
        result.served_wait_time + result.unserved_wait_time + result.still_onboard_wait_time
    )

    # No passenger-mass can have waited longer than the simulation ran,
    # regardless of which outcome it ended up in.
    assert result.served_wait_time / result.total_served <= config.SIM_DURATION_MINUTES
    assert result.unserved_wait_time / result.unserved_passengers <= config.SIM_DURATION_MINUTES


def test_zero_demand_gives_zero_waits_and_transfers_but_buses_still_run():
    stop_distances, route = _single_route()
    demand = np.zeros((5, 5))

    result = TransitSimulator([route], stop_distances, demand, [6.0]).run()

    assert result.total_wait_time == 0.0
    assert result.total_transfers == 0.0
    assert result.total_created == 0.0
    assert result.total_bus_distance > 0.0


def test_higher_frequency_reduces_total_wait_time():
    stop_distances, route = _single_route()
    demand = np.zeros((5, 5))
    demand[0, 4] = 300.0

    low_freq = TransitSimulator([route], stop_distances, demand, [2.0]).run()
    high_freq = TransitSimulator([route], stop_distances, demand, [12.0]).run()

    assert high_freq.total_wait_time < low_freq.total_wait_time


def test_shared_route_od_pair_makes_zero_transfers():
    stop_distances, route = _single_route()
    demand = np.zeros((5, 5))
    demand[0, 4] = 300.0  # both stop 0 and stop 4 are on the one route

    result = TransitSimulator([route], stop_distances, demand, [6.0]).run()

    assert result.total_transfers == 0.0


def test_capacity_binds_leaves_passengers_unserved():
    stop_distances, route = _single_route()
    demand = np.zeros((5, 5))
    demand[0, 4] = 5000.0  # far above what a low-frequency single route can carry

    result = TransitSimulator([route], stop_distances, demand, [1.0]).run()

    assert result.unserved_passengers > 0.0


def test_determinism():
    stop_distances, route = _single_route()
    demand = np.zeros((5, 5))
    demand[0, 4] = 300.0

    result_a = TransitSimulator([route], stop_distances, demand, [6.0]).run()
    result_b = TransitSimulator([route], stop_distances, demand, [6.0]).run()

    assert result_a == result_b


def test_doubling_duration_roughly_doubles_demand_served(monkeypatch):
    # A throughput-limited scenario (heavy demand, low-ish frequency)
    # where only a modest fraction of demand is served in the base
    # window, so there's room for more service time to matter. Buses
    # never retire once dispatched (per spec) and the fleet keeps
    # growing over the run, so served-vs-duration isn't perfectly
    # linear here - "roughly doubles" is checked with a generous
    # tolerance band rather than an exact 2x.
    stop_distances, route = _single_route(spacing=10.0)
    demand = np.zeros((5, 5))
    demand[0, 4] = 700.0

    monkeypatch.setattr(config, "SIM_DURATION_MINUTES", 120.0)
    base = TransitSimulator([route], stop_distances, demand, [4.0]).run()

    monkeypatch.setattr(config, "SIM_DURATION_MINUTES", 240.0)
    doubled = TransitSimulator([route], stop_distances, demand, [4.0]).run()

    assert base.total_served > 0.0
    ratio = doubled.total_served / base.total_served
    assert 1.5 <= ratio <= 3.0
