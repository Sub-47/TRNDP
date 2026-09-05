"""One objective definition shared by search, baselines and exported results."""
from __future__ import annotations
import numpy as np
import config
from city.sim.simulator import SimResult


def objective_values(result: SimResult, transfer_penalty: float | None = None) -> tuple[float,float,float]:
    penalty=config.GA_TRANSFER_PENALTY_MINUTES if transfer_penalty is None else transfer_penalty
    values=(result.total_wait_time+result.total_travel_time+result.total_transfers*penalty,
            result.total_bus_distance,result.incomplete_passengers)
    if not np.all(np.isfinite(values)) or min(values)<0:
        raise ValueError('objectives must be finite and nonnegative')
    return tuple(float(v) for v in values)


def reference_point(demand, routes_per_solution: int, frequency: float) -> np.ndarray:
    """Conservative common bounds for fixed-horizon experiments on one instance.

    Each passenger accrues at most horizon minutes in queue/onboard combined,
    plus MAX_TRANSFERS penalties. Each direction dispatches ceil(horizon*f/60)
    vehicles, each travelling at most horizon*speed cells. A 10% margin keeps
    feasible points strictly inside the reference box. No run-specific fitting.
    """
    transit=float(np.asarray(demand).sum()-np.trace(demand))
    duration=config.SIM_DURATION_MINUTES
    trips=int(np.ceil(duration*frequency/60.)) if frequency>0 else 0
    bounds=[transit*(duration+config.MAX_TRANSFERS*config.GA_TRANSFER_PENALTY_MINUTES),
            routes_per_solution*2*trips*duration*config.BUS_SPEED_CELLS_PER_MINUTE,transit]
    return np.maximum(bounds,1.)*1.1
