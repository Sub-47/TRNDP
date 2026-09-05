"""Small GA over the real generated city, with all production stages included."""
import numpy as np
import pytest
import config
from city.pipeline import build_pipeline
from city.ga.nsga2 import NSGA2
from city.sim.simulator import TransitSimulator
from city.objectives import objective_values


def test_complete_pipeline_reproduces(monkeypatch):
    for k,v in dict(GA_ROUTES_PER_SOLUTION=3,GA_POPULATION=4,GA_GENERATIONS=1,SIM_DURATION_MINUTES=12.).items():
        monkeypatch.setattr(config,k,v)
    a=build_pipeline(city_seed=42); b=build_pipeline(city_seed=42)
    assert sorted(a.graph.edges)==sorted(b.graph.edges)
    assert a.stops==b.stops and a.routes==b.routes
    assert np.array_equal(a.stop_od,b.stop_od)
    assert np.all(np.isfinite(a.stop_od)) and np.all(a.stop_od>=0)
    assert len(a.routes)>=3
    for r in a.routes:
        assert all(a.graph.has_edge(u,v) for u,v in zip(r.road_path,r.road_path[1:]))
        assert sum(r.leg_distances)==pytest.approx(r.length)
    result=TransitSimulator(a.routes[:3],a.stop_distances,a.stop_od,[8.]*3).run()
    result.verify_conservation()
    assert np.all(np.isfinite(objective_values(result)))
    first=NSGA2(a.routes,a.stop_distances,a.stop_od,seed=9).run()
    second=NSGA2(b.routes,b.stop_distances,b.stop_od,seed=9).run()
    assert first.pareto_front==second.pareto_front
    assert first.hypervolume_history==second.hypervolume_history
    for c,o in first.pareto_front:
        assert len(c)==3 and all(0<=i<len(a.routes) for i in c)
        assert np.all(np.isfinite(o))
