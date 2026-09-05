"""Mathematical and exhaustive checks independent of evolutionary success claims."""
import itertools
import numpy as np
import pytest
import config
from city.ga.nsga2 import NSGA2, fast_non_dominated_sort, crowding_distance
from city.routes.route_pool import Route
from city.objectives import objective_values
from city.sim.simulator import TransitSimulator


def hv(points,ref=(1.,1.,1.)):
    ga=object.__new__(NSGA2);ga.reference_point=np.array(ref)
    return ga._hypervolume(np.array(points))

def test_analytic_hypervolumes():
    assert hv([[.5,.5,.5]])==pytest.approx(.125)
    assert hv([[.2,.8,.5],[.8,.2,.5]])==pytest.approx(.14)
    assert hv([[.2,.8,.5],[.8,.2,.5],[.8,.8,.8],[.2,.8,.5]])==pytest.approx(.14)
    assert hv([[1.1,.1,.1]])==0
    assert hv([])==0
    assert hv([[.5,.5,0.]])==pytest.approx(.25)

def test_population_hv_can_decrease_archive_cannot():
    ga=object.__new__(NSGA2);ga.population_size=4
    rng=np.random.default_rng(7)
    for _ in range(10): values=rng.uniform(.05,.95,(8,3))
    ga.reference_point=values[:4].max(axis=0)
    _,selected=ga._select_next_generation([frozenset([i]) for i in range(8)],values)
    old=ga._hypervolume(values[:4]); new=ga._hypervolume(selected)
    assert new < old
    assert ga._hypervolume(values)>=old

def test_constant_objective_does_not_invent_boundary_points():
    assert np.array_equal(crowding_distance(np.ones((4,3)),list(range(4))),np.zeros(4))

def test_tiny_exhaustive_pareto(monkeypatch):
    for key,value in dict(GA_ROUTES_PER_SOLUTION=2,GA_POPULATION=6,GA_GENERATIONS=3,
                          SIM_DURATION_MINUTES=12.,BUS_CAPACITY=4).items():
        monkeypatch.setattr(config,key,value)
    dist=np.abs(np.arange(4)[:,None]-np.arange(4)[None,:])*2.
    pool=[Route([0,1],2.),Route([1,2],2.),Route([2,3],2.),Route([0,1,2,3],6.)]
    od=np.ones((4,4))*4; np.fill_diagonal(od,0)
    exact={frozenset(c):objective_values(TransitSimulator([pool[i] for i in c],dist,od,[8.,8.]).run())
           for c in itertools.combinations(range(4),2)}
    # Independent direct pairwise definition, not the implementation's sorter.
    true={c:o for c,o in exact.items() if not any(
        all(a<=b for a,b in zip(other,o)) and any(a<b for a,b in zip(other,o)) for other in exact.values())}
    result=NSGA2(pool,dist,od,seed=7).run()
    assert dict(result.pareto_front)==true
    assert result.evaluations==6
    assert result.hypervolume_history[-1]==pytest.approx(hv(list(true.values()),result.reference_point))
    assert np.all(np.diff(result.hypervolume_history)>=-1e-12)

def test_sort_matches_independent_definition():
    rng=np.random.default_rng(13);v=rng.random((40,3))
    expected={i for i,o in enumerate(v) if not any(np.all(p<=o) and np.any(p<o) for p in v)}
    assert set(fast_non_dominated_sort(v)[0])==expected

@pytest.mark.parametrize('values', [[[np.nan,1,2]],[[np.inf,1,2]]])
def test_sort_rejects_nonfinite(values):
    with pytest.raises(ValueError):fast_non_dominated_sort(values)
