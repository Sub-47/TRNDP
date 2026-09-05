"""Validate baseline semantics, service boundaries and reproducible evidence."""
import numpy as np
import pytest
from city.experiments import choose_greedy
from city.sim.simulator import TransitSimulator
from city.routes.route_pool import Route
import config

def test_marginal_greedy_avoids_duplicate_coverage():
    routes=[Route([0,1],1),Route([0,1],1),Route([2,3],1)]
    od=np.zeros((4,4));od[0,1]=10;od[2,3]=9
    assert choose_greedy(routes,od,2)==frozenset([0,1])
    assert choose_greedy(routes,od,2,True)==frozenset([0,2])
    # Local demand must not affect either baseline's score.
    np.fill_diagonal(od,1e6)
    assert choose_greedy(routes,od,2,True)==frozenset([0,2])

@pytest.mark.parametrize('key,value',[('SIM_TIME_STEP_MINUTES',0),('SIM_DURATION_MINUTES',-1),
                                      ('BUS_SPEED_CELLS_PER_MINUTE',0),('BUS_CAPACITY',0),
                                      ('DWELL_MINUTES_PER_STOP',-1),('MAX_TRANSFERS',-1)])
def test_simulator_rejects_invalid_parameters(monkeypatch,key,value):
    monkeypatch.setattr(config,key,value)
    with pytest.raises(ValueError):TransitSimulator([],np.zeros((2,2)),np.zeros((2,2)),[])

@pytest.mark.parametrize('od',[np.array([[0,-1],[0,0]]),np.array([[0,np.nan],[0,0]]),np.zeros((2,3))])
def test_simulator_rejects_invalid_demand(od):
    with pytest.raises(ValueError):TransitSimulator([],np.zeros((2,2)),od,[])

def test_inconsistent_legacy_route_length_rejected():
    with pytest.raises(ValueError,match='length disagrees'):
        TransitSimulator([Route([0,1],10)],np.array([[0,2],[2,0]]),np.zeros((2,2)),[1])

def test_bus_route_input_order_does_not_change_outcomes(monkeypatch):
    monkeypatch.setattr(config,'SIM_DURATION_MINUTES',20.)
    dist=np.abs(np.arange(3)[:,None]-np.arange(3)[None,:])*2.
    routes=[Route([0,1],2),Route([1,2],2)]
    od=np.ones((3,3))*5
    a=TransitSimulator(routes,dist,od,[8,12]).run()
    b=TransitSimulator(routes[::-1],dist,od,[12,8]).run()
    assert a==b
