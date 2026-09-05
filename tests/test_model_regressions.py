"""Independent examples for the corrected transport contract."""
import numpy as np
import networkx as nx
import pytest
import config
from city.routes.route_pool import Route, RoutePool
from city.routes.cluster import Cluster
from city.sim.simulator import TransitSimulator

@pytest.fixture(autouse=True)
def settings(monkeypatch):
    for key,value in dict(SIM_DURATION_MINUTES=4., SIM_TIME_STEP_MINUTES=1.,
                          BUS_SPEED_CELLS_PER_MINUTE=1., DWELL_MINUTES_PER_STOP=0.,
                          BUS_CAPACITY=50, MAX_TRANSFERS=2).items():
        monkeypatch.setattr(config,key,value)

def simulate(demand, routes=None, frequencies=None, distances=None):
    n=len(demand)
    if distances is None:
        distances=np.abs(np.arange(n)[:,None]-np.arange(n)[None,:])*2.
    if routes is None:
        routes=[Route([0,1],2.)]
    if frequencies is None:
        frequencies=[60.]*len(routes)
    r=TransitSimulator(routes,distances,np.array(demand,dtype=float),frequencies).run()
    assert r.total_created == pytest.approx(r.total_served+r.queued_passengers+r.abandoned_passengers+r.still_onboard)
    assert r.total_input_demand == pytest.approx(r.total_created+r.local_or_walking_demand)
    assert r.incomplete_passengers == pytest.approx(r.total_created-r.total_served)
    return r

def test_same_stop_is_local_not_unserved():
    r=simulate([[10,0],[0,0]])
    assert r.local_or_walking_demand == 10
    assert r.total_created == r.unserved_passengers == r.total_wait_time == 0

def test_hand_calculated_direct_service():
    # Four unit batches arrive at .5,1.5,2.5,3.5; board at 1,2,3.
    # Two-cell journeys arrive at 3 and 4; third remains onboard.
    r=simulate([[0,4],[0,0]])
    assert r.total_served == 2
    assert r.queued_passengers == r.still_onboard == 1
    assert r.total_wait_time == pytest.approx(2.)
    assert r.total_travel_time == pytest.approx(5.)
    assert r.total_bus_distance == pytest.approx(14.)
    assert r.incomplete_passengers == 2

def test_no_routes_conserves_demand():
    r=simulate([[0,4],[0,0]],routes=[])
    assert r.total_served == r.still_onboard == 0
    assert r.queued_passengers == 4
    assert r.total_wait_time == pytest.approx(8.)

def test_zero_demand_no_phantoms():
    r=simulate([[0,0],[0,0]])
    assert r.total_served == r.total_travel_time == r.total_wait_time == 0
    assert r.total_bus_distance > 0

def test_capacity_relief(monkeypatch):
    monkeypatch.setattr(config,'BUS_CAPACITY',1)
    low=simulate([[0,40],[0,0]])
    monkeypatch.setattr(config,'BUS_CAPACITY',50)
    high=simulate([[0,40],[0,0]])
    assert high.total_served > low.total_served
    assert high.incomplete_passengers < low.incomplete_passengers

def test_transfer_and_disconnected(monkeypatch):
    monkeypatch.setattr(config,'SIM_DURATION_MINUTES',8.)
    od=np.zeros((3,3)); od[0,2]=8
    routes=[Route([0,1],2.),Route([1,2],2.)]
    r=simulate(od,routes)
    assert r.total_served > 0
    assert r.served_transfers == pytest.approx(r.total_served)
    isolated=simulate(od,routes[:1])
    assert isolated.total_served == 0
    assert isolated.total_travel_time == 0

def test_detour_geometry_is_retained(monkeypatch):
    g=nx.Graph()
    a,b,x=(0.,0.),(2.,0.),(1.,2.)
    g.add_edge(a,b,weight=2.);g.add_edge(a,x,weight=3.);g.add_edge(x,b,weight=3.)
    routes=RoutePool(g,[Cluster([0],0,1),Cluster([1],1,1)],[a,b],routes_per_pair=2,min_stops=2).generate()
    assert sorted(r.length for r in routes)==[2.,6.]
    long=max(routes,key=lambda r:r.length)
    assert long.leg_distances == [6.]
    assert all(g.has_edge(u,v) for u,v in zip(long.road_path,long.road_path[1:]))
    monkeypatch.setattr(config,'SIM_DURATION_MINUTES',7.)
    r=simulate([[0,0],[0,0]],[long],[1.])
    assert r.total_bus_distance == pytest.approx(2*long.length)
    # Same horizon and demand: the detour cannot finish what the short route can.
    short=simulate([[0,7],[0,0]],[min(routes,key=lambda r:r.length)],[60.])
    detour=simulate([[0,7],[0,0]],[long],[60.])
    assert detour.total_served < short.total_served
    assert short.total_travel_time == pytest.approx(11.)
    assert detour.total_travel_time == pytest.approx(21.)

def test_nonintegral_horizon_conserves(monkeypatch):
    monkeypatch.setattr(config,'SIM_DURATION_MINUTES',4.3)
    r=simulate([[0,4],[0,0]])
    assert r.total_created == pytest.approx(4.)

def test_zero_frequency_is_no_service():
    r=simulate([[0,4],[0,0]],frequencies=[0.])
    assert r.total_served == r.total_bus_distance == r.still_onboard == 0
    assert r.queued_passengers == 4

def test_substep_journey_has_nonzero_time_and_peak(monkeypatch):
    monkeypatch.setattr(config,'SIM_DURATION_MINUTES',2.)
    r=simulate([[0,2],[0,0]],[Route([0,1],.1)],distances=np.array([[0,.1],[.1,0]]))
    assert r.total_served == 1
    assert r.total_travel_time == pytest.approx(.1)
    assert r.peak_load_factor == pytest.approx(1/50)

def test_transfer_can_initially_move_away(monkeypatch):
    monkeypatch.setattr(config,'SIM_DURATION_MINUTES',8.)
    od=np.zeros((3,3));od[0,2]=8
    # Destination is one cell away, but service requires an initial two-cell
    # journey in the opposite direction to a real interchange.
    dist=np.array([[0,2,1],[2,0,3],[1,3,0]],dtype=float)
    r=simulate(od,[Route([0,1],2.),Route([1,2],3.)],distances=dist)
    assert r.total_served == 3
    assert r.served_transfers == 3
