"""Deterministic fluid demand with chronological, event-resolved vehicle trips.

Demand is discretized into midpoint batches. Vehicles depart both termini at
fixed headways, complete a single direction, then retire. Passenger time is
integrated between events, including dwell. No real traffic calibration is implied.
"""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass
import numpy as np
import config
from city.routes.route_pool import Route


@dataclass
class SimResult:
    """Times are passenger-minutes, distance is cells, counts are fluid mass.

    total_created excludes local/walking demand. unserved_passengers is the
    legacy queued+abandoned diagnostic; incomplete_passengers additionally
    includes onboard passengers and is the GA's third objective.
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
    local_or_walking_demand: float = 0.
    total_input_demand: float = 0.
    queued_passengers: float = 0.
    abandoned_passengers: float = 0.
    served_transfers: float = 0.

    @property
    def incomplete_passengers(self) -> float:
        return self.unserved_passengers + self.still_onboard

    def verify_conservation(self) -> None:
        if not np.isclose(self.total_created, self.total_served + self.queued_passengers
                          + self.abandoned_passengers + self.still_onboard, rtol=1e-9, atol=1e-7):
            raise RuntimeError("transit passenger conservation violated")
        if not np.isclose(self.total_input_demand, self.total_created + self.local_or_walking_demand,
                          rtol=1e-9, atol=1e-7):
            raise RuntimeError("overall demand conservation violated")


class _Bus:
    def __init__(self, route_index: int, direction: int, stops: list[int], n: int, transfers: int):
        self.route_index = route_index
        self.direction = direction
        self.stops = stops
        self.position = 0
        # Indexed by alighting stop, destination, completed transfers.
        self.manifest = np.zeros((n, n, transfers + 1))
        self.manifest_wait = np.zeros_like(self.manifest)
        self.load = 0.
        self.last_event = 0.
        self.departure = 0.
        self.leg_length = 0.


class TransitSimulator:
    def __init__(self, routes: list[Route], stop_distances: np.ndarray,
                 demand: np.ndarray, frequencies: list[float]) -> None:
        self.stop_distances = np.asarray(stop_distances, dtype=float)
        self.demand = np.asarray(demand, dtype=float).copy()
        if self.demand.ndim != 2 or self.demand.shape[0] != self.demand.shape[1]:
            raise ValueError("demand must be a square matrix")
        self.n_stops = len(self.demand)
        if self.stop_distances.shape != self.demand.shape:
            raise ValueError("distance and demand shapes must match")
        if not np.all(np.isfinite(self.demand)) or np.any(self.demand < 0):
            raise ValueError("demand must be finite and nonnegative")
        if np.any(np.isnan(self.stop_distances)) or np.any(self.stop_distances < 0):
            raise ValueError("distances must be nonnegative, without NaN")
        if len(frequencies) != len(routes) or not np.all(np.isfinite(frequencies)) or np.any(np.asarray(frequencies) < 0):
            raise ValueError("frequencies must be finite nonnegative, one per route")
        self.duration = float(config.SIM_DURATION_MINUTES)
        self.step = float(config.SIM_TIME_STEP_MINUTES)
        self.speed = float(config.BUS_SPEED_CELLS_PER_MINUTE)
        self.dwell = float(config.DWELL_MINUTES_PER_STOP)
        self.capacity = float(config.BUS_CAPACITY)
        self.max_transfers = config.MAX_TRANSFERS
        if not all(np.isfinite(v) and v > 0 for v in (self.duration,self.step,self.speed,self.capacity)):
            raise ValueError("duration, time step, speed and capacity must be positive and finite")
        if not np.isfinite(self.dwell) or self.dwell < 0:
            raise ValueError("dwell must be nonnegative and finite")
        if not isinstance(self.max_transfers,int) or self.max_transfers < 0:
            raise ValueError("MAX_TRANSFERS must be a nonnegative integer")
        # Stable geometry ordering removes dependence on chromosome set iteration.
        order=sorted(range(len(routes)), key=lambda i:(tuple(routes[i].stops),tuple(routes[i].leg_distances),frequencies[i]))
        self.routes=[routes[i] for i in order]
        self.frequencies=[frequencies[i] for i in order]
        for route in self.routes:
            if any(not isinstance(s,(int,np.integer)) or s < 0 or s >= self.n_stops for s in route.stops):
                raise ValueError("route stop index outside the distance matrix")
        self.legs=[r.resolved_legs(self.stop_distances) for r in self.routes]
        self.targets=self._boarding_targets()

    def _boarding_targets(self):
        """Minimum-boardings paths on the operating service network.

        Each ride connects any two stops of a route. Every transfer strictly
        reduces remaining rides, preventing cycles. Among equal-ride options
        use actual in-vehicle distance plus network distance as a tie breaker.
        This policy is intentionally not timetable/queue-aware route choice.
        """
        n=self.n_stops
        hops=np.full((n,n),np.inf); np.fill_diagonal(hops,0.)
        for r,f in zip(self.routes,self.frequencies):
            if f > 0:
                hops[np.ix_(r.stops,r.stops)]=1.
        np.fill_diagonal(hops,0.)
        for k in range(n):
            hops=np.minimum(hops,hops[:,k,None]+hops[None,k,:])
        targets={}
        for ri,(r,legs) in enumerate(zip(self.routes,self.legs)):
            for direction in (1,-1):
                stops=r.stops if direction==1 else r.stops[::-1]
                directed_legs=legs if direction==1 else legs[::-1]
                cumulative=np.r_[0.,np.cumsum(directed_legs)]
                for pos,origin in enumerate(stops):
                    table=np.full((n,self.max_transfers+1),-1,dtype=int)
                    for dest in range(n):
                        needed=hops[origin,dest]
                        if not np.isfinite(needed) or needed <= 0:
                            continue
                        choices=[j for j in range(pos+1,len(stops))
                                 if 1+hops[stops[j],dest] == needed]
                        if not choices:
                            continue
                        chosen=min(choices,key=lambda j:(cumulative[j]-cumulative[pos]+self.stop_distances[stops[j],dest],stops[j]))
                        for t in range(self.max_transfers+1):
                            if needed <= self.max_transfers+1-t:
                                table[dest,t]=stops[chosen]
                    targets[ri,direction,pos]=table
        return targets

    @staticmethod
    def _dispatch_times(frequency: float, duration: float) -> np.ndarray:
        return np.arange(0.,duration,60./frequency) if frequency > 0 else np.array([])

    def run(self,on_step=None) -> SimResult:
        n=self.n_stops; T=self.max_transfers+1
        local=float(np.trace(self.demand)); od=self.demand.copy(); np.fill_diagonal(od,0.)
        queue=np.zeros((n,n,T)); waits=np.zeros_like(queue); queue_clock=np.zeros(n)
        events=[]; serial=itertools.count(); active={}; bus_id=itertools.count()
        def push(t,kind,data):
            if t <= self.duration+1e-10:
                heapq.heappush(events,(min(t,self.duration),next(serial),kind,data))
        def accrue(stop,now):
            waits[stop] += queue[stop]*(now-queue_clock[stop]); queue_clock[stop]=now
        for i,start in enumerate(np.arange(0.,self.duration,self.step)):
            width=min(self.step,self.duration-start)
            push(start+width/2,'demand',width/self.duration)
            if on_step is not None:
                push(start+width,'snapshot',i)
        for ri,(route,f) in enumerate(zip(self.routes,self.frequencies)):
            for t in self._dispatch_times(f,self.duration):
                for direction in (1,-1):
                    push(float(t),'dispatch',(ri,direction))
        served=served_wait=served_transfers=travel=distance=peak=transfers=0.
        while events:
            now=events[0][0]; group=[]
            while events and abs(events[0][0]-now) < 1e-10:
                group.append(heapq.heappop(events))
            arriving=[]; snapshots=[]
            # All alighting at a timestamp precedes any boarding at that timestamp.
            for _,_,kind,data in group:
                if kind=='demand':
                    for stop in range(n): accrue(stop,now)
                    queue[:,:,0] += od*data
                elif kind=='snapshot': snapshots.append(data)
                else:
                    if kind=='dispatch':
                        ri,direction=data; stops=self.routes[ri].stops
                        bus=_Bus(ri,direction,stops if direction==1 else stops[::-1],n,self.max_transfers)
                        ident=next(bus_id); active[ident]=bus; bus.last_event=now
                    else:
                        ident=data;bus=active[ident]
                        travel += bus.load*(now-bus.last_event)
                        distance += bus.leg_length
                        bus.last_event=now
                    stop=bus.stops[bus.position]; accrue(stop,now)
                    counts=bus.manifest[stop].copy(); history=bus.manifest_wait[stop].copy()
                    served += counts[stop].sum(); served_wait += history[stop].sum()
                    served_transfers += float(counts[stop] @ np.arange(T))
                    counts[stop]=0.;history[stop]=0.
                    # Only valid preplanned transfers can be present here.
                    if counts[:,-1].sum() > 1e-8: raise RuntimeError('transfer budget exceeded')
                    queue[stop,:,1:] += counts[:,:-1]; waits[stop,:,1:] += history[:,:-1]
                    transfers += counts.sum()
                    bus.manifest[stop]=0.;bus.manifest_wait[stop]=0.
                    bus.load=float(bus.manifest.sum())
                    if bus.position==len(bus.stops)-1:
                        if bus.load > 1e-8: raise RuntimeError('passengers left on retiring vehicle')
                        del active[ident]
                    else: arriving.append((ident,bus,stop))
            for ident,bus,stop in arriving:
                table=self.targets[bus.route_index,bus.direction,bus.position]
                eligible=table>=0
                available=float(queue[stop][eligible].sum())
                fraction=min(1.,max(0.,self.capacity-bus.load)/available) if available else 0.
                # Proportional allocation across eligible destinations avoids index priority.
                boarding=queue[stop]*eligible*fraction; history=waits[stop]*eligible*fraction
                queue[stop] -= boarding; waits[stop] -= history
                destinations,ts=np.nonzero(boarding)
                bus.manifest[table[destinations,ts],destinations,ts] += boarding[destinations,ts]
                bus.manifest_wait[table[destinations,ts],destinations,ts] += history[destinations,ts]
                bus.load=float(bus.manifest.sum()); peak=max(peak,bus.load)
                legs=self.legs[bus.route_index]
                if bus.direction==-1: legs=legs[::-1]
                bus.leg_length=float(legs[bus.position]); bus.departure=now+self.dwell
                bus.position+=1
                push(bus.departure+bus.leg_length/self.speed,'arrive',ident)
            for index in snapshots:
                on_step(index,list(active.values()))
        for stop in range(n): accrue(stop,self.duration)
        onboard=onboard_wait=0.
        for bus in active.values():
            travel += bus.load*(self.duration-bus.last_event)
            distance += min(bus.leg_length,max(0.,self.duration-bus.departure)*self.speed)
            onboard += bus.load; onboard_wait += float(bus.manifest_wait.sum())
        queued=float(queue.sum()); queued_wait=float(waits.sum())
        result=SimResult(total_wait_time=float(served_wait+queued_wait+onboard_wait),
            total_travel_time=float(travel),total_transfers=float(transfers),unserved_passengers=queued,
            total_bus_distance=float(distance),peak_load_factor=float(peak/self.capacity),
            total_created=float(od.sum()),total_served=float(served),still_onboard=float(onboard),
            served_wait_time=float(served_wait),unserved_wait_time=queued_wait,
            still_onboard_wait_time=float(onboard_wait),local_or_walking_demand=local,
            total_input_demand=float(self.demand.sum()),queued_passengers=queued,
            served_transfers=float(served_transfers))
        result.verify_conservation()
        return result
