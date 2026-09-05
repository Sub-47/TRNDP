"""Canonical city-to-route pipeline shared by diagnostics and experiments."""
from __future__ import annotations
from dataclasses import dataclass
import networkx as nx
import numpy as np
import config
from city.models.world import World
from city.managers.map_manager import MapManager
from city.generators.population_generator import PopulationGenerator
from city.roads.segment_grower import RoadNetworkGrower
from city.roads.graph_builder import RoadGraphBuilder
from city.roads.connector import connect_components
from city.roads.loop_closer import close_loops
from city.roads.stop_selector import StopSelector
from city.demand.zone_map import ZoneMap
from city.demand.distance_matrix import DistanceMatrix
from city.demand.gravity_model import GravityModel
from city.routes.cluster import build_stop_distance_matrix, assign_zones_to_stops, DemandClusterer
from city.routes.route_pool import RoutePool, Route

@dataclass
class RoadStages:
    segments: list
    grown_graph: nx.Graph
    connectors: list
    connected_graph: nx.Graph
    loops: list
    graph: nx.Graph
    starts: list
    grower: RoadNetworkGrower

@dataclass
class Pipeline:
    world: World
    graph: nx.Graph
    stops: list
    zones: ZoneMap
    demand: GravityModel
    stop_distances: np.ndarray
    stop_od: np.ndarray
    routes: list[Route]
    stages: RoadStages
    clusters: list


def build_roads(world: World, city_seed: int | None = None, pattern: str | None = None) -> RoadStages:
    seed=config.SEED if city_seed is None else city_seed
    population=world.population.data; obstacle=world.obstacle.data
    generator=PopulationGenerator(world.terrain.data,obstacle,world_size=len(population),seed=seed)
    generator.run()
    starts=[(float(c),float(r)) for r,c in generator.centres]
    grower=RoadNetworkGrower(population,obstacle,starts,pattern=pattern or config.STREET_PATTERN,seed=seed)
    segments=grower.grow(); grown=RoadGraphBuilder(segments).build()
    connectors=connect_components(grown,obstacle)
    connected=RoadGraphBuilder(segments+connectors).build()
    loops=close_loops(connected,obstacle)
    graph=RoadGraphBuilder(segments+connectors+loops).build()
    return RoadStages(segments,grown,connectors,connected,loops,graph,starts,grower)


def build_stop_od_matrix(zones: ZoneMap,demand: GravityModel,stops: list) -> np.ndarray:
    """Retain diagonal mass: the simulator classifies it explicitly as local."""
    if not stops: raise ValueError('cannot aggregate demand without stops')
    assignment=assign_zones_to_stops(zones,stops); n=len(stops)
    index=assignment[:,None]*n+assignment[None,:]
    return np.bincount(index.ravel(),weights=demand.data.ravel(),minlength=n*n).reshape(n,n)


def build_pipeline(city_seed: int | None = None, pattern: str | None = None,
                   period: str = 'BASE', weighted: bool = True, demand_metric: str = 'graph') -> Pipeline:
    seed=config.SEED if city_seed is None else city_seed
    world=World(MapManager(map_source_name=config.MAP_SOURCE,world_size=config.WORLD_SIZE,seed=seed));world.generate()
    stages=build_roads(world,seed,pattern); graph=stages.graph
    stops=StopSelector(graph,world.population.data,target_count=config.TARGET_STOP_COUNT,
                       min_spacing=config.STOP_MIN_SPACING).select()
    if len(stops)<2: raise ValueError('city produced fewer than two stops; record this instance as infeasible')
    zones=ZoneMap.from_maps(world.population,world.obstacle,zone_size=config.ZONE_SIZE)
    if demand_metric not in ('graph','euclidean'): raise ValueError('unknown demand metric')
    dist=DistanceMatrix.from_graph(zones,graph) if demand_metric=='graph' else DistanceMatrix.from_zone_map(zones)
    demand=(GravityModel.from_zones_and_distance(zones,dist) if period=='BASE'
            else GravityModel.for_period(zones,dist,period))
    stop_od=build_stop_od_matrix(zones,demand,stops)
    transit=stop_od.copy(); np.fill_diagonal(transit,0.)
    weights=transit.sum(axis=0)+transit.sum(axis=1) if weighted else np.ones(len(stops))
    stop_distances=build_stop_distance_matrix(graph,stops)
    clusters=DemandClusterer(stops,stop_distances,weights,eps=config.DBSCAN_EPS,min_samples=config.DBSCAN_MIN_SAMPLES).cluster()
    routes=RoutePool(graph,clusters,stops,routes_per_pair=config.ROUTES_PER_PAIR,
                     min_stops=config.ROUTE_MIN_STOPS,max_length=config.ROUTE_MAX_LENGTH).generate()
    for route in routes: route.resolved_legs(stop_distances)
    return Pipeline(world,graph,stops,zones,demand,stop_distances,stop_od,routes,stages,clusters)
