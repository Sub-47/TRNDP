from city.managers.map_manager import MapManager
from city.models.world import World
from city.demand.zone_map import ZoneMap
from city.demand.distance_matrix import DistanceMatrix
from city.demand.gravity_model import GravityModel

mgr = MapManager()
world = World(mgr)
world.generate()

zones = ZoneMap.from_maps(world.population, world.obstacle)
dist = DistanceMatrix.from_zone_map(zones)
demand = GravityModel.from_zones_and_distance(zones, dist)

print(demand.data.shape)
print(demand.data[:5, :5])  # show the first 5x5 block