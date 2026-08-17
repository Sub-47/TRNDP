from city.managers.map_manager import MapManager
from city.models.world import World
from city.demand.zone_map import ZoneMap
from city.demand.distance_matrix import DistanceMatrix
from city.demand.gravity_model import GravityModel


def main() -> None:
    mgr = MapManager()
    world = World(mgr)
    world.generate()

    zones = ZoneMap.from_maps(world.population, world.obstacle)
    dist = DistanceMatrix.from_zone_map(zones)
    demand = GravityModel.from_zones_and_distance(zones, dist)

    print("Gravity demand matrix generated")
    print("shape:", demand.data.shape)
    print("self-trip value at [0,0]:", demand.data[0, 0])
    print("max value:", float(demand.data.max()))
    print("min value:", float(demand.data.min()))
    print("first 5x5 block:")
    print(demand.data[:5, :5])


if __name__ == "__main__":
    main()
