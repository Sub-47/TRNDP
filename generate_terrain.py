from city.managers.map_manager import MapManager
from city.models.world import World


def main():
    world = World(MapManager())
    world.generate()
    world.visualize(show=False)
    world.save()
    print("Done.")


if __name__ == "__main__":
    main()
