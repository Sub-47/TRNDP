"""
main.py

Entry point for the Spatial Urban Modeling Framework (SUMF).

Still deliberately minimal: World and MapManager do all the work.
Which data source is used is controlled entirely by config.MAP_SOURCE.
"""

from city.managers.map_manager import MapManager
from city.models.world import World


def main() -> None:
    map_manager = MapManager()
    world = World(map_manager)
    world.generate()
    world.visualize()
    world.save()
    print("Done.")


if __name__ == "__main__":
    main()
