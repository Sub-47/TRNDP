import matplotlib.pyplot as plt
import numpy as np

from city.managers.map_manager import MapManager
from city.models.world import World
from city.roads.segment_grower import RoadNetworkGrower
import config

world = World(MapManager())
world.generate()

pop = world.population.data
obs = world.obstacle.data

start = np.unravel_index(np.argmax(pop), pop.shape)   # (row, col) = (y, x)
start_xy = (float(start[1]), float(start[0]))         # grower wants (x, y)

grower = RoadNetworkGrower(pop, obs, start_xy)
segments = grower.grow()
print(f"{len(segments)} segments from start {start_xy}")

plt.figure(figsize=(9, 9))
plt.imshow(np.where(obs, np.nan, pop), cmap="inferno")
for (x1, y1), (x2, y2) in segments:
    plt.plot([x1, x2], [y1, y2], color="white", linewidth=0.7)
plt.plot(*start_xy, "co", markersize=6)
plt.title(f"Roads — {config.STREET_PATTERN}, {len(segments)} segments")
plt.savefig("output/roads.png", dpi=150)