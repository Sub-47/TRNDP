import matplotlib.pyplot as plt

from city.managers.map_manager import MapManager
from city.models.world import World

world = World(MapManager())
world.generate()

pop = world.population.data          # the (256, 256) float array

plt.figure(figsize=(8, 8))
plt.imshow(pop, cmap="inferno")      # each cell's value -> a colour
plt.colorbar(label="population density")
plt.title(f"Population field (seed={world.map_manager._source.seed})")
plt.savefig("output/population_map.png", dpi=150)
print("saved output/population_map.png")
import numpy as np

terrain = world.terrain.data
obstacle = world.obstacle.data

# Grey terrain underneath, hot population on top, water/mountains masked out.
masked_pop = np.where(obstacle, np.nan, pop)   # nan cells render transparent

plt.figure(figsize=(8, 8))
plt.imshow(terrain, cmap="Greys", alpha=0.4)   # faint terrain backdrop
plt.imshow(masked_pop, cmap="inferno")         # population on top
plt.colorbar(label="population density")
plt.title("Population over terrain")
plt.savefig("output/population_over_terrain.png", dpi=150)