import numpy as np
from city.managers.map_manager import MapManager
from city.models.world import World
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