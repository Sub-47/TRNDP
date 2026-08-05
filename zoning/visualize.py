import os

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import config
from city.enums.terrain_type import TerrainType


def render_zone_map(world, nodes_df, edges_df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cmap = ListedColormap([config.TERRAIN_COLORS[t.name] for t in TerrainType])
    fig, ax = plt.subplots(figsize=config.IMAGE_SIZE)
    ax.imshow(
        world.terrain.data,
        cmap=cmap,
        vmin=-0.5,
        vmax=len(TerrainType) - 0.5,
        interpolation="nearest",
    )

    node_pos = nodes_df.set_index("node_id")[["x_km", "y_km"]]
    for edge in edges_df.itertuples(index=False):
        x1, y1 = node_pos.loc[edge.source_id]
        x2, y2 = node_pos.loc[edge.target_id]
        ax.plot([x1, x2], [y1, y2], color="red", linewidth=0.8, alpha=0.7, zorder=2)

    sizes = 20 + 200 * (nodes_df["population"] / nodes_df["population"].max())
    ax.scatter(
        nodes_df["x_km"], nodes_df["y_km"],
        s=sizes, c="white", edgecolors="black", linewidths=0.8, zorder=3,
    )

    ax.set_title("Zones and Roads over Terrain")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=config.IMAGE_DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path
