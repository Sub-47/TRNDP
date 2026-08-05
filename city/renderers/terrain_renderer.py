import os

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import config
from city.enums.terrain_type import TerrainType


def _build_colormap():
    return ListedColormap([config.TERRAIN_COLORS[t.name] for t in TerrainType])


def render_terrain_map(
    terrain_classes,
    output_path,
    title=config.FIGURE_TITLE,
    image_size=config.IMAGE_SIZE,
    dpi=config.IMAGE_DPI,
    show=True,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=image_size)
    ax.imshow(
        terrain_classes,
        cmap=_build_colormap(),
        vmin=-0.5,
        vmax=len(TerrainType) - 0.5,
        interpolation="nearest",
    )
    ax.set_title(title)
    ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=config.TERRAIN_COLORS[t.name]) for t in TerrainType]
    ax.legend(
        handles,
        [t.name.title() for t in TerrainType],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(TerrainType),
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return output_path
