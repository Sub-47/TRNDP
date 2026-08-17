"""
city/renderers/terrain_renderer.py

Rendering utilities for the synthetic city terrain.

This module only ever looks at a terrain class array - it has no idea
whether that array came from procedural noise, a raster PNG, or a
hybrid combination, and it never will.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import config
from city.enums.terrain_type import TerrainType


def _build_colormap() -> ListedColormap:
    """Build a ListedColormap ordered to match TerrainType integer values.

    Returns:
        A matplotlib colormap where index N corresponds to the terrain
        type whose enum value is N.
    """
    ordered_colors = [
        config.TERRAIN_COLORS[terrain_type.name] for terrain_type in TerrainType
    ]
    return ListedColormap(ordered_colors)


def render_terrain_map(
    terrain_classes: np.ndarray,
    output_path: str,
    title: str = config.FIGURE_TITLE,
    image_size: tuple[float, float] = config.IMAGE_SIZE,
    dpi: int = config.IMAGE_DPI,
    show: bool = True,
) -> str:
    """Render a colorized terrain classification map and save it to disk.

    Args:
        terrain_classes: Integer array of TerrainType values.
        output_path: Full file path to save the rendered image to.
        title: Figure title.
        image_size: Figure size in inches, as (width, height).
        dpi: Resolution of the saved figure.
        show: Whether to display the figure interactively.

    Returns:
        The path the figure was saved to.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cmap = _build_colormap()
    num_types = len(TerrainType)
    boundaries = np.arange(num_types + 1) - 0.5

    fig, ax = plt.subplots(figsize=image_size)
    ax.imshow(
        terrain_classes,
        cmap=cmap,
        vmin=boundaries[0],
        vmax=boundaries[-1],
        interpolation="nearest",
    )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=config.TERRAIN_COLORS[t.name])
        for t in TerrainType
    ]
    legend_labels = [t.name.title() for t in TerrainType]
    ax.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=num_types,
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)
    return output_path
