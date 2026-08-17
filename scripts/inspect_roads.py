"""
scripts/inspect_roads.py

Standalone diagnostic instrument for the road network, re-run after every
tuning change to the growth/graph parameters in config.py. Generates the
world, grows roads from every population centre (not just argmax), builds
the routing graph, and prints a diagnostic table. Asserts nothing.

Run: python scripts/inspect_roads.py
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.ndimage import distance_transform_edt

import config
from city.generators.population_generator import PopulationGenerator
from city.managers.map_manager import MapManager
from city.models.world import World
from city.renderers.terrain_renderer import _build_colormap
from city.roads.connector import connect_components
from city.roads.graph_builder import RoadGraphBuilder
from city.roads.loop_closer import close_loops
from city.roads.segment_grower import RoadNetworkGrower
from city.roads.stop_selector import StopSelector

COVERAGE_RADII = (2, 4, 8, 16)
OBSTACLE_CROSSING_STEP = 0.25
RASTER_STEP = 0.5


def rasterize_roads(segments: list, world_size: int) -> np.ndarray:
    """Marks every cell a road passes through (not just endpoints).

    Walks each segment in RASTER_STEP-unit steps and marks the rounded
    cell, so the coverage distance field sees continuous road cells
    rather than sparse endpoint dots.
    """
    road_mask = np.zeros((world_size, world_size), dtype=bool)
    for (x1, y1), (x2, y2) in segments:
        length = math.dist((x1, y1), (x2, y2))
        steps = max(int(math.ceil(length / RASTER_STEP)), 1)
        for i in range(steps + 1):
            t = i / steps
            x = x1 + (x2 - x1) * t
            y = y1 + (y2 - y1) * t
            row, col = round(y), round(x)
            if 0 <= row < world_size and 0 <= col < world_size:
                road_mask[row, col] = True
    return road_mask


def count_obstacle_crossings(segments: list, obstacle: np.ndarray) -> int:
    """Counts segments that pass through an obstacle cell at any point
    when walked at a finer resolution (0.25 units) than the grower's own
    unit-step legality check, exposing its sub-unit sampling gap."""
    world_size = obstacle.shape[0]
    crossings = 0
    for (x1, y1), (x2, y2) in segments:
        length = math.dist((x1, y1), (x2, y2))
        steps = max(int(math.ceil(length / OBSTACLE_CROSSING_STEP)), 1)
        hit = False
        for i in range(steps + 1):
            t = i / steps
            x = x1 + (x2 - x1) * t
            y = y1 + (y2 - y1) * t
            row, col = round(y), round(x)
            if 0 <= row < world_size and 0 <= col < world_size and obstacle[row, col]:
                hit = True
                break
        if hit:
            crossings += 1
    return crossings


def population_coverage(population: np.ndarray, road_mask: np.ndarray) -> dict[int, float]:
    """Percentage of total population mass within each radius (in cells)
    of the nearest road cell, weighted by population mass."""
    distance = distance_transform_edt(~road_mask)
    total_mass = population.sum()
    coverage = {}
    for radius in COVERAGE_RADII:
        mass_within = population[distance <= radius].sum()
        coverage[radius] = 100.0 * mass_within / total_mass if total_mass > 0 else 0.0
    return coverage


def render_roads(
    terrain_classes: np.ndarray,
    segments: list,
    output_path: str,
    stops: list | None = None,
) -> None:
    """Saves the usual terrain render with the road network overlaid, and
    stops drawn as distinct markers on top if given."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    from city.enums.terrain_type import TerrainType

    cmap = _build_colormap()
    num_types = len(TerrainType)
    boundaries = np.arange(num_types + 1) - 0.5

    fig, ax = plt.subplots(figsize=config.IMAGE_SIZE)
    ax.imshow(
        terrain_classes,
        cmap=cmap,
        vmin=boundaries[0],
        vmax=boundaries[-1],
        interpolation="nearest",
    )

    for (x1, y1), (x2, y2) in segments:
        ax.plot([x1, x2], [y1, y2], color="black", linewidth=0.6, zorder=2)

    if stops:
        stop_x = [x for x, _y in stops]
        stop_y = [y for _x, y in stops]
        ax.scatter(
            stop_x, stop_y, c="red", marker="o", s=40, edgecolors="white", linewidths=0.6, zorder=3
        )

    ax.set_title("Road Network")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=config.IMAGE_DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    map_manager = MapManager()
    world = World(map_manager)
    world.generate()

    population = world.population.data
    obstacle = world.obstacle.data
    world_size = population.shape[0]

    # PopulationGenerator already computed centres while producing
    # world.population; re-run it here (same terrain/obstacle/seed, so
    # deterministically identical output) just to reach its exposed
    # `.centres` attribute, which World's pipeline otherwise discards.
    population_generator = PopulationGenerator(
        terrain_classes=world.terrain.data,
        obstacle_mask=world.obstacle.data,
        world_size=world_size,
        seed=config.SEED,
    )
    population_generator.run()
    starts = [(float(col), float(row)) for row, col in population_generator.centres]

    grower = RoadNetworkGrower(population, obstacle, starts)
    segments = grower.grow()

    builder = RoadGraphBuilder(segments)
    graph = builder.build()

    components = list(nx.connected_components(graph))
    num_nodes = graph.number_of_nodes()
    largest_component = max((len(c) for c in components), default=0)
    largest_component_pct = 100.0 * largest_component / num_nodes if num_nodes else 0.0

    hit_cap = len(segments) >= config.ROAD_MAX_SEGMENTS

    connectors = connect_components(graph, obstacle)
    connector_length = sum(math.dist(p1, p2) for p1, p2 in connectors)

    all_segments = segments + connectors
    final_builder = RoadGraphBuilder(all_segments)
    final_graph = final_builder.build()

    final_components = list(nx.connected_components(final_graph))
    final_num_nodes = final_graph.number_of_nodes()
    final_largest = max((len(c) for c in final_components), default=0)
    final_largest_pct = 100.0 * final_largest / final_num_nodes if final_num_nodes else 0.0

    loop_edges = close_loops(final_graph, obstacle)
    loop_length = sum(math.dist(p1, p2) for p1, p2 in loop_edges)

    looped_segments = all_segments + loop_edges
    looped_builder = RoadGraphBuilder(looped_segments)
    looped_graph = looped_builder.build()

    looped_components = list(nx.connected_components(looped_graph))
    looped_num_nodes = looped_graph.number_of_nodes()
    looped_largest = max((len(c) for c in looped_components), default=0)
    looped_largest_pct = 100.0 * looped_largest / looped_num_nodes if looped_num_nodes else 0.0

    road_mask = rasterize_roads(looped_segments, world_size)
    coverage = population_coverage(population, road_mask)
    crossings = count_obstacle_crossings(looped_segments, obstacle)

    selector = StopSelector(looped_graph, population)
    stops = selector.select()

    stop_mask = np.zeros((world_size, world_size), dtype=bool)
    for x, y in stops:
        row, col = round(y), round(x)
        if 0 <= row < world_size and 0 <= col < world_size:
            stop_mask[row, col] = True
    stop_coverage = population_coverage(population, stop_mask)

    stop_nearest_neighbor = []
    for i, a in enumerate(stops):
        others = [b for j, b in enumerate(stops) if j != i]
        if others:
            stop_nearest_neighbor.append(min(math.dist(a, b) for b in others))
    mean_stop_spacing = (
        sum(stop_nearest_neighbor) / len(stop_nearest_neighbor) if stop_nearest_neighbor else 0.0
    )
    min_stop_spacing = min(stop_nearest_neighbor) if stop_nearest_neighbor else 0.0

    stops_are_graph_nodes = all(stop in looped_graph.nodes for stop in stops)

    def _component_index(node):
        for idx, component in enumerate(looped_components):
            if node in component:
                return idx
        return None

    stop_component_indices = {_component_index(stop) for stop in stops}
    stops_share_one_component = len(stop_component_indices) <= 1

    print("=" * 60)
    print("ROAD NETWORK DIAGNOSTIC")
    print("=" * 60)
    print(f"starts (population centres)     : {len(starts)}")
    for start in starts:
        print(f"  {start}")
    print(f"segments generated              : {len(segments)}")
    print(f"hit ROAD_MAX_SEGMENTS ({config.ROAD_MAX_SEGMENTS})     : {hit_cap}")
    print(f"redundant_discards               : {grower.redundant_discards}")
    print("-" * 60)
    pre_cycles = graph.number_of_edges() - num_nodes + len(components)
    print("pre-connector graph (grown segments only):")
    print(f"  graph nodes                   : {num_nodes}")
    print(f"  graph edges                   : {graph.number_of_edges()}")
    print(f"  connected components          : {len(components)}")
    print(f"  largest component             : {largest_component_pct:.2f}% of all nodes")
    print(f"  independent cycles            : {pre_cycles}  (edges - nodes + components)")
    for i, component in enumerate(sorted(components, key=len, reverse=True)):
        owning_starts = [s for s in starts if s in component]
        print(f"    component {i}: {len(component)} nodes, starts={owning_starts}")
    print("-" * 60)
    print(f"connector segments added        : {len(connectors)}")
    print(f"connector total length          : {connector_length:.2f} cells")
    print("-" * 60)
    post_cycles = final_graph.number_of_edges() - final_num_nodes + len(final_components)
    print("post-connector graph (grown + connector segments):")
    print(f"  graph nodes                   : {final_num_nodes}")
    print(f"  graph edges                   : {final_graph.number_of_edges()}")
    print(f"  connected components          : {len(final_components)}")
    print(f"  largest component             : {final_largest_pct:.2f}% of all nodes")
    print(f"  independent cycles            : {post_cycles}  (edges - nodes + components)")
    print("-" * 60)
    print(f"loop-closing edges added        : {len(loop_edges)}")
    print(f"loop-closing total length       : {loop_length:.2f} cells")
    print("-" * 60)
    looped_cycles = looped_graph.number_of_edges() - looped_num_nodes + len(looped_components)
    print("post-loop graph (grown + connector + loop-closing segments):")
    print(f"  graph nodes                   : {looped_num_nodes}")
    print(f"  graph edges                   : {looped_graph.number_of_edges()}")
    print(f"  connected components          : {len(looped_components)}")
    print(f"  largest component             : {looped_largest_pct:.2f}% of all nodes")
    print(f"  independent cycles            : {looped_cycles}  (edges - nodes + components)")
    print("-" * 60)
    print("population coverage (weighted by population mass):")
    for radius in COVERAGE_RADII:
        print(f"  within {radius:>2} cells of a road   : {coverage[radius]:.2f}%")
    print("-" * 60)
    print(f"obstacle crossings (0.25-unit walk) : {crossings} / {len(looped_segments)} segments")
    print(f"dropped_segments (graph builder)    : {looped_builder.dropped_segments}")
    print("-" * 60)
    print("stops:")
    print(f"  stops selected                 : {len(stops)} / {selector.target_count} target")
    print(f"  shortfall                      : {selector.shortfall}")
    print("  stop coverage (weighted by population mass):")
    for radius in (2, 4, 8):
        print(f"    within {radius:>2} cells of a stop   : {stop_coverage[radius]:.2f}%")
    print(f"  mean nearest-neighbor spacing  : {mean_stop_spacing:.2f} cells")
    print(f"  min nearest-neighbor spacing   : {min_stop_spacing:.2f} cells")
    print(f"  every stop is a graph node     : {stops_are_graph_nodes}")
    print(f"  all stops share one component  : {stops_share_one_component}")
    print("=" * 60)

    output_path = os.path.join(config.OUTPUT_DIR, "roads.png")
    render_roads(world.terrain.data, looped_segments, output_path, stops=stops)
    print(f"Saved render to {output_path}")


if __name__ == "__main__":
    main()
