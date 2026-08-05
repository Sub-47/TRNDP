import itertools

import numpy as np
import pandas as pd

import config
from city.enums.terrain_type import TerrainType


def _zone_population(population_map, row, col, world_size, radius):
    row_lo, row_hi = max(0, row - radius), min(world_size, row + radius + 1)
    col_lo, col_hi = max(0, col - radius), min(world_size, col + radius + 1)
    local_mean = population_map[row_lo:row_hi, col_lo:col_hi].mean()
    return max(config.ZONE_MIN_POPULATION, int(local_mean * config.ZONE_POPULATION_SCALE))


def _zone_jobs(population, terrain_type):
    multiplier = config.ZONE_JOB_MULTIPLIERS.get(terrain_type.name, 0.3)
    return max(1, int(population * multiplier))


def _place_zones(world):
    terrain = world.terrain.data
    obstacle = world.obstacle.data
    population = world.population.data
    world_size = world.terrain.world_size
    spacing = config.ZONE_GRID_SPACING
    radius = config.ZONE_SAMPLE_RADIUS

    zones = []
    zone_id = itertools.count(1)
    for row in range(spacing // 2, world_size, spacing):
        for col in range(spacing // 2, world_size, spacing):
            if obstacle[row, col]:
                continue

            terrain_type = TerrainType(terrain[row, col])
            zone_population = _zone_population(population, row, col, world_size, radius)
            zones.append(
                {
                    "node_id": next(zone_id),
                    "row": row,
                    "col": col,
                    "terrain_type": terrain_type.name,
                    "population": zone_population,
                    "jobs": _zone_jobs(zone_population, terrain_type),
                }
            )

    return pd.DataFrame(zones)


def _build_edges(zones_df, k_nearest=config.ZONE_KNN, cell_size_km=config.ZONE_CELL_SIZE_KM):
    coords = zones_df[["row", "col"]].to_numpy(dtype=np.float64)
    node_ids = zones_df["node_id"].to_numpy()
    terrain_speed = zones_df["terrain_type"].map(config.ZONE_TERRAIN_SPEED_KMH).to_numpy()

    edges = {}
    for i in range(len(zones_df)):
        distances = np.linalg.norm(coords - coords[i], axis=1)
        distances[i] = np.inf
        nearest = np.argsort(distances)[:k_nearest]

        for j in nearest:
            key = frozenset((node_ids[i], int(node_ids[j])))
            if key in edges:
                continue
            length_km = distances[j] * cell_size_km
            speed_kmh = (terrain_speed[i] + terrain_speed[j]) / 2
            edges[key] = {
                "source_id": node_ids[i],
                "target_id": int(node_ids[j]),
                "length_km": round(length_km, 3),
                "speed_kmh": speed_kmh,
            }

    return pd.DataFrame(edges.values())


def allocate_zones(world):
    zones_df = _place_zones(world)
    if zones_df.empty:
        raise RuntimeError("No zones could be placed; terrain may be entirely ocean/mountains.")

    edges_df = _build_edges(zones_df)

    nodes_df = zones_df.rename(columns={"row": "y_km", "col": "x_km"})[
        ["node_id", "population", "jobs", "x_km", "y_km", "terrain_type"]
    ]
    return nodes_df, edges_df


def write_zone_csvs(nodes_df, edges_df, nodes_path, edges_path):
    nodes_df.to_csv(nodes_path, index=False)
    edges_df.to_csv(edges_path, index=False)
