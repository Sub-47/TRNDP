# Progress Log

## Status: Day 1 objectives complete, terrain + zone allocation bridge added

All five Day 1 goals from `DAY1_GOALS.md` are implemented and verified.
On top of that, a procedural terrain generator and a zone-allocation bridge
now connect terrain to the graph/gravity pipeline end to end.

## What was built

### 1. Project structure
- `data/` — input CSVs and generated output
- `graph/` — graph construction from raw data
- `gravity/` — gravity-model computation
- `city/` — procedural terrain generation (elevation, terrain classification,
  obstacles, population raster)
- `zoning/` — bridges terrain output into graph-ready nodes/edges
- `tests/` — automated verification
- `config.py` — shared configuration for terrain and zone allocation
- `main.py` — single-command entrypoint (hand-built sample data)
- `generate_terrain.py` — terrain-only entrypoint
- `generate_world.py` — full pipeline: terrain -> zones -> graph -> OD matrix

### 2. Sample data (`data/nodes.csv`, `data/edges.csv`)
A small 6-node road network used to develop and test the pipeline:
- **Nodes**: `node_id`, `population`, `jobs`
- **Edges**: `source_id`, `target_id`, `length_km`, `speed_kmh`

### 3. Graph loader (`graph/loader.py`)
`load_graph(nodes_path, edges_path)` reads both CSVs and builds an undirected
`networkx.Graph`:
- Nodes carry `population` and `jobs`.
- Edges carry `length_km`, `speed_kmh`, and a derived `travel_time_h`
  (`length_km / speed_kmh`).

### 4. Gravity model (`gravity/model.py`)
`compute_od_matrix(graph, beta=2.0, k=1.0)`:
- Computes shortest travel-time distance between every pair of nodes
  (`networkx.all_pairs_dijkstra_path_length`, weighted by `travel_time_h`).
- Applies the gravity equation for every origin/destination pair (excluding
  self-pairs):

  ```
  trips(i, j) = k * population(i) * jobs(j) / travel_time_h(i, j) ** beta
  ```

- Returns a `pandas.DataFrame` with columns `origin_id`, `destination_id`,
  `trips`.
- `beta` and `k` are overridable for future calibration.

### 5. Runnable workflow (`main.py`)
Single command from the repo root:

```
.venv/bin/python main.py
```

Loads the graph, computes the OD matrix, and writes it to
`data/od_matrix.csv`.

### 6. Validation (`tests/test_gravity_model.py`)
`.venv/bin/python -m unittest tests.test_gravity_model -v`

Confirms:
- The OD matrix is non-empty.
- It contains `origin_id`, `destination_id`, and `trips` columns.

Sanity-checked manually: the largest flow in the sample network is
`3 → 4` (high population at node 3, high job capacity at node 4, short
travel time between them) — matches expected gravity-model behavior.
Flows are directional/asymmetric by design, since population and jobs
pull independently in each direction.

### 7. Terrain generation (`city/`)
Procedural elevation via Perlin noise (`city/generators/terrain_generator.py`),
classified into `TerrainType` (Ocean/Beach/Plains/Hills/Mountains), plus an
`ObstacleMap` (ocean + mountains) and a placeholder `PopulationMap` raster.
Pluggable sources (`PROCEDURAL`, `RASTER`, `HYBRID`) registered in
`city/sources/`, orchestrated by `MapManager` and `World`
(`city/models/world.py`).

```
.venv/bin/python generate_terrain.py
```

Writes `output/terrain_map.png` and `output/metadata.json`.

### 8. Zone allocation bridge (`zoning/allocator.py`)
Converts a generated `World` into the exact node/edge schema `graph/loader.py`
expects, so the existing gravity pipeline runs unmodified on top of terrain
output:
- **Zone placement**: samples candidate zones on a grid across the terrain,
  skipping obstacle cells (ocean/mountains).
- **Population**: averages the population raster around each zone, scaled to
  a population count (`config.ZONE_POPULATION_SCALE`).
- **Jobs**: derived from population via a per-terrain-type multiplier
  (`config.ZONE_JOB_MULTIPLIERS`) — a placeholder, since the underlying
  population raster is itself just noise for now, not real demographic data.
- **Edges**: each zone connects to its `k` nearest zones
  (`config.ZONE_KNN`) by straight-line distance; speed is derived from the
  terrain type at each endpoint (`config.ZONE_TERRAIN_SPEED_KMH`).

```
.venv/bin/python generate_world.py
```

Runs the full chain — terrain generation -> zone allocation -> graph
loading -> OD matrix — and writes `data/generated_nodes.csv`,
`data/generated_edges.csv`, and `data/generated_od_matrix.csv`. These are
separate from `data/nodes.csv`/`data/edges.csv` (the hand-built sample used
by `main.py` and the test suite), so the original sample data is untouched.

## Not yet built (future stages)
- Calibration of `beta`/`k` against real or more detailed data.
- A real population/jobs data source to replace the placeholder noise raster.
- Obstacle-aware road routing (edges are currently straight-line KNN and
  don't check for water/mountains crossing between two zones).
- Visualization of the graph and OD flows overlaid on the terrain map.
- Handling of intrazonal trips (currently excluded — `i == j` pairs are
  skipped).
- Automated tests for `zoning/allocator.py` (currently verified manually).
