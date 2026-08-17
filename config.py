"""
config.py

Central configuration for the synthetic city terrain generator.

Every tunable parameter used anywhere in the codebase lives here. No module
outside this file should hardcode a numeric constant that affects world
generation, classification, or rendering. This keeps the simulator
reproducible: a given SEED plus this file fully determines the output.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# World / grid parameters
# --------------------------------------------------------------------------

# Number of cells along each axis of the (square) world grid. Kept small
# (128) to match the prototype's target scale; every other cell-based
# constant below is a ratio against this, not an absolute.
WORLD_SIZE: int = 128

# Random seed driving Perlin noise generation. Same seed + same config
# always produces an identical world.
SEED: int = 42

# --------------------------------------------------------------------------
# Elevation / sea level
# --------------------------------------------------------------------------

# Normalized elevation (0.0-1.0) below which a cell is classified as ocean.
SEA_LEVEL: float = 0.30

# --------------------------------------------------------------------------
# Perlin noise parameters (passed to the `noise` library)
# --------------------------------------------------------------------------

# Larger NOISE_SCALE -> smoother, more gradual terrain features. Sampled
# roughly at (col / NOISE_SCALE, row / NOISE_SCALE), so this must scale
# with WORLD_SIZE to keep feature size proportional to the map: halved
# alongside WORLD_SIZE (256 -> 128) so terrain doesn't come out twice as
# coarse relative to the smaller grid and merge the water into one sea.
NOISE_SCALE: float = 50.0

# Number of noise layers summed together. More octaves add finer detail.
OCTAVES: int = 6

# Amplitude falloff per octave.
PERSISTENCE: float = 0.5

# Frequency growth per octave.
LACUNARITY: float = 2.0

# --------------------------------------------------------------------------
# Terrain classification thresholds
# --------------------------------------------------------------------------
# All thresholds are normalized elevation values (0.0-1.0). Elevation below
# SEA_LEVEL is always OCEAN regardless of these thresholds. Above SEA_LEVEL,
# a cell's class is the first threshold (in ascending order) that its
# elevation does not exceed.
#
# Anything above the highest listed threshold is classified as the final
# terrain type (MOUNTAINS).

TERRAIN_THRESHOLDS: dict[str, float] = {
    "BEACH": 0.45,
    "PLAINS": 0.65,
    "HILLS": 0.80,
    # MOUNTAINS has no upper bound: it captures everything above HILLS.
}

# --------------------------------------------------------------------------
# Visualization
# --------------------------------------------------------------------------

# Matplotlib figure size, in inches, as (width, height).
IMAGE_SIZE: tuple[float, float] = (10.0, 10.0)

# Resolution of the saved figure.
IMAGE_DPI: int = 150

# Colors used to render each terrain class. Keys must match TerrainType
# member names in city/terrain.py.
TERRAIN_COLORS: dict[str, str] = {
    "OCEAN": "#1f77b4",
    "BEACH": "#e8d9a0",
    "PLAINS": "#4d9221",
    "HILLS": "#6b6b1f",
    "MOUNTAINS": "#7f7f7f",
}

FIGURE_TITLE: str = "Procedurally Generated Terrain"

# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

OUTPUT_DIR: str = "output"
OUTPUT_FILENAME: str = "terrain_map.png"
METADATA_FILENAME: str = "metadata.json"

# --------------------------------------------------------------------------
# Map source (Milestone 1.5)
# --------------------------------------------------------------------------
# Controls where World's spatial layers come from. Changing this value is
# the *only* change needed to switch data sources; no other code should
# require modification.
#
#   "PROCEDURAL" - everything generated from Perlin noise (default; no
#                  input files required).
#   "RASTER"     - everything loaded from assets/input/*.png; raises a
#                  clear error if a required file is missing.
#   "HYBRID"     - uses raster input per-map where available, generates
#                  the rest procedurally.
MAP_SOURCE: str = "PROCEDURAL"

# Directory RasterSource / HybridSource look in for elevation.png and
# population.png.
ASSET_INPUT_DIR: str = "assets/input"

# Seed offset used by ProceduralSource to derive population-generation
# randomness (city-centre placement) from config.SEED, kept apart from
# the elevation seed so the two layers don't end up correlated.
POPULATION_SEED_OFFSET: int = 9973

# TerrainType member names treated as non-traversable when deriving
# ObstacleMap from TerrainMap.
OBSTACLE_TERRAIN_TYPES: list[str] = ["OCEAN", "MOUNTAINS"]

# Recorded in output/metadata.json for reproducibility.
GENERATOR_VERSION: str = "1.5.0"

# Side length, in cells, of a demand zone. Halved alongside WORLD_SIZE
# (16 -> 8) so the zone grid keeps the same number of zones per axis.
ZONE_SIZE: int = 8
MASK_POPULATION_ON_OBSTACLES: bool = True

# Gravity model parameters
GRAVITY_K: float = 1.0
GRAVITY_BETA: float = 2.0

# Average trips generated per person per day (to/from work, school,
# errands, etc.) - the standard rule-of-thumb unit trip generation is
# measured in. A placeholder for this synthetic city, not calibrated to
# any real one; used to scale each origin zone's row to an actual trip
# count instead of an unnormalised gravity score.
TRIPS_PER_PERSON: float = 2.0

# Multipliers applied to the base (singly-constrained) trip matrix to
# approximate time-of-day demand variation. These are rough
# approximations drawn from urban transport literature - peaks running
# well above the daily average, off-peak well below it - not values
# calibrated to this synthetic city.
DIURNAL_FACTORS: dict[str, float] = {
    "MORNING_PEAK": 1.6,
    "OFF_PEAK": 0.6,
    "EVENING_PEAK": 1.4,
}

# --------------------------------------------------------------------------
# Population generation (city-centre model)
# --------------------------------------------------------------------------
# Population density is modeled as a sum of exponential-decay fields
# radiating from a small number of city centres (one primary CBD plus
# sub-centres), scaled by terrain suitability and zeroed on obstacles.
# This gives the field real spatial structure - a downtown core and
# secondary nodes - for a later DBSCAN clustering phase to find,
# unlike the raw Perlin-noise placeholder it replaces.

# Total number of city centres placed, including the primary CBD.
NUM_CITY_CENTRES: int = 5

# Minimum allowed distance, in grid cells, between any two centres
# (including the primary CBD). Prevents centres from landing on top
# of each other. Halved alongside WORLD_SIZE (30 -> 15) to keep the
# same proportion of the map between centres.
CENTRE_MIN_SEPARATION: int = 15

# Maximum allowed distance, in grid cells, between a sub-centre and
# the primary CBD. Without this bound, sub-centres are picked from the
# entire habitable area and routinely land 50+ cells from the CBD on
# a 128-cell world - producing several disconnected mini-settlements
# rather than one polycentric city with a downtown and nearby nodes.
# If no habitable cell within this radius satisfies
# CENTRE_MIN_SEPARATION, the bound is relaxed for that pick rather
# than failing generation outright.
SUBCENTRE_MAX_DISTANCE: float = 45.0

# Distance, in grid cells, over which a centre's contribution decays
# by a factor of e. Larger values spread and merge centres into broad
# density gradients; smaller values produce sharp, isolated peaks that
# won't form contiguous high-density zones.
CENTRE_DECAY_LENGTH: float = 45.0

# Weight applied to every centre after the first. The primary CBD
# always has weight 1.0; sub-centres are secondary nodes with less
# pull.
SUBCENTRE_WEIGHT: float = 0.5

# --------------------------------------------------------------------------
# Road graph construction
# --------------------------------------------------------------------------

# Grid size (in the same units as segment coordinates) that coordinates are
# snapped to before being used as graph node keys. Two points within this
# tolerance of the same grid cell collapse into a single intersection node.
ROAD_SNAP_TOLERANCE: float = 0.5

# Multiplier applied to density based on each cell's terrain class:
# flat land is the easiest to build a city on, beaches and hills are
# less buildable/desirable, and water/mountains are uninhabitable.
# Keys must match TerrainType member names in city/enums/terrain_type.py.
TERRAIN_SUITABILITY: dict[str, float] = {
    "OCEAN": 0.0,
    "BEACH": 0.7,
    "PLAINS": 1.0,
    "HILLS": 0.3,
    "MOUNTAINS": 0.0,
}

# --------------------------------------------------------------------------
# Road segment growth (Parish & Muller style L-system growth)
# --------------------------------------------------------------------------

STREET_PATTERN: str = "RADIAL"  # "GRID" or "RADIAL"
GRID_BASE_ANGLE_DEG: float = 0.0
RADIAL_SPOKE_BIAS: float = 0.0
# Degrees of extra tolerance given to the outward spoke when snapping.
# Ring roads are geometrically dense near the centre and consume the
# segment budget; biasing outward trades local rings for reach.
# MUST stay well above ROAD_SNAP_TOLERANCE: ROAD_SNAP_TOLERANCE is 0.5, and
# segments shorter than roughly half that collapse to zero length in the
# graph builder. Do not lower it. Halved alongside WORLD_SIZE (8.0 -> 4.0)
# to keep segment length proportional to the map; 4.0 / 0.5 = 8, still
# well clear of the collapse threshold.
ROAD_SEGMENT_LENGTH: float = 4.0
# Restored to the measured optimum (coverage@4: 19.78% at 4 branches vs.
# 15.87% at 8): more initial branches from a single origin just meant more
# rays fighting over the same nearby population, not more reach.
ROAD_INITIAL_BRANCHES: int = 4
# Halved alongside WORLD_SIZE (1500 -> 800): a smaller map needs
# proportionally fewer segments to cover it.
ROAD_MAX_SEGMENTS: int = 800
# 128 / 4.0 = 32 hops to cross the map, so 60 remains ample headroom.
ROAD_MAX_DEPTH: int = 60
ROAD_RAY_COUNT: int = 9
ROAD_RAY_SPREAD_DEG: float = 60.0
ROAD_RAY_SAMPLES: int = 8
ROAD_ROTATION_ATTEMPTS: int = 6
ROAD_ROTATION_STEP_DEG: float = 15.0
ROAD_BRANCH_PROBABILITY: float = 0.45
# Restored to the measured optimum (coverage@4: 19.78% at 0.05 vs. 15.87%
# at 0.02): letting roads wander through near-zero-population gaps just
# burned segment budget away from population, it didn't help reach it.
ROAD_MIN_POPULATION_SCORE: float = 0.05

ROAD_CONNECTOR_MAX_LENGTH: float = 80.0
# Longest connector road that will be built between two otherwise
# disconnected sub-networks. Above this, the fragment is treated as
# genuinely unreachable rather than bridged by an implausible road.

ROAD_MIN_SEPARATION: float = 2.0
# Cells of clearance required between a new road and existing road.
# Below this, parallel roads stack and their sub-edges collapse in the
# graph builder. Raise for sparser, more distinct streets.

ROAD_PROXIMITY_IGNORE: float = 1.0
# Distance from a segment's start exempted from the proximity check, since
# every segment legitimately begins on an existing road end. Kept small:
# ancestry exemption (parent_cells, see segment_grower.py) is now the
# primary mechanism for that, since it is direction-aware and exact; this
# is only a backstop for road ends with no parent (the initial branches).

ROAD_MIN_JUNCTION_LENGTH: float = 1.0
# Shortest connecting segment worth emitting when a road meets existing
# road. Below this the sub-edge collapses at ROAD_SNAP_TOLERANCE anyway.

LOOP_MAX_GAP: float = 12.0
# Nodes farther apart than this are never considered for a loop-closing
# edge, regardless of detour ratio - keeps candidate edges plausible as
# actual streets, not long shortcuts across the map.

LOOP_MIN_DETOUR: float = 4.0
# Minimum ratio of graph distance to Euclidean distance for a node pair
# to count as "topologically far" - exactly where a real city would have
# a connecting street, not just two points that happen to be close.

LOOP_MAX_NEW_EDGES: int = 40
# Upper bound on loop-closing edges added per pass, so a small network
# doesn't get flooded with shortcuts.

TARGET_STOP_COUNT: int = 40
# Originally planned as 93 stops; reduced to 40 because the honest road
# network (discard-on-redundant growth, no oscillation inflating the
# count) has only ~73 nodes total - a fixed target above that would
# always shortfall regardless of spacing.

STOP_MIN_SPACING: float = 6.0
# Minimum distance, in cells, enforced between any two selected stops.
# Without this, every stop lands in the CBD, since that's where the
# population is; spacing is what forces the selection to spread out.

STOP_CATCHMENT_RADIUS: float = 6.0
# Radius, in cells, over which a candidate stop's population score is
# summed (not averaged) - a node deep in a dense area outscores one at
# the same density but a smaller catchment.

# --------------------------------------------------------------------------
# Route pool (DBSCAN clustering + Yen's k-shortest paths)
# --------------------------------------------------------------------------

DBSCAN_EPS: float = 8.0
# Max graph distance, in cells, between stops in the same cluster. Stops
# either side of a lake/obstacle that look close in a straight line but
# are far apart over the road network correctly land in different
# clusters, since clustering runs on graph distance, not coordinates.
# An initial guess of 15.0 was set from Euclidean intuition (~7 cells
# mean stop spacing) but chained nearly every stop into one cluster: this
# project's own demand diagnostic measured graph distance at a median
# 1.6x Euclidean, so typical adjacent stops are really ~11 cells apart
# over the network, and 15.0 reached past nearly every neighbour. 8.0 is
# below that 1.6x-scaled neighbour distance instead of above it.

DBSCAN_MIN_SAMPLES: int = 2
# Kept small because there are only 29 stops total; a larger value would
# leave most stops as noise regardless of how they're actually arranged.

ROUTES_PER_PAIR: int = 5
# Number of Yen's k-shortest paths taken between each pair of cluster
# anchors to seed the candidate route pool.

ROUTE_MIN_STOPS: int = 3
# Routes with fewer stops than this are too short to be a useful transit
# route (little more than a single hop) and are discarded.

ROUTE_MAX_LENGTH: float = 200.0
# Routes longer than this, in cells, are implausible for a single bus
# route and are discarded.

# --------------------------------------------------------------------------
# Transit simulation (discrete-time, the GA's fitness function)
# --------------------------------------------------------------------------

SIM_TIME_STEP_MINUTES: float = 1.0
# Discretisation step. Small relative to DWELL_MINUTES_PER_STOP (0.5) and
# typical inter-stop travel time so bus movement/dwell resolve cleanly.

SIM_DURATION_MINUTES: float = 180.0
# A 3-hour peak period.

BUS_SPEED_CELLS_PER_MINUTE: float = 1.0
# Matches ROAD_SEGMENT_LENGTH's cell units; buses move at 1 cell/minute
# of simulated time, no traffic/congestion modelling.

DWELL_MINUTES_PER_STOP: float = 0.5
# Time a bus spends at each stop for alighting/boarding, regardless of
# how many passengers actually board.

BUS_CAPACITY: int = 50
# Passengers a single bus can carry at once - a generic single-unit bus,
# not calibrated to any real vehicle.

MAX_TRANSFERS: int = 2
# A passenger needing more than this many transfers to reach their
# destination is counted unserved rather than simulated indefinitely.

# --------------------------------------------------------------------------
# Genetic algorithm (NSGA-II route selection)
# --------------------------------------------------------------------------

GA_POPULATION: int = 50
# Chromosomes per generation.

GA_GENERATIONS: int = 100
# Generations to evolve.

GA_ROUTES_PER_SOLUTION: int = 20
# Routes per chromosome. Measured (inspect_routes.py reachability audit):
# 8 routes reach 32% of demand, 20 reach 86%, 40 reach 98%. Eight routes
# can't form a real network to trade off; twenty leaves genuine
# trade-offs between coverage, user cost, and operator cost for the GA
# to search, without approaching the near-total coverage of forty where
# there's little left to optimise.

GA_MUTATION_RATE: float = 0.2
# Probability a child chromosome has one route swapped for a random
# unused pool route.

GA_TRANSFER_PENALTY_MINUTES: float = 5.0
# Minutes of equivalent user cost charged per transfer, so the user-cost
# objective doesn't treat a transfer as free just because it isn't wait
# or travel time on its own.

GA_FIXED_FREQUENCY: float = 8.0
# Buses/hour for every route in every chromosome. The inspect_sim.py
# frequency sweep (4/8/16/32) showed served% and wait/created barely
# move across that whole range with a fixed route set - frequency isn't
# the lever here, route selection is - so it's fixed instead of
# optimised, at a representative mid-sweep value.