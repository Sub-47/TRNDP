# TRNDP — AI-Assisted Transit Network Optimisation

Minor project (ENCT 354), Python 3.12. A synthetic city generator feeding a
genetic algorithm that optimises bus route networks.

## Pipeline
terrain → population → road segments → road graph → connector → loop-closer
→ stops → demand → route pool → GA + simulation

**Phase 1 (city + roads) and Phase 2 (demand) are COMPLETE.** Do not modify
them without being asked. Phase 3 (DBSCAN + Yen's k-shortest paths) and
Phase 4 (multi-objective GA + discrete-time simulation) are next.

## Current state (all verified by diagnostics)
- road graph: 100 nodes, 117 edges, 1 connected component, 18 independent
  cycles, 0 obstacle crossings
- stops: 29 selected (target was 40; the network cannot hold 40 at 6-cell
  spacing — the shortfall is reported, not silent)
- demand: singly-constrained gravity model on graph distances, scale-invariant
  to 0.00% across ZONE_SIZE 4/8/16, diurnal periods working
- 55 tests passing
- measured GA cost: ~0.0008 s per fitness evaluation, so a 100×200 run is
  ~16 seconds. **Performance is not a constraint — design for correctness.**

## Three layers — never conflate these
| Layer | What it is | Count |
|---|---|---|
| road node | an intersection in the road graph | ~100 |
| bus stop | a subset of road nodes that buses serve | 29 |
| demand zone | a ZoneMap cell with population, walking to its nearest stop | 256 |

A demand zone connects to its nearest stop by a walk (access link); a stop sits
on a road node and reaches other stops along road edges.

## Rules
- `config.py` is the single source of truth for every tunable number. No module
  hardcodes a constant affecting generation. Every constant carries a comment
  explaining WHY it has that value.
- Diagnostic scripts under `scripts/` (`inspect_population.py`,
  `inspect_roads.py`, `inspect_demand.py`) assert nothing — they are
  instruments, re-run after any change. Tests verify components; diagnostics
  verify the system. **Both are needed**: a connector bug once passed every
  unit test and was only caught by running the diagnostic.
- Style: `from __future__ import annotations`, full type hints, Google-style
  docstrings, step-method class structure (see `TerrainGenerator`).
- Seeded RNG only — `random.Random(seed)` or `np.random.default_rng(seed)`.
  Never global random state. Every stage reproducible from `config.SEED`.
- A component may discard input, but never silently — count it and log it.
- Precompute the stop-to-stop distance matrix ONCE, outside any GA loop.

## How to work on this
- Read only the files named in the task. Do not explore the repo.
- Run `pytest -q` (conftest.py at the root makes plain `pytest` work).
  Do not run `main.py` or the full pipeline unless asked.
- **Report numbers as they come out. NEVER tune constants to hit a target.**
  If a target is missed, say by how much and stop.
- If a spec is ambiguous or a number looks wrong, ask ONE concise question
  rather than trying several approaches. Specs in this project have contained
  arithmetic errors, and flagging them has caught real bugs every time.

## Hard-won findings — do not relearn these
- The road grower once produced 531 segments; ~80% were an oscillation bug
  where a road U-turned into its own ancestry-exempt parent. The honest
  network is ~104 segments. **Do not chase the old number.**
- Road coverage is ~28% of population within 4 cells; stop coverage ~17%.
  This is the accepted cost of a single road tier (no local-street hierarchy,
  a deliberate simplification of Parish & Müller). Documented limitation, not
  a bug.
- `ROAD_SEGMENT_LENGTH` must stay well above `ROAD_SNAP_TOLERANCE` (0.5) or
  segments collapse to zero length in the graph builder.
- `ROAD_PROXIMITY_IGNORE` and the ancestry (`parent_cells`) exemption are
  coupled; changing one without the other has deadlocked growth before.
- Road growth is NOT L-system string rewriting. It implements Parish &
  Müller's `globalGoals`/`localConstraints` directly, which their paper
  describes as where the real logic lives. **The project proposal still says
  "L-system grammar rules and turtle graphics" — that wording needs updating
  before the defense.**
- Euclidean vs graph distance on this city: median ratio 1.61×, worst 30×
  (zone 4 → zone 5: 8 cells apart, 240 cells to travel). This is the headline
  evidence for the project's central claim.