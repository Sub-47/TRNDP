# TRNDP — AI-Assisted Transit Network Optimisation

Minor project (ENCT 354), Python 3.12. A synthetic city generator feeding a
multi-objective genetic algorithm that optimises bus route networks.

## Pipeline

```
terrain → population → road segments → road graph → connector → loop-closer
→ stops → demand (gravity model) → route pool → NSGA-II GA + transit simulation
```

Each stage lives under `city/`:

| Stage | Module |
|---|---|
| Terrain / population generation | `city/generators/`, `city/sources/` |
| Road network growth + graph | `city/roads/` |
| Demand modelling (gravity model, zones) | `city/demand/` |
| Candidate route pool, clustering | `city/routes/` |
| Transit simulation (discrete-time) | `city/sim/` |
| Multi-objective GA (NSGA-II) | `city/ga/` |

## Running

```bash
pip install -r requirements.txt
pytest -q                       # full test suite
python main.py                  # runs the full pipeline once
python scripts/inspect_ga.py    # GA diagnostic: hypervolume convergence, Pareto front
```

Other `scripts/inspect_*.py` files are standalone diagnostics for individual
pipeline stages (roads, demand, routes, simulation) — they assert nothing,
they just report.

## GA objectives

The NSGA-II search (`city/ga/nsga2.py`) evaluates each candidate route set on
three minimised objectives: user cost (wait + travel time + transfer
penalty), operator cost (total bus distance), and unserved demand.
Convergence is tracked via exact 3D hypervolume (objectives normalised
against a fixed reference point, computed by exact slicing rather than Monte
Carlo sampling — see `docs/` for the writeup, if present locally).

## Status

Config-driven throughout — `config.py` is the single source of truth for
every tunable constant. See `pytest -q` for current test count.
