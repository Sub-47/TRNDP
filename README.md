# TRNDP — Synthetic Transit Network Optimization

A university research prototype integrating terrain/population generation,
road-network demand, demand-weighted DBSCAN/Yen route generation, a fluid
transit simulator, and NSGA-II route-set selection. Distances are synthetic
cells; this is not calibrated to a real transit network.

## Install and verify

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

Tested with Python 3.14.7. Each experiment records dependency versions and an
exact source snapshot. Other Python versions have not been verified.

## Canonical experiments

Run from the repository root. Use a new output directory for each execution:

```bash
.venv/bin/python scripts/run_experiment.py --config configs/smoke.json --output results/my_smoke
.venv/bin/python scripts/run_experiment.py --config configs/final_experiment.json --output results/my_final
.venv/bin/python scripts/validate_model.py results/my_validation
.venv/bin/python scripts/plot_results.py results/my_final
.venv/bin/python scripts/check_results.py results/my_final
```

`smoke.json` runs a small GA for an interactive demonstration. `final_experiment.json`
uses city seed 42, five independent optimizer seeds, population 20, 20 generations,
20 routes per solution, and a 180-minute horizon. This is a finite-budget comparison,
not a claim of global convergence. Final experiments use **0.5-minute demand batches**.
The validation script compares numerical resolutions and runs controlled model checks.

Every experiment saves full config, Git revision and dirty status, source hashes
and source ZIP, dependency versions, geometry, demand arrays, all evaluated
solutions, nondominated archives, histories, runtimes, and figures. Random search
gets exactly each corresponding GA run's unique simulation count. A matched pair
shares the initial random samples but evaluates them separately. Both greedy
baselines use the same inputs, route count, simulator, frequency, and objectives.

Generated files under `results/` and `output/` are local and ignored by Git.
The commands above create the outputs needed for plotting and result checks.

## Model contract

- Generated routes retain full road geometry and actual stop-leg distances.
- Diagonal stop OD demand is local/walking mass, recorded outside transit queues.
- Transit demand equals completed + queued + abandoned + onboard passengers.
- Fitness minimizes passenger time plus transfer penalty, bus distance, and **all
  incomplete transit passengers**, including those onboard at the fixed horizon.
- Vehicles depart both termini at the stated frequency **per direction**, make
  one scheduled one-way trip, and retire. Fleet reuse/deadheading is not modeled.
- Demand is deterministic fluid mass in midpoint batches. Vehicle events are
  chronological; travel/wait times are integrated over actual elapsed intervals.
- Passenger routing minimizes boardings, with a distance tie breaker. It is not
  timetable-optimal, congestion-aware, or an individual stochastic agent model.
- Current-population hypervolume may decrease. The cumulative nondominated archive
  includes every evaluated candidate and has monotonic HV under the common fixed
  reference bounds. "Balanced" is one actual solution minimizing the Euclidean
  norm of its reference-normalized objective vector; it is not a proven knee.

## Layout and diagnostics

`city/pipeline.py` is the shared pipeline; `city/objectives.py` defines fitness;
`city/experiments.py` implements baselines and evidence export. Research code lives
under `city/roads`, `city/demand`, `city/routes`, `city/sim`, and `city/ga`.

```bash
.venv/bin/python scripts/inspect_roads.py
.venv/bin/python scripts/inspect_demand.py
.venv/bin/python scripts/inspect_routes.py
.venv/bin/python scripts/inspect_sim.py
.venv/bin/python scripts/inspect_ga.py
```

`main.py` generates/displays/saves **map layers only**. `inspect_ga.py` runs the
optimizer with `config.py` defaults and prints diagnostics. The canonical runner
above saves the inputs and outputs needed to reproduce and check an experiment.
