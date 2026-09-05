"""Shared baseline selection, evaluation and machine-readable experiment evidence."""
from __future__ import annotations
import csv
import hashlib
import itertools
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import numpy as np
import config
from city.pipeline import Pipeline
from city.sim.simulator import TransitSimulator
from city.objectives import objective_values, reference_point
from city.ga.nsga2 import NSGA2, fast_non_dominated_sort


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        if rows:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def configuration() -> dict:
    return {k:v for k,v in vars(config).items() if k.isupper() and isinstance(v,(str,int,float,bool,dict,list,tuple))}


def provenance() -> dict:
    root=Path(__file__).resolve().parents[1]
    def git(*args):
        return subprocess.check_output(['git',*args],cwd=root,text=True).strip()
    paths=[root/'config.py',root/'requirements.txt']
    for directory in ('city','scripts','tests','configs'):
        paths.extend(p for p in (root/directory).rglob('*') if p.suffix in ('.py','.json'))
    hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    return dict(timestamp=datetime.now(timezone.utc).isoformat(),git_commit=git('rev-parse','HEAD'),
                git_dirty=bool(git('status','--porcelain')),python=sys.version,
                dependencies={name:version(name) for name in ('numpy','networkx','scipy','scikit-learn','matplotlib','noise','Pillow','pytest')},
                source_hashes=hashes,source_fingerprint=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest())


def choose_greedy(routes, od, count: int, marginal: bool = False) -> frozenset[int]:
    if not 0<count<=len(routes): raise ValueError('route count outside pool size')
    weights=np.array(od,dtype=float,copy=True);np.fill_diagonal(weights,0.)
    masks=[]
    for route in routes:
        mask=np.zeros_like(weights,dtype=bool);mask[np.ix_(route.stops,route.stops)]=True;np.fill_diagonal(mask,False)
        masks.append(mask)
    chosen=[]; covered=np.zeros_like(weights,dtype=bool)
    scores=np.array([weights[m].sum() for m in masks])
    for _ in range(count):
        if marginal: scores=np.array([weights[m & ~covered].sum() for m in masks])
        scores[chosen]=-np.inf
        index=int(np.argmax(scores));chosen.append(index);covered |= masks[index]
    return frozenset(chosen)


class Evaluator:
    """Each method owns its cache; count real simulations, not cache lookups."""
    def __init__(self,pipeline: Pipeline):
        self.pipeline=pipeline;self.records={}

    def __call__(self,chromosome):
        c=frozenset(chromosome)
        if c not in self.records:
            p=self.pipeline
            result=TransitSimulator([p.routes[i] for i in sorted(c)],p.stop_distances,p.stop_od,
                                    [config.GA_FIXED_FREQUENCY]*len(c)).run()
            self.records[c]=(objective_values(result),result)
        return self.records[c][0]


def nondominated(records: dict) -> list:
    keys=list(records)
    if not keys:return []
    values=[records[c][0] for c in keys]
    return [keys[i] for i in fast_non_dominated_sort(values)[0]]


def representatives(front: list, evaluator: Evaluator, ref) -> dict:
    values=np.array([evaluator(c) for c in front])
    names=('minimum_user_cost','minimum_operator_distance','maximum_completion')
    result={name:front[int(np.argmin(values[:,i]))] for i,name in enumerate(names)}
    # One actual chromosome nearest the origin in common reference-normalized space.
    result['balanced']=front[int(np.argmin(np.linalg.norm(values/ref,axis=1)))]
    return result


def solution_row(c,evaluator: Evaluator,ref) -> dict:
    objectives,result=evaluator.records[c]
    metrics=asdict(result);transit=result.total_created
    metrics.update(incomplete_passengers=result.incomplete_passengers,
                   completed_percent=100*result.total_served/transit if transit else 0.,
                   queued_percent=100*result.queued_passengers/transit if transit else 0.,
                   onboard_percent=100*result.still_onboard/transit if transit else 0.,
                   incomplete_percent=100*result.incomplete_passengers/transit if transit else 0.,
                   local_percent=100*result.local_or_walking_demand/result.total_input_demand if result.total_input_demand else 0.,
                   wait_per_created=result.total_wait_time/transit if transit else 0.,
                   served_wait_per_completed=result.served_wait_time/result.total_served if result.total_served else 0.,
                   travel_per_created=result.total_travel_time/transit if transit else 0.)
    return dict(routes=json.dumps(sorted(c)),user_cost=objectives[0],operator_distance=objectives[1],
                incomplete=objectives[2],normalized_user=objectives[0]/ref[0],
                normalized_operator=objectives[1]/ref[1],normalized_incomplete=objectives[2]/ref[2],**metrics)


def save_method(path,method,seed,evaluator,front,ref,history,elapsed) -> dict:
    rows=[solution_row(c,evaluator,ref) for c in evaluator.records]
    write_csv(path/'solutions.csv',rows)
    write_csv(path/'pareto_front.csv',[solution_row(c,evaluator,ref) for c in front])
    write_csv(path/'history.csv',history)
    reps=representatives(front,evaluator,ref)
    hv=object.__new__(NSGA2);hv.reference_point=ref
    summary=dict(method=method,seed=seed,evaluations=len(evaluator.records),runtime_seconds=elapsed,
                 hypervolume=hv._hypervolume(np.array([evaluator(c) for c in front])),
                 pareto_size=len(front),reference=ref.tolist(),
                 representatives={name:solution_row(c,evaluator,ref) for name,c in reps.items()},
                 denominator='transit demand excludes separately reported local/walking trips')
    write_json(path/'summary.json',summary)
    return summary


def run_methods(pipeline: Pipeline,seeds: list[int],output: Path) -> list[dict]:
    output.mkdir(parents=True,exist_ok=True)
    ref=reference_point(pipeline.stop_od,config.GA_ROUTES_PER_SOLUTION,config.GA_FIXED_FREQUENCY)
    summaries=[]
    for method in ('greedy','marginal_coverage'):
        start=time.perf_counter();evaluator=Evaluator(pipeline)
        c=choose_greedy(pipeline.routes,pipeline.stop_od,config.GA_ROUTES_PER_SOLUTION,method=='marginal_coverage')
        evaluator(c)
        summaries.append(save_method(output/method,method,None,evaluator,[c],ref,[],time.perf_counter()-start))
    for seed in seeds:
        path=output/f'nsga2_seed_{seed}';path.mkdir(parents=True,exist_ok=True)
        evaluator=Evaluator(pipeline);history=[]
        def record(row):
            history.append(row)
            write_csv(path/'history.csv',history)
        ga=NSGA2(pipeline.routes,pipeline.stop_distances,pipeline.stop_od,seed=seed,reference=ref,evaluator=evaluator)
        start=time.perf_counter();result=ga.run(on_generation=record);elapsed=time.perf_counter()-start
        summaries.append(save_method(path,'nsga2',seed,evaluator,[c for c,o in result.pareto_front],ref,history,elapsed))
        print(f'NSGA-II seed {seed}: {len(evaluator.records)} evaluations, {elapsed:.2f} s',flush=True)
        # Exactly match the GA's unique simulation count for this seed.
        budget=len(evaluator.records);random_eval=Evaluator(pipeline);rng=np.random.default_rng(seed)
        start=time.perf_counter()
        while len(random_eval.records)<budget:
            c=frozenset(int(i) for i in rng.choice(len(pipeline.routes),size=config.GA_ROUTES_PER_SOLUTION,replace=False))
            random_eval(c)
        elapsed=time.perf_counter()-start
        summaries.append(save_method(output/f'random_seed_{seed}','random',seed,random_eval,
                                     nondominated(random_eval.records),ref,[],elapsed))
        print(f'Random seed {seed}: {len(random_eval.records)} evaluations, {elapsed:.2f} s',flush=True)
        write_json(output/'summary.json',summaries)
    return summaries


def save_pipeline(p: Pipeline,output: Path) -> None:
    write_json(output/'network.json',dict(nodes=list(p.graph.nodes),
        edges=[dict(source=a,target=b,distance=float(d['weight'])) for a,b,d in p.graph.edges(data=True)],
        stops=p.stops,routes=[asdict(r) for r in p.routes],
        metrics=dict(nodes=len(p.graph),edges=p.graph.number_of_edges(),stops=len(p.stops),routes=len(p.routes)),
        terrain=p.world.terrain.data.tolist()))
    np.savez_compressed(output/'demand.npz',stop_od=p.stop_od,stop_distances=p.stop_distances,
                        zone_od=p.demand.data,population=p.world.population.data)
