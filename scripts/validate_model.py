"""Save numerical, exhaustive and controlled sensitivity evidence."""
from __future__ import annotations
import itertools
import os
import sys
import time
from pathlib import Path
from dataclasses import replace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('MPLCONFIGDIR','/tmp/trndp-mpl')
import numpy as np
import config
from city.pipeline import build_pipeline,build_stop_od_matrix
from city.demand.distance_matrix import DistanceMatrix
from city.demand.gravity_model import GravityModel
from city.routes.route_pool import Route
from city.sim.simulator import TransitSimulator
from city.objectives import objective_values,reference_point
from city.ga.nsga2 import NSGA2
from city.experiments import (write_json,write_csv,configuration,provenance,choose_greedy,
                              Evaluator,solution_row,save_pipeline,run_methods)


def main():
    root=Path(sys.argv[1] if len(sys.argv)>1 else 'results/model_validation')
    if root.exists() and any(root.iterdir()):raise FileExistsError(root)
    root.mkdir(parents=True)
    write_json(root/'metadata.json',provenance());write_json(root/'config.json',configuration())
    config.SIM_TIME_STEP_MINUTES=0.5
    original={k:getattr(config,k) for k in ('SIM_TIME_STEP_MINUTES','SIM_DURATION_MINUTES','GA_ROUTES_PER_SOLUTION','GA_POPULATION','GA_GENERATIONS')}
    # Prespecified numerical criterion: <=5% difference from the .25 minute run
    # on nonzero aggregate metrics in these selected scenarios. Not a universal error bound.
    metrics=('total_served','total_wait_time','total_travel_time','total_bus_distance','incomplete_passengers')
    p=build_pipeline(city_seed=42)
    chosen=choose_greedy(p.routes,p.stop_od,20,True)
    cases=[('controlled', [Route([0,1,2],7.8,[3.7,4.1])],
            np.array([[0,3.7,7.8],[3.7,0,4.1],[7.8,4.1,0]]),
            np.array([[0.,0,80],[0,0,0],[40,0,0]]),60.),
           ('city42_marginal',[p.routes[i] for i in sorted(chosen)],p.stop_distances,p.stop_od,180.)]
    numeric=[]
    for name,routes,dist,od,duration in cases:
        config.SIM_DURATION_MINUTES=duration
        rows=[]
        for step in (1.,.5,.25):
            config.SIM_TIME_STEP_MINUTES=step
            result=TransitSimulator(routes,dist,od,[8.]*len(routes)).run()
            rows.append(dict(scenario=name,time_step=step,**{m:float(getattr(result,m)) for m in metrics}))
        for row in rows:
            for metric in metrics:
                denominator=abs(rows[-1][metric])
                row[f'{metric}_relative_difference_percent']=100*abs(row[metric]-rows[-1][metric])/denominator if denominator else 0.
            numeric.append(row)
    write_csv(root/'time_step_sensitivity.csv',numeric)
    for k,v in original.items():setattr(config,k,v)
    max_difference=max(row[f'{m}_relative_difference_percent'] for row in numeric for m in metrics)
    write_json(root/'numerical_summary.json',dict(criterion_percent=5.,max_difference_percent=max_difference,
              passed=max_difference<=5.,interpretation='selected scenarios only, relative to 0.25-minute resolution'))
    final_rows=[r for r in numeric if r['time_step']==0.5]
    final_difference=max(r[f'{m}_relative_difference_percent'] for r in final_rows for m in metrics)
    write_json(root/'final_resolution_summary.json',dict(time_step_minutes=.5,criterion_percent=5.,
               max_difference_percent=final_difference,passed=final_difference<=5.,
               note='1-minute failed and is retained in numerical_summary.json; final resolution chosen by refinement'))
    # Exhaustive tiny front, independently filtered. Also measure a genuinely
    # evolutionary run with population smaller than the six-solution space.
    config.GA_ROUTES_PER_SOLUTION=2;config.GA_POPULATION=3;config.GA_GENERATIONS=15;config.SIM_DURATION_MINUTES=12.
    dist=np.abs(np.arange(4)[:,None]-np.arange(4)[None,:])*2.
    pool=[Route([0,1],2.),Route([1,2],2.),Route([2,3],2.),Route([0,1,2,3],6.)]
    od=np.ones((4,4))*4;np.fill_diagonal(od,0.)
    exact={frozenset(c):objective_values(TransitSimulator([pool[i] for i in c],dist,od,[8.,8.]).run())
           for c in itertools.combinations(range(4),2)}
    true={c:o for c,o in exact.items() if not any(all(a<=b for a,b in zip(other,o)) and any(a<b for a,b in zip(other,o)) for other in exact.values())}
    exhaustive=[]
    for seed in range(1,6):
        result=NSGA2(pool,dist,od,seed=seed).run();found=dict(result.pareto_front)
        exhaustive.append(dict(seed=seed,evaluations=result.evaluations,true_front_size=len(true),
            recovered=len(set(found)&set(true)),coverage_percent=100*len(set(found)&set(true))/len(true),
            exact_match=found==true,hypervolume=result.hypervolume_history[-1],
            history=result.hypervolume_history,solutions=[dict(routes=sorted(c),objectives=o) for c,o in found.items()]))
    write_json(root/'exhaustive_pareto.json',dict(all_solutions=[dict(routes=sorted(c),objectives=o) for c,o in exact.items()],
               true_front=[dict(routes=sorted(c),objectives=o) for c,o in true.items()],runs=exhaustive))
    for k,v in original.items():setattr(config,k,v)
    # Controlled ablations: the chosen chromosome is always saved, never mixed minima.
    ref=reference_point(p.stop_od,20,config.GA_FIXED_FREQUENCY);rows=[]
    def score(label,instance,c,scoring='graph BASE'):
        evaluator=Evaluator(instance);evaluator(c)
        rows.append(dict(experiment=label,scoring_demand=scoring,pool_size=len(instance.routes),
                         **solution_row(c,evaluator,ref)))
    score('graph_selection',p,chosen)
    euclidean=GravityModel.from_zones_and_distance(p.zones,DistanceMatrix.from_zone_map(p.zones))
    euclidean_od=build_stop_od_matrix(p.zones,euclidean,p.stops)
    ec=choose_greedy(p.routes,euclidean_od,20,True)
    score('euclidean_selection_common_graph_scoring',p,ec)
    np.savez_compressed(root/'ablation_demand.npz',graph_scoring=p.stop_od,euclidean_selection=euclidean_od)
    unweighted=build_pipeline(city_seed=42,weighted=False)
    budget=min(len(p.routes),len(unweighted.routes),300)
    for label,instance in [('weighted',p),('unweighted',unweighted)]:
        idx=sorted(np.random.default_rng(17).choice(len(instance.routes),budget,replace=False))
        limited=replace(instance,routes=[instance.routes[i] for i in idx],stop_od=p.stop_od)
        c=choose_greedy(limited.routes,p.stop_od,20,True)
        save_pipeline(limited,root/label)
        score(label+'_equal_pool_budget',limited,c)
    for period,factor in config.DIURNAL_FACTORS.items():
        instance=replace(p,stop_od=p.stop_od*factor)
        score(period,instance,chosen,scoring=f'graph {period}; deliberate demand stress test')
    write_csv(root/'controlled_comparisons.csv',rows)
    write_json(root/'comparison_design.json',dict(
        selection='marginal direct-OD coverage greedy',candidate_budget=budget,candidate_sample_seed=17,
        weighted_note='equal pool budget, same graph BASE evaluation demand, one seed; exploratory',
        euclidean_note='same routes and evaluation demand; only selection demand changes',
        periods_note='fixed network, changed load; not evidence that an optimizer improves across periods'))
    # Separate small-budget generalization runs: no pooling with five city42 GA seeds.
    config.GA_POPULATION=8;config.GA_GENERATIONS=5
    for city_seed,pattern in [(43,'RADIAL'),(44,'RADIAL'),(42,'GRID')]:
        output=root/f'city{city_seed}_{pattern.lower()}'
        instance=build_pipeline(city_seed=city_seed,pattern=pattern)
        save_pipeline(instance,output)
        write_json(output/'config.json',dict(model=configuration(),city_seed=city_seed,ga_seeds=[1],pattern=pattern))
        write_json(output/'metadata.json',provenance())
        summaries=run_methods(instance,[1],output);write_json(output/'summary.json',summaries)
        from scripts.plot_results import plot_results
        plot_results(output)
    for k,v in original.items():setattr(config,k,v)
    print('Numerical maximum relative difference (%)',max_difference,flush=True)

if __name__=='__main__':main()
