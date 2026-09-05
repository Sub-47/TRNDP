"""Regenerate defense figures solely from an experiment's saved JSON/CSV files."""
from __future__ import annotations
import csv
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('MPLCONFIGDIR','/tmp/trndp-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np


def plot_results(root: Path):
    summaries=json.loads((root/'summary.json').read_text());network=json.loads((root/'network.json').read_text())
    dest=root/'figures';dest.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'figure.dpi':130,'savefig.dpi':180})
    def save(fig,name):
        fig.tight_layout();fig.savefig(dest/f'{name}.png');fig.savefig(dest/f'{name}.pdf');plt.close(fig)
    fig=plt.figure(figsize=(9,7));ax=fig.add_subplot(projection='3d')
    for method,color in [('greedy','black'),('marginal_coverage','orange'),('random','gray'),('nsga2','tab:blue')]:
        points=[]
        for s in summaries:
            if s['method']!=method:continue
            folder=method if s['seed'] is None else f"{method}_seed_{s['seed']}"
            with (root/folder/'pareto_front.csv').open() as f:
                points.extend([[float(r[k]) for k in ('user_cost','operator_distance','incomplete')] for r in csv.DictReader(f)])
        if points:
            v=np.array(points);ax.scatter(*v.T,label=method,c=color,s=12,alpha=.65)
    ax.set(xlabel='User cost (passenger-min)',ylabel='Bus distance (cells)',zlabel='Incomplete transit trips',title='Nondominated solutions within each method/run')
    ax.legend();save(fig,'pareto_front')
    fig,ax=plt.subplots(figsize=(8,4))
    for s in summaries:
        if s['method']!='nsga2':continue
        with (root/f"nsga2_seed_{s['seed']}"/'history.csv').open() as f:rows=list(csv.DictReader(f))
        ax.plot([int(r['generation']) for r in rows],[float(r['archive_hypervolume']) for r in rows],label=f"Seed {s['seed']}")
    ax.set(xlabel='Generation',ylabel='Archive hypervolume (reference-normalized)',title='Cumulative archive; common fixed reference');ax.legend();save(fig,'hypervolume')
    methods=['greedy','marginal_coverage','random','nsga2'];groups=[[s for s in summaries if s['method']==m] for m in methods]
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,metric,label in [(axes[0],'completed_percent','Completed transit trips (%)'),(axes[1],'user_cost','User cost (passenger-min)'),(axes[2],'operator_distance','Bus distance (cells)')]:
        values=[[s['representatives']['balanced'][metric] for s in g] for g in groups]
        means=[np.mean(v) for v in values]; std=[np.std(v,ddof=1) if len(v)>1 else 0 for v in values]
        ax.bar(methods,means,yerr=std,capsize=4);ax.set_ylabel(label);ax.tick_params(axis='x',rotation=20);ax.set_ylim(bottom=0)
    fig.suptitle('Actual balanced solutions: mean ± sample SD across optimizer seeds');save(fig,'baseline_comparison')
    fig,ax=plt.subplots(figsize=(8,4));bottom=np.zeros(len(methods))
    for key,label in [('completed_percent','Completed'),('queued_percent','Queued'),('onboard_percent','Onboard')]:
        values=np.array([np.mean([s['representatives']['balanced'][key] for s in g]) for g in groups])
        ax.bar(methods,values,bottom=bottom,label=label);bottom+=values
    ax.set(ylabel='Percentage of transit-required demand',ylim=(0,100),title='Passenger outcomes at the fixed horizon (local trips excluded)');ax.legend();save(fig,'passenger_outcomes')
    fig,ax=plt.subplots(figsize=(7,4))
    for i,g in enumerate(groups):
        values=[s['hypervolume'] for s in g];ax.scatter([i]*len(values),values,s=35)
    ax.set_xticks(range(len(methods)),methods);ax.set(ylabel='Final archive hypervolume',title='Seed variation; deterministic baselines have one observation');save(fig,'seed_variation')
    exemplar=next(s for s in summaries if s['method']=='nsga2')
    for label,row in exemplar['representatives'].items():
        fig,ax=plt.subplots(figsize=(9,8))
        terrain_names=['Ocean','Beach','Plains','Hills','Mountains']
        terrain_colors=['#b9d9eb','#f6edcb','#e5efdf','#dce2c9','#d1d1d1']
        ax.imshow(network['terrain'],cmap=ListedColormap(terrain_colors),vmin=-.5,vmax=4.5)

        for edge in network['edges']:
            a,b=edge['source'],edge['target'];ax.plot([a[0],b[0]],[a[1],b[1]],color='gray',lw=.5)
        for number,index in enumerate(json.loads(row['routes'])):
            route=network['routes'][index];path=np.array(route['road_path'])
            ax.plot(path[:,0],path[:,1],lw=1.6,color=plt.cm.tab20(number%20),alpha=.8)
        stops=np.array(network['stops']);ax.scatter(stops[:,0],stops[:,1],s=8,c='black',label='Bus stops')
        ax.set(xlabel='x (cells)',ylabel='y (cells)',title=f"{label.replace('_',' ')} — GA seed {exemplar['seed']}\nCompleted {row['completed_percent']:.1f}% of transit demand")
        handles=[Patch(facecolor=c,label=n) for n,c in zip(terrain_names,terrain_colors)]
        handles.extend([Line2D([0],[0],color='tab:red',lw=2,label='Selected routes (individual colors)'),
                        Line2D([0],[0],color='gray',lw=.5,label='Road network'),
                        Line2D([0],[0],marker='o',color='black',lw=0,label='Bus stops')])
        ax.legend(handles=handles,loc='upper right',fontsize=8)
        save(fig,f'routes_{label}')

if __name__=='__main__':plot_results(Path(sys.argv[1]))
