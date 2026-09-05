"""Audit saved experiment results and compare deterministic reruns."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from city.ga.nsga2 import NSGA2
from city.experiments import write_json


def check(root: Path) -> dict:
    metadata=json.loads((root/'metadata.json').read_text())
    assert metadata['status']=='complete', 'experiment did not complete'
    summaries=json.loads((root/'summary.json').read_text())
    network=json.loads((root/'network.json').read_text())
    distances={(tuple(e['source']),tuple(e['target'])):e['distance'] for e in network['edges']}
    for route in network['routes']:
        edges=list(zip(route['road_path'],route['road_path'][1:]))
        total=0.
        for a,b in edges:
            key=(tuple(a),tuple(b));reverse=key[::-1]
            assert key in distances or reverse in distances,'invalid road edge'
            total+=distances[key] if key in distances else distances[reverse]
        assert np.isclose(total,route['length'])
        assert np.isclose(sum(route['leg_distances']),route['length'])
    with zipfile.ZipFile(root/'source_snapshot.zip') as archive:
        for name,digest in metadata['source_hashes'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest()==digest, f'source hash mismatch: {name}'
    total=0
    for summary in summaries:
        method,seed=summary['method'],summary['seed']
        folder=root/(method if seed is None else f'{method}_seed_{seed}')
        rows=list(csv.DictReader((folder/'solutions.csv').open()))
        assert len(rows)==summary['evaluations']
        seen=set()
        for r in rows:
            route_ids=tuple(json.loads(r['routes']));assert route_ids not in seen;seen.add(route_ids)
            assert len(route_ids)==len(set(route_ids))
            spec=json.loads((root/'config.json').read_text())
            assert len(route_ids)==spec['model']['GA_ROUTES_PER_SOLUTION']
            assert all(0<=i<len(network['routes']) for i in route_ids)
            values={k:float(v) for k,v in r.items() if k!='routes'}
            assert all(np.isfinite(v) and v>=-1e-7 for v in values.values())
            assert np.isclose(values['total_created'],values['total_served']+values['queued_passengers']+values['abandoned_passengers']+values['still_onboard'])
            assert np.isclose(values['total_input_demand'],values['total_created']+values['local_or_walking_demand'])
            assert np.isclose(values['incomplete'],values['total_created']-values['total_served'])
            assert np.isclose(values['total_wait_time'],values['served_wait_time']+values['unserved_wait_time']+values['still_onboard_wait_time'])
            assert values['peak_load_factor']<=1+1e-9
        front=list(csv.DictReader((folder/'pareto_front.csv').open()))
        objectives=np.array([[float(r[k]) for k in ('user_cost','operator_distance','incomplete')] for r in front])
        for row in objectives:
            assert not any(np.all(other<=row) and np.any(other<row) for other in objectives)
        all_values=np.array([[float(r[k]) for k in ('user_cost','operator_distance','incomplete')] for r in rows])
        for point in objectives:
            assert not np.any(np.all(all_values<=point,axis=1) & np.any(all_values<point,axis=1))
        for point in all_values:
            assert np.any(np.all(objectives<=point,axis=1)), 'archive omitted an evaluated nondominated point'
        hv=object.__new__(NSGA2);hv.reference_point=np.array(summary['reference'])
        assert np.isclose(hv._hypervolume(objectives),summary['hypervolume'])
        for rep in summary['representatives'].values():
            assert any(r['routes']==rep['routes'] and all(float(r[k])==rep[k] for k in ('user_cost','operator_distance','incomplete')) for r in front)
        if method=='nsga2':
            history=list(csv.DictReader((folder/'history.csv').open()))
            assert np.all(np.diff([float(r['archive_hypervolume']) for r in history])>=-1e-12)
            assert np.isclose(float(history[-1]['archive_hypervolume']),summary['hypervolume'])
            partner=next(s for s in summaries if s['method']=='random' and s['seed']==seed)
            assert partner['evaluations']==summary['evaluations']
        total+=len(rows)
    figures=list((root/'figures').glob('*.png'))
    assert len(figures)>=9 and all(p.stat().st_size>1000 for p in figures)
    return dict(passed=True,solutions_checked=total,routes_checked=len(network['routes']),figures_checked=len(figures))


def compare(a: Path,b: Path) -> dict:
    # Timestamps, wall times, dirty status and source/documentation updates are
    # not deterministic scientific outputs. Numeric arrays and all search rows are.
    files=['config.json','network.json']
    files += [str(p.relative_to(a)) for p in a.rglob('*.csv')]
    for name in files:
        assert (a/name).read_bytes()==(b/name).read_bytes(),f'rerun mismatch: {name}'
    with np.load(a/'demand.npz') as left,np.load(b/'demand.npz') as right:
        assert left.files==right.files
        for name in left.files:assert np.array_equal(left[name],right[name]),f'demand mismatch: {name}'
    return dict(passed=True,files_compared=len(files),demand_arrays_compared=len(left.files),
                excluded='timestamps, runtime, source/provenance metadata; scientific CSV and inputs match exactly')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);parser.add_argument('--compare',type=Path)
    args=parser.parse_args();result=check(args.directory)
    if args.compare:result['reproducibility']=compare(args.directory,args.compare)
    write_json(args.directory/'audit.json',result);print(json.dumps(result,indent=2))
