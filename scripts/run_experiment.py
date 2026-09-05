"""Canonical runner. Configuration is applied before importing model modules."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('MPLCONFIGDIR','/tmp/trndp-mpl')
import config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();spec=json.loads(args.config.read_text())
    allowed={'city_seed','ga_seeds','period','pattern','weighted','demand_metric','overrides'}
    if set(spec)-allowed:raise ValueError(f'unknown experiment keys: {set(spec)-allowed}')
    for key,value in spec.get('overrides',{}).items():
        if not key.isupper() or not hasattr(config,key):raise ValueError(f'unknown config key {key}')
        setattr(config,key,value)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'{args.output} is not empty; choose a new output directory')
    from city.pipeline import build_pipeline
    from city.experiments import write_json,configuration,provenance,save_pipeline,run_methods
    args.output.mkdir(parents=True,exist_ok=True)
    write_json(args.output/'config.json',dict(experiment=spec,model=configuration()))
    metadata=provenance();metadata.update(city_seed=spec['city_seed'],ga_seeds=spec['ga_seeds'],
                                         demand_seed=None,demand_model='deterministic gravity',status='running')
    write_json(args.output/'metadata.json',metadata)
    source_root=Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(args.output/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in metadata['source_hashes']:
            archive.write(source_root/name,arcname=name)
    p=build_pipeline(city_seed=spec['city_seed'],pattern=spec.get('pattern'),period=spec.get('period','BASE'),
                     weighted=spec.get('weighted',True),demand_metric=spec.get('demand_metric','graph'))
    save_pipeline(p,args.output)
    summaries=run_methods(p,spec['ga_seeds'],args.output)
    write_json(args.output/'summary.json',summaries)
    metadata['status']='complete';write_json(args.output/'metadata.json',metadata)
    from scripts.plot_results import plot_results
    plot_results(args.output)

if __name__=='__main__':main()
