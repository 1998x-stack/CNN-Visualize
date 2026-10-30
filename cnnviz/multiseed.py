"""Reproducible, explicit multi-seed experiment orchestrator.

Use one fixed split and probe set. Changing only training initialization and
minibatch order controls one source of seed variation. No hidden downloads.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from itertools import combinations

import numpy as np

from . import dynamics
from .research import run_research
from .concepts import compare_runs, _aligned


def study(*, prefix='concept-digits',dataset='digits',variant='zf-small',seeds=(11,22,33),
          split_seed=42,epochs=8,train_limit=1024,val_limit=160,probe_count=100,
          interval=16,root='data',download=False,out='artifacts/dynamics'):
    if len(set(seeds))!=len(seeds) or len(seeds)<2 or len(seeds)>8:
        raise ValueError('Need 2..8 unique training seeds')
    if probe_count>val_limit and val_limit>0:
        raise ValueError('probe_count cannot exceed val_limit')
    manifests=[]
    for seed in seeds:
        name=f'{prefix}-s{seed}'
        m=run_research(name=name,dataset=dataset,variant=variant,seed=seed,
            split_seed=split_seed,epochs=epochs,train_limit=train_limit,
            val_limit=val_limit,probe_count=probe_count,interval=interval,
            root=root,download=download,out=out)
        manifests.append(m)
    orig=dynamics.BASE_DIR
    dynamics.BASE_DIR=Path(out).resolve()
    for func in [dynamics.load_run,dynamics.probe_data,dynamics.model_at,dynamics.activations_at,dynamics.activation_gram]:func.cache_clear()
    try:
        pairs=[]
        for a,b in combinations(manifests,2):
            _aligned(a['run_id'],b['run_id'])
            ca=compare_runs(a['run_id'],b['run_id'],a['snapshots'][-1]['step'],
                b['snapshots'][-1]['step'],5,bootstrap=40)
            pairs.append({'a':a['run_id'],'b':b['run_id'],'final_layer_cka':ca['cka'],
                'probe_bootstrap_ci95':ca['bootstrap_ci95'],
                'matched_response_corr':ca['channel_mean_corr']})
    finally:
        dynamics.BASE_DIR=orig
        for func in [dynamics.load_run,dynamics.probe_data,dynamics.model_at,dynamics.activations_at,dynamics.activation_gram]:func.cache_clear()
    accuracy=np.array([m['metrics'][-1]['val_accuracy'] for m in manifests])
    report={'schema':1,'dataset':dataset,'variant':variant,'split_seed':split_seed,
        'probe_count':len(manifests[0]['probe_indices']),'seeds':list(seeds),
        'runs':[{'run_id':m['run_id'],'seed':m['seed'],
           'final_val_accuracy':m['metrics'][-1]['val_accuracy'],
           'final_step':m['snapshots'][-1]['step']} for m in manifests],
        'mean_accuracy':float(accuracy.mean()),
        'std_accuracy_sample':float(accuracy.std(ddof=1)),
        'pairwise':pairs,
        'limitations':'Few training seeds. Mean and sample SD describe these runs only; pairwise CKA bootstrap CI resamples probes, not independent models. Digits is 8x8 upsampled when selected.'}
    dest=Path(out)/f'{prefix}-summary.json';dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--prefix',default='concept-digits')
    p.add_argument('--dataset',choices=['digits','cifar10'],default='digits')
    p.add_argument('--variant',choices=['zf-small','alex-small'],default='zf-small')
    p.add_argument('--seeds',type=int,nargs='+',default=[11,22,33])
    p.add_argument('--split-seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=8)
    p.add_argument('--train-limit',type=int,default=1024)
    p.add_argument('--val-limit',type=int,default=160)
    p.add_argument('--probe-count',type=int,default=100)
    p.add_argument('--interval',type=int,default=16)
    p.add_argument('--root',default='data')
    p.add_argument('--download',action='store_true')
    p.add_argument('--out',default='artifacts/dynamics')
    report=study(**vars(p.parse_args()))
    print('Multi-seed report:',json.dumps(report,indent=2))
if __name__=='__main__':main()