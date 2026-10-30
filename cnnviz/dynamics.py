"""Scientifically explicit, fixed-probe training-dynamics measurements.

Linear CKA uses centered Gram matrices of flattened activations from the SAME
held-out samples, across all checkpoints. It is not a causal explanation.
"""
from __future__ import annotations
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans

from .research import FiveLayerCNN, research_datasets, _deterministic_subset, _tensor_hist

BASE_DIR = Path(__import__('os').getenv('CNNVIZ_DYNAMICS_ARTIFACTS', str(Path(__file__).resolve().parent.parent / 'artifacts' / 'dynamics')))


def run_path(name: str) -> Path:
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,60}', name):
        raise ValueError('Invalid run ID')
    dest = BASE_DIR / name
    if not (dest / 'manifest.json').is_file():
        raise ValueError('Run not found')
    return dest


@lru_cache(maxsize=8)
def load_run(name: str) -> dict:
    obj = json.loads((run_path(name) / 'manifest.json').read_text(encoding='utf-8'))
    if obj.get('schema') != 1 or obj.get('run_id') != name:
        raise ValueError('Run manifest is inconsistent')
    return obj


@lru_cache(maxsize=8)
def probe_data(name: str):
    meta = load_run(name)
    # CIFAR-10 explicitly requires local dataset. No implicit network downloads in GET.
    _, val, _, _ = research_datasets(meta['dataset'], root=meta.get('data_root','data'), download=False, seed=meta.get('split_seed',meta['seed']))
    val = _deterministic_subset(val, meta['config']['val_limit'], meta.get('split_seed',meta['seed']) + 1)
    pairs = [val[int(i)] for i in meta['probe_indices']]
    x = torch.stack([sample[0] for sample in pairs])
    y = torch.tensor([int(sample[1]) for sample in pairs])
    return x, y


@lru_cache(maxsize=24)
def model_at(name: str, step: int):
    meta = load_run(name)
    if step not in {s['step'] for s in meta['snapshots']}:
        raise ValueError('Step has no saved checkpoint')
    model = FiveLayerCNN(meta['channels'], len(meta['classes']), meta['variant'])
    checkpoint = run_path(name) / f'step_{step:06d}.pt'
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    model.eval()
    return model


@lru_cache(maxsize=24)
def activations_at(name: str, step: int):
    model = model_at(name, step)
    x, _ = probe_data(name)
    with torch.no_grad():
        logits, traces = model.forward_trace(x)
    return logits.detach().cpu(), tuple(t.detach().cpu() for t in traces)


def centered_gram(feature: torch.Tensor) -> np.ndarray:
    if feature.shape[0] < 3:
        raise ValueError('CKA needs at least 3 distinct probe rows')
    x = feature.reshape(feature.shape[0], -1).numpy().astype('float64')
    x -= x.mean(0, keepdims=True)
    gram = x @ x.T
    # Centering twice is safe and minimizes accumulated floating-point bias.
    gram -= gram.mean(0, keepdims=True)
    gram -= gram.mean(1, keepdims=True)
    gram += gram.mean()
    return gram


def linear_cka_gram(a: np.ndarray, b: np.ndarray) -> float:
    # Identical centered Gram matrices must give 1, zero Gram is undefined.
    dot = float((a*b).sum())
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return max(0., min(1., dot/norm)) if norm > 1e-18 else 0.


@lru_cache(maxsize=150)
def activation_gram(name: str, step: int, layer: int) -> np.ndarray:
    if layer not in range(1,6): raise ValueError('Layer must be 1..5')
    return centered_gram(activations_at(name,step)[1][layer-1])


def temporal_cka(name: str, layer: int) -> dict:
    snaps = load_run(name)['snapshots']
    if len(probe_data(name)[0]) < 3: raise ValueError('Insufficient probe samples')
    grams = [activation_gram(name,s['step'],layer) for s in snaps]
    matrix = [[round(linear_cka_gram(a,b), 6) for b in grams] for a in grams]
    return {'run_id':name,'layer':layer,'steps':[s['step'] for s in snaps],
            'matrix':matrix, 'probe_count':len(probe_data(name)[0]),
            'method':'Centered biased linear CKA, identical held-out samples, flattened layer activations',
            'warning':'CKA measures similarity of representations; it does not prove semantic identity or causality.'}


def snapshot_diagnostics(name: str, step: int, layer: int) -> dict:
    if layer not in range(1,6): raise ValueError('Layer must be 1..5')
    meta=load_run(name)
    logits,features=activations_at(name,step)
    conv=getattr(model_at(name,step),f'conv{layer}')
    weights=conv.weight.detach()
    channel_rms=features[layer-1].square().mean((0,2,3)).sqrt()
    events={event['step']:event for event in meta['events']}
    event=events.get(step)
    return {'step':step,'layer':layer,'weight_histogram':_tensor_hist(weights),
            'weight_l2':float(weights.double().norm()),
            'channel_rms':[round(float(x),6) for x in channel_rms],
            'gradient_histogram':event['gradient_histogram'][f'conv{layer}'] if event and event['gradient_histogram'] else None,
            'gradient_preclip':event['gradient_layers_preclip'][f'conv{layer}'] if event else None,
            'update_l2':event['update_layers_l2'][f'conv{layer}'] if event else None,
            'top_responding_channels':torch.argsort(channel_rms,descending=True)[:8].tolist(),
            'note':'Gradient is from the training minibatch at that step (before clipping). Activation uses held-out probes.'}


def channel_clusters(name: str, step: int, layer: int, groups: int=4) -> dict:
    if not (2 <= groups <= 8):raise ValueError('Clusters must be 2..8')
    x,y=probe_data(name)
    features=activations_at(name,step)[1][layer-1]
    # [channels, probes] channel response signatures; every row is independently standardized.
    response=features.mean((-2,-1)).T.numpy().astype('float64')
    mean=response.mean(1,keepdims=True)
    std=response.std(1,keepdims=True)
    z=(response-mean)/np.maximum(std,1e-8)
    k=min(groups,len(z))
    assignments=KMeans(n_clusters=k,random_state=42,n_init=10).fit_predict(z)
    result=[]
    for i in range(len(z)):
        per_label=[]
        for target in range(len(load_run(name)['classes'])):
            mask=(y.numpy()==target)
            per_label.append(round(float(response[i,mask].mean()),6) if mask.any() else None)
        result.append({'channel':i, 'cluster':int(assignments[i]),
                       'mean_response':round(float(mean[i,0]),6),
                       'response_by_class':per_label})
    counts=[int((assignments==j).sum()) for j in range(k)]
    return {'step':step,'layer':layer,'groups':k,'counts':counts,
            'channels':result,'probe_count':len(x),
            'note':'KMeans of standardized per-image channel mean responses. Clusters are checkpoint-local, and cluster IDs are arbitrary.'}


def pair_probe(name: str, step_a: int, step_b: int, layer: int, channel: int, sample_index: int) -> dict:
    from .comparison import ActivationScale, compare_activation, render_activation
    from .imaging import input_image
    
    if layer not in range(1,6): raise ValueError('Layer must be 1..5')
    xx, labels=probe_data(name)
    if not (0 <= sample_index < len(xx)): raise ValueError('Sample out of range')
    out_a,fa=activations_at(name,step_a)
    out_b,fb=activations_at(name,step_b)
    f_a=fa[layer-1][sample_index];f_b=fb[layer-1][sample_index]
    if not (0<=channel<len(f_a)): raise ValueError('Channel out of range')
    # Fixed color range across ALL stored steps for this one probe/layer, across channels.
    ceiling=max(float(activations_at(name,s['step'])[1][layer-1][sample_index].max())
                for s in load_run(name)['snapshots'])
    scale=ActivationScale(0.,max(ceiling,1e-9),'probe_layer_all_steps_all_channels')
    a,b=f_a[channel],f_b[channel]
    compare=compare_activation(a,b,scale)
    probs_a=out_a[sample_index].softmax(0)
    probs_b=out_b[sample_index].softmax(0)
    return {'step_a':step_a,'step_b':step_b,'layer':layer,'channel':channel,
        'sample_index':sample_index,'label':int(labels[sample_index]),
        'input_image':input_image(xx[sample_index],upscale=5),
        'scale': scale.as_dict(),
        'activation_a':render_activation(a,scale=scale,mode='fixed',upscale=max(6,128//a.shape[-1])),
        'activation_b':render_activation(b,scale=scale,mode='fixed',upscale=max(6,128//b.shape[-1])),
        'prediction_a':int(probs_a.argmax()),'prediction_b':int(probs_b.argmax()),
        'true_probability_a':float(probs_a[labels[sample_index]]),
        'true_probability_b':float(probs_b[labels[sample_index]]),
        'cka':round(linear_cka_gram(activation_gram(name,step_a,layer),activation_gram(name,step_b,layer)),6),
        **compare}