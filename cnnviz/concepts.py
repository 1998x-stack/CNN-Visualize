"""V0.5 representation/concept discovery over *measured*, aligned held-out probes.

Interpretation discipline: response rankings are candidate concepts, matched units are
correlated response signatures, linear CKA measures representation similarity, and
occlusion measures behavior under an out-of-distribution intervention.
"""
from __future__ import annotations

import hashlib
import math
from functools import lru_cache

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F

from . import dynamics as dyn
from .imaging import input_image
from .comparison import ActivationScale, render_activation


def _check(name: str, step: int, layer: int):
    if not 1 <= layer <= 5:
        raise ValueError('Layer must be 1..5')
    dyn.model_at(name, step)  # validates actual checkpoint, without interpolating


def _samples(name: str):
    x,y = dyn.probe_data(name)
    return x,y


def _aligned(left: str, right: str):
    x1,y1 = _samples(left)
    x2,y2 = _samples(right)
    m1,m2=dyn.load_run(left),dyn.load_run(right)
    if (m1['dataset']!=m2['dataset'] or len(x1)!=len(x2)
            or not torch.equal(x1,x2) or not torch.equal(y1,y2)):
        raise ValueError('Runs are not aligned: use the same fixed split, subset and held-out probe IDs')
    if len(x1)<4:
        raise ValueError('At least four aligned held-out samples are required')
    return x1,y1


def _fingerprint(x: torch.Tensor, y: torch.Tensor) -> str:
    h=hashlib.sha256()
    h.update(x.detach().contiguous().numpy().tobytes())
    h.update(y.detach().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def response_signature(name: str, step: int, layer: int):
    _check(name,step,layer)
    _,features=dyn.activations_at(name,step)
    return features[layer-1].mean((-2,-1)).T.detach().numpy().astype(np.float64) # [C,N]


def _normalized_rows(values: np.ndarray):
    values = values-values.mean(axis=1,keepdims=True)
    norm=np.linalg.norm(values,axis=1,keepdims=True)
    return values/np.maximum(norm,1e-12)


def rank_concepts(name: str, step: int, layer: int, channel: int, k: int=8):
    """Top-k by sample-wise spatial *peak*, not a human-labeled concept classifier."""
    _check(name,step,layer)
    if not 1<=k<=16: raise ValueError('Top-k must be 1..16')
    x,y=_samples(name)
    activation=dyn.activations_at(name,step)[1][layer-1]
    if not 0<=channel<activation.shape[1]: raise ValueError('Channel out of range')
    maps=activation[:,channel].detach()
    score=maps.flatten(1).max(1).values
    ranking=torch.argsort(score,descending=True,stable=True)[:min(k,len(score))]
    ceiling=max(float(maps.max()),1e-9)
    scale=ActivationScale(0,ceiling,'same_channel_same_checkpoint_all_probes')
    records=[]
    for ix in ranking.tolist():
        records.append({'probe_index':ix,'label':int(y[ix]),
            'peak_response':round(float(score[ix]),6),
            'mean_response':round(float(maps[ix].mean()),6),
            'input':input_image(x[ix],upscale=5),
            'activation':render_activation(maps[ix],scale=scale,mode='fixed',upscale=max(5,128//maps.shape[-1]))})
    counts=np.bincount(y.numpy(),minlength=len(dyn.load_run(name)['classes']))
    top_counts=np.bincount(y[ranking].numpy(),minlength=len(counts))
    dominant=int(top_counts.argmax())
    return {'run_id':name,'step':step,'layer':layer,'channel':channel,'pool_size':len(x),
        'probe_fingerprint':_fingerprint(x,y),'ranking':'maximum spatial ReLU activation',
        'scale':scale.as_dict(),'top_k':records,'counts_in_top_k':top_counts.tolist(),
        'dominant_label':dominant,'top_k_purity':round(float(top_counts[dominant]/len(ranking)),4),
        'pool_prior':round(float(counts[dominant]/len(y)),4),
        'warning':'Top-k is a ranked set of held-out examples, NOT proof of a stable semantic concept. Probe pool is limited.'}


def match_channels(left: str, right: str, step_left: int, step_right: int, layer: int, k: int=8):
    _aligned(left,right)
    a=response_signature(left,step_left,layer)
    b=response_signature(right,step_right,layer)
    similarity=_normalized_rows(a)@_normalized_rows(b).T
    r,c=linear_sum_assignment(-similarity)
    matched=sorted([{'channel_a':int(i),'channel_b':int(j),'response_corr':round(float(similarity[i,j]),5)}
                    for i,j in zip(r,c)],key=lambda row:row['response_corr'],reverse=True)
    return {'run_a':left,'run_b':right,'step_a':step_left,'step_b':step_right,'layer':layer,
        'matched':matched,'top_matches':matched[:k],
        'mean_matched_corr':round(float(np.mean([z['response_corr'] for z in matched])),5),
        'num_channels_a':len(a),'num_channels_b':len(b),
        'note':'One-to-one Hungarian matching maximizes Pearson correlation of channel mean responses to identical probes. Correspondence is descriptive, not semantic identity.'}


def _cka_features(x: torch.Tensor,y: torch.Tensor) -> float:
    # Centered, biased linear CKA, with >= 4 probe samples. No cross-run row permutation.
    a=dyn.centered_gram(x)
    b=dyn.centered_gram(y)
    return float(dyn.linear_cka_gram(a,b))


def compare_runs(left: str,right: str,step_left: int,step_right: int,layer: int,
                 bootstrap: int=60,seed: int=2026):
    x,y=_aligned(left,right)
    _check(left,step_left,layer);_check(right,step_right,layer)
    a=dyn.activations_at(left,step_left)[1][layer-1].reshape(len(x),-1)
    b=dyn.activations_at(right,step_right)[1][layer-1].reshape(len(x),-1)
    point=_cka_features(a,b)
    if not 10<=bootstrap<=200: raise ValueError('Bootstrap replicates must be 10..200')
    rng=np.random.default_rng(seed)
    vals=[]
    for _ in range(bootstrap):
        indices=torch.from_numpy(rng.integers(0,len(x),len(x))).long()
        vals.append(_cka_features(a[indices],b[indices]))
    lo,hi=np.quantile(vals,[.025,.975])
    matches=match_channels(left,right,step_left,step_right,layer)
    return {'run_a':left,'run_b':right,'step_a':step_left,'step_b':step_right,'layer':layer,
      'probe_fingerprint':_fingerprint(x,y),'probe_count':len(x),
      'cka':round(point,6),'bootstrap_ci95':[round(float(lo),6),round(float(hi),6)],
      'replicates':bootstrap,'bootstrap_seed':seed,
      'channel_mean_corr':matches['mean_matched_corr'],'matches':matches['top_matches'],
      'warning':'Percentile bootstrap over shared probes; conditional on this finite dataset and model pair. Not independent multi-seed confidence interval; matching correlations do not establish identical semantics.'}


def _cam(model: torch.nn.Module, img: torch.Tensor, target: int):
    with torch.no_grad():
        logits, features=model.forward_trace(img[None])
        last=features[-1][0]
        coeff=model.fc.weight[target].detach()
        cam=F.relu((last*coeff[:,None,None]).sum(0))
        return logits.softmax(-1)[0],F.interpolate(cam[None,None],size=img.shape[-2:],mode='bilinear',align_corners=False)[0,0]


def faithfulness(name: str,step: int,sample_index: int,target_class: int,
                 grid: int=4,random_trials: int=8,seed: int=42):
    """Patch deletion: class probability after cumulative occlusion; random-order control."""
    _check(name,step,5)
    x,y=_samples(name)
    if not 0<=sample_index<len(x):raise ValueError('Sample out of range')
    model=dyn.model_at(name,step)
    nclasses=len(dyn.load_run(name)['classes'])
    if not 0<=target_class<nclasses:raise ValueError('Target out of range')
    if grid not in (4,8):raise ValueError('Patch grid must be 4 or 8')
    if not 2<=random_trials<=16:raise ValueError('Invalid number of random trials')
    img=x[sample_index]
    probabilities,cam=_cam(model,img,target_class)
    h,w=img.shape[-2:]
    boxes=[];scores=[]
    for i in range(grid):
        for j in range(grid):
            y0=i*h//grid;y1=(i+1)*h//grid;x0=j*w//grid;x1=(j+1)*w//grid
            boxes.append((y0,y1,x0,x1))
            scores.append(float(cam[y0:y1,x0:x1].mean()))
    order=np.argsort(-np.array(scores),kind='stable')
    # report every 1/8 coverage; full sweep is possible but slower for larger checkpoints
    counts=sorted(set([0,*[math.ceil(len(boxes)*t/8) for t in range(1,9)]]))
    schedules=[order]+[np.random.default_rng(seed+z).permutation(len(boxes)) for z in range(random_trials)]
    inputs=[]
    for schedule in schedules:
        for count in counts:
            new=img.clone()
            for idx in schedule[:count]:
                y0,y1,x0,x1=boxes[int(idx)]
                new[:,y0:y1,x0:x1]=0 # zero-filled OOD intervention
            inputs.append(new)
    with torch.no_grad():
        probs=model(torch.stack(inputs)).softmax(1)[:,target_class].reshape(len(schedules),len(counts)).cpu().numpy()
    ranked=probs[0];control=probs[1:].mean(0)
    fracs=np.array(counts,dtype=np.float64)/len(boxes)
    ranked_auc=float(np.trapezoid(ranked,fracs))
    control_auc=float(np.trapezoid(control,fracs))
    return {'run_id':name,'step':step,'sample_index':sample_index,'label':int(y[sample_index]),
        'target_class':target_class,'model_prediction':int(probabilities.argmax()),
        'probability_before':round(float(probabilities[target_class]),6),
        'fractions':fracs.tolist(),'top_cam_deletion':[round(float(z),6) for z in ranked],
        'random_deletion_mean':[round(float(z),6) for z in control],
        'auc_top_cam':round(ranked_auc,6),'auc_random':round(control_auc,6),
        'auc_random_minus_cam':round(control_auc-ranked_auc,6),
        'random_trials':random_trials,'grid':grid,
        'note':'Lower CAM-order AUC than random suggests concentration on influential patches; zero occlusion is OOD and results are input/target-specific. This is NOT proof of causal neuron semantics.'}


def matched_examples(left: str,right: str,step_left: int,step_right: int,
                     layer: int,channel_left: int,channel_right: int,k: int=6):
    """Paired Top-k panels show whether correlated units prefer the SAME samples."""
    _aligned(left,right)
    a=rank_concepts(left,step_left,layer,channel_left,k)
    b=rank_concepts(right,step_right,layer,channel_right,k)
    ids_a={item['probe_index'] for item in a['top_k']}
    ids_b={item['probe_index'] for item in b['top_k']}
    intersection=sorted(ids_a&ids_b)
    return {'run_a':left,'run_b':right,'layer':layer,'channel_a':channel_left,'channel_b':channel_right,
         'topk_a':a['top_k'],'topk_b':b['top_k'],
         'intersection':intersection,'intersection_size':len(intersection),
         'jaccard':round(len(intersection)/max(1,len(ids_a|ids_b)),5),
         'dominant_label_a':a['dominant_label'],'dominant_label_b':b['dominant_label'],
         'note':'Same held-out probe IDs; Top-k overlap and label agreement measure response stability, not equivalence of semantic concepts.'}