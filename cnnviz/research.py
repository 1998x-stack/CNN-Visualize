"""V0.4 research experiments, independent of legacy V0.3 checkpoint semantics.

This is a small AlexNet/ZFNet-*inspired* 5-convolution model for 32x32 inputs,
NOT a reproduction of their ImageNet architectures or published accuracy.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from .steps import parameter_statistics, state_copy
from .data import LABELS_CIFAR, LABELS_DIGITS


class FiveLayerCNN(nn.Module):
    """5 conv layers. Early sampling is the controlled architecture difference."""
    def __init__(self, channels: int = 1, classes: int = 10, variant: str = 'zf-small'):
        super().__init__()
        if variant not in ('zf-small', 'alex-small'):
            raise ValueError('Expected zf-small or alex-small')
        self.variant = variant
        self.conv1 = nn.Conv2d(channels, 24, 7 if variant == 'zf-small' else 11,
                               stride=2 if variant == 'zf-small' else 4,
                               padding=3 if variant == 'zf-small' else 5)
        self.conv2 = nn.Conv2d(24, 32, 5, padding=2)
        self.conv3 = nn.Conv2d(32, 48, 3, padding=1)
        self.conv4 = nn.Conv2d(48, 48, 3, padding=1)
        self.conv5 = nn.Conv2d(48, 64, 3, padding=1)
        self.fc = nn.Linear(64, classes)

    def forward_trace(self, x):
        a1 = F.relu(self.conv1(x))
        a2 = F.relu(self.conv2(F.max_pool2d(a1, 2)))
        a3 = F.relu(self.conv3(F.max_pool2d(a2, 2)))
        a4 = F.relu(self.conv4(a3))
        a5 = F.relu(self.conv5(a4))
        return self.fc(a5.mean(dim=(-2, -1))), [a1, a2, a3, a4, a5]

    def forward(self, x):
        return self.forward_trace(x)[0]


def research_datasets(name: str, root='data', download=False, seed=42):
    """Use native CIFAR-10 32x32; Digits 8x8 -> 32x32 is *explicitly interpolated*."""
    if name == 'digits':
        raw = load_digits()
        xx = torch.from_numpy(np.ascontiguousarray(raw.images.astype('float32') / 16))[:, None]
        xx = F.interpolate(xx, size=(32, 32), mode='nearest')
        yy = torch.tensor(raw.target, dtype=torch.long)
        tr, val = train_test_split(np.arange(len(yy)), test_size=.2,
                                   random_state=seed, stratify=raw.target)
        return (TensorDataset(xx[tr], yy[tr]), TensorDataset(xx[val], yy[val]),
               LABELS_DIGITS, {'native_resolution': [8, 8], 'model_resolution': [32, 32],
                               'interpolated': True, 'split': 'stratified_80_20'})
    if name == 'cifar10':
        from torchvision.datasets import CIFAR10
        from torchvision.transforms import ToTensor
        transform = ToTensor()
        train = CIFAR10(root=root, train=True, download=download, transform=transform)
        test = CIFAR10(root=root, train=False, download=download, transform=transform)
        return train, test, LABELS_CIFAR, {'native_resolution': [32, 32],
            'model_resolution': [32, 32], 'interpolated': False, 'split': 'official_train_test'}
    raise ValueError('Unknown dataset')


def _deterministic_subset(dataset, limit, seed):
    if not limit or limit >= len(dataset):
        return dataset
    # Same subset within a run, reproducibly selected independently of DataLoader shuffle.
    idx = np.random.default_rng(seed).permutation(len(dataset))[:limit].tolist()
    return Subset(dataset, idx)


def _tensor_hist(tensor, bins=20):
    values = tensor.detach().float().flatten()
    if not values.numel(): return {'counts': [], 'edges': []}
    lo, hi = float(values.min()), float(values.max())
    if lo == hi: hi = lo + 1e-6
    counts = torch.histc(values, bins=bins, min=lo, max=hi)
    return {'counts': [int(i) for i in counts.tolist()],
            'edges': [round(float(v), 7) for v in torch.linspace(lo, hi, bins+1).tolist()]}


def run_research(*, name='research-digits-zf', dataset='digits', variant='zf-small',
                 epochs=15, seed=42, train_limit=1024, val_limit=160, interval=16,
                 root='data', download=False, out='artifacts/dynamics', split_seed=None, probe_count=40):
    if not name or not all(c.isalnum() or c in '-_' for c in name) or len(name) > 60:
        raise ValueError('run name must be a simple slug (max 60 chars)')
    if epochs < 1 or epochs > 100 or interval < 1 or interval > 10000 or not 4 <= probe_count <= 512:
        raise ValueError('Invalid epochs or interval')
    torch.set_num_threads(min(4, torch.get_num_threads()))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    split_seed = seed if split_seed is None else split_seed
    train_set, val_set, labels, provenance = research_datasets(dataset, root, download, split_seed)
    train_set = _deterministic_subset(train_set, train_limit, split_seed)
    val_set = _deterministic_subset(val_set, val_limit, split_seed + 1)
    channels = 1 if dataset == 'digits' else 3
    model = FiveLayerCNN(channels, len(labels), variant)
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    loader = DataLoader(train_set, batch_size=64, shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0)
    val_loader = DataLoader(val_set, batch_size=80, shuffle=False, num_workers=0)
    directory = Path(out) / name
    directory.mkdir(parents=True, exist_ok=True)
    events, snapshots, epochs_metrics = [], [], []
    global_step = 0
    torch.save(state_copy(model), directory / 'step_000000.pt')
    snapshots.append({'step': 0, 'epoch': 0, 'path': 'step_000000.pt', 'reason': 'initialization'})
    for ep in range(1, epochs+1):
        model.train()
        for batch, (x, y) in enumerate(loader, 1):
            before = {n: p.detach().clone() for n,p in model.named_parameters()}
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            # Snapshot gradient distribution before any clipping.
            layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc']
            preclip_grad = {k: math.sqrt(sum(float(p.grad.detach().double().square().sum())
                    for p in getattr(model, k).parameters() if p.grad is not None)) for k in layers}
            save_step = (global_step + 1) % interval == 0 or batch == len(loader)
            grad_hist = {k: _tensor_hist(getattr(model,k).weight.grad) for k in layers} if save_step else None
            norm = float(nn.utils.clip_grad_norm_(model.parameters(), 5.0))
            optimizer.step()
            global_step += 1
            delta = parameter_statistics(model, before)
            event = {'step': global_step, 'epoch': ep, 'batch': batch,
                     'batch_loss_preupdate': float(loss.detach()),
                     'batch_accuracy_preupdate': float((logits.argmax(1)==y).float().mean()),
                     'gradient_global_preclip': norm, 'gradient_layers_preclip': preclip_grad,
                     'update_global_l2': delta['global'], 'update_layers_l2': delta['by_layer'],
                     'gradient_histogram': grad_hist,
                     'has_checkpoint': global_step % interval == 0 or batch == len(loader)}
            events.append(event)
            if event['has_checkpoint']:
                fname = f'step_{global_step:06d}.pt'
                torch.save(state_copy(model), directory / fname)
                snapshots.append({'step': global_step, 'epoch': ep, 'path': fname,
                                  'reason': 'epoch_end' if batch == len(loader) else 'interval'})
        model.eval()
        correct=total=0; cumulative=0.
        with torch.no_grad():
            for x,y in val_loader:
                logits=model(x)
                cumulative += float(F.cross_entropy(logits,y,reduction='sum'))
                correct += int((logits.argmax(1)==y).sum()); total+=len(y)
        epochs_metrics.append({'epoch': ep, 'val_accuracy':correct/total,'val_loss':cumulative/total})
        print(f'{name} epoch {ep}: val_acc {correct/total:.1%}', flush=True)
    # A fixed held-out probe set shared by all checkpoints; its labels are not used for fitting.
    probe_idx = list(range(min(probe_count,len(val_set))))
    manifest = {'schema': 1, 'run_id': name, 'dataset': dataset, 'variant': variant,
        'model_family': '5-layer AlexNet/ZFNet-inspired 32px educational CNN; NOT original ImageNet architecture',
        'channels': channels, 'classes': labels, 'seed': seed, 'epochs': epochs,
        'train_count': len(train_set), 'validation_count': len(val_set), 'split_seed': split_seed,
        'data_root': str(root),
        'environment': {'torch': torch.__version__, 'numpy': np.__version__, 'sklearn': sklearn.__version__, 'python': __import__('platform').python_version()},
        'checkpoint_sha256': {s['path']: hashlib.sha256((directory / s['path']).read_bytes()).hexdigest() for s in snapshots},
        'probe_indices': probe_idx, 'provenance': provenance,
        'snapshots': snapshots, 'events': events, 'metrics': epochs_metrics,
        'config': {'batch_size': 64, 'optimizer': 'Adam', 'lr': .003,
                   'clip_norm': 5, 'step_interval': interval, 'train_limit':train_limit, 'val_limit':val_limit},
        'interpretation': {'cka': 'Centered biased linear CKA on SAME held-out probe images, flattened spatial activations; descriptive similarity, not causal importance.',
           'clusters': 'Within-checkpoint response-vector clustering of channels, not fixed semantic entities across runs.',
           'dataset': 'Digits images are upsampled from 8x8; only CIFAR-10 offers genuine 32x32 source content.'}}
    tmp=directory/'manifest.json.tmp';tmp.write_text(json.dumps(manifest,ensure_ascii=False),encoding='utf-8');tmp.replace(directory/'manifest.json')
    return manifest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--name', default='research-digits-zf')
    parser.add_argument('--dataset', choices=['digits','cifar10'], default='digits')
    parser.add_argument('--variant', choices=['zf-small','alex-small'], default='zf-small')
    parser.add_argument('--epochs',type=int,default=15)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--train-limit',type=int,default=1024)
    parser.add_argument('--val-limit',type=int,default=160)
    parser.add_argument('--interval',type=int,default=16)
    parser.add_argument('--root',default='data')
    parser.add_argument('--download',action='store_true')
    parser.add_argument('--out',default='artifacts/dynamics')
    parser.add_argument('--split-seed',type=int,default=None,help='Fix held-out split across different training seeds')
    parser.add_argument('--probe-count',type=int,default=40,help='Held-out examples used for all concept analyses')
    run_research(**vars(parser.parse_args()))

if __name__=='__main__':main()