"""Step checkpoints and training-event metadata; compatible with Epoch-only runs.

A model snapshot is a post-optimizer-step state, never a synthetic interpolation.
Events are written for every optimizer step, while snapshots are deliberately sparse.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch


def checkpoint_steps(run_dir: Path) -> dict:
    path = run_dir / "steps.json"
    if not path.exists():
        return {"schema": 1, "events": [], "snapshots": []}
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("schema") != 1 or not isinstance(doc.get("events"), list) or not isinstance(doc.get("snapshots"), list):
        raise ValueError("Invalid step timeline schema")
    return doc


def parameter_statistics(model: torch.nn.Module, before: dict[str, torch.Tensor]) -> dict:
    """Actual parameter delta induced by optimizer.step(), not lr * gradient."""
    totals: dict[str, float] = {}
    global_sq = 0.0
    for name, p in model.named_parameters():
        difference = p.detach() - before[name]
        sq = float(difference.double().square().sum())
        global_sq += sq
        group = name.split(".")[0]
        totals[group] = totals.get(group, 0.0) + sq
    return {"global": math.sqrt(global_sq), "by_layer": {name: math.sqrt(v) for name, v in totals.items()}}


def gradient_statistics(model: torch.nn.Module) -> dict:
    norms = {}
    for name, module in (("conv1", model.conv1), ("conv2", model.conv2),
                         ("conv3", model.conv3), ("fc", model.fc)):
        norms[name] = math.sqrt(sum(float(p.grad.detach().double().square().sum())
                                    for p in module.parameters() if p.grad is not None))
    return norms


def state_copy(model: torch.nn.Module) -> dict:
    return {name: p.detach().clone().cpu() for name, p in model.state_dict().items()}