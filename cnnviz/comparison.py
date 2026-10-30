"""Scientifically comparable activation rendering across checkpoint time.

Fixed scales are computed per input sample and layer across EVERY checkpoint
and channel. They are never inferred from one selected epoch or image tile.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch

from .imaging import heatmap


@dataclass(frozen=True)
class ActivationScale:
    low: float
    high: float
    scope: str = "sample_layer_all_epochs_all_channels"

    def as_dict(self) -> dict:
        return {"low": self.low, "high": self.high, "scope": self.scope}


def render_activation(values: torch.Tensor, *, scale: ActivationScale, mode: str, upscale: int = 8) -> str:
    if mode == "fixed":
        return heatmap(values, low=scale.low, high=scale.high, upscale=upscale)
    if mode == "local":
        return heatmap(values, upscale=upscale)
    raise ValueError("scale_mode must be fixed or local")


def compare_activation(first: torch.Tensor, second: torch.Tensor, scale: ActivationScale) -> dict:
    """Difference B-A. Signed zero-centred limits held fixed to the layer's scale.

    The raw tensors (not rendered PNGs) are used for all statistics.
    """
    if first.shape != second.shape:
        raise ValueError("Compared activations must have the same spatial grid")
    a = first.detach().float()
    b = second.detach().float()
    d = b - a
    a_norm = float(a.square().sum().sqrt())
    b_norm = float(b.square().sum().sqrt())
    cosine = float((a * b).sum() / (a_norm * b_norm)) if min(a_norm, b_norm) > 1e-12 else None
    bound = max(abs(scale.low), abs(scale.high), 1e-9)
    return {
        "difference": heatmap(d, signed=True, signed_limit=bound, upscale=max(6, 128 // a.shape[-1])),
        "delta_mean": float(d.mean()),
        "rms_difference": float(d.square().mean().sqrt()),
        "l1_mean_difference": float(d.abs().mean()),
        "cosine_similarity": cosine,
        "max_absolute_difference": float(d.abs().max()),
        "difference_scale": {"low": -bound, "high": bound, "zero": 0.0,
                             "scope": "fixed_symmetric_layer_scale"},
    }