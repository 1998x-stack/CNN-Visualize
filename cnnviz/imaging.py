"""PNG visual encoders. Consistent coloring across epochs/layers is explicit."""
from __future__ import annotations
import base64
import io
from PIL import Image
import numpy as np
import torch
from torch.nn import functional as F


def png_data(array: np.ndarray, mode: str = "RGB", upscale: int = 1) -> str:
    arr = np.asarray(array).astype("uint8")
    im = Image.fromarray(arr, mode=mode)
    if upscale > 1:
        im = im.resize((im.width * upscale, im.height * upscale), Image.Resampling.NEAREST)
    buffer = io.BytesIO()
    im.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def normalize(t: torch.Tensor | np.ndarray, low: float | None = None, high: float | None = None) -> np.ndarray:
    a = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    a = a.astype("float32")
    lo = float(np.min(a)) if low is None else low
    hi = float(np.max(a)) if high is None else high
    if hi - lo < 1e-9:
        return np.zeros_like(a, dtype="float32")
    return np.clip((a - lo) / (hi - lo), 0, 1)


def heatmap(t: torch.Tensor | np.ndarray, signed: bool = False, upscale: int = 8) -> str:
    a = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    if signed:
        scale = max(float(np.max(np.abs(a))), 1e-9)
        v = np.clip(a / scale, -1, 1)
        rgb = np.stack([242 - 135 * np.maximum(-v, 0), 242 - 175 * np.abs(v), 242 - 125 * np.maximum(v, 0)], axis=-1)
    else:
        v = normalize(a)
        # ink-blue → periwinkle → warm coral, high contrast on a white UI
        keys = np.array([[15, 23, 42], [54, 85, 160], [128, 112, 221], [239, 142, 136], [255, 221, 177]])
        tscaled = v * (len(keys) - 1)
        k = np.minimum(tscaled.astype(int), len(keys) - 2)
        rgb = keys[k] * (1 - (tscaled - k))[..., None] + keys[k + 1] * (tscaled - k)[..., None]
    return png_data(np.uint8(np.clip(rgb, 0, 255)), upscale=upscale)


def input_image(x: torch.Tensor, upscale: int = 12) -> str:
    a = x.detach().cpu().numpy()
    if a.shape[0] == 1:
        v = np.uint8(np.clip(a[0] * 255, 0, 255))
        rgb = np.stack([v, v, v], axis=-1)
    else:
        rgb = np.uint8(np.clip(np.moveaxis(a, 0, -1) * 255, 0, 255))
    return png_data(rgb, upscale=upscale)


def resized_heat(a: torch.Tensor, size: tuple[int, int] = (16, 16)) -> torch.Tensor:
    return F.interpolate(a[None, None], size=size, mode="bilinear", align_corners=False)[0, 0]