"""FastAPI read-only checkpoint explorer and a memory-only image upload endpoint."""
from __future__ import annotations
import io
import json
import os
import uuid
from functools import lru_cache
from pathlib import Path
from threading import Lock

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from torch.nn import functional as F

from .data import get_dataset
from .imaging import heatmap, input_image, normalize, resized_heat
from .model import InspectableCNN, deconv_projection

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = Path(os.getenv("CNNVIZ_ARTIFACTS", str(ROOT / "artifacts" / "demo")))
app = FastAPI(title="CNN Visualize · Learning Journey", version="0.1.0")
_uploaded: dict[str, torch.Tensor] = {}
_upload_lock = Lock()


@lru_cache(maxsize=1)
def context():
    path = ARTIFACT_DIR / "manifest.json"
    if not path.is_file():
        raise RuntimeError("No training artifacts. Run: python -m cnnviz.train --epochs 10")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    data, targets, _, _, _, _ = get_dataset(manifest["dataset"], download=False, seed=manifest["seed"])
    return manifest, data, targets


@lru_cache(maxsize=8)
def checkpoint(epoch: int):
    manifest, _, _ = context()
    if not (0 <= epoch <= manifest["epochs"]):
        raise ValueError("Invalid epoch")
    filename = ARTIFACT_DIR / f"epoch_{epoch:03d}.pt"
    if not filename.is_file():
        raise ValueError("Checkpoint missing")
    model = InspectableCNN(channels=manifest["channels"], classes=len(manifest["classes"]))
    model.load_state_dict(torch.load(filename, map_location="cpu", weights_only=True))
    model.eval()
    return model


def sample_tensor(sample_id: str):
    manifest, data, targets = context()
    if sample_id.startswith("upload-"):
        with _upload_lock:
            if sample_id not in _uploaded:
                raise ValueError("Uploaded image has expired")
            return _uploaded[sample_id], None
    if not sample_id.isascii() or not sample_id.isdigit():
        raise ValueError("Invalid sample id")
    allowed = {s["id"] for s in manifest["samples"]}
    if sample_id not in allowed:
        raise ValueError("Sample not in gallery")
    index = int(sample_id)
    return data[index:index + 1], int(targets[index])


def render_feature_map(feature: torch.Tensor) -> str:
    return heatmap(feature, upscale=max(6, 128 // feature.shape[-1]))


def explanation(sample_id: str, epoch: int, layer: int, channel: int):
    if layer not in (1, 2, 3):
        raise ValueError("Layer must be 1,2,3")
    model = checkpoint(epoch)
    tensor, target = sample_tensor(sample_id)
    x = tensor.detach().clone().requires_grad_(True)
    logits, trace = model.forward_trace(x)
    probs = logits.softmax(1)[0]
    predicted = int(probs.argmax())
    maps = trace["features"][layer - 1]
    if channel < 0 or channel >= maps.shape[1]:
        raise ValueError("Channel out of range")
    # Class-conditional derivatives use predicted score, not the supplied label.
    last = trace["features"][-1]
    last.retain_grad()
    logits[0, predicted].backward()
    grad = x.grad.detach().abs()[0].amax(dim=0)
    weights = last.grad.detach().mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * last.detach()).sum(dim=1))[0]
    cam = resized_heat(cam, size=(16, 16))
    with torch.no_grad():
        projected = deconv_projection(model, trace, layer, channel)[0].mean(dim=0)
        selected_conv = (model.conv1, model.conv2, model.conv3)[layer - 1]
        kernel = selected_conv.weight[channel].detach().mean(dim=0)
        # One 4x4 mask per position: batched, avoids 49 sequential model runs.
        positions = [(r, c) for r in range(0, 13, 2) for c in range(0, 13, 2)]
        blocked = x.detach().repeat(len(positions), 1, 1, 1)
        for i, (r, c) in enumerate(positions):
            blocked[i, :, r:r + 4, c:c + 4] = 0.0
        shifted = model(blocked).softmax(1)[:, predicted]
        drop = (probs[predicted].detach() - shifted).cpu().numpy()
        occlusion = np.zeros((16, 16), dtype="float32")
        counts = np.zeros_like(occlusion)
        for (r, c), value in zip(positions, drop):
            occlusion[r:r + 4, c:c + 4] += float(value)
            counts[r:r + 4, c:c + 4] += 1
        occlusion /= np.maximum(counts, 1)
        occlusion = np.maximum(occlusion, 0)  # only probability decreases are emphasized
    active = maps[0, channel].detach()
    top = torch.topk(probs, k=min(5, len(probs)))
    return {"sample_id": sample_id, "epoch": epoch, "layer": layer, "channel": channel,
            "true_class": target, "predicted_class": predicted, "confidence": round(float(probs[predicted].detach()), 6),
            "correct": None if target is None else bool(target == predicted),
            "probabilities": [round(float(v), 6) for v in probs.detach()],
            "top_predictions": [{"index": int(k), "probability": round(float(v.detach()), 6)} for k, v in zip(top.indices, top.values)],
            "input": input_image(tensor[0]),
            "feature_maps": [render_feature_map(feat) for feat in maps[0].detach()],
            "feature": render_feature_map(active), "filter": heatmap(kernel, signed=True, upscale=24),
            "deconv": heatmap(projected, signed=True, upscale=12),
            "saliency": heatmap(grad, upscale=12), "gradcam": heatmap(cam, upscale=12),
            "occlusion": heatmap(occlusion, upscale=12),
            "activation_mean": round(float(active.mean()), 6),
            "activation_max": round(float(active.max()), 6),
            "activation_sparsity": round(float((active == 0).float().mean()), 6),
            "feature_shape": list(maps.shape[1:]),
            "warnings": {"deconv": "Single-peak feature projection, not a class attribution or true image reconstruction.",
                         "gradcam": "Class-conditional map for the currently predicted class.",
                         "occlusion": "16x16 input, 4x4 zero patch with stride 2; values are probability decreases."}}


@app.get("/api/health")
def health():
    return {"ok": True, "has_artifacts": (ARTIFACT_DIR / "manifest.json").is_file()}


@app.get("/api/manifest")
def manifest():
    try:
        meta, data, _ = context()
        obj = dict(meta)
        obj["samples"] = [dict(s, image=input_image(data[int(s["id"])], upscale=8)) for s in meta["samples"]]
        return obj
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/trajectory")
def trajectory(sample_id: str):
    try:
        meta, _, _ = context()
        tensor, label = sample_tensor(sample_id)
        evolution = []
        with torch.no_grad():
            for epoch in range(meta["epochs"] + 1):
                probs = checkpoint(epoch)(tensor).softmax(1)[0]
                prediction = int(probs.argmax())
                evolution.append({"epoch": epoch, "predicted_class": prediction,
                                  "confidence": round(float(probs[prediction]), 6),
                                  "true_class_probability": None if label is None else round(float(probs[label]), 6),
                                  "probabilities": [round(float(v), 6) for v in probs]})
        return {"sample_id": sample_id, "true_class": label, "evolution": evolution}
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/inspect")
def inspect(sample_id: str, epoch: int = Query(0, ge=0),
            layer: int = Query(1, ge=1, le=3), channel: int = Query(0, ge=0)):
    try:
        return explanation(sample_id, epoch, layer, channel)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    raw = await file.read(5 * 1024 * 1024 + 1)
    if len(raw) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Maximum upload size is 5MB")
    try:
        with Image.open(io.BytesIO(raw)) as img:
            img.verify()
        with Image.open(io.BytesIO(raw)) as img:
            img = ImageOps.exif_transpose(img)
            if img.width * img.height > 20_000_000:
                raise HTTPException(status_code=413, detail="Image resolution too large")
            channels = context()[0]["channels"]
            img = img.convert("RGB" if channels == 3 else "L")
            img = ImageOps.fit(img, (16, 16), method=Image.Resampling.BILINEAR)
            a = np.array(img, dtype="float32") / 255.0
            if channels == 1:
                x = torch.from_numpy(a[None, None])
            else:
                x = torch.from_numpy(np.moveaxis(a, -1, 0)[None])
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=415, detail="Invalid image") from exc
    sample_id = "upload-" + uuid.uuid4().hex[:16]
    with _upload_lock:
        if len(_uploaded) >= 30:
            _uploaded.pop(next(iter(_uploaded)))
        _uploaded[sample_id] = x
    return {"id": sample_id, "label": "Custom image", "label_index": None, "image": input_image(x[0], upscale=8)}


@app.get("/")
def homepage():
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/app.css")
def stylesheet():
    return FileResponse(ROOT / "web" / "app.css", media_type="text/css")


@app.get("/app.js")
def javascript():
    return FileResponse(ROOT / "web" / "app.js", media_type="text/javascript")