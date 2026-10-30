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

from .comparison import ActivationScale, compare_activation, render_activation
from .data import get_dataset
from .imaging import heatmap, input_image, normalize, resized_heat
from .model import InspectableCNN, deconv_projection

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = Path(os.getenv("CNNVIZ_ARTIFACTS", str(ROOT / "artifacts" / "demo")))
app = FastAPI(title="CNN Visualize · Learning Journey", version="0.3.0")
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


@lru_cache(maxsize=96)
def fixed_activation_scale(sample_id: str, layer: int) -> ActivationScale:
    """One deterministic color domain for sample/layer across all channels/epochs.

    Cache is bounded and only contains scale values, never uploaded pixels.
    Re-run the server if a training artifact is replaced in place.
    """
    manifest, _, _ = context()
    tensor, _ = sample_tensor(sample_id)
    maximum = 0.0
    with torch.no_grad():
        for epoch in range(manifest["epochs"] + 1):
            _, trace = checkpoint(epoch).forward_trace(tensor)
            maximum = max(maximum, float(trace["features"][layer - 1].max()))
    return ActivationScale(low=0.0, high=max(maximum, 1e-9))


def render_feature_map(feature: torch.Tensor, *, scale: ActivationScale, mode: str) -> str:
    return render_activation(feature, scale=scale, mode=mode,
                             upscale=max(6, 128 // feature.shape[-1]))


def explanation(sample_id: str, epoch: int, layer: int, channel: int,
                target_class: int | None = None, scale_mode: str = "fixed"):
    if layer not in (1, 2, 3):
        raise ValueError("Layer must be 1,2,3")
    if scale_mode not in ("fixed", "local"):
        raise ValueError("scale_mode must be fixed or local")
    model = checkpoint(epoch)
    tensor, target = sample_tensor(sample_id)
    x = tensor.detach().clone().requires_grad_(True)
    logits, trace = model.forward_trace(x)
    probs = logits.softmax(1)[0]
    predicted = int(probs.argmax())
    if target_class is not None and not (0 <= target_class < probs.shape[0]):
        raise ValueError("Target class out of range")
    attribution_target = predicted if target_class is None else int(target_class)
    maps = trace["features"][layer - 1]
    if channel < 0 or channel >= maps.shape[1]:
        raise ValueError("Channel out of range")
    # Every attribution below is conditioned on the *same explicit* target.
    # Predictions and accuracy remain based on actual argmax, independently.
    last = trace["features"][-1]
    last.retain_grad()
    logits[0, attribution_target].backward()
    grad = x.grad.detach().abs()[0].amax(dim=0)
    weights = last.grad.detach().mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * last.detach()).sum(dim=1))[0]
    cam = resized_heat(cam, size=(16, 16))
    with torch.no_grad():
        projected = deconv_projection(model, trace, layer, channel)[0].mean(dim=0)
        selected_conv = (model.conv1, model.conv2, model.conv3)[layer - 1]
        kernel = selected_conv.weight[channel].detach().mean(dim=0)
        positions = [(r, c) for r in range(0, 13, 2) for c in range(0, 13, 2)]
        blocked = x.detach().repeat(len(positions), 1, 1, 1)
        for i, (r, c) in enumerate(positions):
            blocked[i, :, r:r + 4, c:c + 4] = 0.0
        shifted = model(blocked).softmax(1)[:, attribution_target]
        drop = (probs[attribution_target].detach() - shifted).cpu().numpy()
        occlusion = np.zeros((16, 16), dtype="float32")
        counts = np.zeros_like(occlusion)
        for (r, c), value in zip(positions, drop):
            occlusion[r:r + 4, c:c + 4] += float(value)
            counts[r:r + 4, c:c + 4] += 1
        occlusion /= np.maximum(counts, 1)
        occlusion = np.maximum(occlusion, 0)
    scale = fixed_activation_scale(sample_id, layer) if scale_mode == "fixed" else ActivationScale(0, 1, "per_tile_dynamic")
    active = maps[0, channel].detach()
    top = torch.topk(probs, k=min(5, len(probs)))
    return {"sample_id": sample_id, "epoch": epoch, "layer": layer, "channel": channel,
            "true_class": target, "predicted_class": predicted,
            "attribution_target_class": attribution_target,
            "attribution_target_probability": round(float(probs[attribution_target].detach()), 6),
            "attribution_mode": "predicted" if target_class is None else "fixed",
            "confidence": round(float(probs[predicted].detach()), 6),
            "correct": None if target is None else bool(target == predicted),
            "probabilities": [round(float(v), 6) for v in probs.detach()],
            "top_predictions": [{"index": int(k), "probability": round(float(v.detach()), 6)} for k, v in zip(top.indices, top.values)],
            "input": input_image(tensor[0]),
            "feature_maps": [render_feature_map(feat, scale=scale, mode=scale_mode) for feat in maps[0].detach()],
            "feature": render_feature_map(active, scale=scale, mode=scale_mode),
            "feature_scale": scale.as_dict() if scale_mode == "fixed" else {"scope": "per_tile_dynamic"},
            "scale_mode": scale_mode,
            "filter": heatmap(kernel, signed=True, upscale=24),
            "deconv": heatmap(projected, signed=True, upscale=12),
            "saliency": heatmap(grad, upscale=12), "gradcam": heatmap(cam, upscale=12),
            "occlusion": heatmap(occlusion, upscale=12),
            "activation_mean": round(float(active.mean()), 6),
            "activation_max": round(float(active.max()), 6),
            "activation_sparsity": round(float((active == 0).float().mean()), 6),
            "feature_shape": list(maps.shape[1:]),
            "warnings": {"deconv": "Single-peak feature projection, not a class attribution or true reconstruction.",
                         "gradcam": "Class-conditional on attribution_target_class; overlay is per-image normalized and not quantitative across epochs.",
                         "occlusion": "16x16 input, 4x4 zero patch stride 2; positive probability drops for attribution_target_class.",
                         "feature": "Fixed: common [0,max] over every epoch/channel for the selected sample and layer; Local: each tile rescaled separately."}}


def comparison(sample_id: str, epoch_a: int, epoch_b: int, layer: int, channel: int,
               scale_mode: str = "fixed"):
    if layer not in (1, 2, 3) or scale_mode not in ("fixed", "local"):
        raise ValueError("Invalid layer or scale mode")
    tensor, _ = sample_tensor(sample_id)
    with torch.no_grad():
        _, trace_a = checkpoint(epoch_a).forward_trace(tensor)
        _, trace_b = checkpoint(epoch_b).forward_trace(tensor)
        channels = trace_a["features"][layer - 1].shape[1]
        if not (0 <= channel < channels):
            raise ValueError("Channel out of range")
        a = trace_a["features"][layer - 1][0, channel]
        b = trace_b["features"][layer - 1][0, channel]
        scale = fixed_activation_scale(sample_id, layer)
        stats = compare_activation(a, b, scale)
        return {"sample_id": sample_id, "layer": layer, "channel": channel,
                "epoch_a": epoch_a, "epoch_b": epoch_b, "scale_mode": scale_mode,
                "scale": scale.as_dict() if scale_mode == "fixed" else {"scope": "per_tile_dynamic"},
                "a": render_feature_map(a, scale=scale, mode=scale_mode),
                "b": render_feature_map(b, scale=scale, mode=scale_mode),
                **stats,
                "note": "B−A on the same sample, layer and channel; signed difference is fixed to ±layer max. No causal claim."}


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


@app.get("/api/channel-trajectory")
def channel_trajectory(sample_id: str, layer: int = Query(1, ge=1, le=3),
                       channel: int = Query(0, ge=0)):
    """Absolute channel statistics across *real* checkpoints for the same input.

    Unlike per-image normalized heatmaps, mean activations here can be compared
    numerically across epochs, because all checkpoints share one model lineage.
    """
    try:
        meta, _, _ = context()
        tensor, _ = sample_tensor(sample_id)
        history = []
        prev = None
        with torch.no_grad():
            for epoch in range(meta["epochs"] + 1):
                model = checkpoint(epoch)
                _, trace = model.forward_trace(tensor)
                feature = trace["features"][layer - 1]
                if channel >= feature.shape[1]:
                    raise ValueError("Channel out of range")
                active = feature[0, channel]
                filter_weights = (model.conv1, model.conv2, model.conv3)[layer - 1].weight[channel]
                drift = 0.0 if prev is None else float((active - prev).square().mean().sqrt())
                history.append({"epoch": epoch, "mean": round(float(active.mean()), 6),
                                "max": round(float(active.max()), 6),
                                "sparsity": round(float((active == 0).float().mean()), 6),
                                "activation_rms_change": round(drift, 6),
                                "filter_l2": round(float(filter_weights.norm()), 6)})
                prev = active
        return {"sample_id": sample_id, "layer": layer, "channel": channel,
                "history": history,
                "note": "Absolute mean and RMS difference (same spatial grid/channel across checkpoints). Not heatmap-normalized."}
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/inspect")
def inspect(sample_id: str, epoch: int = Query(0, ge=0),
            layer: int = Query(1, ge=1, le=3), channel: int = Query(0, ge=0),
            target_class: int | None = Query(None, ge=0),
            scale_mode: str = Query("fixed", pattern="^(fixed|local)$")):
    try:
        return explanation(sample_id, epoch, layer, channel, target_class, scale_mode)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/compare")
def compare(sample_id: str, epoch_a: int = Query(0, ge=0), epoch_b: int = Query(0, ge=0),
            layer: int = Query(1, ge=1, le=3), channel: int = Query(0, ge=0),
            scale_mode: str = Query("fixed", pattern="^(fixed|local)$")):
    try:
        return comparison(sample_id, epoch_a, epoch_b, layer, channel, scale_mode)
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


# V0.3: step-level observatory. Inference is deliberately lazy: each response
# is recomputed from genuine post-optimizer checkpoints for the selected image.
from .steps import checkpoint_steps


@lru_cache(maxsize=1)
def step_timeline() -> dict:
    return checkpoint_steps(ARTIFACT_DIR)


@lru_cache(maxsize=12)
def step_checkpoint(step: int) -> InspectableCNN:
    manifest, _, _ = context()
    if not manifest.get("step_observatory", {}).get("available"):
        raise ValueError("Step recordings unavailable; retrain with --step-interval N")
    ref = next((x for x in step_timeline()["snapshots"] if x["step"] == step), None)
    if ref is None:
        raise ValueError("Step is not a stored checkpoint; select a marked snapshot")
    # Only filenames generated by the trainer; never accept an untrusted path.
    filename = "epoch_000.pt" if step == 0 else f"step_{step:06d}.pt"
    if ref["path"] != filename:
        raise ValueError("Step checkpoint manifest mismatch")
    model = InspectableCNN(channels=manifest["channels"], classes=len(manifest["classes"]))
    model.load_state_dict(torch.load(ARTIFACT_DIR / filename, map_location="cpu", weights_only=True))
    model.eval()
    return model


@lru_cache(maxsize=64)
def step_activation_scale(sample_id: str, layer: int) -> ActivationScale:
    if layer not in (1, 2, 3):
        raise ValueError("Invalid layer")
    tensor, _ = sample_tensor(sample_id)
    high = 0.0
    with torch.no_grad():
        for ref in step_timeline()["snapshots"]:
            _, trace = step_checkpoint(ref["step"]).forward_trace(tensor)
            high = max(high, float(trace["features"][layer - 1].max()))
    return ActivationScale(0.0, max(high, 1e-9), "sample_layer_all_saved_steps_all_channels")


@app.get("/api/steps")
def steps():
    meta, _, _ = context()
    recording = meta.get("step_observatory", {"available": False})
    if not recording.get("available"):
        return {"available": False, "events": [], "snapshots": [], "message": "Train with --step-interval 4"}
    return {"available": True, "run_id": meta["run_id"],
            "recording": recording, **step_timeline()}


@app.get("/api/step-probe")
def step_probe(sample_id: str, step_a: int = Query(0, ge=0),
               step_b: int = Query(0, ge=0), layer: int = Query(1, ge=1, le=3),
               channel: int = Query(0, ge=0), target_class: int = Query(0, ge=0)):
    """One image + one class + one channel: raw signed B−A after two real updates.

    Grads are NOT inferred from a heatmap. This endpoint is a probe of frozen
    saved model parameters, independent of the minibatches used for training.
    """
    try:
        meta, _, _ = context()
        if target_class >= len(meta["classes"]):
            raise ValueError("Target class out of range")
        tensor, actual_label = sample_tensor(sample_id)
        model_a, model_b = step_checkpoint(step_a), step_checkpoint(step_b)
        with torch.no_grad():
            logits_a, trace_a = model_a.forward_trace(tensor)
            logits_b, trace_b = model_b.forward_trace(tensor)
            feature_a = trace_a["features"][layer-1]
            feature_b = trace_b["features"][layer-1]
            if channel >= feature_a.shape[1]:
                raise ValueError("Channel out of range")
            a, b = feature_a[0, channel], feature_b[0, channel]
            scale = step_activation_scale(sample_id, layer)
            pair = compare_activation(a, b, scale)
            pa = logits_a.softmax(1)[0]
            pb = logits_b.softmax(1)[0]
            kernel_a = (model_a.conv1, model_a.conv2, model_a.conv3)[layer-1].weight[channel]
            kernel_b = (model_b.conv1, model_b.conv2, model_b.conv3)[layer-1].weight[channel]
            filter_change = kernel_b - kernel_a
            channel_drift = (feature_b-feature_a).square().mean(dim=(0, 2, 3)).sqrt()
            interesting = torch.argsort(channel_drift, descending=True)[:min(5,len(channel_drift))]
            activ_a = trace_a["features"][2].mean((-2,-1))[0]
            activ_b = trace_b["features"][2].mean((-2,-1))[0]
            # Exact logit decomposition of GAP + Linear head. This is NOT causal attribution.
            contribution_a = model_a.fc.weight[target_class] * activ_a
            contribution_b = model_b.fc.weight[target_class] * activ_b
            top = torch.argsort(contribution_b.abs(), descending=True)[:8]
            return {
                "sample_id": sample_id, "step_a": step_a, "step_b": step_b,
                "layer": layer, "channel": channel, "target_class": target_class,
                "true_class": actual_label,
                "input": input_image(tensor[0]),
                "a": render_feature_map(a, scale=scale, mode="fixed"),
                "b": render_feature_map(b, scale=scale, mode="fixed"),
                "filter_delta": heatmap(filter_change.mean(0), signed=True, upscale=24),
                "kernel_delta_l2": float(filter_change.double().norm()),
                "feature_mean_a": float(a.mean()), "feature_mean_b": float(b.mean()),
                "top_changing_channels": [{"channel": int(i), "activation_rms_difference": float(channel_drift[i])}
                                          for i in interesting],
                "predicted_a": int(pa.argmax()), "predicted_b": int(pb.argmax()),
                "probability_a": float(pa[target_class]), "probability_b": float(pb[target_class]),
                "probability_delta": float(pb[target_class] - pa[target_class]),
                "logit_a": float(logits_a[0, target_class]), "logit_b": float(logits_b[0, target_class]),
                "logit_delta": float(logits_b[0, target_class] - logits_a[0, target_class]),
                "class_contributions": [{"channel": int(i), "a": float(contribution_a[i]),
                    "b": float(contribution_b[i]), "delta": float(contribution_b[i]-contribution_a[i])}
                    for i in top],
                "class_contributions_note": "Exact terms W_c,k * GAP(A_k) for the final Conv3, excluding bias; not an intervention or causal importance estimate.",
                "scale": scale.as_dict(), **pair,
                "note": "B−A across actual saved post-step models. Per-step loss and gradients refer to the TRAINING minibatch, whereas feature changes are measured on the chosen probe image.",
            }
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

# V0.4 is isolated from the V0.3 16px experiment: its run, checkpoints and
# held-out probe population are independently versioned and explicitly named.
from . import dynamics as dyn


@app.get('/api/dynamics/runs')
def dynamics_runs():
    if not dyn.BASE_DIR.exists():
        return {'runs': [], 'message': 'Create a research run with python -m cnnviz.research'}
    runs=[]
    for entry in sorted(dyn.BASE_DIR.iterdir()):
        if entry.is_dir() and (entry/'manifest.json').is_file():
            try:
                m=dyn.load_run(entry.name)
                runs.append({'run_id':m['run_id'],'dataset':m['dataset'],
                             'variant':m['variant'],'snapshots':len(m['snapshots']),
                             'steps':len(m['events']), 'epochs':m['epochs'],
                             'final_accuracy':m['metrics'][-1]['val_accuracy'] if m['metrics'] else None})
            except (ValueError, OSError, KeyError, json.JSONDecodeError):
                continue
    return {'runs':runs}


@app.get('/api/dynamics/manifest')
def dynamics_manifest(run_id: str):
    try:
        return dyn.load_run(run_id)
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=404,detail=str(exc)) from exc


@app.get('/api/dynamics/cka')
def dynamics_cka(run_id: str,layer: int=Query(1,ge=1,le=5)):
    try:
        return dyn.temporal_cka(run_id,layer)
    except (ValueError, OSError, RuntimeError) as exc:
        raise HTTPException(status_code=400,detail=str(exc)) from exc


@app.get('/api/dynamics/snapshot')
def dynamics_snapshot(run_id: str,step: int=Query(0,ge=0),layer: int=Query(1,ge=1,le=5)):
    try:
        return dyn.snapshot_diagnostics(run_id,step,layer)
    except (ValueError, OSError, RuntimeError) as exc:
        raise HTTPException(status_code=400,detail=str(exc)) from exc


@app.get('/api/dynamics/clusters')
def dynamics_clusters(run_id: str,step: int=Query(0,ge=0),
                      layer: int=Query(1,ge=1,le=5),groups: int=Query(4,ge=2,le=8)):
    try:
        return dyn.channel_clusters(run_id,step,layer,groups)
    except (ValueError, OSError, RuntimeError) as exc:
        raise HTTPException(status_code=400,detail=str(exc)) from exc


@app.get('/api/dynamics/probe')
def dynamics_probe(run_id: str,step_a: int=Query(0,ge=0),step_b: int=Query(0,ge=0),
                   layer: int=Query(1,ge=1,le=5),channel: int=Query(0,ge=0),
                   sample_index: int=Query(0,ge=0)):
    try:
        return dyn.pair_probe(run_id,step_a,step_b,layer,channel,sample_index)
    except (ValueError, OSError, RuntimeError) as exc:
        raise HTTPException(status_code=400,detail=str(exc)) from exc


@app.get('/dynamics.css')
def dynamics_stylesheet():
    return FileResponse(ROOT/'web'/'dynamics.css',media_type='text/css')


@app.get('/dynamics.js')
def dynamics_javascript():
    return FileResponse(ROOT/'web'/'dynamics.js',media_type='text/javascript')

# V0.5 is intentionally a read-only exploration API. Training is CLI-only.
from . import concepts as concept_lab

@app.get('/api/concepts/topk')
def concepts_topk(run_id: str,step: int=Query(...,ge=0),layer: int=Query(5,ge=1,le=5),
                  channel: int=Query(0,ge=0),k: int=Query(8,ge=1,le=16)):
    try:
        return concept_lab.rank_concepts(run_id,step,layer,channel,k)
    except (ValueError,RuntimeError,OSError) as exc:
        raise HTTPException(status_code=400,detail=str(exc)) from exc

@app.get('/api/concepts/compare')
def concepts_compare(run_a: str,run_b: str,step_a: int=Query(...,ge=0),step_b: int=Query(...,ge=0),
                     layer: int=Query(5,ge=1,le=5),bootstrap: int=Query(60,ge=10,le=200)):
    try:
        return concept_lab.compare_runs(run_a,run_b,step_a,step_b,layer,bootstrap)
    except (ValueError,RuntimeError,OSError) as exc:
        raise HTTPException(status_code=400,detail=str(exc)) from exc

@app.get('/api/concepts/faithfulness')
def concepts_faithfulness(run_id: str,step: int=Query(...,ge=0),sample_index: int=Query(0,ge=0),
                          target_class: int=Query(0,ge=0),grid: int=Query(4,ge=4,le=8)):
    try:
        return concept_lab.faithfulness(run_id,step,sample_index,target_class,grid=grid)
    except (ValueError,RuntimeError,OSError) as exc:
        raise HTTPException(status_code=400,detail=str(exc)) from exc

@app.get('/concepts.js')
def concepts_javascript():
    return FileResponse(ROOT/'web'/'concepts.js',media_type='text/javascript')

@app.get('/concepts.css')
def concepts_stylesheet():
    return FileResponse(ROOT/'web'/'concepts.css',media_type='text/css')

@app.get('/api/concepts/matched-examples')
def concepts_matched_examples(run_a: str,run_b: str,step_a: int=Query(...,ge=0),step_b: int=Query(...,ge=0),
                       layer: int=Query(5,ge=1,le=5),channel_a: int=Query(...,ge=0),
                       channel_b: int=Query(...,ge=0),k: int=Query(6,ge=1,le=16)):
    try:
        return concept_lab.matched_examples(run_a,run_b,step_a,step_b,layer,channel_a,channel_b,k)
    except (ValueError,RuntimeError,OSError) as exc:
        raise HTTPException(status_code=400,detail=str(exc)) from exc