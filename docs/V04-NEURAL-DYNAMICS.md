# V0.4 · Neural Dynamics Observatory

## Scope and experimental claims

This feature is **not** an ImageNet ZFNet replication. Two **small, 5-convolution, 32px** educational models are compared. The controlled early sampling variants are:

- `zf-small`: Conv1 7×7, stride 2, padding 3; channels 24 → 32 → 48 → 48 → 64.
- `alex-small`: Conv1 11×11, stride 4, padding 5; otherwise identical downstream widths.

Both apply maxpool after the first two blocks, then global average pool + linear classification. They are not the exact AlexNet or Zeiler & Fergus networks. There are no claims that relative performance in this toy setting transfers to ImageNet or settles any architectural debate.

The **offline Digits** dataset has genuine **8×8** source pixels interpolated to **32×32** using nearest-neighbor: this introduces *no new image detail*. CIFAR-10 is supported through a separate optional download and uses **native 32×32** source images and the official 50k/10k split. CIFAR-10 is **not included** in the demo and **not silently downloaded**.

## Data flow

```text
Research training: cnnviz.research (isolated from V0.3)
  -> dataset & fixed train/validation partition
  -> 5-layer CNN, seeded Adam + real optimizer.step()
  -> step events: minibatch loss, preclip gradients, per-layer post-step deltas
  -> sparse post-step checkpoints (interval + epoch end)
  -> manifest: environment versions, immutable SHA256 list, dataset provenance

Research analysis: cnnviz.dynamics
  -> same fixed held-out probe images across all checkpoints
  -> forward_trace of selected real weight state
     -> raw full spatial activations [samples, channels, height, width]
     -> centered linear CKA across checkpoints at a fixed layer
     -> per-channel response-vectors + within-checkpoint KMeans groups
     -> parameter/gradient histograms and actual update norms
     -> same sample/channel A/B image, signed spatial differences
  -> independent FastAPI APIs -> vanilla JS Observatory section
```

## Mathematical definitions

### Linear CKA, fixed probe alignment

For identical held-out samples under two frozen network states, let `X ∈ R^(n×d1)` and `Y ∈ R^(n×d2)` denote *flattened activations*, with rows aligned by sample identity. We subtract per-column means, then compute `K = XXᵀ`, `L = YYᵀ`. The centered linear CKA used here is

```
CKA(K,L) = tr(KL) / sqrt(tr(K²) tr(L²))
```

The result lies in `[0,1]` when the denominator is positive. The implementation uses `float64` for stability. A degenerate zero-Gram comparison currently returns `0` and should be interpreted as **undefined/degenerate**, not evidence of independence. A user should not compare matrices from different probe sets, data augmentations, or architectures without controlling data alignment and feature extraction. With only 40 probe images, estimates are descriptive, sample-dependent and potentially biased. A higher-quality experiment should add larger probes, repeated seeds, bootstrap bands and a debiased estimator.

**Reference:** Kornblith et al. (2019), [Similarity of Neural Network Representations Revisited](https://proceedings.mlr.press/v97/kornblith19a.html), ICML 2019.

### Response grouping

For a single checkpoint and layer, channel `k` receives a vector of mean-spatial responses `r_k ∈ R^n` on the **same** `n` probe images. Each channel's vector is standardized (constant vectors use an epsilon denominator). Deterministic KMeans clusters the **channels**; the UI shows cluster membership and the average raw response to probe images from each class. Cluster names/IDs are arbitrary **within each checkpoint**. Cross-run or cross-epoch cluster correspondences are **not established**.

### Mini-batch gradient distribution vs test-image features

`gradient_histogram` is computed from the actual **training mini-batch before gradient clipping**. Parameter L2 differences are measured immediately **after** Adam update, not approximated by `learning_rate × gradient`. Probe activations are independently computed on a **held-out** data subset. Changes between these quantities show association through a common checkpoint; they are not a full causal attribution of a particular training example.

### A/B color-scale contract

For one held-out probe image and one layer, a common non-negative feature color range is calculated from the **max of all channels at all saved checkpoints**, not independently per image. Signed spatial differences preserve negative/positive values and use a symmetric scale. Reported RMS and cosine values are computed on the raw activation tensors, never PNG colors. Only stored snapshots can be compared: missing weights are not interpolated.

## API contract

- `GET /api/dynamics/runs`: available named local runs and final validation metrics.
- `GET /api/dynamics/manifest?run_id=...`: versioned run metadata, step events, snapshots, provenance and SHA256 hashes.
- `GET /api/dynamics/cka?run_id=...&layer=1`: square symmetric real checkpoint CKA matrix with ordered step IDs.
- `GET /api/dynamics/snapshot?run_id=...&step=...&layer=1`: parameter and corresponding saved-step gradient histograms, norms and top channels.
- `GET /api/dynamics/clusters?run_id=...&step=...&layer=1&groups=4`: per-checkpoint channel response clustering and class response summaries.
- `GET /api/dynamics/probe?run_id=...&step_a=0&step_b=240&layer=3&channel=1&sample_index=0`: one image at two real saved model states, signed difference, probabilities, and pairwise CKA.

HTTP GET never downloads data or modifies a model. Invalid run slugs, sample indexes, unavailable snapshots and incorrect channels are rejected. Data are cached for interactive exploration; restart the process after modifying an artifact in place.

## Reproduce

```bash
pip install -r requirements.txt
# Existing legacy 16px / 3-conv experiment remains independent.
python -m cnnviz.train --dataset digits --epochs 10
# Two 32px / 5-conv research runs; source digits are still native 8px.
python -m cnnviz.research --name research-digits-zf-v04 --dataset digits --variant zf-small \
    --epochs 15 --train-limit 1024 --val-limit 160 --interval 16
python -m cnnviz.research --name research-digits-alex-v04 --dataset digits --variant alex-small \
    --epochs 15 --train-limit 1024 --val-limit 160 --interval 16
uvicorn cnnviz.server:app --port 8000
```

Native CIFAR-10 example (download occurs **only** when explicitly requested):

```bash
python -m cnnviz.research --name cifar-zf --dataset cifar10 --variant zf-small \
    --epochs 20 --train-limit 0 --val-limit 0 --interval 100 --download
```

The CIFAR-10 route is available but is not claimed as measured or validated at full scale in this delivery.

## Quality gates and limitations

- `pytest -q`: legacy checkpoint tests, architecture shapes, linear CKA numerical invariances and symmetry, genuine step checkpoint provenance, cluster membership totals, probe PNG/differences, and request bounds.
- `PYTHONPATH=. python tests/browser_dynamics.py`: browser-backed real API responses through an in-process TestClient transport, automated desktop/mobile UI interactions, script errors and horizontal overflow, without depending on localhost browser access.
- `node --check web/dynamics.js`: static syntax verification.
- Optional browser script requires `pip install playwright` and system Chromium. It is not required by default CI.

**Further work:** real high-resolution CIFAR-10 experiments with adequate training compute, more than 40 held-out probe images, multiple seeds and confidence bands, cross-run channel matching, unbiased CKA, meaningful neuron concept discovery with curated object-part datasets, scalable inference/CKA caching, and secure model lifecycle management. Attribution fidelity still requires independent intervention and randomization tests; CKA is not causal.