# V0.5 · Representation & Concept Discovery

## Goal and falsifiable claims

The UI does **not** assert that a neuron has acquired a named concept. It exposes testable proxies: (i) which fixed held-out images produce top spatial activations, (ii) whether two neurons agree on the **same** held-out probe responses, (iii) whether their broader representations exhibit similar Gram matrices, and (iv) whether changing an image in regions ranked by the target-specific Class Activation Map changes the target-class prediction.

This is not a reproduction of Zeiler & Fergus's original ZFNet/ImageNet results. The five-conv networks have small widths and global-average-pooling heads. For this architecture, class-conditional Grad-CAM is proportional to ReLU of channel-weighted CAM. No use of language-model semantic labeling is implied.

## Architecture

```text
Research trainer (research.py)
    same held-out split/probe across random seeds (split_seed=42)
    train seed controls initialization / minibatch permutations
    real optimizer events, step checkpoint + SHA256 manifest
        |
        v
Dyn reader (dynamics.py)
    probe_data: same x/y ordering, no shuffled requests
    activations_at: saved state dict only
        |
        v
Concept algorithms (concepts.py)
    rank_concepts       -> Top-k images + fixed per-channel scale
    match_channels      -> standardized mean-response correlation + Hungarian
    matched_examples    -> Top-k sample overlap + Jaccard
    compare_runs        -> linear CKA + sample bootstrap CI
    faithfulness        -> target CAM patch deletion vs random-order control
        |
        v
FastAPI read-only endpoints -> web/concepts.js -> web/concepts.css
```

## Exact data agreement

- `research_datasets('digits')` returns 8×8-origin Digits upscaled to 32×32 with nearest-neighbor; **no additional detail** is created.
- `research_datasets('cifar10')` returns native 32×32 CIFAR-10 images, official train/test. Dataset download requires `--download`. **CIFAR-10 has not been trained as part of this offline demonstration.**
- Training/subsampling/probe selection are now split from training RNG through `--split-seed`. The split and sampled training examples remain identical across seeds, while parameter initialization and DataLoader order vary.
- `_aligned()` checks complete probe input tensor and labels, not merely dataset names, seeds or array dimensions. Mismatched runs raise HTTP 400.
- All rankings, CKA and matching use **held-out** samples. `probe_indices` and each response's 16-character SHA256 prefix describe the exact probe population. This SHA prefix is for identification, not cryptographic authorization.

## Measures

### Top-k images

For checkpoint s, layer l and channel c, rank each held-out image by

`score(x,c) = max_{spatial u,v} activation_{s,l,c}(x)[u,v]`.

Return the *actual* image, spatial activation map, peak and mean response. Each activation tile within the selected channel and checkpoint uses the same absolute `[0, max]` color scale. The dominant class fraction is `count(dominant class among top k)/k`, and the pool prior is `count(dominant class among all probes)/n`. This remains a **candidate response preference** rather than a semantic label.

### Channel correspondence

For every channel, compute the vector of mean activation responses across **the same** n probe images. Center and unit-normalize the response vectors. Pearson correlation between two channels is their dot product. Use `scipy.optimize.linear_sum_assignment` on negative correlations to find a globally optimal **one-to-one** matching (not separate greedy maxima). Constant-response channels are mapped to zeros. Similarity can be negative. Reports include all assignments, the mean matched correlation and the top eight associations. Matched Top-k panels offer an independent sample-overlap test: `Jaccard = |Top-k(A)∩Top-k(B)| / |Top-k(A)∪Top-k(B)|`.

**No automatic semantic-identity conclusion** follows from correlation or overlap.

### Across-run CKA

For aligned probe rows X and Y (flattened activations), compute centered Gram matrices `K=H XXᵀ H` and `L=H YYᵀ H`, with `H=I-11ᵀ/n`.

`linear_CKA = <K,L>_F / (||K||_F ||L||_F)`.

Bootstrap 60 resamples of the **same probe indices** and report percentile 2.5% and 97.5%. This quantifies within-pair probe sampling sensitivity, NOT uncertainty over training random seeds, datasets or classes. Due to centered biased CKA and sampling with replacement it is descriptive. Multi-seed comparisons are shown separately; they should not be pooled into a fictitious independent model confidence interval.

### Occlusion faithfulness

For the last Conv layer, target class c, compute `CAM = ReLU(Σ_k w_{c,k} A_k)` and bilinearly upsample to the model input. Split a 32×32 image into 4×4 non-overlapping patches, rank by average CAM score, and progressively fill the top patches with zero. Evaluate target-class probability after each cumulative removal; repeat with 8 seeded random-order controls. `AUC_random - AUC_cam > 0` indicates target-high regions are *more sensitive to this intervention* than random regions for that image/target, not reliable cause discovery. Zero fill is out of distribution. The result can be negative and should **never** be forced positive.

## API

| Endpoint | Required inputs | Output |
| --- | --- | --- |
| `GET /api/concepts/topk` | `run_id,step,layer,channel` | ranked probe images and maps, dominant label, fixed scale |
| `GET /api/concepts/compare` | `run_a,run_b,step_a,step_b,layer` | aligned CKA, bootstrap interval, matching |
| `GET /api/concepts/matched-examples` | plus `channel_a,channel_b` | side-by-side top-k, sample ID overlap/Jaccard |
| `GET /api/concepts/faithfulness` | `run_id,step,sample_index,target_class` | class-specific deletion curves and AUC |

Input parameters are validated. All APIs are **read-only**; training happens via CLI. No unknown Step is interpolated.

## Reproduce demo results (CPU)

```bash
python -m cnnviz.train --dataset digits --epochs 10 --step-interval 4
python -m cnnviz.multiseed --prefix v05-digits-zf --dataset digits \
  --variant zf-small --seeds 11 22 33 --split-seed 42 \
  --epochs 10 --train-limit 1024 --val-limit 160 --probe-count 120 --interval 16
uvicorn cnnviz.server:app --host 127.0.0.1 --port 8000
```

Observed (one execution): final held-out accuracy 88.125%, 86.875%, 91.25%; mean 88.75%, sample standard deviation 2.253 percentage points. Pairwise Conv5 CKA scores 0.7600, 0.4022, 0.4938. These are **Digits**, with n=160 validation examples and a shared pool of n=120 probes, not CIFAR-10.

Run genuine 32×32 CIFAR-10 research on a machine with adequate hardware and dataset access:

```bash
python -m cnnviz.multiseed --prefix cifar-study --dataset cifar10 \
  --variant zf-small --download --seeds 11 22 33 \
  --split-seed 42 --epochs 30 --train-limit 0 --val-limit 600 \
  --probe-count 120 --interval 500
```

This is an **execution recipe only**. Real CIFAR-10 accuracy and CPU/GPU throughput have **not** been measured here. The mini CNN is not tuned to achieve benchmark CIFAR-10 performance; adjust computation/regularization, then report test accuracy without tuning against the official test split.

## Quality gates

```bash
pytest -q
node --check web/concepts.js
python tests/browser_concepts.py    # optional Playwright + Chromium
```

Important open limitations: lack of independent semantic concept annotations (e.g. Broden/ConceptSHAP), no class-stratified bootstrap, no large-seed population, no cross-architecture channel-neuron transfer proof, OOD nature of zero occlusions, no multi-GPU training orchestration, and lack of optimizer-state restart checkpoints. The V0.5 browser is an exploratory measurement surface, not a certificate of faithful explanations.