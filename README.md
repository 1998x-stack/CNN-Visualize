# CNN Visualize · Learning Journey

> **An Apple-inspired interactive lab for watching convolutional neural networks learn, checkpoint by checkpoint.**

Based on [Zeiler & Fergus, *Visualizing and Understanding Convolutional Networks* (ECCV 2014)](https://arxiv.org/abs/1311.2901). This is a small, scientifically explicit reimplementation and visualization lab, not a claim to reproduce the exact original ZFNet ImageNet result.


## V0.5 · Representation & Concept Discovery (NEW)

V0.5 introduces a fifth-page-section **What did it learn?** with real held-out Top-k response galleries, fixed-scale feature maps, Hungarian one-to-one cross-run channel matching, aligned cross-run linear CKA with probe bootstrap intervals, matched neuron Top-k overlap, and target-class CAM deletion vs random-occlusion curves. This is scientific measurement of candidate concepts, **not automated or proven semantic labeling**.

The offline demo contains three independent training seeds on **Digits** with **one identical split and 120 identical held-out probe images**, using the 5-conv `zf-small` model: 10 epochs / 160 real optimizer updates per seed, final validation accuracies 88.125%, 86.875%, 91.25% (on a small 160-image validation set). Pairwise final-layer CKA: 0.760, 0.402, 0.494. The fixed split permits meaningful per-image comparison; seed variation can still produce very different internal representations.

```bash
# Existing source clone needs real research checkpoints, unlike the offline demo ZIP.
python -m cnnviz.train --dataset digits --epochs 10 --step-interval 4
python -m cnnviz.multiseed --prefix v05-digits-zf --dataset digits \
  --variant zf-small --seeds 11 22 33 --split-seed 42 \
  --epochs 10 --train-limit 1024 --val-limit 160 --probe-count 120 --interval 16
uvicorn cnnviz.server:app --host 127.0.0.1 --port 8000
```

An opt-in **native 32×32 CIFAR-10 multi-seed training pipeline** is available through `python -m cnnviz.multiseed --dataset cifar10 --download ...`; actual CIFAR-10 runs are **not** part of this delivery or claimed as verified. Do not confuse Digits images upscaled from 8×8 with native 32×32 data.

See **[V0.5 research design, API, math, experimental setup and limitations](docs/V05-CONCEPT-DISCOVERY.md)**.

## V0.4 · Neural Dynamics Observatory

The existing 16×16 three-convolution Epoch/Step explorer is **unchanged**. V0.4 adds a **separate** experiment family with **five convolutional layers**, real **32×32 model inputs**, two **AlexNet/ZFNet-inspired** first-layer configurations, fixed-probe **temporal linear CKA**, layer-specific weight/gradient histograms, KMeans channel-response groupings, and a new Apple-style **Observe the unseen** interactive dashboard. These are **small educational architectures**, not the original ImageNet models.

To reproduce the offline research runs (source Digits are native **8×8**, interpolated to 32×32; **not** genuine 32px source detail):

```bash
python -m cnnviz.train --dataset digits --epochs 10 --step-interval 4
python -m cnnviz.research --name research-digits-zf-v04 --variant zf-small --epochs 15 --train-limit 1024 --val-limit 160 --interval 16
python -m cnnviz.research --name research-digits-alex-v04 --variant alex-small --epochs 15 --train-limit 1024 --val-limit 160 --interval 16
uvicorn cnnviz.server:app --host 127.0.0.1 --port 8000
```

Choose a V0.4 research run at the bottom of the homepage, select `Conv 01..05`, click a cell in the CKA matrix to choose Step A/B, then select a neuron channel and fixed held-out sample. The UI displays **actual saved weights**, same-scale A/B activations and signed difference, per-layer gradient distributions and channel-response families. No unsaved Step weights are fabricated.

**Native 32×32 CIFAR-10** is supported via `python -m cnnviz.research --dataset cifar10 --download ...`; it is **not** downloaded silently and not part of the measured offline demo. The current demo's two 15-Epoch, 1024-training-sample/160-validation-sample Digits experiments achieve 97.5% (`zf-small`) and 96.875% (`alex-small`) validation accuracy **on that one seed/split**, not a rigorous architecture benchmark. Each has 240 real optimizer events and 16 saved checkpoints. More seeds, CIFAR-10 and larger probe sets are future work.

See **[V0.4 scientific design, equations, APIs and limitations](docs/V04-NEURAL-DYNAMICS.md)**.

## Quickstart (offline, CPU)

```bash
python -m venv .venv
source .venv/bin/activate                # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m cnnviz.train --dataset digits --epochs 10
uvicorn cnnviz.server:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000/**. On first launch after cloning you **must generate checkpoints** with the training command above. For faster smoke testing use `--epochs 2 --train-limit 300`.

## How it works

1. Select a held-out validation image from a gallery (four per class), or upload an image (stored **in server memory only**; no uploaded labels are assumed).
2. Drag the **Epoch timeline** or press Play to load weights from the *actual checkpoint*. Epoch 0 is a randomly initialized, untrained network.
3. Explore the classification probabilities, predicted class, per-image true-class probability trajectory, training/validation loss and accuracy.
4. Toggle four separate explanations: **input saliency**, **Grad-CAM**, **occlusion sensitivity** (4×4 zero patch on a 16×16 input), and **DeconvNet projection** (single peak activation, stored MaxPool switches, positive backward rectification and transposed convolution).
5. Switch among **three convolutional layers** and select any channel in the activation atlas. See the learned filter, selected feature activation, projected pixel pattern, mean, sparsity and **cross-epoch channel learning curve with RMS activation drift**.

The detailed visual design and scientific interpretation specification is in [`docs/VISUALIZATION-SYSTEM.md`](docs/VISUALIZATION-SYSTEM.md).

**Legacy note:** the V0.3 Epoch/Step explorer uses real checkpoints and locks an explicit target class for gradient explanations. Its fixed-color mode supports quantitative comparisons. DeconvNet remains a single-peak feature projection, not a class attribution.

## Datasets

| Dataset | Availability | Input | Split | Role |
|---|---|---|---|---|
| `digits` | Offline through scikit-learn | Original 8×8 grayscale → nearest-upscaled 16×16 | Fixed stratified 80/20 seed 42 | Default development & unit testing |
| `cifar10` | Optional torchvision download | Original 32×32 RGB → bilinear 16×16 | Official 50k/10k | More meaningful color scenes; longer training |

```bash
python -m cnnviz.train --dataset cifar10 --download --epochs 20 --out artifacts/cifar10
CNNVIZ_ARTIFACTS=artifacts/cifar10 uvicorn cnnviz.server:app --port 8000
```

CIFAR-10 is not downloaded silently. Its provenance/rights belong to the original data provider; review its terms before redistributing examples. No dataset images or trained models are shipped in GitHub source by default. `artifacts/` and `data/` are gitignored.

## Repository layout

```text
cnnviz/
  data.py       # Offline digits + CIFAR-10 tensor loaders and partitions
  model.py      # Three Conv blocks + pool switches + DeconvNet projection
  train.py      # Train + save all Epoch state_dict checkpoints and metrics
  imaging.py    # Reproducible PNG, filter sign/activation maps
  server.py     # FastAPI checkpoint inference, explanations, upload endpoint
web/
  index.html    # Accessible responsive lab structure
  app.css       # Apple-style frosted-glass, gradients, responsive UI
  app.js        # Timeline, gallery, layer/channel and heatmap interaction
artifacts/      # Generated weights and manifests (ignored)
tests/          # Model, API, upload and reproducibility smoke tests
```

## Design principles

- **No fabricated scientific data**: epochs and explanations are computed from actual PyTorch weights and inputs, not a screenshot animation.
- **Each question has a dedicated view**: temporal probability, global optimization loss, layer/channel activation, target-class gradient, backward feature projection, input occlusion intervention.
- **Make uncertainty visible**: distinguish reconstruction from attribution, and clarify the low 16×16 demonstration resolution.
- **Privacy by default**: uploads are image-decoded, size-limited, held only in memory, evicted after 30 images, and never used for training.
- **Reproducibility**: fixed seeds, dataset provenance, checkpoint per Epoch, validation split, `weights_only=True` checkpoint loading.

## API contract

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Server and checkpoint readiness |
| `/api/manifest` | GET | Run metadata, loss curves, gallery images |
| `/api/trajectory?sample_id=...` | GET | Per-Epoch probabilities for one input |
| `/api/inspect?sample_id=...&epoch=...&layer=...&channel=...` | GET | Real intermediate activations and explanation images |
| `/api/channel-trajectory?sample_id=...&layer=...&channel=...` | GET | Absolute channel activation stats and RMS change across all checkpoints |
| `/api/upload` | POST | Upload ≤5MB image as multipart field `file` |

## Historical roadmap (superseded by the actual release notes below)

**V0.2:** pre- and post-activation comparison; consistent cross-epoch heatmap scales; aligned nearest-neighbor top activations; global class-wise activation drift and similarity (CKA), deterministic gallery presets; network graph with clickable connections.

**V0.3:** ZFNet/AlexNet 224×224 research-scale implementation; train-time gradient/parameter histograms; dense inference trace serialization; class-aware deletion/insertion metrics and randomization sanity checks; exact sample provenance.

**V0.4:** checkpoint comparisons with side-by-side image, weight distribution and feature trajectory; video capture/export and automated benchmark reports.

## Test

```bash
pytest -q
```

## V0.2.2 — controlled temporal microscope

- Lock any target class across epochs (gallery defaults to its true label;
  unlabeled upload defaults to its Epoch-0 predicted class). Predictions remain
  independent; attribution heatmaps target the locked class.
- Fixed, quantitative color domains for the same sample/layer across all
  saved epochs and channels; optional Local contrast expansion **only for
  visual inspection**, never intensity comparisons.
- A/B Epoch inspector with common-scale feature maps, fixed signed B−A
  difference and raw RMS/mean/L1/cosine statistics.
- Cancellable / generation-guarded browser requests and reset placeholders;
  cached model inference safe for subsequent autograd attributions.
- Mean/max/last global gradient norms captured over every minibatch, all
  measured **before gradient clipping**. Legacy manifest values are labelled.

Run `pytest -q` and `node --check web/app.js`. With Python Playwright and
Chromium installed, run `python tests/browser_smoke.py` for a headless browser
regression on the real in-process FastAPI results (no live HTTP required).
See [docs/VISUALIZATION-SYSTEM.md](docs/VISUALIZATION-SYSTEM.md) for the
color-scale scope, API semantics and scientific limitations.

## V0.3 — Step-level Learning Observatory

See all actual minibatch updates, not just Epoch snapshots. The new **See every update** section includes a per-Step Loss / preclip Gradient / postupdate Weight-Δ chart, an event scrubber for every optimizer step, and a distinct saved-checkpoint scrubber for the snapshots that really exist. Lock the image, layer, channel and target class to compare two trained states. View common-scale activations, signed B−A, filter-weight changes, fixed-class prediction changes and exact GAP+Linear per-channel contributions. Jump straight to the channels whose spatial activations changed the most.

```bash
python -m cnnviz.train --dataset digits --epochs 10 --step-interval 4
uvicorn cnnviz.server:app --host 127.0.0.1 --port 8000
```

- The trainer emits `artifacts/demo/steps.json` with **every** optimizer step event; `step_XXXXXX.pt` is emitted only for saved steps and epoch ends.
- `/api/steps` provides all training events, including per-layer gradient norms and actual optimizer parameter deltas; `/api/step-probe` evaluates two **real** saved checkpoints on the same held-out image.
- Selecting an unsaved event updates the optimization metrics only, **not** the Step A/B model maps. The page explains the distinction prominently.
- Older Epoch-only artifacts still work; the Step area reports how to enable recording.
- `--step-interval 1` records every model state (at higher storage cost), or `--no-step-records` turns step recording off entirely.
- Caution: model snapshots do not include the Adam optimizer state; timeline replay is inference only, not resumable training.

Run `pytest -q` and `node --check web/app.js`. For optional offline Chromium smoke tests: `CNNVIZ_ARTIFACTS=$PWD/artifacts/demo python tests/browser_smoke.py`.
For numerical/visualization definitions see [docs/VISUALIZATION-SYSTEM.md](docs/VISUALIZATION-SYSTEM.md).