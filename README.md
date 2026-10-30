# CNN Visualize · Learning Journey

> **An Apple-inspired interactive lab for watching convolutional neural networks learn, checkpoint by checkpoint.**

Based on [Zeiler & Fergus, *Visualizing and Understanding Convolutional Networks* (ECCV 2014)](https://arxiv.org/abs/1311.2901). This is a small, scientifically explicit reimplementation and visualization lab, not a claim to reproduce the exact original ZFNet ImageNet result.

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

**Important distinction:** the Epoch playback replays stored training checkpoints; it does not visualize individual optimizer steps. Grad-CAM and saliency are class-conditional for the **current predicted class**. DeconvNet does **not** establish class attribution. Colors are normalized **per image**, so they support qualitative spatial comparison but **not** quantitative heatmap intensity comparisons across epochs; rely on numeric activation means for that.

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

## Scientific roadmap

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