# Visualization System: a temporal microscope for neural networks

**Product principle:** The user must be able to ask *what did this image look like to the network at epoch E*, *what changed since epoch E−1*, *which features were responsible for a local response*, and *does that interpretation survive an intervention?* Each view answers exactly one of those questions. A visually attractive heatmap is never treated as a quantitative explanation by itself.

## 1. User experience narrative

**Scene A — Choose one image.** A persistent gallery of held-out images is grouped by ground-truth class (4 per class) and supports uploaded custom images. The original input remains visible at every checkpoint. Uploads are unlabeled and never used for training.

**Scene B — Rewind to Epoch 0.** Begin with randomly initialized weights. The untrained classifier should make uncertain or incorrect guesses. The website explicitly labels the initial state as **untrained**, not as learned edges. An optional Play control advances checkpoints; time is an independent dimension across all other inspector views.

**Scene C — Watch the belief update.** Epoch timeline, class probability bars, per-image ground-truth probability trajectory, and training/validation loss respond to the selected checkpoint. Display predicted vs true label and success/failure without hiding non-monotonic learning.

**Scene D — Open a layer.** The user chooses Conv1, Conv2, or Conv3, and then one of their 12/20/28 channels. The atlas shows actual spatial activations from the current checkpoint. The selected channel opens its signed filter, mean activation, zero fraction and a **cross-epoch channel trajectory with absolute numeric scaling**. A marker indicates the current checkpoint and Δ RMS reports activation change from the previous one.

**Scene E — Interrogate the explanation.** Overlay class-conditional saliency and Grad-CAM, display the selected unit's DeconvNet projection, and compare against the occlusion response. Tooltips state clearly that DeconvNet targets one activation, whereas Grad-CAM targets a class; these are different scientific objects.

**Scene F — Form a testable hypothesis.** Example: “Channel 07 responds more strongly to vertical strokes after epoch 5.” Rewind/advance checkpoints while holding the input and channel constant; inspect the absolute mean trajectory to verify. Switch images to check whether the pattern generalizes. Use occlusion to test whether a high-response input area actually matters to the class score.

## 2. Data and lifecycle

```text
scikit-learn Digits (offline) / torchvision CIFAR-10 (optional)
     │
     ▼
train.py: seeded split, deterministic gallery, optimizer loop
     │
     ├── artifacts/<run>/manifest.json (experiment + metrics + sample IDs)
     └── artifacts/<run>/epoch_NNN.pt (one real state_dict per epoch)
                      │
                      ▼
FastAPI: sample → checkpoint → trace (pool switches + feature maps)
                      │
                      ├── probabilities, pred/actual, loss/time metadata
                      ├── Grad-CAM and input saliency (autograd)
                      ├── one-peak DeconvNet projection (unpool + transposed conv)
                      ├── occlusion delta over 49 patch positions (batched)
                      ├── channel activation atlas and signed convolution kernel
                      └── channel-trajectory over full checkpoint lineage
                      │
                      ▼
Browser: shared {sample, epoch, layer, channel, explanation} state
           ├── responsive image stage with opacity control
           ├── checkpoint scrubbing and autoplay
           ├── probability/time/optimization charts
           └── layer/channel explorer + numerically comparable trajectory
```

No training happens on page load. The browser requests saved checkpoint-derived explanations; the model is in eval mode. The playback is **epoch playback**, not a real-time animation of each gradient update. Later versions can capture batches, optimizer states, per-parameter gradients and weight histograms.

## 3. Visualization objects and interpretation

| Object | Source | Normalization | Scientific meaning | Known limitation |
|---|---|---|---|---|
| Original image | Held-out sample or upload | Image values [0,1] | Input to the current checkpoint | Digits are 8→16 nearest upscaled; CIFAR-10 is 32→16 resized |
| Feature-map atlas | Layer post-ReLU tensor | Per-map display stretch | Spatial response of a selected channel | Image colors are not absolute activation values |
| Kernel weight view | Mean across input channels of chosen output filter | Signed diverging | Signed weight pattern | Averaging input channels discards information |
| DeconvNet projection | Largest selected channel activation; MaxUnpool switches | Signed diverging | Input-space back-projection of one activation | Not a faithful inverse / class attribution |
| Saliency | ∂ predicted logit / ∂ input (magnitude) | Per-image display stretch | Local differential sensitivity | Gradient saturation, high-frequency noise |
| Grad-CAM | Weighted post-ReLU Conv3 activations | Per-image display stretch | Low-resolution class-conditional localization | Spatially coarse and not causal |
| Occlusion | Probability drop after zero-masking 4×4 patches | Per-image display stretch, positive drop only | Response to input intervention | Mask artifacts; masked patch can be out of distribution |
| Channel mean trajectory | Mean of same channel across checkpoints | **Raw absolute values** | How an activation scale changes through learning | Different checkpoints may reparameterize features |
| Channel Δ RMS | RMS difference from previous checkpoint | **Raw absolute values** | Magnitude of representational change on same sample | Not a measure of causal relevance |
| Loss/accuracy | Train batch metrics and held-out validation | Absolute metric units | Model-level optimization/generalization | Training metrics aggregated over minibatches |

**Two distinct coordinate systems must be kept separate:** (i) visual heatmap palettes normalized per frame for interpretability and (ii) real activation units, only the latter usable for across-epoch numerical comparisons. Display-level heatmap intensity must not be interpreted as “more activation after epoch 7”.

## 4. Visual design tokens / interaction language

- **Canvas:** light mist `#f7f8fc`, gradient ambient orbs, white cards with 20–30px rounded corners and hairline borders.
- **Color meanings:** Indigo/periwinkle for model/selection signals; warm coral for separate validation series; green for correct; soft red for incorrect. No arbitrary “success green” for data science claims.
- **Typography:** system stack with SF Pro fallbacks, very restrained all-caps micro-labels; large weight/letter-spacing contrasts and generous whitespace; retain bilingual Chinese explanatory copy.
- **Motion:** one continuous time slider changes all panes; 1600ms autoplay per checkpoint. Reactivity occurs through server computations, not CSS-generated pseudo-data. Respect `prefers-reduced-motion`.
- **Desktop:** sample sidebar + central input/timeline + right probability/loss. Activation atlas and reconstruction beneath the primary narrative.
- **Mobile:** stacked cards, horizontally scrollable sample gallery, smaller 4-column activation atlas. Keyboard ArrowLeft/ArrowRight scrubs epochs when focus is outside controls.
- **Failure:** non-blocking toast, stable prior view on transient API error; clear instruction if checkpoints are missing.
- **Accessibility:** semantic sections, per-button aria-labels, input labels, image alt text, keyboard accessible uploads, focus rings.

## 5. Implemented API contract

- `GET /api/manifest`: dataset provenance, seed, architecture, validation gallery, metric arrays.
- `GET /api/trajectory?sample_id=<id>`: per-epoch probabilities, predicted class and true-class probability (null for upload).
- `GET /api/inspect?sample_id=<id>&epoch=E&layer=L&channel=C`: probability ranking + feature-map data URIs + filter/deconv/saliency/Grad-CAM/occlusion + activation stats.
- `GET /api/channel-trajectory?sample_id=<id>&layer=L&channel=C`: per-epoch absolute channel mean, maximum, sparsity, filter norm and RMS activation drift.
- `POST /api/upload`: decoded, resized image held in memory with a generated opaque sample ID. File size capped at 5 MB and pixels at 20 MP.

### Recommended optimization path

**V0.2:** Serve PNG tiles by hash and lightweight numerical JSON separately; compress output and cache static `{checkpoint,sample,layer}` feature maps; precompute trajectories for gallery; use ETags. This is more efficient than repeatedly transferring 28 images encoded as base64 in each `/api/inspect` response.

**V0.3:** Add an event log `{global_step, batch_id, optimizer_lr, loss, gradient_norm, weight_delta, timestamp}` with checkpoint sampling policy. User could inspect actual optimization steps while accurately distinguishing them from epochs. Add persistent comparison and synchronized split-screen scrubber with fixed colormap scales.

**V0.4:** More rigorous paper-faithful reproduction at 224×224 with original ZFNet, top-k examples for each feature across the dataset, robust occlusion and class-conditioned deletion/insertion metrics, randomization sanity checks, CKA/representational similarity and multiple seeds. Add experimental quality gates before publishing interpretations.

## 6. Verification

- Unit tests: all layer shapes, Deconv projection output shape and finite values, invalid layer/channel requests.
- Dataset tests: 1797 Digits, stratified train/validation separation, reproducible gallery.
- Integration: train real model for 2 epochs, read checkpoint 0..2, validate response images, probability simplex, image upload, channel trajectory.
- Browser: offline browser rendering with real API fixture data; Epoch, Layer and explanation toggles; screenshots desktop and mobile.
- Always log and report which modes were tested and which are extrapolated. This first version is *not* a reproduction of the authors' ImageNet performance.

## 7. Sources and dataset provenance

- Zeiler & Fergus, **Visualizing and Understanding Convolutional Networks**, ECCV 2014 — https://arxiv.org/abs/1311.2901
- scikit-learn Digits dataset — https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html
- torchvision CIFAR-10 — https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html
- PyTorch MaxUnpool2d — https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxUnpool2d.html

## 8. V0.2.1 + V0.2.2: controlled comparisons (implemented)

### Fixed-class explanation

The browser **locks `target_class` at sample selection**: a gallery sample uses
its known ground-truth class; an unlabeled upload uses its epoch-0 predicted
class and is clearly labeled as such. The dropdown can lock any other class.
Prediction always shows the checkpoint argmax separately. Requests include
`target_class` so saliency, Grad-CAM, occlusion and the confidence timeline
all refer to the *same fixed class*. The old `/api/inspect` behavior remains
available if the query parameter is omitted: explain the currently predicted
class; it is not appropriate for temporal interpretation.

### Fixed activation color scale

By default feature-map atlas uses `scale_mode=fixed` with range:

    low = 0; high = max over {all checkpoints, all channels, same image, same layer}

The server computes the scale from real post-ReLU activation tensors and
caches the scalar (`maxsize=96`). A Fixed range is shared by atlas tiles,
selected channel and both endpoints in the comparison. In Local mode, each
map is stretched independently to reveal weak structures; **Local colors
must never be compared across time**. The selected mode and numerical bounds
are returned by `/api/inspect` and `/api/compare`.

The signed B−A difference is always displayed on a **separate, zero-centered
symmetric palette with constant ±layer scale**, including when endpoints use
Local stretching. `delta_mean`, `rms_difference`, `l1_mean_difference`,
`cosine_similarity` and `max_absolute_difference` are computed from original
tensors before rendering; cosine is null if either map has zero norm.

The fixed scale is *per sample and per layer*, not a universal dataset-wide
scale. Only compare heatmap intensities **within this scope**. Activations
are nonnegative post-ReLU. Signed differences mean relative change, not
signed input gradients or causal effects.

### New API and contract

- `GET /api/inspect?sample_id=...&epoch=...&layer=...&channel=...&target_class=3&scale_mode=fixed`
  returns class-bound attributions, actual predictions and atlas heatmaps.
- `GET /api/compare?sample_id=...&epoch_a=0&epoch_b=8&layer=2&channel=4&scale_mode=fixed`
  returns common-scale A/B PNGs, difference PNG and raw numerical comparisons.
- Training metrics now include `grad_norm_mean`, `grad_norm_max`,
  `grad_norm_last`, `grad_norm_batches` and the definition in the manifest.
  All values are **global pre-clipping L2 norms** over minibatches; old
  `grad_norm` remains an alias of the last batch for compatibility.
  Old artifacts without `grad_norm_mean` are visibly labeled `LEGACY`.

### Async and model-cache correctness

All sample selections invalidate outstanding trajectory, inspection,
channel-trajectory and comparison work *synchronously*, with request aborts,
monotonic generations and matching identifiers before rendering. Epoch,
channel, class and scale changes likewise protect their respective views.
On a state transition, stale images and metrics are cleared rather than
mislabelled as belonging to the newly selected checkpoint. Duplicate
A/B requests are deduplicated via a comparison state key.

Cached PyTorch model weights **must not be first created inside
`torch.inference_mode()`** because those tensors cannot later be saved for
backpropagation when computing class explanations. Comparison code therefore
uses `torch.no_grad()` instead; a regression test performs comparison before
attribution after clearing the cache.

### Verification

- `pytest -q` includes mathematical A/B checks against directly computed
  activation tensors, fixed-color checks, class-target and backward compatibility,
  gradient aggregation and cache-after-inference regression cases.
- `node --check web/app.js` validates frontend syntax.
- `python tests/browser_smoke.py` (optional Playwright + system Chromium) uses
  the actual FastAPI in-process TestClient as the API source, checks fixed
  class, Local toggle, A/B selectors, layer switch, rapid sample-switch
  consistency, JavaScript errors and mobile overflow without opening local HTTP.
- Still **epoch checkpoints**, not per optimizer step. No rigorous
  paper-faithful ZFNet deconvolution numeric parity or attribution fidelity
  certification yet.

---

## V0.3 · Step-level Learning Observatory

### Scientific invariants

1. One `event` is **one actual optimizer.step**. Recorded `loss_before_update` and `batch_accuracy_before_update` are measured on its TRAIN minibatch **before** the update. `gradient_norm_preclip` / `gradient_norm_layers_preclip` are the gradient magnitudes before gradient clipping; clipping threshold is 5.0.
2. `weight_delta_post_step` is the actual parameter norm `||θ_after − θ_before||₂` **after** the Adam update (not `lr×gradient`). The `weight_delta_layers` values partition this norm by module.
3. **Every** optimizer event is available in `steps.json`; only `snapshots` can be replayed as a model. `--step-interval N` saves every Nth step plus every Epoch's final step; step zero aliases the `epoch_000.pt` initialization. Missing weight snapshots are never interpolated.
4. A Step image is the same held-out image evaluated on two saved models. It is **not** the training minibatch underlying the Step loss or gradient. Dataset splits, target class, layer and channel are held constant within comparisons.
5. Fixed feature color scale is `[0, maximum activation]` across all channels and **all saved Step snapshots**, for the selected sample/layer. Raw spatial B−A is displayed with a symmetric fixed `±maximum`; the difference statistics operate on raw tensors. Kernel difference uses an independently normalized signed scale; do not compare kernel tile colors across different pairs.
6. The final `Global Average Pool → Linear` head allows exact algebraic class-logit contributions, `W[c,k]·mean(A[k])`, excluding bias. This is a **decomposition**, not causal importance or a claim about dataset-wide semantics.
7. Epoch snapshots are bitwise identical with and without Step observer instrumentation (verified for Digits seed 42, 10 epochs). No random step observer operations are used during training.
8. No optimizer state is persisted with Step inference snapshots, so rewinding the slider is inference replay, **not** resuming/branching training. User uploads reside in server memory and are never part of training.

### Files and API

- `artifacts/demo/steps.json`: `{schema, step_interval, global_steps, events, snapshots, semantics}`.
- `artifacts/demo/step_000004.pt`: a stored post-optimizer `state_dict`; `epoch_000.pt` is Step 0.
- `GET /api/steps`: step events, availability and stored snapshot references. With legacy Epoch-only artifacts, returns `available=false`, rather than inventing observations.
- `GET /api/step-probe?sample_id=...&step_a=4&step_b=20&layer=3&channel=5&target_class=7`: two actual saved model forward passes; raw activation comparison, filter change norm, prediction changes, fixed-class probabilities, and exact GAP+Linear contributions.
- The API rejects steps without a saved checkpoint (HTTP 400). The UI keeps a separate **event** scrubber and **saved-snapshot** scrubber to make that separation visible.
- Step data is intentionally generated at inference time for the selected image; saving all image × Step × Layer activations would be costly and redundant.

### Reproduce and verify

```bash
python -m cnnviz.train --dataset digits --epochs 10 --step-interval 4 --out artifacts/demo
pytest -q
node --check web/app.js
python tests/browser_smoke.py  # requires Playwright + Chromium
uvicorn cnnviz.server:app --host 127.0.0.1 --port 8000
```

For a lighter run use `--train-limit 180`; for a full model at each step use `--step-interval 1` (higher disk use). Use `--no-step-records` to run in Epoch-only compatibility mode.

### Explicit limitations and follow-on experiments

- The primary trainer is a three-convolution 16×16 Digits/CIFAR-10 demo, not the original ImageNet ZFNet. Grad-CAM for the GAP+Linear model is related to ordinary CAM.
- Individual channels are comparable **within the same training lineage**; they are not automatically aligned across unrelated seeds/runs.
- Step data is minibatch-level. The Step event plot can be noisy, and changes in batch composition may dominate apparent loss movements. For data-consistent trends, use a fixed held-out input in `/api/step-probe`.
- V0.4 candidates: separate per-module optimizer moment statistics, optimizer state snapshots for genuine branch/resume, dataset-agnostic high-resolution traces, and larger CNN architecture support.