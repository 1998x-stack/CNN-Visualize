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