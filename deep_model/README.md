# Deep Contour-Ensemble Landmark Model

Best validated model in this repository: **1.309 mm** mean landmark error on the
held-out validation split (60 ears / 30 subjects) — a **30 % improvement over the
classical Dense-V4 pipeline** (1.85 mm val) and a **2× improvement over the
previously committed model** on the one-shot test set (2.65 mm).

> **Target (from the organizers):** < 0.5 mm is "good", < 0.2 mm "very good", because
> these landmarks feed pinna measurements used for HRTFs
> ([Dinakaran et al., ICASSP 2018](https://depositonce.tu-berlin.de/items/f9757195-f1a2-493a-809e-a4a4a7de7f49):
> ~1 mm surface precision preserves localization cues). We are **not there yet** —
> see [the analysis of what limits us](#what-limits-us-measured) — but the ground
> truth is precise enough (~0.006 mm off-surface) that sub-0.5 mm is achievable in
> principle, so the remaining error is *model* error, not label noise.

![results](results/deep_results.png)

| Metric (validation, 60 ears) | Result |
| --- | ---: |
| Mean landmark error (MLE) | **1.309 mm** |
| Median landmark error | 1.123 mm |
| RMSE | 1.577 mm |
| 95 % CI for the mean | [1.222, 1.394] mm |
| Worst-ear mean | 2.224 mm *(classical: 6.36)* |
| Best-ear mean | 0.687 mm |
| Success rate @ 2 mm | 82.1 % |
| Success rate @ 3 mm | 93.5 % |
| Success rate @ 5 mm | 98.9 % |
| HRTF-critical SR @ 2 mm | 89.0 % |
| HRTF-critical MLE | 1.068 mm |

## Architecture

The model refines a coarse landmark estimate (the classical pipeline's per-mesh
extraction, ~3.7 mm) into a precise one, operating on the ear **point cloud** in a
canonical, coarse-centred frame.

```
ear point cloud (2048 pts, canonical frame)  +  coarse 85-landmark estimate
        │
        ▼
[1. DGCNN backbone]      ── 3 EdgeConv layers (static kNN graph) + global context
        │                   → rich per-point local-geometry features
        ▼
[2. Iterative offset→snap head]  (×4 passes, per landmark)
        │   • OFFSET: pooled local-window feature → unconstrained displacement
        │            (relocates the query — reaches landmarks a local window can't)
        │   • SNAP:   soft-argmax over the K=48 nearest surface points
        │            (keeps the prediction ON the surface = sub-point precision)
        ▼
[3. Contour-structured refinement]  ← the decisive lever
        │   Per anatomical contour (Helix / Antihelix-Concha / Outer boundary /
        │   Cross-section), a small 1-D convolution ALONG the ordered landmarks,
        │   with its OWN weights per contour. Fixes "along-contour sliding" — the
        │   error a purely local head cannot see (see below).
        ▼
[4. 4-seed ensemble]     ── average the raw predictions of 4 independently-trained
                            seeds. Decorrelates residual error.
        ▼
85 refined landmark coordinates
```

### Why the contour head — the key finding

A per-region error analysis showed that **~60 % of the error is *tangential*** —
landmarks sliding *along* their contour rather than off the surface (worst on the
Outer boundary: 1.59 mm tangential vs 0.64 perpendicular). Along a smooth rim there
is no *local* geometric cue for exactly where a landmark sits, so a local soft-argmax
window physically cannot fix it — only *contour-level* context can. The
region-specific 1-D convolution supplies exactly that, and it was the single largest
architectural gain (1.50 → 1.375 mm single-model) as well as making the ensemble
seeds decorrelate twice as effectively.

Higher point-cloud resolution (8192 vs 2048 pts) was tested and did **not** help —
it only sharpens the already-small perpendicular error.

## Files

| File | Role |
| --- | --- |
| `deep_infer.py` | Torch-free NumPy forward pass for one network (parity-verified to 5e-6 vs the PyTorch reference). |
| `deep_predict.py` | `DeepEnsemble` — averages N networks × optional TTA, blends with the SSM projection. |
| `deep_stage.py` | Pipeline stage: frame a raw mesh around a coarse estimate, run the ensemble, handle left/right mirroring. |
| `evaluate_deep.py` | Reproduce the metrics + figure from committed predictions. |
| `weights/gpu_cont_s{0..3}.npz` | The 4 trained seeds (~3.3 MB each, NumPy arrays — no PyTorch needed). |
| `ssm.npz` | Train-only statistical shape model (mean + 30 components) used for framing / optional projection. |
| `val_errors.npz` | Per-landmark error **distances** of the 4-seed ensemble on validation (no landmark coordinates — no challenge data published), for reproducible metrics. |

**No PyTorch is required at inference** — training was done in PyTorch on GPU, then
the weights were exported as plain NumPy arrays and the forward pass reimplemented in
NumPy (`deep_infer.py`), verified to match the PyTorch output to 5e-6.

## Reproduce the metrics and figure

```bash
python -m deep_model.evaluate_deep
```

This reads `val_errors.npz` (error distances only), writes `results/metrics.json`,
and renders `results/deep_results.png`. It needs no raw challenge data and no PyTorch.

## Run the model on a mesh

```python
import numpy as np, glob
from deep_model.deep_stage import deep_refine, load_ensemble

ssm = np.load("deep_model/ssm.npz")
ens = load_ensemble(sorted(glob.glob("deep_model/weights/gpu_cont_s*.npz")),
                    ssm["ssm_mean"], ssm["ssm_comp"], blend=0.0, tta=False)

# coarse_world: (85,3) coarse estimate from the classical pipeline (world frame)
# mesh_verts:   (V,3) mesh vertices (world frame)
landmarks = deep_refine(mesh_verts, coarse_world, ens, ssm["ssm_mean"].reshape(85, 3),
                        side="left")   # or side="right"
```

Best recipe is the **4-seed ensemble, raw** (`blend=0.0`, `tta=False`): with the
contour head + ensemble already removing the tangential error, the SSM projection
and TTA no longer help.

## What limits us (measured)

**Correction to an earlier conclusion:** we previously called ~1.33 mm an
"annotation-noise floor". That was **wrong** — it was calibrated against a mistaken
1.29 mm target. Direct measurement shows the ground truth is highly precise:
landmarks sit **0.006 mm** from the mesh surface (96.5 % within 0.05 mm), contour
spacing is **algorithmically equidistant** (gap CV 0.011–0.018 on two contours), and
intrinsic point jitter is only ~0.15 mm. **The remaining error is model error.**

The dominant component is **tangential** — sliding *along* the contour (1.06 mm vs
0.61 mm perpendicular). Measured cause: the landmarks have **no local geometric
signature** to detect. Surface curvature along the contour is flat (~5°/mm) at *every*
scale down to the 0.5 mm mesh resolution, and the nearest curvature feature sits
1.2–1.4 mm away (no better than random). **The GT was built by tracing contours,
projecting to the surface, and resampling by arc length** — so 30 of the 85 landmarks
are pure arc-length samples with no detectable local identity. A per-point detector
fundamentally cannot place them; that is the wrong framing, and it is why the levers
below all fail.

Levers that were *measured* (not guessed) and do **not** work:

| lever | result | why |
|---|---|---|
| more ensemble seeds | saturates (4-seed 1.329 = 6-seed 1.330) | seeds correlated; true asymptote ~1.33 |
| synthetic data (SSM+TPS) | wash-to-worse | 30-comp SSM too coarse; source-anchored variant neutral |
| curvature input channels | **worse** (1.474 vs 1.395 single) | signal is perpendicular (small); extra channels overfit 280 ears |
| richer SSM prior / projection | ≤ 0 mm | 60–70 % of error energy is *inside* the SSM subspace |
| cascade / tighter crops | dead | point density uncorrelated with error (r=0.06) |
| relational / global head | ≈ 0 | the boundary slides as a unit; added capacity overfits |
| bilateral symmetry | dead | a person's L/R ears differ 1.59 mm > our error |
| TTA, Huber, confidence-gating | ≤ 0.015 (within noise) | attack the same variance term |
| **hard equal-arc-length layer** | **worse** (1.43 vs 1.40) | correct idea, but resampling propagates the *endpoint* error (lm 74 is one of the worst) across the whole contour — even with 6× endpoint loss weight |

What **does** work, and is shipped: **exact point-to-triangle surface projection**
(`surfproj.py`). GT lies 0.006 mm off the surface, raw predictions ~0.17 mm off, so
projecting along the normal is a free systematic gain: **1.329 → 1.309 mm**,
improving **100 % of ears** (paired t-test p = 2e-29).

## The path toward the 0.5 mm target

The GT's own generative process is the blueprint — **contours, not points**:

1. **Dense-surface SSM + non-rigid ICP (recommended).** Register one landmarked
   template ear onto every training ear (GT landmarks as anchors) to get dense
   correspondence, PCA a dense-vertex shape model in which the 85 landmarks are fixed
   barycentric points, then at test time fit that model to the *whole surface*.
   Thousands of surface points over-determine each landmark's along-contour position —
   the only measured mechanism that can fix tangential error — then arc-length-resample
   to reproduce the GT construction. Estimated 0.5–0.7 mm; gated by contour-endpoint
   accuracy. **Run the cheap kill-gate first:** check that transported-template +
   resampling reproduces val GT before building the full SSM.
2. **Contour-endpoint detector.** All arc-length machinery hinges on the ~8 contour
   endpoints; they are currently the worst landmarks (idx 70–74 ≈ 2.5 mm).
3. **More real labeled ears** (external 3D-ear data → backbone pretraining):
   **SONICOM** (200 subj, MIT), **AudioEar3D** (112 scans), **HUTUBS** (96 subj, CC-BY —
   co-created by Huawei, so check challenge rules *and* subject overlap for leakage);
   the **York Ear Model** could replace the 30-comp SSM prior.
4. At submission time: train the final weights on **train + val** (340 ears).

> ⚠️ Before using any external dataset, confirm the competition permits external data
> and verify no subject overlap with the challenge set.

## Training

Trained on an RTX A6000 (PyTorch). The training script and the leakage-clean dataset
builder are kept out of this package (they depend on the raw challenge data). The
dataset is leakage-audited: the SSM is fit on the training split only, the coarse
init uses per-mesh geometry only, and the test split is never loaded.
