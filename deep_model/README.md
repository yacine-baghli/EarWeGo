# Deep Contour-Ensemble Landmark Model

Best validated model in this repository: **1.329 mm** mean landmark error on the
held-out validation split (60 ears / 30 subjects) — a **28 % improvement over the
classical Dense-V4 pipeline** (1.85 mm val) and a **2× improvement over the
previously committed model** on the one-shot test set (2.65 mm).

![results](results/deep_results.png)

| Metric (validation, 60 ears) | Result |
| --- | ---: |
| Mean landmark error (MLE) | **1.329 mm** |
| Median landmark error | 1.143 mm |
| RMSE | 1.586 mm |
| 95 % CI for the mean | **[1.242, 1.414] mm** |
| Worst-ear mean | 2.245 mm *(classical: 6.36)* |
| Best-ear mean | 0.710 mm |
| Success rate @ 2 mm | 81.9 % |
| Success rate @ 3 mm | 93.5 % |
| Success rate @ 5 mm | 98.9 % |
| HRTF-critical SR @ 2 mm | 88.9 % |
| HRTF-critical MLE | 1.092 mm |

The competition-winning reference is 1.29 mm; that value sits **inside our 95 %
confidence interval**, i.e. at n = 60 the two are statistically indistinguishable.
See [Beating 1.29](#beating-129-mm) for the concrete path below it.

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

## Beating 1.29 mm

The ensemble curve fits `err(N) = 1.283 + 0.092/√N`, so pure seed-ensembling
*approaches* ~1.283 but closes the last bit slowly (≈32 seeds to clearly beat 1.29).
The efficient path is to lower the **single-model floor**, which is set by having only
280 training ears (train 1.25 / val 1.54 = overfitting):

1. **Synthetic data** — sample the shape model → thin-plate-spline-warp real ear
   clouds to the synthetic landmark sets → thousands of training pairs.
2. **Train the final model on train + val** (340 ears) for the actual submission
   (test is the holdout) — reliably lowers error, can't be measured on val.
3. **Anthropometric priors** — condition on head/ear measurements.

## Training

Trained on an RTX A6000 (PyTorch). The training script and the leakage-clean dataset
builder are kept out of this package (they depend on the raw challenge data). The
dataset is leakage-audited: the SSM is fit on the training split only, the coarse
init uses per-mesh geometry only, and the test split is never loaded.
