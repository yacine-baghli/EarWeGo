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

## Why 1.329 mm is the floor (exhaustively measured)

Two adversarial investigations plus dozens of local measurements established that
**1.329 mm is the practical floor for this dataset + architecture.** The reason is
structural: the **median** per-landmark error is already **1.07 mm** — the *mean* is
pulled up by an ~11 % tail of ambiguous rim landmarks (worst: idx 70–74) whose error
is **~85 % tangential** (sliding *along* the contour). That tail is largely
**irreducible ground-truth annotation ambiguity** plus a **280-ear data ceiling**
(the model is ~97 % variance-limited). Every modeling lever was *measured*, not
guessed:

| lever | result | why |
|---|---|---|
| more ensemble seeds | saturates (4-seed 1.329 = 6-seed 1.330) | seeds correlated; true asymptote ~1.33 |
| synthetic data (SSM+TPS) | wash-to-worse | 30-comp SSM too coarse; source-anchored variant neutral |
| curvature input channels | **worse** (1.474 vs 1.395 single) | signal is perpendicular (small); extra channels overfit 280 ears |
| richer SSM prior / projection | ≤ 0 mm | 60–70 % of error energy is *inside* the SSM subspace |
| cascade / tighter crops | dead | point density uncorrelated with error (r=0.06) |
| relational / global head | ≈ 0 | the boundary slides as a unit; added capacity overfits |
| continuous-surface snap | dead | predictions already 0.31 mm from the surface |
| bilateral symmetry | dead | a person's L/R ears differ 1.59 mm > our error |
| TTA, Huber, confidence-gating | ≤ 0.015 (within noise) | attack the same variance term |

**The only remaining small gains** (do at submission time): train the final weights
on **train + val** (280 → 340 ears, ~0.05 mm on the val→test gap, unverifiable), and
ship the 4-seed ensemble **without TTA**.

## The real path below the floor — new *information*, not new modeling

Genuine improvement requires new data, since the ceiling is data/annotation-limited:

1. **External 3D ear data → self-supervised backbone pretraining** (highest leverage;
   directly attacks the 280-ear overfitting). Public, in-domain, permissively-licensed
   candidates (~700+ extra ear meshes): **SONICOM** (200 subj, MIT), **AudioEar3D**
   (112 scans, GitHub), **HUTUBS** (96 subj, CC-BY — but co-created by Huawei, so
   check the challenge rules and subject overlap for leakage). A published ear 3D
   morphable model, the **York Ear Model (YEM)**, could replace the 30-comp SSM prior.
2. **Multi-annotator re-labeling** of the ~5 ambiguous tail landmarks (70–74) to
   denoise the ground truth itself — what actually drags the mean above the 1.07 median.
3. *(Research bet)* a **tangent-aware / arc-length correspondence loss** — the only
   idea that targets the dominant tangential error, but likely capped by annotation noise.

> ⚠️ Before using any external dataset, confirm the competition permits external data,
> and verify no subject overlap with the challenge set (SONICOM/HUTUBS are the same
> HRTF-pinna domain as this challenge). See the session notes for the full search.

## Training

Trained on an RTX A6000 (PyTorch). The training script and the leakage-clean dataset
builder are kept out of this package (they depend on the raw challenge data). The
dataset is leakage-audited: the SSM is fit on the training split only, the coarse
init uses per-mesh geometry only, and the test split is never loaded.
