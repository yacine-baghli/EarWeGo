# Huawei Tech Arena 2026 — Pinna Landmark Extraction

![system_overview](img/Overview.png)

This repository contains a professional implementation of a hybrid 3D geometry pipeline to automatically extract **85 pinna (outer ear) landmarks** from human head scans. The solution runs **entirely landmark-free at test time**, utilizing surface curvature, shape priors, and statistical regression to achieve high precision and robustness.

## Best Validated Version — Deep Contour Ensemble + Dense SSM (1.294 mm)

The best model is a **deep point-cloud network that refines the classical estimate**,
reaching a validation mean landmark error of **1.294 mm** (60 ears / 30 subjects) —
a **31 % improvement over the classical Dense-V4 pipeline** and a **2× improvement**
over the previously committed model on the one-shot test set (2.65 mm).

**Target:** the organizers specify **< 0.5 mm** ("good") and **< 0.2 mm** ("very good"),
since these landmarks feed pinna measurements used for HRTFs. We are not there yet; the
ground truth is precise to ~0.006 mm, so the remaining error is *model* error and the
target is reachable in principle — the measured obstacle and the concrete plan are in
[`deep_model/README.md`](deep_model/README.md).

![deep results](deep_model/results/deep_results.png)

| Validation metric (60 ears) | Deep ensemble | Classical Dense-V4 |
| --- | ---: | ---: |
| Mean landmark error | **1.294 mm** | 1.874 mm |
| Median landmark error | 1.120 mm | 1.718 mm |
| RMSE | 1.560 mm | — |
| 95 % CI for the mean | [1.206, 1.379] mm | — |
| Worst-ear mean | 2.216 mm | 6.36 mm |
| Success rate @ 2 mm | **82.2 %** | 65.5 % |
| Success rate @ 3 mm | **93.7 %** | 84.1 % |
| Success rate @ 5 mm | **98.9 %** | 95.9 % |
| HRTF-critical SR @ 2 mm | 88.7 % | — |

```bash
python -m deep_model.evaluate_deep   # reproduce the metrics + figure (no PyTorch, no raw data)
```

---

## Architecture of the best model (1.294 mm)

The deep model does not predict landmarks from scratch: it **refines** the classical
pipeline's coarse estimate (~3.68 mm, [see below](#system-architecture)). Every stage
below is justified by a measurement, and its contribution was measured in isolation.

```
raw head mesh (PLY)  ──►  classical pipeline  ──►  coarse 85 landmarks (~3.68 mm)
                                                          │
   ┌──────────────────────────────────────────────────────┘
   ▼
[0] CANONICAL FRAMING          crop the ear around the coarse estimate (±14 mm),
                               rotation-align to the SSM mean shape, centre on the
                               coarse centroid, keep true mm scale, sample 2048 pts.
                               Right ears are mirrored into a common left frame.
   ▼
[1] DGCNN BACKBONE             3 × EdgeConv on a static k-NN graph (k=20) over the
                               point cloud → 64/128/128 features, concatenated (320)
                               → fused to 256, then max-pooled global context mixed
                               back per point → 256-d per-point features.
   ▼
[2] ITERATIVE OFFSET → SNAP    ×4 passes, independently per landmark:
    HEAD                       • OFFSET  pooled (mean+max) features of the K=48 points
                                 nearest the current query + a per-landmark embedding
                                 → MLP → an *unconstrained* 3-D displacement.
                                 (Without this the model cannot reach landmarks that
                                 lie outside the initial window — it plateaus at 2.76 mm.)
                               • SNAP    re-window at the relocated query, attention
                                 (softmax) over the K=48 nearest surface points →
                                 expectation = a convex combination of real surface
                                 points, so the prediction stays ON the surface and is
                                 sub-point-spacing precise (expressivity floor 0.095 mm).
   ▼
[3] CONTOUR-STRUCTURED         the 85 landmarks form 4 ordered anatomical contours.
    REFINEMENT                 Each contour gets its OWN 1-D convolution stack
                               (kernels 5→3→1, width 96) running ALONG the ordered
                               landmarks, taking [position, backbone feature (64),
                               landmark embedding (32)].  ← largest single gain
   ▼
[4] 4-SEED ENSEMBLE            average the raw predictions of 4 independently trained
                               seeds (measured to saturate at 4: 6 seeds = 1.330).
   ▼
[5] SURFACE PROJECTION         exact point-to-triangle snap onto the mesh (pure NumPy).
   ▼
[6] DENSE-SSM HYBRID FIT       a dense-vertex ear shape model is fitted to the target
                               surface *and* to these landmarks (closed form), then
                               blended at α=0.3 and re-projected.
   ▼
85 landmark coordinates (85 × 3)
```

### Measured contribution of each stage (validation, 60 ears)

| stage | val MLE | Δ |
| --- | ---: | ---: |
| classical coarse estimate (input to the deep model) | 3.68 mm | — |
| [1]+[2] DGCNN + iterative offset→snap head | 1.50 mm | −2.18 |
| [3] + contour-structured refinement | 1.375 mm | −0.13 |
| [4] + 4-seed ensemble | 1.329 mm | −0.046 |
| [5] + exact surface projection | 1.309 mm | −0.020 |
| [6] + dense-SSM hybrid blend | **1.294 mm** | −0.015 |

### Why the contour head is the key architectural idea

A per-region error decomposition showed **~60 % of the error is *tangential*** —
landmarks sliding *along* their contour, not off the surface (worst contour: 1.59 mm
tangential vs 0.64 perpendicular). Along a smooth rim there is **no local geometric
cue** for where a landmark sits (surface curvature along the contour is flat, ~5°/mm at
every scale down to the 0.5 mm mesh resolution), so a purely local window physically
cannot resolve it — only contour-level context can. Giving each anatomical region its
own 1-D convolution along the ordered landmarks supplies exactly that, and it also made
the ensemble seeds decorrelate about twice as effectively.

### Implementation details

| | |
| --- | --- |
| Input | 2048-point ear cloud (canonical frame, true mm) + 85 coarse landmarks |
| Backbone | DGCNN, static graph k=20, 3 EdgeConv layers, 256-d per-point features |
| Head | K=48 window, T=4 offset→snap passes, per-landmark embeddings (32-d) |
| Parameters | 813 k per seed (3.25 MB) — deliberately small for 280 training ears |
| Loss | MSE on the final output + deep supervision on every pass (offset ×0.4, snap ×1.0, later passes weighted more) |
| Optimiser | AdamW, lr 1.5e-3, weight decay 5e-4, cosine schedule, 1200 epochs, batch 16 |
| Augmentation | random 1280-point subsample, random-axis rotation ±34°, scale ±10 %, surface jitter 0.25 mm, coarse-init jitter 0.9 mm |
| Training data | 280 ears (140 subjects); left/right unified by mirroring |
| Dense SSM \[6\] | 120 components over 23 252 template vertices (96 % variance), built by landmark-anchored non-rigid ICP on the **training split only** |
| Inference | **pure NumPy, no PyTorch** — parity-verified to 3.6e-6 against the PyTorch reference |

Trained in PyTorch on an RTX A6000; the weights are exported as plain NumPy arrays and
the forward pass is reimplemented in NumPy, so the submission needs no deep-learning
framework. Full details, the negative results (what was measured *not* to work), and
the path toward the 0.5 mm target are in
**[`deep_model/README.md`](deep_model/README.md)**.

**What limits us is now measured.** The ground truth is precise — landmarks sit
**0.006 mm** from the surface and contour spacing is algorithmically equidistant — so
the remaining error is *model* error, not label noise. It is dominated by **tangential**
sliding *along* the contours, and those landmarks provably have **no local geometric
signature** to detect (curvature is flat along the contour at every scale; the GT was
built by tracing contours and resampling by arc length). A per-point detector is
therefore the wrong framing; the route to the 0.5 mm target is curve-based (dense-surface
SSM + non-rigid registration) — see
[`deep_model/README.md`](deep_model/README.md#what-limits-us-measured).

<details>
<summary><b>Classical baseline — Dense V4 (1.8738 mm)</b></summary>

The best classical pipeline is **Dense V4**, validation MLE **1.8738 mm** (30 subjects):
median 1.7177 mm, SR@2/3/5 mm = 65.5 / 84.1 / 95.9 %. It is the coarse initialiser
for the deep model above.

- [Browse the Dense V4 source snapshot](versions/dense_v4_1.8738mm/)
- [View aggregate validation metrics](versions/dense_v4_1.8738mm/validation_metrics.json)
- Source commit: [`21bdc53`](https://github.com/yacine-baghli/EarWeGo/commit/21bdc53bb21ffbb8dcc0026108efcb014a025926)
</details>

The published artifacts contain source code, model weights (NumPy), configuration,
and aggregate metrics only. Challenge data, participant split lists, host metadata,
and per-subject results are **not** published.

## System Architecture — classical pipeline (coarse stage)

> This is the **classical** pipeline. On its own it reaches 1.874 mm (Dense-V4); in the
> current best model it serves as the **coarse initialiser** (~3.68 mm for the v1
> configuration used to train the deep model) that the
> [deep model above](#architecture-of-the-best-model-1294-mm) refines to 1.294 mm.
> It runs entirely landmark-free at test time.

The pipeline processes raw 3D head meshes using six main stages:

```
3D Head Scan (PLY)
  │
  ▼
[1. Automatic Ear Detection]  ◄── Curvature Analysis & Learned Spatial Bounding Box
  │
  ▼
[2. Coarse Template Alignment] ◄── Rigid Iterative Closest Point (ICP)
  │
  ▼
[3. Statistical Shape Model]   ◄── Regularized Shape Prior via GPA + PCA Projection
  │
  ▼
[4. Coordinate Residual Regs] ◄── GBR Models regressing reconstruction errors
  │
  ▼
[5. KNN Shape Blending]       ◄── Weighted blending with local training shapes
  │
  ▼
[6. Mesh Surface Snapping]    ◄── Final proximity projection onto target geometry
  │
  ▼
85 Landmark Coordinates (85x3)
```

1. **Automatic Ear Detection**: Resolves the localization problem without requiring landmarks at test time. Utilizes a learned spatial bounding box and local surface curvature (ears have 5.5x higher curvature than the skull) to isolate left and right ears.
2. **Template Alignment (ICP)**: Aligns left/right mean ear shape templates to the isolated ear mesh vertices using rigid Iterative Closest Point (ICP).
3. **Statistical Shape Model (SSM)**: Projects the coarsely aligned points onto a unified shape space trained via Generalized Procrustes Analysis (GPA) and Principal Component Analysis (PCA) to regularize the shape.
4. **Residual Correction**: Uses 255 separate Gradient Boosting Regressors (GBR) to predict coordinate-wise residuals between the regularized SSM reconstruction and the true landmark geometry.
5. **K-Nearest Neighbors (KNN) Blending**: Computes similarity weights in the shape coefficient space and blends the predictions with the nearest neighborhood of local training shapes.
6. **Surface Snapping**: Projects predicted landmark points to the nearest coordinate on the target mesh surface.
<img width="2491" height="1922" alt="image" src="https://github.com/user-attachments/assets/0aa5f6ee-3f07-4031-a758-efff87c1c904" />
<img width="2083" height="742" alt="image" src="https://github.com/user-attachments/assets/e21431eb-ba83-4423-9c51-84acd9a46c6e" />

---

## Repository Structure

```
Huawei_tech_arena/
├── deep_model/                 # BEST MODEL (1.294 mm) — torch-free NumPy inference
│   ├── deep_infer.py           # NumPy forward pass of one network (DGCNN + heads)
│   ├── deep_predict.py         # DeepEnsemble: N networks (+ optional TTA / SSM blend)
│   ├── deep_stage.py           # Pipeline stage: framing, mirroring, projection
│   ├── surfproj.py             # Exact point-to-triangle surface projection
│   ├── dense_fit.py            # Dense-SSM hybrid fit (closed form)
│   ├── evaluate_deep.py        # Reproduce metrics + results figure
│   ├── weights/                # 4 trained seeds (813 k params each, NumPy)
│   ├── dense_ssm.npz           # Dense-vertex ear shape model (train split only)
│   └── results/                # metrics.json + deep_results.png
├── models/                     # Saved model pickle checkpoints (.pkl)
├── output/                     # Diagnostic reports, evaluation stats, and plots
├── scratch/                    # Temporary/experimental scripts
├── src/                        # Core package files
│   ├── __init__.py
│   ├── dataset.py              # Compatible dataset loader
│   ├── estimator.py            # Official LandmarkExtractor submission class
│   ├── metrics.py              # Official mean landmark distance metric
│   ├── ear_detector.py         # Curvature-based ear region segmenter
│   ├── predictor.py            # Landmark predictor class (SSM + GBR + KNN)
│   ├── geometry.py             # Alignment, GPA, and SSM math
│   ├── visualize.py            # Diagnostic and 3D visualization tools
│   ├── evaluation.py           # 6-Dimensional Rigorous Evaluation suite
│   └── eval_plots.py           # Diagnostic dashboard plotting code
├── versions/
│   └── dense_v4_1.8738mm/      # Source-only snapshot of the best validated version
├── requirements.txt            # Package dependencies
├── train.py                    # Script to train and save checkpoints
├── evaluate.py                 # Script to evaluate model performance
└── README.md                   # User documentation
```

---

## Getting Started

### Installation

Clone this repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

### Dataset Structure

Ensure the dataset folders are structured as follows:
```
2026 Munich Tech Arena - Datas/
└── 2026 Munich Tech Arena - Datas/
    ├── mesh/
    │   ├── P0001.ply
    │   └── ...
    └── landmarks/
        ├── P0001_left_ear_landmarks.csv
        ├── P0001_right_ear_landmarks.csv
        └── ...
```

---

## Usage Pipeline

### 1. Training

To train the `EarDetector` and `LandmarkPredictor` and save the checkpoints to the `models/` directory:

```bash
python train.py --mesh-dir "path/to/mesh" --landmarks-dir "path/to/landmarks"
```

*Arguments:*
- `--n-components`: Number of Statistical Shape Model PCA components (default: `30`).
- `--k-neighbors`: Number of neighbors for KNN shape blending (default: `7`).
- `--blend-alpha`: Weight for SSM+GBR predictions vs KNN shapes (default: `0.6`).
- `--models-dir`: Output directory for checkpoints (default: `models`).
- `--n-mesh-samples`: Number of mesh samples to compile the mean ear template (default: `30`).

### 2. Evaluation

To run the evaluation pipeline using the official metrics and the rigorous 6-dimensional report suite:

```bash
python evaluate.py --mesh-dir "path/to/mesh" --landmarks-dir "path/to/landmarks"
```

*Arguments:*
- `--diagnostic`: If set to `True`, runs the full 6D diagnostic analysis and saves plots to `--output-dir` (default: `True`).
- `--quick-test`: If set to a positive integer $N$, evaluates only the first $N$ subjects.
- `--output-dir`: Output directory for reports and figures (default: `output`).

---

## Challenge Submission

For evaluation on the hidden test set, the challenge platform will load the `LandmarkExtractor` class in `src/estimator.py` and call:

```python
extractor = LandmarkExtractor()
pred_left, pred_right = extractor.extract(mesh)
```

The extractor automatically handles model loading and landmark-free prediction. To prepare your submission:
1. Ensure you have run `python train.py` to generate the checkpoints in `models/`.
2. Include the `models/ear_detector.pkl` and `models/landmark_predictor.pkl` files in your final submission zip.
3. Ensure the folder structure matches the **Repository Structure** defined above.
