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

Full architecture, the key findings (≈60 % of the error is *tangential* — landmark
sliding *along* contours — which a per-region **contour-structured refinement head**
partly corrects, plus exact **point-to-triangle surface projection** that improved
100 % of ears), the torch-free NumPy inference (parity-verified to 4e-6), and
reproduction instructions are in **[`deep_model/README.md`](deep_model/README.md)**.

```bash
python -m deep_model.evaluate_deep   # reproduce the metrics + figure (no PyTorch, no raw data)
```

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

## System Architecture

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
