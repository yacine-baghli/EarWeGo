# Huawei Tech Arena 2026 — Pinna Landmark Extraction

This repository contains a professional implementation of a hybrid 3D geometry pipeline to automatically extract **85 pinna (outer ear) landmarks** from human head scans. The solution runs **entirely landmark-free at test time**, utilizing surface curvature, shape priors, and statistical regression to achieve high precision and robustness.

## System Architecture

The pipeline processes raw 3D head meshes using five main stages:

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
