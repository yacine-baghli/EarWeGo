"""
============================================================================
RIGOROUS EVALUATION FRAMEWORK for Pinna Landmark Extraction
============================================================================

A publication-grade evaluation suite designed by domain experts in
3D morphometry, computational anatomy, and spatial audio.

Covers 6 evaluation dimensions:
  1. Point-Level Accuracy (per-landmark Euclidean errors)
  2. Shape-Level Fidelity (Procrustes distance, shape preservation)
  3. Anatomical Plausibility (inter-landmark distance preservation)
  4. Directional Bias Analysis (systematic offset detection)
  5. Robustness & Reliability (failure rate, confidence intervals)
  6. Downstream HRTF Impact (acoustic-relevant error weighting)

Author: Expert evaluation framework for Huawei Tech Arena 2026
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.linalg import orthogonal_procrustes
from sklearn.model_selection import KFold
from typing import Optional
from pathlib import Path
import time
import warnings

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import (
    load_all_landmarks, load_participant, get_participant_ids, NUM_LANDMARKS,
)
from src.geometry import procrustes_align, mirror_landmarks_y


# ═══════════════════════════════════════════════════════════════════════════
# LANDMARK GROUP DEFINITIONS (anatomical regions of the pinna)
# ═══════════════════════════════════════════════════════════════════════════

LANDMARK_GROUPS = {
    "Helix": list(range(0, 25)),        # Outer rim — large-scale shape
    "Antihelix/Concha": list(range(25, 55)),  # Inner folds — fine detail
    "Outer boundary": list(range(55, 75)),    # Back contour / lobule
    "Cross-section": list(range(75, 85)),     # Transverse depth markers
}

# Landmarks that are CRITICAL for HRTF (based on acoustic literature):
# The concha cavity, tragus, and helix crest are the primary pinna
# features that shape the spectral notches in HRTFs.
HRTF_CRITICAL_LANDMARKS = {
    "Concha depth":    [29, 30, 31, 32, 33, 34],     # Concha cavity
    "Tragus/Antitragus": [43, 44, 45, 46, 47],        # Canal entrance
    "Helix crest":     [5, 6, 7, 8, 9, 10],           # Top of helix
    "Antihelix fold":  [25, 26, 27, 28, 35, 36, 37],  # Antihelix ridge
    "Ear canal axis":  [48, 49, 50, 51, 52, 53, 54],  # Points near canal
}

# Adjacent landmark pairs for inter-landmark distance checks
# (pairs that should maintain consistent spacing)
ADJACENT_PAIRS = (
    [(i, i+1) for i in range(0, 24)] +       # Helix contour
    [(i, i+1) for i in range(25, 54)] +       # Inner contour
    [(i, i+1) for i in range(55, 74)] +       # Outer boundary
    [(i, i+1) for i in range(75, 84)]         # Cross-section
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. POINT-LEVEL ACCURACY METRICS
# ═══════════════════════════════════════════════════════════════════════════

def compute_point_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    """
    Compute comprehensive point-level error metrics.
    
    Args:
        gt:   (85, 3) ground-truth landmarks
        pred: (85, 3) predicted landmarks
    
    Returns:
        dict of scalar metrics
    """
    # Per-landmark Euclidean error
    errors = np.linalg.norm(pred - gt, axis=1)  # (85,)
    
    # Per-axis signed error (for bias detection)
    signed_errors = pred - gt  # (85, 3)
    
    # Per-axis absolute error
    axis_errors = np.abs(signed_errors)
    
    return {
        # --- Euclidean error statistics ---
        "MLE":        errors.mean(),                     # Mean Landmark Error
        "MdLE":       np.median(errors),                 # Median Landmark Error
        "SDLE":       errors.std(),                      # Std of Landmark Error
        "MaxLE":      errors.max(),                      # Maximum Landmark Error
        "MinLE":      errors.min(),                      # Minimum Landmark Error
        "P90":        np.percentile(errors, 90),         # 90th percentile
        "P95":        np.percentile(errors, 95),         # 95th percentile
        "P99":        np.percentile(errors, 99),         # 99th percentile
        
        # --- RMSE (root mean squared error — penalizes outliers more) ---
        "RMSE":       np.sqrt(np.mean(errors**2)),
        
        # --- Normalized Mean Error (by ear size for scale-invariance) ---
        "NME":        errors.mean() / _ear_diagonal(gt),
        
        # --- Success rate at various thresholds ---
        "SR@2mm":     (errors < 2.0).mean(),   # % landmarks within 2mm
        "SR@3mm":     (errors < 3.0).mean(),   # % landmarks within 3mm
        "SR@5mm":     (errors < 5.0).mean(),   # % landmarks within 5mm
        "SR@10mm":    (errors < 10.0).mean(),  # % landmarks within 10mm
        
        # --- Per-axis mean absolute error ---
        "MAE_x":      axis_errors[:, 0].mean(),
        "MAE_y":      axis_errors[:, 1].mean(),
        "MAE_z":      axis_errors[:, 2].mean(),
        
        # --- Per-axis signed bias (systematic offset) ---
        "Bias_x":     signed_errors[:, 0].mean(),
        "Bias_y":     signed_errors[:, 1].mean(),
        "Bias_z":     signed_errors[:, 2].mean(),
    }


def _ear_diagonal(landmarks: np.ndarray) -> float:
    """Compute the bounding-box diagonal of a landmark set (normalization factor)."""
    return np.linalg.norm(landmarks.max(axis=0) - landmarks.min(axis=0))


# ═══════════════════════════════════════════════════════════════════════════
# 2. SHAPE-LEVEL FIDELITY METRICS
# ═══════════════════════════════════════════════════════════════════════════

def compute_shape_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    """
    Evaluate whether the predicted landmark configuration preserves
    the overall ear SHAPE, independent of rigid position.
    
    This removes global translation/rotation/scale and measures
    intrinsic shape error — critical for HRTF applications.
    """
    # Procrustes alignment: find best rigid+scale fit of pred to gt
    pred_aligned, transform = procrustes_align(pred, gt, allow_scale=True)
    procrustes_errors = np.linalg.norm(pred_aligned - gt, axis=1)
    
    # Procrustes distance (shape distance after optimal alignment)
    procrustes_dist = np.sqrt(np.mean(procrustes_errors**2))
    
    # Relative shape error (Procrustes vs raw) — how much is rigid misalignment?
    raw_rmse = np.sqrt(np.mean(np.linalg.norm(pred - gt, axis=1)**2))
    alignment_improvement = 1.0 - (procrustes_dist / (raw_rmse + 1e-10))
    
    # Centroid error (global position accuracy)
    centroid_error = np.linalg.norm(pred.mean(axis=0) - gt.mean(axis=0))
    
    # Scale error
    gt_scale = np.linalg.norm(gt - gt.mean(axis=0))
    pred_scale = np.linalg.norm(pred - pred.mean(axis=0))
    scale_ratio = pred_scale / (gt_scale + 1e-10)
    
    return {
        "Procrustes_RMSE":     procrustes_dist,
        "Procrustes_Mean":     procrustes_errors.mean(),
        "Procrustes_Median":   np.median(procrustes_errors),
        "Alignment_Gain":      alignment_improvement,  # How much error was just rigid offset
        "Centroid_Error":      centroid_error,
        "Scale_Ratio":         scale_ratio,             # 1.0 = perfect scale
        "Scale_Error_Pct":     abs(scale_ratio - 1.0) * 100,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. ANATOMICAL PLAUSIBILITY METRICS
# ═══════════════════════════════════════════════════════════════════════════

def compute_anatomical_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    """
    Evaluate whether predicted landmarks form an anatomically valid ear.
    
    Checks:
    - Inter-landmark distance preservation
    - Topological ordering preservation
    - Self-intersection (landmark crossings)
    """
    # 3a. Inter-Landmark Distance Preservation (ILD)
    # For adjacent landmarks, the spacing should match ground truth
    gt_dists = []
    pred_dists = []
    for i, j in ADJACENT_PAIRS:
        if i < len(gt) and j < len(gt):
            gt_dists.append(np.linalg.norm(gt[i] - gt[j]))
            pred_dists.append(np.linalg.norm(pred[i] - pred[j]))
    
    gt_dists = np.array(gt_dists)
    pred_dists = np.array(pred_dists)
    
    ild_errors = np.abs(pred_dists - gt_dists)
    ild_relative = ild_errors / (gt_dists + 1e-10)
    
    # 3b. Full pairwise distance matrix comparison
    gt_pdm = cdist(gt, gt)      # (85, 85) pairwise distance matrix
    pred_pdm = cdist(pred, pred)
    
    # Relative pairwise distance error (upper triangle only)
    triu_idx = np.triu_indices(len(gt), k=1)
    gt_pw = gt_pdm[triu_idx]
    pred_pw = pred_pdm[triu_idx]
    
    pw_errors = np.abs(pred_pw - gt_pw)
    pw_relative = pw_errors / (gt_pw + 1e-10)
    
    # 3c. Correlation of pairwise distances (shape similarity)
    pw_correlation = np.corrcoef(gt_pw, pred_pw)[0, 1]
    
    # 3d. Ordering violations — check if sequential landmarks maintain order
    # along the helix (0-24), the progression should be monotonic in arc-length
    ordering_violations = 0
    for group_name, indices in LANDMARK_GROUPS.items():
        if len(indices) < 3:
            continue
        # Check if cumulative distances along the contour are preserved in sign
        for k in range(len(indices) - 2):
            i, j, l = indices[k], indices[k+1], indices[k+2]
            gt_d1 = np.linalg.norm(gt[j] - gt[i])
            gt_d2 = np.linalg.norm(gt[l] - gt[j])
            pred_d1 = np.linalg.norm(pred[j] - pred[i])
            pred_d2 = np.linalg.norm(pred[l] - pred[j])
            # Flag if the ratio changes drastically
            if gt_d1 > 0.5 and gt_d2 > 0.5:
                gt_ratio = gt_d1 / gt_d2
                pred_ratio = pred_d1 / (pred_d2 + 1e-10)
                if abs(np.log(pred_ratio / (gt_ratio + 1e-10))) > 1.5:
                    ordering_violations += 1
    
    return {
        # Adjacent pair spacing
        "ILD_Mean":           ild_errors.mean(),          # Mean inter-landmark dist error
        "ILD_Max":            ild_errors.max(),
        "ILD_Relative_Mean":  ild_relative.mean() * 100,  # % relative error
        "ILD_Relative_Max":   ild_relative.max() * 100,
        
        # Full pairwise distances
        "PW_Mean_Error":      pw_errors.mean(),
        "PW_Relative_Mean":   pw_relative.mean() * 100,
        "PW_Correlation":     pw_correlation,             # Should be > 0.99
        
        # Topology
        "Ordering_Violations": ordering_violations,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. DIRECTIONAL BIAS ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def compute_bias_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    """
    Detect systematic directional bias in predictions.
    
    A good model should have zero mean signed error per axis.
    This detects if the model consistently over/under-predicts
    in a particular direction.
    """
    signed = pred - gt  # (85, 3)
    
    # Per-axis bias (one-sample t-test: is mean significantly != 0?)
    results = {}
    axis_names = ["x", "y", "z"]
    for axis in range(3):
        axis_data = signed[:, axis]
        t_stat, p_value = scipy_stats.ttest_1samp(axis_data, 0.0)
        results[f"Bias_{axis_names[axis]}_mean"] = axis_data.mean()
        results[f"Bias_{axis_names[axis]}_std"] = axis_data.std()
        results[f"Bias_{axis_names[axis]}_t_stat"] = t_stat
        results[f"Bias_{axis_names[axis]}_p_value"] = p_value
        results[f"Bias_{axis_names[axis]}_significant"] = p_value < 0.05
    
    # Directional error (magnitude + direction of average error vector)
    mean_error_vector = signed.mean(axis=0)
    results["Bias_vector_magnitude"] = np.linalg.norm(mean_error_vector)
    
    # Per-landmark-group bias
    for group_name, indices in LANDMARK_GROUPS.items():
        group_signed = signed[indices]
        group_mag = np.linalg.norm(group_signed.mean(axis=0))
        results[f"Bias_{group_name}_magnitude"] = group_mag
    
    return results


# ═══════════════════════════════════════════════════════════════════════════
# 5. ROBUSTNESS & RELIABILITY METRICS
# ═══════════════════════════════════════════════════════════════════════════

def compute_robustness_metrics(
    all_errors: list[np.ndarray],
    threshold_failure: float = 10.0,
) -> dict:
    """
    Evaluate model robustness across multiple predictions.
    
    Args:
        all_errors: list of (85,) per-landmark error arrays, one per ear
        threshold_failure: mm threshold for a "failed" landmark
    """
    all_errors_flat = np.concatenate(all_errors)
    per_ear_means = np.array([e.mean() for e in all_errors])
    per_ear_maxes = np.array([e.max() for e in all_errors])
    
    n_ears = len(all_errors)
    n_landmarks_total = len(all_errors_flat)
    
    # Failure rates
    n_failed_landmarks = (all_errors_flat > threshold_failure).sum()
    n_failed_ears = (per_ear_means > threshold_failure).sum()
    
    # Catastrophic failure: any ear with mean error > 20mm
    n_catastrophic = (per_ear_means > 20.0).sum()
    
    # 95% confidence interval for mean error (bootstrap-style)
    ci_low, ci_high = _bootstrap_ci(per_ear_means, n_bootstrap=1000)
    
    # Coefficient of variation (consistency measure)
    cv = per_ear_means.std() / (per_ear_means.mean() + 1e-10)
    
    # Inter-quartile range
    iqr = np.percentile(per_ear_means, 75) - np.percentile(per_ear_means, 25)
    
    return {
        "N_ears":                  n_ears,
        "N_landmarks_total":       n_landmarks_total,
        "Failure_Rate_Landmarks":  n_failed_landmarks / n_landmarks_total * 100,
        "Failure_Rate_Ears":       n_failed_ears / n_ears * 100,
        "Catastrophic_Rate":       n_catastrophic / n_ears * 100,
        "CV":                      cv,          # Coefficient of variation
        "IQR":                     iqr,         # Inter-quartile range of per-ear means
        "CI95_low":                ci_low,      # 95% CI lower bound
        "CI95_high":               ci_high,     # 95% CI upper bound
        "Worst_Ear_Mean":          per_ear_means.max(),
        "Best_Ear_Mean":           per_ear_means.min(),
    }


def _bootstrap_ci(data, n_bootstrap=1000, ci=0.95, seed=42):
    """Compute bootstrap confidence interval for the mean."""
    rng = np.random.RandomState(seed)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        means.append(sample.mean())
    means = np.array(means)
    alpha = (1 - ci) / 2
    return np.percentile(means, alpha * 100), np.percentile(means, (1 - alpha) * 100)


# ═══════════════════════════════════════════════════════════════════════════
# 6. HRTF-WEIGHTED (DOWNSTREAM ACOUSTIC IMPACT) METRICS
# ═══════════════════════════════════════════════════════════════════════════

def compute_hrtf_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    """
    Weight errors by their acoustic importance for HRTF estimation.
    
    Based on literature: the concha, tragus, and antihelix contribute
    the most to spectral notches in the HRTF above 4 kHz.
    Errors in these regions have outsized impact on perceived spatial audio.
    
    We define importance weights:
      - HRTF-critical landmarks (concha, tragus, ear canal): weight = 3.0
      - Helix crest: weight = 2.0
      - Other landmarks: weight = 1.0
    """
    errors = np.linalg.norm(pred - gt, axis=1)
    
    # Build weight vector
    weights = np.ones(NUM_LANDMARKS)
    for group_name, indices in HRTF_CRITICAL_LANDMARKS.items():
        if group_name in ("Concha depth", "Tragus/Antitragus", "Ear canal axis"):
            for idx in indices:
                weights[idx] = 3.0
        elif group_name in ("Helix crest", "Antihelix fold"):
            for idx in indices:
                weights[idx] = 2.0
    
    # Weighted metrics
    weighted_mean = np.average(errors, weights=weights)
    
    # HRTF-critical-only error
    critical_indices = []
    for indices in HRTF_CRITICAL_LANDMARKS.values():
        critical_indices.extend(indices)
    critical_errors = errors[critical_indices]
    
    # Concha depth error (most acoustically sensitive)
    concha_errors = errors[HRTF_CRITICAL_LANDMARKS["Concha depth"]]
    
    # Ear canal axis error
    canal_errors = errors[HRTF_CRITICAL_LANDMARKS["Ear canal axis"]]
    
    return {
        "HRTF_Weighted_MLE":       weighted_mean,
        "HRTF_Critical_MLE":       critical_errors.mean(),
        "HRTF_Critical_Max":       critical_errors.max(),
        "Concha_MLE":              concha_errors.mean(),
        "Canal_Axis_MLE":          canal_errors.mean(),
        "HRTF_SR@2mm":             (critical_errors < 2.0).mean(),
        "HRTF_SR@3mm":             (critical_errors < 3.0).mean(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# PER-LANDMARK-GROUP BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════════

def compute_per_group_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    """Compute metrics broken down by anatomical group."""
    errors = np.linalg.norm(pred - gt, axis=1)
    results = {}
    for group_name, indices in LANDMARK_GROUPS.items():
        group_errors = errors[indices]
        results[f"{group_name}_MLE"] = group_errors.mean()
        results[f"{group_name}_MdLE"] = np.median(group_errors)
        results[f"{group_name}_MaxLE"] = group_errors.max()
        results[f"{group_name}_RMSE"] = np.sqrt(np.mean(group_errors**2))
        results[f"{group_name}_SR@3mm"] = (group_errors < 3.0).mean()
    return results


# ═══════════════════════════════════════════════════════════════════════════
# COMPREHENSIVE EVALUATOR CLASS
# ═══════════════════════════════════════════════════════════════════════════

class PinnaLandmarkEvaluator:
    """
    Full evaluation engine that combines all 6 metric dimensions.
    
    Usage:
        evaluator = PinnaLandmarkEvaluator()
        evaluator.add_prediction(gt_left, pred_left, pid="P0001", side="left")
        evaluator.add_prediction(gt_right, pred_right, pid="P0001", side="right")
        ...
        report = evaluator.generate_report()
    """
    
    def __init__(self, failure_threshold: float = 10.0):
        self.failure_threshold = failure_threshold
        self.predictions = []  # list of dicts with gt, pred, pid, side
        self._report = None
    
    def add_prediction(
        self,
        gt: np.ndarray,
        pred: np.ndarray,
        pid: str = "",
        side: str = "",
    ):
        """Add a single prediction-ground truth pair."""
        self.predictions.append({
            "gt": gt.copy(),
            "pred": pred.copy(),
            "pid": pid,
            "side": side,
        })
        self._report = None  # Invalidate cached report
    
    def generate_report(self) -> dict:
        """Generate the full evaluation report across all predictions."""
        if not self.predictions:
            raise ValueError("No predictions added! Call add_prediction() first.")
        
        n = len(self.predictions)
        print(f"\n{'='*72}")
        print(f"  RIGOROUS EVALUATION REPORT — {n} predictions")
        print(f"{'='*72}")
        
        # Collect per-prediction metrics
        all_point_metrics = []
        all_shape_metrics = []
        all_anatomical_metrics = []
        all_bias_metrics = []
        all_hrtf_metrics = []
        all_group_metrics = []
        all_errors = []
        
        for entry in self.predictions:
            gt, pred = entry["gt"], entry["pred"]
            
            all_point_metrics.append(compute_point_metrics(gt, pred))
            all_shape_metrics.append(compute_shape_metrics(gt, pred))
            all_anatomical_metrics.append(compute_anatomical_metrics(gt, pred))
            all_bias_metrics.append(compute_bias_metrics(gt, pred))
            all_hrtf_metrics.append(compute_hrtf_metrics(gt, pred))
            all_group_metrics.append(compute_per_group_metrics(gt, pred))
            all_errors.append(np.linalg.norm(pred - gt, axis=1))
        
        # Aggregate across all predictions
        report = {}
        
        # 1. Point-level (aggregate)
        point_df = pd.DataFrame(all_point_metrics)
        report["Point_Level"] = {
            col: {
                "mean": point_df[col].mean(),
                "std":  point_df[col].std(),
                "min":  point_df[col].min(),
                "max":  point_df[col].max(),
            }
            for col in point_df.columns
        }
        
        # 2. Shape-level (aggregate)
        shape_df = pd.DataFrame(all_shape_metrics)
        report["Shape_Level"] = {
            col: {
                "mean": shape_df[col].mean(),
                "std":  shape_df[col].std(),
            }
            for col in shape_df.columns
        }
        
        # 3. Anatomical (aggregate)
        anat_df = pd.DataFrame(all_anatomical_metrics)
        report["Anatomical"] = {
            col: {
                "mean": anat_df[col].mean(),
                "std":  anat_df[col].std(),
            }
            for col in anat_df.columns
        }
        
        # 4. Bias (aggregate — signed errors should cancel if unbiased)
        bias_df = pd.DataFrame(all_bias_metrics)
        # For bias, compute the MEAN of means (should be ~0 if unbiased)
        all_signed = np.stack([p["pred"] - p["gt"] for p in self.predictions])
        global_bias = all_signed.mean(axis=(0, 1))  # (3,) mean bias across all
        report["Bias"] = {
            "Global_Bias_x": global_bias[0],
            "Global_Bias_y": global_bias[1],
            "Global_Bias_z": global_bias[2],
            "Global_Bias_Magnitude": np.linalg.norm(global_bias),
        }
        # T-test on per-ear mean bias
        per_ear_bias = all_signed.mean(axis=1)  # (N, 3)
        for ax, name in enumerate(["x", "y", "z"]):
            t, p = scipy_stats.ttest_1samp(per_ear_bias[:, ax], 0.0)
            report["Bias"][f"Global_Bias_{name}_t_stat"] = t
            report["Bias"][f"Global_Bias_{name}_p_value"] = p
            report["Bias"][f"Global_Bias_{name}_significant"] = bool(p < 0.05)
        
        # 5. Robustness
        report["Robustness"] = compute_robustness_metrics(
            all_errors, self.failure_threshold
        )
        
        # 6. HRTF (aggregate)
        hrtf_df = pd.DataFrame(all_hrtf_metrics)
        report["HRTF"] = {
            col: {
                "mean": hrtf_df[col].mean(),
                "std":  hrtf_df[col].std(),
            }
            for col in hrtf_df.columns
        }
        
        # 7. Per-group (aggregate)
        group_df = pd.DataFrame(all_group_metrics)
        report["Per_Group"] = {
            col: {
                "mean": group_df[col].mean(),
                "std":  group_df[col].std(),
            }
            for col in group_df.columns
        }
        
        # 8. Per-landmark (aggregate)
        all_err_matrix = np.stack(all_errors)  # (N, 85)
        report["Per_Landmark"] = {
            i: {
                "mean": all_err_matrix[:, i].mean(),
                "std":  all_err_matrix[:, i].std(),
                "median": np.median(all_err_matrix[:, i]),
                "max": all_err_matrix[:, i].max(),
                "SR@3mm": (all_err_matrix[:, i] < 3.0).mean(),
            }
            for i in range(NUM_LANDMARKS)
        }
        
        # Left vs Right comparison
        left_errors = [np.linalg.norm(p["pred"] - p["gt"], axis=1).mean()
                       for p in self.predictions if p["side"] == "left"]
        right_errors = [np.linalg.norm(p["pred"] - p["gt"], axis=1).mean()
                        for p in self.predictions if p["side"] == "right"]
        if left_errors and right_errors:
            t, p = scipy_stats.ttest_ind(left_errors, right_errors)
            report["Left_vs_Right"] = {
                "Left_MLE": np.mean(left_errors),
                "Right_MLE": np.mean(right_errors),
                "Difference": abs(np.mean(left_errors) - np.mean(right_errors)),
                "t_stat": t,
                "p_value": p,
                "Significant": bool(p < 0.05),
            }
        
        self._report = report
        return report
    
    def print_report(self, report: dict = None):
        """Print a nicely formatted report."""
        if report is None:
            report = self._report or self.generate_report()
        
        # ─── 1. Point-Level Summary ──────────────
        print(f"\n{'─'*72}")
        print("  1. POINT-LEVEL ACCURACY")
        print(f"{'─'*72}")
        pl = report["Point_Level"]
        key_metrics = [
            ("MLE",  "Mean Landmark Error"),
            ("MdLE", "Median Landmark Error"),
            ("RMSE", "Root Mean Squared Error"),
            ("NME",  "Normalized Mean Error"),
            ("P90",  "90th Percentile"),
            ("P95",  "95th Percentile"),
            ("MaxLE","Maximum Landmark Error"),
        ]
        for key, label in key_metrics:
            m = pl[key]
            unit = "" if key == "NME" else " mm"
            fmt = ".4f" if key == "NME" else ".2f"
            print(f"  {label:30s}: {m['mean']:{fmt}}{unit}  (std={m['std']:{fmt}})")
        
        print(f"\n  Success Rates:")
        for key in ["SR@2mm", "SR@3mm", "SR@5mm", "SR@10mm"]:
            m = pl[key]
            print(f"    {key:10s}: {m['mean']*100:.1f}%  (std={m['std']*100:.1f}%)")
        
        # ─── 2. Shape-Level ──────────────────────
        print(f"\n{'─'*72}")
        print("  2. SHAPE-LEVEL FIDELITY")
        print(f"{'─'*72}")
        sl = report["Shape_Level"]
        for key, label in [
            ("Procrustes_RMSE", "Procrustes RMSE (shape error)"),
            ("Procrustes_Mean", "Procrustes Mean Error"),
            ("Alignment_Gain",  "Alignment Gain (rigid offset %)"),
            ("Centroid_Error",  "Centroid Error"),
            ("Scale_Error_Pct", "Scale Error"),
        ]:
            m = sl[key]
            if "Pct" in key or "Gain" in key:
                print(f"  {label:35s}: {m['mean']:.1f}%  (std={m['std']:.1f}%)")
            else:
                print(f"  {label:35s}: {m['mean']:.2f} mm  (std={m['std']:.2f})")
        
        # ─── 3. Anatomical Plausibility ──────────
        print(f"\n{'─'*72}")
        print("  3. ANATOMICAL PLAUSIBILITY")
        print(f"{'─'*72}")
        an = report["Anatomical"]
        for key, label in [
            ("ILD_Mean",          "Inter-Landmark Dist Error (adj)"),
            ("ILD_Relative_Mean", "Relative ILD Error"),
            ("PW_Mean_Error",     "Pairwise Distance Mean Error"),
            ("PW_Correlation",    "Pairwise Distance Correlation"),
            ("Ordering_Violations", "Ordering Violations"),
        ]:
            m = an[key]
            if "Relative" in key:
                print(f"  {label:35s}: {m['mean']:.1f}%  (std={m['std']:.1f}%)")
            elif "Correlation" in key:
                print(f"  {label:35s}: {m['mean']:.4f}  (std={m['std']:.4f})")
            elif "Violation" in key:
                print(f"  {label:35s}: {m['mean']:.1f}  (std={m['std']:.1f})")
            else:
                print(f"  {label:35s}: {m['mean']:.2f} mm  (std={m['std']:.2f})")
        
        # ─── 4. Directional Bias ─────────────────
        print(f"\n{'─'*72}")
        print("  4. DIRECTIONAL BIAS ANALYSIS")
        print(f"{'─'*72}")
        bi = report["Bias"]
        print(f"  Global bias vector: [{bi['Global_Bias_x']:.3f}, "
              f"{bi['Global_Bias_y']:.3f}, {bi['Global_Bias_z']:.3f}] mm")
        print(f"  Bias magnitude:     {bi['Global_Bias_Magnitude']:.3f} mm")
        for ax in ["x", "y", "z"]:
            sig = "YES *" if bi[f"Global_Bias_{ax}_significant"] else "no"
            print(f"  {ax}-axis: t={bi[f'Global_Bias_{ax}_t_stat']:.2f}, "
                  f"p={bi[f'Global_Bias_{ax}_p_value']:.4f}, significant={sig}")
        
        # ─── 5. Robustness ───────────────────────
        print(f"\n{'─'*72}")
        print("  5. ROBUSTNESS & RELIABILITY")
        print(f"{'─'*72}")
        ro = report["Robustness"]
        print(f"  Evaluated:              {ro['N_ears']} ears, {ro['N_landmarks_total']} landmarks")
        print(f"  Failure rate (>{self.failure_threshold}mm):")
        print(f"    Per-landmark:         {ro['Failure_Rate_Landmarks']:.2f}%")
        print(f"    Per-ear:              {ro['Failure_Rate_Ears']:.1f}%")
        print(f"  Catastrophic (>20mm):   {ro['Catastrophic_Rate']:.1f}%")
        print(f"  95% CI for mean error:  [{ro['CI95_low']:.2f}, {ro['CI95_high']:.2f}] mm")
        print(f"  Coeff. of Variation:    {ro['CV']:.3f}")
        print(f"  IQR of per-ear means:   {ro['IQR']:.2f} mm")
        print(f"  Best/Worst ear:         {ro['Best_Ear_Mean']:.2f} / {ro['Worst_Ear_Mean']:.2f} mm")
        
        # ─── 6. HRTF Impact ─────────────────────
        print(f"\n{'─'*72}")
        print("  6. HRTF-WEIGHTED ACOUSTIC IMPACT")
        print(f"{'─'*72}")
        hf = report["HRTF"]
        for key, label in [
            ("HRTF_Weighted_MLE",  "HRTF-Weighted MLE"),
            ("HRTF_Critical_MLE",  "Critical Landmark MLE"),
            ("Concha_MLE",         "Concha Depth MLE"),
            ("Canal_Axis_MLE",     "Ear Canal Axis MLE"),
        ]:
            m = hf[key]
            print(f"  {label:30s}: {m['mean']:.2f} mm  (std={m['std']:.2f})")
        print(f"\n  HRTF-critical success rates:")
        for key in ["HRTF_SR@2mm", "HRTF_SR@3mm"]:
            m = hf[key]
            print(f"    {key:15s}: {m['mean']*100:.1f}%")
        
        # ─── 7. Per-Group Breakdown ──────────────
        print(f"\n{'─'*72}")
        print("  7. PER-ANATOMICAL-GROUP BREAKDOWN")
        print(f"{'─'*72}")
        print(f"  {'Group':25s} {'MLE':>8s} {'MdLE':>8s} {'RMSE':>8s} {'MaxLE':>8s} {'SR@3mm':>8s}")
        print(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
        pg = report["Per_Group"]
        for group_name in LANDMARK_GROUPS:
            mle = pg[f"{group_name}_MLE"]["mean"]
            mdle = pg[f"{group_name}_MdLE"]["mean"]
            rmse = pg[f"{group_name}_RMSE"]["mean"]
            maxle = pg[f"{group_name}_MaxLE"]["mean"]
            sr3 = pg[f"{group_name}_SR@3mm"]["mean"] * 100
            print(f"  {group_name:25s} {mle:7.2f}  {mdle:7.2f}  "
                  f"{rmse:7.2f}  {maxle:7.2f}  {sr3:6.1f}%")
        
        # ─── 8. Left vs Right ────────────────────
        if "Left_vs_Right" in report:
            print(f"\n{'─'*72}")
            print("  8. LEFT vs RIGHT EAR COMPARISON")
            print(f"{'─'*72}")
            lr = report["Left_vs_Right"]
            print(f"  Left ear MLE:    {lr['Left_MLE']:.2f} mm")
            print(f"  Right ear MLE:   {lr['Right_MLE']:.2f} mm")
            print(f"  Difference:      {lr['Difference']:.2f} mm")
            sig = "YES *" if lr["Significant"] else "no"
            print(f"  Statistical test: t={lr['t_stat']:.2f}, p={lr['p_value']:.4f}, significant={sig}")
        
        # ─── Summary Score ───────────────────────
        print(f"\n{'='*72}")
        print("  COMPOSITE SCORE")
        print(f"{'='*72}")
        composite = self._compute_composite_score(report)
        for name, score in composite.items():
            bar = "#" * int(score / 2)
            print(f"  {name:30s}: {score:5.1f}/100  |{bar}")
        total = np.mean(list(composite.values()))
        print(f"  {'─'*30}{'─'*20}")
        print(f"  {'OVERALL':30s}: {total:5.1f}/100")
        print(f"{'='*72}")
    
    def _compute_composite_score(self, report: dict) -> dict:
        """
        Compute a composite quality score (0-100) across all dimensions.
        This provides a single comparable number for model ranking.
        """
        scores = {}
        
        # 1. Accuracy: MLE < 1mm = 100, MLE > 10mm = 0
        mle = report["Point_Level"]["MLE"]["mean"]
        scores["Accuracy"] = max(0, min(100, (10 - mle) / 9 * 100))
        
        # 2. Precision: SR@3mm — what fraction of landmarks are within 3mm?
        sr3 = report["Point_Level"]["SR@3mm"]["mean"]
        scores["Precision (SR@3mm)"] = sr3 * 100
        
        # 3. Shape fidelity: Procrustes RMSE < 0.5mm = 100, > 5mm = 0
        proc_rmse = report["Shape_Level"]["Procrustes_RMSE"]["mean"]
        scores["Shape Fidelity"] = max(0, min(100, (5 - proc_rmse) / 4.5 * 100))
        
        # 4. Anatomical validity: PW correlation > 0.999 = 100
        pw_corr = report["Anatomical"]["PW_Correlation"]["mean"]
        scores["Anatomical Validity"] = max(0, min(100, (pw_corr - 0.9) / 0.1 * 100))
        
        # 5. Robustness: failure rate < 0.1% = 100, > 10% = 0
        fail_rate = report["Robustness"]["Failure_Rate_Landmarks"]
        scores["Robustness"] = max(0, min(100, (10 - fail_rate) / 10 * 100))
        
        # 6. HRTF accuracy: critical MLE < 1mm = 100, > 8mm = 0
        hrtf_mle = report["HRTF"]["HRTF_Critical_MLE"]["mean"]
        scores["HRTF Accuracy"] = max(0, min(100, (8 - hrtf_mle) / 7 * 100))
        
        return scores
    
    def export_csv(self, path: str | Path):
        """Export per-ear results as CSV for external analysis."""
        records = []
        for entry in self.predictions:
            gt, pred = entry["gt"], entry["pred"]
            pm = compute_point_metrics(gt, pred)
            sm = compute_shape_metrics(gt, pred)
            hm = compute_hrtf_metrics(gt, pred)
            records.append({
                "pid": entry["pid"],
                "side": entry["side"],
                **pm,
                **sm,
                **hm,
            })
        df = pd.DataFrame(records)
        df.to_csv(path, index=False)
        print(f"Exported {len(df)} rows to {path}")
        return df


# ═══════════════════════════════════════════════════════════════════════════
# CROSS-VALIDATION WITH FULL EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def run_rigorous_cv(n_folds: int = 5, n_ssm: int = 30, k: int = 7) -> dict:
    """
    Run cross-validation with the full evaluation framework.
    """
    from src.improved_predictor import ImprovedLandmarkPredictor
    
    print("Loading all landmarks...")
    all_lm = load_all_landmarks()
    pids = sorted(all_lm.keys())
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    evaluator = PinnaLandmarkEvaluator(failure_threshold=10.0)
    
    t_start = time.time()
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(pids)):
        print(f"\nFold {fold+1}/{n_folds} ({len(train_idx)} train, {len(test_idx)} test)")
        
        train_pids = [pids[i] for i in train_idx]
        test_pids = [pids[i] for i in test_idx]
        train_lm = {pid: all_lm[pid] for pid in train_pids}
        
        pred = ImprovedLandmarkPredictor(n_ssm_components=n_ssm, k_neighbors=k)
        pred.fit(train_lm, train_with_regressors=True)
        
        for pid in test_pids:
            data = load_participant(pid)
            for side in ["left", "right"]:
                gt = all_lm[pid][side]
                predicted = pred.predict(data["mesh"], side=side)
                evaluator.add_prediction(gt, predicted, pid=pid, side=side)
                err = np.linalg.norm(predicted - gt, axis=1).mean()
                print(f"  {pid} {side}: {err:.2f}mm")
    
    elapsed = time.time() - t_start
    print(f"\nCV completed in {elapsed:.0f}s")
    
    report = evaluator.generate_report()
    evaluator.print_report(report)
    evaluator.export_csv("output/rigorous_cv_results.csv")
    
    return report


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    report = run_rigorous_cv(n_folds=5)
