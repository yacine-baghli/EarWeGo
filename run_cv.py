"""
============================================================================
CROSS-VALIDATION RUNNER — Pinna Landmark Extraction (v2.0)
============================================================================

Runs a proper, leakage-free 5-fold cross-validation on the dataset.
In each fold, the EarDetector and LandmarkPredictor are trained on the 
training subjects and evaluated on the test subjects without test-time landmarks.

Usage:
    python run_cv.py
"""

import numpy as np
from sklearn.model_selection import KFold
from pathlib import Path
import sys
import time

# Ensure core package is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.dataset import Dataset
from src.ear_detector import EarDetector
from src.predictor import LandmarkPredictor
from src.metrics import compute_mean_landmark_distance
from src.data_loader import NUM_LANDMARKS


def main():
    print("=" * 72)
    print("  LEAKAGE-FREE 5-FOLD CROSS-VALIDATION HARNESS (v2.0)")
    print("=" * 72)
    
    # Paths
    mesh_dir = "2026 Munich Tech Arena - Datas/2026 Munich Tech Arena - Datas/mesh"
    landmarks_dir = "2026 Munich Tech Arena - Datas/2026 Munich Tech Arena - Datas/landmarks"
    
    dataset = Dataset(mesh_dir=mesh_dir, landmarks_dir=landmarks_dir)
    pids = sorted(dataset.subject_ids)
    n_subjects = len(pids)
    
    print(f"Loaded {n_subjects} subjects.")
    
    # Reconstruct all landmarks dictionary for fitting folds
    print("Formatting training database...")
    all_landmarks = {}
    for idx in range(n_subjects):
        pid = dataset.get_identifier(idx)
        _, lm_l, lm_r = dataset[idx]
        all_landmarks[pid] = {
            "left": lm_l,
            "right": lm_r
        }
        
    N_FOLDS = 5
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
    all_errors = []
    per_landmark_errors = {i: [] for i in range(NUM_LANDMARKS)}
    
    t_start = time.time()
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(pids)):
        print(f"\nFold {fold+1}/{N_FOLDS} ({len(train_idx)} train, {len(test_idx)} test)")
        print("-" * 50)
        
        train_pids = [pids[i] for i in train_idx]
        test_pids = [pids[i] for i in test_idx]
        train_lm = {pid: all_landmarks[pid] for pid in train_pids}
        
        # 1. Fit detector on training split only
        print("  Fitting Ear Detector...")
        detector = EarDetector()
        detector.fit(train_lm, n_mesh_samples=20)
        
        # 2. Fit predictor on training split only
        print("  Fitting Landmark Predictor...")
        predictor = LandmarkPredictor(n_ssm_components=30, k_neighbors=7)
        predictor.fit(train_lm, train_with_regressors=True)
        
        # 3. Evaluate on test split
        fold_errors = []
        for pid in test_pids:
            subject_idx = pids.index(pid)
            mesh, gt_left, gt_right = dataset[subject_idx]
            
            for side, gt in [("left", gt_left), ("right", gt_right)]:
                try:
                    predicted = predictor.predict(mesh, side=side, ear_detector=detector)
                    err = compute_mean_landmark_distance(predicted, gt)
                    fold_errors.append(err)
                    all_errors.append(err)
                    
                    # Track per-landmark errors
                    diffs = np.linalg.norm(predicted - gt, axis=1)
                    for i in range(NUM_LANDMARKS):
                        per_landmark_errors[i].append(diffs[i])
                except Exception as e:
                    print(f"    Error on {pid} {side}: {e}")
                    
            left_err = fold_errors[-2]
            right_err = fold_errors[-1]
            print(f"    {pid}: L={left_err:.2f} mm | R={right_err:.2f} mm | avg={np.mean([left_err, right_err]):.2f} mm")
            
        print(f"  Fold {fold+1} mean error: {np.mean(fold_errors):.4f} mm")
        
    elapsed = time.time() - t_start
    all_errors = np.array(all_errors)
    
    # 4. Print overall CV summary report
    print("\n" + "=" * 72)
    print("  CROSS-VALIDATION HARNESS COMPLETE")
    print("=" * 72)
    print(f"  Mean Distance (MD):  {all_errors.mean():.4f} mm")
    print(f"  Median error:        {np.median(all_errors):.4f} mm")
    print(f"  Std error:           {all_errors.std():.4f} mm")
    print(f"  90th percentile:     {np.percentile(all_errors, 90):.4f} mm")
    print(f"  95th percentile:     {np.percentile(all_errors, 95):.4f} mm")
    print(f"  Max error:           {all_errors.max():.4f} mm")
    print(f"  Total elapsed time:  {elapsed:.1f}s")
    print("=" * 72)
    
    # Save results to output
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    np.savez(
        output_dir / "cv_results.npz",
        all_errors=all_errors,
        per_landmark_errors={str(k): np.array(v) for k, v in per_landmark_errors.items()},
    )
    print(f"CV results successfully saved to: {output_dir / 'cv_results.npz'}")


if __name__ == "__main__":
    main()
