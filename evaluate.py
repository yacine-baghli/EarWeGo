"""
============================================================================
EVALUATION PIPELINE — Huawei Tech Arena 2026 Pinna Landmark Extraction
============================================================================

Runs the official challenge evaluation metric on the dataset.
Optionally performs a detailed 6-dimensional diagnostic analysis and 
saves report figures to the 'output/' directory.

Usage:
    python evaluate.py --mesh-dir "path/to/mesh" --landmarks-dir "path/to/landmarks"
"""

import argparse
import numpy as np
from pathlib import Path
import sys
import time

# Ensure core package is in the Python search path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.dataset import Dataset
from src.estimator import LandmarkExtractor
from src.metrics import compute_mean_landmark_distance
from src.evaluation import PinnaLandmarkEvaluator
from src.eval_plots import plot_evaluation_dashboard


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Pinna Landmark Extraction Performance")
    
    # Dataset configurations
    parser.add_argument(
        "--mesh-dir",
        type=str,
        default="2026 Munich Tech Arena - Datas/2026 Munich Tech Arena - Datas/mesh",
        help="Path to directory containing PLY mesh files"
    )
    parser.add_argument(
        "--landmarks-dir",
        type=str,
        default="2026 Munich Tech Arena - Datas/2026 Munich Tech Arena - Datas/landmarks",
        help="Path to directory containing CSV landmark files"
    )
    
    # Checkpoint configuration
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory containing trained model checkpoints"
    )
    
    # Diagnostic configurations
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        default=True,
        help="Run rigorous 6-dimensional evaluation and save plots"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save evaluation reports and plots"
    )
    parser.add_argument(
        "--quick-test",
        type=int,
        default=0,
        help="If positive, runs evaluation only on the first N subjects"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    mesh_path = Path(args.mesh_dir)
    landmarks_path = Path(args.landmarks_dir)
    models_path = Path(args.models_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("=" * 72)
    print("  EVALUATION PIPELINE — PINNA LANDMARK EXTRACTION")
    print("=" * 72)
    print(f"Mesh directory:      {mesh_path}")
    print(f"Landmark directory:  {landmarks_path}")
    print(f"Models directory:    {models_path}")
    print(f"Diagnostic reports:  {args.diagnostic}")
    print("=" * 72)
    
    if not mesh_path.exists() or not landmarks_path.exists():
        print(f"Error: Dataset directories do not exist. Please check the paths.")
        sys.exit(1)
        
    t_start = time.time()
    
    # 1. Load dataset
    print("\n[1/3] Loading dataset...")
    dataset = Dataset(mesh_dir=str(mesh_path), landmarks_dir=str(landmarks_path))
    num_subjects = len(dataset)
    print(f"  Loaded {num_subjects} subjects.")
    
    # Limit if quick-test is active
    if args.quick_test > 0:
        eval_count = min(args.quick_test, num_subjects)
        print(f"  [Quick Test Mode] Evaluating on the first {eval_count} subjects only.")
    else:
        eval_count = num_subjects
        
    # 2. Instantiate LandmarkExtractor
    print("\n[2/3] Instantiating official LandmarkExtractor...")
    try:
        extractor = LandmarkExtractor(
            detector_path=str(models_path / "ear_detector.pkl"),
            predictor_path=str(models_path / "landmark_predictor.pkl")
        )
    except Exception as e:
        print(f"Error instantiating extractor: {e}")
        print("Please verify that models are trained and checkpoints exist.")
        sys.exit(1)
        
    # 3. Perform prediction and compute official metrics
    print("\n[3/3] Running prediction and metric computation...")
    subject_distances = []
    
    # Diagnostics evaluator
    diag_evaluator = PinnaLandmarkEvaluator(failure_threshold=10.0) if args.diagnostic else None
    
    for idx in range(eval_count):
        subject_id = dataset.get_identifier(idx)
        mesh, gt_left, gt_right = dataset[idx]
        
        # Call official extract method
        try:
            pred_left, pred_right = extractor.extract(mesh)
        except Exception as e:
            print(f"  Subject {subject_id} failed: {e}")
            continue
            
        # Compute official distances
        d_left = compute_mean_landmark_distance(pred_left, gt_left)
        d_right = compute_mean_landmark_distance(pred_right, gt_right)
        d_subject = (d_left + d_right) / 2
        
        print(f"  Subject {idx+1}/{eval_count} ({subject_id}): MD = {d_subject:.3f} mm (L: {d_left:.3f} | R: {d_right:.3f})")
        subject_distances.append(d_subject)
        
        # Collect for diagnostic evaluation
        if args.diagnostic:
            diag_evaluator.add_prediction(gt_left, pred_left, pid=subject_id, side="left")
            diag_evaluator.add_prediction(gt_right, pred_right, pid=subject_id, side="right")
            
    # Print official overall score
    overall_mean = np.mean(subject_distances)
    print("\n" + "=" * 72)
    print("  EVALUATION SUMMARY")
    print("=" * 72)
    print(f"  Official Mean Distance (MD): {overall_mean:.4f} mm")
    print(f"  Standard Deviation:          {np.std(subject_distances):.4f} mm")
    print(f"  Median Distance:             {np.median(subject_distances):.4f} mm")
    print(f"  Max Distance:                {np.max(subject_distances):.4f} mm")
    print(f"  Min Distance:                {np.min(subject_distances):.4f} mm")
    print(f"  Total processing time:       {time.time() - t_start:.1f}s")
    print("=" * 72)
    
    # Produce detailed diagnostics report and plots
    if args.diagnostic and len(subject_distances) > 0:
        print("\nGenerating diagnostic reports and plots...")
        report = diag_evaluator.generate_report()
        diag_evaluator.print_report(report)
        diag_evaluator.export_csv(output_path / "rigorous_evaluation_results.csv")
        plot_evaluation_dashboard(diag_evaluator, save_dir=str(output_path))
        print(f"Reports saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
