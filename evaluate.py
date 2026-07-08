"""
============================================================================
EVALUATION PIPELINE — Huawei Tech Arena 2026 Pinna Landmark Extraction
============================================================================

Runs the official challenge evaluation metric on a trained experiment.
Saves all results (CSV, plots, summary) into the experiment's results/ folder.

Usage:
    python evaluate.py --experiment-dir experiments/baseline_20260708_194000
    python evaluate.py --experiment-dir experiments/baseline_20260708_194000 --split test
"""

import argparse
import json
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
    
    # Experiment directory (preferred)
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default=None,
        help="Path to a versioned experiment directory (e.g. experiments/baseline_20260708_194000). "
             "Models are loaded from <experiment-dir>/models/ and results saved to <experiment-dir>/results/"
    )
    
    # Legacy fallback: direct model/output paths
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="(Legacy) Directory containing trained model checkpoints"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="(Legacy) Directory to save evaluation reports and plots"
    )
    
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
    
    # Diagnostic configurations
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="The dataset partition split to evaluate on (default: 'val')"
    )
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        default=True,
        help="Run rigorous 6-dimensional evaluation and save plots"
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
    
    # Resolve model and output directories based on experiment-dir or legacy paths
    if args.experiment_dir:
        exp_dir = Path(args.experiment_dir)
        if not exp_dir.exists():
            print(f"Error: Experiment directory '{exp_dir}' does not exist.")
            sys.exit(1)
        models_path = exp_dir / "models"
        output_path = exp_dir / "results"
        output_path.mkdir(exist_ok=True)
        exp_name = exp_dir.name
    else:
        models_path = Path(args.models_dir)
        output_path = Path(args.output_dir)
        output_path.mkdir(exist_ok=True)
        exp_name = "(legacy)"
    
    print("=" * 72)
    print("  EVALUATION PIPELINE -- PINNA LANDMARK EXTRACTION")
    print("=" * 72)
    print(f"Experiment:          {exp_name}")
    print(f"Models directory:    {models_path}")
    print(f"Results directory:   {output_path}")
    print(f"Mesh directory:      {mesh_path}")
    print(f"Landmark directory:  {landmarks_path}")
    print(f"Evaluation split:    {args.split}")
    print(f"Diagnostic reports:  {args.diagnostic}")
    print("=" * 72)
    
    if not mesh_path.exists() or not landmarks_path.exists():
        print(f"Error: Dataset directories do not exist. Please check the paths.")
        sys.exit(1)
        
    t_start = time.time()
    
    # 1. Load dataset with split configurations
    print("\n[1/3] Loading dataset...")
    dataset = Dataset(mesh_dir=str(mesh_path), landmarks_dir=str(landmarks_path), split=args.split)
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
        
    # --- Loud Leakage Guard ---
    from src.splits import load_splits
    try:
        train_pids, val_pids, test_pids = load_splits(mesh_dir=mesh_path)
        
        # Guard 1: Verify split files are disjoint
        overlap_train_val = set(train_pids) & set(val_pids)
        overlap_train_test = set(train_pids) & set(test_pids)
        overlap_val_test = set(val_pids) & set(test_pids)
        if overlap_train_val or overlap_train_test or overlap_val_test:
            print("\n" + "!" * 72)
            print("  CRITICAL ERROR: Overlapping PIDs found between split files!")
            print(f"  Train/Val overlap: {overlap_train_val}")
            print(f"  Train/Test overlap: {overlap_train_test}")
            print(f"  Val/Test overlap: {overlap_val_test}")
            print("!" * 72 + "\n")
            sys.exit(2)
            
        # Guard 2: Verify model has not trained on evaluation partition
        if hasattr(extractor, "predictor") and extractor.predictor is not None:
            if hasattr(extractor.predictor, "pids") and extractor.predictor.pids is not None:
                trained_pids = set(extractor.predictor.pids)
                eval_pids = set(dataset.subject_ids)
                leakage = trained_pids & eval_pids
                
                if leakage:
                    print("\n" + "!" * 72)
                    print("  CRITICAL ERROR: DATA LEAKAGE DETECTED!")
                    print(f"  The model checkpoint was trained on PIDs in the evaluation split '{args.split}'!")
                    print(f"  Leakage count: {len(leakage)} PIDs")
                    print(f"  Leaked PIDs: {sorted(list(leakage))[:15]}...")
                    print("!" * 72 + "\n")
                    sys.exit(3)
    except Exception as e:
        # If splits files aren't found on clean workspace or missing dataset
        print(f"  [Leakage Guard Warning] Could not run splits validation check: {e}")
        
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
    elapsed = time.time() - t_start
    
    print("\n" + "=" * 72)
    print("  EVALUATION SUMMARY")
    print("=" * 72)
    print(f"  Official Mean Distance (MD): {overall_mean:.4f} mm")
    print(f"  Standard Deviation:          {np.std(subject_distances):.4f} mm")
    print(f"  Median Distance:             {np.median(subject_distances):.4f} mm")
    print(f"  Max Distance:                {np.max(subject_distances):.4f} mm")
    print(f"  Min Distance:                {np.min(subject_distances):.4f} mm")
    print(f"  Total processing time:       {elapsed:.1f}s")
    print("=" * 72)
    
    # Save evaluation summary JSON into experiment directory
    eval_summary = {
        "experiment": exp_name if args.experiment_dir else "legacy",
        "split": args.split,
        "num_subjects": len(subject_distances),
        "mean_distance_mm": round(float(overall_mean), 4),
        "std_distance_mm": round(float(np.std(subject_distances)), 4),
        "median_distance_mm": round(float(np.median(subject_distances)), 4),
        "max_distance_mm": round(float(np.max(subject_distances)), 4),
        "min_distance_mm": round(float(np.min(subject_distances)), 4),
        "processing_time_s": round(elapsed, 1),
        "per_subject": {
            dataset.get_identifier(i): round(float(d), 4)
            for i, d in enumerate(subject_distances)
        },
    }
    summary_file = output_path / f"eval_summary_{args.split}.json"
    with open(summary_file, "w") as f:
        json.dump(eval_summary, f, indent=2)
    print(f"\nEvaluation summary saved to: {summary_file}")
    
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
