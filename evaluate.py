"""
============================================================================
EVALUATION PIPELINE — Huawei Tech Arena 2026 Pinna Landmark Extraction
============================================================================

Evaluates a trained model on the frozen split and writes results into the
run directory. Updates runs/index.csv with headline metrics.

Usage:
    python evaluate.py --run runs/<run_id>
    python evaluate.py --run runs/<run_id> --split test
    python evaluate.py --models-dir models  # legacy mode
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
    
    # Run-based interface (preferred)
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Path to a run directory (e.g. runs/20260708_1530_baseline_ab12cd). "
             "Loads weights from <run>/weights/ and saves results to <run>/results/"
    )
    
    # Legacy fallback
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
    
    # Dataset
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
    
    # Evaluation options
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
    
    # ── Resolve model and output paths ───────────────────────────────────────
    run_dir = None
    if args.run:
        run_dir = Path(args.run)
        if not run_dir.exists():
            print(f"Error: Run directory '{run_dir}' does not exist.")
            sys.exit(1)
        models_path = run_dir / "weights"
        output_path = run_dir / "results"
        output_path.mkdir(exist_ok=True)
        run_name = run_dir.name
    else:
        models_path = Path(args.models_dir)
        output_path = Path(args.output_dir)
        output_path.mkdir(exist_ok=True)
        run_name = "(legacy)"
    
    # ── Test split warning ───────────────────────────────────────────────────
    if args.split == "test":
        print("\n" + "!" * 72)
        print("  WARNING: Evaluating on TEST split.")
        print("  This should be a one-shot final check. Do not iterate on test results.")
        print("!" * 72 + "\n")
    
    print("=" * 72)
    print("  EVALUATION PIPELINE -- PINNA LANDMARK EXTRACTION")
    print("=" * 72)
    print(f"Run:                 {run_name}")
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
    
    # 1. Load dataset with split
    print("\n[1/3] Loading dataset...")
    dataset = Dataset(mesh_dir=str(mesh_path), landmarks_dir=str(landmarks_path), split=args.split)
    num_subjects = len(dataset)
    print(f"  Loaded {num_subjects} subjects.")
    
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
        
    # ── Loud Leakage Guard ───────────────────────────────────────────────────
    from src.splits import load_splits
    try:
        train_pids, val_pids, test_pids = load_splits(mesh_dir=mesh_path)
        
        # Guard 1: Verify split files are disjoint
        overlap_tv = set(train_pids) & set(val_pids)
        overlap_tt = set(train_pids) & set(test_pids)
        overlap_vt = set(val_pids) & set(test_pids)
        if overlap_tv or overlap_tt or overlap_vt:
            print("\n" + "!" * 72)
            print("  CRITICAL ERROR: Overlapping PIDs found between split files!")
            print("!" * 72 + "\n")
            sys.exit(2)
            
        # Guard 2: Verify model's shape bank does not overlap eval split
        if hasattr(extractor, "predictor") and extractor.predictor is not None:
            if hasattr(extractor.predictor, "pids") and extractor.predictor.pids is not None:
                trained_pids = set(extractor.predictor.pids)
                eval_pids = set(dataset.subject_ids)
                leakage = trained_pids & eval_pids
                
                if leakage:
                    print("\n" + "!" * 72)
                    print("  CRITICAL ERROR: DATA LEAKAGE DETECTED!")
                    print(f"  The model's shape bank contains PIDs in the '{args.split}' split!")
                    print(f"  Leakage count: {len(leakage)} PIDs")
                    print(f"  Leaked PIDs: {sorted(list(leakage))[:15]}...")
                    print("!" * 72 + "\n")
                    sys.exit(3)
                    
            # Guard 3: Assert shape bank contains ONLY train-split subjects
            if hasattr(extractor.predictor, "pids") and extractor.predictor.pids is not None:
                shape_bank = set(extractor.predictor.pids)
                non_train = shape_bank - set(train_pids)
                if non_train:
                    print("\n" + "!" * 72)
                    print("  WARNING: Shape bank contains non-train PIDs!")
                    print(f"  Non-train PIDs in shape bank: {sorted(non_train)[:10]}...")
                    print("  KNN blending may leak val/test data.")
                    print("!" * 72 + "\n")
                    
    except Exception as e:
        print(f"  [Leakage Guard Warning] Could not run splits validation check: {e}")
        
    # 3. Predict and compute metrics
    print("\n[3/3] Running prediction and metric computation...")
    subject_distances = []
    diag_evaluator = PinnaLandmarkEvaluator(failure_threshold=10.0) if args.diagnostic else None
    
    for idx in range(eval_count):
        subject_id = dataset.get_identifier(idx)
        mesh, gt_left, gt_right = dataset[idx]
        
        try:
            pred_left, pred_right = extractor.extract(mesh)
        except Exception as e:
            print(f"  Subject {subject_id} failed: {e}")
            continue
            
        d_left = compute_mean_landmark_distance(pred_left, gt_left)
        d_right = compute_mean_landmark_distance(pred_right, gt_right)
        d_subject = (d_left + d_right) / 2
        
        print(f"  Subject {idx+1}/{eval_count} ({subject_id}): MD = {d_subject:.3f} mm (L: {d_left:.3f} | R: {d_right:.3f})")
        subject_distances.append(d_subject)
        
        if args.diagnostic:
            diag_evaluator.add_prediction(gt_left, pred_left, pid=subject_id, side="left")
            diag_evaluator.add_prediction(gt_right, pred_right, pid=subject_id, side="right")
            
    # ── Compute summary ──────────────────────────────────────────────────────
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
    
    # ── Build summary dict ───────────────────────────────────────────────────
    eval_summary = {
        "run": run_name if args.run else "legacy",
        "split": args.split,
        "num_subjects": len(subject_distances),
        "mean_distance_mm": round(float(overall_mean), 4),
        "std_distance_mm": round(float(np.std(subject_distances)), 4),
        "median_distance_mm": round(float(np.median(subject_distances)), 4),
        "max_distance_mm": round(float(np.max(subject_distances)), 4),
        "min_distance_mm": round(float(np.min(subject_distances)), 4),
        "processing_time_s": round(elapsed, 1),
    }
    
    # Add P90/P95 and success rates
    all_dists = np.array(subject_distances)
    eval_summary["P90_mm"] = round(float(np.percentile(all_dists, 90)), 4)
    eval_summary["P95_mm"] = round(float(np.percentile(all_dists, 95)), 4)
    
    # Per-subject breakdown
    eval_summary["per_subject"] = {
        dataset.get_identifier(i): round(float(d), 4)
        for i, d in enumerate(subject_distances)
    }
    
    # Extract rigorous metrics if available
    if args.diagnostic and diag_evaluator:
        report = diag_evaluator.generate_report()
        pl = report.get("Point_Level", {})
        
        # Success rates
        for sr_key in ["SR@2mm", "SR@3mm", "SR@5mm"]:
            if sr_key in pl:
                eval_summary[sr_key] = round(float(pl[sr_key]["mean"]) * 100, 1)
        
        # Per-region
        pg = report.get("Per_Group", {})
        for group in ["Helix", "Antihelix/Concha", "Outer boundary", "Cross-section"]:
            key = f"{group}_MLE"
            if key in pg:
                eval_summary[f"{group}_MLE_mm"] = round(float(pg[key]["mean"]), 4)
        
        # HRTF
        hf = report.get("HRTF", {})
        if "Concha_MLE" in hf:
            eval_summary["Concha_MLE_mm"] = round(float(hf["Concha_MLE"]["mean"]), 4)
    
    # ── Save summary JSON ────────────────────────────────────────────────────
    summary_file = output_path / f"summary_{args.split}.json"
    with open(summary_file, "w") as f:
        json.dump(eval_summary, f, indent=2)
    print(f"\nEvaluation summary saved to: {summary_file}")
    
    # ── Save diagnostic reports ──────────────────────────────────────────────
    if args.diagnostic and len(subject_distances) > 0:
        print("\nGenerating diagnostic reports and plots...")
        diag_evaluator.print_report(report)
        
        metrics_file = output_path / f"metrics_{args.split}.csv"
        diag_evaluator.export_csv(metrics_file)
        plot_evaluation_dashboard(diag_evaluator, save_dir=str(output_path))
        
        # Rename dashboard plots with split suffix
        for name in ["evaluation_dashboard.png", "shape_fidelity_analysis.png", "hrtf_impact_analysis.png"]:
            src = output_path / name
            dst = output_path / name.replace(".png", f"_{args.split}.png")
            if src.exists() and not dst.exists():
                src.rename(dst)
            elif src.exists():
                # Overwrite
                import shutil
                shutil.move(str(src), str(dst))
        
        print(f"Reports saved to: {output_path.resolve()}")
    
    # ── Register in index.csv ────────────────────────────────────────────────
    if run_dir:
        from src.runs import register_run
        register_run(run_dir, eval_summary)
    
    # ── Update metadata with eval duration ───────────────────────────────────
    if run_dir:
        meta_path = run_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            durations = meta.get("durations", {})
            durations[f"eval_{args.split}_s"] = round(elapsed, 1)
            meta["durations"] = durations
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2, default=str)


if __name__ == "__main__":
    main()
