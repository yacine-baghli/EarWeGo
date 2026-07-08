"""
============================================================================
TRAINING PIPELINE — Huawei Tech Arena 2026 Pinna Landmark Extraction
============================================================================

Config-driven training: loads hyperparameters from YAML, trains on the
frozen train split, and saves weights + JSON sidecar into a run directory.

Usage:
    python train.py --config configs/v1_baseline.yaml --name baseline
    python train.py --config configs/v2_arclen.yaml --name arclen --set model.k_neighbors=10
    python train.py  # (back-compat: uses configs/base.yaml, saves to models/)
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys
import time

# Ensure core package is in the Python search path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config, resolve_data_paths
from src.dataset import Dataset
from src.ear_detector import EarDetector
from src.predictor import LandmarkPredictor
from src.splits import load_splits


def parse_args():
    parser = argparse.ArgumentParser(description="Train Pinna Landmark Detector and Predictor")
    
    # Config-driven interface
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to YAML config file (default: configs/base.yaml)"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="baseline",
        help="Name for this experiment run"
    )
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        help="Override config values, e.g. --set model.k_neighbors=10 model.blend_alpha=0.5"
    )
    
    # Split override
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Override the training split (default: 'train')"
    )
    parser.add_argument(
        "--include-val",
        action="store_true",
        help="Combine training and validation splits for final model training"
    )
    
    # Output: if --run-dir is provided, save into that run directory.
    # Otherwise create a new run (or use --output-dir for legacy mode).
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Save weights into an existing run directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="(Legacy) Override output directory for model checkpoints"
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Skip creating a run directory, save directly to models/"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # ── Load config ──────────────────────────────────────────────────────────
    overrides = {}
    for item in args.set:
        if "=" in item:
            key, val = item.split("=", 1)
            overrides[key] = val
    
    cfg = load_config(args.config, overrides=overrides if overrides else None)
    cfg = resolve_data_paths(cfg)
    
    # Extract config sections
    model_cfg = cfg.get("model", {})
    detector_cfg = cfg.get("detector", {})
    seed = cfg.get("seed", 42)
    
    # Paths
    mesh_path = Path(cfg["data"]["mesh_dir"])
    landmarks_path = Path(cfg["data"]["landmarks_dir"])
    
    # ── Determine output directory ───────────────────────────────────────────
    run_dir = None
    if args.run_dir:
        run_dir = Path(args.run_dir)
        weights_path = run_dir / "weights"
        weights_path.mkdir(parents=True, exist_ok=True)
    elif args.no_run or args.output_dir:
        weights_path = Path(args.output_dir or "models")
        weights_path.mkdir(parents=True, exist_ok=True)
    else:
        # Create a new run directory
        from src.runs import new_run
        run_dir = new_run(args.name, cfg)
        weights_path = run_dir / "weights"
    
    # Split
    train_split = args.split or "train"
    
    print("=" * 72)
    print("  TRAINING PIPELINE -- PINNA LANDMARK EXTRACTION")
    print("=" * 72)
    print(f"Config:              {args.config}")
    print(f"Config hash:         {cfg.get('_config_hash', 'N/A')}")
    if run_dir:
        print(f"Run directory:       {run_dir}")
    print(f"Weights directory:   {weights_path}")
    print(f"Mesh directory:      {mesh_path}")
    print(f"Landmark directory:  {landmarks_path}")
    if args.include_val:
        print(f"Training split:      train + val (Combined)")
    else:
        print(f"Training split:      {train_split}")
    print(f"SSM components:      {model_cfg.get('n_ssm_components', 30)}")
    print(f"KNN neighbors:       {model_cfg.get('k_neighbors', 7)}")
    print(f"Blend weight (alpha): {model_cfg.get('blend_alpha', 0.6)}")
    print(f"GBR estimators:      {model_cfg.get('gbr_n_estimators', 50)}")
    print(f"Seed:                {seed}")
    print("=" * 72)
    
    # Loud guard: Prevent training on test partition
    if train_split == "test" and not args.include_val:
        print("\n" + "!" * 72)
        print("  CRITICAL ERROR: Training on the 'test' partition is strictly forbidden!")
        print("  To protect evaluation validity, training must never touch test subjects.")
        print("!" * 72 + "\n")
        sys.exit(1)
        
    if not mesh_path.exists() or not landmarks_path.exists():
        print(f"Error: Dataset directories do not exist. Please check the paths.")
        sys.exit(1)
        
    t_start = time.time()
    
    # ── Load splits for leakage assertion ────────────────────────────────────
    train_pids_split, val_pids_split, test_pids_split = load_splits(mesh_dir=mesh_path)
    
    # 1. Load dataset
    print("\n[1/3] Loading dataset...")
    if args.include_val:
        dataset_train = Dataset(mesh_dir=str(mesh_path), landmarks_dir=str(landmarks_path), split="train")
        dataset_val = Dataset(mesh_dir=str(mesh_path), landmarks_dir=str(landmarks_path), split="val")
        dataset = Dataset(mesh_dir=str(mesh_path), landmarks_dir=str(landmarks_path))
        dataset.subject_ids = sorted(list(set(dataset_train.subject_ids) | set(dataset_val.subject_ids)))
    else:
        dataset = Dataset(mesh_dir=str(mesh_path), landmarks_dir=str(landmarks_path), split=train_split)
        
    num_subjects = len(dataset)
    print(f"  Loaded {num_subjects} subjects successfully.")
    
    # Assert train does not overlap with val or test
    actual_pids = set(dataset.subject_ids)
    val_test = set(val_pids_split) | set(test_pids_split)
    leakage = actual_pids & val_test
    if leakage and not args.include_val:
        print("\n" + "!" * 72)
        print(f"  CRITICAL: Train set overlaps with val/test! Leaked PIDs: {sorted(leakage)[:10]}")
        print("!" * 72 + "\n")
        sys.exit(2)
    
    # Format landmarks dictionary
    all_landmarks = {}
    for idx in range(num_subjects):
        pid = dataset.get_identifier(idx)
        _, lm_l, lm_r = dataset[idx]
        all_landmarks[pid] = {"left": lm_l, "right": lm_r}
        
    # 2. Fit Ear Detector
    print("\n[2/3] Fitting Ear Detector (Landmark-Free Segmentation)...")
    detector = EarDetector(
        curvature_radius=detector_cfg.get("curvature_radius", 3.0),
        curvature_threshold_factor=detector_cfg.get("curvature_threshold_factor", 2.5),
        ear_margin=detector_cfg.get("ear_margin", 20.0),
        y_percentile_lateral=detector_cfg.get("y_percentile_lateral", 90),
        n_curvature_samples=detector_cfg.get("n_curvature_samples", 10000),
    )
    detector.fit(all_landmarks, n_mesh_samples=detector_cfg.get("n_mesh_samples", 30))
    detector_file = weights_path / "ear_detector.pkl"
    detector.save(detector_file)
    
    # 3. Fit Landmark Predictor (SSM + GBR + KNN)
    print("\n[3/3] Fitting Landmark Predictor (SSM + GBR + KNN)...")
    predictor = LandmarkPredictor(
        n_ssm_components=model_cfg.get("n_ssm_components", 30),
        k_neighbors=model_cfg.get("k_neighbors", 7),
        feature_radius=model_cfg.get("feature_radius", 5.0),
        blend_alpha=model_cfg.get("blend_alpha", 0.6),
    )
    predictor.fit(all_landmarks, train_with_regressors=model_cfg.get("use_regressors", True))
    predictor_file = weights_path / "landmark_predictor.pkl"
    predictor.save(predictor_file)
    
    # ── Leakage guard: verify shape bank contains ONLY train PIDs ────────────
    shape_bank_pids = set(predictor.pids)
    if not args.include_val:
        non_train = shape_bank_pids - set(train_pids_split)
        if non_train:
            print("\n" + "!" * 72)
            print(f"  CRITICAL: Shape bank contains non-train PIDs: {sorted(non_train)[:10]}")
            print("  The KNN blending would leak val/test data at inference!")
            print("!" * 72 + "\n")
            sys.exit(3)
        print(f"  Shape bank leakage guard: PASSED ({len(shape_bank_pids)} PIDs, all in train split)")
    
    elapsed = time.time() - t_start
    
    # 4. Save JSON sidecar (stdlib-only, for estimator inference)
    sidecar = {
        "config_hash": cfg.get("_config_hash", ""),
        "name": args.name,
        "trained_at": datetime.now().isoformat(),
        "train_split": "train+val" if args.include_val else train_split,
        "train_pids": sorted(all_landmarks.keys()),
        "num_subjects": num_subjects,
        "seed": seed,
        "model": {
            "n_ssm_components": model_cfg.get("n_ssm_components", 30),
            "k_neighbors": model_cfg.get("k_neighbors", 7),
            "blend_alpha": model_cfg.get("blend_alpha", 0.6),
        },
        "training_time_s": round(elapsed, 1),
    }
    sidecar_file = weights_path / "model_info.json"
    with open(sidecar_file, "w") as f:
        json.dump(sidecar, f, indent=2)
    
    # 5. Write metadata if we have a run directory
    if run_dir:
        from src.runs import write_metadata
        write_metadata(
            run_dir, name=args.name, config=cfg,
            train_pids=sorted(all_landmarks.keys()),
            val_pids=val_pids_split,
            test_pids=test_pids_split,
            durations={"train_s": round(elapsed, 1)},
        )
    
    print(f"\nTraining pipeline completed successfully in {elapsed:.1f}s.")
    if run_dir:
        print(f"Run directory:       {run_dir.resolve()}")
        print(f"\nTo evaluate, run:")
        print(f"  python evaluate.py --run \"{run_dir}\"")
    else:
        print(f"Checkpoints saved to: {weights_path.resolve()}")
        print(f"\nTo evaluate, run:")
        print(f"  python evaluate.py --models-dir \"{weights_path}\"")
    print("=" * 72)


if __name__ == "__main__":
    main()
