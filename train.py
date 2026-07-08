"""
============================================================================
TRAINING PIPELINE — Huawei Tech Arena 2026 Pinna Landmark Extraction
============================================================================

Trains the EarDetector and LandmarkPredictor modules on the training dataset
and serializes the models into a versioned experiment directory.

Each run creates:
    experiments/<name>_<YYYYMMDD_HHMMSS>/
        models/          <- ear_detector.pkl, landmark_predictor.pkl
        config.json      <- full hyperparameter snapshot

Usage:
    python train.py --experiment-name baseline_ssm
    python train.py --experiment-name deep_gbr --n-components 50 --k-neighbors 10
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys
import time

# Ensure core package is in the Python search path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.dataset import Dataset
from src.ear_detector import EarDetector
from src.predictor import LandmarkPredictor


def parse_args():
    parser = argparse.ArgumentParser(description="Train Pinna Landmark Detector and Predictor")
    
    # Experiment naming
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="baseline",
        help="Name for this experiment run (folder: experiments/<name>_<timestamp>)"
    )
    parser.add_argument(
        "--experiments-root",
        type=str,
        default="experiments",
        help="Root directory for all experiments"
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
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="The dataset partition split to train on (default: 'train')"
    )
    parser.add_argument(
        "--include-val",
        action="store_true",
        help="Combine training and validation splits for final model training"
    )
    
    # Model hyperparameters
    parser.add_argument(
        "--n-components",
        type=int,
        default=30,
        help="Number of Statistical Shape Model PCA components"
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=7,
        help="Number of neighbors for KNN shape blending"
    )
    parser.add_argument(
        "--blend-alpha",
        type=float,
        default=0.6,
        help="Weight for SSM+GBR predictions (residual weight)"
    )
    parser.add_argument(
        "--n-mesh-samples",
        type=int,
        default=30,
        help="Number of mesh samples to compile the mean ear template"
    )
    
    return parser.parse_args()


def create_experiment_dir(args):
    """Create a timestamped experiment directory and save config."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{args.experiment_name}_{timestamp}"
    exp_dir = Path(args.experiments_root) / exp_name
    
    models_dir = exp_dir / "models"
    results_dir = exp_dir / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full config snapshot
    config = {
        "experiment_name": args.experiment_name,
        "timestamp": timestamp,
        "split": args.split,
        "include_val": args.include_val,
        "n_components": args.n_components,
        "k_neighbors": args.k_neighbors,
        "blend_alpha": args.blend_alpha,
        "n_mesh_samples": args.n_mesh_samples,
        "mesh_dir": args.mesh_dir,
        "landmarks_dir": args.landmarks_dir,
    }
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    return exp_dir, models_dir, results_dir


def main():
    args = parse_args()
    
    mesh_path = Path(args.mesh_dir)
    landmarks_path = Path(args.landmarks_dir)
    
    # Create versioned experiment directory
    exp_dir, models_path, results_path = create_experiment_dir(args)
    
    print("=" * 72)
    print("  TRAINING PIPELINE -- PINNA LANDMARK EXTRACTION")
    print("=" * 72)
    print(f"Experiment:          {exp_dir}")
    print(f"Mesh directory:      {mesh_path}")
    print(f"Landmark directory:  {landmarks_path}")
    if args.include_val:
        print(f"Training split:      train + val (Combined)")
    else:
        print(f"Training split:      {args.split}")
    print(f"SSM components:      {args.n_components}")
    print(f"KNN neighbors:       {args.k_neighbors}")
    print(f"Blend weight (alpha): {args.blend_alpha}")
    print("=" * 72)
    
    # Loud guard: Prevent training on test partition to protect evaluation validity
    if args.split == "test" and not args.include_val:
        print("\n" + "!" * 72)
        print("  CRITICAL ERROR: Training on the 'test' partition is strictly forbidden!")
        print("  To protect evaluation validity, training must never touch test subjects.")
        print("!" * 72 + "\n")
        sys.exit(1)
        
    if not mesh_path.exists() or not landmarks_path.exists():
        print(f"Error: Dataset directories do not exist. Please check the paths.")
        sys.exit(1)
        
    t_start = time.time()
    
    # 1. Load dataset using the official Dataset class with split configurations
    print("\n[1/3] Loading dataset...")
    if args.include_val:
        # Load train and validation splits and merge their PIDs
        dataset_train = Dataset(mesh_dir=str(mesh_path), landmarks_dir=str(landmarks_path), split="train")
        dataset_val = Dataset(mesh_dir=str(mesh_path), landmarks_dir=str(landmarks_path), split="val")
        
        dataset = Dataset(mesh_dir=str(mesh_path), landmarks_dir=str(landmarks_path))
        dataset.subject_ids = sorted(list(set(dataset_train.subject_ids) | set(dataset_val.subject_ids)))
    else:
        dataset = Dataset(mesh_dir=str(mesh_path), landmarks_dir=str(landmarks_path), split=args.split)
        
    num_subjects = len(dataset)
    print(f"  Loaded {num_subjects} subjects successfully.")
    
    # Format landmarks dictionary for fitting
    all_landmarks = {}
    for idx in range(num_subjects):
        pid = dataset.get_identifier(idx)
        _, lm_l, lm_r = dataset[idx]
        all_landmarks[pid] = {
            "left": lm_l,
            "right": lm_r
        }
        
    # 2. Fit Ear Detector
    print("\n[2/3] Fitting Ear Detector (Landmark-Free Segmentation)...")
    detector = EarDetector()
    detector.fit(all_landmarks, n_mesh_samples=args.n_mesh_samples)
    detector_file = models_path / "ear_detector.pkl"
    detector.save(detector_file)
    
    # 3. Fit Landmark Predictor (SSM + GBR + KNN)
    print("\n[3/3] Fitting Landmark Predictor (SSM + GBR + KNN)...")
    predictor = LandmarkPredictor(
        n_ssm_components=args.n_components,
        k_neighbors=args.k_neighbors,
        blend_alpha=args.blend_alpha
    )
    predictor.fit(all_landmarks, train_with_regressors=True)
    predictor_file = models_path / "landmark_predictor.pkl"
    predictor.save(predictor_file)
    
    elapsed = time.time() - t_start
    
    # Save training summary to experiment directory
    summary = {
        "training_time_s": round(elapsed, 1),
        "num_subjects": num_subjects,
        "ssm_variance_explained": getattr(predictor, "variance_explained_", None),
    }
    with open(exp_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nTraining pipeline completed successfully in {elapsed:.1f}s.")
    print(f"Experiment saved to: {exp_dir.resolve()}")
    print("=" * 72)
    print(f"\nTo evaluate this experiment, run:")
    print(f"  python evaluate.py --experiment-dir \"{exp_dir}\"")
    print("=" * 72)


if __name__ == "__main__":
    main()
