"""
============================================================================
TRAINING PIPELINE — Huawei Tech Arena 2026 Pinna Landmark Extraction
============================================================================

Trains the EarDetector and LandmarkPredictor modules on the training dataset
and serializes the models to the 'models/' directory.

Usage:
    python train.py --mesh-dir "path/to/mesh" --landmarks-dir "path/to/landmarks"
"""

import argparse
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
    
    # Output configurations
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory to save trained model files"
    )
    parser.add_argument(
        "--n-mesh-samples",
        type=int,
        default=30,
        help="Number of mesh samples to compile the mean ear template"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    mesh_path = Path(args.mesh_dir)
    landmarks_path = Path(args.landmarks_dir)
    models_path = Path(args.models_dir)
    models_path.mkdir(exist_ok=True)
    
    print("=" * 72)
    print("  TRAINING PIPELINE — PINNA LANDMARK EXTRACTION")
    print("=" * 72)
    print(f"Mesh directory:      {mesh_path}")
    print(f"Landmark directory:  {landmarks_path}")
    print(f"SSM components:      {args.n_components}")
    print(f"KNN neighbors:       {args.k_neighbors}")
    print(f"Blend weight (α):    {args.blend_alpha}")
    print("=" * 72)
    
    if not mesh_path.exists() or not landmarks_path.exists():
        print(f"Error: Dataset directories do not exist. Please check the paths.")
        sys.exit(1)
        
    t_start = time.time()
    
    # 1. Load dataset using the official Dataset class
    print("\n[1/3] Loading dataset...")
    dataset = Dataset(mesh_dir=str(mesh_path), landmarks_dir=str(landmarks_path))
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
    print(f"\nTraining pipeline completed successfully in {elapsed:.1f}s.")
    print(f"Checkpoints saved to: {models_path.resolve()}")
    print("=" * 72)


if __name__ == "__main__":
    main()
