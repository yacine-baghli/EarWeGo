"""
============================================================================
LANDMARK PREDICTOR — Statistical Shape Model + Gradient Boosting Regressor
============================================================================

Predicts pinna landmarks from 3D ear mesh geometry using a hybrid approach:
  1. Coarse alignment via template-based Iterative Closest Point (ICP).
  2. Projection of aligned landmarks onto a statistical shape model (SSM)
     trained via Generalized Procrustes Analysis (GPA) and PCA.
  3. Residual correction using Gradient Boosting Regressors (GBR) trained
     to predict the error between the regularized SSM shape and actual landmarks.
  4. Non-linear blending with local shapes via K-Nearest Neighbors (KNN).
  5. Surface snapping to project final predictions back onto the ear mesh surface.

Author: Antigravity AI Landmark Predictor
"""

import numpy as np
import pickle
from pathlib import Path
from scipy.spatial import cKDTree
from sklearn.ensemble import GradientBoostingRegressor
import trimesh

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import (
    load_all_landmarks, load_participant, get_participant_ids,
    extract_ear_region, NUM_LANDMARKS,
)
from src.geometry import (
    procrustes_align, apply_procrustes_transform,
    mirror_landmarks_y, generalized_procrustes,
    StatisticalShapeModel, icp, robust_icp, snap_to_mesh,
)


class LandmarkPredictor:
    """
    Landmark predictor using Statistical Shape Model (SSM) and
    Gradient Boosting Regressors (GBR) for residual error correction.
    """
    
    def __init__(
        self,
        n_ssm_components: int = 30,
        k_neighbors: int = 7,
        feature_radius: float = 5.0,
        blend_alpha: float = 0.6,
    ):
        """
        Args:
            n_ssm_components: Number of PCA components for the SSM.
            k_neighbors: Number of nearest neighbors to blend for KNN shape refinement.
            feature_radius: Local curvature feature radius (not used in default GBR).
            blend_alpha: Blending weight between SSM+GBR (alpha) and KNN predictions (1-alpha).
        """
        self.n_ssm_components = n_ssm_components
        self.k_neighbors = k_neighbors
        self.feature_radius = feature_radius
        self.blend_alpha = blend_alpha
        
        self.ssm = StatisticalShapeModel(n_components=n_ssm_components)
        self.mean_shape_left = None
        self.mean_shape_right = None
        self.aligned_shapes = None
        self.ssm_coefficients = None
        self.all_landmarks = None
        self.pids = None
        self.regressors = None
        self.fitted = False
    
    def fit(self, all_landmarks: dict = None, train_with_regressors: bool = True):
        """
        Fit the SSM and GBR regressors on training landmarks.
        
        Args:
            all_landmarks: dict mapping pid -> {'left': (85,3), 'right': (85,3)}.
                           If None, loads all landmarks from the dataset.
            train_with_regressors: Whether to train per-landmark GBR models.
        """
        if all_landmarks is None:
            all_landmarks = load_all_landmarks()
        
        self.all_landmarks = all_landmarks
        self.pids = sorted(all_landmarks.keys())
        n = len(self.pids)
        
        # Collect shapes
        left_shapes = np.stack([all_landmarks[pid]["left"] for pid in self.pids])
        right_shapes = np.stack([all_landmarks[pid]["right"] for pid in self.pids])
        
        # Store raw means
        self.mean_shape_left = left_shapes.mean(axis=0)
        self.mean_shape_right = right_shapes.mean(axis=0)
        
        # Mirror right ears for a unified "left-like" SSM shape space
        right_mirrored = right_shapes.copy()
        right_mirrored[:, :, 1] *= -1
        
        all_shapes = np.concatenate([left_shapes, right_mirrored], axis=0)
        
        # Run Generalized Procrustes Analysis
        print("  Running GPA...")
        self.aligned_shapes, _ = generalized_procrustes(all_shapes)
        
        # Fit Statistical Shape Model
        print("  Fitting SSM...")
        self.ssm.fit(self.aligned_shapes)
        
        # Project shapes to ssm space for KNN query neighbors
        self.ssm_coefficients = np.array([
            self.ssm.project(s) for s in self.aligned_shapes
        ])
        
        # Train regressors on shape residuals
        if train_with_regressors:
            print("  Training per-landmark regressors...")
            self._train_regressors()
        
        self.fitted = True
        print("  LandmarkPredictor fitted!")
    
    def _train_regressors(self):
        """
        Train coordinate-wise GBR models to regress the residual errors
        between actual GPA shapes and their reconstructed SSM shapes.
        """
        n = len(self.pids)
        X_data = []
        Y_data = {i: [] for i in range(NUM_LANDMARKS)}
        
        # We augment each training sample with different random perturbations to simulate alignment noise
        n_augmentations = 5
        np.random.seed(42)
        from scipy.spatial.transform import Rotation
        
        for idx in range(2 * n):
            actual = self.aligned_shapes[idx]
            
            # Clean sample
            coeff = self.ssm_coefficients[idx]
            reconstructed = self.ssm.reconstruct(coeff)
            X_data.append(coeff)
            for i in range(NUM_LANDMARKS):
                Y_data[i].append(actual[i] - reconstructed[i])
                
            # Perturbed samples
            for _ in range(n_augmentations):
                # Small random translation
                t = np.random.normal(0, 1.5, size=3)  # std = 1.5mm
                # Small random rotation
                angles = np.random.normal(0, 1.0, size=3)  # std = 1.0 degree
                R = Rotation.from_euler('xyz', angles, degrees=True).as_matrix()
                # Small random scale
                s = np.random.normal(1.0, 0.02)  # std = 2%
                
                # Apply rigid perturbation
                perturbed = s * (actual @ R) + t
                
                # Align perturbed back to SSM space (simulating test-time ICP alignment)
                aligned_p, _ = procrustes_align(perturbed, self.ssm.get_mean_shape(), allow_scale=True)
                
                # Project and reconstruct
                coeff_p = self.ssm.project(aligned_p)
                reconstructed_p = self.ssm.reconstruct(coeff_p)
                
                X_data.append(coeff_p)
                for i in range(NUM_LANDMARKS):
                    # Target residual is the difference between the true GPA coordinate and the perturbed SSM reconstruction
                    Y_data[i].append(actual[i] - reconstructed_p[i])
        
        X_data = np.array(X_data)
        
        # Fit 85 * 3 = 255 separate GBR regressors
        self.regressors = {}
        for i in range(NUM_LANDMARKS):
            y = np.array(Y_data[i])  # (2N, 3)
            regs = []
            for axis in range(3):
                reg = GradientBoostingRegressor(
                    n_estimators=50,
                    max_depth=3,
                    learning_rate=0.1,
                    subsample=0.8,
                    random_state=42,
                )
                reg.fit(X_data, y[:, axis])
                regs.append(reg)
            self.regressors[i] = regs
            
    def predict(
        self,
        mesh: trimesh.Trimesh = None,
        side: str = "left",
        known_landmarks: np.ndarray = None,
        ear_detector = None,
    ) -> np.ndarray:
        """
        Predict landmarks for a specific ear mesh.
        
        Args:
            mesh: Full head trimesh.Trimesh.
            side: 'left' or 'right'.
            known_landmarks: Optional starting landmarks (skips coarse alignment).
            ear_detector: Optional EarDetector object for automatic localization.
        
        Returns:
            (85, 3) array of predicted landmark coordinates.
        """
        if not self.fitted:
            raise RuntimeError("LandmarkPredictor not fitted!")
            
        # Step 1: Initialize coarse landmark location
        if known_landmarks is not None:
            initial = known_landmarks.copy()
        elif mesh is not None:
            template = self.mean_shape_left if side == "left" else self.mean_shape_right
            mesh_verts = np.array(mesh.vertices)
            
            # Segment ear region
            if ear_detector is not None:
                ear_verts, _, _ = ear_detector.detect(mesh, side=side)
            else:
                # Default: use mean shape bounding box as guide
                template_min = template.min(axis=0) - 20.0
                template_max = template.max(axis=0) + 20.0
                ear_mask = np.all(
                    (mesh_verts >= template_min) & (mesh_verts <= template_max),
                    axis=1
                )
                
                if ear_mask.sum() > 100:
                    ear_verts = mesh_verts[ear_mask]
                else:
                    ear_mask = mesh_verts[:, 1] > 0 if side == "left" else mesh_verts[:, 1] < 0
                    ear_verts = mesh_verts[ear_mask]
            
            # Align template using bounded distance ICP
            try:
                initial, _, _ = icp(
                    template, ear_verts,
                    max_iterations=100,
                    tolerance=1e-8,
                    max_correspondence_dist=15.0,
                )
            except Exception:
                initial = template.copy()
        else:
            raise ValueError("Must provide either mesh or known_landmarks")
            
        # Step 2: Project alignment to SSM space
        if side == "right":
            initial_for_ssm = mirror_landmarks_y(initial)
        else:
            initial_for_ssm = initial
            
        aligned, transform = procrustes_align(
            initial_for_ssm, self.ssm.get_mean_shape(), allow_scale=True
        )
        coeff = self.ssm.project(aligned)
        
        # Step 3: SSM reconstruction with GBR residual correction
        reconstructed = self.ssm.reconstruct(coeff)
        if self.regressors is not None:
            for i in range(NUM_LANDMARKS):
                for axis in range(3):
                    residual = self.regressors[i][axis].predict(coeff.reshape(1, -1))[0]
                    reconstructed[i, axis] += residual
                    
        # Step 4: Transform back to original coordinate space
        inv_transform = {
            "R": transform["R"].T,
            "t_src": transform["t_tgt"],
            "t_tgt": transform["t_src"],
            "s": 1.0 / transform["s"],
        }
        result = apply_procrustes_transform(reconstructed, inv_transform)
        if side == "right":
            result = mirror_landmarks_y(result)
            
        # Step 5: KNN blending with local shapes
        knn_result = self._knn_predict(coeff, side)
        result = self.blend_alpha * result + (1.0 - self.blend_alpha) * knn_result
        
        # Step 6: Snap to nearest mesh vertices using KDTree
        if mesh is not None:
            try:
                result = snap_to_mesh(result, np.array(mesh.vertices))
            except Exception:
                pass
                
        return result
        
    def _knn_predict(self, query_coeff: np.ndarray, side: str) -> np.ndarray:
        """Helper to compute inverse-distance-weighted KNN landmark blend."""
        distances = np.linalg.norm(self.ssm_coefficients - query_coeff, axis=1)
        knn_idx = np.argsort(distances)[:self.k_neighbors]
        
        knn_dists = distances[knn_idx]
        weights = 1.0 / (knn_dists + 1e-10)
        weights /= weights.sum()
        
        n = len(self.pids)
        predicted = np.zeros((NUM_LANDMARKS, 3))
        
        for i, idx in enumerate(knn_idx):
            if idx < n:
                pid = self.pids[idx]
            else:
                pid = self.pids[idx - n]
                
            shape = self.all_landmarks[pid][side].copy()
            predicted += weights[i] * shape
            
        return predicted
        
    def save(self, path: str | Path):
        """Serialize and save predictor states using pickle."""
        with open(path, "wb") as f:
            pickle.dump({
                "n_ssm_components": self.n_ssm_components,
                "k_neighbors": self.k_neighbors,
                "feature_radius": self.feature_radius,
                "blend_alpha": self.blend_alpha,
                "ssm": self.ssm,
                "mean_shape_left": self.mean_shape_left,
                "mean_shape_right": self.mean_shape_right,
                "aligned_shapes": self.aligned_shapes,
                "ssm_coefficients": self.ssm_coefficients,
                "all_landmarks": self.all_landmarks,
                "pids": self.pids,
                "regressors": self.regressors,
            }, f)
        print(f"LandmarkPredictor saved to {path}")
        
    def load(self, path: str | Path):
        """De-serialize and restore predictor state from pickle file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        for key, val in data.items():
            setattr(self, key, val)
        self.fitted = True
        print(f"LandmarkPredictor loaded from {path}")
