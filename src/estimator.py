"""
============================================================================
LANDMARK EXTRACTOR — Huawei Tech Arena 2026 Submission Class
============================================================================

Official submission interface. DO NOT add non-stdlib/non-stack imports.
Runtime imports: numpy, scipy, trimesh, scikit-learn, json (stdlib).
NO yaml, NO config system at top level.

Weight resolution order:
    1. MODEL_PATH environment variable
    2. models/submission symlink/junction
    3. models/best symlink/junction
    4. models/submission.txt pointer file
    5. models/best.txt pointer file
    6. models/ directory (legacy fallback)
    7. Clear error
"""

import json
import os
from typing import Tuple
from pathlib import Path

import numpy as np
from trimesh import Trimesh


class LandmarkExtractor:
    """
    Landmark extractor implementation for automatic evaluation on the challenge platform.
    Integrates our EarDetector and LandmarkPredictor.
    """

    def __init__(self, detector_path: str = None, predictor_path: str = None):
        """
        Initialize the landmark extractor.
        
        Resolves weight paths using the following priority:
        1. Explicit arguments (detector_path, predictor_path)
        2. MODEL_PATH environment variable
        3. models/submission symlink/junction
        4. models/best symlink/junction  
        5. models/submission.txt or models/best.txt pointer files
        6. models/ directory (legacy fallback)
        """
        src_dir = Path(__file__).resolve().parent
        self._root_dir = src_dir.parent
        
        if detector_path and predictor_path:
            self.detector_path = Path(detector_path)
            self.predictor_path = Path(predictor_path)
        else:
            weights_dir = self._resolve_weights_dir()
            self.detector_path = weights_dir / "ear_detector.pkl"
            self.predictor_path = weights_dir / "landmark_predictor.pkl"
        
        self.detector = None
        self.predictor = None
        self._model_info = None
        
        # Attempt to load checkpoints immediately if they exist
        self._load_models(silent=True)

    def _resolve_weights_dir(self) -> Path:
        """
        Resolve the weights directory using the priority chain.
        Handles real symlinks, Windows junctions, and .txt pointer files.
        """
        models_dir = self._root_dir / "models"
        
        # 1. MODEL_PATH environment variable
        env_path = os.environ.get("MODEL_PATH")
        if env_path:
            p = Path(env_path)
            if p.is_file():
                # Points to a specific .pkl file — use its parent
                return p.parent
            elif p.is_dir():
                return p
        
        # 2-3. Check symlinks/junctions: submission first, then best
        for pointer_name in ("submission", "best"):
            pointer = models_dir / pointer_name
            # is_dir() follows symlinks and junctions on Windows
            if pointer.is_dir():
                return pointer.resolve()
        
        # 4-5. Check .txt pointer files
        for pointer_name in ("submission", "best"):
            txt_file = models_dir / f"{pointer_name}.txt"
            if txt_file.exists():
                target_str = txt_file.read_text().strip()
                target = Path(target_str)
                if not target.is_absolute():
                    target = self._root_dir / target
                if target.is_dir():
                    return target
        
        # 6. Legacy: models/ directory itself
        if models_dir.is_dir():
            detector_check = models_dir / "ear_detector.pkl"
            if detector_check.exists():
                return models_dir
        
        # 7. Clear error
        raise FileNotFoundError(
            "No model weights found. Resolution tried:\n"
            "  1. MODEL_PATH env var (not set)\n"
            "  2. models/submission (not found)\n"
            "  3. models/best (not found)\n"
            "  4. models/submission.txt (not found)\n"
            "  5. models/best.txt (not found)\n"
            "  6. models/ directory (no weights)\n"
            "\nRun: python scripts/promote.py --run <run_id> --as submission"
        )

    def _load_models(self, silent: bool = False):
        """Load EarDetector and LandmarkPredictor states from disk."""
        if self.detector is not None and self.predictor is not None:
            return True
            
        if not self.detector_path.exists():
            if not silent:
                raise FileNotFoundError(f"EarDetector checkpoint not found at: {self.detector_path}")
            return False
            
        if not self.predictor_path.exists():
            if not silent:
                raise FileNotFoundError(f"LandmarkPredictor checkpoint not found at: {self.predictor_path}")
            return False
            
        from src.ear_detector import EarDetector
        from src.predictor import LandmarkPredictor
        
        self.detector = EarDetector()
        self.detector.load(self.detector_path)
        
        self.predictor = LandmarkPredictor()
        self.predictor.load(self.predictor_path)
        
        # Load JSON sidecar if present (stdlib only)
        sidecar_path = self.detector_path.parent / "model_info.json"
        if sidecar_path.exists():
            with open(sidecar_path) as f:
                self._model_info = json.load(f)
        
        return True

    def extract(self, mesh: Trimesh) -> Tuple[np.ndarray, np.ndarray]:
        """
        Method to extract left and right ear landmarks from a 3D mesh.
        Both output arrays are of shape (85, 3), containing the landmark
        coordinates in the correct order.
        
        This method runs entirely landmark-free at test time.
        Refinement flags are loaded from the JSON sidecar if present.
        """
        # Ensure models are loaded
        if not self._load_models(silent=False):
            raise RuntimeError("Models could not be loaded. Please run training first to generate checkpoints.")
        
        # Read refinement flags from the JSON sidecar (no yaml dependency)
        refine = {}
        if self._model_info and "refine" in self._model_info:
            refine = self._model_info["refine"]
            
        # Predict left and right ears
        pred_left = self.predictor.predict(mesh, side="left", ear_detector=self.detector, refine=refine)
        pred_right = self.predictor.predict(mesh, side="right", ear_detector=self.detector, refine=refine)
        
        return pred_left, pred_right
