from typing import Tuple
import numpy as np
from pathlib import Path
from trimesh import Trimesh


class LandmarkExtractor:
    """
    Landmark extractor implementation for automatic evaluation on the challenge platform.
    Integrates our EarDetector and LandmarkPredictor.
    """

    def __init__(self, detector_path: str = None, predictor_path: str = None):
        """
        Initialize the landmark extractor.
        Loads pre-trained model checkpoints from the 'models/' directory.
        """
        src_dir = Path(__file__).resolve().parent
        models_dir = src_dir.parent / "models"
        
        self.detector_path = Path(detector_path) if detector_path else models_dir / "ear_detector.pkl"
        self.predictor_path = Path(predictor_path) if predictor_path else models_dir / "landmark_predictor.pkl"
        
        self.detector = None
        self.predictor = None
        
        # Attempt to load checkpoints immediately if they exist
        self._load_models(silent=True)

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
        return True

    def extract(self, mesh: Trimesh) -> Tuple[np.ndarray, np.ndarray]:
        """
        Method to extract left and right ear landmarks from a 3D mesh.
        Both output arrays are of shape (85, 3), containing the landmark
        coordinates in the correct order.
        
        This method runs entirely land-mark free at test time.
        """
        # Ensure models are loaded
        if not self._load_models(silent=False):
            raise RuntimeError("Models could not be loaded. Please run training first to generate checkpoints.")
            
        # Predict left and right ears
        pred_left = self.predictor.predict(mesh, side="left", ear_detector=self.detector)
        pred_right = self.predictor.predict(mesh, side="right", ear_detector=self.detector)
        
        return pred_left, pred_right
