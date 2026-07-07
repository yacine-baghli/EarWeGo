"""
Geometric processing utilities for ear meshes and landmarks.
Includes alignment (Procrustes, ICP), curvature computation, and feature extraction.
"""

import numpy as np
from scipy.spatial import KDTree, cKDTree
from scipy.spatial.transform import Rotation
from scipy.linalg import orthogonal_procrustes
from typing import Optional


# ─── Procrustes Alignment ────────────────────────────────────────────────────

def procrustes_align(
    source: np.ndarray,
    target: np.ndarray,
    allow_scale: bool = False,
) -> tuple[np.ndarray, dict]:
    """
    Align source points to target points using Procrustes analysis.
    
    Args:
        source: (N, 3) source points to transform.
        target: (N, 3) target points (reference).
        allow_scale: If True, also optimize scale.
    
    Returns:
        aligned: (N, 3) aligned source points.
        transform: dict with 'R' (rotation), 't' (translation), 's' (scale).
    """
    # Center both point sets
    src_mean = source.mean(axis=0)
    tgt_mean = target.mean(axis=0)
    
    src_centered = source - src_mean
    tgt_centered = target - tgt_mean
    
    # Find optimal rotation
    R, _ = orthogonal_procrustes(src_centered, tgt_centered)
    
    # Compute scale if allowed
    if allow_scale:
        src_rotated = src_centered @ R
        s = np.sum(tgt_centered * src_rotated) / np.sum(src_rotated * src_rotated)
    else:
        s = 1.0
    
    # Apply transformation
    aligned = s * (src_centered @ R) + tgt_mean
    
    transform = {
        "R": R,
        "t_src": src_mean,
        "t_tgt": tgt_mean,
        "s": s,
    }
    
    return aligned, transform


def apply_procrustes_transform(
    points: np.ndarray,
    transform: dict,
) -> np.ndarray:
    """Apply a previously computed Procrustes transform to new points."""
    centered = points - transform["t_src"]
    return transform["s"] * (centered @ transform["R"]) + transform["t_tgt"]


# ─── Iterative Closest Point (ICP) ──────────────────────────────────────────

def icp(
    source: np.ndarray,
    target: np.ndarray,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    max_correspondence_dist: float = float("inf"),
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Simple point-to-point ICP alignment.
    
    Args:
        source: (N, 3) source points.
        target: (M, 3) target points (reference).
        max_iterations: Maximum number of ICP iterations.
        tolerance: Convergence threshold on mean distance change.
        max_correspondence_dist: Reject correspondences farther than this.
    
    Returns:
        aligned: (N, 3) aligned source points.
        R_total: (3, 3) total rotation matrix.
        t_total: (3,) total translation vector.
    """
    src = source.copy()
    target_tree = cKDTree(target)
    
    R_total = np.eye(3)
    t_total = np.zeros(3)
    prev_error = float("inf")
    
    for iteration in range(max_iterations):
        # Find closest points in target for each source point
        distances, indices = target_tree.query(src)
        
        # Filter by max distance
        valid = distances < max_correspondence_dist
        if valid.sum() < 4:
            break
        
        src_valid = src[valid]
        tgt_valid = target[indices[valid]]
        
        # Compute optimal rigid transform
        src_mean = src_valid.mean(axis=0)
        tgt_mean = tgt_valid.mean(axis=0)
        
        src_centered = src_valid - src_mean
        tgt_centered = tgt_valid - tgt_mean
        
        H = src_centered.T @ tgt_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Handle reflection
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        t = tgt_mean - R @ src_mean
        
        # Apply transform
        src = (R @ src.T).T + t
        
        # Accumulate transforms
        R_total = R @ R_total
        t_total = R @ t_total + t
        
        # Check convergence
        mean_error = distances[valid].mean()
        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
    
    return src, R_total, t_total


# ─── Mirror Landmarks ───────────────────────────────────────────────────────

def mirror_landmarks_y(landmarks: np.ndarray) -> np.ndarray:
    """Mirror landmarks across Y=0 plane (to make right ear comparable to left)."""
    mirrored = landmarks.copy()
    mirrored[:, 1] *= -1
    return mirrored


# ─── Surface Features ───────────────────────────────────────────────────────

def compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute per-vertex normals by averaging adjacent face normals."""
    normals = np.zeros_like(vertices)
    
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    face_normals = np.cross(v1 - v0, v2 - v0)
    # Normalize face normals
    norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    face_normals /= norms
    
    # Accumulate to vertices
    for i in range(3):
        np.add.at(normals, faces[:, i], face_normals)
    
    # Normalize vertex normals
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normals /= norms
    
    return normals


def compute_curvature_estimate(
    vertices: np.ndarray,
    faces: np.ndarray,
    k_neighbors: int = 15,
) -> np.ndarray:
    """
    Estimate mean curvature at each vertex using local PCA.
    
    Returns:
        curvatures: (N,) approximate mean curvature values.
    """
    tree = cKDTree(vertices)
    curvatures = np.zeros(len(vertices))
    
    for i in range(len(vertices)):
        _, idx = tree.query(vertices[i], k=k_neighbors)
        neighbors = vertices[idx]
        
        # Local PCA
        centered = neighbors - neighbors.mean(axis=0)
        cov = centered.T @ centered / len(neighbors)
        eigenvalues = np.linalg.eigvalsh(cov)
        
        # Smallest eigenvalue relative to others indicates curvature
        eigenvalues = np.sort(eigenvalues)
        curvatures[i] = eigenvalues[0] / (eigenvalues.sum() + 1e-10)
    
    return curvatures


def compute_local_features(
    vertices: np.ndarray,
    point: np.ndarray,
    radius: float = 5.0,
    tree: Optional[cKDTree] = None,
) -> np.ndarray:
    """
    Compute local geometric features around a point on the mesh.
    
    Returns:
        Feature vector containing local statistics.
    """
    if tree is None:
        tree = cKDTree(vertices)
    
    idx = tree.query_ball_point(point, radius)
    if len(idx) < 3:
        return np.zeros(12)
    
    local_pts = vertices[idx]
    centered = local_pts - point
    
    # PCA eigenvalues
    cov = centered.T @ centered / len(local_pts)
    eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
    
    # Features: eigenvalues, ratios, density, spread
    features = np.array([
        eigenvalues[0],
        eigenvalues[1],
        eigenvalues[2],
        eigenvalues[1] / (eigenvalues[0] + 1e-10),  # planarity
        eigenvalues[2] / (eigenvalues[0] + 1e-10),  # linearity
        len(idx),  # point density
        np.linalg.norm(centered, axis=1).mean(),  # mean distance
        np.linalg.norm(centered, axis=1).std(),   # distance std
        centered[:, 0].std(),
        centered[:, 1].std(),
        centered[:, 2].std(),
        eigenvalues[2] / (eigenvalues[1] + 1e-10),  # sphericity
    ])
    
    return features


# ─── Landmark Snapping ───────────────────────────────────────────────────────

def snap_to_mesh(
    landmarks: np.ndarray,
    mesh_vertices: np.ndarray,
) -> np.ndarray:
    """Snap landmark positions to nearest mesh vertex."""
    tree = cKDTree(mesh_vertices)
    _, indices = tree.query(landmarks)
    return mesh_vertices[indices]


def snap_to_surface(
    landmarks: np.ndarray,
    mesh,
) -> np.ndarray:
    """Snap landmark positions to nearest point on mesh surface (face projection)."""
    closest_points, _, _ = mesh.nearest.on_surface(landmarks)
    return closest_points


# ─── Statistical Shape Model ────────────────────────────────────────────────

class StatisticalShapeModel:
    """
    PCA-based Statistical Shape Model (SSM) for ear landmarks.
    """
    
    def __init__(self, n_components: int = 20):
        self.n_components = n_components
        self.mean_shape = None
        self.components = None
        self.eigenvalues = None
        self.explained_variance_ratio = None
    
    def fit(self, shapes: np.ndarray):
        """
        Fit SSM to a set of aligned shapes.
        
        Args:
            shapes: (N, 85, 3) array of aligned landmark configurations.
        """
        n_samples = shapes.shape[0]
        flat_shapes = shapes.reshape(n_samples, -1)  # (N, 255)
        
        # Compute mean shape
        self.mean_shape = flat_shapes.mean(axis=0)
        
        # Center the data
        centered = flat_shapes - self.mean_shape
        
        # PCA via SVD
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        
        n_comp = min(self.n_components, len(S))
        self.components = Vt[:n_comp]  # (n_comp, 255)
        self.eigenvalues = (S[:n_comp] ** 2) / (n_samples - 1)
        
        total_var = (S ** 2).sum() / (n_samples - 1)
        self.explained_variance_ratio = self.eigenvalues / total_var
        
        print(f"SSM fitted with {n_comp} components, "
              f"explaining {self.explained_variance_ratio.sum():.1%} of variance")
    
    def project(self, shape: np.ndarray) -> np.ndarray:
        """Project a shape onto the SSM, returning coefficients."""
        flat = shape.flatten() - self.mean_shape
        return flat @ self.components.T
    
    def reconstruct(self, coefficients: np.ndarray) -> np.ndarray:
        """Reconstruct a shape from SSM coefficients."""
        flat = self.mean_shape + coefficients @ self.components
        return flat.reshape(85, 3)
    
    def get_mean_shape(self) -> np.ndarray:
        """Return mean shape as (85, 3)."""
        return self.mean_shape.reshape(85, 3)
    
    def sample(self, n_samples: int = 1, scale: float = 1.0) -> np.ndarray:
        """Sample random shapes from the model."""
        stds = np.sqrt(self.eigenvalues)
        coeffs = np.random.randn(n_samples, self.n_components) * stds * scale
        shapes = []
        for c in coeffs:
            shapes.append(self.reconstruct(c))
        return np.stack(shapes)


# ─── Generalized Procrustes Analysis ─────────────────────────────────────────

def generalized_procrustes(
    shapes: np.ndarray,
    max_iterations: int = 20,
    tolerance: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generalized Procrustes Analysis to align multiple shapes.
    
    Args:
        shapes: (N, K, 3) array of N shapes with K points each.
    
    Returns:
        aligned_shapes: (N, K, 3) aligned shapes.
        mean_shape: (K, 3) consensus mean shape.
    """
    n_shapes = shapes.shape[0]
    aligned = shapes.copy()
    
    # Initialize mean as first shape (centered)
    mean_shape = aligned[0] - aligned[0].mean(axis=0)
    mean_shape /= np.linalg.norm(mean_shape)
    
    for iteration in range(max_iterations):
        # Align each shape to current mean
        for i in range(n_shapes):
            aligned[i], _ = procrustes_align(aligned[i], mean_shape, allow_scale=True)
        
        # Update mean
        new_mean = aligned.mean(axis=0)
        new_mean -= new_mean.mean(axis=0)  # Re-center
        
        # Check convergence
        diff = np.linalg.norm(new_mean - mean_shape)
        mean_shape = new_mean
        
        if diff < tolerance:
            break
    
    return aligned, mean_shape
