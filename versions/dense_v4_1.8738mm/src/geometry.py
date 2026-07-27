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


def robust_icp(
    source: np.ndarray,
    target: np.ndarray,
    max_iterations: int = 100,
    tolerance: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Robust ICP with multi-start translation offsets to prevent local minima drift.
    
    Args:
        source: (N, 3) template landmarks.
        target: (M, 3) segmented mesh vertices.
        max_iterations: Max full ICP iterations.
        tolerance: Convergence threshold.
        
    Returns:
        aligned: (N, 3) aligned landmarks.
        R: rotation matrix.
        t: translation vector.
    """
    # Initialize centroids
    src_mean = source.mean(axis=0)
    tgt_mean = target.mean(axis=0)
    
    # Pre-align centroids
    centroid_aligned_source = source + (tgt_mean - src_mean)
    
    # Offsets in X, Y, Z to test (mm)
    offsets = [
        np.array([0.0, 0.0, 0.0]),
        np.array([15.0, 0.0, 0.0]),
        np.array([-15.0, 0.0, 0.0]),
        np.array([0.0, 10.0, 0.0]),
        np.array([0.0, -10.0, 0.0]),
        np.array([0.0, 0.0, 15.0]),
        np.array([0.0, 0.0, -15.0]),
    ]
    
    best_error = float("inf")
    best_start_src = centroid_aligned_source
    
    # Find the best translation offset using a quick ICP run
    target_tree = cKDTree(target)
    for offset in offsets:
        shifted_src = centroid_aligned_source + offset
        try:
            # Quick 15-iteration ICP run
            src_temp, _, _ = icp(
                shifted_src, target, max_iterations=15, tolerance=1e-4
            )
            dists, _ = target_tree.query(src_temp)
            mean_err = dists.mean()
            
            if mean_err < best_error:
                best_error = mean_err
                best_start_src = shifted_src
        except Exception:
            pass
            
    # Run full ICP from the best starting position
    return icp(
        best_start_src, target, max_iterations=max_iterations, tolerance=tolerance
    )


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


# ─── Contour Spline Resampling ────────────────────────────────────────────────

def resample_contours(landmarks: np.ndarray) -> np.ndarray:
    """
    Enforce anatomical contour spacing constraints via cubic spline resampling.
    63 of the 85 landmarks are intermediate points along 4 contours:
      1. Outer Helix (0-24, anchors at 0, 6, 22, 24)
      2. Concha Outline (25-54, anchors at 25, 33, 42, 46, 50, 54)
      3. Inner Helix (55-74, anchors at 55, 64; extrapolated to 74)
      4. Superior Antihelix (75-84, anchors at 75, 84)
      
    Args:
        landmarks: (85, 3) array of predicted landmarks.
        
    Returns:
        resampled: (85, 3) array with contour constraints strictly enforced.
    """
    from scipy.interpolate import CubicSpline
    
    resampled = landmarks.copy()
    
    # ─── Helper function to resample a single segment of a spline ───
    def resample_spline_segment(spline, t_start, t_end, num_points):
        t_high = np.linspace(t_start, t_end, 500)
        pts_high = spline(t_high)
        
        # Cumulative arc lengths
        dists = np.linalg.norm(np.diff(pts_high, axis=0), axis=1)
        arc_lengths = np.concatenate(([0], np.cumsum(dists)))
        total_len = arc_lengths[-1]
        
        if total_len < 1e-6:
            return np.linspace(spline(t_start), spline(t_end), num_points)
            
        # Target equally spaced arc lengths
        target_lens = np.linspace(0, total_len, num_points)
        
        # Interpolate parameter t as function of arc length
        t_sampled = np.interp(target_lens, arc_lengths, t_high)
        return spline(t_sampled)

    # ─── 1. Outer Helix (0-24) ───
    # Fit spline to the raw predicted outer helix points to capture shape
    idx_helix = np.arange(25)
    helix_pts = landmarks[idx_helix]
    t_helix = np.concatenate(([0], np.cumsum(np.linalg.norm(np.diff(helix_pts, axis=0), axis=1))))
    spline_helix = CubicSpline(t_helix, helix_pts, bc_type='natural')
    
    # Resample each segment between anchors: [0, 6], [6, 22], [22, 24]
    anchors_helix = [0, 6, 22, 24]
    for start_idx, end_idx in zip(anchors_helix[:-1], anchors_helix[1:]):
        n_pts = end_idx - start_idx + 1
        resampled[start_idx:end_idx+1] = resample_spline_segment(
            spline_helix, t_helix[start_idx], t_helix[end_idx], n_pts
        )

    # ─── 2. Concha Outline (25-54) ───
    idx_concha = np.arange(25, 55)
    concha_pts = landmarks[idx_concha]
    t_concha = np.concatenate(([0], np.cumsum(np.linalg.norm(np.diff(concha_pts, axis=0), axis=1))))
    spline_concha = CubicSpline(t_concha, concha_pts, bc_type='natural')
    
    # Anchors: 25, 33, 42, 46, 50, 54
    anchors_concha = [25, 33, 42, 46, 50, 54]
    for start_idx, end_idx in zip(anchors_concha[:-1], anchors_concha[1:]):
        n_pts = end_idx - start_idx + 1
        # Convert absolute indices to concha-relative indices for parameter lookup
        rel_start = start_idx - 25
        rel_end = end_idx - 25
        resampled[start_idx:end_idx+1] = resample_spline_segment(
            spline_concha, t_concha[rel_start], t_concha[rel_end], n_pts
        )

    # ─── 3. Inner Helix (55-74) ───
    # Spacing between 64 and 74 must equal the spacing of the segment 55-64
    idx_inner = np.arange(55, 65)  # Points 55 to 64
    inner_pts = landmarks[idx_inner]
    t_inner = np.concatenate(([0], np.cumsum(np.linalg.norm(np.diff(inner_pts, axis=0), axis=1))))
    spline_inner = CubicSpline(t_inner, inner_pts, bc_type='natural')
    
    # Resample the active segment 55-64 (10 points)
    resampled_55_64 = resample_spline_segment(spline_inner, t_inner[0], t_inner[-1], 10)
    resampled[55:65] = resampled_55_64
    
    # Spacing interval
    dists_55_64 = np.linalg.norm(np.diff(resampled_55_64, axis=0), axis=1)
    step_size = dists_55_64.mean()
    
    # Extrapolate for remaining 10 points (65 to 74)
    # We sample beyond t_inner[-1] at increments that match step_size along the curve
    t_extrapolated = []
    t_curr = t_inner[-1]
    
    # Determine local derivative at the end of the spline to estimate parameter step
    deriv = spline_inner.derivative()(t_curr)
    deriv_norm = np.linalg.norm(deriv)
    t_step = step_size / (deriv_norm + 1e-10)
    
    for _ in range(10):
        t_curr += t_step
        t_extrapolated.append(t_curr)
        # Update t_step dynamically based on the local derivative at the new point
        new_deriv = spline_inner.derivative()(t_curr)
        t_step = step_size / (np.linalg.norm(new_deriv) + 1e-10)
        
    resampled[65:75] = spline_inner(t_extrapolated)

    # ─── 4. Superior Antihelix (75-84) ───
    idx_antihelix = np.arange(75, 85)
    anti_pts = landmarks[idx_antihelix]
    t_anti = np.concatenate(([0], np.cumsum(np.linalg.norm(np.diff(anti_pts, axis=0), axis=1))))
    spline_anti = CubicSpline(t_anti, anti_pts, bc_type='natural')
    
    resampled[75:85] = resample_spline_segment(spline_anti, t_anti[0], t_anti[-1], 10)
    
    return resampled

