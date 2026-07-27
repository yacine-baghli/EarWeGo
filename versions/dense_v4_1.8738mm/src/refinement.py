"""
============================================================================
REFINEMENT — Post-prediction refinement stages for landmark quality
============================================================================

Three toggleable stages, designed for A/B comparison via the config system:

  1. clamp_scale      — Limit the Procrustes scale factor to a safe range
  2. resample_contours — Re-exported from geometry.py (equal arc-length spacing)
  3. selective_snap    — Snap only non-contour (anchor) landmarks to the mesh
                         surface, preserving the contour spacing from step 2.

Order: clamp → SSM reconstruct → inverse transform → KNN blend → resample → selective snap

Usage:
    from src.refinement import clamp_scale, resample_contours, selective_snap
"""

import numpy as np
from src.geometry import resample_contours  # noqa: F401 — re-export


# ─── Anchor / contour index definitions ──────────────────────────────────────
# These are the anatomically-defined anchor landmarks that are NOT derived
# from spline interpolation. They should be snapped to the surface.
# Everything else is a contour-interpolated point that should keep its spacing.

CONTOUR_RANGES = [
    (0, 25),     # Outer Helix        (indices 0–24)
    (25, 55),    # Concha Outline      (indices 25–54)
    (55, 75),    # Inner Helix         (indices 55–74)
    (75, 85),    # Superior Antihelix  (indices 75–84)
]

# Anchors: contour endpoints and anatomical keypoints that are safe to snap
ANCHOR_INDICES = {
    # Outer Helix anchors
    0, 6, 22, 24,
    # Concha anchors
    25, 33, 42, 46, 50, 54,
    # Inner Helix anchors
    55, 64,
    # Superior Antihelix anchors
    75, 84,
}

# Intermediate (non-anchor) contour points — should NOT be individually snapped
CONTOUR_INDICES = set()
for lo, hi in CONTOUR_RANGES:
    for i in range(lo, hi):
        if i not in ANCHOR_INDICES:
            CONTOUR_INDICES.add(i)


def clamp_scale(transform: dict, lo: float = 0.92, hi: float = 1.08) -> dict:
    """
    Clamp the Procrustes scale factor to [lo, hi].
    
    Extreme scale factors indicate ICP alignment drifted onto a wrong region
    (e.g. the skull instead of the pinna). Clamping the inverse transform keeps
    the back-projected landmarks at a physically plausible size.
    
    Args:
        transform: Procrustes transform dict with keys {R, t_src, t_tgt, s}.
        lo: Minimum allowed scale factor.
        hi: Maximum allowed scale factor.
    
    Returns:
        New transform dict with s clamped (original is not mutated).
    """
    clamped = dict(transform)
    clamped["s"] = float(np.clip(transform["s"], lo, hi))
    return clamped


def selective_snap(
    landmarks: np.ndarray,
    mesh,
    snap_anchors: bool = True,
    snap_contours: bool = False,
    max_snap_dist_mm: float = 5.0,
) -> np.ndarray:
    """
    Snap landmarks to the nearest mesh surface point, selectively.
    
    By default, only anchor landmarks (contour endpoints / keypoints) are
    snapped. Contour-interpolated points are left untouched so that the
    equal-arc-length spacing from resample_contours() is preserved.
    
    A distance guard prevents catastrophic snaps: if the nearest surface
    point is farther than max_snap_dist_mm, the landmark stays in place.
    
    Args:
        landmarks:   (85, 3) predicted landmark array.
        mesh:        trimesh.Trimesh of the ear/head.
        snap_anchors:   If True, snap anchor landmarks to the surface.
        snap_contours:  If True, also snap contour points (use with caution).
        max_snap_dist_mm: Maximum distance to allow snapping (mm).
    
    Returns:
        (85, 3) array with selected landmarks snapped.
    """
    result = landmarks.copy()
    
    # Determine which indices to snap
    snap_idx = set()
    if snap_anchors:
        snap_idx |= ANCHOR_INDICES
    if snap_contours:
        snap_idx |= CONTOUR_INDICES
    
    if not snap_idx:
        return result
    
    # Compute nearest surface points for all landmarks at once (batch query)
    try:
        closest_pts, distances, _ = mesh.nearest.on_surface(landmarks)
    except Exception:
        # Fallback to vertex-only snap via KDTree
        from scipy.spatial import cKDTree
        tree = cKDTree(np.array(mesh.vertices))
        distances, indices = tree.query(landmarks)
        closest_pts = np.array(mesh.vertices)[indices]
    
    # Apply selectively with distance guard
    for idx in snap_idx:
        if distances[idx] <= max_snap_dist_mm:
            result[idx] = closest_pts[idx]
    
    return result
