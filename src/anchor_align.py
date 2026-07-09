"""
============================================================================
ANCHOR-DRIVEN ALIGNMENT — pose + SSM shape solve from a sparse anchor set
============================================================================

Given noisy world-space observations of a SUBSET of landmarks ("anchors"),
jointly estimate the similarity pose (scale, rotation, translation) and the
Statistical Shape Model coefficients that best explain them, then reconstruct
all 85 landmarks from the SSM.

Motivation (see docs / memory): 40% of the v1 baseline's 3.68mm error is pose.
Solving pose from distinctive anchor correspondences instead of blind template
ICP has a validated oracle ceiling of ~1.84mm and tolerates ~2mm anchor noise.

estimator-safe: numpy + scipy only. No yaml / config imports.
"""
import numpy as np

from src.geometry import procrustes_align, apply_procrustes_transform

# 4 anatomical contours (consecutive-index chains). Never interpolate across a
# break. Only the last two are equidistant; see finding-contour-structure.
CONTOURS = [(0, 24), (25, 54), (55, 74), (75, 84)]
NUM_LANDMARKS = 85


def default_anchor_indices(keep_every: int = 3) -> list[int]:
    """Every `keep_every`-th index within each contour, plus both endpoints.
    keep_every=3 -> ~32 anchors, near the flat part of the accuracy curve."""
    a = []
    for (s, e) in CONTOURS:
        idx = list(range(s, e + 1))
        a += idx[::keep_every] + [s, e]
    return sorted(set(a))


def solve_pose_and_shape(
    ssm,
    anchor_idx,
    anchor_targets: np.ndarray,
    n_iter: int = 4,
    ridge: float = 5.0,
) -> np.ndarray:
    """
    Jointly estimate similarity pose + SSM coefficients from anchor observations.

    Args:
        ssm: fitted StatisticalShapeModel (mean_shape flat, components (nc,255)).
        anchor_idx: iterable of landmark indices that are observed.
        anchor_targets: (len(anchor_idx), 3) observed world positions, in the
            SAME space the SSM was trained in (mirror right ears to left first).
        n_iter: alternating-minimization iterations.
        ridge: L2 penalty on SSM coefficients (regularizes against anchor noise).

    Returns:
        (85, 3) full landmark prediction in the observation world space.
    """
    mu = ssm.get_mean_shape()                                   # (85,3)
    comps = ssm.components.reshape(-1, NUM_LANDMARKS, 3)        # (nc,85,3)
    nc = comps.shape[0]
    a = np.asarray(list(anchor_idx))
    T_a = np.asarray(anchor_targets, dtype=np.float64)

    # Precompute the anchor design matrix for the coefficient LSQ.
    Ca = comps[:, a, :].reshape(nc, -1).T                       # (3*na, nc)
    G = Ca.T @ Ca + ridge * np.eye(nc)

    b = np.zeros(nc)
    tf = None
    for _ in range(n_iter):
        shape = mu + np.tensordot(b, comps, axes=1)             # ssm frame
        # Pose that maps current SSM-frame anchors onto the observed anchors.
        _, tf = procrustes_align(shape[a], T_a, allow_scale=True)
        inv = {"R": tf["R"].T, "t_src": tf["t_tgt"],
               "t_tgt": tf["t_src"], "s": 1.0 / tf["s"]}
        # Bring observed anchors into the SSM frame, then refit coefficients.
        T_a_frame = apply_procrustes_transform(T_a, inv)
        rhs = Ca.T @ (T_a_frame - mu[a]).reshape(-1)
        b = np.linalg.solve(G, rhs)

    shape = mu + np.tensordot(b, comps, axes=1)
    return apply_procrustes_transform(shape, tf)
