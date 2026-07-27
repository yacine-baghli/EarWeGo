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
from scipy.linalg import orthogonal_procrustes

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


def _weighted_procrustes(src, tgt, w):
    """Similarity transform (s, R, t) mapping src onto tgt with per-point weights.
    Same dict convention as geometry.procrustes_align:
        aligned = s * (src - t_src) @ R + t_tgt.
    Weighting √w on centered points reduces to the standard Procrustes objective."""
    W = w.sum()
    mc_s = (w[:, None] * src).sum(0) / W
    mc_t = (w[:, None] * tgt).sum(0) / W
    Xc, Yc = src - mc_s, tgt - mc_t
    sw = np.sqrt(w)[:, None]
    R, _ = orthogonal_procrustes(sw * Xc, sw * Yc)
    XR = Xc @ R
    s = (w * (Yc * XR).sum(1)).sum() / (w * (XR * XR).sum(1)).sum()
    return {"R": R, "t_src": mc_s, "t_tgt": mc_t, "s": s}


def solve_pose_and_shape(
    ssm,
    anchor_idx,
    anchor_targets: np.ndarray,
    n_iter: int = 4,
    ridge: float = 5.0,
    robust: bool = False,
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
        robust: if True, iteratively down-weight anchors whose residual is large
            (Huber IRLS). A no-op when all anchors agree; rejects gross outliers
            (mis-detected anchors) so one bad point can't drag the whole pose.

    Returns:
        (85, 3) full landmark prediction in the observation world space.
    """
    mu = ssm.get_mean_shape()                                   # (85,3)
    comps = ssm.components.reshape(-1, NUM_LANDMARKS, 3)        # (nc,85,3)
    nc = comps.shape[0]
    a = np.asarray(list(anchor_idx))
    T_a = np.asarray(anchor_targets, dtype=np.float64)
    na = len(a)

    Ca = comps[:, a, :].reshape(nc, -1).T                       # (3*na, nc)
    w = np.ones(na)
    G = Ca.T @ Ca + ridge * np.eye(nc)                          # reused if not robust

    b = np.zeros(nc)
    tf = None
    for _ in range(n_iter):
        shape = mu + np.tensordot(b, comps, axes=1)             # ssm frame
        # Pose mapping current SSM-frame anchors onto the observed anchors.
        if robust:
            tf = _weighted_procrustes(shape[a], T_a, w)
        else:
            _, tf = procrustes_align(shape[a], T_a, allow_scale=True)
        inv = {"R": tf["R"].T, "t_src": tf["t_tgt"],
               "t_tgt": tf["t_src"], "s": 1.0 / tf["s"]}
        # Bring observed anchors into the SSM frame, then refit coefficients.
        T_a_frame = apply_procrustes_transform(T_a, inv)
        resid_frame = (T_a_frame - mu[a])                       # (na,3)
        if robust:
            wr = np.repeat(w, 3)[:, None]
            Gw = (Ca * wr).T @ Ca + ridge * np.eye(nc)
            rhs = (Ca * wr).T @ resid_frame.reshape(-1)
            b = np.linalg.solve(Gw, rhs)
        else:
            rhs = Ca.T @ resid_frame.reshape(-1)
            b = np.linalg.solve(G, rhs)

        if robust:
            # Update weights from per-anchor world residuals (Huber).
            shape = mu + np.tensordot(b, comps, axes=1)
            pred_a = apply_procrustes_transform(shape[a], tf)
            r = np.linalg.norm(pred_a - T_a, axis=1)
            delta = max(2.0, 2.0 * np.median(r))               # adaptive; no-op if all small
            w = np.where(r <= delta, 1.0, delta / np.maximum(r, 1e-9))

    shape = mu + np.tensordot(b, comps, axes=1)
    return apply_procrustes_transform(shape, tf)
