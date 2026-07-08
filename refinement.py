"""
Post-hoc refinements for predicted pinna landmarks, targeting the three error
sources the evaluation diagnostics exposed:

  1. clamp_scale          -> kills the 10-16% scale-error tail ears
  2. resample_contours    -> enforces the equal-arc-length definition of the 63
                             non-anchor landmarks; handles the inner-helix
                             continuation (worst region). Reduces inner-helix / helix error.
  3. selective_snap       -> snap only outer-contour points to the surface, so the
                             concha-bowl and inner-helix points aren't pulled off.

All functions are pure (operate on (85,3) arrays); drop them into predictor.py's
extract() at the marked points. Nothing here needs the dataset.
"""
from __future__ import annotations
import numpy as np

# ── contour layout (absolute landmark indices, ordered along each contour) ──
# anchors are the clearly-defined points from the challenge brief; everything
# between consecutive anchors is equally spaced by arc length.
CONTOURS = {
    "outer_helix":   {"span": (0, 25),  "anchors": [0, 6, 22, 24]},
    "concha":        {"span": (25, 55), "anchors": [25, 33, 42, 46, 50, 54]},
    "inner_helix":   {"span": (55, 75), "anchors": [55, 64, 74]},   # 64->74 = continuation
    "sup_antihelix": {"span": (75, 85), "anchors": [75, 84]},
}
# outer contours that genuinely lie on the mesh surface (safe to snap)
SURFACE_CONTOURS = ("outer_helix", "concha")


# ─────────────────────────────────────────────────────────────────────────────
def clamp_scale(transform: dict, lo: float = 0.92, hi: float = 1.08) -> dict:
    """
    Clamp the similarity scale from procrustes_align to a plausible range.

    The predictor estimates scale when fitting the ICP-initialised shape to the
    SSM mean; a bad init yields absurd scales (the tail ears). Clamping to a band
    learned from the SSM training scales removes those blow-ups. Returns a *new*
    transform dict (does not mutate the input).
    """
    s = float(transform["s"])
    t = dict(transform)
    t["s"] = float(np.clip(s, lo, hi))
    return t


# ─────────────────────────────────────────────────────────────────────────────
def _arclength_resample_segment(pts: np.ndarray, count: int) -> np.ndarray:
    """
    Given the ordered points of one contour (anchors + intermediates), return the
    contour resampled so intermediates are equally spaced by arc length between
    the fixed endpoints. `pts` here is ONE segment: endpoints stay, interior gets
    `count` points total including endpoints.
    """
    from scipy.interpolate import splprep, splev

    pts = np.asarray(pts, float)
    m = len(pts)
    if m < 2:
        return pts
    # fit a smooth spline through the segment's points (degree<=3, light smoothing)
    k = min(3, m - 1)
    try:
        tck, _ = splprep(pts.T, k=k, s=m * 0.1)
    except Exception:
        # fallback: straight-line equal spacing between endpoints
        return np.linspace(pts[0], pts[-1], count)

    # dense sample -> cumulative arc length -> pick equal-arclength params
    uu = np.linspace(0, 1, 400)
    dense = np.array(splev(uu, tck)).T
    seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 0:
        return np.linspace(pts[0], pts[-1], count)
    targets = np.linspace(0, total, count)
    u_at = np.interp(targets, cum, uu)
    return np.array(splev(u_at, tck)).T


def resample_contours(landmarks: np.ndarray, contours: dict = CONTOURS) -> np.ndarray:
    """
    Re-impose equal-arc-length spacing on every contour, keeping anchor points
    fixed. This enforces the exact rule the ground-truth landmarks were built
    with, so the 63 non-anchor points snap toward their defined positions.
    """
    out = np.asarray(landmarks, float).copy()
    for c in contours.values():
        lo, hi = c["span"]
        anchors = c["anchors"]
        # resample each anchor-to-anchor segment independently
        for a, b in zip(anchors[:-1], anchors[1:]):
            count = b - a + 1                     # inclusive of both anchors
            seg_pts = out[a:b + 1]
            new = _arclength_resample_segment(seg_pts, count)
            # keep the anchors exactly where the model put them
            new[0], new[-1] = out[a], out[b]
            out[a:b + 1] = new
    return out


# ─────────────────────────────────────────────────────────────────────────────
def selective_snap(result: np.ndarray, mesh, contours: dict = CONTOURS,
                   snap: tuple = SURFACE_CONTOURS) -> np.ndarray:
    """
    Snap ONLY the outer-contour landmarks to the mesh surface. The concha-bowl
    and inner-helix points sit off the outer surface, so snapping them to the
    nearest surface point (as the current Step 6 does for all 85) drags them
    away from the truth.
    """
    idxs = []
    for name in snap:
        lo, hi = contours[name]["span"]
        idxs.extend(range(lo, hi))
    idxs = np.array(sorted(set(idxs)))
    out = np.asarray(result, float).copy()
    try:
        closest, _, _ = mesh.nearest.on_surface(out[idxs])
        out[idxs] = closest
    except Exception:
        pass
    return out
