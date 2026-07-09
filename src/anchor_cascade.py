"""
============================================================================
ANCHOR CASCADE — supervised per-anchor geometric refinement
============================================================================

Refines a coarse landmark prediction by regressing per-anchor corrections from
local mesh geometry, iterated in a cascade, then reconstructing all 85 points
via the anchor-driven pose+SSM solve (`anchor_align.solve_pose_and_shape`).

Validated (val, no leakage): 3.68 -> 2.42mm vs a 1.84mm oracle ceiling. Attacks
the pose component of the error (40% of the baseline) that global SSM+ICP misses.

estimator-safe: numpy + scipy + scikit-learn only. No yaml / config imports.
Works entirely in the SSM's mirrored-left canonical space; the caller mirrors
right ears in/out (as the existing predictor already does).
"""
import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor

from src.geometry import procrustes_align, apply_procrustes_transform
from src.anchor_align import default_anchor_indices, solve_pose_and_shape

# Local-descriptor neighbourhood radii (mm). 4 scales.
RADII = (2.0, 4.0, 6.0, 9.0)
# [surf_var, planarity, linearity, centroid_off(3), density,
#  oriented_normal(3), signed_plane_offset]
FEAT_PER_RADIUS = 11


def _inv_tf(tf):
    return {"R": tf["R"].T, "t_src": tf["t_tgt"], "t_tgt": tf["t_src"], "s": 1.0 / tf["s"]}


def _features(cloud: np.ndarray, tree: cKDTree, q: np.ndarray, gc: np.ndarray) -> np.ndarray:
    """Rotation-consistent local descriptors at query points q (Na,3), in the
    canonical (mm-scale) frame. gc = cloud centroid, used to orient normals
    outward so their sign is consistent across ears.
    Returns (Na, FEAT_PER_RADIUS*len(RADII))."""
    feats = []
    for r in RADII:
        nb_lists = tree.query_ball_point(q, r)
        block = np.zeros((len(q), FEAT_PER_RADIUS))
        for i, nb in enumerate(nb_lists):
            if len(nb) < 4:
                continue
            P = cloud[nb]
            c = P.mean(0)
            cov = (P - c).T @ (P - c) / len(nb)
            w, V = np.linalg.eigh(cov)          # w ascending; columns = eigenvecs
            e = np.clip(w, 1e-12, None)
            s = e.sum()
            n = V[:, 0]                          # normal = smallest-eigenvalue dir
            if n @ (q[i] - gc) < 0:              # orient outward (consistent sign)
                n = -n
            block[i, 0] = e[0] / s               # surface variation (curvature)
            block[i, 1] = (e[1] - e[0]) / e[2]   # planarity
            block[i, 2] = (e[2] - e[1]) / e[2]   # linearity
            block[i, 3:6] = c - q[i]             # centroid offset (directional)
            block[i, 6] = len(nb)                # density
            block[i, 7:10] = n                   # oriented surface normal
            block[i, 10] = (q[i] - c) @ n        # signed offset from local plane
        feats.append(block)
    return np.hstack(feats)


class AnchorCascade:
    """Cascaded per-anchor regression refiner. Serialized inside the predictor."""

    def __init__(self, keep_every: int = 3, n_stages: int = 2,
                 n_estimators: int = 120, max_depth: int = 12,
                 min_samples_leaf: int = 3, ridge: float = 5.0):
        self.anchor_idx = default_anchor_indices(keep_every)
        self.n_stages = n_stages
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.ridge = ridge
        self.stages = []  # list of {anchor_idx: RandomForestRegressor}
        self.fitted = False

    # ---- geometry helpers -------------------------------------------------
    def _canonical(self, mesh_verts, coarse_pred, ssm_mean, crop_margin=12.0):
        """Map ear cloud + coarse anchors into a canonical ROTATION frame that
        PRESERVES mm scale (so descriptor radii stay meaningful — the SSM mean is
        unit-normalized, ~0.3 diagonal, so we must NOT adopt its scale). Only the
        orientation is canonicalized; the pose+scale solve handles the rest.
        Inputs in mirrored-left space. Returns (cloud, tree, coarse_canon, R, c0)
        where world = canon @ R + c0."""
        tf = procrustes_align(ssm_mean, coarse_pred, allow_scale=True)[1]  # ssm->world
        R = tf["R"]              # ssm->world rotation; world->canon is R.T
        c0 = tf["t_tgt"]         # == coarse_pred.mean(0)
        lo, hi = coarse_pred.min(0) - crop_margin, coarse_pred.max(0) + crop_margin
        m = np.all((mesh_verts >= lo) & (mesh_verts <= hi), axis=1)
        cloud_w = mesh_verts[m] if m.any() else mesh_verts
        cloud = (cloud_w - c0) @ R.T                              # canonical, mm-scale
        tree = cKDTree(cloud)
        coarse_canon = (coarse_pred[self.anchor_idx] - c0) @ R.T
        return cloud, tree, coarse_canon, R, c0

    # ---- training ---------------------------------------------------------
    def fit(self, samples, ssm, verbose=True):
        """samples: iterable of (mesh_verts, coarse_pred, gt_full) — all in
        mirrored-left space. Trains the cascade in place."""
        ssm_mean = ssm.get_mean_shape()
        A = self.anchor_idx
        prepped = []  # (cloud, tree, coarse_canon, true_canon)
        for mesh_verts, coarse_pred, gt_full in samples:
            cloud, tree, coarse_canon, R, c0 = self._canonical(mesh_verts, coarse_pred, ssm_mean)
            true_canon = (gt_full[A] - c0) @ R.T
            prepped.append((cloud, tree, coarse_canon, true_canon))
        cur = [p[2].copy() for p in prepped]  # current estimate per sample (canonical)

        self.stages = []
        for st in range(self.n_stages):
            X = {a: [] for a in A}
            Y = {a: [] for a in A}
            for (cloud, tree, _, true_canon), c in zip(prepped, cur):
                f = _features(cloud, tree, c, cloud.mean(0))
                for k, a in enumerate(A):
                    X[a].append(f[k]); Y[a].append(true_canon[k] - c[k])
            stage = {}
            for a in A:
                rf = RandomForestRegressor(
                    n_estimators=self.n_estimators, max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf, n_jobs=-1, random_state=42)
                rf.fit(np.asarray(X[a]), np.asarray(Y[a]))
                stage[a] = rf
            # advance current estimates
            for j, (cloud, tree, _, _) in enumerate(prepped):
                f = _features(cloud, tree, cur[j], cloud.mean(0))
                cur[j] = cur[j] + np.array([stage[a].predict(f[k:k+1])[0]
                                            for k, a in enumerate(A)])
            self.stages.append(stage)
            if verbose:
                err = np.mean([np.linalg.norm(c - p[3], axis=1).mean()
                               for c, p in zip(cur, prepped)])
                print(f"    [anchor-cascade] stage {st+1}/{self.n_stages}: "
                      f"train anchor err {err:.3f}mm")
        self.fitted = True

    # ---- inference --------------------------------------------------------
    def refine(self, mesh_verts, coarse_pred, ssm):
        """Return refined full-85 prediction (mirrored-left space)."""
        if not self.fitted:
            return coarse_pred
        A = self.anchor_idx
        cloud, tree, cur, R, c0 = self._canonical(mesh_verts, coarse_pred, ssm.get_mean_shape())
        gc = cloud.mean(0)
        for stage in self.stages:
            f = _features(cloud, tree, cur, gc)
            cur = cur + np.array([stage[a].predict(f[k:k+1])[0]
                                  for k, a in enumerate(A)])
        anchors_world = cur @ R + c0
        return solve_pose_and_shape(ssm, A, anchors_world, ridge=self.ridge)
