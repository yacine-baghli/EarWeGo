"""
============================================================================
DENSE CASCADE — all-85 geometric refinement with coarse-to-fine + SSM-project
============================================================================

Beats the anchor-solve design by detecting ALL 85 landmarks (not just anchors)
and denoising with SSM projection. Each round: build a rotation-only canonical
frame from the current estimate, cascade-regress all 85 offsets from local
geometry, then SSM-project (denoise) and feed back as the next round's coarse.

Validated (val): 3.68 -> 1.96 (single round) -> 1.85mm (3 rounds). Uses its OWN
raw-shape SSM (mm-scale mean) so the descriptor radii and projection behave
correctly and it reproduces the experiment exactly.

estimator-safe: numpy + scipy + scikit-learn only.
"""
import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor

from src.geometry import StatisticalShapeModel, procrustes_align, apply_procrustes_transform
from src.anchor_cascade import _features
from src.data_loader import NUM_LANDMARKS

ALL = np.arange(NUM_LANDMARKS)


class DenseCascade:
    def __init__(self, n_rounds=2, n_stages=2, n_ssm=30, n_estimators=25,
                 max_depth=10, min_samples_leaf=3, crop=20.0):
        self.n_rounds = n_rounds
        self.n_stages = n_stages
        self.n_ssm = n_ssm
        self.rf_kw = dict(n_estimators=n_estimators, max_depth=max_depth,
                          min_samples_leaf=min_samples_leaf, n_jobs=4, random_state=42)
        self.crop = crop
        self.ssm = None
        self.rounds = []       # rounds[r] = list of stages; stage = {i: RandomForest}
        self.fitted = False

    # ---- geometry helpers -------------------------------------------------
    def _frame(self, coarse):
        """Rotation-only canonical frame (preserve mm scale). world = canon@R + c0."""
        tf = procrustes_align(self.ssm.get_mean_shape(), coarse, allow_scale=True)[1]
        return tf["R"], tf["t_tgt"]

    def _project(self, pts):
        """SSM-project (denoise) a full-85 shape (world) via its own SSM."""
        aligned, tf = procrustes_align(pts, self.ssm.get_mean_shape(), allow_scale=True)
        recon = self.ssm.reconstruct(self.ssm.project(aligned))
        inv = {"R": tf["R"].T, "t_src": tf["t_tgt"], "t_tgt": tf["t_src"], "s": 1.0 / tf["s"]}
        return apply_procrustes_transform(recon, inv)

    def _canon_cloud(self, cloud_world, cur, R, c0):
        cl = (cloud_world - c0) @ R.T
        return cl, cKDTree(cl), (cur - c0) @ R.T

    # ---- training ---------------------------------------------------------
    def fit(self, samples, verbose=True):
        """samples: iterable of (cloud_world, coarse_pred, gt_full) in mirrored-left
        space. cloud_world is the (pre-cropped) ear point cloud."""
        samples = list(samples)
        clouds = [s[0] for s in samples]
        cur = [s[1].copy() for s in samples]          # world coarse, updated per round
        gts = [s[2] for s in samples]

        self.ssm = StatisticalShapeModel(self.n_ssm)
        self.ssm.fit(np.stack(gts))                    # own raw-shape SSM (mm scale)

        self.rounds = []
        for r in range(self.n_rounds):
            frames = [self._frame(cur[j]) for j in range(len(samples))]
            cl, trees, cur_c, true_c = [], [], [], []
            for j in range(len(samples)):
                R, c0 = frames[j]
                a, t, b = self._canon_cloud(clouds[j], cur[j], R, c0)
                cl.append(a); trees.append(t); cur_c.append(b)
                true_c.append((gts[j] - c0) @ R.T)
            stages = []
            for st in range(self.n_stages):
                X = {i: [] for i in ALL}; Y = {i: [] for i in ALL}
                for j in range(len(samples)):
                    f = _features(cl[j], trees[j], cur_c[j], cl[j].mean(0))
                    for i in ALL:
                        X[i].append(f[i]); Y[i].append(true_c[j][i] - cur_c[j][i])
                models = {}
                for i in ALL:
                    rf = RandomForestRegressor(**self.rf_kw)
                    rf.fit(np.asarray(X[i]), np.asarray(Y[i])); models[i] = rf
                for j in range(len(samples)):
                    f = _features(cl[j], trees[j], cur_c[j], cl[j].mean(0))
                    cur_c[j] = cur_c[j] + np.array([models[i].predict(f[i:i+1])[0] for i in ALL])
                stages.append(models)
            # map back to world + SSM-project -> next round coarse
            for j in range(len(samples)):
                R, c0 = frames[j]
                cur[j] = self._project(cur_c[j] @ R + c0)
            self.rounds.append(stages)
            if verbose:
                err = np.mean([np.linalg.norm(cur[j] - gts[j], axis=1).mean()
                               for j in range(len(samples))])
                print(f"    [dense-cascade] round {r+1}/{self.n_rounds}: train err {err:.3f}mm")
        self.fitted = True

    # ---- inference --------------------------------------------------------
    def refine(self, mesh_verts, coarse_pred):
        """Return refined full-85 (mirrored-left space)."""
        if not self.fitted:
            return coarse_pred
        lo, hi = coarse_pred.min(0) - self.crop, coarse_pred.max(0) + self.crop
        m = np.all((mesh_verts >= lo) & (mesh_verts <= hi), axis=1)
        cloud_world = mesh_verts[m] if m.any() else mesh_verts
        cur = coarse_pred
        for stages in self.rounds:
            R, c0 = self._frame(cur)
            cl, tree, cur_c = self._canon_cloud(cloud_world, cur, R, c0)
            for models in stages:
                f = _features(cl, tree, cur_c, cl.mean(0))
                cur_c = cur_c + np.array([models[i].predict(f[i:i+1])[0] for i in ALL])
            cur = self._project(cur_c @ R + c0)
        return cur
