"""
Dense-surface SSM hybrid fit (numpy, torch-free).

A dense-vertex statistical shape model of the ear (built by landmark-anchored
non-rigid ICP of one template onto all 280 training ears, then PCA) is fitted to a
new ear using BOTH:
  * the target ear SURFACE (dense, sub-mm accurate, but correspondence-ambiguous:
    many shape/pose configurations explain the same surface, differing by sliding), and
  * the deep model's predicted landmarks (good correspondence, but noisy).
Each term fixes the other's weakness. The model is LINEAR in the shape coefficients
and the PCA components are orthonormal, so the coefficients have an exact
closed-form MAP solution -- no iterative optimizer needed.

The fitted landmarks are then blended with the deep prediction (the fit alone is
slightly worse than the deep model; its value is decorrelated error). Measured on
validation: deep+surfproj 1.309 -> blend 1.298 -> +surface projection 1.294 mm.
The blend gain is robust (split-half: +0.011mm out-of-sample, positive in 100% of
200 repetitions).
"""
import numpy as np
from scipy.spatial import cKDTree


class DenseSSMFit:
    def __init__(self, path, kuse=120):
        z = np.load(path)
        self.mu = z["mean"].astype(np.float64)                 # (3n,)
        self.comps = z["comps"][:kuse].astype(np.float64)      # (K,3n) orthonormal
        self.eig = z["eig"][:kuse].astype(np.float64)          # (K,)
        self.F = z["template_F"].astype(np.int64)
        self.bf = z["bary_f"].astype(np.int64)
        self.bw = z["bary_w"].astype(np.float64)
        self.n = len(self.mu) // 3
        self.tri = self.F[self.bf]                             # (85,3) landmark triangles
        # landmark response of each component (needed for the landmark data term)
        self.Clm = np.stack([self._transport(c.reshape(self.n, 3)).reshape(-1)
                             for c in self.comps])             # (K,255)
        self.lm_mean = self._transport(self.mu.reshape(self.n, 3)).reshape(-1)
        self.G_lm = self.Clm @ self.Clm.T

    def _transport(self, V):
        return (self.bw[..., None] * V[self.tri]).sum(1)        # (85,3)

    @staticmethod
    def _wprocrustes(A, B, w):
        wn = w / max(w.sum(), 1e-9)
        ca = (wn[:, None] * A).sum(0); cb = (wn[:, None] * B).sum(0)
        A0, B0 = A - ca, B - cb
        U, _, Vt = np.linalg.svd((wn[:, None] * A0).T @ B0)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U = U.copy(); U[:, -1] *= -1; R = U @ Vt
        s = (wn[:, None] * B0 * (A0 @ R)).sum() / max((wn[:, None] * A0 * A0).sum(), 1e-9)
        return s, R, cb - s * (ca @ R)

    def fit(self, cloud, deep_lms, w_lm=300.0, sig=(9, 4, 1, 0.36, 0.16, 0.09),
            iters=4, cut=4.0):
        """cloud (P,3) target ear surface points, deep_lms (85,3) -> fitted (85,3)"""
        cloud = np.asarray(cloud, float); Ld = np.asarray(deep_lms, float)
        tree = cKDTree(cloud)
        K = len(self.comps)
        mean_V = self.mu.reshape(self.n, 3)
        c = np.zeros(K)
        s, R, t = self._wprocrustes(self._transport(mean_V), Ld, np.ones(85))
        I = np.eye(K); invlam = 1.0 / self.eig
        for sig2 in sig:
            for _ in range(iters):
                V = (self.mu + c @ self.comps).reshape(self.n, 3)
                Vw = s * (V @ R) + t
                dist, idx = tree.query(Vw)
                tgt = cloud[idx]
                w = (dist < cut).astype(float)
                if w.sum() < 100:
                    w = np.ones_like(w)
                A_src = np.vstack([V, self._transport(V)])
                B_dst = np.vstack([tgt, Ld])
                wc = np.concatenate([w, w_lm * 20.0 * np.ones(85)])
                s, R, t = self._wprocrustes(A_src, B_dst, wc)
                tgt_m = ((tgt - t) @ R.T) / s
                Ld_m = ((Ld - t) @ R.T) / s
                A = I / sig2 + w_lm * self.G_lm + np.diag(invlam)
                b = (self.comps @ (tgt_m.reshape(-1) - self.mu)) / sig2 \
                    + w_lm * (self.Clm @ (Ld_m.reshape(-1) - self.lm_mean))
                c = np.linalg.solve(A, b)
        V = (self.mu + c @ self.comps).reshape(self.n, 3)
        return self._transport(s * (V @ R) + t)

    def refine(self, cloud, deep_lms, alpha=0.3, **kw):
        """fitted landmarks blended with the deep prediction (alpha = SSM weight)"""
        got = self.fit(cloud, deep_lms, **kw)
        return alpha * got + (1 - alpha) * np.asarray(deep_lms, float)
