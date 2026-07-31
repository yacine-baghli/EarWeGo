"""3D thin-plate spline warp — maps a source landmark set to a target set and
applies the smooth deformation to arbitrary points (e.g. an ear point cloud)."""
import numpy as np


class TPS3D:
    def __init__(self, reg=0.0):
        self.reg = reg  # optional smoothing regularization

    def fit(self, src, dst):
        src = np.asarray(src, float); dst = np.asarray(dst, float)
        n = len(src)
        d = np.linalg.norm(src[:, None, :] - src[None, :, :], axis=2)  # (n,n)
        K = d                                                          # 3D biharmonic kernel U(r)=r
        if self.reg:
            K = K + self.reg * np.eye(n)
        P = np.hstack([np.ones((n, 1)), src])                          # (n,4)
        L = np.zeros((n + 4, n + 4))
        L[:n, :n] = K; L[:n, n:] = P; L[n:, :n] = P.T
        Y = np.vstack([dst, np.zeros((4, 3))])                         # (n+4,3)
        sol = np.linalg.solve(L, Y)
        self.w = sol[:n]; self.a = sol[n:]; self.src = src
        return self

    def transform(self, pts):
        pts = np.asarray(pts, float)
        d = np.linalg.norm(pts[:, None, :] - self.src[None, :, :], axis=2)  # (m,n)
        U = d
        P = np.hstack([np.ones((len(pts), 1)), pts])                       # (m,4)
        return U @ self.w + P @ self.a


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    src = rng.randn(85, 3) * 10
    dst = src + rng.randn(85, 3) * 1.5            # small plausible deformation
    tps = TPS3D().fit(src, dst)
    err = np.abs(tps.transform(src) - dst).max()
    print(f"exact interpolation error at control pts: {err:.2e}")
    # smoothness: a nearby point moves ~like its neighbours
    p = src[10] + np.array([0.1, 0.1, 0.1])
    moved = tps.transform(p[None])[0] - p
    ctrl_move = dst[10] - src[10]
    print(f"nearby-point displacement vs control displacement: "
          f"{np.linalg.norm(moved - ctrl_move):.3f}mm (should be small)")
