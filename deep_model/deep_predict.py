"""
Submission-time deep inference: ENSEMBLE of nets x TTA rotations -> blend with SSM.
Operates in the canonical (framed) coordinate system. Torch-free.

The pipeline supplies (cloud_canonical, coarse_canonical) via the same framing as
preprocess_deep; this returns the 85 refined landmarks in that same frame.
"""
import numpy as np
from .deep_infer import DeepNet


def _rot(ax, a):
    x, y, z = ax; c, s, C = np.cos(a), np.sin(a), 1 - np.cos(a)
    return np.array([[c+x*x*C, x*y*C-z*s, x*z*C+y*s],
                     [y*x*C+z*s, c+y*y*C, y*z*C-x*s],
                     [z*x*C-y*s, z*y*C+x*s, c+z*z*C]])


TTA_ROTS = [np.eye(3)] + [_rot(a, ang) for a in ([1, 0, 0], [0, 1, 0], [0, 0, 1])
                          for ang in (0.18, -0.18)]


def _procrustes(src, tgt):
    ms, mt = src.mean(0), tgt.mean(0)
    A, B = src - ms, tgt - mt
    U, _, Vt = np.linalg.svd(A.T @ B); R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1; R = U @ Vt
    s = (B * (A @ R)).sum() / (A * A).sum()
    return s, R, ms, mt


def ssm_project(pts, mean, comp):
    NL = len(pts)
    s, R, ms, mt = _procrustes(pts, mean.reshape(NL, 3))
    aligned = s * ((pts - ms) @ R) + mt
    coeff = (aligned.flatten() - mean) @ comp.T
    recon = (mean + coeff @ comp).reshape(NL, 3)
    return ((recon - mt) @ R.T) / s + ms


class DeepEnsemble:
    def __init__(self, weight_paths, ssm_mean, ssm_comp, blend=0.3, tta=True):
        self.nets = [DeepNet(p) for p in weight_paths]
        self.mean = np.asarray(ssm_mean, float)
        self.comp = np.asarray(ssm_comp, float)
        self.blend = blend
        self.rots = TTA_ROTS if tta else [np.eye(3)]

    def predict(self, cloud, coarse):
        cloud = np.asarray(cloud, float); coarse = np.asarray(coarse, float)
        acc = np.zeros((coarse.shape[0], 3)); n = 0
        for M in self.rots:
            cl, co = cloud @ M.T, coarse @ M.T
            for net in self.nets:
                acc += net.predict(cl, co) @ M
                n += 1
        raw = acc / n
        proj = ssm_project(raw, self.mean, self.comp)
        return self.blend * proj + (1 - self.blend) * raw
