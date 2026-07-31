"""
Per-landmark dense-patch features + local frames for the orthogonal corrector.

At each FROZEN OOF-predicted landmark, build the orthonormal frame (t, b, n) -- contour
tangent from neighbouring predicted landmarks, oriented mesh normal re-orthogonalised
against t, b = n x t -- and describe the dense local surface IN THAT FRAME, so every
feature is rotation-invariant and tells the head which way the true surface lies.

Output: scratch/ortho_feats.npz  feats (340,85,16), base, gt, t, b, n, fold, subj
"""
import os, sys
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.splits import get_split
from src.dataset import Dataset

MESH = "2026 Munich Tech Arena - Datas/2026 Munich Tech Arena - Datas/mesh"
LM = "2026 Munich Tech Arena - Datas/2026 Munich Tech Arena - Datas/landmarks"
MIRROR = np.array([1., -1., 1.])
CONT = [(0, 24), (25, 54), (55, 74), (75, 84)]
RADII = (2.0, 5.0)

z = np.load("scratch/oof_final.npz")
P, G, fold = z["pred"].astype(np.float64), z["gt"].astype(np.float64), z["fold"]
NE = len(P)
tr = get_split("train", mesh_dir=Path(MESH)); va = get_split("val", mesh_dir=Path(MESH))
order = [(p, s) for p in tr for s in ("left", "right")] + \
        [(p, s) for p in va for s in ("left", "right")]
ds = Dataset(MESH, LM); pid2idx = {q: i for i, q in enumerate(ds.subject_ids)}

FEAT = 14   # 4 (nearest-surface offset in frame + dist) + 2 radii x 5 descriptors
feats = np.zeros((NE, 85, FEAT), np.float32)
Tt = np.zeros((NE, 85, 3)); Bb = np.zeros((NE, 85, 3)); Nn = np.zeros((NE, 85, 3))
cache = {}
for i, (pid, side) in enumerate(order):
    if pid not in cache:
        m = ds[pid2idx[pid]][0]
        cache[pid] = (np.asarray(m.vertices), np.asarray(m.faces))
        if len(cache) > 25:
            cache.pop(next(iter(cache)))
    V, F = cache[pid]
    if side == "right":
        V = V * MIRROR; F = F[:, ::-1]
    lo_, hi_ = P[i].min(0) - 10, P[i].max(0) + 10
    vin = np.all((V >= lo_) & (V <= hi_), axis=1)
    fm = vin[F].all(axis=1); Fs = F[fm]
    keep = np.unique(Fs) if len(Fs) else np.where(vin)[0]
    remap = -np.ones(len(V), int); remap[keep] = np.arange(len(keep))
    Vc = V[keep]; Fc = remap[Fs] if len(Fs) else None
    VN = np.zeros_like(Vc)
    if Fc is not None and len(Fc):
        fn = np.cross(Vc[Fc[:, 1]] - Vc[Fc[:, 0]], Vc[Fc[:, 2]] - Vc[Fc[:, 0]])
        for c in range(3):
            np.add.at(VN, Fc[:, c], fn)
    nr = np.linalg.norm(VN, axis=1, keepdims=True)
    VN = np.where(nr > 1e-9, VN / np.maximum(nr, 1e-9), np.array([0., 0., 1.]))
    tree = cKDTree(Vc)
    for lo, hi in CONT:
        Pc = P[i, lo:hi + 1]
        t = np.gradient(Pc, axis=0)
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
        dnn, jnn = tree.query(Pc)
        nn = VN[jnn]
        nn = nn - (nn * t).sum(1, keepdims=True) * t
        nn /= np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-9)
        b = np.cross(nn, t)
        b /= np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-9)
        Tt[i, lo:hi + 1] = t; Bb[i, lo:hi + 1] = b; Nn[i, lo:hi + 1] = nn
        # nearest-surface offset expressed in the local frame
        off = Vc[jnn] - Pc
        col = [ (off*t).sum(1), (off*b).sum(1), (off*nn).sum(1), dnn ]
        for r in RADII:
            nb = tree.query_ball_point(Pc, r)
            cb = np.zeros(len(Pc)); cn = np.zeros(len(Pc)); ct = np.zeros(len(Pc))
            sv = np.zeros(len(Pc)); nrm_dev = np.zeros(len(Pc))
            for k, idxs in enumerate(nb):
                if len(idxs) >= 5:
                    Q = Vc[idxs] - Pc[k]
                    ct[k] = (Q @ t[k]).mean(); cb[k] = (Q @ b[k]).mean(); cn[k] = (Q @ nn[k]).mean()
                    Qc = Q - Q.mean(0)
                    ev = np.linalg.eigvalsh(Qc.T @ Qc / len(idxs))
                    sv[k] = max(ev[0], 0) / max(ev.sum(), 1e-12)
                    nrm_dev[k] = 1.0 - np.clip((VN[idxs] @ nn[k]).mean(), -1, 1)
            col += [cb, cn, sv, nrm_dev, ct]
        feats[i, lo:hi + 1] = np.stack(col, axis=1)[:, :FEAT]
    if (i + 1) % 40 == 0:
        print(f"  {i+1}/{NE}", flush=True)

np.savez_compressed("scratch/ortho_feats.npz", feats=feats, base=P, gt=G,
                    t=Tt, b=Bb, n=Nn, fold=fold, subj=np.arange(NE) // 2)
print(f"saved scratch/ortho_feats.npz feats {feats.shape}")
