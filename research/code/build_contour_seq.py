"""
Surface-conditioned sequence features for ONE contour (step 3-4 of the plan).

For each ear: walk T ordered locations along the PREDICTED contour polyline and, at each
location, read the DENSE mesh: nearest-surface offset, oriented normal, multi-scale local
curvature, plus the polyline tangent and normalised arc length.

This is the evidence the phase must be inferred from -- the previous ridge experiment
failed precisely because it only saw the predicted landmark configuration, which cannot
reveal its own phase error.

    CONTOUR=inner python scratch/build_contour_seq.py
Output: scratch/seq_<contour>.npz  seq (340,T,C), plus GT/pred contours and subject ids.
"""
import os, sys
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.splits import get_split
from src.dataset import Dataset

MESH = "2026 Munich Tech Arena - Datas/2026 Munich Tech Arena - Datas/mesh"
LM = "2026 Munich Tech Arena - Datas/2026 Munich Tech Arena - Datas/landmarks"
MIRROR = np.array([1., -1., 1.])
NAMES = {"outer": (0, 24), "concha": (25, 54), "inner": (55, 74), "antihelix": (75, 84)}
CNAME = os.environ.get("CONTOUR", "inner")
LO, HI = NAMES[CNAME]
T = int(os.environ.get("T", "192"))
RADII = (1.0, 2.5)

z = np.load("scratch/oof_final.npz")
P_all, G_all, fold = z["pred"].astype(float), z["gt"].astype(float), z["fold"]
tr = get_split("train", mesh_dir=Path(MESH)); va = get_split("val", mesh_dir=Path(MESH))
order = [(p, s) for p in tr for s in ("left", "right")] + \
        [(p, s) for p in va for s in ("left", "right")]
ds = Dataset(MESH, LM); pid2idx = {p: i for i, p in enumerate(ds.subject_ids)}


def arc(P):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]


def eval_poly(P, sq):
    s = arc(P); out = np.empty((len(sq), 3))
    for k, t in enumerate(sq):
        j = min(max(np.searchsorted(s, t) - 1, 0), len(P) - 2)
        f = (t - s[j]) / max(s[j + 1] - s[j], 1e-12)
        out[k] = P[j] + f * (P[j + 1] - P[j])
    return out


cache = {}
seqs = np.zeros((len(P_all), T, 13), np.float32)
for i, (pid, side) in enumerate(order):
    if pid not in cache:
        m = ds[pid2idx[pid]][0]
        cache[pid] = (np.asarray(m.vertices), np.asarray(m.faces))
        if len(cache) > 30:
            cache.pop(next(iter(cache)))
    V, F = cache[pid]
    if side == "right":
        V = V * MIRROR; F = F[:, ::-1]
    Pc = P_all[i, LO:HI + 1]
    # dense local crop + vertex normals
    lo_, hi_ = Pc.min(0) - 8, Pc.max(0) + 8
    vin = np.all((V >= lo_) & (V <= hi_), axis=1)
    fm = vin[F].all(axis=1); Fs = F[fm]
    keep = np.unique(Fs) if len(Fs) else np.arange(len(V))
    remap = -np.ones(len(V), int); remap[keep] = np.arange(len(keep))
    Vc = V[keep]; Fc = remap[Fs] if len(Fs) else None
    VN = np.zeros_like(Vc)
    if Fc is not None and len(Fc):
        fn = np.cross(Vc[Fc[:, 1]] - Vc[Fc[:, 0]], Vc[Fc[:, 2]] - Vc[Fc[:, 0]])
        for c in range(3):
            np.add.at(VN, Fc[:, c], fn)
    nrm = np.linalg.norm(VN, axis=1, keepdims=True)
    VN = np.where(nrm > 1e-9, VN / np.maximum(nrm, 1e-9), np.array([0., 0., 1.]))
    tree = cKDTree(Vc)
    # T ordered locations along the predicted polyline
    s = arc(Pc); L = s[-1]
    sq = np.linspace(0, L, T)
    Q = eval_poly(Pc, sq)
    d, j = tree.query(Q)
    off = Vc[j] - Q                                  # nearest-surface offset
    nn = VN[j]                                       # oriented normal
    # tangent of the polyline
    tg = np.gradient(Q, axis=0)
    tg /= (np.linalg.norm(tg, axis=1, keepdims=True) + 1e-9)
    # multi-scale local curvature (surface variation) at the surface point
    curv = np.zeros((T, len(RADII)), np.float32)
    for ri, r in enumerate(RADII):
        nb = tree.query_ball_point(Vc[j], r)
        for k, idxs in enumerate(nb):
            if len(idxs) >= 5:
                Pp = Vc[idxs] - Vc[idxs].mean(0)
                ev = np.linalg.eigvalsh(Pp.T @ Pp / len(idxs))
                curv[k, ri] = max(ev[0], 0) / max(ev.sum(), 1e-12)
    cen = Q.mean(0); sc = max(L, 1e-6)
    feat = np.concatenate([(Q - cen) / sc, off, nn, tg,
                           (sq / sc)[:, None], curv], axis=1)   # 3+3+3+3+1+2 = 15 -> trim
    seqs[i] = feat[:, :13] if feat.shape[1] >= 13 else np.pad(feat, ((0, 0), (0, 13 - feat.shape[1])))
    if (i + 1) % 40 == 0:
        print(f"  {i+1}/{len(order)}", flush=True)

np.savez_compressed(f"scratch/seq_{CNAME}.npz", seq=seqs,
                    pred=P_all[:, LO:HI + 1], gt=G_all[:, LO:HI + 1],
                    fold=fold, subj=np.arange(len(P_all)) // 2)
print(f"saved scratch/seq_{CNAME}.npz  seq {seqs.shape} for contour {CNAME} ({LO}-{HI})")
