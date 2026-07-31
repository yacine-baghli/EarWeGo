"""
Ablation A: alternative surface samplers for all 340 ears.

The shipped `_frame` uses np.random.randint over cropped VERTICES => sampling WITH
replacement (duplicate points, which also bias the kNN graph) and density-biased
(dense mesh regions oversampled) rather than uniform over surface AREA.

Samplers:
  repl   : current behaviour (randint over vertices, with replacement)
  norepl : uniform over vertices WITHOUT replacement
  area   : area-weighted uniform sampling over triangle SURFACE (barycentric)
  fps    : farthest-point sampling (seeded, from a random subset for tractability)

    SAMPLER=area M=4 python scratch/build_clouds_sampler.py
Output: scratch/clouds_<sampler>.npz  clouds (340,M,2048,3)
"""
import os, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.splits import get_split
from src.dataset import Dataset

MESH = "2026 Munich Tech Arena - Datas/2026 Munich Tech Arena - Datas/mesh"
LM = "2026 Munich Tech Arena - Datas/2026 Munich Tech Arena - Datas/landmarks"
MIRROR = np.array([1., -1., 1.])
NPTS, MARGIN = int(os.environ.get("NPTS", "2048")), 14.0
M = int(os.environ.get("M", "4"))
SAMPLER = os.environ.get("SAMPLER", "area")

d = np.load("scratch/deep_dataset.npz", allow_pickle=True)
coarse, Rm, c0 = d["coarse"], d["R"], d["c0"]
tr = get_split("train", mesh_dir=Path(MESH)); va = get_split("val", mesh_dir=Path(MESH))
order = [(p, s) for p in tr for s in ("left", "right")] + \
        [(p, s) for p in va for s in ("left", "right")]
ds = Dataset(MESH, LM); pid2idx = {q: i for i, q in enumerate(ds.subject_ids)}


def sample(Vc, Fc, npts, rng):
    if SAMPLER == "repl":
        return Vc[rng.randint(0, len(Vc), npts)]
    if SAMPLER == "norepl":
        k = min(npts, len(Vc))
        idx = rng.choice(len(Vc), k, replace=False)
        if k < npts:
            idx = np.concatenate([idx, rng.choice(len(Vc), npts - k, replace=True)])
        return Vc[idx]
    if SAMPLER == "area":
        if Fc is None or len(Fc) == 0:
            return Vc[rng.randint(0, len(Vc), npts)]
        A, B, C = Vc[Fc[:, 0]], Vc[Fc[:, 1]], Vc[Fc[:, 2]]
        ar = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)
        p = ar / max(ar.sum(), 1e-12)
        fi = rng.choice(len(Fc), npts, p=p)                 # area-weighted faces
        u = rng.rand(npts, 1); v = rng.rand(npts, 1)
        fl = (u + v) > 1
        u[fl] = 1 - u[fl]; v[fl] = 1 - v[fl]                # uniform in the triangle
        return A[fi] + u * (B[fi] - A[fi]) + v * (C[fi] - A[fi])
    if SAMPLER == "fps":
        pool = Vc[rng.choice(len(Vc), min(len(Vc), 6000), replace=False)]
        sel = np.empty(npts, int); sel[0] = rng.randint(len(pool))
        dmin = np.linalg.norm(pool - pool[sel[0]], axis=1)
        for k in range(1, npts):
            sel[k] = int(np.argmax(dmin))
            dmin = np.minimum(dmin, np.linalg.norm(pool - pool[sel[k]], axis=1))
        return pool[sel]
    raise ValueError(SAMPLER)


cache = {}
out = np.zeros((len(order), M, NPTS, 3), np.float32)
for i, (pid, side) in enumerate(order):
    if pid not in cache:
        m = ds[pid2idx[pid]][0]
        cache[pid] = (np.asarray(m.vertices), np.asarray(m.faces))
        if len(cache) > 25:
            cache.pop(next(iter(cache)))
    V, F = cache[pid]
    if side == "right":
        V = V * MIRROR; F = F[:, ::-1]
    R, cc = Rm[i], c0[i]
    cw = coarse[i] @ R + cc
    lo, hi = cw.min(0) - MARGIN, cw.max(0) + MARGIN
    vin = np.all((V >= lo) & (V <= hi), axis=1)
    fm = vin[F].all(axis=1); Fs = F[fm]
    keep = np.unique(Fs) if len(Fs) else np.where(vin)[0]
    if len(keep) < 10:
        keep = np.arange(len(V)); Fs = F
    remap = -np.ones(len(V), int); remap[keep] = np.arange(len(keep))
    Vc = V[keep]; Fc = remap[Fs] if len(Fs) else None
    Vl = (Vc - cc) @ R.T                                   # canonical frame
    Fl = Fc
    for j in range(M):
        rng = np.random.RandomState(1000 + 97 * i + j)
        out[i, j] = sample(Vl, Fl, NPTS, rng).astype(np.float32)
    if (i + 1) % 60 == 0:
        print(f"  {i+1}/{len(order)}", flush=True)

fn = f"scratch/clouds_{SAMPLER}{'' if NPTS==2048 else '_'+str(NPTS)}.npz"
np.savez_compressed(fn, clouds=out, coarse=coarse, true=d["true"], R=Rm, c0=c0,
                    split=d["split"])
print(f"saved {fn} {out.shape} ({os.path.getsize(fn)/1e6:.1f} MB) sampler={SAMPLER}")
