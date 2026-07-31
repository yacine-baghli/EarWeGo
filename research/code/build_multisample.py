"""
Fresh-surface-sample TTA (brief v2 §7.2).

Our dataset stores ONE fixed 2048-point sample per ear, and our TTA only rotated that
same sample -- which cannot average away the variance caused by WHICH points were
sampled. The comparison pipeline reports ~0.24 mm of prediction displacement when the
surface sample changes, so that variance is real and currently unaveraged.

Builds M independent 2048-point samples per val ear using the SAME framing (R, c0), so
the only thing that changes is the surface sample.

Output: scratch/val_multisample.npz  clouds (60, M, 2048, 3)
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
NPTS = 2048
M = int(os.environ.get("M", "8"))
MARGIN = 14.0

d = np.load("scratch/deep_dataset.npz", allow_pickle=True)
coarse, true, Rm, c0, split = d["coarse"], d["true"], d["R"], d["c0"], d["split"]
va = np.where(split == "val")[0]
vp = get_split("val", mesh_dir=Path(MESH))
ds = Dataset(MESH, LM); pid2idx = {p: i for i, p in enumerate(ds.subject_ids)}
order = [(p, s) for p in vp for s in ("left", "right")]
cache = {}
out = np.zeros((len(va), M, NPTS, 3), np.float32)
for k, i in enumerate(va):
    pid, side = order[k]
    if pid not in cache:
        m = ds[pid2idx[pid]][0]
        cache[pid] = np.asarray(m.vertices)
    V = cache[pid]
    if side == "right":
        V = V * MIRROR
    R, cc = Rm[i], c0[i]
    cw = coarse[i] @ R + cc                       # coarse landmarks in world frame
    lo, hi = cw.min(0) - MARGIN, cw.max(0) + MARGIN
    msk = np.all((V >= lo) & (V <= hi), axis=1)
    crop = V[msk] if msk.any() else V
    cl = (crop - cc) @ R.T                        # canonical frame
    for j in range(M):
        rng = np.random.RandomState(1000 + 97 * k + j)
        idx = rng.randint(0, len(cl), NPTS)       # independent fresh sample
        out[k, j] = cl[idx].astype(np.float32)
    if (k + 1) % 20 == 0:
        print(f"  {k+1}/{len(va)}", flush=True)

np.savez_compressed("scratch/val_multisample.npz", clouds=out,
                    coarse=coarse[va], true=true[va], R=Rm[va], c0=c0[va])
# how different are the samples? (mean nearest-point distance between two samples)
from scipy.spatial import cKDTree
dd = [cKDTree(out[k, 0]).query(out[k, 1])[0].mean() for k in range(min(10, len(va)))]
print(f"saved val_multisample.npz {out.shape} ({os.path.getsize('scratch/val_multisample.npz')/1e6:.1f} MB)")
print(f"mean nearest-point distance between two independent samples: {np.mean(dd):.3f} mm")
