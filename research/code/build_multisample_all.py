"""
Fresh-sample clouds for ALL 340 ears (train+val), so the CV fold models can produce
OOF predictions through the EXACT final pipeline (fresh-sample TTA included).

Output: scratch/all_multisample.npz  clouds (340, M, 2048, 3)
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
NPTS, MARGIN = 2048, 14.0
M = int(os.environ.get("M", "4"))

d = np.load("scratch/deep_dataset.npz", allow_pickle=True)
coarse, true, Rm, c0, split = d["coarse"], d["true"], d["R"], d["c0"], d["split"]
tr = get_split("train", mesh_dir=Path(MESH)); va = get_split("val", mesh_dir=Path(MESH))
order = [(p, s) for p in tr for s in ("left", "right")] + \
        [(p, s) for p in va for s in ("left", "right")]
assert len(order) == len(split)
ds = Dataset(MESH, LM); pid2idx = {p: i for i, p in enumerate(ds.subject_ids)}
cache = {}
out = np.zeros((len(order), M, NPTS, 3), np.float32)
for i, (pid, side) in enumerate(order):
    if pid not in cache:
        cache[pid] = np.asarray(ds[pid2idx[pid]][0].vertices)
    V = cache[pid]
    if side == "right":
        V = V * MIRROR
    R, cc = Rm[i], c0[i]
    cw = coarse[i] @ R + cc
    lo, hi = cw.min(0) - MARGIN, cw.max(0) + MARGIN
    msk = np.all((V >= lo) & (V <= hi), axis=1)
    cl = ((V[msk] if msk.any() else V) - cc) @ R.T
    for j in range(M):
        rng = np.random.RandomState(1000 + 97 * i + j)
        out[i, j] = cl[rng.randint(0, len(cl), NPTS)].astype(np.float32)
    if (i + 1) % 60 == 0:
        print(f"  {i+1}/{len(order)}", flush=True)
np.savez_compressed("scratch/all_multisample.npz", clouds=out, coarse=coarse,
                    true=true, R=Rm, c0=c0, split=split)
print(f"saved all_multisample.npz {out.shape} "
      f"({os.path.getsize('scratch/all_multisample.npz')/1e6:.1f} MB)")
