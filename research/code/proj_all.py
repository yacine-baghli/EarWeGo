"""Surface-project all 340 OOF predictions (train+val ear order)."""
import os, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.splits import get_split
from src.dataset import Dataset
from surfproj import SurfaceProjector

MESH = "2026 Munich Tech Arena - Datas/2026 Munich Tech Arena - Datas/mesh"
LM = "2026 Munich Tech Arena - Datas/2026 Munich Tech Arena - Datas/landmarks"
MIRROR = np.array([1., -1., 1.])

z = np.load("scratch/oof_tta.npz")
P = z["pred"].astype(float); gt = z["gt"].astype(float); fold = z["fold"]
tr = get_split("train", mesh_dir=Path(MESH)); va = get_split("val", mesh_dir=Path(MESH))
order = [(p, s) for p in tr for s in ("left", "right")] + \
        [(p, s) for p in va for s in ("left", "right")]
assert len(order) == len(P)
ds = Dataset(MESH, LM); pid2idx = {p: i for i, p in enumerate(ds.subject_ids)}
out = P.copy(); cache = {}
for i, (pid, side) in enumerate(order):
    if pid not in cache:
        m = ds[pid2idx[pid]][0]
        cache[pid] = (np.asarray(m.vertices), np.asarray(m.faces))
        if len(cache) > 40:
            cache.pop(next(iter(cache)))
    V, F = cache[pid]
    V = V * MIRROR if side == "right" else V
    lo, hi = P[i].min(0) - 8, P[i].max(0) + 8
    vin = np.all((V >= lo) & (V <= hi), axis=1)
    fm = vin[F].any(axis=1); Fs = F[fm]
    if len(Fs) == 0:
        continue
    keep = np.unique(Fs); remap = -np.ones(len(V), int); remap[keep] = np.arange(len(keep))
    out[i] = SurfaceProjector(V[keep], remap[Fs]).project(P[i])[0]
    if (i + 1) % 40 == 0:
        print(f"  {i+1}/{len(order)}", flush=True)
print(f"before {np.linalg.norm(P-gt,axis=2).mean():.4f}  after-proj "
      f"{np.linalg.norm(out-gt,axis=2).mean():.4f}")
np.savez("scratch/oof_final.npz", pred=out, gt=gt, fold=fold)
print("saved scratch/oof_final.npz")
