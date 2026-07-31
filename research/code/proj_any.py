"""Surface-project an arbitrary (60,85,3) val prediction array. Usage:
    python scratch/proj_any.py IN.npy OUT.npy   (or IN.npz with key 'pred')"""
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

src = sys.argv[1]
P = np.load(src)
if src.endswith(".npz"):
    P = P["pred"]
P = P.astype(float)
gt = np.load("scratch/val_surfproj.npz")["gt"].astype(float)

va = get_split("val", mesh_dir=Path(MESH))
ds = Dataset(MESH, LM); pid2idx = {p: i for i, p in enumerate(ds.subject_ids)}
order = [(p, s) for p in va for s in ("left", "right")]
out = P.copy(); cache = {}
for i, (p, side) in enumerate(order):
    if p not in cache:
        m = ds[pid2idx[p]][0]
        cache[p] = (np.asarray(m.vertices), np.asarray(m.faces))
    V, F = cache[p]
    V = V * MIRROR if side == "right" else V
    lo, hi = P[i].min(0) - 8, P[i].max(0) + 8
    vin = np.all((V >= lo) & (V <= hi), axis=1)
    fm = vin[F].any(axis=1); Fs = F[fm]
    keep = np.unique(Fs); remap = -np.ones(len(V), int); remap[keep] = np.arange(len(keep))
    out[i] = SurfaceProjector(V[keep], remap[Fs]).project(P[i])[0]
    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{len(order)}", flush=True)
print(f"before {np.linalg.norm(P-gt,axis=2).mean():.4f}mm  after-proj "
      f"{np.linalg.norm(out-gt,axis=2).mean():.4f}mm")
np.save(sys.argv[2], out)
print("saved", sys.argv[2])
