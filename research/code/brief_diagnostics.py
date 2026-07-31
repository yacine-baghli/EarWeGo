"""
The three diagnostics from the brief that we had NOT yet run.

D1 (§4.7) COARSE DISTRIBUTION SHIFT — the deep model refines a coarse estimate from
   the v1 classical model. That classical model was FITTED ON TRAIN, so train ears get
   in-sample (optimistically good) coarse inputs while val/test ears get out-of-sample
   ones. If the gap is large, the refiner was trained on easier inputs than it sees at
   test time. This is a real, fixable bug (fix = out-of-fold coarse predictions).

D2 (§5.5) PROCRUSTES DIAGNOSTIC — rigidly align predictions to GT and recompute. A big
   drop means a global pose/scale/crop bias (systematic, fixable); a small drop means
   the error is genuinely local.

D3 (§6.4) EUCLIDEAN vs GEODESIC error — the pinna is folded, so a prediction can be
   close in 3D but on the WRONG FOLD. If geodesic error >> euclidean error, we have
   wrong-fold responses and geodesic supervision matters.
"""
import os, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.geometry import procrustes_align
from src.splits import get_split
from src.dataset import Dataset
from scipy.spatial import cKDTree
import scipy.sparse as sp
import scipy.sparse.csgraph as csg

MESH = "2026 Munich Tech Arena - Datas/2026 Munich Tech Arena - Datas/mesh"
LM = "2026 Munich Tech Arena - Datas/2026 Munich Tech Arena - Datas/landmarks"
MIRROR = np.array([1., -1., 1.])
CONT = [(0, 24), (25, 54), (55, 74), (75, 84)]
NM = ["Helix", "Antihelix", "Concha/outer", "Lobe"]

print("=" * 70)
print("D1  COARSE DISTRIBUTION SHIFT (train in-sample vs val out-of-sample)")
print("=" * 70)
d = np.load("scratch/deep_dataset.npz", allow_pickle=True)
coarse, true, split = d["coarse"], d["true"], d["split"]
etr = np.linalg.norm(coarse[split == "train"] - true[split == "train"], axis=2)
eva = np.linalg.norm(coarse[split == "val"] - true[split == "val"], axis=2)
print(f"  coarse (v1) error on TRAIN ears : {etr.mean():.3f} mm   (n={len(etr)})")
print(f"  coarse (v1) error on VAL   ears : {eva.mean():.3f} mm   (n={len(eva)})")
print(f"  SHIFT = {eva.mean()-etr.mean():+.3f} mm "
      f"({(eva.mean()/etr.mean()-1)*100:+.0f}%)")
print(f"  => {'REAL train/test mismatch: the refiner saw EASIER inputs than at test' if eva.mean() > etr.mean()*1.10 else 'no meaningful shift'}")

print()
print("=" * 70)
print("D2  PROCRUSTES DIAGNOSTIC (is there a global pose/scale bias?)")
print("=" * 70)
z = np.load("scratch/val_surfproj.npz")
pred = z["proj"].astype(float); gt = z["gt"].astype(float)
raw = np.linalg.norm(pred - gt, axis=2)
rig, sim = [], []
for i in range(len(gt)):
    a_r = procrustes_align(pred[i], gt[i], allow_scale=False)[0]
    a_s = procrustes_align(pred[i], gt[i], allow_scale=True)[0]
    rig.append(np.linalg.norm(a_r - gt[i], axis=1))
    sim.append(np.linalg.norm(a_s - gt[i], axis=1))
rig = np.array(rig); sim = np.array(sim)
print(f"  as-is                       : {raw.mean():.3f} mm")
print(f"  after rigid align (R,t)     : {rig.mean():.3f} mm  ({(1-rig.mean()/raw.mean())*100:.1f}% of error is global pose)")
print(f"  after similarity (R,t,s)    : {sim.mean():.3f} mm  ({(1-sim.mean()/raw.mean())*100:.1f}% is pose+scale)")
# per-ear scale bias: are we systematically too big/small?
scales = [procrustes_align(pred[i], gt[i], allow_scale=True)[1]["s"] for i in range(len(gt))]
print(f"  fitted scale: mean {np.mean(scales):.4f} (1.0 = unbiased), sd {np.std(scales):.4f}")
print(f"  => {'SYSTEMATIC pose/scale bias worth fixing' if rig.mean() < raw.mean()*0.85 else 'error is genuinely LOCAL, not a global transform bug'}")

print()
print("=" * 70)
print("D3  EUCLIDEAN vs GEODESIC error (are we on the WRONG FOLD?)")
print("=" * 70)
va = get_split("val", mesh_dir=Path(MESH))
ds = Dataset(MESH, LM); pid2idx = {p: i for i, p in enumerate(ds.subject_ids)}
order = [(p, s) for p in va for s in ("left", "right")]
N_CHECK = int(os.environ.get("N_CHECK", "10"))
geo_all, euc_all, per_cont = [], [], {n: [[], []] for n in NM}
for k in range(N_CHECK):
    pid, side = order[k]
    m = ds[pid2idx[pid]][0]
    V = np.asarray(m.vertices); F = np.asarray(m.faces)
    if side == "right":
        V = V * MIRROR
    # crop around the landmarks, build the surface graph
    lo, hi = gt[k].min(0) - 10, gt[k].max(0) + 10
    vin = np.all((V >= lo) & (V <= hi), axis=1)
    fm = vin[F].all(axis=1); Fs = F[fm]
    keep = np.unique(Fs); remap = -np.ones(len(V), int); remap[keep] = np.arange(len(keep))
    Vc = V[keep]; Fc = remap[Fs]
    e = np.vstack([Fc[:, [0, 1]], Fc[:, [1, 2]], Fc[:, [2, 0]]])
    w = np.linalg.norm(Vc[e[:, 0]] - Vc[e[:, 1]], axis=1)
    G = sp.coo_matrix((np.r_[w, w], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                      shape=(len(Vc), len(Vc))).tocsr()
    tree = cKDTree(Vc)
    gi = tree.query(gt[k])[1]; pi = tree.query(pred[k])[1]
    D = csg.dijkstra(G, indices=gi, min_only=False)          # (85, nv)
    for j in range(85):
        gd = D[j, pi[j]]
        ed = np.linalg.norm(pred[k][j] - gt[k][j])
        if np.isfinite(gd):
            geo_all.append(gd); euc_all.append(ed)
            for (a, b), nm in zip(CONT, NM):
                if a <= j <= b:
                    per_cont[nm][0].append(gd); per_cont[nm][1].append(ed)
geo_all = np.array(geo_all); euc_all = np.array(euc_all)
print(f"  on {N_CHECK} val ears, {len(geo_all)} landmarks with a connected path:")
print(f"  mean EUCLIDEAN error {euc_all.mean():.3f} mm | mean GEODESIC error {geo_all.mean():.3f} mm")
print(f"  ratio geodesic/euclidean = {geo_all.mean()/euc_all.mean():.2f}x")
print(f"  landmarks with ratio > 2 (likely WRONG FOLD): {(geo_all > 2*euc_all).mean()*100:.1f}%")
print(f"  landmarks with ratio > 3                    : {(geo_all > 3*euc_all).mean()*100:.1f}%")
for nm in NM:
    g, ee = np.array(per_cont[nm][0]), np.array(per_cont[nm][1])
    if len(g):
        print(f"    {nm:14s} euc {ee.mean():.3f}  geo {g.mean():.3f}  ratio {g.mean()/ee.mean():.2f}x"
              f"  >2x in {(g > 2*ee).mean()*100:.0f}%")
print(f"  => {'WRONG-FOLD errors are significant -> geodesic supervision matters' if (geo_all > 2*euc_all).mean() > 0.15 else 'errors are mostly ALONG the surface, not across folds'}")
