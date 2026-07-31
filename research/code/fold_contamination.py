"""
How fold-contaminated is the soft-argmax attention window?

The model attends to the K=48 EUCLIDEAN nearest cloud points around each query and
outputs a convex combination of them. On a folded surface those neighbours can lie on
the far side of a rim/crease, so the expectation is pulled off the correct fold.

Measures, for each GT landmark on the 2048-pt cloud the model actually uses:
  * fraction of the K=48 window that is geodesically far (>2x euclidean) = contaminated
  * the error floor if the window were restricted to geodesically-consistent points
  * whether a cheap NORMAL-AGREEMENT test can identify the contaminated neighbours
    (a shippable proxy for geodesic locality)
"""
import os, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.splits import get_split
from src.dataset import Dataset
from scipy.spatial import cKDTree
import scipy.sparse as sp
import scipy.sparse.csgraph as csg

MESH = "2026 Munich Tech Arena - Datas/2026 Munich Tech Arena - Datas/mesh"
LM = "2026 Munich Tech Arena - Datas/2026 Munich Tech Arena - Datas/landmarks"
MIRROR = np.array([1., -1., 1.])
K = 48
N_EAR = int(os.environ.get("N_EAR", "8"))

d = np.load("scratch/deep_dataset.npz", allow_pickle=True)
clouds, true, Rm, c0, split = d["clouds"], d["true"], d["R"], d["c0"], d["split"]
va = np.where(split == "val")[0]
vpids = get_split("val", mesh_dir=Path(MESH))
ds = Dataset(MESH, LM); pid2idx = {p: i for i, p in enumerate(ds.subject_ids)}
order = [(p, s) for p in vpids for s in ("left", "right")]

frac_bad, floor_gain, normal_auc = [], [], []
for k in range(N_EAR):
    i = va[k]; pid, side = order[k]
    m = ds[pid2idx[pid]][0]
    V = np.asarray(m.vertices); F = np.asarray(m.faces)
    if side == "right":
        V = V * MIRROR
    R, cc = Rm[i], c0[i]
    cloud_w = clouds[i] @ R + cc                    # the 2048 pts the model uses (world)
    gt_w = true[i] @ R + cc
    # surface graph on a crop of the full-res mesh
    lo, hi = gt_w.min(0) - 10, gt_w.max(0) + 10
    vin = np.all((V >= lo) & (V <= hi), axis=1)
    fm = vin[F].all(axis=1); Fs = F[fm]
    keep = np.unique(Fs); remap = -np.ones(len(V), int); remap[keep] = np.arange(len(keep))
    Vc = V[keep]; Fc = remap[Fs]
    e = np.vstack([Fc[:, [0, 1]], Fc[:, [1, 2]], Fc[:, [2, 0]]])
    w = np.linalg.norm(Vc[e[:, 0]] - Vc[e[:, 1]], axis=1)
    G = sp.coo_matrix((np.r_[w, w], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                      shape=(len(Vc), len(Vc))).tocsr()
    # per-vertex normals (area-weighted) for the normal-agreement proxy
    fn = np.cross(Vc[Fc[:, 1]] - Vc[Fc[:, 0]], Vc[Fc[:, 2]] - Vc[Fc[:, 0]])
    VN = np.zeros_like(Vc)
    for c in range(3):
        np.add.at(VN, Fc[:, c], fn)
    VN /= (np.linalg.norm(VN, axis=1, keepdims=True) + 1e-12)
    tree = cKDTree(Vc)
    ci = tree.query(cloud_w)[1]                    # cloud pt -> nearest crop vertex
    gi = tree.query(gt_w)[1]
    ctree = cKDTree(cloud_w)
    for j in range(85):
        nn = ctree.query(gt_w[j], k=K)[1]           # the model's attention window
        D = csg.dijkstra(G, indices=gi[j])          # geodesic from the GT landmark
        gd = D[ci[nn]]
        ed = np.linalg.norm(cloud_w[nn] - gt_w[j], axis=1)
        ok = np.isfinite(gd)
        bad = ok & (gd > 2.0 * np.maximum(ed, 0.3))  # geodesically far = other fold
        frac_bad.append(bad.mean())
        # normal agreement of the GT landmark's own normal vs each neighbour's
        ndot = VN[ci[nn]] @ VN[gi[j]]
        normal_auc.append(np.nan if bad.sum() in (0, len(bad)) else
                          ((ndot[~bad][:, None] > ndot[bad][None, :]).mean()))
        # error floor: best convex combo of GOOD vs ALL neighbours (centroid proxy)
        if bad.sum() < len(bad):
            all_c = cloud_w[nn].mean(0)
            good_c = cloud_w[nn][~bad].mean(0)
            floor_gain.append(np.linalg.norm(all_c - gt_w[j]) - np.linalg.norm(good_c - gt_w[j]))
frac_bad = np.array(frac_bad); floor_gain = np.array(floor_gain)
na = np.array(normal_auc, float); na = na[~np.isnan(na)]
print(f"on {N_EAR} val ears ({len(frac_bad)} landmark windows, K={K}):")
print(f"  mean fraction of the attention window on a DIFFERENT FOLD : {frac_bad.mean()*100:.1f}%")
print(f"  windows with >25% contamination                            : {(frac_bad>0.25).mean()*100:.1f}%")
print(f"  windows with >50% contamination                            : {(frac_bad>0.50).mean()*100:.1f}%")
print(f"  windows that are CLEAN (0%)                                : {(frac_bad==0).mean()*100:.1f}%")
print(f"  centroid error change if contaminated pts removed          : {floor_gain.mean():+.3f} mm"
      f"  (positive = removing them HELPS)")
print(f"  NORMAL-AGREEMENT separates good/bad neighbours: AUC {na.mean():.3f}"
      f"   ({'usable cheap proxy' if na.mean()>0.7 else 'weak proxy'})")
