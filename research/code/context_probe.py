"""
FULL-HEAD / BILATERAL CONTEXT PROBE.

Purpose: break the confound. The left/right correlations show a shared per-subject factor
in the oracle corrections, but subject, scan session, preprocessing and annotation session
are mutually confounded. If that factor is a SCAN / ALIGNMENT effect it should be
predictable from observable global head, mesh-quality, crop and alignment features. If it
is annotation-only, those features carry no signal.

Features (all observable at test time, NO ground truth, NO subject ID):
  head geometry     bounding-box dimensions, PCA axis lengths + anisotropy
  mesh quality      vertex/face counts, edge-length mean/sd/percentiles, per-ear crop
                    vertex density, nearest-neighbour spacing
  crop / alignment  coarse-to-SSM-mean Procrustes scale, rotation angle, residual;
                    crop extent and orientation (frame axes as angles)
  bilateral         left/right coarse asymmetry (Procrustes distance between the two
                    coarse shapes after mirroring), and BOTH ears' coarse shape embeddings
                    (so an ear can be informed by its partner)

Targets: the oracle per-contour affine offset and stretch, and the mean across-contour
offset. Subject-grouped 5-fold CV (the frozen folds). Ridge + gradient boosting.

    python research/code/context_probe.py
Writes research/results/context_probe.json
"""
import os, sys, json
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.splits import get_split
from src.dataset import Dataset
from src.geometry import procrustes_align

MESH = "2026 Munich Tech Arena - Datas/2026 Munich Tech Arena - Datas/mesh"
LM = "2026 Munich Tech Arena - Datas/2026 Munich Tech Arena - Datas/landmarks"
MIRROR = np.array([1., -1., 1.])
RNG_C = {"outer_helix": (0, 24), "concha": (25, 54),
         "inner_helix": (55, 74), "sup._antihelix": (75, 84)}

oof = np.load("scratch/oof_final.npz")
BASE = oof["pred"].astype(np.float64); GT = oof["gt"].astype(np.float64); fold = oof["fold"]
par = np.load("scratch/oracles_v2_params.npz")
ofe = np.load("scratch/ortho_feats.npz")
bdir = ofe["b"].astype(np.float64)
NE = len(BASE)
dd = np.load("scratch/deep_dataset.npz", allow_pickle=True)
coarse, Rm, c0 = dd["coarse"], dd["R"], dd["c0"]
ssm = np.load("deep_model/ssm.npz")
mean_shape = ssm["ssm_mean"].reshape(85, 3)

tr = get_split("train", mesh_dir=Path(MESH)); va = get_split("val", mesh_dir=Path(MESH))
order = [(p, s) for p in tr for s in ("left", "right")] + \
        [(p, s) for p in va for s in ("left", "right")]
ds = Dataset(MESH, LM); pid2idx = {q: i for i, q in enumerate(ds.subject_ids)}

# ---------------- per-ear observable features ----------------
head_cache = {}
rows = []
for i, (pid, side) in enumerate(order):
    if pid not in head_cache:
        m = ds[pid2idx[pid]][0]
        V = np.asarray(m.vertices); F = np.asarray(m.faces)
        # global head descriptors (side-independent)
        ext = V.max(0) - V.min(0)
        Vc = V - V.mean(0)
        ev = np.linalg.eigvalsh(Vc.T @ Vc / len(Vc))[::-1]
        el = np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1)
        sub = V[np.random.RandomState(0).choice(len(V), min(20000, len(V)), replace=False)]
        sp = cKDTree(sub).query(sub, k=2)[0][:, 1]
        head_cache[pid] = dict(V=V, F=F, ext=ext, ev=ev,
                               nv=len(V), nf=len(F),
                               el=[el.mean(), el.std(), np.percentile(el, 90)],
                               sp=[sp.mean(), sp.std()])
        if len(head_cache) > 20:
            head_cache.pop(next(iter(head_cache)))
    H = head_cache[pid]
    V = H["V"] * (MIRROR if side == "right" else 1.0)
    R, cc = Rm[i], c0[i]
    cw = coarse[i] @ R + cc
    lo, hi = cw.min(0) - 14, cw.max(0) + 14
    msk = np.all((V >= lo) & (V <= hi), axis=1)
    crop = V[msk] if msk.any() else V
    ctree = cKDTree(crop[np.random.RandomState(1).choice(len(crop), min(4000, len(crop)), replace=False)])
    csp = ctree.query(ctree.data, k=2)[0][:, 1]
    # coarse -> SSM-mean alignment descriptors
    _, tf = procrustes_align(mean_shape, cw, allow_scale=True)
    ang = np.arccos(np.clip((np.trace(tf["R"]) - 1) / 2, -1, 1))
    resid = np.linalg.norm(procrustes_align(mean_shape, cw, allow_scale=True)[0] - cw, axis=1).mean()
    f = [*H["ext"], *np.sqrt(np.maximum(H["ev"], 0)), H["ev"][0] / max(H["ev"][2], 1e-9),
         H["nv"] / 1e5, H["nf"] / 1e5, *H["el"], *H["sp"],
         len(crop) / 1e4, csp.mean(), csp.std(),
         tf["s"], ang, resid,
         *(cw.max(0) - cw.min(0)),
         *R[0], *R[2],                                    # crop frame orientation
         1.0 if side == "right" else 0.0]
    rows.append(f)
    if (i + 1) % 40 == 0:
        print(f"  {i+1}/{NE}", flush=True)
Xown = np.array(rows, float)

# bilateral: partner features + left/right coarse asymmetry + both coarse embeddings
partner = np.array([i + 1 if i % 2 == 0 else i - 1 for i in range(NE)])
asym = np.zeros(NE)
emb = np.zeros((NE, 30))
for i in range(NE):
    A = coarse[i]; B = coarse[partner[i]]
    asym[i] = np.linalg.norm(procrustes_align(B, A, allow_scale=True)[0] - A, axis=1).mean()
    c = A.mean(0); sc = np.linalg.norm(A - c) / np.sqrt(85)
    q = ((A - c) / sc).ravel()
    emb[i] = q[:30]
X = np.hstack([Xown, Xown[partner], asym[:, None], emb, emb[partner]])
print(f"features: own {Xown.shape[1]}, + partner, + asymmetry, + both embeddings -> {X.shape}")

# ---------------- can they predict the oracle corrections? ----------------
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor
targets = {}
for k, (lo, hi) in RNG_C.items():
    p = par[k]
    targets[f"{k}/offset"] = p[:, 1]
    targets[f"{k}/stretch"] = p[:, 0]
    targets[f"{k}/across"] = (((BASE - GT) * bdir).sum(-1))[:, lo:hi + 1].mean(1)

out = {"note": ("Do observable global head / mesh-quality / crop / alignment / bilateral "
                "features predict the oracle corrections? Subject-grouped 5-fold CV, no "
                "subject ID. Signal => the shared per-subject factor is at least partly a "
                "SCAN/ALIGNMENT effect (and exploitable). No signal => consistent with an "
                "annotation-only factor, though it cannot prove it."),
       "n_features": int(X.shape[1]), "targets": {}}
print(f"\n{'target':26s} {'sd(y)':>7s} {'ridge R2':>9s} {'GBM R2':>8s}  verdict")
for nm, y in targets.items():
    pr_r = np.zeros(NE); pr_g = np.zeros(NE)
    for f_ in range(5):
        te = fold == f_; tr_ = ~te
        mu, sd = X[tr_].mean(0), X[tr_].std(0) + 1e-9
        pr_r[te] = RidgeCV(alphas=np.logspace(-2, 5, 16)).fit((X[tr_] - mu) / sd, y[tr_]).predict((X[te] - mu) / sd)
        pr_g[te] = HistGradientBoostingRegressor(max_depth=3, max_iter=200,
                                                 learning_rate=0.05, l2_regularization=1.0
                                                 ).fit(X[tr_], y[tr_]).predict(X[te])
    def r2(p):
        return 1 - ((y - p) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
    rr, rg = r2(pr_r), r2(pr_g)
    v = "SIGNAL" if max(rr, rg) > 0.05 else "none"
    print(f"{nm:26s} {y.std():7.3f} {rr:9.3f} {rg:8.3f}  {v}")
    out["targets"][nm] = {"sd": round(float(y.std()), 4), "ridge_R2_oof": round(float(rr), 4),
                          "gbm_R2_oof": round(float(rg), 4),
                          "corr_ridge": round(float(np.corrcoef(y, pr_r)[0, 1]), 4),
                          "verdict": v}
best = max(max(v["ridge_R2_oof"], v["gbm_R2_oof"]) for v in out["targets"].values())
out["max_R2_across_targets"] = round(float(best), 4)
out["conclusion"] = ("scan/alignment context carries predictive signal" if best > 0.05 else
                     "no exploitable scan/alignment signal found in these features")
json.dump(out, open("research/results/context_probe.json", "w"), indent=1)
print(f"\nmax OOF R2 across targets: {best:.3f} -> {out['conclusion']}")
print("wrote research/results/context_probe.json")
