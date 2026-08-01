"""
CROSS-FAMILY ENSEMBLE + SURFACE PROJECTION -- the end-to-end honest number.

Members are three architecturally unrelated backbones, each with out-of-fold predictions
over all 340 development ears from the frozen folds:
  dgcnn3   the adopted corrected-normals model, 3-seed prediction ensemble
  kpconv   kernel-point convolution, 8192 face-sampled points
  ptv3     serialized-attention transformer, 8192 face-sampled points

Equal weights, deliberately. Fitted weights are the next step and belong in
ensemble_oof.py with nested-OOF disjointness; equal weighting needs no fitting at all,
so this number carries no selection optimism.

The dense-SSM blend is omitted: measured at +0.0019mm on the corrected-normals model, it
became harmful once normals fixed the across-contour and off-surface error it used to
supply (research/results/full_pipeline_normalsfix.json).

    python research/code/ensemble_final.py
Writes research/results/ensemble_final.json
"""
import os, sys, json
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "deep_model"))
from surfproj import SurfaceProjector

W = "scratch"
of = np.load(f"{W}/ortho_feats.npz")
GT, SUBJ = of["gt"].astype(float), of["subj"]
NE = len(GT)
md = np.load(f"{W}/mesh_data.npz")
V, F, VP, FP, R, C0 = (md["verts"], md["faces"], md["v_ptr"], md["f_ptr"],
                       md["R"].astype(float), md["c0"].astype(float))
CONT = [(0, 24, "outer_helix"), (25, 54, "concha"),
        (55, 74, "inner_helix"), (75, 84, "sup._antihelix")]


def assemble(get):
    P = np.full((NE, 85, 3), np.nan)
    for f in range(5):
        pred, idx = get(f)
        P[idx] = pred
    assert not np.isnan(P).any(), "an ear was never held out"
    return P


def dgcnn(f):
    idx = np.array(json.load(open(f"{W}/screen_normalsfix_s0_f{f}.json"))["val_ear_index"])
    return np.mean([np.load(f"{W}/screen_normalsfix_s{s}_f{f}.npy") for s in (0, 1, 2)],
                   0).astype(float), idx


def famA(tag):
    def g(f):
        idx = np.array(json.load(open(f"{W}/famA_{tag}_f{f}.json"))["val_ear_index"])
        return np.load(f"{W}/famA_{tag}_f{f}.npy").astype(float), idx
    return g


M = {"dgcnn3": assemble(dgcnn), "kpconv": assemble(famA("kpconv")),
     "ptv3": assemble(famA("ptv3"))}
mle = lambda P: float(np.linalg.norm(P - GT, axis=2).mean())
per_ear = lambda P: np.linalg.norm(P - GT, axis=2).mean(1)

ENS = np.mean(list(M.values()), 0)
PROJ = ENS.copy()
worst = 0.0
for i in range(NE):
    v = V[VP[i]:VP[i + 1]].astype(float) @ R[i] + C0[i]      # canonical -> prediction frame
    f_ = F[FP[i]:FP[i + 1]].astype(np.int64) - VP[i]
    worst = max(worst, float(np.median(cKDTree(v).query(GT[i])[0])))
    PROJ[i] = SurfaceProjector(v, f_).project(ENS[i])[0]
    if (i + 1) % 80 == 0:
        print(f"  projected {i+1}/{NE}", flush=True)
assert worst < 2.0, f"frame mismatch: worst GT-to-vertex median {worst:.2f}mm"

stages = {k: round(mle(v), 4) for k, v in M.items()}
stages["ensemble_equal_weight"] = round(mle(ENS), 4)
stages["plus_surface_projection"] = round(mle(PROJ), 4)
print(f"\nframe check worst {worst:.3f}mm")
for k, v in stages.items():
    print(f"  {k:26s} {v:.4f}")

us = np.unique(SUBJ)
per = {s: np.where(SUBJ == s)[0] for s in us}
rng = np.random.RandomState(23)
out = {"members": list(M), "n_ears": NE, "stages_mm": stages,
       "note": ("equal weights, no fitting, so no selection optimism; dense-SSM blend "
                "omitted because it measured +0.0019mm once normals were corrected")}
for base, nm in ((M["dgcnn3"], "dgcnn3"), (ENS, "ensemble")):
    d = per_ear(PROJ) - per_ear(base)
    dr = np.array([d[np.concatenate([per[s] for s in rng.choice(us, len(us), True)])].mean()
                   for _ in range(20000)])
    lo, hi = np.percentile(dr, [2.5, 97.5])
    out[f"final_vs_{nm}"] = {"delta_mm": round(float(d.mean()), 4),
                             "ci95": [round(float(lo), 4), round(float(hi), 4)],
                             "p_negative": round(float((dr < 0).mean()), 4)}
    print(f"  final vs {nm:9s} {d.mean():+.4f} mm  CI [{lo:+.4f}, {hi:+.4f}]  "
          f"P(<0)={(dr < 0).mean():.3f}")

dd = np.linalg.norm(PROJ - GT, axis=2)
out["per_contour_mm"] = {nm: round(float(dd[:, lo:hi + 1].mean()), 4) for lo, hi, nm in CONT}
out["median_mm"] = round(float(np.median(dd)), 4)
out["p90_mm"] = round(float(np.percentile(dd, 90)), 4)
print(f"  per-contour {out['per_contour_mm']}")
print(f"  median {out['median_mm']}  p90 {out['p90_mm']}")
json.dump(out, open("research/results/ensemble_final.json", "w"), indent=1)
print("wrote research/results/ensemble_final.json")
