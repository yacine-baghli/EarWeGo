"""
THE HONEST FULL-PIPELINE POOLED-OOF BASELINE.

Every OOF number quoted so far has been the raw network. The shipped pipeline also
applies exact point-to-triangle surface projection and a dense-SSM MAP blend, and those
were never measured out-of-fold, for a specific reason: deep_model/dense_ssm.npz was
fitted over all 280 ears of the ORIGINAL train split, which contains validation ears of
every CV fold. Using it in an OOF evaluation would fit the shape basis partly on the ears
being scored.

build_template.py already solves that -- it rebuilds the dense mean and PCA basis per
fold from that fold's TRAINING ears only. This script reuses those fold-safe bases, so
each stage is measured on ears the corresponding basis never saw.

Stages, each reported separately so the contribution of each is visible:
  1. single seed        (mean over seeds of the per-seed pooled OOF)
  2. seed ensemble      (mean PREDICTION over seeds -- the shippable quantity)
  3. + surface projection   exact point-to-triangle, onto the cropped submesh
  4. + dense-SSM blend      fold-safe basis, alpha from env

    SEEDS=0,1,2 TAG=base python research/code/full_pipeline_oof.py
Writes research/results/full_pipeline_<tag>.json
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "deep_model"))
from surfproj import SurfaceProjector
from dense_fit import DenseSSMFit

WORK = "scratch"
TAG = os.environ.get("TAG", "base")
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
ALPHA = float(os.environ.get("SSM_ALPHA", "0.3"))
KUSE = int(os.environ.get("SSM_KUSE", "120"))
CONT = [(0, 24, "outer_helix"), (25, 54, "concha"),
        (55, 74, "inner_helix"), (75, 84, "sup._antihelix")]

of = np.load(f"{WORK}/ortho_feats.npz")
GT, SUBJ = of["gt"].astype(np.float64), of["subj"]
NE = len(GT)
md = np.load(f"{WORK}/mesh_data.npz")
V_ALL, F_ALL, VP, FP = md["verts"], md["faces"], md["v_ptr"], md["f_ptr"]
R_ALL, C0_ALL = md["R"].astype(np.float64), md["c0"].astype(np.float64)


def ear_mesh(i):
    """Cropped submesh in the PREDICTION frame.

    mesh_data.npz stores geometry coarse-centred (canonical): world = v @ R + c0.
    Predictions and ortho_feats['gt'] live in the mirrored-world frame. Getting this
    backwards silently produced a 73.5mm 'result' rather than an error, so the frame is
    now asserted against the ground truth every time, not assumed.
    """
    v = V_ALL[VP[i]:VP[i + 1]].astype(np.float64) @ R_ALL[i] + C0_ALL[i]
    f_ = F_ALL[FP[i]:FP[i + 1]].astype(np.int64) - VP[i]        # global -> local indices
    assert f_.min() >= 0 and f_.max() < len(v), f"ear {i}: face index out of range"
    return v, f_

# ---- assemble the OOF prediction per seed, then the seed ensemble
P = np.full((len(SEEDS), NE, 85, 3), np.nan)
fold = np.full(NE, -1)
for si, s in enumerate(SEEDS):
    for f in range(5):
        p = f"{WORK}/screen_{TAG}_s{s}_f{f}.json"
        assert os.path.exists(p) and os.path.getsize(p) > 0, f"missing {p}"
        idx = np.array(json.load(open(p))["val_ear_index"])
        P[si, idx] = np.load(p[:-5] + ".npy")
        fold[idx] = f
assert not np.isnan(P).any(), "some ear was never held out by some seed"
assert (fold >= 0).all()

ENS = P.mean(0)


def mle(X):
    return float(np.linalg.norm(X - GT, axis=2).mean())


def per_ear(X):
    return np.linalg.norm(X - GT, axis=2).mean(1)


stage = {"single_seed_mean": float(np.mean([mle(P[i]) for i in range(len(SEEDS))])),
         "seed_ensemble": mle(ENS)}
print(f"1. single seed (mean of {len(SEEDS)})   {stage['single_seed_mean']:.4f}")
print(f"2. seed ensemble                  {stage['seed_ensemble']:.4f}"
      f"   ({stage['seed_ensemble']-stage['single_seed_mean']:+.4f})")

# ---- 3. exact surface projection onto the cropped submesh (canonical frame, as stored)
from scipy.spatial import cKDTree
PROJ = ENS.copy()
frame_chk = []
for i in range(NE):
    v, f_ = ear_mesh(i)
    frame_chk.append(float(np.median(cKDTree(v).query(GT[i])[0])))
    PROJ[i] = SurfaceProjector(v, f_).project(ENS[i])[0]
    if (i + 1) % 60 == 0:
        print(f"   projected {i+1}/{NE}", flush=True)
fc = float(np.max(frame_chk))
assert fc < 2.0, (f"FRAME MISMATCH: worst ear has GT {fc:.2f}mm from its nearest mesh "
                 f"vertex; the mesh is not in the prediction frame")
print(f"   frame check: GT->nearest-vertex median per ear, worst {fc:.3f}mm (< 2.0)")
stage["plus_surface_projection"] = mle(PROJ)
print(f"3. + surface projection           {stage['plus_surface_projection']:.4f}"
      f"   ({stage['plus_surface_projection']-stage['seed_ensemble']:+.4f})")

# ---- 4. dense-SSM blend with a FOLD-SAFE basis (built from training ears only)
BLEND = PROJ.copy()
fits = {}
for f in range(5):
    z = np.load(f"{WORK}/template_f{f}.npz")
    assert int(z["fold"]) == f, "template fold label mismatch"
    tmp = f"{WORK}/_ssm_f{f}.npz"
    np.savez(tmp, mean=z["mean_V"].reshape(-1), comps=z["comps"], eig=z["eig"],
             template_F=z["template_F"], bary_f=z["bary_f"], bary_w=z["bary_w"])
    fits[f] = DenseSSMFit(tmp, kuse=KUSE)
for i in range(NE):
    v, _ = ear_mesh(i)
    try:
        lm = fits[fold[i]].fit(v, PROJ[i])            # (cloud, deep_lms) -> fitted (85,3)
        BLEND[i] = (1 - ALPHA) * PROJ[i] + ALPHA * lm
    except Exception as e:                     # a fit that fails must not silently pass through
        print(f"   ! ear {i} SSM fit failed ({type(e).__name__}: {e}); left unblended")
    if (i + 1) % 60 == 0:
        print(f"   blended {i+1}/{NE}", flush=True)
stage["plus_dense_ssm_blend"] = mle(BLEND)
print(f"4. + dense-SSM blend (fold-safe)  {stage['plus_dense_ssm_blend']:.4f}"
      f"   ({stage['plus_dense_ssm_blend']-stage['plus_surface_projection']:+.4f})")

d = np.linalg.norm(BLEND - GT, axis=2)
pc = {nm: round(float(d[:, lo:hi + 1].mean()), 4) for lo, hi, nm in CONT}
print(f"\nFINAL full-pipeline pooled OOF ({NE} ears): {mle(BLEND):.4f} mm")
print(f"per-contour {pc}")

# paired bootstrap of the whole pipeline against the single-seed raw network
us = np.unique(SUBJ)
per = {s: np.where(SUBJ == s)[0] for s in us}
diff = per_ear(BLEND) - per_ear(P[0])
rng = np.random.RandomState(9)
dr = np.array([diff[np.concatenate([per[s] for s in rng.choice(us, len(us), True)])].mean()
               for _ in range(20000)])
lo, hi = np.percentile(dr, [2.5, 97.5])
print(f"vs seed-0 raw network: {diff.mean():+.4f} mm  CI [{lo:+.4f}, {hi:+.4f}]")

json.dump({"tag": TAG, "seeds": SEEDS, "n_ears": int(NE), "ssm_alpha": ALPHA,
           "ssm_kuse": KUSE, "stages_mm": {k: round(v, 4) for k, v in stage.items()},
           "final_per_contour_mm": pc,
           "vs_seed0_raw": {"delta_mm": round(float(diff.mean()), 4),
                            "ci95": [round(float(lo), 4), round(float(hi), 4)]},
           "note": ("dense-SSM basis is rebuilt per fold from that fold's TRAINING ears "
                    "only (build_template.py), so it never saw the ears it scores. The "
                    "shipped deep_model/dense_ssm.npz is NOT used -- it was fitted over "
                    "all 280 original-train ears, which include CV validation ears.")},
          open(f"research/results/full_pipeline_{TAG}.json", "w"), indent=1)
print(f"wrote research/results/full_pipeline_{TAG}.json")
