"""
WHAT THE PROFILE DECODER'S PLACEMENT OPERATOR DOES TO REAL PREDICTIONS.

fam_profile.py trains a model end to end with the placement in the graph. This measures
the operator ALONE, applied post hoc to the best existing out-of-fold predictions (the
equal-weight dgcnn3 + kpconv + ptv3 ensemble that ensemble_final.py scores), so the
structural claim is checked on real curves before a GPU run is spent on it.

READ THE RESULT AS AN INDICATION, NOT AS THE FAMILY'S SCORE. These predictions were
trained with a free-XYZ ordered-MSE objective; a model trained WITH the placement can
put its curve somewhere else. The post-hoc number is the answer to "is the population
profile a better phase than the one the ensemble predicts, on the curve the ensemble
already draws", which is the premise the family rests on.

LEAKAGE. Fold f's validation ears are scored with the profile from scratch/profile_f<f>.npz,
which build_profile.py computed from fold f's TRAINING ears; the mask is re-asserted here
against the very val_ear_index the predictions were written with. The rows marked LEAKY
use GT endpoints and are upper bounds, labelled as such, never a shippable number.

    python research/code/profile_apply.py
Writes research/results/profile_apply.json
"""
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fam_profile import CONTOURS, CNAMES, NC, arc_profile, place_on_polyline, reposition

W = os.environ.get("WORK", "scratch")
GT = np.load(f"{W}/ortho_feats.npz")["gt"].astype(np.float64)
NE = len(GT)
PROF = [int(x) for x in os.environ.get("PROFILE_CONTOURS", "2,3").split(",")]

P = np.full((NE, 85, 3), np.nan)
FOLD = np.full(NE, -1)
for f in range(5):
    idx = np.array(json.load(open(f"{W}/screen_normalsfix_s0_f{f}.json"))["val_ear_index"])
    m = [np.mean([np.load(f"{W}/screen_normalsfix_s{s}_f{f}.npy") for s in (0, 1, 2)], 0)]
    for t in ("kpconv", "ptv3"):
        j = np.array(json.load(open(f"{W}/famA_{t}_f{f}.json"))["val_ear_index"])
        assert (j == idx).all(), f"{t} fold {f} held out different ears"
        m.append(np.load(f"{W}/famA_{t}_f{f}.npy"))
    P[idx], FOLD[idx] = np.mean(m, 0).astype(np.float64), f
assert not np.isnan(P).any() and (FOLD >= 0).all(), "an ear was never held out"

prof, unif = {}, {}
for f in range(5):
    z = np.load(f"{W}/profile_f{f}.npz")
    assert int(z["fold"]) == f
    m = z["train_ear_mask"].astype(bool)
    assert not m[FOLD == f].any(), f"profile_f{f}.npz saw fold {f}'s validation ears -- LEAK"
    assert int(m.sum()) == int((FOLD != f).sum())
    prof[f] = [z[f"prof_c{ci}"] for ci in range(NC)]
print(f"{NE} OOF ears, 5 folds, profiles proven fold-safe against the prediction files")
for ci, (lo, hi) in enumerate(CONTOURS):
    unif[ci] = np.linspace(0, 1, hi - lo + 1)

T = lambda a: torch.tensor(a, dtype=torch.float64)
mle = lambda A, sl: float(np.linalg.norm(A[:, sl] - GT[:, sl], axis=2).mean())


def apply(which, source, ends):
    """source: 'fold' | 'uniform'.  ends: None (the prediction's own) | 'gt' (LEAKY)."""
    out = P.copy()
    for ci in which:
        lo, hi = CONTOURS[ci]
        for f in range(5):
            k = FOLD == f
            C = T(P[k, lo:hi + 1])
            if ends == "gt":
                C = reposition(C, T(GT[k, lo]), T(GT[k, hi]), "similarity")
            s = T((prof[f][ci] if source == "fold" else unif[ci]))[None].expand(len(C), -1)
            out[k, lo:hi + 1] = place_on_polyline(C, s).numpy()
    return out


def repos_only(which):
    out = P.copy()
    for ci in which:
        lo, hi = CONTOURS[ci]
        out[:, lo:hi + 1] = reposition(T(P[:, lo:hi + 1]), T(GT[:, lo]), T(GT[:, hi]),
                                       "similarity").numpy()
    return out


def subst_ends(which):
    """Only the 2 endpoint LANDMARKS of each contour become GT; the interior is untouched.
    Separates 'those 4 landmarks are wrong' from 'the whole curve is displaced'."""
    out = P.copy()
    for ci in which:
        out[:, list(CONTOURS[ci])] = GT[:, list(CONTOURS[ci])]
    return out


ALL = list(range(NC))
runs = [("baseline (equal-weight ensemble, no projection)", P, ""),
        ("fold-mean profile, own endpoints", apply(PROF, "fold", None), ""),
        ("uniform profile, own endpoints", apply(PROF, "uniform", None), ""),
        ("fold-mean profile on ALL FOUR contours", apply(ALL, "fold", None), ""),
        ("GT endpoint landmarks only, interior untouched", subst_ends(PROF), "LEAKY"),
        ("GT endpoints, predicted phase (reposition only)", repos_only(PROF), "LEAKY"),
        ("GT endpoints + fold-mean profile", apply(PROF, "fold", "gt"), "LEAKY"),
        ("GT endpoints + uniform profile", apply(PROF, "uniform", "gt"), "LEAKY")]

sel = np.concatenate([np.arange(CONTOURS[ci][0], CONTOURS[ci][1] + 1) for ci in PROF])
base = np.linalg.norm(P - GT, axis=2)
out = {"members": ["dgcnn3", "kpconv", "ptv3"], "n_ears": int(NE),
       "profile_contours": [CNAMES[ci] for ci in PROF],
       "caveat": ("post-hoc operator on predictions trained WITHOUT it; indicative of the "
                  "premise, not of the family's score. No surface projection is applied, so "
                  "the baseline row is above the shipped 1.1897mm."),
       "rows": []}
hdr = f"{'':52s}{'all 85':>9s}{'the 30':>9s}" + "".join(f"{CNAMES[ci][:9]:>11s}" for ci in ALL)
print("\n" + hdr)
for nm, A, tag in runs:
    r = {"name": nm, "leaky": tag == "LEAKY", "all85_mm": round(mle(A, slice(None)), 4),
         "profile_landmarks_mm": round(float(np.linalg.norm(A[:, sel] - GT[:, sel], axis=2).mean()), 4),
         "per_contour_mm": {CNAMES[ci]: round(mle(A, slice(CONTOURS[ci][0], CONTOURS[ci][1] + 1)), 4)
                            for ci in ALL}}
    r["delta_all85_mm"] = round(r["all85_mm"] - mle(P, slice(None)), 4)
    out["rows"].append(r)
    print(f"{(tag + ' ' + nm).strip():52s}{r['all85_mm']:9.4f}{r['profile_landmarks_mm']:9.4f}"
          + "".join(f"{v:11.4f}" for v in r["per_contour_mm"].values()))

# per-subject bootstrap of the one row that could actually ship
d = np.linalg.norm(apply(PROF, "fold", None) - GT, axis=2).mean(1) - base.mean(1)
subj = np.arange(NE) // 2
us = np.unique(subj)
per = {s: np.where(subj == s)[0] for s in us}
rng = np.random.RandomState(23)
dr = np.array([d[np.concatenate([per[s] for s in rng.choice(us, len(us), True)])].mean()
               for _ in range(20000)])
lo, hi = np.percentile(dr, [2.5, 97.5])
out["fold_profile_vs_baseline"] = {"delta_mm": round(float(d.mean()), 4),
                                   "ci95": [round(float(lo), 4), round(float(hi), 4)],
                                   "p_negative": round(float((dr < 0).mean()), 4)}
print(f"\nfold-mean profile vs baseline: {d.mean():+.4f} mm  CI [{lo:+.4f}, {hi:+.4f}]  "
      f"P(<0)={(dr < 0).mean():.3f}")

# ---- how good must an endpoint predictor be? The oracle rows use PERFECT endpoints; this
# degrades them with isotropic noise and reads off where the placement stops paying. The
# answer is what fam_endpoint.py has to hit, so it is measured rather than guessed.
EP_L = [i for ci in PROF for i in CONTOURS[ci]]
ep_err = float(np.linalg.norm(P[:, EP_L] - GT[:, EP_L], axis=2).mean())
print(f"\nendpoint sensitivity (the ensemble's own endpoint error on these 4 landmarks is "
      f"{ep_err:.4f} mm)")
# UNITS. sigma is the PER-AXIS Gaussian sd; ep_err above, and every mm in the table, is a
# MEAN EUCLIDEAN NORM, and E||N(0,sigma^2 I_3)|| = 1.5958*sigma. Reading sigma directly
# against ep_err understates the tolerable endpoint error by that factor, so the realised
# mean norm is measured per draw and it is THAT column the break-even is taken on.
print(f"  {'sigma mm':>9s}{'mean |ep err| mm':>18s}{'reposition only':>17s}{'+ fold profile':>16s}"
      f"{'gain of profile':>17s}")
sweep = []
for sig in (0.0, 0.25, 0.5, 0.625, 0.75, 1.0, 1.5, 2.0):
    rs = np.random.RandomState(7)
    a = b_ = e_ = 0.0
    for rep in range(3):                       # average 3 noise draws, same draws for both
        E = GT + rs.randn(*GT.shape) * sig
        e_ += float(np.linalg.norm(E[:, EP_L] - GT[:, EP_L], axis=2).mean()) / 3
        Q = P.copy()
        for ci in PROF:
            lo, hi = CONTOURS[ci]
            Q[:, lo:hi + 1] = reposition(T(P[:, lo:hi + 1]), T(E[:, lo]), T(E[:, hi]),
                                         "similarity").numpy()
        a += float(np.linalg.norm(Q[:, sel] - GT[:, sel], axis=2).mean()) / 3
        R_ = Q.copy()
        for ci in PROF:
            lo, hi = CONTOURS[ci]
            for f in range(5):
                k = FOLD == f
                s = T(prof[f][ci])[None].expand(int(k.sum()), -1)
                R_[k, lo:hi + 1] = place_on_polyline(T(Q[k, lo:hi + 1]), s).numpy()
        b_ += float(np.linalg.norm(R_[:, sel] - GT[:, sel], axis=2).mean()) / 3
    sweep.append({"sigma_mm": sig, "mean_endpoint_error_mm": round(e_, 4),
                  "reposition_only_mm": round(a, 4),
                  "plus_profile_mm": round(b_, 4), "profile_gain_mm": round(b_ - a, 4)})
    print(f"  {sig:9.3f}{e_:18.4f}{a:17.4f}{b_:16.4f}{b_ - a:+17.4f}")

# break-even, read on the mean-norm column so it is comparable with ep_err
b30 = out["rows"][0]["profile_landmarks_mm"]
xs = np.array([r["mean_endpoint_error_mm"] for r in sweep])
ys = np.array([r["reposition_only_mm"] for r in sweep])
i = int(np.searchsorted(ys, b30))
be = float(xs[i - 1] + (xs[i] - xs[i - 1]) * (b30 - ys[i - 1]) / (ys[i] - ys[i - 1]))
print(f"  break-even (reposition-only == the {b30:.4f} mm baseline on the 30): "
      f"{be:.3f} mm mean endpoint error = sigma {be / 1.5958:.3f}; the ensemble is at "
      f"{ep_err:.3f} mm, so fam_endpoint.py must be {ep_err / be:.2f}x better")
out["endpoint_sensitivity"] = {"ensemble_endpoint_error_mm": round(ep_err, 4),
                               "baseline_30_landmarks_mm": b30,
                               "break_even_mean_endpoint_error_mm": round(be, 3),
                               "required_improvement_factor": round(ep_err / be, 2),
                               "leaky": True,
                               "noise_sweep": sweep,
                               "units": ("sigma_mm is the PER-AXIS Gaussian sd; every other mm "
                                         "here is a MEAN EUCLIDEAN NORM, and "
                                         "E||N(0,sigma^2 I_3)|| = 1.5958*sigma. The break-even "
                                         "is taken on mean_endpoint_error_mm so that it is "
                                         "comparable with ensemble_endpoint_error_mm; comparing "
                                         "sigma with it directly overstates the bar by 1.6x."),
                               "note": ("isotropic Gaussian noise added to the GT endpoints, "
                                        "3 draws averaged; 'reposition only' keeps the "
                                        "ensemble's own phase, '+ fold profile' replaces it. "
                                        "The noise is INDEPENDENT of the curve's own error, "
                                        "which is the right model for an external endpoint "
                                        "predictor but pessimistic for one that shares the "
                                        "backbone: the ensemble's own endpoints are off by "
                                        f"{ep_err:.2f} mm yet repositioning onto them is the "
                                        "identity, because that error is the curve's error.")}

g = out["rows"][6]["per_contour_mm"]
b = out["rows"][0]["per_contour_mm"]
out["conclusion"] = (
    f"On the curves this ensemble already draws, imposing the fold-mean profile with the "
    f"model's OWN endpoints moves the 30 profile landmarks "
    f"{out['rows'][1]['profile_landmarks_mm'] - out['rows'][0]['profile_landmarks_mm']:+.4f} mm "
    f"and the pooled 85 by {out['rows'][1]['delta_all85_mm']:+.4f} mm. With GT endpoints "
    f"(LEAKY) the same placement reaches "
    + ", ".join(f"{CNAMES[ci]} {b[CNAMES[ci]]:.4f} -> {g[CNAMES[ci]]:.4f}" for ci in PROF)
    + f". Forcing the profile on all four contours costs "
    f"{out['rows'][3]['delta_all85_mm']:+.4f} mm, which is why PROFILE_CONTOURS defaults to "
    f"the two contours whose profile is measurably constant.")
print("\n" + out["conclusion"])
json.dump(out, open("research/results/profile_apply.json", "w"), indent=1)
print("wrote research/results/profile_apply.json")
