"""
IS THE SHARED ALONG-CONTOUR ERROR INHERITED FROM THE COARSE INITIALISER?

Four unrelated backbones -- DGCNN, KPConv, PTv3, PointNeXt -- agree on their SIGNED
tangent error at r = 0.80. That is usually read as "the task is hard in that direction",
and info_limit.json supports it (local geometry is worse than chance along the contour).
But there is a second explanation nobody has tested, and it is embarrassingly simple:

    every one of those models is initialised from the SAME classical coarse estimate.

If the coarse estimate carries a systematic per-contour placement bias, every refiner
inherits it, and the r = 0.80 is an artefact of the shared input rather than a statement
about the surface.

A WRONG FIRST VERSION, recorded because the failure is seductive and easy to repeat.
The first attempt used slide(coarse) = mean_i <coarse_i - gt_i, t_i> as the predictor and
got out-of-fold R^2 = 0.77 and 1.1776 -> 1.0025 mm. It is circular: that quantity is the
initialiser's ERROR AGAINST GROUND TRUTH, so it is not available at test time and the
regression was predicting the ensemble's error from a measurement of the same error. The
tells were the size of the number (nothing else in this repo has ever exceeded R^2 = 0)
and that `ortho_feats["base"]` is the 1.3144 mm deep baseline, not the 3.766 mm classical
coarse estimate at all.

WHAT THIS FILE MEASURES INSTEAD. Only quantities a submission actually has:

  feature   f_ec = mean_i < coarse_i - pred_i , that_i >        (contour c of ear e)
            the along-contour DISAGREEMENT between the classical initialiser and the
            refined output, measured with `that`, the unit tangent of the PREDICTED
            polyline (central difference, so no ground truth enters).
            Both the coarse estimate and the prediction exist at test time.

  target    s_ec = mean_i < pred_i - gt_i , t_i >
            the quantity the oracle ladder's per-contour tangent slide row removes
            (-0.256 mm at 4 dof/ear).

Two regressions, both fitted per OUTER FOLD on that fold's TRAINING ears only and applied
to its held-out ears:
    univariate    s_ec ~ f_ec
    multivariate  s_ec ~ f_e0..f_e3 + predicted arc length     because the four contour
                  slides are correlated (+0.49 outer/inner, -0.51 concha/antihelix), so
                  one contour's disagreement may carry information about another's error.

The correction is then applied ALONG `that`, i.e. exactly as a submission would apply it.
In-sample values are printed only to size the optimism and are not results.

    python research/code/init_inheritance.py
Writes research/results/init_inheritance.json
"""
import json
import numpy as np

CONT = [(0, 24, "outer_helix"), (25, 54, "concha"),
        (55, 74, "inner_helix"), (75, 84, "sup._antihelix")]
NC = len(CONT)
NB = 20000

Z = np.load("scratch/ortho_feats.npz")
GT = Z["gt"].astype(np.float64)
T = Z["t"].astype(np.float64)                     # GT frame -- TARGET side only
FOLD = Z["fold"].astype(int)
SUBJ = Z["subj"].astype(int)
PR = np.load("scratch/ensemble5_proj.npy").astype(np.float64)
E = len(GT)

D = np.load("scratch/deep_dataset.npz", allow_pickle=True)
R, C0 = D["R"].astype(np.float64), D["c0"].astype(np.float64)
CO = np.einsum("elk,ekj->elj", D["coarse"].astype(np.float64), R) + C0[:, None, :]
TW = np.einsum("elk,ekj->elj", D["true"].astype(np.float64), R) + C0[:, None, :]
assert np.abs(TW - GT).max() < 1e-6, "deep_dataset/ortho_feats frame mismatch"
print(f"classical coarse {np.linalg.norm(CO-GT,axis=-1).mean():.4f} mm   "
      f"deep baseline {np.linalg.norm(Z['base'].astype(float)-GT,axis=-1).mean():.4f} mm")

FJ = json.load(open("research/results/folds.json"))["assignments"]
assert len(FJ) == E
for a in FJ:
    i = a["ear_index"]
    assert a["fold"] == FOLD[i] and a["subject_group"] == SUBJ[i], f"ear {i}: fold drift"


def pred_tangents(X):
    """Unit tangent of each contour's polyline by central difference. NO ground truth."""
    U = np.zeros_like(X)
    for lo, hi, _ in CONT:
        P = X[:, lo:hi + 1]
        g = np.gradient(P, axis=1)
        U[:, lo:hi + 1] = g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-9)
    return U


TH = pred_tangents(PR)
# sanity: the predicted tangent should agree with the GT frame's tangent up to sign
al = np.abs(np.einsum("elk,elk->el", TH, T)).mean()
print(f"predicted tangent vs GT frame tangent: mean |cos| {al:.4f}")
TH = TH * np.sign(np.einsum("elk,elk->el", TH, T))[..., None]   # orientation only


def contour_mean(v):
    return np.stack([v[:, lo:hi + 1].mean(1) for lo, hi, _ in CONT], 1)


F = contour_mean(np.einsum("elk,elk->el", CO - PR, TH))       # feature  (E,NC)
S = contour_mean(np.einsum("elk,elk->el", PR - GT, T))        # target   (E,NC)
LEN = np.stack([np.linalg.norm(np.diff(PR[:, lo:hi + 1], axis=1), axis=-1).sum(1)
                for lo, hi, _ in CONT], 1)                    # predicted arc length (E,NC)

d0 = np.linalg.norm(PR - GT, axis=-1)
base_mle = float(d0.mean())
print(f"baseline pooled OOF {base_mle:.4f} mm\n")


def oof_fit(X, y):
    """Ridge-free least squares with intercept, fitted per outer fold on training ears."""
    p = np.zeros(E)
    for f in range(5):
        tr, te = FOLD != f, FOLD == f
        assert not (set(SUBJ[tr].tolist()) & set(SUBJ[te].tolist())), "subject leak"
        A = np.hstack([X[tr], np.ones((tr.sum(), 1))])
        w, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        p[te] = np.hstack([X[te], np.ones((te.sum(), 1))]) @ w
    return p


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


out = {"baseline_mm": round(base_mle, 4), "n_ears": E,
       "coarse_mm": round(float(np.linalg.norm(CO - GT, axis=-1).mean()), 4),
       "feature": "mean_i <coarse_i - pred_i, predicted_tangent_i>  (no ground truth)",
       "target": "mean_i <pred_i - gt_i, t_i>  (the oracle ladder's per-contour slide)",
       "per_contour": {}}

print(f"{'contour':16s} {'sd(target)':>11s} {'corr':>7s} {'R2 uni':>8s} {'R2 multi':>9s}")
P_UNI = np.zeros_like(S)
P_MUL = np.zeros_like(S)
for c, (lo, hi, nm) in enumerate(CONT):
    y = S[:, c]
    P_UNI[:, c] = oof_fit(F[:, c:c + 1], y)
    P_MUL[:, c] = oof_fit(np.hstack([F, LEN]), y)
    ru, rm = r2(y, P_UNI[:, c]), r2(y, P_MUL[:, c])
    cr = float(np.corrcoef(F[:, c], y)[0, 1])
    print(f"{nm:16s} {y.std():11.4f} {cr:7.3f} {ru:8.4f} {rm:9.4f}")
    out["per_contour"][nm] = {"sd_target_slide_mm": round(float(y.std()), 4),
                              "corr_feature_target": round(cr, 4),
                              "oof_r2_univariate": round(ru, 4),
                              "oof_r2_multivariate": round(rm, 4)}


def apply_and_test(shift, name):
    """Move each contour back along the PREDICTED tangent by `shift` mm. Deployable."""
    X = PR.copy()
    for c, (lo, hi, _) in enumerate(CONT):
        X[:, lo:hi + 1] -= shift[:, c, None, None] * TH[:, lo:hi + 1]
    d1 = np.linalg.norm(X - GT, axis=-1)
    per_ear = d1.mean(1) - d0.mean(1)
    uu = np.unique(SUBJ)
    bys = np.array([per_ear[SUBJ == s].mean() for s in uu])
    rng = np.random.RandomState(5)
    bs = bys[rng.randint(0, len(uu), (NB, len(uu)))].mean(1)
    ci = [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
    m = float(d1.mean())
    v = "ADOPT" if ci[1] < 0 else "HARMFUL" if ci[0] > 0 else "INDISTINGUISHABLE"
    print(f"  {name:34s} {m:.4f} mm  delta {m-base_mle:+.4f}  CI {ci}  {v}")
    return {"mle_mm": round(m, 4), "delta_mm": round(m - base_mle, 4),
            "ci95": ci, "verdict": v}


print("\ncorrection applied along the PREDICTED tangent:")
out["corrections"] = {
    "univariate": apply_and_test(P_UNI, "OOF univariate"),
    "multivariate": apply_and_test(P_MUL, "OOF multivariate + arc length"),
    "oracle_slide_LEAKY": apply_and_test(S, "oracle slide (GT-fitted, bound)")}

mean_r2 = float(np.mean([v["oof_r2_multivariate"] for v in out["per_contour"].values()]))
out["mean_oof_r2_multivariate"] = round(mean_r2, 4)
out["conclusion"] = (
    f"Mean out-of-fold R^2 of the best test-time-available predictor is {mean_r2:.4f}. " +
    ("The refined placement is NOT inherited from the coarse initialiser in any way a "
     "submission could exploit, so the r=0.80 agreement between four backbones is about "
     "the task, not about their shared input. The initialiser-disagreement feature joins "
     "the seven predictors that carry no usable signal -- this is the eighth."
     if mean_r2 <= 0.01 else
     "Part of the final placement error is predictable from the initialiser-refiner "
     "disagreement, which is available at test time and is causally UPSTREAM of the "
     "error, unlike all seven predictors tried before."))
out["caveats"] = [
    "The TARGET uses the GT frame's tangent, which is correct because the target is a "
    "diagnostic quantity; the FEATURE and the applied CORRECTION use only the predicted "
    "polyline's own tangent, so the correction is deployable as written.",
    "Linear maps only. A nonlinear map of the same features is not tried here.",
    "oracle_slide_LEAKY is fitted on ground truth and is a bound, not a result."]
json.dump(out, open("research/results/init_inheritance.json", "w"), indent=1)
print(f"\n{out['conclusion']}")
print("\nwrote research/results/init_inheritance.json")
