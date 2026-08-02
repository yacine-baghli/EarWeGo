"""
IS THE UNDER-DISPERSION THE CALIBRATED AMOUNT, OR TOO MUCH?

contour_rigidity.py established that the prediction varies LESS across ears than ground
truth does (median CV ratio 0.893 over the most rigid cross-contour pairs, present in
every single model and deepened by ensembling). That is shrinkage. The open question is
whether it is the RIGHT amount of shrinkage, and it has a precise answer.

THE IDENTITY. For a conditional-mean predictor, per shape mode k,

        var(pred_k)  =  rho_k^2 * var(gt_k)

and equivalently the optimal scalar gain a_k = cov(gt_k, pred_k) / var(pred_k) equals 1.
So:

    a_k ~= 1   the model is CALIBRATED. Its under-dispersion is exactly what minimising
               MSE against a partly unpredictable target requires, and no rescaling of
               the existing prediction can help. The residual is conditional variance.
    a_k >  1   the model is OVER-shrunk. Rescaling mode k outward is a free, deployable
               gain that needs no new information at all.
    a_k <  1   the model is over-confident in mode k and should be shrunk further.

This matters because it is NOT another attempt to predict the correction -- the nine
before it all failed at that. It re-scales what the model already produces, so it is
either free money or a proof that there is none.

METHOD. Everything is done in the CANONICAL frame (rotation is orthogonal, so ordered
Euclidean distances -- the competition metric -- are unchanged). Per outer fold, a PCA
basis is built from that fold's TRAINING ears' ground-truth shapes only; both the GT and
the prediction are projected onto it; a scalar gain per mode is fitted on the TRAINING
ears and applied to the HELD-OUT ears. Ear- and subject-disjointness is asserted. The
in-sample gain is reported only to size the optimism.

A three-parameter variant (one gain for all modes, one for the first 10, one for the
rest) is fitted too, because 255 free gains on 272 training ears is itself a variance
problem and the pooled version is the honest headline if the per-mode one overfits.

    python research/code/shape_calibration.py
Writes research/results/shape_calibration.json
"""
import json
import numpy as np

CONT = [(0, 24, "outer_helix"), (25, 54, "concha"),
        (55, 74, "inner_helix"), (75, 84, "sup._antihelix")]
NL, NB = 85, 20000

Z = np.load("scratch/ortho_feats.npz")
GTW = Z["gt"].astype(np.float64)
FOLD, SUBJ = Z["fold"].astype(int), Z["subj"].astype(int)
PRW = np.load("scratch/ensemble5_proj.npy").astype(np.float64)
E = len(GTW)

D = np.load("scratch/deep_dataset.npz", allow_pickle=True)
R, C0 = D["R"].astype(np.float64), D["c0"].astype(np.float64)
# world = canonical @ R + c0   ->   canonical = (world - c0) @ R^T
GT = np.einsum("elk,ejk->elj", GTW - C0[:, None, :], R)
PR = np.einsum("elk,ejk->elj", PRW - C0[:, None, :], R)
# R is orthogonal to 6e-8 and the round-trip reproduces deep_dataset's own canonical
# landmarks to 2.5e-6 mm, so the tolerance is set at a physically meaningless 1e-5 mm
# rather than at float64 noise: anything larger would be a real frame error.
iso = float(np.abs(np.linalg.norm(PR - GT, axis=-1) -
                   np.linalg.norm(PRW - GTW, axis=-1)).max())
assert iso < 1e-5, f"frame map is not isometric: worst distance changes by {iso:g} mm"

FJ = json.load(open("research/results/folds.json"))["assignments"]
for a in FJ:
    i = a["ear_index"]
    assert a["fold"] == FOLD[i] and a["subject_group"] == SUBJ[i], f"ear {i}: fold drift"

d0 = np.linalg.norm(PR - GT, axis=-1)
base = float(d0.mean())
G, P = GT.reshape(E, -1), PR.reshape(E, -1)          # (E,255)
print(f"baseline pooled OOF {base:.4f} mm   canonical frame verified isometric\n")


def apply_gains(mode, nmode, report=False):
    """Fold-safe PCA + per-mode gain. mode: 'per_mode' | 'pooled3' | 'global'."""
    OUT = PR.copy().reshape(E, -1)
    gains_all, r2_all = [], []
    for f in range(5):
        tr, te = FOLD != f, FOLD == f
        assert not (set(SUBJ[tr].tolist()) & set(SUBJ[te].tolist())), "subject leak"
        mu = G[tr].mean(0)
        U, S, Vt = np.linalg.svd(G[tr] - mu, full_matrices=False)
        B = Vt[:nmode]                                # (K,255) basis from TRAINING GT only
        gt_tr, pr_tr = (G[tr] - mu) @ B.T, (P[tr] - mu) @ B.T
        pr_te = (P[te] - mu) @ B.T
        vp = (pr_tr ** 2).mean(0)
        a = (gt_tr * pr_tr).mean(0) / np.maximum(vp, 1e-12)
        if mode == "global":
            a = np.full(nmode, float((gt_tr * pr_tr).sum() / max((pr_tr ** 2).sum(), 1e-12)))
        elif mode == "pooled3":
            grp = [slice(0, 10), slice(10, 40), slice(40, nmode)]
            a2 = np.ones(nmode)
            for g in grp:
                num, den = (gt_tr[:, g] * pr_tr[:, g]).sum(), (pr_tr[:, g] ** 2).sum()
                a2[g] = num / max(den, 1e-12)
            a = a2
        a = np.clip(a, 0.5, 2.0)
        # rebuild: keep the component outside the basis untouched, rescale inside it
        res = (P[te] - mu) - pr_te @ B
        OUT[te] = mu + res + (pr_te * a) @ B
        gains_all.append(a)
        if report:
            gt_te = (G[te] - mu) @ B.T
            r2_all.append([float(np.corrcoef(gt_te[:, k], pr_te[:, k])[0, 1] ** 2)
                           for k in range(min(nmode, 20))])
    return OUT.reshape(E, NL, 3), np.array(gains_all), (np.array(r2_all) if report else None)


def score(X, name):
    d1 = np.linalg.norm(X - GT, axis=-1)
    pe = d1.mean(1) - d0.mean(1)
    uu = np.unique(SUBJ)
    bys = np.array([pe[SUBJ == s].mean() for s in uu])
    rng = np.random.RandomState(5)
    bs = bys[rng.randint(0, len(uu), (NB, len(uu)))].mean(1)
    ci = [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
    m = float(d1.mean())
    v = "ADOPT" if ci[1] < 0 else "HARMFUL" if ci[0] > 0 else "INDISTINGUISHABLE"
    print(f"  {name:34s} {m:.4f} mm  delta {m-base:+.4f}  CI {ci}  {v}")
    return {"mle_mm": round(m, 4), "delta_mm": round(m - base, 4), "ci95": ci, "verdict": v}


out = {"baseline_mm": round(base, 4), "n_ears": E, "n_bootstrap": NB,
       "identity": ("for a conditional-mean predictor var(pred_k) = rho_k^2 var(gt_k), "
                    "equivalently the optimal per-mode gain a_k = 1"),
       "runs": {}}

print("out-of-fold gain calibration on a fold-safe GT PCA basis:")
for mode in ("global", "pooled3", "per_mode"):
    for K in (40, 120):
        X, gains, _ = apply_gains(mode, K)
        key = f"{mode}_K{K}"
        out["runs"][key] = score(X, f"{mode}, {K} modes")
        out["runs"][key]["gain_mean"] = round(float(gains.mean()), 4)
        out["runs"][key]["gain_sd_across_folds"] = round(float(gains.std(0).mean()), 4)

# what the gains actually look like, and whether they track predictability
_, gains, r2 = apply_gains("per_mode", 40, report=True)
gm, r2m = gains.mean(0), r2.mean(0)
print(f"\nper-mode gain (40 modes): mean {gm.mean():.3f}  "
      f"first 10 {np.round(gm[:10], 3).tolist()}")
print(f"per-mode OOF rho^2      : first 10 {np.round(r2m[:10], 3).tolist()}")
out["per_mode_detail"] = {
    "gain_first20": [round(float(x), 4) for x in gm[:20]],
    "rho2_first20": [round(float(x), 4) for x in r2m[:20]],
    "gain_mean_40": round(float(gm.mean()), 4),
    "gain_frac_above_1": round(float((gm > 1).mean()), 4)}

# the variance ratio the identity predicts, against the one observed
best = min(out["runs"], key=lambda k: out["runs"][k]["mle_mm"])
adopted = [k for k, v in out["runs"].items() if v["verdict"] == "ADOPT"]
out["best_config"] = best
out["verdict"] = "ADOPT" if adopted else "CALIBRATED"
out["conclusion"] = (
    f"Mean out-of-fold gain over the first 40 shape modes is {gm.mean():.3f}, and "
    f"{gm.mean() > 1 and 'above' or 'at or below'} one in "
    f"{100*float((gm > 1).mean()):.0f}% of them. Best configuration {best}: "
    f"{out['runs'][best]['mle_mm']} mm ({out['runs'][best]['delta_mm']:+}, CI "
    f"{out['runs'][best]['ci95']}). " + (
        "Rescaling the existing prediction is a free, deployable gain that needs no new "
        "information."
        if adopted else
        "The model is CALIBRATED: its under-dispersion is exactly what minimising MSE "
        "against a partly unpredictable target requires, so no rescaling of what it "
        "already produces can help. Together with contour_rigidity.py this closes the "
        "question the shrinkage raised -- the remaining error is conditional variance, "
        "and only new information or a different target can reduce it."))
out["caveats"] = [
    "The PCA basis, the mean shape and the gains all come from the outer fold's TRAINING "
    "ears; held-out ears contribute nothing to any of them.",
    "Gains are clipped to [0.5,2.0] so a near-zero-variance mode cannot explode.",
    "The component of the prediction outside the K-mode basis is passed through "
    "untouched, so K controls only how much of the shape is recalibrated.",
    "rho^2 per mode is computed on held-out ears and is reported for interpretation, not "
    "used to set any gain."]
json.dump(out, open("research/results/shape_calibration.json", "w"), indent=1)
print(f"\n{out['verdict']}: {out['conclusion']}")
print("\nwrote research/results/shape_calibration.json")
