"""
THE LOSS OPTIMISES THE MEAN. THE METRIC WANTS THE MEDIAN. DOES THE GAP COST US ANYTHING?

Every model in this repo trains ordered MSE, which is minimised by the conditional MEAN.
The competition metric is the mean ordered Euclidean DISTANCE, and

        argmin_c  E|| x - c ||     is the geometric median of x,
        argmin_c  E|| x - c ||^2   is its mean.

Those coincide only for a symmetric distribution. Ours is not symmetric: the pooled error
has mean 1.1776mm against median 0.9305 and p90 2.3323, a heavy right tail. So the models
are provably optimising the wrong functional, and the question is whether the difference
is worth anything in millimetres.

shape_calibration.py showed the model is a well-calibrated conditional-MEAN predictor
(optimal per-mode gain 1.000). That result says nothing about whether the mean is the
right target -- it confirms the model hits the target it was given. This file asks
whether the target itself is misaligned with the score.

THREE TESTS, none of which needs new information.

(1) PER-LANDMARK CONSTANT OFFSET, mean vs geometric median.
    A landmark's residuals across ears form a cloud. The offset that minimises the mean
    DISTANCE over that cloud is its geometric median; the offset that minimises squared
    distance is its mean. Both are fitted on the outer fold's TRAINING ears and applied to
    its held-out ears -- 255 parameters, the lowest-variance corrector possible, and the
    one previous correction attempts skipped in favour of learned regressors.
    If the model is unbiased AND the residuals are symmetric, both give zero. If the
    geometric-median offset beats the mean offset, the mean/median mismatch is real.

(2) AGGREGATION RULE over the 7 trained networks. The shipped ensemble takes an
    arithmetic mean. Under this metric the geometric median (Weiszfeld) is the matched
    aggregator and is also robust to a member that fails on one ear. Coordinate-wise
    median and a trimmed mean are included as cheaper relatives.

(3) MEMBER WEIGHTING. The shipped mean is over 3 MEMBERS (dgcnn 3 seeds, kpconv 2, ptv3
    2), so a dgcnn seed carries 1/9 and a kpconv seed 1/6. Averaging the 7 NETWORKS
    equally is a different rule and has not been measured; it is reported alongside so
    the aggregation comparison is not confounded by the weighting change.

Everything is scored pooled out-of-fold over all 340 ears with a paired bootstrap that
resamples SUBJECTS. No surface projection is applied to the variants, so the baseline row
here is the pre-projection ensemble, and projection is applied at the end to the winner
only if one exists.

    python research/code/metric_alignment.py
Writes research/results/metric_alignment.json
"""
import json
import numpy as np

NB = 20000
W = "scratch"
Z = np.load(f"{W}/ortho_feats.npz")
GT = Z["gt"].astype(np.float64)
FOLD, SUBJ = Z["fold"].astype(int), Z["subj"].astype(int)
E, NL = GT.shape[0], GT.shape[1]

FJ = json.load(open("research/results/folds.json"))["assignments"]
for a in FJ:
    i = a["ear_index"]
    assert a["fold"] == FOLD[i] and a["subject_group"] == SUBJ[i], f"ear {i}: fold drift"


def pooled(tag_fmt, seeds):
    """(S,E,85,3) pooled-OOF stack, one slice per seed."""
    out = []
    for s in seeds:
        P = np.full_like(GT, np.nan)
        for f in range(5):
            P[FOLD == f] = np.load(f"{W}/{tag_fmt.format(s=s, f=f)}.npy").astype(np.float64)
        assert not np.isnan(P).any(), f"{tag_fmt} seed {s}: pooled OOF has holes"
        out.append(P)
    return np.stack(out)


DG = pooled("screen_normalsfix_s{s}_f{f}", (0, 1, 2))
KP = np.stack([pooled("famA_kpconv_f{f}", (0,))[0], pooled("famA_kpconv_s1_f{f}", (0,))[0]])
PT = np.stack([pooled("famA_ptv3_f{f}", (0,))[0], pooled("famA_ptv3_s1_f{f}", (0,))[0]])
NET = np.concatenate([DG, KP, PT])                      # (7,E,85,3) every trained network
MEMBER = np.stack([DG.mean(0), KP.mean(0), PT.mean(0)])  # (3,E,85,3) the shipped grouping
print(f"{len(NET)} networks, {len(MEMBER)} members")


def mle(P):
    return float(np.linalg.norm(P - GT, axis=-1).mean())


BASE = MEMBER.mean(0)
base = mle(BASE)
ref = np.load(f"{W}/ensemble5_ens.npy").astype(np.float64)
print(f"reassembled member-mean {base:.4f} mm vs shipped ensemble5_ens {mle(ref):.4f} mm "
      f"(max |diff| {np.abs(BASE-ref).max():.2e} mm)")
assert np.abs(BASE - ref).max() < 1e-6, "reassembly does not reproduce the shipped ensemble"


def paired(P, name, against=None, seed=5):
    A = against if against is not None else BASE
    da = np.linalg.norm(A - GT, axis=-1).mean(1)
    db = np.linalg.norm(P - GT, axis=-1).mean(1)
    diff = db - da
    uu = np.unique(SUBJ)
    bys = np.array([diff[SUBJ == s].mean() for s in uu])
    rng = np.random.RandomState(seed)
    bs = bys[rng.randint(0, len(uu), (NB, len(uu)))].mean(1)
    ci = [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
    m = mle(P)
    v = "ADOPT" if ci[1] < 0 else "HARMFUL" if ci[0] > 0 else "INDISTINGUISHABLE"
    print(f"  {name:38s} {m:.4f}  delta {m-mle(A):+.4f}  CI {ci}  {v}")
    return {"mle_mm": round(m, 4), "delta_mm": round(m - mle(A), 4), "ci95": ci, "verdict": v}


def geo_median(X, axis=0, iters=64, eps=1e-9):
    """Weiszfeld over `axis`. Falls back to the mean when a point coincides with it."""
    c = X.mean(axis)
    for _ in range(iters):
        d = np.linalg.norm(X - c[None], axis=-1)          # (n, ...)
        w = 1.0 / np.maximum(d, eps)
        c_new = (X * w[..., None]).sum(0) / w.sum(0)[..., None]
        if np.abs(c_new - c).max() < 1e-10:
            c = c_new
            break
        c = c_new
    return c


out = {"baseline_member_mean_mm": round(base, 4), "n_ears": E, "n_networks": len(NET),
       "n_bootstrap": NB,
       "why": ("argmin E||x-c|| is the geometric median, argmin E||x-c||^2 is the mean; "
               "the models train the second and are scored on the first"),
       "tests": {}}

# ---------------------------------------------- (1) per-landmark constant offset
print(f"\nbaseline (shipped member mean) {base:.4f} mm")
print("\n(1) per-landmark constant offset, fitted out-of-fold:")
for kind in ("mean", "geometric_median"):
    P = BASE.copy()
    off_norm = []
    for f in range(5):
        tr, te = FOLD != f, FOLD == f
        assert not (set(SUBJ[tr].tolist()) & set(SUBJ[te].tolist())), "subject leak"
        Rr = (BASE - GT)[tr]                               # (n,85,3) training residuals
        o = Rr.mean(0) if kind == "mean" else geo_median(Rr)
        P[te] = BASE[te] - o[None]
        off_norm.append(float(np.linalg.norm(o, axis=-1).mean()))
    out["tests"][f"offset_{kind}"] = paired(P, f"offset = training {kind}")
    out["tests"][f"offset_{kind}"]["mean_offset_mm"] = round(float(np.mean(off_norm)), 4)

# ---------------------------------------------------------- (2) aggregation rule
print("\n(2) aggregation rule over the 7 networks:")
rules = {
    "net_mean_equal7": NET.mean(0),
    "net_geometric_median": geo_median(NET),
    "net_coordinate_median": np.median(NET, 0),
    "net_trimmed_mean_1": np.sort(NET, 0)[1:-1].mean(0),
    "member_geometric_median": geo_median(MEMBER),
    "member_coordinate_median": np.median(MEMBER, 0),
}
for k, P in rules.items():
    out["tests"][k] = paired(P, k)

# ------------------------------------------- (3) is the residual actually skewed?
Rz = BASE - GT
nrm = np.linalg.norm(Rz, axis=-1)
gm_all = geo_median(Rz.reshape(-1, 3)[:, None]).ravel()
out["residual_shape"] = {
    "pooled_mean_mm": round(float(nrm.mean()), 4),
    "pooled_median_mm": round(float(np.median(nrm)), 4),
    "pooled_p90_mm": round(float(np.percentile(nrm, 90)), 4),
    "mean_over_median": round(float(nrm.mean() / np.median(nrm)), 4),
    "global_residual_vector_mean_mm": round(float(np.linalg.norm(Rz.reshape(-1, 3).mean(0))), 4),
    "global_residual_vector_geomedian_mm": round(float(np.linalg.norm(gm_all)), 4),
    "per_landmark_bias_mm": round(float(np.linalg.norm(Rz.mean(0), axis=-1).mean()), 4)}
print(f"\nresidual shape: mean/median {out['residual_shape']['mean_over_median']}, "
      f"per-landmark systematic bias {out['residual_shape']['per_landmark_bias_mm']} mm")

best = min(out["tests"], key=lambda k: out["tests"][k]["mle_mm"])
adopted = [k for k, v in out["tests"].items() if v["verdict"] == "ADOPT"]
out["best"] = best
out["verdict"] = "ADOPT" if adopted else "NULL"
out["conclusion"] = (
    f"Best variant {best} at {out['tests'][best]['mle_mm']} mm "
    f"({out['tests'][best]['delta_mm']:+}, CI {out['tests'][best]['ci95']}). " + (
        f"Aligning the aggregation or the offset with the metric is a free gain: "
        f"{', '.join(adopted)}."
        if adopted else
        "The mean/median mismatch is real in the distribution -- pooled mean over median "
        f"is {out['residual_shape']['mean_over_median']} -- but it is NOT exploitable "
        "post hoc. The per-landmark systematic bias is only "
        f"{out['residual_shape']['per_landmark_bias_mm']} mm, so there is no constant to "
        "remove, and with seven strongly correlated networks the geometric median and the "
        "mean are nearly the same point. Fixing this would require training against the "
        "distance rather than its square, not re-aggregating what MSE already produced."))
out["caveats"] = [
    "No surface projection anywhere here, so every row is comparable to the 1.1827mm "
    "pre-projection ensemble and not to the 1.1776mm shipped figure.",
    "The geometric median over 7 highly correlated members is close to their mean by "
    "construction; a null here does NOT show that a distance-based training loss is null.",
    "Offsets are fitted on the outer fold's TRAINING ears only; ear- and subject-"
    "disjointness is asserted per fold.",
    "net_mean_equal7 changes the member weighting as well as nothing else, and is "
    "reported so the aggregation rows are not read as if weighting were held fixed."]
json.dump(out, open("research/results/metric_alignment.json", "w"), indent=1)
print(f"\n{out['verdict']}: {out['conclusion']}")
print("\nwrote research/results/metric_alignment.json")
