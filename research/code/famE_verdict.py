"""
FAMILY E VERDICT: does a subject's OTHER ear carry usable information about this one?

WHY THIS IS THE EXPERIMENT THAT MATTERS. shape_calibration.py showed the model is a
CALIBRATED conditional-mean predictor: its under-dispersion is exactly what minimising
MSE against a partly unpredictable target requires, so no architecture, resolution,
ensemble or rescaling of the existing output can reclaim the 0.575mm the oracle ladder
leaves. That leaves only new information or a different target. The one new-information
channel that is measurable AND available at test time is the other ear: the per-contour
slides of a subject's two ears correlate +0.30..+0.42, and the oracle corrections
correlate 0.35..0.56, against a geometry-matched different-subject null of |r| <= 0.19.

MODE is the only difference between the arms. Identical parameter count (995,248),
identical state_dict keys, identical data file, one shared context-set table read by
NEEDS / the augmenter / the model. So a difference here is the bilateral information and
not an architecture change. context_probe.json already showed 121 HAND-CRAFTED global and
bilateral features carry nothing (max OOF R^2 -0.0149); this asks whether a LEARNED
encoder trained end to end against the landmark loss does better.

THE HONEST PRIOR IS NO, and the value is then in the control: if bilateral == single
within the noise, "use the other ear" is closed with a like-for-like architecture rather
than a feature-engineering proxy.

STATISTICS. Pooled out-of-fold over all 340 ears per (arm, seed); the arms are compared
PAIRED per ear, and the bootstrap resamples SUBJECTS (both ears move together) because
the two ears of a subject are not independent -- which is the very correlation under
test. Fold and seed variance are decomposed so the delta can be read against the right
noise scale: comparing it to the seed spread of a single fold would understate the
uncertainty, and comparing it to the fold spread would overstate it.

Tolerates runs still in flight: it reports what is complete and says what is missing.

    python research/code/famE_verdict.py
Writes research/results/family_E.json
"""
import glob
import json
import os
import numpy as np

CONT = [(0, 24, "outer_helix"), (25, 54, "concha"),
        (55, 74, "inner_helix"), (75, 84, "sup._antihelix")]
NB = 20000
ARMS = ("single", "bilat")
SEEDS = (0, 1, 2)
FOLDS = (0, 1, 2, 3, 4)

Z = np.load("scratch/ortho_feats.npz")
GT = Z["gt"].astype(np.float64)
T = Z["t"].astype(np.float64)
FOLD, SUBJ = Z["fold"].astype(int), Z["subj"].astype(int)
E = len(GT)

FJ = json.load(open("research/results/folds.json"))["assignments"]
for a in FJ:
    i = a["ear_index"]
    assert a["fold"] == FOLD[i] and a["subject_group"] == SUBJ[i], f"ear {i}: fold drift"


def assemble(arm, seed):
    """Pooled OOF prediction for one (arm, seed), or None if a fold is missing."""
    P = np.full_like(GT, np.nan)
    for f in FOLDS:
        base = f"scratch/famE_{arm}_s{seed}_f{f}"
        if not (os.path.exists(base + ".npy") and os.path.exists(base + ".json")):
            return None, f"fold {f} missing"
        j = json.load(open(base + ".json"))
        idx = np.asarray(j["val_ear_index"], int)
        # the run's own held-out set must BE the frozen fold -- not merely overlap it
        assert set(idx.tolist()) == set(np.where(FOLD == f)[0].tolist()), \
            f"{base}: val_ear_index is not fold {f}"
        P[idx] = np.load(base + ".npy").astype(np.float64)
    assert not np.isnan(P).any(), "pooled OOF has holes"
    return P, None


def mle(P):
    return float(np.linalg.norm(P - GT, axis=-1).mean())


def paired(Pa, Pb, seed=5):
    """Pb - Pa, bootstrapped over SUBJECTS so a subject's two ears move together."""
    da = np.linalg.norm(Pa - GT, axis=-1).mean(1)
    db = np.linalg.norm(Pb - GT, axis=-1).mean(1)
    diff = db - da
    uu = np.unique(SUBJ)
    bys = np.array([diff[SUBJ == s].mean() for s in uu])
    rng = np.random.RandomState(seed)
    bs = bys[rng.randint(0, len(uu), (NB, len(uu)))].mean(1)
    return (round(float(diff.mean()), 4),
            [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
            round(float((bs < 0).mean()), 4))


def directional(P):
    e = P - GT
    et = np.einsum("elk,elk->el", e, T)
    return {"tangent_rmse_mm": round(float(np.sqrt((et ** 2).mean())), 4),
            "per_contour_mm": {nm: round(float(np.linalg.norm(
                (P - GT)[:, lo:hi + 1], axis=-1).mean()), 4) for lo, hi, nm in CONT}}


out = {"question": "does a subject's other ear carry usable information about this one?",
       "control": "MODE is the only difference; 995,248 params and identical state_dict "
                  "keys in both arms",
       "n_ears": E, "n_bootstrap": NB, "arms": {}, "missing": []}

POOL = {}
print(f"{'arm':9s} {'seed':>4s} {'pooled OOF':>11s}   per-fold")
for arm in ARMS:
    for s in SEEDS:
        P, why = assemble(arm, s)
        if P is None:
            out["missing"].append(f"{arm} seed {s}: {why}")
            print(f"{arm:9s} {s:4d}   -- {why}")
            continue
        POOL[(arm, s)] = P
        pf = [round(float(np.linalg.norm((P - GT)[FOLD == f], axis=-1).mean()), 4)
              for f in FOLDS]
        print(f"{arm:9s} {s:4d} {mle(P):11.4f}   {pf}")
        out["arms"].setdefault(arm, {})[f"seed{s}"] = {
            "pooled_oof_mm": round(mle(P), 4), "per_fold_mm": pf, **directional(P)}

done_seeds = sorted({s for (a, s) in POOL if all((x, s) in POOL for x in ARMS)})
out["seeds_complete_in_both_arms"] = done_seeds
if not done_seeds:
    json.dump(out, open("research/results/family_E.json", "w"), indent=1)
    raise SystemExit("\nno seed is complete in both arms yet -- nothing to compare")

print(f"\npaired bilateral - single, subject-level bootstrap ({NB} draws):")
out["per_seed_delta"] = {}
for s in done_seeds:
    d, ci, p = paired(POOL[("single", s)], POOL[("bilat", s)])
    v = "ADOPT" if ci[1] < 0 else "HARMFUL" if ci[0] > 0 else "INDISTINGUISHABLE"
    print(f"  seed {s}: {d:+.4f} mm  CI {ci}  P(delta<0) {p}  {v}")
    out["per_seed_delta"][f"seed{s}"] = {"delta_mm": d, "ci95": ci, "p_negative": p,
                                         "verdict": v}

# seed-ensembled arms: the comparison the shipped pipeline would actually make
if len(done_seeds) >= 2:
    Ea = np.mean([POOL[("single", s)] for s in done_seeds], 0)
    Eb = np.mean([POOL[("bilat", s)] for s in done_seeds], 0)
    d, ci, p = paired(Ea, Eb)
    v = "ADOPT" if ci[1] < 0 else "HARMFUL" if ci[0] > 0 else "INDISTINGUISHABLE"
    print(f"\n  {len(done_seeds)}-seed ensemble: single {mle(Ea):.4f}  bilateral {mle(Eb):.4f}"
          f"  delta {d:+.4f}  CI {ci}  {v}")
    out["seed_ensembled"] = {"n_seeds": len(done_seeds),
                             "single_mm": round(mle(Ea), 4), "bilateral_mm": round(mle(Eb), 4),
                             "delta_mm": d, "ci95": ci, "p_negative": p, "verdict": v}

# noise scales the delta has to be read against
for arm in ARMS:
    ss = [s for s in done_seeds if (arm, s) in POOL]
    if len(ss) >= 2:
        M = np.array([[float(np.linalg.norm((POOL[(arm, s)] - GT)[FOLD == f], axis=-1).mean())
                       for s in ss] for f in FOLDS])
        out["arms"][arm]["variance"] = {
            "fold_sd": round(float(M.mean(1).std(ddof=1)), 4),
            "seed_sd": round(float(M.mean(0).std(ddof=1)), 4),
            "resid_sd": round(float((M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True)
                                     + M.mean()).std(ddof=1)), 4)}

nseed = len(done_seeds)
final = out.get("seed_ensembled") or out["per_seed_delta"][f"seed{done_seeds[0]}"]
out["verdict"] = final["verdict"]
out["complete"] = (len(done_seeds) == len(SEEDS))
out["conclusion"] = (
    f"With {nseed} of {len(SEEDS)} seeds complete in both arms, bilateral minus single is "
    f"{final['delta_mm']:+} mm (CI {final['ci95']}). " + (
        "The other ear carries usable information: a learned encoder finds what 121 "
        "hand-crafted bilateral features could not."
        if final["verdict"] == "ADOPT" else
        "Bilateral context is indistinguishable from single-ear at this sample size. "
        "Together with context_probe.json's 121 hand-crafted features this closes the "
        "'use the other ear' idea with a like-for-like architecture: the shared "
        "per-subject factor is real in the ORACLE corrections but is not recoverable "
        "from the observable geometry of either ear."
        if final["verdict"] == "INDISTINGUISHABLE" else
        "Bilateral context HURTS -- the extra input is acting as a nuisance variable."))
if not out["complete"]:
    out["conclusion"] += (f" NOT FINAL: {len(out['missing'])} run group(s) still in "
                          f"flight; the seed spread below is the scale to read it against.")
out["caveats"] = [
    "This family scores ~1.35mm, weaker than the 1.26mm DGCNN baseline, so a bilateral "
    "win here answers the scientific question without necessarily producing a better "
    "ensemble member. Porting the context to the strong backbone would be the follow-up.",
    "The bootstrap resamples SUBJECTS, not ears: the two ears of a subject are not "
    "independent, and that dependence is exactly what is under test.",
    "Raw-network OOF -- no surface projection, no TTA -- so these are not comparable to "
    "the 1.1776mm shipped figure.",
    "In MODE=single the context encoder's is_head input column never fires, so those "
    "weights exist and take no gradient. That is the price of a bit-for-bit identical "
    "parameter list and is cheaper than a differently-shaped comparison."]
json.dump(out, open("research/results/family_E.json", "w"), indent=1)
print(f"\n{out['verdict']}: {out['conclusion']}")
print("\nwrote research/results/family_E.json")
