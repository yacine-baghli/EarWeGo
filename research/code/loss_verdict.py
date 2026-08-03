"""
DOES TRAINING AGAINST THE METRIC BEAT TRAINING AGAINST ITS SQUARE?

argmin_c E||x-c||^2 is the conditional MEAN; argmin_c E||x-c|| is the geometric MEDIAN.
Every model here trains the first and is scored on the second, and the pooled error is
skewed (mean 1.1827mm, median 0.9305, ratio 1.27) so the two functionals do not coincide.
metric_alignment.py established the gap is not recoverable AFTER training -- per-landmark
offsets and geometric-median aggregation over seven correlated networks are null or
harmful -- which leaves changing the objective as the only test.

THE ARMS. gpu_screen.py VARIANT=normals on screen_data_2048nrm.npz, 1200 epochs, identical
LR and deep-supervision weights, differing ONLY in LOSSFN:
    mse     the shipped objective, verified BIT-IDENTICAL to the pre-change code
    dist    sqrt(d2 + 1e-8), the metric itself
    phuber  delta^2 (sqrt(1 + d2/delta^2) - 1) at delta = 1mm: MSE below 1mm, metric above
The mse arm needs no new training: screen_normalsfix is that exact configuration at
5 folds x 3 seeds, pooled OOF 1.2292mm (3-seed ensemble) / 1.2675-1.2728 per seed.

WHY THE BASELINE ANCHOR MATTERS. Comparing a new seed-0 arm against baseline SEED 0 alone
confounds the loss change with which seed the baseline happened to draw: on fold 0 the
baseline's three seeds are 1.2800 / 1.2580 / 1.2577, a 0.022mm spread, and seed 0 is the
worst of them. So the honest per-fold reference is the baseline's SEED MEAN, and both are
reported. A seed-matched comparison is also given because it is the cleaner paired design
once the new arm has its own three seeds.

FILENAMES. gpu_screen.py writes screen_<OUTTAG>_s<SEED>_f<FOLD>.json and the launcher
passed an OUTTAG already containing _s<seed>_f<fold>, so the remote files are DOUBLE
suffixed: screen_loss_dist_s0_f2_s0_f2.json. This reader accepts both spellings, because
globbing the intuitive name silently finds nothing -- which already cost an hour of this
session.

    python research/code/loss_verdict.py
Writes research/results/loss_verdict.json
"""
import glob
import json
import os
import numpy as np

NB = 20000
W = "scratch"
ARMS = ("dist", "phuber")
BASE_SEEDS = (0, 1, 2)

Z = np.load(f"{W}/ortho_feats.npz")
GT = Z["gt"].astype(np.float64)
FOLD, SUBJ = Z["fold"].astype(int), Z["subj"].astype(int)
E = len(GT)

FJ = json.load(open("research/results/folds.json"))["assignments"]
for a in FJ:
    i = a["ear_index"]
    assert a["fold"] == FOLD[i] and a["subject_group"] == SUBJ[i], f"ear {i}: fold drift"


def find(arm, seed, fold):
    """Both the intuitive and the double-suffixed spelling."""
    for pat in (f"{W}/screen_loss_{arm}_s{seed}_f{fold}_s{seed}_f{fold}",
                f"{W}/screen_loss_{arm}_s{seed}_f{fold}",
                f"{W}/loss_{arm}_s{seed}_f{fold}"):
        if os.path.exists(pat + ".npy") and os.path.exists(pat + ".json"):
            return pat
    return None


def assemble(prefix_fn, seed):
    P = np.full_like(GT, np.nan)
    for f in range(5):
        base = prefix_fn(seed, f)
        if base is None:
            return None, f"fold {f} missing"
        j = json.load(open(base + ".json"))
        idx = np.asarray(j["val_ear_index"], int)
        assert set(idx.tolist()) == set(np.where(FOLD == f)[0].tolist()), \
            f"{base}: val_ear_index is not fold {f}"
        P[idx] = np.load(base + ".npy").astype(np.float64)
    assert not np.isnan(P).any(), "pooled OOF has holes"
    return P, None


def mle(P):
    return float(np.linalg.norm(P - GT, axis=-1).mean())


def paired(Pa, Pb, seed=5):
    """Pb - Pa, resampling SUBJECTS so a subject's two ears move together."""
    da = np.linalg.norm(Pa - GT, axis=-1).mean(1)
    db = np.linalg.norm(Pb - GT, axis=-1).mean(1)
    diff = db - da
    uu = np.unique(SUBJ)
    bys = np.array([diff[SUBJ == s].mean() for s in uu])
    rng = np.random.RandomState(seed)
    bs = bys[rng.randint(0, len(uu), (NB, len(uu)))].mean(1)
    ci = [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
    v = "ADOPT" if ci[1] < 0 else "HARMFUL" if ci[0] > 0 else "INDISTINGUISHABLE"
    return round(float(diff.mean()), 4), ci, round(float((bs < 0).mean()), 4), v


# ------------------------------------------------------------------ the mse baseline
BASE = {}
for s in BASE_SEEDS:
    P, why = assemble(lambda sd, f, s=s: (f"{W}/screen_normalsfix_s{s}_f{f}"
                                          if os.path.exists(f"{W}/screen_normalsfix_s{s}_f{f}.npy")
                                          else None), s)
    if P is not None:
        BASE[s] = P
assert BASE, "no mse baseline seeds found"
BASE_ENS = np.mean(list(BASE.values()), 0)
base_fold_mean = np.array([[float(np.linalg.norm((BASE[s] - GT)[FOLD == f], axis=-1).mean())
                            for f in range(5)] for s in sorted(BASE)])
print(f"mse baseline: {len(BASE)} seeds, per-seed pooled "
      f"{[round(mle(BASE[s]), 4) for s in sorted(BASE)]}, "
      f"{len(BASE)}-seed ensemble {mle(BASE_ENS):.4f} mm")
print(f"  per-fold seed mean {np.round(base_fold_mean.mean(0), 4).tolist()}")
print(f"  per-fold seed sd   {np.round(base_fold_mean.std(0, ddof=1), 4).tolist()}")

out = {"question": "does training against the metric beat training against its square?",
       "control": "VARIANT, data, LR, epochs and deep-supervision weights all identical; "
                  "LOSSFN is the only difference, and LOSSFN=mse is bit-identical to the "
                  "shipped objective",
       "n_ears": E, "n_bootstrap": NB,
       "mse_baseline": {"seeds": sorted(BASE),
                        "per_seed_pooled_mm": [round(mle(BASE[s]), 4) for s in sorted(BASE)],
                        "seed_ensemble_mm": round(mle(BASE_ENS), 4),
                        "per_fold_seed_mean_mm": [round(x, 4) for x in base_fold_mean.mean(0)],
                        "per_fold_seed_sd_mm": [round(x, 4) for x in base_fold_mean.std(0, ddof=1)]},
       "arms": {}, "partial": []}

for arm in ARMS:
    seeds_done = []
    for s in range(3):
        P, why = assemble(lambda sd, f, a=arm, s=s: find(a, s, f), s)
        if P is None:
            got = sum(1 for f in range(5) if find(arm, s, f))
            if got:
                out["partial"].append(f"{arm} seed {s}: {got}/5 folds")
                pf = {f: round(float(np.linalg.norm(
                    np.load(find(arm, s, f) + ".npy").astype(np.float64)
                    - GT[FOLD == f], axis=-1).mean()), 4) for f in range(5) if find(arm, s, f)}
                bm = {f: round(float(base_fold_mean.mean(0)[f]), 4) for f in pf}
                print(f"\n{arm} seed {s}: {got}/5 folds -- INCOMPLETE, per fold vs baseline seed mean")
                for f in pf:
                    print(f"    fold {f}: {pf[f]:.4f} vs {bm[f]:.4f}  delta {pf[f]-bm[f]:+.4f}")
                out["arms"].setdefault(arm, {})[f"seed{s}_partial"] = {
                    "folds_done": sorted(pf), "per_fold_mm": pf,
                    "baseline_seed_mean_mm": bm,
                    "per_fold_delta_mm": {f: round(pf[f] - bm[f], 4) for f in pf}}
            continue
        seeds_done.append(s)
        d_sm, ci_sm, p_sm, v_sm = paired(BASE_ENS, P)
        rec = {"pooled_oof_mm": round(mle(P), 4),
               "per_fold_mm": [round(float(np.linalg.norm((P - GT)[FOLD == f], axis=-1).mean()), 4)
                               for f in range(5)],
               "vs_mse_seed_ensemble": {"delta_mm": d_sm, "ci95": ci_sm,
                                        "p_negative": p_sm, "verdict": v_sm}}
        if s in BASE:
            d, ci, p, v = paired(BASE[s], P)
            rec["vs_mse_same_seed"] = {"delta_mm": d, "ci95": ci, "p_negative": p, "verdict": v}
        out["arms"].setdefault(arm, {})[f"seed{s}"] = rec
        print(f"\n{arm} seed {s}: pooled {mle(P):.4f} mm")
        print(f"    vs mse {len(BASE)}-seed ensemble: {d_sm:+.4f}  CI {ci_sm}  {v_sm}")
        if s in BASE:
            print(f"    vs mse same seed        : {rec['vs_mse_same_seed']['delta_mm']:+.4f}"
                  f"  CI {rec['vs_mse_same_seed']['ci95']}"
                  f"  {rec['vs_mse_same_seed']['verdict']}")
    if len(seeds_done) >= 2:
        Pe = np.mean([assemble(lambda sd, f, a=arm, s=s: find(a, s, f), s)[0]
                      for s in seeds_done], 0)
        # SEED-MATCHED: the mse side must use the SAME seeds. Averaging 2 arm seeds against
        # a 3-seed mse ensemble measures how many seeds each side had, not the objective --
        # it reported dist as +0.0012 when the matched comparison is -0.0089. This is the
        # mirror of the single-seed-vs-3-seed error already fixed above; both directions of
        # the mistake are easy and both were made.
        matched = [s for s in seeds_done if s in BASE]
        MB = np.mean([BASE[s] for s in matched], 0)
        d, ci, p, v = paired(MB, Pe)
        out["arms"][arm]["seed_ensembled"] = {
            "n_seeds": len(seeds_done), "seeds": matched, "mle_mm": round(mle(Pe), 4),
            "mse_same_seeds_mm": round(mle(MB), 4),
            "delta_vs_mse_same_seeds_mm": d, "ci95": ci, "p_negative": p, "verdict": v,
            "mse_all_seed_ensemble_mm": round(mle(BASE_ENS), 4)}
        print(f"\n{arm} {len(seeds_done)}-seed ensemble (seeds {matched}): {mle(Pe):.4f} mm"
              f"  vs mse same seeds {mle(MB):.4f}  delta {d:+.4f}  CI {ci}  {v}")
        if len(matched) < len(BASE):
            print(f"    (mse {len(BASE)}-seed ensemble is {mle(BASE_ENS):.4f}; NOT the "
                  f"comparison -- different seed counts)")

full = {a: v for a, v in out["arms"].items() if "seed_ensembled" in v or "seed0" in v}
if not full:
    out["verdict"] = "INCOMPLETE"
    out["conclusion"] = ("No arm has a complete 5-fold pooled OOF yet. Per-fold partials "
                         "above are compared against the baseline's SEED MEAN, which is "
                         "the right anchor while the new arm has only one seed.")
else:
    def score_of(v):
        """A seed-ensembled record stores mle_mm; a single-seed record pooled_oof_mm."""
        r = v.get("seed_ensembled") or v.get("seed0") or {}
        return r.get("mle_mm", r.get("pooled_oof_mm", 9e9))

    best = min(full, key=lambda a: score_of(full[a]))
    rec = full[best].get("seed_ensembled") or full[best]["seed0"]
    if "delta_vs_mse_same_seeds_mm" in rec:   # seed-MATCHED ensemble comparison
        dv, vv = rec["delta_vs_mse_same_seeds_mm"], rec["verdict"]
        anchor = f"the mse ensemble over the SAME seeds {rec['seeds']}"
    else:
        # A ONE-SEED arm must not be compared against a THREE-SEED ensemble: that measures
        # seed-ensembling, not the loss. The seed-matched row is the only fair one here.
        sm = rec.get("vs_mse_same_seed") or rec["vs_mse_seed_ensemble"]
        dv, vv = sm["delta_mm"], sm["verdict"]
        anchor = ("the mse arm at the SAME seed" if "vs_mse_same_seed" in rec
                  else f"the mse {len(BASE)}-seed ensemble")
    out["verdict"] = vv
    out["best_arm"] = best
    out["headline_anchor"] = anchor
    out["fair_comparison_note"] = (
        "Every headline comparison is SEED MATCHED. Mismatching the seed count measures "
        "seed-ensembling rather than the objective, and it misleads in BOTH directions: a "
        "1-seed dist arm against the 3-seed mse ensemble reads +0.0286mm (arm looks bad), "
        "while a 2-seed dist ensemble against the 3-seed mse ensemble reads +0.0012mm when "
        "the matched 2-vs-2 comparison is -0.0089mm (arm looks neutral instead of ahead). "
        "Both errors were made in this file before being fixed.")
    out["conclusion"] = (
        f"Best arm {best} at {score_of(full[best])} mm, {dv:+} vs {anchor}. " + (
            "Training against the metric rather than its square is a real gain, and it is "
            "architecture-independent -- the same change should be tried on kpconv and "
            "ptv3, where a correlated improvement in every ensemble member compounds."
            if vv == "ADOPT" else
            "The objective mismatch is real in the distribution but does not pay in "
            "millimetres. Combined with metric_alignment.py, the mean/median gap is closed "
            "both post hoc and at training time."
            if vv == "INDISTINGUISHABLE" else
            "Training against the metric is WORSE. The likely cause is optimisation, not "
            "the functional: a distance loss has bounded gradients that do not shrink near "
            "convergence, and the LR was deliberately not retuned."))

# ------------------------------------------------------- multiplicity, stated not hidden
from math import comb  # noqa: E402
per_seed = [r["vs_mse_same_seed"]["delta_mm"]
            for a in out["arms"].values() for k, r in a.items()
            if k.startswith("seed") and "_partial" not in k and "vs_mse_same_seed" in r]
if per_seed:
    n = len(per_seed)
    kneg = sum(1 for v in per_seed if v < 0)
    sign_p = sum(comb(n, i) for i in range(kneg, n + 1)) / 2 ** n
    out["pooled_across_arms"] = {
        "note": ("dist and phuber are not independent tests -- both are sub-quadratic on "
                 "the tail and succeed or fail for the same reason -- so the honest "
                 "summary is the pooled sign test over every seed of either arm against "
                 "mse at that same seed."),
        "per_seed_deltas_mm": per_seed, "n_runs": n, "n_negative": kneg,
        "mean_delta_mm": round(sum(per_seed) / n, 4),
        "sign_test_p_one_sided": round(sign_p, 4)}
    print(f"\npooled across arms: {kneg}/{n} runs negative, mean "
          f"{sum(per_seed)/n:+.4f} mm, sign test p = {sign_p:.4f}")

adopt_marginal = [k for k, v in out["arms"].items()
                  for kk, r in v.items() if kk == "seed_ensembled"
                  and r.get("verdict") == "ADOPT" and r["ci95"][1] > -0.005]
out["multiplicity_caveat"] = (
    "THE PER-ARM VERDICTS ABOVE ARE NOMINAL AND NOT MULTIPLICITY-CORRECTED. Two arms were "
    "run and each was inspected at 1, 2 and 3 seeds, so at least six tests were performed "
    "on the same hypothesis. Any verdict that flips to ADOPT on a barely-excluded interval "
    "is optional stopping: the nominal p-value does not hold under a procedure that looks "
    "repeatedly and stops at the first crossing. "
    + (f"Currently marginal: {adopt_marginal} (CI upper bound within 0.005mm of zero). "
       if adopt_marginal else "")
    + "Treat the pooled sign test as the summary and require a pre-registered confirmation "
      "run before adopting any single arm.")
out["caveats"] = [
    "Raw-network OOF: no TTA, no surface projection, no dense-SSM blend. Comparable to the "
    "1.2292mm mse baseline row, NOT to the 1.1776mm shipped figure.",
    "A 5-fold x 1-seed arm is a screen. It pools all 340 ears so it is not one-fold "
    "screening, but it is under-powered on the SEED axis; the baseline's per-fold seed sd "
    "is reported so the delta can be read against it.",
    "LR and deep-supervision weights were NOT retuned for the new losses, deliberately, so "
    "the arm differs in one thing. A harmful result therefore does not separate the "
    "functional from its optimisation.",
    "phuber's delta=1mm was chosen a priori from the pooled median error, not tuned."]
json.dump(out, open("research/results/loss_verdict.json", "w"), indent=1)
print(f"\n{out['verdict']}: {out['conclusion']}")
print("\nwrote research/results/loss_verdict.json")
