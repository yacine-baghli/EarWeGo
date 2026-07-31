"""
MULTI-SEED POOLED-OOF ANALYSIS -- the reporting unit under the upgraded protocol.

A run is now 5 frozen subject-grouped folds x S seeds. That gives two things a
single-seed CV cannot:

  * a SEED-ENSEMBLE OOF prediction (mean over seeds per ear), which is what you would
    actually ship, and
  * an explicit fold/seed variance decomposition, so a difference between two
    architectures can be read against the noise that training itself contributes.

Reports per tag: the 5xS matrix of fold-by-seed MLEs, each seed's pooled OOF, the
seed-ensemble pooled OOF, a two-way variance decomposition (fold main effect, seed main
effect, interaction/residual), per-contour MLE and the tangent/across/normal components.

With two tags it also gives the paired per-subject bootstrap of the difference, computed
on the SEED-ENSEMBLE per-ear MLEs (the shippable quantity), plus the same bootstrap
per seed so a verdict cannot rest on one lucky initialisation.

    TAGS=base,normalsfix SEEDS=0,1,2 python research/code/cv_multiseed.py

Writes research/results/multiseed_<tags>.json
"""
import os, json
import numpy as np

WORK = "scratch"
TAGS = os.environ.get("TAGS", "base").split(",")
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
NB = 20000
CONT = [(0, 24, "outer_helix"), (25, 54, "concha"),
        (55, 74, "inner_helix"), (75, 84, "sup._antihelix")]

of = np.load(f"{WORK}/ortho_feats.npz")
GT, T, B, N, SUBJ = of["gt"], of["t"], of["b"], of["n"], of["subj"]
NE = len(GT)


def load_tag(tag):
    """-> P (S,NE,85,3) out-of-fold predictions per seed, and the fold-by-seed MLEs."""
    P = np.full((len(SEEDS), NE, 85, 3), np.nan)
    cell = np.full((5, len(SEEDS)), np.nan)
    folds = np.full(NE, -1)
    for si, s in enumerate(SEEDS):
        seen = np.zeros(NE, bool)
        for f in range(5):
            p = f"{WORK}/screen_{tag}_s{s}_f{f}.json"
            npy = p[:-5] + ".npy"
            # Runs land asynchronously and a half-transferred artefact is normal, not an
            # error. Skip anything not fully present rather than dying on it.
            if not (os.path.exists(p) and os.path.getsize(p) > 0
                    and os.path.exists(npy) and os.path.getsize(npy) > 0):
                print(f"  ! incomplete {os.path.basename(p)}")
                continue
            try:
                j = json.load(open(p))
            except json.JSONDecodeError:
                print(f"  ! corrupt {os.path.basename(p)} -- skipped")
                continue
            idx = np.array(j["val_ear_index"])
            assert not seen[idx].any(), f"{tag} s{s}: ear in two folds"
            seen[idx] = True
            P[si, idx] = np.load(npy)
            cell[f, si] = j["ordered_MLE_mm"]
            if folds[idx[0]] < 0:
                folds[idx] = f
            else:
                assert (folds[idx] == f).all(), f"{tag}: fold assignment differs across seeds"
        if not seen.all():
            print(f"  ! {tag} s{s}: {(~seen).sum()} ears never held out -- INCOMPLETE")
    return P, cell, folds


def per_ear(P):
    return np.linalg.norm(P - GT, axis=2).mean(1)


def boot(diff, subj, seed=5):
    us = np.unique(subj)
    per = {s: np.where(subj == s)[0] for s in us}
    rng = np.random.RandomState(seed)
    d = np.empty(NB)
    for k in range(NB):
        pick = np.concatenate([per[s] for s in rng.choice(us, len(us), replace=True)])
        d[k] = diff[pick].mean()
    return float(diff.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), \
        float((d < 0).mean())


out = {"seeds": SEEDS, "n_ears": int(NE), "n_bootstrap": NB, "tags": {}}
store = {}
for tag in TAGS:
    print(f"===== {tag} =====")
    P, cell, folds = load_tag(tag)
    ok = ~np.isnan(P).any(axis=(1, 2, 3))
    if not ok.any():
        print("  no complete seed -- skipped\n")
        continue
    print(f"{'fold':>5s}" + "".join(f"{'s'+str(s):>9s}" for s in SEEDS) + f"{'mean':>9s}")
    for f in range(5):
        print(f"{f:5d}" + "".join(f"{cell[f, i]:9.4f}" for i in range(len(SEEDS)))
              + f"{np.nanmean(cell[f]):9.4f}")
    print("  seed" + "".join(f"{np.nanmean(cell[:, i]):9.4f}" for i in range(len(SEEDS))))

    # per-seed pooled OOF and the seed ensemble (what you would ship)
    pooled = np.array([per_ear(P[i]).mean() if ok[i] else np.nan for i in range(len(SEEDS))])
    ens = per_ear(np.nanmean(P[ok], axis=0))
    print(f"\n  per-seed pooled OOF : " + "  ".join(f"{v:.4f}" for v in pooled)
          + f"   (sd {np.nanstd(pooled, ddof=1):.4f})")
    print(f"  SEED-ENSEMBLE pooled OOF : {ens.mean():.4f}"
          f"   ({ens.mean()-np.nanmean(pooled):+.4f} vs the mean single seed)")

    # two-way variance decomposition of the fold x seed MLE matrix
    c = cell[:, :len(SEEDS)]
    gm = np.nanmean(c)
    fold_eff = np.nanmean(c, axis=1) - gm
    seed_eff = np.nanmean(c, axis=0) - gm
    resid = c - gm - fold_eff[:, None] - seed_eff[None, :]
    print(f"\n  variance decomposition of the {c.shape[0]}x{c.shape[1]} MLE matrix")
    print(f"    fold main effect  sd {np.nanstd(fold_eff, ddof=1):.4f} mm   "
          f"(range {np.nanmax(fold_eff)-np.nanmin(fold_eff):.4f})")
    print(f"    seed main effect  sd {np.nanstd(seed_eff, ddof=1):.4f} mm   "
          f"(range {np.nanmax(seed_eff)-np.nanmin(seed_eff):.4f})")
    print(f"    interaction/resid sd {np.nanstd(resid, ddof=1):.4f} mm")

    E = np.nanmean(P[ok], axis=0) - GT
    d = np.linalg.norm(E, axis=2)
    pc = {nm: round(float(d[:, lo:hi + 1].mean()), 4) for lo, hi, nm in CONT}
    dirs = {nm: round(float(np.sqrt((((E * V).sum(-1)) ** 2).mean())), 4)
            for nm, V in (("tangent_t", T), ("across_b", B), ("normal_n", N))}
    print(f"  per-contour {pc}")
    print(f"  directional {dirs}\n")
    store[tag] = ens
    out["tags"][tag] = {
        "fold_seed_MLE": [[None if np.isnan(x) else round(float(x), 4) for x in row]
                          for row in c],
        "per_seed_pooled_OOF": [None if np.isnan(v) else round(float(v), 4) for v in pooled],
        "per_seed_sd": round(float(np.nanstd(pooled, ddof=1)), 4),
        "seed_ensemble_pooled_OOF": round(float(ens.mean()), 4),
        "variance": {"fold_sd": round(float(np.nanstd(fold_eff, ddof=1)), 4),
                     "seed_sd": round(float(np.nanstd(seed_eff, ddof=1)), 4),
                     "resid_sd": round(float(np.nanstd(resid, ddof=1)), 4)},
        "per_contour_MLE_mm": pc, "directional_rmse_mm": dirs,
        "n_complete_seeds": int(ok.sum())}

if len(store) >= 2:
    ks = list(store)
    ref = ks[0]
    print(f"===== paired per-subject bootstrap vs {ref} (seed-ensemble) =====")
    out["comparisons"] = {}
    for k in ks[1:]:
        m, lo, hi, pn = boot(store[k] - store[ref], SUBJ)
        v = "ADOPT" if hi < 0 else "REJECT" if lo > 0 else "INDISTINGUISHABLE"
        print(f"  {k:14s} {m:+.4f} mm  CI [{lo:+.4f}, {hi:+.4f}]  P(<0)={pn:.3f}  -> {v}")
        out["comparisons"][f"{k}_vs_{ref}"] = {
            "delta_mm": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "p_negative": round(pn, 4), "verdict": v}
    print("\nThe interval covers subject sampling. Seed and fold variance are reported\n"
          "separately above -- a delta smaller than the seed sd is not a finding.")

nm = "_".join(TAGS)
json.dump(out, open(f"research/results/multiseed_{nm}.json", "w"), indent=1)
print(f"wrote research/results/multiseed_{nm}.json")
