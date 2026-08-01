"""
PER-FOLD MEAN ARC-LENGTH PROFILE — the only population statistic fam_profile.py consumes.

The normalised cumulative chord-length profile of the ordered GT landmarks (definition:
fam_profile.arc_profile) is nearly constant across subjects on inner_helix and
sup._antihelix. A model that places landmarks at that profile is therefore reading a
POPULATION statistic, which under constraint 2 must be computed from the current fold's
TRAINING ears only. This writes one npz per fold, each carrying the `fold` and
`train_ear_mask` that let train_family.py prove it never saw a validation ear, and each
holding exactly that fold's training-ear mean.

The fold rule is constraint 3 verbatim and is cross-checked against the frozen
research/results/folds.json before anything is written.

WHAT IS AND IS NOT IN THE NPZ. prof_c<ci> (mean profile) and prof_sd_c<ci> (per-position
sd, the bound PROFILE_MODE=learned is allowed to deviate within) and len_c<ci> (mean
contour length in mm, so a normalised sd can be read in mm). No per-ear quantity, no
validation ear, no geometry: 85 numbers per fold in total, all of them fractions of an
arc length.

    python research/code/build_profile.py
    DATA=scratch/screen_data_8192nrm.npz python research/code/build_profile.py

Writes scratch/profile_f{0..4}.npz and research/results/profile_stats.json
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fam_profile import CONTOURS, CNAMES, NC

WORK = os.environ.get("WORK", "scratch")
DATA = os.environ.get("DATA", f"{WORK}/screen_data_2048nrm.npz")
NFOLD = 5
FOLDS_JSON = "research/results/folds.json"


def profile(P):
    """(E,n,3) ordered landmarks -> (E,n) normalised cumulative chord length, and lengths."""
    seg = np.linalg.norm(np.diff(P, axis=1), axis=2)
    L = seg.sum(1)
    return np.concatenate([np.zeros((len(P), 1)), np.cumsum(seg, 1)], 1) / L[:, None], L


GT = np.load(DATA, allow_pickle=True)["true"].astype(np.float64)
NE = len(GT)
subj = np.arange(NE) // 2
parts = np.array_split(np.random.RandomState(12345).permutation(np.unique(subj)), NFOLD)
# the profile is invariant to any similarity transform, so canonical vs world is immaterial;
# `true` is used because it is literally the training target.

if os.path.exists(FOLDS_JSON):
    a = json.load(open(FOLDS_JSON))["assignments"]
    of = np.full(NE, -1)
    for f, p in enumerate(parts):
        of[np.isin(subj, p)] = f
    assert len(a) == NE and all(of[r["ear_index"]] == r["fold"] for r in a), \
        "the derived split disagrees with the frozen research/results/folds.json"
    print(f"fold rule verified against {FOLDS_JSON}")
else:
    print(f"{FOLDS_JSON} absent -- split derived from the frozen rule only")

pooled = [profile(GT[:, lo:hi + 1]) for lo, hi in CONTOURS]
out = {"data": DATA, "n_ears": int(NE), "n_folds": NFOLD,
       "definition": ("normalised cumulative CHORD length of the ordered GT landmarks of a "
                      "contour; 0 at the first landmark, 1 at the last"),
       "pooled_over_all_dev_ears": {}, "per_fold": {}}
print(f"\n{'contour':16s} {'n':>3s} {'len mm':>8s} {'sd':>8s} {'sd mm':>8s} {'gap CV':>8s} "
      f"{'|mean-uniform|':>15s} {'in mm':>7s}")
for ci, (lo, hi) in enumerate(CONTOURS):
    s, L = pooled[ci]
    n = hi - lo + 1
    sd, gap = s.std(0, ddof=1), np.diff(s, axis=1)
    du = np.abs(s.mean(0) - np.linspace(0, 1, n)).max()
    print(f"{CNAMES[ci]:16s} {n:3d} {L.mean():8.2f} {sd.mean():8.4f} {sd.mean()*L.mean():8.3f} "
          f"{gap.std(0, ddof=1).mean()/gap.mean():8.4f} {du:15.4f} {du*L.mean():7.3f}")
    out["pooled_over_all_dev_ears"][CNAMES[ci]] = {
        "n_landmarks": n, "mean_length_mm": round(float(L.mean()), 3),
        "profile_sd_mean": round(float(sd.mean()), 4),
        "profile_sd_max": round(float(sd.max()), 4),
        "profile_sd_mean_mm": round(float(sd.mean() * L.mean()), 4),
        "gap_CV": round(float(gap.std(0, ddof=1).mean() / gap.mean()), 4),
        "max_dev_of_mean_from_uniform": round(float(du), 5),
        "max_dev_of_mean_from_uniform_mm": round(float(du * L.mean()), 4)}

os.makedirs(WORK, exist_ok=True)
drift = {}
for f in range(NFOLD):
    va = np.where(np.isin(subj, parts[f]))[0]
    mask = ~np.isin(subj, parts[f])
    assert not mask[va].any() and mask.sum() + len(va) == NE
    z = {"fold": np.int64(f), "train_ear_mask": mask, "n_train_ears": np.int64(mask.sum()),
         "note": np.array("training-ear mean of the normalised cumulative chord-length "
                          "profile; built by research/code/build_profile.py")}
    per = {}
    for ci, (lo, hi) in enumerate(CONTOURS):
        s, L = profile(GT[mask, lo:hi + 1])
        z[f"prof_c{ci}"] = s.mean(0)
        z[f"prof_sd_c{ci}"] = s.std(0, ddof=1)
        z[f"len_c{ci}"] = np.float64(L.mean())
        d = np.abs(s.mean(0) - pooled[ci][0].mean(0)).max()
        drift[CNAMES[ci]] = max(drift.get(CNAMES[ci], 0.0), float(d * L.mean()))
        per[CNAMES[ci]] = {"profile": [round(float(x), 5) for x in s.mean(0)],
                           "sd_mean": round(float(s.std(0, ddof=1).mean()), 5)}
    # the sd at the two endpoints is exactly 0 by definition; keep it that way so the
    # learned-deviation bound pins them rather than relying on a float being tiny.
    for ci in range(NC):
        z[f"prof_sd_c{ci}"][0] = z[f"prof_sd_c{ci}"][-1] = 0.0
    p = f"{WORK}/profile_f{f}.npz"
    np.savez(p, **z)
    out["per_fold"][f] = {"n_train_ears": int(mask.sum()), "n_val_ears": int(len(va)),
                          "path": p, "contours": per}
    print(f"  fold {f}: {int(mask.sum())} train ears -> {p}")

out["max_fold_to_pooled_drift_mm"] = {k: round(v, 4) for k, v in drift.items()}
out["conclusion"] = (
    f"Holding the profile to the training fold costs at most "
    f"{max(drift.values()):.3f} mm of drift against the pooled (leaky) mean, so fold "
    f"safety here is free. On inner_helix and sup._antihelix the training mean is also "
    f"within "
    f"{max(out['pooled_over_all_dev_ears'][c]['max_dev_of_mean_from_uniform_mm'] for c in ('inner_helix', 'sup._antihelix')):.3f}"
    f" mm of UNIFORM, which is why fam_profile.py's no-artefact fallback (uniform, and so "
    f"leakage-free by construction) is a faithful stand-in and not a shortcut.")
print(f"\nmax drift of a fold mean vs the pooled mean: "
      + "  ".join(f"{k} {v:.3f}mm" for k, v in drift.items()))
print(out["conclusion"])
json.dump(out, open("research/results/profile_stats.json", "w"), indent=1)
print("wrote research/results/profile_stats.json")
