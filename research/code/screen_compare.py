"""
SCREENING COMPARISON — turn the raw per-run reports into a decision table.

Reads every scratch/screen_<variant>_s<seed>_f<fold>.json produced by gpu_screen.py
(plus the matching .npy of validation predictions) and reports, against a single
reference run:

  ordered MLE, per-contour MLE, the tangent/across/normal directional components,
  fresh-sample prediction variance, parameter count, runtime, and a PAIRED
  PER-SUBJECT BOOTSTRAP interval on the MLE difference vs the reference.

Two rules this script exists to enforce:

  1. TRAINING NOISE FIRST. base seed0 vs base seed1 defines the noise band. A variant
     whose gain does not clear that band is not evidence of anything. If the two base
     seeds are not both present the script says so and refuses to rank variants.

  2. PAIRED, SUBJECT-GROUPED. Runs on the same fold share validation ears, so the
     comparison is paired per ear and resampled by SUBJECT (both ears move together),
     never by ear — the two ears of a subject are not independent.

Directional components use a FROZEN orthonormal basis (t, b, n) taken from
scratch/ortho_feats.npz, i.e. built once on the baseline OOF predictions with the mesh
normals. Holding the basis fixed is deliberate: it makes the components comparable
across variants. It is an approximation for a variant whose landmarks moved
appreciably, since the true tangent is defined on that variant's own polyline.

    python research/code/screen_compare.py            # ranks everything found
    REF=base_s0 python research/code/screen_compare.py

Writes research/results/screening.json
"""
import os, json, glob
import numpy as np

WORK = "scratch"
REF = os.environ.get("REF", "base_s0")
NB = 10000
CONT = [(0, 24, "outer_helix"), (25, 54, "concha"),
        (55, 74, "inner_helix"), (75, 84, "sup._antihelix")]

of = np.load(f"{WORK}/ortho_feats.npz")
GT_ALL, T_ALL, B_ALL, N_ALL = of["gt"], of["t"], of["b"], of["n"]
SUBJ = of["subj"]

runs = {}
for p in sorted(glob.glob(f"{WORK}/screen_*.json")):
    r = json.load(open(p))
    key = f"{r['variant']}_s{r['seed']}"
    npy = p[:-5] + ".npy"
    if not os.path.exists(npy):
        print(f"  ! {key}: predictions missing, skipped")
        continue
    idx = np.array(r["val_ear_index"])
    P = np.load(npy).astype(np.float64)
    E = P - GT_ALL[idx]
    d = np.linalg.norm(E, axis=2)                      # (nval, 85)
    assert abs(d.mean() - r["ordered_MLE_mm"]) < 2e-3, \
        f"{key}: predictions disagree with reported MLE ({d.mean():.4f} vs {r['ordered_MLE_mm']})"
    r["_per_ear"] = d.mean(1)
    r["_idx"] = idx
    r["_subj"] = SUBJ[idx]
    # directional RMSE components in the frozen frame (energy, not mean |.|)
    r["_dir"] = {nm: float(np.sqrt((((E * V[idx]).sum(-1)) ** 2).mean()))
                 for nm, V in (("tangent_t", T_ALL), ("across_b", B_ALL), ("normal_n", N_ALL))}
    r["_dir_total"] = float(np.sqrt((d ** 2).mean()))
    runs[key] = r

if not runs:
    raise SystemExit("no screening reports in scratch/ — run run_screen.sh first")
print(f"loaded {len(runs)} run(s): {', '.join(runs)}\n")


def paired_boot(a, b, subj, rng):
    """Paired per-subject bootstrap of mean(b) - mean(a) over the shared ears."""
    us = np.unique(subj)
    per = {s: np.where(subj == s)[0] for s in us}
    diff = b - a
    draws = np.empty(NB)
    for k in range(NB):
        pick = np.concatenate([per[s] for s in rng.choice(us, len(us), replace=True)])
        draws[k] = diff[pick].mean()
    return float(diff.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


# ---------- training-noise band from the two base seeds ----------
noise = None
if "base_s0" in runs and "base_s1" in runs:
    a, b = runs["base_s0"], runs["base_s1"]
    assert np.array_equal(a["_idx"], b["_idx"]), "base seeds ran on different val sets"
    m, lo, hi = paired_boot(a["_per_ear"], b["_per_ear"], a["_subj"], np.random.RandomState(7))
    noise = {"seed_delta_mm": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)],
             "half_width_mm": round((hi - lo) / 2, 4),
             "base_s0_MLE": a["ordered_MLE_mm"], "base_s1_MLE": b["ordered_MLE_mm"]}
    print(f"TRAINING NOISE  base_s0 {a['ordered_MLE_mm']:.4f} vs base_s1 {b['ordered_MLE_mm']:.4f}"
          f"  ->  delta {m:+.4f} mm  [{lo:+.4f}, {hi:+.4f}]")
    print(f"A variant must beat |{abs(m):.4f}| mm AND have a CI excluding zero to count.\n")
else:
    print("!! both base seeds not present -> training noise unknown; variants NOT ranked.\n")

# ---------- table ----------
ref = runs.get(REF)
hdr = (f"{'run':12s} {'MLE':>7s} {'med':>6s} {'P90':>6s} | "
       f"{'outer':>6s} {'conch':>6s} {'inner':>6s} {'antih':>6s} | "
       f"{'rms_t':>6s} {'rms_b':>6s} {'rms_n':>6s} | {'svar':>5s} {'par':>6s} {'min':>5s} | "
       f"{'vs ref':>8s} {'ci95':>18s}")
print(hdr); print("-" * len(hdr))
out = {"reference": REF, "n_bootstrap": NB, "training_noise": noise,
       "basis_note": ("directional components use the frozen (t,b,n) basis from "
                      "ortho_feats.npz, built on the baseline OOF predictions with mesh "
                      "normals, held fixed so components are comparable across variants"),
       "runs": {}}
rows = []
for k, r in runs.items():
    pc = list(r["per_contour_MLE_mm"].values())
    cmp_s, cmp = "        ", None
    if ref is not None and k != REF:
        if np.array_equal(r["_idx"], ref["_idx"]):
            m, lo, hi = paired_boot(ref["_per_ear"], r["_per_ear"], r["_subj"],
                                    np.random.RandomState(11))
            cmp = {"delta_mm": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)],
                   "excludes_zero": bool(lo > 0 or hi < 0),
                   "beats_seed_noise": bool(noise and m < -abs(noise["seed_delta_mm"]) and hi < 0)}
            cmp_s = f"{m:+8.4f}"
            ci_s = f"[{lo:+.4f},{hi:+.4f}]"
        else:
            ci_s = "different fold"
    else:
        ci_s = "(reference)" if k == REF else ""
    print(f"{k:12s} {r['ordered_MLE_mm']:7.4f} {r['median_mm']:6.3f} {r['P90_mm']:6.3f} | "
          f"{pc[0]:6.3f} {pc[1]:6.3f} {pc[2]:6.3f} {pc[3]:6.3f} | "
          f"{r['_dir']['tangent_t']:6.3f} {r['_dir']['across_b']:6.3f} {r['_dir']['normal_n']:6.3f} | "
          f"{r['fresh_sample_pred_variance_mm']:5.3f} {r['params']/1000:6.0f} "
          f"{r['runtime_s']/60:5.1f} | {cmp_s} {ci_s:>18s}")
    out["runs"][k] = {"ordered_MLE_mm": r["ordered_MLE_mm"], "median_mm": r["median_mm"],
                      "P90_mm": r["P90_mm"], "per_contour_MLE_mm": r["per_contour_MLE_mm"],
                      "directional_rmse_mm": {kk: round(vv, 4) for kk, vv in r["_dir"].items()},
                      "total_rmse_mm": round(r["_dir_total"], 4),
                      "fresh_sample_pred_variance_mm": r["fresh_sample_pred_variance_mm"],
                      "params": r["params"], "runtime_s": r["runtime_s"],
                      "epochs": r["epochs"], "fold": r["fold"],
                      "n_val_ears": len(r["_idx"]), "config": r["config"],
                      "train_val_curve": r["train_val_curve"],
                      "vs_reference": cmp}
    if cmp:
        rows.append((k, cmp))

# ---------- promotion verdict ----------
print()
if noise is None:
    print("VERDICT withheld: run base with both seeds before ranking variants.")
else:
    ok = [(k, c) for k, c in rows if c["beats_seed_noise"]]
    ok.sort(key=lambda x: x[1]["delta_mm"])
    if ok:
        print("PROMOTE to 5-fold CV (gain clears seed noise, CI excludes zero):")
        for k, c in ok[:2]:
            print(f"  {k:12s} {c['delta_mm']:+.4f} mm  CI {c['ci95']}")
        out["promote"] = [k for k, _ in ok[:2]]
    else:
        print("PROMOTE none: no variant's gain clears the seed-to-seed noise band.")
        out["promote"] = []
    print("\nOne fold only. These intervals cover val-ear sampling, not fold choice;\n"
          "the promoted variants still need the full 5-fold CV before any claim.")

json.dump(out, open("research/results/screening.json", "w"), indent=1)
print("\nwrote research/results/screening.json")
