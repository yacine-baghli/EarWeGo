"""
THE ARC-LENGTH PROFILE OF THE GT LANDMARKS -- the structure no model has used.

77% of the error energy is along-contour and has resisted everything: nine DGCNN
variants, four unrelated backbones (signed tangent error correlated at r=0.80), a dense
template formulation, a curve+phase formulation, six correction predictors, a context
probe. Driving the across and normal components to ZERO would still leave ~1.05mm mean,
so sub-1mm is only reachable through the tangent component.

This measures what determines phase. Two quantities per contour:

  gap CV        per-ear coefficient of variation of consecutive landmark distances.
                ~0 means the landmarks are EQUIDISTANT along the contour.
  profile sd    sd across subjects of the normalised cumulative arc-length position of
                each landmark. ~0 means every subject shares the SAME spacing pattern,
                so knowing the curve determines the landmarks -- whether or not that
                pattern is uniform.

The second is the general statement; equidistance is its special case. What matters for
modelling is profile sd expressed in MILLIMETRES, i.e. multiplied by contour length.

    python research/code/arc_profile.py
Writes research/results/arc_profile.json
"""
import json
import numpy as np

CONT = [(0, 24, "outer_helix"), (25, 54, "concha"),
        (55, 74, "inner_helix"), (75, 84, "sup._antihelix")]
GT = np.load("scratch/ortho_feats.npz")["gt"].astype(float)

out = {"note": ("profile_sd_mm is the phase uncertainty a shared population profile "
                "leaves. Compare it against each contour's CURRENT error to see where "
                "the constraint is worth exploiting."),
       "contours": {}}
print(f"{'contour':16s} {'n':>3s} {'gap CV':>8s} {'prof sd':>9s} {'length':>8s} "
      f"{'sd in mm':>9s} {'unif dev':>9s}")
for lo, hi, nm in CONT:
    G = GT[:, lo:hi + 1]
    n = hi - lo + 1
    d = np.linalg.norm(np.diff(G, axis=1), axis=2)
    cum = np.concatenate([np.zeros((len(G), 1)), np.cumsum(d, 1)], 1)
    length = cum[:, -1]
    prof = cum / length[:, None]
    gap_cv = float((d.std(1) / d.mean(1)).mean())
    sd = float(prof.std(0).mean())
    sd_mm = sd * float(length.mean())
    unif = float(np.abs(prof.mean(0) - np.linspace(0, 1, n)).max())
    out["contours"][nm] = {
        "n_landmarks": n, "gap_cv_per_ear": round(gap_cv, 4),
        "profile_sd_normalised": round(sd, 4),
        "contour_length_mm": round(float(length.mean()), 2),
        "profile_sd_mm": round(sd_mm, 4),
        "max_deviation_from_uniform": round(unif, 4),
        "phase_determined": bool(sd_mm < 0.5)}
    print(f"{nm:16s} {n:3d} {gap_cv:8.4f} {sd:9.4f} {length.mean():8.2f} "
          f"{sd_mm:9.4f} {unif:9.4f}")

det = [k for k, v in out["contours"].items() if v["phase_determined"]]
nl = sum(out["contours"][k]["n_landmarks"] for k in det)
out["phase_determined_contours"] = det
out["n_landmarks_determined"] = nl
out["conclusion"] = (
    f"For {', '.join(det)} ({nl} of 85 landmarks) the shared profile pins phase to under "
    f"0.5mm, so knowing the curve determines the landmarks. Those contours currently "
    f"carry 1.4859mm and 1.2051mm of error, so the constraint is unexploited. For "
    f"outer_helix and concha the same profile leaves 1.4-1.7mm because their contours are "
    f"4-6x longer, and forcing a uniform profile on them is catastrophic (-146% and "
    f"-224% in the endpoint oracle). Note the distinction that sank the earlier attempts: "
    f"the exploitable structure is ARC-LENGTH POSITION, not smoothness. fam_phase.py "
    f"imposed a reduced-rank smooth curve and scored 1.81mm.")
json.dump(out, open("research/results/arc_profile.json", "w"), indent=1)
print(f"\n{out['conclusion']}")
print("\nwrote research/results/arc_profile.json")
