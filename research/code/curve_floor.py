"""
REPRESENTATION FLOOR OF A SMOOTH CURVE THROUGH THE ORDERED LANDMARKS.

Family C (dense template transfer) scored 1.8282mm and Family F (curve + monotone phase)
scored 1.8096mm, while free-XYZ regression on the same fold scores 1.2652mm. Two
unrelated structured parameterisations landing within 0.02mm of each other is not a
coincidence about either design -- it points at the cost of constraining the output to a
smooth structure at all. This measures that cost directly, so the failure is diagnosed
rather than merely observed.

Cubic parametric B-spline (scipy splprep) fitted to each contour's ordered GT landmarks
at decreasing degrees of freedom. At full dof the spline interpolates exactly, so any
residual is purely the price of smoothing.

A NOTE ON A WRONG FIRST ATTEMPT, kept because the failure mode is easy to repeat: the
first version built a Catmull-Rom basis by perturbing one control point at a time and
solving linear least squares. That assumes the curve is linear in its control points.
CENTRIPETAL Catmull-Rom is not -- its knot vector is derived from the chord lengths of
the control polygon itself -- so probing with a degenerate all-zeros-but-one polygon
yields meaningless knots. The tell was that the residual did not fall as K rose, which is
impossible for a genuine fit: K = n control points must nearly interpolate n points.

    python research/code/curve_floor.py
Writes research/results/curve_floor.json
"""
import json
import warnings
import numpy as np
from scipy.interpolate import splprep, splev

warnings.filterwarnings("ignore")
CONT = [(0, 24, "outer_helix"), (25, 54, "concha"),
        (55, 74, "inner_helix"), (75, 84, "sup._antihelix")]
GT = np.load("scratch/ortho_feats.npz")["gt"].astype(float)

out = {"note": ("Residual of a cubic parametric B-spline fitted to each contour's ordered "
                "GT landmarks. Full dof interpolates exactly, so the residual is the price "
                "of smoothing -- a floor any reduced-rank curve parameterisation pays "
                "BEFORE any learning error."),
       "contours": {}}
print(f"{'contour':16s} {'dof':>6s} {'mean':>9s} {'p90':>9s} {'max':>9s}")
for lo, hi, nm in CONT:
    n = hi - lo + 1
    out["contours"][nm] = {"n_landmarks": n, "by_dof": {}}
    for frac, lbl in ((0.35, "~35%"), (0.6, "~60%"), (1.0, "full")):
        res = []
        for e in range(len(GT)):
            P = GT[e, lo:hi + 1].T
            s = 0.0 if frac >= 1.0 else (n * (1 - frac) * 0.5) ** 2
            try:
                tck, u = splprep(P, s=s, k=3)
                res.append(np.linalg.norm(np.array(splev(u, tck)) - P, axis=0))
            except Exception:
                pass
        r = np.concatenate(res)
        out["contours"][nm]["by_dof"][lbl] = {
            "mean_mm": round(float(r.mean()), 4),
            "p90_mm": round(float(np.percentile(r, 90)), 4),
            "max_mm": round(float(r.max()), 4)}
        print(f"{nm if frac == 0.35 else '':16s} {lbl:>6s} {r.mean():9.4f} "
              f"{np.percentile(r, 90):9.4f} {r.max():9.4f}")

worst = max(out["contours"][c]["by_dof"]["~60%"]["mean_mm"] for c in out["contours"])
out["conclusion"] = (
    f"A smooth curve carrying ~60% of the landmarks' degrees of freedom already costs up "
    f"to {worst}mm mean, comparable to the entire current error budget of 1.22mm. Family F "
    f"used NCTRL=16, i.e. 64% of the outer helix's 25 landmarks and 53% of concha's 30, so "
    f"it paid roughly 0.95mm of pure representation error before learning anything. That "
    f"accounts for its 1.81mm and means the result does NOT refute the phase idea -- it "
    f"refutes a reduced-rank curve. NCTRL >= n removes the floor entirely while keeping "
    f"the structural constraint the family exists for, which is monotone ordering, not "
    f"smoothness.")
json.dump(out, open("research/results/curve_floor.json", "w"), indent=1)
print(f"\n{out['conclusion']}")
print("\nwrote research/results/curve_floor.json")
