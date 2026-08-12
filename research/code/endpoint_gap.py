"""
THE LARGEST UNEXPLOITED GAP: the 8 contour endpoints carry MORE information than the
average landmark, and our model is WORSE on them than on the average landmark.

Three measured facts that only make sense together.

(1) ENDPOINTS ARE MORE IDENTIFIABLE. From info_limit.json's per-landmark table, endpoints
    (0, 24, 25, 54, 55, 74, 75, 84) against the 77 mid-contour landmarks:
        lambda_along (ambiguity in the dominant direction)   0.78x   <- LESS ambiguous
        surface sharpness                                    1.53x   <- MORE distinctive
    That is the one place where local geometry says something along the contour, which
    info_limit otherwise measured at exactly chance (1.00x) over all 85.

(2) OUR MODEL IS WORSE THERE. Same landmarks, our own pooled OOF error: 1.12x the
    mid-contour error. The single worst landmark of all 85 is #74, an inner-helix
    endpoint, at 2.53mm -- and #74 is one of only EIGHT landmarks info_limit classifies as
    geometrically determined. The surface says where it is and we are 2.5mm off.
    Mechanism, not mystery: the loss weights all 85 landmarks equally, so the 8 points
    that gate the placement of 30 others get 8/85 of the gradient and no extra capacity.

(3) FIXING THEM IS WORTH -0.13 TO -0.20mm, AND THE REQUIREMENT IS QUANTIFIED.
    profile_oracle.json: for inner_helix and sup._antihelix the population arc-length
    profile is nearly subject-invariant (profile sd 0.32mm / 0.12mm), so two endpoints
    determine the other 28 landmarks. With GT endpoints that is 1.4859 -> 1.0634 and
    1.2016 -> 0.3296, i.e. -0.2007mm on all 85 (-0.1264mm if the oracle-exact anchors are
    not credited). The breakeven bar under purely tangential endpoint error is 0.77mm for
    inner_helix, which today carries 1.6887mm -- 2.2x too much -- and 1.27mm for
    sup._antihelix, which carries 0.988mm and is already inside its bar.

So the target is explicit: HALVE the along-contour error on 4 landmarks (55, 74, 75, 84).
That is not a new architecture, it is a re-allocation of capacity and loss weight toward
the points that gate everything else, at a resolution the 2048-point backbones never see.

This file recomputes (1) and (2) from the current best prediction so the claim tracks the
pipeline rather than a stale artefact, and reports what each endpoint would need.

    python research/code/endpoint_gap.py
Writes research/results/endpoint_gap.json
"""
import json
import os
import numpy as np

W = "scratch"
CONT = [(0, 24, "outer_helix"), (25, 54, "concha"),
        (55, 74, "inner_helix"), (75, 84, "sup._antihelix")]
ENDS = [i for lo, hi, _ in CONT for i in (lo, hi)]
MIDS = [i for i in range(85) if i not in ENDS]
# breakeven bars on PURELY TANGENTIAL endpoint error, from profile_oracle.json
BARS = {"inner_helix": 0.77, "sup._antihelix": 1.27}

of = np.load(f"{W}/ortho_feats.npz")
GT, T = of["gt"].astype(np.float64), of["t"].astype(np.float64)
PRED = os.environ.get("PRED", f"{W}/ensemble_best_proj.npy")
if not os.path.exists(PRED):
    PRED = f"{W}/ensemble5_proj.npy"
P = np.load(PRED).astype(np.float64)
print(f"prediction: {PRED}  pooled {np.linalg.norm(P - GT, axis=2).mean():.4f} mm")

err = np.linalg.norm(P - GT, axis=2)                      # (E,85)
tan = np.abs(np.einsum("elk,elk->el", P - GT, T))          # along-contour magnitude

IL = json.load(open("research/results/info_limit.json"))["per_landmark"]
det = set(json.load(open("research/results/info_limit.json"))["geometrically_determined"])


def mean_of(idx, arr):
    return float(arr[:, idx].mean())


out = {"prediction": PRED,
       "pooled_mm": round(float(err.mean()), 4),
       "claim": ("endpoints are more identifiable from local geometry than mid-contour "
                 "landmarks, and our model is worse on them"),
       "endpoints": ENDS,
       "identifiability_from_info_limit": {}, "our_error": {}, "per_endpoint": {}}

print(f"\n{'':24s} {'endpoints(8)':>13s} {'mid-contour(77)':>16s} {'ratio':>7s}")
for key, lbl in (("lam_along", "lambda ALONG (ambig)"), ("S", "surface sharpness"),
                 ("lam", "lambda (all dir)"), ("nn1_gt", "descriptor match mm")):
    a = float(np.mean([IL[key][i] for i in ENDS]))
    b = float(np.mean([IL[key][i] for i in MIDS]))
    out["identifiability_from_info_limit"][lbl] = {"endpoints": round(a, 4),
                                                   "mid": round(b, 4),
                                                   "ratio": round(a / b, 3)}
    print(f"{lbl:24s} {a:13.4f} {b:16.4f} {a / b:7.3f}")
for arr, lbl in ((err, "our error mm"), (tan, "our ALONG-contour mm")):
    a, b = mean_of(ENDS, arr), mean_of(MIDS, arr)
    out["our_error"][lbl] = {"endpoints": round(a, 4), "mid": round(b, 4),
                             "ratio": round(a / b, 3)}
    print(f"{lbl:24s} {a:13.4f} {b:16.4f} {a / b:7.3f}")

print(f"\n{'lm':>4s} {'contour':15s} {'err':>7s} {'along':>7s} {'bar':>6s} {'need':>7s} {'det?':>5s}")
for lo, hi, nm in CONT:
    for i in (lo, hi):
        e, t = float(err[:, i].mean()), float(tan[:, i].mean())
        bar = BARS.get(nm)
        need = "" if bar is None else ("OK" if t <= bar else f"{t / bar:.2f}x")
        out["per_endpoint"][i] = {"contour": nm, "err_mm": round(e, 4),
                                  "along_mm": round(t, 4), "tangential_bar_mm": bar,
                                  "over_bar_factor": (None if bar is None
                                                      else round(t / bar, 2)),
                                  "geometrically_determined": i in det}
        print(f"{i:4d} {nm:15s} {e:7.4f} {t:7.4f} "
              f"{('-' if bar is None else f'{bar:.2f}'):>6s} {need:>7s} "
              f"{'yes' if i in det else '':>5s}")

worst = max(range(85), key=lambda i: err[:, i].mean())
out["worst_landmark"] = {"index": int(worst), "err_mm": round(float(err[:, worst].mean()), 4),
                         "is_endpoint": worst in ENDS,
                         "geometrically_determined": worst in det}
out["prize"] = {
    "source": "research/results/profile_oracle.json",
    "gt_endpoints_k2_gain_all85_mm": -0.2007,
    "gt_endpoints_k2_gain_not_crediting_anchors_mm": -0.1264,
    "contours_unlocked": ["inner_helix", "sup._antihelix"],
    "n_landmarks_unlocked": 30,
    "requirement": ("halve the along-contour error on landmarks 55 and 74 (inner_helix); "
                    "sup._antihelix's endpoints 75/84 are already at or inside their bar")}
out["conclusion"] = (
    f"Endpoints are {1 / out['identifiability_from_info_limit']['lambda ALONG (ambig)']['ratio']:.2f}x "
    f"LESS ambiguous along the contour and "
    f"{out['identifiability_from_info_limit']['surface sharpness']['ratio']:.2f}x sharper than "
    f"mid-contour landmarks, yet our error on them is "
    f"{out['our_error']['our error mm']['ratio']:.2f}x higher. Landmark {worst} is the worst "
    f"of all 85 at {out['worst_landmark']['err_mm']}mm and is one of only 8 that info_limit "
    f"calls geometrically determined. The loss weights all 85 equally, so the 8 landmarks "
    f"that gate the arc-length placement of 30 others receive 8/85 of the gradient. This is "
    f"the largest gap in the repository between information that demonstrably EXISTS in the "
    f"surface and information the model is not using.")
out["caveats"] = [
    "lambda, sharpness and the descriptor match are DESCRIPTIVE quantities computed at the "
    "true landmark (info_limit's own caveat); only its section-1 matcher is held out.",
    "The endpoint/mid split is 8 landmarks against 77, so the endpoint means carry roughly "
    "3x the standard error of the mid-contour means.",
    "The -0.2007mm prize assumes GT-exact endpoints. anchor_deployable.json measured that "
    "reparameterising between PREDICTED anchors is harmful at every k with today's endpoint "
    "error -- which is the same statement as being 2.2x outside the bar, not a contradiction.",
    "Sharpness being higher at endpoints is consistent with locality_limit.json: sharpness "
    "helps ACROSS and hurts ALONG. The load-bearing number here is lambda_along, which is "
    "directional and favours the endpoints independently of sharpness."]
json.dump(out, open("research/results/endpoint_gap.json", "w"), indent=1)
print(f"\n{out['conclusion']}")
print("\nwrote research/results/endpoint_gap.json")
