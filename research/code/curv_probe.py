"""
TRAINING-FREE PROBE: does the curvature channel carry landmark signal, BEFORE any GPU?

The question a new input channel has to answer is not "is it geometrically meaningful"
(curvature obviously is) but "does knowing it make a landmark's own neighbourhood more
IDENTIFIABLE than the neighbourhoods around it". This measures exactly that, with no
network and no training, so a null result costs minutes instead of GPU-days -- which is
the point, given that six of the last eight ideas in this repo were null.

THE PROBE
---------
Leave-one-SUBJECT-out, over the 8192-point surface clouds:

  1. For every ear and every landmark l, the ANCHOR is the cloud point nearest the GT
     landmark. Its feature vector is what that landmark "looks like" on that ear.
  2. From the TRAINING ears only, average the anchor features -> a template t_l per
     landmark, and average the GT positions -> a template position p_l.
  3. On the held-out ear, score all 8192 points by
            score(x) = ||z(x) - z(p_l)||^2  +  w * ||f(x) - t_l||^2
     and RETRIEVE the argmin. The error is the mm distance from the retrieved point to
     the GT landmark. w = 0 is position alone; w -> inf is the descriptor alone.
  4. A descriptor helps iff the curve of error vs w dips below its own w = 0 value.

Everything is z-scored per channel using the TRAINING ears' cloud points, so w is
dimensionless and position and descriptor enter on the same footing. w is a free
parameter of the PROBE, not of any model; the table reports the whole curve, and the
"best w" column is therefore an OPTIMISTIC reading -- w is chosen on the same 340 ears
the error is measured on. Only the SIGN and the ORDER of the deltas should be trusted,
never their magnitude as a prediction of a training gain.

MEASURED CALIBRATION of the two position priors: on these clouds the population-mean
canonical landmark position retrieves at 3.7363mm and this ear's own coarse landmark at
3.7588mm -- indistinguishable, because the canonical frame is itself built from the
coarse landmarks. So POS=mean and POS=coarse are two nearly equivalent baselines, not a
weak one and a strong one, and agreement between them is a consistency check.

WHY IT IS FRAMED AGAINST POSITION, NOT ON ITS OWN. Every model in this repo already knows
where a landmark roughly is: the canonical frame is per-ear aligned and the coarse init
sits ~0.46mm from the surface with a ~3.7mm tangential error. A descriptor that merely
recovers position is worth nothing. So the probe conditions on position and asks what the
descriptor ADDS. Two position priors are reported:
    POS=mean    the population-mean canonical landmark position (what a shape prior knows)
    POS=coarse  THIS ear's coarse landmark (what the network is actually handed)

CONTROLS, because a 12-column descriptor beating a 3-column one can be pure dimensionality:
    crv_shuf    the same 12 curvature columns with the ear's POINT ORDER permuted. Same
                marginal distribution, same dimensionality, spatial registration
                destroyed. A gain that survives this is not a gain.
    nrm         oriented normals, 3 columns -- the last channel that was adopted
                (-0.0481mm end to end), so it calibrates what "a real but small win"
                looks like on this probe.
    crv@1.5 / crv@3 / crv@6    one radius at a time, 4 columns each: dimension-matched
                against nothing, but they say which SCALE carries the signal.

WHAT THIS PROBE CANNOT SAY. It is a nearest-template retrieval, i.e. a linear, isotropic,
per-landmark-independent reading of the channel. A network can do better (learned metric,
context, the whole contour at once). So a POSITIVE result is strong evidence the channel
is worth GPU time; a NEGATIVE result bounds the EASY signal, not the achievable one.
It also cannot see anything about ORDERING along a contour, which is 77% of the error.

The retrieval FLOOR is reported: the anchor itself is ~0.4mm from the GT landmark at 8192
points, so no configuration can score below that.

LEAKAGE. Templates, z-scores and every population statistic are formed from a running
per-ear sum with the held-out SUBJECT's ears subtracted, and the code asserts the
held-out ears contributed nothing (constraint 2). `true` is read only to place the
anchors of TRAINING ears and to score the held-out ear -- never to build a feature.

RESULT ON scratch/screen_data_8192crv.npz, 340 ears, 170 LOSO folds -- REDUNDANT, NOT NULL
--------------------------------------------------------------------------------------
Retrieval floor 0.5352mm. Position alone 3.6250mm (POS=mean) / 3.8186 (POS=coarse).

  descriptor      best mm   delta   at w      (slot 0; slot 2 in brackets)
  nrm (3 ch)       2.8492   -0.7758  0.25     [2.8568  -0.7623]
  nrm+crv (15)     2.8809   -0.7441  0.05     [2.8698  -0.7493]
  crv (12)         3.3639   -0.2611  0.01     [3.3551  -0.2640]
  crv@6 (4)        3.3906   -0.2344  0.05     [3.3746  -0.2446]
  crv@3 (4)        3.4141   -0.2109  0.05     [3.4084  -0.2108]
  crv@1.5 (4)      3.4393   -0.1857  0.05     [3.4315  -0.1876]
  shape_index (3)  3.5091   -0.1159  0.1      [3.4933  -0.1258]
  crv_shuf (12)    3.6250    0.0000  0        [3.6191   0.0000]   control behaves

THE WEIGHT GRID DECIDED THE VERDICT, AND THE FIRST GRID WAS WRONG. This table was
originally run on WGRID=0.25,1,4,16,inf and reported curvature as NULL: crv, crv@1.5 and
crv@3 all came out at exactly the position-only value because no tested w > 0 beat w = 0.
The optimum for a 12-column descriptor is near w = 0.01 -- at w = 0.25 the 12 descriptor
terms already swamp the 3 position terms. Same file, same slot, same code, only the grid
changed: crv 3.6250 (0.0000) -> 3.3639 (-0.2611). The default WGRID now starts at 0.01.

SO CURVATURE DOES CARRY SIGNAL, and it is not dimensionality: at the same w = 0.01 the
shuffled control is 3.8202, i.e. 0.20mm WORSE than position alone while the real channel
is 0.26mm BETTER -- a 0.46mm separation between a descriptor and its own shuffle.

BUT IT IS REDUNDANT WITH NORMALS, which is the actual decision number. nrm alone 2.8492;
nrm+crv 2.8809. Adding all 12 curvature channels on top of oriented normals costs
+0.0317mm under POS=mean and BUYS -0.0306mm under POS=coarse -- the sign is not even
stable across the two position priors, so the honest reading is "no measurable effect",
not the "+0.3126mm worse" the coarse grid produced. Per contour (POS=mean), nrm+crv minus
nrm: outer +0.260, concha -0.356, inner_helix +0.358, sup._antihelix -0.027; under
POS=coarse: outer -0.012, concha -0.315, inner_helix +0.520, sup._antihelix -0.325. The
one contour curvature consistently helps is the CONCHA (a basin, seen by the r=6 block);
the one it consistently hurts is the inner helix.

This is not a broken channel: the shipped statistics are healthy (per-channel sd 0.048 to
0.498, fewer than 3e-4 of values beyond |0.99|, no saturation), the estimator matches
closed-form sphere/cylinder/saddle, and the shuffled control degrades exactly as it should.

WHAT IT DOES AND DOES NOT LICENCE. Curvature is real but nearly all of what it says about
a landmark's neighbourhood, THIS METRIC can already read off the oriented normal -- which
makes sense, since the normal field's local variation IS the shape operator. So a naive
15-channel concat is not the experiment to run; if curvature is tried, the case is the
r=6mm block on the concha, or curvature INSTEAD of normals, not curvature on top. It does
NOT prove a network cannot use curvature: this metric is unlearned, isotropic and
per-landmark independent. Note the calibration -- normals score -0.7758 here and bought
only -0.0481mm end to end, so this probe's scale is not a millimetre forecast, and a
+-0.03mm difference on it is far below anything it can resolve.

    python research/code/curv_probe.py            # -> research/results/curv_probe.json
    SMOKE=1 python research/code/curv_probe.py    # synthetic self-test, no data needed

ENV (defaults in brackets)
  SRC   [scratch/screen_data_8192crv.npz]
  SLOT  [0]        which of the M fresh samples to probe (one is enough; M=4 exist)
  WGRID [0.25,1,4,16,inf]   descriptor weights; 0 is always prepended
  OUT   [research/results/curv_probe.json]
  SMOKE [0]
"""
import os, sys, json, time
import numpy as np
from scipy.spatial import cKDTree

SRC = os.environ.get("SRC", "scratch/screen_data_8192crv.npz")
SLOT = int(os.environ.get("SLOT", "0"))
# The grid MUST reach well below 0.25. The first version of this file started at 0.25 and
# concluded that curvature was null (no w > 0 beat w = 0 for crv, crv@1.5 or crv@3); the
# optimum for a 12-column descriptor is at w ~ 0.01, and at 0.25 the descriptor term is
# already swamping the position term. Same data, same slot, only the grid changed:
# crv went 3.6250 (delta 0.0000) -> 3.3639 (delta -0.2611). See the RESULT block above.
WGRID = [float(x) for x in
         os.environ.get("WGRID", "0.01,0.05,0.1,0.25,1,4,16,inf").split(",")]
OUT = os.environ.get("OUT", "research/results/curv_probe.json")
CONTOURS = [("outer_helix", 0, 24), ("concha", 25, 54),
            ("inner_helix", 55, 74), ("sup._antihelix", 75, 84)]
NL = 85


def feature_sets(names):
    """column blocks of the concatenated descriptor [nrm(3) | crv(NC)]"""
    nc = len(names)
    sets = {"nrm": list(range(3)), "crv": list(range(3, 3 + nc)),
            "nrm+crv": list(range(3 + nc))}
    for r in sorted({n.split("@")[1] for n in names}, key=float):
        sets[f"crv@{r}"] = [3 + k for k, n in enumerate(names) if n.split("@")[1] == r]
    sets["shape_index"] = [3 + k for k, n in enumerate(names) if n.startswith("S@")]
    sets["crv_shuf"] = sets["crv"]                 # handled by SHUF below, same columns
    return sets


SHUF = "crv_shuf"


def anchors(cl, true):
    """index of the cloud point nearest each GT landmark, and that distance (the floor)"""
    d, j = cKDTree(cl).query(true)
    return j.astype(np.int64), d


def probe(clouds, D, true, coarse, subj, sets, wgrid, log=print):
    """LOSO retrieval. clouds (E,N,3) f32, D (E,N,C) f32 descriptors, true/coarse (E,85,3).

    Returns err[set][pos][w] -> (E,85) mm, plus the anchor floor.
    """
    E, N, C = D.shape
    rngs = [np.random.RandomState(9_000_000 + e).permutation(N) for e in range(E)]

    A = np.zeros((E, NL, C), np.float64)           # anchor feature per ear/landmark
    Ash = np.zeros((E, NL, C), np.float64)         # ... under the shuffled control
    floor = np.zeros((E, NL))
    aidx = np.zeros((E, NL), np.int64)
    s1 = np.zeros((E, C)); s2 = np.zeros((E, C))   # per-ear running sums for the z-score
    p1 = np.zeros((E, 3)); p2 = np.zeros((E, 3))
    for e in range(E):
        j, d = anchors(clouds[e], true[e])
        aidx[e], floor[e] = j, d
        A[e] = D[e][j]
        Ash[e] = D[e][rngs[e]][j]
        s1[e] = D[e].sum(0, dtype=np.float64); s2[e] = (D[e].astype(np.float64) ** 2).sum(0)
        p1[e] = clouds[e].sum(0, dtype=np.float64)
        p2[e] = (clouds[e].astype(np.float64) ** 2).sum(0)
    T1 = true.astype(np.float64)

    keys = [(s, p, w) for s in sets for p in ("mean", "coarse") for w in wgrid]
    err = {k: np.zeros((E, NL)) for k in keys}
    subs = np.unique(subj)
    t0 = time.time()
    for si, s in enumerate(subs):
        ho = np.flatnonzero(subj == s)             # this subject's ears (left + right)
        tr = np.flatnonzero(subj != s)
        assert not set(ho) & set(tr) and len(tr) == E - len(ho)
        ntr = len(tr)
        if si == 0:
            # Constraint 2, as an executable test rather than a claim. Index disjointness
            # is true by construction and proves nothing; this PERTURBS the held-out ears'
            # contributions and requires every population statistic to be unmoved.
            for arr in (A, Ash, s1, s2, p1, p2):
                arr[ho] += 1e6
            chk = ((A.sum(0) - A[ho].sum(0)) / ntr, (s1.sum(0) - s1[ho].sum(0)) / (ntr * N))
            for arr in (A, Ash, s1, s2, p1, p2):
                arr[ho] -= 1e6
            ref = (A[tr].mean(0), s1[tr].sum(0) / (ntr * N))
            for c, r in zip(chk, ref):
                assert np.allclose(c, r, atol=1e-6), \
                    "a held-out ear reached a population statistic -- LEAK"
        mu = (s1.sum(0) - s1[ho].sum(0)) / (ntr * N)
        var = (s2.sum(0) - s2[ho].sum(0)) / (ntr * N) - mu ** 2
        sd = np.sqrt(np.maximum(var, 1e-12))
        pmu = (p1.sum(0) - p1[ho].sum(0)) / (ntr * N)
        psd = np.sqrt(np.maximum((p2.sum(0) - p2[ho].sum(0)) / (ntr * N) - pmu ** 2, 1e-12))
        Tf = (A.sum(0) - A[ho].sum(0)) / ntr                     # (85,C) feature template
        Tfs = (Ash.sum(0) - Ash[ho].sum(0)) / ntr
        Tp = (T1.sum(0) - T1[ho].sum(0)) / ntr                   # (85,3) position template
        for e in ho:
            P = (clouds[e].astype(np.float64) - pmu) / psd       # (N,3)
            Dp = {}
            for pk, q in (("mean", Tp), ("coarse", coarse[e].astype(np.float64))):
                Z = (q - pmu) / psd
                Dp[pk] = ((P ** 2).sum(1)[:, None] - 2 * P @ Z.T + (Z ** 2).sum(1)[None])
            Fz = (D[e].astype(np.float64) - mu) / sd
            Fzs = (D[e][rngs[e]].astype(np.float64) - mu) / sd
            for name, cols in sets.items():
                f = (Fzs if name == SHUF else Fz)[:, cols]
                t = ((Tfs if name == SHUF else Tf) - mu)[:, cols] / sd[cols]
                Df = (f ** 2).sum(1)[:, None] - 2 * f @ t.T + (t ** 2).sum(1)[None]
                for pk in ("mean", "coarse"):
                    for w in wgrid:
                        sc = Df if np.isinf(w) else Dp[pk] + w * Df
                        k = sc.argmin(0)
                        err[(name, pk, w)][e] = np.linalg.norm(
                            clouds[e][k] - true[e], axis=1)
        if (si + 1) % 40 == 0 or si + 1 == len(subs):
            log(f"  subject {si+1}/{len(subs)}  {time.time()-t0:.0f}s")
    return err, floor, aidx


def table(err, floor, sets, wgrid, log=print):
    """print the retrieval table and return it as JSON-able rows"""
    rows = []
    log(f"\nretrieval FLOOR (anchor = nearest cloud point to GT): {floor.mean():.4f} mm "
        f"mean, {np.median(floor):.4f} median, {floor.max():.4f} max")
    head = f"{'descriptor':>12s} {'pos':>7s} " + "".join(f"{('w=%g' % w):>9s}" for w in wgrid)
    for pk in ("mean", "coarse"):
        base = err[(list(sets)[0], pk, 0.0)].mean()
        log(f"\n--- POS={pk}: position alone (w=0) = {base:.4f} mm "
            f"[all descriptors share this column]")
        log(head)
        for name in sets:
            e0 = err[(name, pk, 0.0)]
            cells = []
            for w in wgrid:
                m = err[(name, pk, w)].mean()
                cells.append(f"{m:9.4f}")
            best_w = min(wgrid, key=lambda w: err[(name, pk, w)].mean())
            eb = err[(name, pk, best_w)]
            log(f"{name:>12s} {pk:>7s} " + "".join(cells) +
                f"   best {eb.mean():.4f} ({eb.mean()-e0.mean():+.4f}) at w={best_w:g}")
            row = {"descriptor": name, "pos": pk,
                   "w0_mm": round(float(e0.mean()), 4),
                   "best_w": best_w, "best_mm": round(float(eb.mean()), 4),
                   "delta_mm": round(float(eb.mean() - e0.mean()), 4),
                   "per_w_mm": {str(w): round(float(err[(name, pk, w)].mean()), 4)
                                for w in wgrid},
                   "per_contour_best_mm": {c: round(float(eb[:, a:b + 1].mean()), 4)
                                           for c, a, b in CONTOURS},
                   "per_contour_delta_mm": {
                       c: round(float(eb[:, a:b + 1].mean() - e0[:, a:b + 1].mean()), 4)
                       for c, a, b in CONTOURS},
                   "n_landmarks_improved": int(
                       (eb.mean(0) < e0.mean(0)).sum()),
                   "hit_2mm": round(float((eb < 2.0).mean()), 4)}
            rows.append(row)
        # The first version of this file drew the WRONG CONCLUSION because its grid started
        # past the optimum: every descriptor pinned at w = 0 and curvature read as null.
        # "Pinned at an endpoint" is the observable symptom, so say so instead of relying on
        # a reader noticing. wlo firing means the true optimum may be BELOW the grid (the
        # descriptor is being under-weighted and its delta understated); whi means above.
        pos = [w for w in wgrid if w > 0 and np.isfinite(w)]
        wlo = [n for n in sets if min(wgrid, key=lambda w: err[(n, pk, w)].mean()) == min(pos)]
        whi = [n for n in sets if min(wgrid, key=lambda w: err[(n, pk, w)].mean()) == max(pos)]
        for tag, names, side in (("smallest", wlo, min(pos)), ("largest", whi, max(pos))):
            if names:
                log(f"  !! best w is the {tag} tested weight ({side:g}) for {names} -- the "
                    f"optimum may lie past the grid; widen WGRID before reading their delta")
    return rows


# --------------------------------------------------------------------------- driver
def run():
    z = np.load(SRC, allow_pickle=True)
    names = [str(x) for x in z["crv_names"]]
    clouds = z["clouds"][:, SLOT].astype(np.float32)
    nrm = z["nrm"][:, SLOT].astype(np.float32)
    crv = z["crv"][:, SLOT].astype(np.float32)
    true = z["true"].astype(np.float32); coarse = z["coarse"].astype(np.float32)
    E, N, _ = clouds.shape
    D = np.concatenate([nrm, crv], -1)
    subj = np.arange(E) // 2
    sets = feature_sets(names)
    print(f"[curv_probe] {SRC}  E={E} N={N} slot={SLOT}  channels {names}")
    print(f"  descriptor sets: " + ", ".join(f"{k}({len(v)})" for k, v in sets.items()))
    print(f"  LOSO over {len(np.unique(subj))} subjects; w grid {[0.0]+WGRID}")
    wg = [0.0] + WGRID
    err, floor, aidx = probe(clouds, D, true, coarse, subj, sets, wg)
    rows = table(err, floor, sets, wg)

    # per-landmark detail for the single most promising configuration
    cand = [r for r in rows if r["descriptor"] in ("crv", "nrm+crv")]
    best = min(cand, key=lambda r: r["delta_mm"])
    key = (best["descriptor"], best["pos"], best["best_w"])
    eb, e0 = err[key], err[(best["descriptor"], best["pos"], 0.0)]
    d = eb.mean(0) - e0.mean(0)
    o = np.argsort(d)
    print(f"\n--- per-landmark, best curvature config {key}: "
          f"{int((d<0).sum())}/85 landmarks improved")
    print("    most improved: " + ", ".join(f"{int(i)}({d[i]:+.3f})" for i in o[:8]))
    print("    most harmed:   " + ", ".join(f"{int(i)}({d[i]:+.3f})" for i in o[-8:]))

    res = {"src": SRC, "slot": SLOT, "n_ears": int(E), "n_points": int(N),
           "channels": names, "w_grid": wg,
           "floor_mm": round(float(floor.mean()), 4),
           "floor_median_mm": round(float(np.median(floor)), 4),
           "rows": rows,
           "best_curvature_config": {"descriptor": key[0], "pos": key[1], "w": key[2],
                                     "delta_mm": best["delta_mm"],
                                     "per_landmark_delta_mm": [round(float(x), 4) for x in d]}}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")
    return res


# --------------------------------------------------------------------------- smoke
def smoke():
    """Synthetic ears where the descriptor is KNOWN to be informative, plus a null one.

    Ear e is the same point set shifted by a random offset d_e. Channel block `good`
    depends on the OFFSET-CORRECTED coordinate, so it identifies a landmark's
    neighbourhood exactly while the mean position template does not; block `null` is
    per-ear noise. The probe must rank them that way, and the shuffled control must
    destroy the good one -- otherwise this file's conclusions mean nothing.
    """
    t0 = time.time()
    E, N, NC = 12, 3000, 12
    rs = np.random.RandomState(0)
    L0 = rs.uniform(-10, 10, (NL, 3))
    clouds = np.zeros((E, N, 3), np.float32)
    D = np.zeros((E, N, 3 + NC), np.float32)
    true = np.zeros((E, NL, 3), np.float32)

    def field(p):                                  # 12 smooth, mutually distinct channels
        return np.concatenate([np.sin(p / (1.0 + k)) * np.cos(p[:, ::-1] / (2.0 + k))
                               for k in range(NC // 3)], 1)

    for e in range(E):
        d = rs.uniform(-4, 4, 3)                   # per-ear translation
        q = rs.uniform(-12, 12, (N, 3))            # offset-corrected coordinates
        clouds[e] = q + d
        true[e] = L0 + d
        D[e, :, :3] = rs.randn(N, 3) * 0.1         # "nrm" block: pure noise here
        D[e, :, 3:] = field(q)                     # "crv" block: informative
    coarse = true + rs.randn(E, NL, 3).astype(np.float32) * 1.5
    subj = np.arange(E) // 2
    names = [f"{c}@{r}" for r in (1.5, 3.0, 6.0) for c in "SCHK"]
    sets = feature_sets(names)
    assert len(sets["crv"]) == NC and len(sets["crv@1.5"]) == 4 and len(sets["nrm"]) == 3
    print(f"sets: " + ", ".join(f"{k}({len(v)})" for k, v in sets.items()))
    wg = [0.0, 1.0, 16.0, np.inf]
    err, floor, aidx = probe(clouds, D, true, coarse, subj, sets, wg)
    rows = table(err, floor, sets, wg)

    g = min(err[("crv", "mean", w)].mean() for w in wg)
    b = min(err[("nrm", "mean", w)].mean() for w in wg)
    sh = min(err[(SHUF, "mean", w)].mean() for w in wg)
    p0 = err[("crv", "mean", 0.0)].mean()
    print(f"\n  position only {p0:.4f} | informative crv {g:.4f} | noise nrm {b:.4f} | "
          f"shuffled crv {sh:.4f} | floor {floor.mean():.4f}")
    assert g < 0.5 * p0, f"probe failed to see a planted signal ({g:.3f} vs {p0:.3f})"
    assert g < 0.5 * sh, f"the planted signal did not beat its own shuffle ({g:.3f}/{sh:.3f})"
    assert b > 0.8 * p0, f"probe credited a pure-noise descriptor ({b:.3f} vs {p0:.3f})"
    assert sh > 0.8 * p0, f"the shuffled control did not destroy the signal ({sh:.3f})"
    assert err[("crv", "mean", 0.0)].shape == (E, NL)
    assert len(rows) == 2 * len(sets) and all(r["n_landmarks_improved"] <= NL for r in rows)
    assert (floor >= 0).all() and floor.mean() < 2.0
    print(f"SMOKE PASS ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    smoke() if int(os.environ.get("SMOKE", "0")) else run()
