"""
WHAT THE ARC-LENGTH-PROFILE CONSTRAINT CAN AND CANNOT BUY -- the exact map.

The finding this exists to bound: the normalised cumulative arc-length profile of the GT
landmarks along each contour is nearly CONSTANT across subjects, most sharply for
inner_helix and sup._antihelix. If phase is determined by the curve, then 30 of the 85
landmarks should not be carrying 1.49 and 1.21 mm of error. This file measures, per
contour, how much of that is actually recoverable and under what oracle.

Four questions, each a separate row, each explicit about what is ORACLE and what is
PREDICTED:

  1 POPULATION-PROFILE FLOOR   landmarks placed on the GT curve at the TRAINING-fold mean
    profile. Everything is oracle except the profile. This is the irreducible cost of
    assuming subjects share a profile -- no model can beat it while making that
    assumption. Reported against the uniform (equidistant) profile, which is the special
    case the two tight contours are claimed to satisfy.

  2 ANCHOR-CONDITIONED ORACLE  k GT landmarks are given (k = 2 endpoints, then 3..KMAX
    spread by index). Each anchor-bounded sub-polyline of the PREDICTED curve is mapped
    onto its two GT anchors by a similarity, and the interior landmarks are placed at the
    population profile renormalised within that segment. Curve geometry stays PREDICTED;
    ground truth supplies only the k anchors and (via the training fold) the profile.
    k=2 is exactly "give me the two endpoints". Reported twice: over all n landmarks, and
    over the n-k NON-ANCHOR landmarks only, because the k anchors are oracle-exact and
    would otherwise flatter the number.

  3 CURVE vs PHASE   the free per-point distance from each GT landmark to the predicted
    polyline is the part of the error NO reparameterisation can remove ("the curve is
    wrong"); everything above it is phase. Measured on the raw predicted polyline and on
    the endpoint-aligned one, so the two are separated per contour.

  4 COMBINE   the overall MLE if inner_helix and sup._antihelix were solved to their k=2
    oracle and the other two contours left untouched, plus the zero-oracle variant
    (reparameterise the predicted curve in place, no GT anchors at all) which is the only
    row here that is directly deployable.

NOT A MODEL. No training, no fitting beyond a mean over training ears.

LEAKAGE. The population profile is the only fitted quantity. population_profile() takes
FOLD explicitly, derives the training ears from the frozen subject-grouped split itself,
and asserts both ear-disjointness and SUBJECT-disjointness from that fold's validation
ears before it averages anything. Every ear is scored with the profile of the fold that
held it out.

A PREVIOUS FAILURE THIS DOES NOT REPEAT: research/code/fam_phase.py imposed SMOOTHNESS (a
rank-16 Catmull-Rom) and scored 1.81 mm; research/results/curve_floor.json then measured
that a curve at ~60% of the landmarks' dof costs up to 0.98 mm before any learning.
Nothing here reduces the rank of anything -- the predicted polyline is used at full rank
and only its PARAMETERISATION is touched.

    python research/code/profile_oracle.py            # smoke test, then the analysis
    SRC=ens python research/code/profile_oracle.py    # against the unprojected ensemble

Env (defaults): WORK=scratch  SRC=proj (proj|ens)  KMAX=6  SCALE=1 (similarity, not
rigid, in the anchor alignment)  CACHE=1  SMOKE_ONLY=0
OUT=research/results/profile_oracle.json
Writes OUT, and caches the assembled prediction at $WORK/ensemble3_{ens,proj}.npy.
"""
import os, sys, json
import numpy as np

WORK = os.environ.get("WORK", "scratch")
SRC = os.environ.get("SRC", "proj")
KMAX = int(os.environ.get("KMAX", "6"))
SIGMAS = [float(s) for s in
          os.environ.get("SIGMAS", "0,0.25,0.5,0.75,1.0,1.5,2.0").split(",")]
NREP = int(os.environ.get("NREP", "3"))
SCALE = os.environ.get("SCALE", "1") == "1"
CACHE = os.environ.get("CACHE", "1") == "1"
OUT = os.environ.get("OUT", "research/results/profile_oracle.json")
NFOLD = 5
CONT = [(0, 24, "outer_helix"), (25, 54, "concha"),
        (55, 74, "inner_helix"), (75, 84, "sup._antihelix")]
TIGHT = ("inner_helix", "sup._antihelix")      # the two contours the finding is about


# ------------------------------------------------------------------ curve primitives
def arc(P):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]


def profile(P):
    """normalised cumulative arc-length position of each vertex of the polyline P."""
    s = arc(P)
    return s / max(s[-1], 1e-12)


def resample(P, u):
    """piecewise-linear evaluation of polyline P at normalised arc lengths u, clipped to
    [0,1]. No extrapolation: every placement here is interior to the curve by
    construction, since u[0]=0 and u[-1]=1 for any profile."""
    s = arc(P)
    t = np.clip(np.asarray(u, float), 0.0, 1.0) * s[-1]
    j = np.clip(np.searchsorted(s, t) - 1, 0, len(P) - 2)
    f = (t - s[j]) / np.maximum(s[j + 1] - s[j], 1e-12)
    return P[j] + f[:, None] * (P[j + 1] - P[j])


def dist_to_poly(Q, P):
    """min distance from each of Q (m,3) to the polyline P (n,3). FREE per-point
    reparameterisation, not even monotone -- a lower bound on any phase correction."""
    a, ab = P[:-1], np.diff(P, axis=0)
    t = np.clip(((Q[:, None] - a) * ab).sum(-1) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
    return np.linalg.norm(a + t[..., None] * ab - Q[:, None], axis=-1).min(1)


def min_rot(a, b):
    """minimal rotation carrying direction a onto direction b."""
    u, v = a / max(np.linalg.norm(a), 1e-12), b / max(np.linalg.norm(b), 1e-12)
    c = float(u @ v)
    if c > 1 - 1e-12:
        return np.eye(3)
    if c < -1 + 1e-12:                       # antiparallel: any axis perpendicular to u
        w = np.eye(3)[int(np.argmin(np.abs(u)))]
        w = np.cross(u, w); w /= np.linalg.norm(w)
        return -np.eye(3) + 2 * np.outer(w, w)
    k = np.cross(u, v)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def sim_fit(Xs, Xt, scale=True):
    """-> callable mapping source points onto target points by a similarity.

    k=2 is special-cased deliberately. With two correspondences the centred covariance has
    rank 1, so Umeyama's rotation is UNDETERMINED about the chord axis and LAPACK's
    arbitrary choice of the null singular vectors would decide it. The minimal rotation of
    the source chord onto the target chord is the well-defined choice, and with the scale
    free it maps BOTH anchors exactly onto their GT positions."""
    ms, mt = Xs.mean(0), Xt.mean(0)
    A, B = Xs - ms, Xt - mt
    if len(Xs) == 2:
        R = min_rot(A[1] - A[0], B[1] - B[0])
    else:
        U, _, Vt = np.linalg.svd(B.T @ A)
        R = U @ np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)))]) @ Vt
    s = 1.0
    if scale:
        den = float((A ** 2).sum())
        s = float((B * (A @ R.T)).sum()) / den if den > 1e-12 else 1.0
    return lambda P: (P - ms) @ R.T * s + mt


def anchor_idx(n, k):
    """k anchor positions spread by INDEX along a contour of n landmarks. Index, not arc
    length: a model predicts specific landmarks, not arc-length stations."""
    a = np.unique(np.round(np.linspace(0, n - 1, k)).astype(int))
    assert len(a) == k and a[0] == 0 and a[-1] == n - 1, (n, k, a)
    return a


def place(P, u, A_gt=None, aidx=None, scale=True):
    """Place n landmarks on the PREDICTED polyline P at population profile u.

    A_gt/aidx None  -> no oracle at all: reparameterise P in place (deployable variant).
    otherwise       -> for k>=3 a GLOBAL similarity onto all k anchors first, because two
                       correspondences leave the ROLL about their chord free and only a
                       third anchor pins it; then each anchor-bounded sub-polyline is
                       mapped onto its two GT anchors so every anchor lands exactly on GT,
                       and its interior is placed at u renormalised within the segment.
                       At k=2 the global step is a no-op and this is exactly "give me the
                       two endpoints".
    """
    if A_gt is None:
        return resample(P, u)
    if len(aidx) >= 3:
        P = sim_fit(P[aidx], A_gt, scale)(P)
    out = np.empty_like(P)
    for j in range(len(aidx) - 1):
        i0, i1 = int(aidx[j]), int(aidx[j + 1])
        S = P[i0:i1 + 1]
        T = sim_fit(np.stack([S[0], S[-1]]), np.stack([A_gt[j], A_gt[j + 1]]), scale)
        w = (u[i0:i1 + 1] - u[i0]) / max(u[i1] - u[i0], 1e-12)
        out[i0:i1 + 1] = resample(T(S), w)
    return out


# ------------------------------------------------------------------ frozen folds
def frozen_folds(ne):
    """Constraint 3, verbatim (identical to train_family.frozen_folds)."""
    subj = np.arange(ne) // 2
    parts = np.array_split(np.random.RandomState(12345).permutation(np.unique(subj)), NFOLD)
    return subj, [np.asarray(p) for p in parts]


def split(fold, subj, parts):
    va = np.where(np.isin(subj, parts[fold]))[0]
    tr = np.where(~np.isin(subj, parts[fold]))[0]
    assert not np.intersect1d(tr, va).size, f"fold {fold}: ear in both halves"
    assert not (set(subj[tr].tolist()) & set(subj[va].tolist())), \
        f"fold {fold}: a SUBJECT straddles the split -- grouping broken"
    assert len(tr) + len(va) == len(subj)
    return tr, va


def population_profile(GT, lo, hi, fold, subj, parts):
    """Mean normalised arc-length profile of one contour, from fold's TRAINING ears ONLY.

    Takes FOLD explicitly and re-derives the split here rather than trusting a caller's
    index array (constraint 2): this is the only fitted quantity in the file, so it is the
    whole leakage surface.
    """
    tr, va = split(fold, subj, parts)
    assert not np.isin(tr, va).any(), "LEAK: a validation ear is in the profile sample"
    U = np.stack([profile(GT[e, lo:hi + 1]) for e in tr])
    m = U.mean(0)
    assert m[0] == 0.0 and abs(m[-1] - 1.0) < 1e-12 and (np.diff(m) > 0).all()
    return m, U, tr, va


# ------------------------------------------------------------------ the analysis
def anchor_noise(P, GT, subj, parts, mode="iso", sigmas=SIGMAS, nrep=NREP, scale=SCALE,
                 conts=CONT):
    """Row 5, the number that decides whether row 2 is reachable. The k=2 oracle hands out
    EXACT endpoints; a model would predict them. Perturb the two anchors by noise of sd
    sigma and re-run, so the gain is read against how well the endpoints would actually
    have to be localised. sigma=0 reproduces the oracle exactly.

    mode='iso'      isotropic 3D noise.
    mode='tangent'  noise ALONG the contour only, which is what these models actually make
                    (tangent RMSE 1.40 of 1.58 total). Isotropic noise alone would be the
                    wrong proxy and would set the accuracy bar in the wrong place.

    The perturbed anchors are counted in the error, as they must be: they ARE two of the
    landmarks being scored.

    UNITS. sigma is the sd of a Gaussian; current_anchor_error_mm is a mean DISTANCE. They
    are not comparable and must never be printed side by side without the conversion:
    E|N3(0,s)| = 2*sqrt(2/pi)*s and E|N1(0,s)| = sqrt(2/pi)*s. breakeven_mean_disp_mm is
    the breakeven in the unit of the measured columns, and the tangent mode must be read
    against the TANGENTIAL component of today's endpoint error, not against its norm.
    """
    E = len(GT)
    c_disp = 2 * np.sqrt(2 / np.pi) if mode == "iso" else np.sqrt(2 / np.pi)
    fold_of = np.full(E, -1)
    for f in range(NFOLD):
        fold_of[split(f, subj, parts)[1]] = f
    out = {}
    for lo, hi, nm in conts:
        n = hi - lo + 1
        a = anchor_idx(n, 2)
        u = {f: population_profile(GT, lo, hi, f, subj, parts)[0] for f in range(NFOLD)}
        # today's endpoint error, split along/across the contour at each endpoint: the
        # tangent bar is only meaningful against the tangential column.
        er = P[:, lo:hi + 1][:, a] - GT[:, lo:hi + 1][:, a]
        Tg = np.stack([GT[:, lo + 1] - GT[:, lo], GT[:, hi] - GT[:, hi - 1]], 1)
        Tg /= np.linalg.norm(Tg, axis=2, keepdims=True)
        sg = (er * Tg).sum(-1)
        cur = float(np.linalg.norm(er, axis=2).mean())
        cur_t = float(np.abs(sg).mean())
        cur_p = float(np.linalg.norm(er - sg[..., None] * Tg, axis=2).mean())
        curve = {}
        for sg in sigmas:
            rng = np.random.RandomState(4242)
            v = []
            for r in range(nrep if sg > 0 else 1):
                for e in range(E):
                    G, Q = GT[e, lo:hi + 1], P[e, lo:hi + 1]
                    if mode == "iso":
                        d = rng.randn(2, 3)
                    else:                       # along the contour at each endpoint
                        T = np.stack([G[1] - G[0], G[-1] - G[-2]])
                        d = T / np.linalg.norm(T, axis=1, keepdims=True) * rng.randn(2, 1)
                    X = place(Q, u[fold_of[e]], G[a] + d * sg, a, scale)
                    v.append(np.linalg.norm(X - G, axis=1).mean())
            curve[sg] = round(float(np.mean(v)), 4)
        base = float(np.linalg.norm(P[:, lo:hi + 1] - GT[:, lo:hi + 1], axis=2).mean())
        s = np.array(sigmas)
        g = base - np.array([curve[x] for x in sigmas])          # gain, decreasing in sigma
        be = None
        for i in range(len(s) - 1):
            if g[i] > 0 >= g[i + 1]:
                be = round(float(s[i] + (s[i + 1] - s[i]) * g[i] / (g[i] - g[i + 1])), 3)
                break
        out[nm] = {"mode": mode, "current_anchor_error_mm": round(cur, 4),
                   "current_anchor_tangential_mm": round(cur_t, 4),
                   "current_anchor_perp_mm": round(cur_p, 4),
                   "baseline_mm": round(base, 4), "by_sigma_mm": curve,
                   "breakeven_sigma_mm": be,
                   "breakeven_mean_disp_mm": None if be is None else round(c_disp * be, 4),
                   "compare_against": ("current_anchor_error_mm" if mode == "iso"
                                       else "current_anchor_tangential_mm"),
                   "nrep": nrep if max(sigmas) > 0 else 1}
    return out


def analyse(P, GT, subj, parts, kmax=KMAX, scale=SCALE, conts=CONT):
    """P = PREDICTED landmarks (E,85,3), GT = ground truth (E,85,3). Everything below is
    per contour; every ear is scored with the profile of the fold that HELD IT OUT."""
    E = len(GT)
    assert P.shape == GT.shape == (E, 85, 3), (P.shape, GT.shape)
    fold_of, seen = np.full(E, -1), np.zeros(E, int)
    for f in range(NFOLD):
        va = split(f, subj, parts)[1]
        fold_of[va] = f
        seen[va] += 1
    # not cosmetic: with overlapping parts an ear silently gets the LAST fold that claimed
    # it, and is then scored with a profile the other fold trained on -- a leak that every
    # shape check passes.
    assert (seen == 1).all(), \
        f"{int((seen != 1).sum())} ears are not in exactly one fold's validation set"
    prof = {(nm, f): population_profile(GT, lo, hi, f, subj, parts)
            for lo, hi, nm in conts for f in range(NFOLD)}

    res, PRED = {}, {}
    for lo, hi, nm in conts:
        n = hi - lo + 1
        ks = [k for k in range(2, kmax + 1) if k <= n]
        acc = {"base": [], "len": [], "gt_uniform": [], "gt_pop": [],
               "floor_raw": [], "floor_aligned": [], "noanchor": [],
               **{f"k{k}": [] for k in ks}, **{f"k{k}_na": [] for k in ks},
               **{f"g{k}": [] for k in ks}}
        store = {f"k{k}": np.empty((E, n, 3)) for k in ks}
        store["noanchor"] = np.empty((E, n, 3))
        uni = np.linspace(0, 1, n)
        for e in range(E):
            u = prof[(nm, fold_of[e])][0]
            G, Q = GT[e, lo:hi + 1], P[e, lo:hi + 1]
            acc["base"].append(np.linalg.norm(Q - G, axis=1).mean())
            acc["len"].append(arc(G)[-1])
            # 1 population-profile floor: GT curve, population profile
            acc["gt_uniform"].append(np.linalg.norm(resample(G, uni) - G, axis=1).mean())
            acc["gt_pop"].append(np.linalg.norm(resample(G, u) - G, axis=1).mean())
            # 3 curve-vs-phase floors on the PREDICTED curve
            acc["floor_raw"].append(dist_to_poly(G, Q).mean())
            T2 = sim_fit(np.stack([Q[0], Q[-1]]), np.stack([G[0], G[-1]]), scale)
            acc["floor_aligned"].append(dist_to_poly(G, T2(Q)).mean())
            # zero-oracle: reparameterise the predicted curve in place
            X = place(Q, u)
            store["noanchor"][e] = X
            acc["noanchor"].append(np.linalg.norm(X - G, axis=1).mean())
            # 2 anchor-conditioned oracle
            for k in ks:
                a = anchor_idx(n, k)
                X = place(Q, u, G[a], a, scale)
                store[f"k{k}"][e] = X
                d = np.linalg.norm(X - G, axis=1)
                acc[f"k{k}"].append(d.mean())
                acc[f"k{k}_na"].append(np.delete(d, a).mean())
                # the SAME procedure on the GT curve: the floor for that many anchors.
                # Row 1's floor is only the k=2 case -- extra anchors reset the accumulated
                # phase drift, so the profile assumption gets cheaper as k grows and the
                # k=2 floor is NOT a bound on k>2.
                acc[f"g{k}"].append(np.linalg.norm(place(G, u, G[a], a, scale) - G,
                                                   axis=1).mean())
        m = {k: float(np.mean(v)) for k, v in acc.items()}
        # sd of the profile itself, against the TRAINING mean of each ear's own fold
        sdu = float(np.mean([np.std(prof[(nm, f)][1], 0, ddof=1).mean() for f in range(NFOLD)]))
        b, fr, fa, p2 = m["base"], m["floor_raw"], m["floor_aligned"], m["k2"]
        res[nm] = {"n": n, "ks": ks, "mean_len_mm": round(m["len"], 3),
                   "profile_sd": round(sdu, 4),
                   "profile_sd_mm": round(sdu * m["len"], 4),
                   "baseline_mm": round(b, 4),
                   "gt_curve_uniform_mm": round(m["gt_uniform"], 4),
                   "gt_curve_popprofile_mm": round(m["gt_pop"], 4),
                   "pred_curve_freepoint_floor_mm": round(fr, 4),
                   "pred_curve_freepoint_floor_gtaligned_mm": round(fa, 4),
                   "zero_oracle_reparam_mm": round(m["noanchor"], 4),
                   "zero_oracle_delta_mm": round(m["noanchor"] - b, 4),
                   "anchor_oracle_mm": {k: round(m[f"k{k}"], 4) for k in ks},
                   "anchor_oracle_nonanchor_mm": {k: round(m[f"k{k}_na"], 4) for k in ks},
                   "gt_curve_anchor_floor_mm": {k: round(m[f"g{k}"], 4) for k in ks},
                   # in-place phase share: what ANY reparameterisation of the predicted
                   # curve, left where it is, could at most recover
                   "inplace_phase_share_pct": round(100 * (b - fr) / b, 1),
                   # negative = the endpoint alignment DESTROYS the curve's geometry
                   "gt_endpoint_alignment_gain_mm": round(fr - fa, 4),
                   "k2_gain_mm": round(b - p2, 4),
                   "k2_pct_of_its_own_ceiling": (round(100 * (b - p2) / (b - fa), 1)
                                                 if b - fa > 1e-6 else None)}
        PRED[nm] = store
    return res, PRED, fold_of


def combine(P, GT, PRED, res, conts=CONT):
    """Row 4. Replace ONLY the two tight contours with their k=2 oracle; everything else
    stays exactly as predicted. Reported three ways so the oracle's own contribution is
    visible rather than buried."""
    lm = lambda X: float(np.linalg.norm(X - GT, axis=2).mean())
    out = {"baseline_all85_mm": round(lm(P), 4), "replaced": list(TIGHT)}
    for tag, use_anchor_credit, key in (("k2_oracle", True, "k2"),
                                        ("k2_oracle_no_anchor_credit", False, "k2"),
                                        ("zero_oracle_reparam", True, "noanchor")):
        X = P.copy()
        for lo, hi, nm in conts:
            if nm not in TIGHT:
                continue
            Y = PRED[nm][key].copy()
            if not use_anchor_credit:                 # anchors keep their PREDICTED value
                a = anchor_idx(hi - lo + 1, 2)
                Y[:, a] = P[:, lo:hi + 1][:, a]
            X[:, lo:hi + 1] = Y
        out[tag] = {"all85_mm": round(lm(X), 4), "delta_mm": round(lm(X) - lm(P), 4)}
    X = P.copy()                                       # zero-oracle applied to ALL four
    for lo, hi, nm in conts:
        X[:, lo:hi + 1] = PRED[nm]["noanchor"]
    out["zero_oracle_reparam_all_contours"] = {"all85_mm": round(lm(X), 4),
                                               "delta_mm": round(lm(X) - lm(P), 4)}
    return out


# ------------------------------------------------------------------ reporting
def report(res, comb, src, noise=None, conts=CONT):
    W = sum(r["n"] for r in res.values())
    agg = lambda k: sum(res[nm][k] * res[nm]["n"] for _, _, nm in conts) / W
    print(f"\n{'='*96}\nprediction source: {src}   |   {W} landmarks, 4 contours, "
          f"5 frozen folds, TRAINING-fold profiles only\n{'='*96}")

    print("\n1  POPULATION-PROFILE FLOOR -- oracle GT CURVE, only the profile is fitted")
    print(f"{'contour':16s}{'n':>4s}{'len_mm':>9s}{'sd(u)':>8s}{'sd_mm':>8s}"
          f"{'baseline':>10s}{'@uniform':>10s}{'@popprof':>10s}")
    for _, _, nm in conts:
        r = res[nm]
        print(f"{nm:16s}{r['n']:4d}{r['mean_len_mm']:9.2f}{r['profile_sd']:8.4f}"
              f"{r['profile_sd_mm']:8.3f}{r['baseline_mm']:10.4f}"
              f"{r['gt_curve_uniform_mm']:10.4f}{r['gt_curve_popprofile_mm']:10.4f}")
    print(f"{'ALL 85':16s}{W:4d}{'':9s}{'':8s}{'':8s}{agg('baseline_mm'):10.4f}"
          f"{agg('gt_curve_uniform_mm'):10.4f}{agg('gt_curve_popprofile_mm'):10.4f}")

    ks = sorted({k for r in res.values() for k in r["ks"]})
    cell = lambda d, k: f"{d[k]:9.4f}" if k in d else f"{'-':>9s}"
    for lbl, key in (
            ("PREDICTED curve, all n landmarks (the k anchors are oracle-exact)",
             "anchor_oracle_mm"),
            ("PREDICTED curve, the n-k NON-ANCHOR landmarks only", "anchor_oracle_nonanchor_mm"),
            ("floor: the SAME procedure on the ORACLE GT curve -- profile cost alone",
             "gt_curve_anchor_floor_mm")):
        print(f"\n2  ANCHOR-CONDITIONED ORACLE -- k GT anchors + population profile\n   {lbl}")
        print(f"{'contour':16s}{'base':>9s}" + "".join(f"{'k='+str(k):>9s}" for k in ks))
        for _, _, nm in conts:
            r = res[nm]
            print(f"{nm:16s}{r['baseline_mm']:9.4f}"
                  + "".join(cell(r[key], k) for k in ks))

    print("\n3  CURVE vs PHASE, per contour")
    print(f"{'contour':16s}{'base':>9s}{'curve_floor':>12s}{'phase':>9s}{'share':>8s}"
          f"{'algn_gain':>11s}{'k2':>9s}{'k2_gain':>9s}{'of_ceil':>9s}")
    for _, _, nm in conts:
        r = res[nm]
        b, fr = r["baseline_mm"], r["pred_curve_freepoint_floor_mm"]
        c = r["k2_pct_of_its_own_ceiling"]
        print(f"{nm:16s}{b:9.4f}{fr:12.4f}{b-fr:9.4f}{r['inplace_phase_share_pct']:7.1f}%"
              f"{r['gt_endpoint_alignment_gain_mm']:+11.4f}{r['anchor_oracle_mm'][2]:9.4f}"
              f"{r['k2_gain_mm']:+9.4f}" + (f"{c:8.1f}%" if c is not None else f"{'n/a':>9s}"))
    print("   curve_floor = free per-point distance from GT to the PREDICTED polyline, left "
          "where it is:\n     the part of the error no reparameterisation can touch. "
          "phase = base - curve_floor.\n   algn_gain = how much moving the curve onto the "
          "two GT ENDPOINTS improves that floor;\n     negative means the rigid "
          "repositioning DESTROYS the curve geometry.\n   of_ceil = the k=2 gain as a "
          "share of (base - endpoint-aligned floor), i.e. of what that\n     "
          "repositioned curve could ever give.")

    print("\n   zero-oracle reparameterisation (NO GT anywhere -- the only deployable row)")
    print(f"{'contour':16s}{'base':>9s}{'reparam':>10s}{'delta':>10s}")
    for _, _, nm in conts:
        r = res[nm]
        print(f"{nm:16s}{r['baseline_mm']:9.4f}{r['zero_oracle_reparam_mm']:10.4f}"
              f"{r['zero_oracle_delta_mm']:+10.4f}")

    print(f"\n4  COMBINE -- {' + '.join(TIGHT)} solved, the other two untouched")
    print(f"   baseline all-85                       {comb['baseline_all85_mm']:.4f} mm")
    for k in ("k2_oracle", "k2_oracle_no_anchor_credit", "zero_oracle_reparam",
              "zero_oracle_reparam_all_contours"):
        print(f"   {k:36s}  {comb[k]['all85_mm']:.4f} mm  ({comb[k]['delta_mm']:+.4f})")

    for mode, nz in (noise or {}).items():
        sg = sorted(next(iter(nz.values()))["by_sigma_mm"])
        print(f"\n5  HOW GOOD DO THE TWO ANCHORS HAVE TO BE -- k=2 with the endpoints "
              f"perturbed by\n   {mode.upper()} noise of sd sigma (the perturbed anchors "
              f"are scored, as a model's would be)")
        print(f"{'contour':16s}{'base':>9s}{'now':>8s}{'now_t':>8s}"
              + "".join(f"{'s='+str(s):>9s}" for s in sg)
              + f"{'be_sigma':>10s}{'be_mm':>8s}{'vs now':>9s}")
        for _, _, nm in conts:
            r = nz[nm]
            be, bm = r["breakeven_sigma_mm"], r["breakeven_mean_disp_mm"]
            now = r[r["compare_against"]]
            print(f"{nm:16s}{r['baseline_mm']:9.4f}{r['current_anchor_error_mm']:8.3f}"
                  f"{r['current_anchor_tangential_mm']:8.3f}"
                  + "".join(f"{r['by_sigma_mm'][s]:9.4f}" for s in sg)
                  + (f"{be:10.3f}{bm:8.3f}{bm/now:8.2f}x" if be is not None
                     else f"{'>max':>10s}{'-':>8s}{'-':>9s}"))
        print("   now / now_t = the CURRENT mean endpoint error of those two landmarks, in "
              "full 3D and\n     along the contour. be_sigma is a Gaussian SD and is NOT "
              "comparable to them; be_mm is\n     the same breakeven as a mean displacement "
              f"(E|N{'3' if mode == 'iso' else '1'}(0,s)| = "
              f"{2*np.sqrt(2/np.pi) if mode == 'iso' else np.sqrt(2/np.pi):.3f}s), which is."
              f"\n     vs now = be_mm / {'now' if mode == 'iso' else 'now_t'}; below 1.00x "
              "the endpoints are not good enough today.")


def conclude(res, comb, noise):
    f = lambda nm, k: res[nm][k]
    be = lambda m, nm: noise[m][nm]["breakeven_mean_disp_mm"]
    sig = lambda m, nm: (f"{be(m, nm):.2f}mm" if be(m, nm) is not None
                         else f">{max(SIGMAS):.1f}mm")
    now = lambda m, nm: noise[m][nm][noise[m][nm]["compare_against"]]
    # first k at which the procedure beats the contour's own baseline, on the honest
    # (non-anchor) row: the credited row flatters it by k oracle-exact landmarks
    firstk = lambda nm: next((k for k in res[nm]["ks"]
                              if res[nm]["anchor_oracle_nonanchor_mm"][k] < res[nm]["baseline_mm"]),
                             None)
    return (
        f"The shared-profile assumption is FREE for inner_helix ({f('inner_helix','gt_curve_popprofile_mm')}mm "
        f"on the GT curve) and sup._antihelix ({f('sup._antihelix','gt_curve_popprofile_mm')}mm), and "
        f"UNUSABLE for outer_helix ({f('outer_helix','gt_curve_popprofile_mm')}mm) and concha "
        f"({f('concha','gt_curve_popprofile_mm')}mm) -- both of those floors sit at or above the error "
        f"those contours already have ({f('outer_helix','baseline_mm')} / {f('concha','baseline_mm')}mm), so no "
        f"model can use a shared whole-contour profile there. Two GT endpoints take inner_helix "
        f"{f('inner_helix','baseline_mm')}->{f('inner_helix','anchor_oracle_mm')[2]} and sup._antihelix "
        f"{f('sup._antihelix','baseline_mm')}->{f('sup._antihelix','anchor_oracle_mm')[2]}, worth "
        f"{comb['k2_oracle']['delta_mm']}mm on all 85 ({comb['k2_oracle']['all85_mm']}mm), or "
        f"{comb['k2_oracle_no_anchor_credit']['delta_mm']}mm once the two oracle-exact anchors per contour are "
        f"NOT credited. outer_helix and concha need k>={firstk('outer_helix')}/"
        f"{firstk('concha')} anchors before the procedure beats their baseline on the "
        f"non-anchor landmarks (k=3 already beats outer_helix's baseline if the 3 "
        f"oracle-exact anchors are credited), and even at k=6 their floor on the GT CURVE is "
        f"{f('outer_helix','gt_curve_anchor_floor_mm')[6]}/{f('concha','gt_curve_anchor_floor_mm')[6]}mm while the "
        f"predicted curve gives {f('outer_helix','anchor_oracle_mm')[6]}/{f('concha','anchor_oracle_mm')[6]}mm -- "
        f"so what limits them is curve GEOMETRY, not phase. With no oracle at all, "
        f"reparameterising in place HURTS every contour except sup._antihelix "
        f"({f('sup._antihelix','zero_oracle_delta_mm')}mm); there is no free lunch here. THE BAR (all figures "
        f"MEAN DISPLACEMENTS, not Gaussian sds -- the raw breakeven sigmas are not comparable "
        f"to a measured mean error): under PURELY TANGENTIAL endpoint error the k=2 gain dies "
        f"at {sig('tangent','inner_helix')} for inner_helix, whose endpoints carry "
        f"{now('tangent','inner_helix')}mm of tangential error today ("
        f"{now('tangent','inner_helix')/be('tangent','inner_helix'):.1f}x too much), and at "
        f"{sig('tangent','sup._antihelix')} for sup._antihelix, which carries "
        f"{now('tangent','sup._antihelix')}mm and is already INSIDE that bar. Under ISOTROPIC "
        f"endpoint noise the bars are {sig('iso','inner_helix')} and {sig('iso','sup._antihelix')} against "
        f"today's full 3D endpoint error of {now('iso','inner_helix')} and {now('iso','sup._antihelix')}mm, "
        f"i.e. inner_helix is 2x outside and sup._antihelix sits ON its bar. Real endpoint "
        f"error is neither: inner_helix's is "
        f"{100*noise['tangent']['inner_helix']['current_anchor_tangential_mm']/noise['iso']['inner_helix']['current_anchor_error_mm']:.0f}% "
        f"tangential so its tangent bar is the honest one, sup._antihelix's is only "
        f"{100*noise['tangent']['sup._antihelix']['current_anchor_tangential_mm']/noise['iso']['sup._antihelix']['current_anchor_error_mm']:.0f}% "
        f"tangential so its true bar sits between the two and it is at BREAKEVEN, not clear "
        f"of it. So inner_helix needs its endpoints roughly halved along the contour and "
        f"sup._antihelix needs its across-contour endpoint error cut before either pays. The "
        f"compute goes on ENDPOINT LOCALISATION for those two contours, not on a global "
        f"profile prior and not on outer_helix or concha.")


# ------------------------------------------------------------------ data
def load_pred(work, src, gt):
    """The 3-model equal-weight OOF ensemble of ensemble_final.py, optionally + exact
    surface projection. Cached, because the projection is 340 KD-trees."""
    what = ("equal-weight OOF ensemble of dgcnn3+kpconv+ptv3" +
            ("" if src == "ens" else " + exact surface projection"))
    cp = f"{work}/ensemble3_{src}.npy"
    if CACHE and os.path.exists(cp):
        # STALE-CACHE RISK: this does not track the mtimes of the member .npy files.
        # Delete cp (or set CACHE=0) after any member is retrained.
        return np.load(cp), f"{what} [cached {cp}]"
    ne = len(gt)

    def assemble(get):
        Q = np.full((ne, 85, 3), np.nan)
        for f in range(NFOLD):
            p, i = get(f)
            Q[i] = p
        assert not np.isnan(Q).any(), "an ear was never held out"
        return Q

    def dgcnn(f):
        i = np.array(json.load(open(f"{work}/screen_normalsfix_s0_f{f}.json"))["val_ear_index"])
        return np.mean([np.load(f"{work}/screen_normalsfix_s{s}_f{f}.npy")
                        for s in (0, 1, 2)], 0).astype(float), i

    def famA(t):
        def g(f):
            i = np.array(json.load(open(f"{work}/famA_{t}_f{f}.json"))["val_ear_index"])
            return np.load(f"{work}/famA_{t}_f{f}.npy").astype(float), i
        return g

    ENS = np.mean([assemble(dgcnn), assemble(famA("kpconv")), assemble(famA("ptv3"))], 0)
    if src == "ens":
        np.save(cp, ENS)
        return ENS, what
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "deep_model"))
    from scipy.spatial import cKDTree
    from surfproj import SurfaceProjector
    md = np.load(f"{work}/mesh_data.npz")
    V, F, VP, FP = md["verts"], md["faces"], md["v_ptr"], md["f_ptr"]
    R, C0 = md["R"].astype(float), md["c0"].astype(float)
    PR, worst = ENS.copy(), 0.0
    for i in range(ne):
        v = V[VP[i]:VP[i + 1]].astype(float) @ R[i] + C0[i]
        worst = max(worst, float(np.median(cKDTree(v).query(gt[i])[0])))
        PR[i] = SurfaceProjector(v, F[FP[i]:FP[i + 1]].astype(np.int64) - VP[i]).project(ENS[i])[0]
    assert worst < 2.0, f"frame mismatch: worst GT-to-vertex median {worst:.2f}mm"
    np.save(cp, PR)
    return PR, f"{what} (frame check {worst:.3f}mm)"


def main():
    of = np.load(f"{WORK}/ortho_feats.npz")
    GT = of["gt"].astype(float)
    subj, parts = frozen_folds(len(GT))
    assert (subj == of["subj"]).all(), "frozen subject rule disagrees with ortho_feats.subj"
    P, note = load_pred(WORK, SRC, GT)
    res, PRED, fold_of = analyse(P, GT, subj, parts)
    comb = combine(P, GT, PRED, res)
    noise = {m: anchor_noise(P, GT, subj, parts, m) for m in ("iso", "tangent")}
    report(res, comb, note, noise)
    W = sum(r["n"] for r in res.values())
    out = {"prediction_source": {"key": SRC, "note": note,
                                 "all85_mm": comb["baseline_all85_mm"]},
           "n_ears": int(len(GT)), "n_folds": NFOLD, "kmax": KMAX,
           "anchor_alignment": "similarity (rotation+uniform scale)" if SCALE else "rigid",
           "leakage": ("every population profile is the mean over the TRAINING ears of the "
                       "fold that held the scored ear out; population_profile() takes FOLD, "
                       "re-derives the split from the frozen rule and asserts ear- and "
                       "SUBJECT-disjointness before averaging"),
           "definitions": {
               "gt_curve_uniform_mm": "ORACLE curve (GT polyline), landmarks re-placed at "
                                      "equal arc length. Nothing predicted.",
               "gt_curve_popprofile_mm": "ORACLE curve (GT polyline), landmarks re-placed "
                                         "at the TRAINING-fold mean profile. The floor of "
                                         "assuming a shared profile.",
               "anchor_oracle_mm": "PREDICTED curve. k GT landmarks given; each "
                                   "anchor-bounded sub-polyline mapped onto its two GT "
                                   "anchors by a similarity, interior placed at the "
                                   "population profile. Includes the k oracle-exact "
                                   "anchors in the mean.",
               "anchor_oracle_nonanchor_mm": "same, averaged over the n-k landmarks the "
                                             "oracle did NOT supply.",
               "pred_curve_freepoint_floor_mm": "distance from each GT landmark to the "
                                                "PREDICTED polyline, free and non-monotone. "
                                                "The part no reparameterisation can remove.",
               "gt_curve_anchor_floor_mm": "the k-anchor procedure run on the ORACLE GT "
                                           "curve: the profile cost alone at that k. Row "
                                           "1's floor is only its k=2 case -- more anchors "
                                           "reset the accumulated phase drift.",
               "zero_oracle_reparam_mm": "PREDICTED curve reparameterised to the population "
                                         "profile IN PLACE. No ground truth anywhere. The "
                                         "only deployable row.",
               "inplace_phase_share_pct": "100*(baseline - pred_curve_freepoint_floor)/"
                                          "baseline: the share of the error any in-place "
                                          "reparameterisation could at most remove.",
               "gt_endpoint_alignment_gain_mm": "freepoint_floor minus its GT-endpoint-"
                                                "aligned version. NEGATIVE means moving the "
                                                "curve onto the two GT endpoints makes its "
                                                "geometry worse.",
               "anchor_noise": "k=2 with the two anchors perturbed by Gaussian noise of sd "
                               "sigma (iso = 3D, tangent = along the contour), the perturbed "
                               "anchors scored. sigma=0 is the oracle. breakeven_sigma_mm is "
                               "an SD and is NOT comparable to a measured mean error; "
                               "breakeven_mean_disp_mm is the same breakeven as a mean "
                               "displacement and IS. Read it against the field named by "
                               "compare_against."},
           "conclusion": conclude(res, comb, noise),
           "contours": res, "combine": comb, "anchor_noise": noise,
           "all85": {k: round(sum(res[nm][k] * res[nm]["n"] for _, _, nm in CONT) / W, 4)
                     for k in ("baseline_mm", "gt_curve_uniform_mm", "gt_curve_popprofile_mm",
                               "pred_curve_freepoint_floor_mm",
                               "pred_curve_freepoint_floor_gtaligned_mm",
                               "zero_oracle_reparam_mm")}}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\n{out['conclusion']}")
    print(f"\nwrote {OUT}")
    return out


# ------------------------------------------------------------------ smoke test
def _fake(ne=40, seed=0, jitter=0.0, noise=0.6):
    """Synthetic ears: each contour is a polyline built from a smoothly turning direction
    walk with chord lengths taken straight from a per-contour profile. Its OWN chord
    profile is therefore that profile exactly, so at jitter=0 every ear shares it and the
    population-profile floor has a known answer of zero."""
    rng = np.random.RandomState(seed)
    GT = np.empty((ne, 85, 3))
    U = {nm: np.r_[0.0, np.sort(rng.rand(hi - lo - 1)) * 0.9 + 0.05, 1.0]
         for lo, hi, nm in CONT}
    for e in range(ne):
        for lo, hi, nm in CONT:
            n = hi - lo + 1
            u = np.clip(U[nm] + np.r_[0.0, rng.randn(n - 2) * jitter, 0.0], 1e-4, 1 - 1e-4)
            u = np.r_[0.0, np.sort(u[1:-1]), 1.0]
            d = rng.randn(3)
            D = np.empty((n - 1, 3))
            for j in range(n - 1):
                d = d + rng.randn(3) * 0.45
                D[j] = d / np.linalg.norm(d)
            L = 20.0 + 80.0 * rng.rand()
            GT[e, lo:hi + 1] = np.r_[np.zeros((1, 3)),
                                     np.cumsum(D * (np.diff(u) * L)[:, None], 0)] \
                + rng.randn(3) * 10
    P = GT + rng.randn(ne, 85, 3) * noise
    return GT, P


def smoke():
    print("=" * 96)
    print("SMOKE -- primitives, leakage guard, then the FULL analysis on synthetic ears")
    rng = np.random.RandomState(1)

    # a1 round trip: resampling a polyline at its OWN profile is the identity
    Q = np.cumsum(rng.randn(12, 3), 0)
    assert np.abs(resample(Q, profile(Q)) - Q).max() < 1e-9

    # a2 similarity fit: k=2 puts both anchors exactly on target; k>=3 recovers an exact
    #    similarity exactly (this is the transform every anchor oracle row depends on)
    G2 = np.cumsum(rng.randn(2, 3), 0) * 3
    T = sim_fit(Q[[0, -1]], G2)
    assert np.abs(T(Q)[[0, -1]] - G2).max() < 1e-9, "k=2 alignment misses its anchors"
    R = min_rot(rng.randn(3), rng.randn(3))
    assert np.abs(R @ R.T - np.eye(3)).max() < 1e-9 and abs(np.linalg.det(R) - 1) < 1e-9
    Y = Q @ R.T * 2.3 + np.array([5.0, -1.0, 2.0])
    assert np.abs(sim_fit(Q, Y)(Q) - Y).max() < 1e-8, "Umeyama fit is not exact"
    assert np.abs(min_rot(np.array([1., 0, 0]), np.array([-1., 0, 0]))
                  @ np.array([1., 0, 0]) + np.array([1., 0, 0])).max() < 1e-9

    # a3 free-point floor of a polyline against its own vertices is zero
    assert dist_to_poly(Q, Q).max() < 1e-9
    print(f"  primitives OK  resample/profile round-trip {np.abs(resample(Q, profile(Q))-Q).max():.2e}"
          f"  |  k=2 anchor residual {np.abs(T(Q)[[0,-1]]-G2).max():.2e}"
          f"  |  Umeyama residual {np.abs(sim_fit(Q, Y)(Q)-Y).max():.2e}")

    # b the leakage surface. Three checks: the folds partition the ears, the profile is
    #    built from exactly the complement of the scored ear's fold, and EXCLUDING the
    #    validation ears actually changes the answer -- otherwise the guard guards nothing.
    GT, P = _fake(jitter=0.0)
    subj, parts = frozen_folds(len(GT))
    cover = np.zeros(len(GT), int)
    for f in range(NFOLD):
        tr, va = split(f, subj, parts)
        cover[va] += 1
        assert not np.isin(tr, va).any() and len(tr) + len(va) == len(GT)
    assert (cover == 1).all(), "the frozen folds do not partition the ears"
    bad = [p.copy() for p in parts]
    bad[0] = np.append(bad[0], parts[1][0])          # one subject now claimed by two folds
    try:
        analyse(P, GT, subj, bad, kmax=2)
        raise SystemExit("an ear in two folds was ACCEPTED -- the leakage guard is dead")
    except AssertionError as e:
        assert "exactly one fold" in str(e), str(e)
    GTj, _ = _fake(jitter=0.03, seed=7)
    u0, U0, tr0, va0 = population_profile(GTj, 55, 74, 0, subj, parts)
    leaky = np.stack([profile(GTj[e, 55:75]) for e in range(len(GTj))]).mean(0)
    assert np.abs(u0 - leaky).max() > 1e-6, \
        "including the validation ears changes nothing, so this test proves nothing"
    print(f"  folds partition {len(GT)} ears; an ear in two folds is refused; fold-0 "
          f"profile uses {len(tr0)} training ears ({len(va0)} held out) and differs from "
          f"the all-ears profile by {np.abs(u0 - leaky).max():.2e}")

    # c exact-profile ears -> the population-profile floor and the k=2 oracle on a
    #   similarity-transformed GT curve must both be ~0
    res, PRED, fold_of = analyse(P, GT, subj, parts, kmax=4)
    assert (fold_of >= 0).all() and PRED["inner_helix"]["k2"].shape == (len(GT), 20, 3)
    fl = max(res[nm]["gt_curve_popprofile_mm"] for _, _, nm in CONT)
    assert fl < 1e-6, f"shared-profile floor should be 0 on exact-profile ears, got {fl}"
    # a curve that IS the GT curve under a similarity must be recovered EXACTLY once the
    # anchors pin the rotation. Two anchors cannot: the roll about their chord is free, so
    # k=2 leaves a real residual and k=3 does not. That asymmetry is the whole point of
    # the global fit in place(), so it is asserted in both directions.
    # the transform must be inside the model's hypothesis class: a similarity when SCALE,
    # rigid when not, otherwise SCALE=0 fails a test about the ROLL for the wrong reason.
    Rr = min_rot(np.array([0., 0, 1]), np.array([1., 2, 3]))
    Psim = GT @ Rr.T * (1.4 if SCALE else 1.0) + np.array([3., 1, -2])
    r23, _, _ = analyse(Psim, GT, subj, parts, kmax=3)
    w2 = max(r23[nm]["anchor_oracle_mm"][2] for _, _, nm in CONT)
    w3 = max(r23[nm]["anchor_oracle_mm"][3] for _, _, nm in CONT)
    assert w3 < 1e-6, f"k=3 must be exact on a similarity-transformed GT curve: {w3}"
    assert w2 > 1e-3, "k=2 recovered a roll it cannot see -- the test is not testing"
    print(f"  exact-profile ears: population floor {fl:.2e} mm  |  GT curve under a "
          f"similarity: k=2 {w2:.4f} mm (chord-roll ambiguity), k=3 {w3:.2e} mm")

    # d the analysis must SEE a pure phase error and must NOT invent a gain out of iid
    #    noise. Same exact-profile ears, predictions slid along the CORRECT curve.
    slide = GT.copy()
    for e in range(len(GT)):
        for lo, hi, nm in CONT:
            n = hi - lo + 1
            v = np.clip(profile(GT[e, lo:hi + 1]) + rng.randn(n) * 0.05, 0, 1)
            slide[e, lo:hi + 1] = resample(GT[e, lo:hi + 1], np.r_[0.0, np.sort(v[1:-1]), 1.0])
    rp, _, _ = analyse(slide, GT, subj, parts, kmax=2)
    for _, _, nm in CONT:
        r = rp[nm]
        # not 0: sliding the landmarks changes which corners the POLYLINE cuts, so the
        # slid curve is not the GT curve and the secant difference survives. >=70% is the
        # bar -- below it the oracle would not be measuring phase at all.
        assert r["anchor_oracle_mm"][2] < 0.30 * r["baseline_mm"], \
            f"{nm}: pure phase error not recovered ({r['baseline_mm']:.4f} -> " \
            f"{r['anchor_oracle_mm'][2]:.4f}) -- the oracle does not measure phase"
    print("  pure phase error (GT curve, slid landmarks) recovered: "
          + "  ".join(f"{nm.split('_')[0]} {rp[nm]['baseline_mm']:.3f}->"
                      f"{rp[nm]['anchor_oracle_mm'][2]:.3f}"
                      f"({100*(1-rp[nm]['anchor_oracle_mm'][2]/rp[nm]['baseline_mm']):.0f}%)"
                      for _, _, nm in CONT))

    # e the full analysis path on ears with a JITTERED profile and noisy predictions
    GT, P = _fake(jitter=0.02, noise=0.6, seed=3)
    res, PRED, _ = analyse(P, GT, subj, parts, kmax=4)
    comb = combine(P, GT, PRED, res)
    for _, _, nm in CONT:
        r = res[nm]
        # the free-point floor is a lower bound on ANY reparameterisation by construction
        assert r["pred_curve_freepoint_floor_mm"] <= r["baseline_mm"] + 1e-9
        assert r["gt_curve_popprofile_mm"] > 0
        # monotonicity in k is an EXPECTATION, not an invariant (a segment-local alignment
        # can be worse than a global one), so it is reported rather than asserted
        a = r["anchor_oracle_mm"]
        if not all(a[k + 1] <= a[k] + 1e-9 for k in r["ks"][:-1]):
            print(f"  note: {nm} anchor curve is NOT monotone in k: {a}")
    assert PRED["outer_helix"]["noanchor"].shape == (len(GT), 25, 3)
    nz = {m: anchor_noise(P, GT, subj, parts, m, sigmas=[0.0, 0.5, 1.0], nrep=2)
          for m in ("iso", "tangent")}
    for m in nz:
        for _, _, nm in CONT:
            c = nz[m][nm]["by_sigma_mm"]
            assert abs(c[0.0] - res[nm]["anchor_oracle_mm"][2]) < 1e-9, \
                f"{m}/{nm}: sigma=0 must reproduce the k=2 oracle ({c[0.0]} vs " \
                f"{res[nm]['anchor_oracle_mm'][2]})"
            assert c[1.0] > c[0.5] > c[0.0], f"{m}/{nm}: noise did not hurt monotonically {c}"
    report(res, comb, "SYNTHETIC 40 ears, profile jitter 0.02, prediction noise 0.6mm", nz)
    print("\n  the (E,85,3) contract: " + " ".join(
        f"{nm}{PRED[nm]['k2'].shape}" for _, _, nm in CONT)
        + f" -> reassembled {combine(P, GT, PRED, res)['baseline_all85_mm']:.4f} mm over "
          f"{P.shape} predictions")
    assert P.shape == GT.shape == (40, 85, 3)
    print("SMOKE PASS")
    print("=" * 96)


if __name__ == "__main__":
    smoke()
    if os.environ.get("SMOKE_ONLY", "0") != "1":
        if os.path.exists(f"{WORK}/ortho_feats.npz"):
            main()
        else:
            print(f"\n{WORK}/ortho_feats.npz absent -- analysis skipped, smoke test only")
