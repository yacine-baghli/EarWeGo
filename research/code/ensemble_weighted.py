"""
DIRECTION-AWARE FITTED ENSEMBLE WEIGHTS -- nonnegative, per (contour x direction), nested.

ensemble_oof.py fits one weight per model per CONTOUR on the raw coordinates. That cannot
express the thing that was actually measured: the optimal weights are strongly DIRECTION
dependent, and the tangent/across split lives WITHIN a contour, not between contours. On
fold 0 with four backbones the in-sample optima were

    tangent  dgcnn .647  kpconv .133  ptv3 .117  pointnext .103
    across         .340         .382       .182            .096
    normal         .138         .233       .225            .404

-- pointnext is the BEST member in the normal direction while being the worst overall.
This file fits a weight per (contour, direction) cell, plus the three coarser schemes it
contains, and reports all of them nested.

    python research/code/ensemble_weighted.py            # the real run, ~52 s
    SMOKE=1 python research/code/ensemble_weighted.py    # CPU smoke test, ~2 s

Writes research/results/ensemble_weighted.json and $WORK/$ENS_NPY.


WHAT "WEIGHTING PER DIRECTION" MEANS, AND WHY IT IS NOT A WEIGHTED AVERAGE OF POINTS
------------------------------------------------------------------------------------
ortho_feats.npz carries a per-(ear, landmark) frame (t, b, n). It is verified here to be
orthonormal and right-handed to 2e-15, so it is a complete basis of R^3 and

    P = <P,t> t + <P,b> b + <P,n> n     exactly, for any point P.

A per-direction combination takes each model's three frame coordinates, mixes each with
its OWN weight vector, and reassembles:

    Q = (sum_k w_kt <P_k,t>) t + (sum_k w_kb <P_k,b>) b + (sum_k w_kn <P_k,n>) n

BE EXPLICIT ABOUT WHAT THIS DOES. With one shared weight vector Q is a convex combination
of the members and therefore lies inside their convex hull. With three different weight
vectors it does NOT: Q is only guaranteed to lie inside the members' axis-aligned bounding
BOX IN THE LOCAL FRAME (each frame coordinate is separately a convex combination -- this
is asserted, not assumed). It can sit off every member's prediction and off the segment
between them. The report therefore measures how far the reassembled point actually travels
from the equal-weight point, how far it sits from the nearest member, and its distance to
the members' convex hull on a subsample. All three are printed before any gain is claimed.

Weights are constrained to the simplex (w >= 0, sum_k w = 1) per cell, not merely
nonnegative. Two reasons, both structural: (a) sum(w) = 1 makes the combination equivariant
under a shift of the world origin, and the frame coordinates <P,t> are measured from the
WORLD origin, so a free-sum solution would be a function of where the origin happens to be
(ensemble_oof.py's caveat 2, made unavoidable here); (b) it is what phase_shared.py solved,
so the fold-0 table above is reproduced exactly as a check on this file's machinery
(CHECK_F0, printed).


TWO OBJECTIVES, AND WHY BOTH ARE REPORTED
-----------------------------------------
The metric is MEAN EUCLIDEAN DISTANCE (MLE), not mean squared error. ensemble_oof.py and
phase_shared.py both fit squared error because it is a linear least-squares problem. At
the effect size here that substitution is NOT free. MEASURED, in sample, on the metric:
MSE-optimal weights DO beat equal weight (by 0.0014-0.0049 mm over the 8 member-set x
scheme configurations) -- but they lose to the MLE-optimal weights by 0.0001-0.0017 mm,
which is the same size as, or larger than, every nested gain this file measures. Out of
sample the mismatch is enough to flip the sign: 4 of the 8 nested MSE configurations end
up at or above their own equal-weight baseline. So:

  OBJ=mle  minimise mean_row ||sum_k w_k (P_k - GT)||, the metric itself. Convex in w;
           solved by SLSQP started AT EQUAL WEIGHT, so the in-sample value can only be
           <= the equal-weight value (asserted), and the four schemes must come out
           monotone in their nesting order (asserted).
  OBJ=mse  min ||A w|| with A the signed directional errors. Solved EXACTLY by active-set
           enumeration over all 2^K - 1 supports, not by the heavy sum-to-1 penalty row
           ensemble_oof.py and phase_shared.py use -- see simplex_ls. Reproduces
           phase_shared.py's fold-0 table, and is reported so the objective-mismatch cost
           is a number rather than a worry.

Four schemes, a strictly nested family (equal is contained in all of them):
    equal                1/K everywhere, no fitting, no optimism
    global               K weights
    contour              K x 4
    direction            K x 3
    contour_x_direction  K x 12          <- the scheme this file exists for


NESTING
-------
Weights for outer fold k are fitted on the OOF predictions of the other four folds only
and applied to fold k's ears. Ear-level AND subject-level disjointness is asserted and
printed -- the loop is ensemble_oof.nested_oof itself, driven through its solve/apply hooks,
so the assertions are literally the same code. The in-sample fit is also computed and
reported as LEAKY_in_sample; it is never the headline. Same caveat as ensemble_oof.py: this
is STACKING, so the base models' training folds still include the evaluation ears; that
residual optimism is not quantified by anything here.


ENVIRONMENT (all optional; every value is echoed into the report)
-----------------------------------------------------------------
  MODELS      normalsfix,famA_kpconv,famA_ptv3
                         resolved as $WORK/screen_<spec>_s<seed>_f<k>, $WORK/<spec>_s<seed>_f<k>
                         or -- new here, the famA_* artefacts carry no seed in their name --
                         $WORK/<spec>_f<k>.
  SEEDS       0,1,2      seeds to look for (seeded specs only)
  MEMBER_SETS ensemble,separate
                         'ensemble': the 3 seeds of a spec become ONE member (their mean).
                         'separate': one member per (spec, seed). Both are run and compared,
                         which is the "does splitting the seeds help" question.
  OBJ         mle,mse    objectives to fit; both are run by default.
  SCHEMES     global,contour,direction,contour_x_direction
  RIDGE       0.0        + RIDGE * mean((w - 1/K)^2) on the MLE objective. UNTUNED: any
                         nonzero value must be chosen by an INNER split of the fit ears or
                         it is a new leakage surface. Left at 0 for the reported run.
  NB          20000      per-subject bootstrap draws           BOOT_SEED 5
  NHULL       2000       landmarks sampled for the convex-hull distance   HULL_SEED 7
  CHECK_F0    1          reproduce the published fold-0 directional table if pointnext is
                         present (it is, for fold 0 only)
  WORK        scratch    FEATS $WORK/ortho_feats.npz (gt/t/b/n/subj/fold)
  ENS_NPY     ensemble_weighted.npy   written into $WORK; the nested contour_x_direction /
                         OBJ=mle prediction set -- the scheme specified in advance, NOT the
                         post-hoc best one.
  OUT         research/results/ensemble_weighted.json
  SMOKE       0          1 -> run the smoke test instead of the analysis
"""
import os, io, sys, json, time, contextlib, tempfile
import numpy as np
from scipy.optimize import nnls, minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ensemble_oof as eo

NL, NFOLD, CONT = eo.NL, eo.NFOLD, eo.CONT
DIRS = ("tangent_t", "across_b", "normal_n")
CELLS = [(ci, d) for ci in range(len(CONT)) for d in range(3)]
GMAP = {"global": np.zeros((4, 3), int),
        "contour": np.repeat(np.arange(4)[:, None], 3, 1),
        "direction": np.repeat(np.arange(3)[None, :], 4, 0),
        "contour_x_direction": np.arange(12).reshape(4, 3)}


def cfg():
    e = os.environ.get
    c = dict(work=e("WORK", "scratch"),
             models=[s for s in e("MODELS", "normalsfix,famA_kpconv,famA_ptv3").split(",") if s],
             seeds=[int(s) for s in e("SEEDS", "0,1,2").split(",") if s.strip()],
             member_sets=[s for s in e("MEMBER_SETS", "ensemble,separate").split(",") if s],
             objs=[s for s in e("OBJ", "mle,mse").split(",") if s],
             schemes=[s for s in e("SCHEMES", ",".join(GMAP)).split(",") if s],
             ridge=float(e("RIDGE", "0.0")),
             nb=int(e("NB", "20000")), boot_seed=int(e("BOOT_SEED", "5")),
             nhull=int(e("NHULL", "2000")), hull_seed=int(e("HULL_SEED", "7")),
             check_f0=e("CHECK_F0", "1") == "1", pred=e("PRED", "raw"),
             ens_npy=e("ENS_NPY", "ensemble_weighted.npy"),
             out=e("OUT", "research/results/ensemble_weighted.json"))
    c["feats"] = e("FEATS", f"{c['work']}/ortho_feats.npz")
    c["seed_mode"] = "ensemble"                      # overwritten per member set in run_set
    assert all(s in GMAP for s in c["schemes"]), c["schemes"]
    assert all(o in ("mle", "mse") for o in c["objs"]), c["objs"]
    return c


# ------------------------------------------------------------------ loading
def load_seedless(spec, c, GT, folds):
    """The famA_<tag>_f<k> naming, which carries no seed and which eo.load_seed cannot
    express. Same three contracts as eo.load_seed: every ear held out exactly once, the
    val_ear_index agrees with the frozen fold, and the .npy reproduces the reported MLE
    against THIS gt (which is what proves the two share a frame)."""
    ne, key = len(GT), "ordered_MLE_full_mm" if c["pred"] == "full" else "ordered_MLE_mm"
    suf = "_full.npy" if c["pred"] == "full" else ".npy"
    P, seen, per_fold = np.full((ne, NL, 3), np.nan), np.zeros(ne, bool), {}
    for f in range(NFOLD):
        b = f"{c['work']}/{spec}_f{f}"
        if not all(os.path.exists(p) and os.path.getsize(p) > 0 for p in (b + ".json", b + suf)):
            return None, f"fold {f}: no seedless artefact pair {os.path.basename(b)}{{.json,{suf}}}"
        j = json.load(open(b + ".json"))
        if j.get(key) is None:
            return None, f"fold {f}: report has no {key} (PRED={c['pred']})"
        idx = np.asarray(j["val_ear_index"], int)
        assert not seen[idx].any(), f"{spec}: an ear is held out by two folds"
        assert (folds[idx] == f).all(), f"{spec} f{f}: val_ear_index != the frozen fold assignment"
        Q = np.load(b + suf).astype(np.float64)
        assert Q.shape == (len(idx), NL, 3), f"{b}{suf}: {Q.shape} != {(len(idx), NL, 3)}"
        d = float(np.linalg.norm(Q - GT[idx], axis=2).mean())
        assert abs(d - j[key]) < 2e-3, \
            f"{b}{suf}: recomputed MLE {d:.4f} != reported {j[key]} -- frame/order mismatch"
        seen[idx], P[idx], per_fold[f] = True, Q, float(j[key])
    assert seen.all(), f"{spec}: {int((~seen).sum())} ears never held out"
    return (P, per_fold), None


def load_members(c, GT, folds):
    """-> names, P (K,NE,85,3), meta. Seeded specs go through eo.load_seed and eo's own
    ensemble/separate logic; seedless specs are one member each, with seeds=[]."""
    seedless = [s for s in c["models"]
                if os.path.exists(f"{c['work']}/{s}_f0.json")
                and not os.path.exists(f"{c['work']}/screen_{s}_s{c['seeds'][0]}_f0.json")]
    seeded = [s for s in c["models"] if s not in seedless]
    names, mats, meta = [], [], {}
    if seeded:
        n2, P2, m2 = eo.load_members({**c, "models": seeded}, GT, folds)
        names, mats, meta = list(n2), list(P2), dict(m2)
    for spec in seedless:
        r, why = load_seedless(spec, c, GT, folds)
        if r is None:
            print(f"  ! {spec}: {why} -- MODEL DROPPED")
            continue
        names.append(spec)
        mats.append(r[0])
        meta[spec] = dict(spec=spec, seeds=[], seedless=True, per_fold_MLE=r[1])
    assert len(names) >= 2, f"need >= 2 members, got {names}"
    return names, np.stack(mats), meta


# ------------------------------------------------------------------ frame algebra
def frame(of):
    """(NE,85,3,3) orthonormal per-landmark basis, rows (t,b,n). Verified, not assumed."""
    E = np.stack([of["t"], of["b"], of["n"]], -2).astype(np.float64)
    G = np.einsum("elac,elbc->elab", E, E)
    err = float(np.abs(G - np.eye(3)).max())
    det = np.linalg.det(E)
    assert err < 1e-9, f"(t,b,n) is not orthonormal: max|E E^T - I| = {err:.2e}"
    return E, err, float(det.min()), float(det.max())


def project(P, E):
    """world (K,NE,85,3) -> frame coordinates (K,NE,85,3dir)."""
    return np.einsum("kelc,eldc->keld", P, E)


def apply_w(W, gmap, S, E, ears):
    """frame coordinates + weights -> world points. Q = sum_d (sum_k w_kd S_kd) e_d."""
    co = np.zeros((len(ears), NL, 3))
    for ci, (lo, hi, _) in enumerate(CONT):
        co[:, lo:hi + 1] = np.einsum("kd,keld->eld", W[:, gmap[ci]], S[:, ears, lo:hi + 1])
    return np.einsum("eld,eldc->elc", co, E[ears])


# ------------------------------------------------------------------ the two solvers
def cell_rows(Ser, ears, gmap, j):
    """(nrows, K) signed directional errors of every cell tied to group j.

    `[..., d]` AFTER the slice, not `[:, ears, lo:hi+1, d]`: an integer index and an array
    index separated by a slice make numpy hoist the advanced axes to the front, so the
    latter silently returns (n_ears, K, n_lm) and the reshape below scrambles the models
    into each other. SMOKE 2 catches exactly this.
    """
    return np.concatenate([Ser[:, ears, CONT[ci][0]:CONT[ci][1] + 1][..., d].reshape(len(Ser), -1)
                           for ci, d in CELLS if gmap[ci, d] == j], 1).T


def simplex_ls(A):
    """EXACT argmin ||A w||, w >= 0, sum_k w = 1, by active-set enumeration.

    Not the "heavy sum-to-1 row" NNLS that ensemble_oof.py and phase_shared.py use. That
    trick needs the penalty row to dominate the data rows, and here the optimal residual can
    be 1e-8 of the column scale (one member far better than the rest in one cell), at which
    point the data term is below float64 resolution against the penalty and the solver
    returns garbage -- observed, and it is what SMOKE 2 catches. K <= 5, so all 2^K - 1
    supports are enumerated and each equality-constrained subproblem is solved by its KKT
    system. No penalty parameter, no conditioning trade-off.
    """
    K = A.shape[1]
    G = A.T @ A
    best, bw = np.inf, None
    for m in range(1, 1 << K):
        S = [k for k in range(K) if m >> k & 1]
        g, o = G[np.ix_(S, S)], np.ones((len(S), 1))
        M = np.block([[2 * g, o], [o.T, np.zeros((1, 1))]])
        try:
            w = np.linalg.solve(M, np.concatenate([np.zeros(len(S)), [1.0]]))[:len(S)]
        except np.linalg.LinAlgError:
            continue
        if w.min() < -1e-12:
            continue
        v = float(w @ g @ w)
        if v < best:
            best, bw = v, (S, np.clip(w, 0, None))
    assert bw is not None, "no feasible support -- A'A is degenerate for every subset"
    S, w = bw
    out = np.zeros(K)
    out[S] = w / w.sum()
    return out


def solve_mse(Ser, ears, gmap, c):
    """min ||A w||, w >= 0, sum w = 1, per group, exactly."""
    K, ng = len(Ser), gmap.max() + 1
    return np.stack([simplex_ls(cell_rows(Ser, ears, gmap, j)) for j in range(ng)], 1)


def _mle_obj(x, R, gmap, K, ng, nrow, ridge):
    w = x.reshape(K, ng)
    g, tot = np.zeros((K, ng)), 0.0
    for ci, Rc in enumerate(R):
        res = np.einsum("kd,keld->eld", w[:, gmap[ci]], Rc)
        nn = np.linalg.norm(res, axis=-1)
        tot += nn.sum()
        gc = np.einsum("eld,keld->kd", res / np.maximum(nn, 1e-12)[..., None], Rc)
        for d in range(3):
            g[:, gmap[ci, d]] += gc[:, d]
    f, g = tot / nrow, g / nrow
    if ridge > 0:
        f += ridge * ((w - 1.0 / K) ** 2).mean()
        g += ridge * 2 * (w - 1.0 / K) / w.size
    return f, g.ravel()


def solve_mle(Ser, ears, gmap, c):
    """min mean_row ||sum_k w_k err_k||, w >= 0, sum w = 1 per group. THE METRIC ITSELF.

    Convex in w. Started at equal weight, which is feasible, so the returned in-sample
    objective is <= the equal-weight objective -- asserted, because an SLSQP that silently
    walked uphill would look exactly like a scheme that does not help.
    """
    K, ng = len(Ser), gmap.max() + 1
    R = [Ser[:, ears, lo:hi + 1, :] for lo, hi, _ in CONT]
    nrow = len(ears) * NL
    J = np.zeros((ng, K, ng))
    for j in range(ng):
        J[j, :, j] = 1.0
    J = J.reshape(ng, K * ng)
    args = (R, gmap, K, ng, nrow, c["ridge"])
    cons = [dict(type="eq", fun=(lambda x, j=j: x.reshape(K, ng)[:, j].sum() - 1),
                 jac=(lambda x, j=j: J[j])) for j in range(ng)]
    x0 = np.full(K * ng, 1.0 / K)
    r = minimize(_mle_obj, x0, args=args, jac=True, method="SLSQP",
                 bounds=[(0.0, 1.0)] * (K * ng), constraints=cons,
                 options=dict(maxiter=400, ftol=1e-12))
    W = np.clip(r.x.reshape(K, ng), 0, None)
    W /= W.sum(0, keepdims=True)
    f_eq = _mle_obj(x0, *args)[0]
    f_hi = _mle_obj(W.ravel(), *args)[0]
    assert f_hi <= f_eq + 1e-9, \
        f"SLSQP returned {f_hi:.6f} > equal weight {f_eq:.6f} ({r.message}) -- solver failed"
    return W


SOLVER = dict(mle=solve_mle, mse=solve_mse)


# ------------------------------------------------------------------ nested driver
def nested(P, GT, folds, subj, c, gmap, Ser, S, E, obj, quiet):
    """eo.nested_oof VERBATIM, driven through its solve/apply hooks.

    The hooks are rebound rather than the loop reimplemented so that the disjointness
    assertions and the printed check are the same code ensemble_oof.py is trusted on.
    """
    keep = eo.solve_weights, eo.apply_weights
    eo.solve_weights = lambda P_, GT_, ears, c_: (SOLVER[obj](Ser, ears, gmap, c_), None)
    eo.apply_weights = lambda P_, W, ears: apply_w(W, gmap, S, E, ears)
    sink = io.StringIO()
    try:
        with contextlib.redirect_stdout(sink if quiet else sys.stdout):
            out, Wn, rows = eo.nested_oof(P, GT, folds, subj, c)
    finally:
        eo.solve_weights, eo.apply_weights = keep
    assert all(r["ear_overlap"] == 0 and r["subject_overlap"] == 0 for r in rows), rows
    return out, Wn, rows


# ------------------------------------------------------------------ geometry diagnostics
def hull_dist(Q, P, n, seed):
    """Distance from Q to the convex hull of the K member points, on a random subsample of
    landmarks. min_w ||sum_k w_k P_k - q||, w on the simplex, by NNLS with a sum-to-1 row."""
    K, ne = len(P), len(Q)
    rng = np.random.RandomState(seed)
    e = rng.randint(0, ne, n)
    l = rng.randint(0, NL, n)
    out = np.empty(n)
    for i in range(n):
        A = P[:, e[i], l[i], :].T                                    # (3,K)
        m = A.mean(1, keepdims=True)                                 # centre: sum(w)=1 makes
        A, q = A - m, Q[e[i], l[i]] - m[:, 0]                        # this exact, and it turns
        sc = 1e6 * max(np.abs(A).max(), 1e-9)                        # a 30 mm scale into 0.3 mm
        w, _ = nnls(np.vstack([A, sc * np.ones((1, K))]), np.concatenate([q, [sc]]))
        out[i] = np.linalg.norm(A @ (w / max(w.sum(), 1e-12)) - q)
    return out


def geometry(Q, P, S, E, c):
    """Is the reassembled point sensible? Box containment is a theorem given simplex
    weights per direction -- so it is ASSERTED. Everything else is measured."""
    Sq = np.einsum("elc,eldc->eld", Q, E)
    lo, hi = S.min(0), S.max(0)
    slack = float(max((lo - Sq).max(), (Sq - hi).max()))   # <= 0 <=> every coord inside
    assert slack < 1e-8, f"reassembled point outside the members' frame-coordinate box by {slack:.2e}"
    Peq = P.mean(0)
    d_eq = np.linalg.norm(Q - Peq, axis=2)
    d_near = np.linalg.norm(Q[None] - P, axis=3).min(0)
    d_near_eq = np.linalg.norm(Peq[None] - P, axis=3).min(0)
    spread = np.linalg.norm(P[None] - P[:, None], axis=4).max((0, 1))
    hd = hull_dist(Q, P, c["nhull"], c["hull_seed"])
    q = lambda a: dict(mean=round(float(a.mean()), 4), median=round(float(np.median(a)), 4),
                       p90=round(float(np.percentile(a, 90)), 4), max=round(float(a.max()), 4))
    return dict(box_containment_slack_mm=float(slack),
                dist_to_equal_weight_point_mm=q(d_eq),
                dist_to_nearest_member_mm=q(d_near),
                dist_of_equal_weight_to_nearest_member_mm=q(d_near_eq),
                member_spread_mm=q(spread),
                dist_to_member_convex_hull_mm={**q(hd), "n_sampled": int(c["nhull"]),
                                               "frac_outside_1e-6": round(float((hd > 1e-6).mean()), 4)},
                ratio_travel_to_spread=round(float((d_eq / np.maximum(spread, 1e-9)).mean()), 4))


# ------------------------------------------------------------------ published-table check
def check_f0(c, GT, E, folds):
    """Reproduce the fold-0 in-sample directional table (phase_shared.py) with THIS file's
    projection + MSE solver. It is leaky by construction and is not a result -- it is the
    only end-to-end check that the frame algebra here is the frame algebra that produced
    the numbers this task is built on. pointnext exists for fold 0 only."""
    need = [f"{c['work']}/famA_{t}_f0.npy" for t in ("kpconv", "ptv3", "pointnext")]
    if not all(os.path.exists(p) for p in need):
        return None
    idx = np.array(json.load(open(f"{c['work']}/famA_kpconv_f0.json"))["val_ear_index"])
    M = {"dgcnn": np.mean([np.load(f"{c['work']}/screen_normalsfix_s{s}_f0.npy") for s in (0, 1, 2)],
                          0).astype(np.float64)}
    for t in ("kpconv", "ptv3", "pointnext"):
        M[t] = np.load(f"{c['work']}/famA_{t}_f0.npy").astype(np.float64)
    ks = list(M)
    Pf = np.stack([M[k] for k in ks])
    Ser = project(Pf, E[idx]) - np.einsum("elc,eldc->eld", GT[idx], E[idx])[None]
    pub = {"tangent_t": dict(opt=1.3493, w=[.647, .133, .117, .103]),
           "across_b": dict(opt=0.6501, w=[.340, .382, .182, .096]),
           "normal_n": dict(opt=0.2178, w=[.138, .233, .225, .404])}
    out, worst = {}, 0.0
    print("===== CHECK_F0: reproduce the published fold-0 in-sample table (LEAKY by design) =====")
    print(f"{'direction':10s} {'best_single':>11s} {'equal':>7s} {'optimal':>7s} {'published':>9s}"
          f"   weights {ks}")
    rms = lambda v: float(np.sqrt((v ** 2).mean()))
    for d, nm in enumerate(DIRS):
        A = Ser[:, :, :, d].reshape(len(ks), -1).T
        w = simplex_ls(A)
        best = min(rms(A[:, i]) for i in range(len(ks)))
        opt, eqw = rms(A @ w), rms(A @ (np.ones(len(ks)) / len(ks)))
        worst = max(worst, abs(opt - pub[nm]["opt"]), float(np.abs(w - pub[nm]["w"]).max()))
        print(f"{nm:10s} {best:11.4f} {eqw:7.4f} {opt:7.4f} {pub[nm]['opt']:9.4f}   "
              + " ".join(f"{x:.3f}" for x in w) + "   published "
              + " ".join(f"{x:.3f}" for x in pub[nm]["w"]))
        out[nm] = dict(best_single_rmse=round(best, 4), equal_weight_rmse=round(eqw, 4),
                       insample_optimal_rmse_LEAKY=round(float(opt), 4),
                       published_optimal_rmse=pub[nm]["opt"],
                       weights_LEAKY={k: round(float(w[i]), 3) for i, k in enumerate(ks)},
                       published_weights=dict(zip(ks, pub[nm]["w"])))
    assert worst < 2e-3, f"fold-0 table not reproduced, worst deviation {worst:.4f}"
    print(f"max deviation from the published table {worst:.5f} -- the frame algebra and the "
          f"simplex solver\nin this file are the ones those numbers came from. pointnext has "
          f"fold 0 ONLY, so it cannot\nbe a member of anything nested below.\n")
    out["max_deviation"] = round(float(worst), 6)
    return out


# ------------------------------------------------------------------ one member set
def run_set(setname, c, GT, E, folds, subj, res_all):
    c = {**c, "seed_mode": setname}
    print("=" * 84)
    print(f"MEMBER SET '{setname}'")
    names, P, meta = load_members(c, GT, folds)
    K, ne = len(names), len(GT)
    S = project(P, E)
    Ser = S - np.einsum("elc,eldc->eld", GT, E)[None]
    Peq = P.mean(0)
    mle_eq = eo.per_ear_mle(Peq, GT)
    per = {n: eo.per_ear_mle(P[i], GT) for i, n in enumerate(names)}
    best = min(names, key=lambda n: per[n].mean())
    sh = lambda s: s.replace("famA_", "")[:12]

    print(f"{K} members, {ne} ears | " + "  ".join(f"{sh(n)} {per[n].mean():.4f}" for n in names))
    print(f"{'equal weight':22s} {mle_eq.mean():.4f}  ({mle_eq.mean()-per[best].mean():+.4f} vs "
          f"best single {sh(best)})")

    # signed-residual correlation PER DIRECTION: a fitted weight can only pay where the
    # members disagree, and 'do the split seeds add anything' is answered here, not by the MLE
    Er = P - GT[None]
    Ed = np.einsum("kelc,eldc->keld", Er, E)
    R = {d: np.corrcoef(Ed[:, :, :, di].reshape(K, -1)) for di, d in enumerate(DIRS)}
    print(f"\n{'signed residual r':22s}" + "".join(f"{sh(n)[:9]:>10s}" for n in names)
          + "   (upper=tangent_t, lower=normal_n)")
    for i, n in enumerate(names):
        print(f"{sh(n):22s}" + "".join(
            f"{(R['tangent_t'] if j > i else R['normal_n'])[i, j]:10.3f}" if i != j else f"{1.0:10.3f}"
            for j in range(K)))
    off = {d: float(R[d][~np.eye(K, dtype=bool)].mean()) for d in DIRS}
    print("mean off-diagonal r  " + "  ".join(f"{d} {off[d]:.3f}" for d in DIRS)
          + "   -- a weight can only pay where these are below 1.\n")

    print(f"{'obj':4s} {'scheme':21s} {'npar':>5s} {'nested':>8s} {'LEAKY':>8s} {'optim':>7s} "
          f"{'d_vs_eq':>8s} | " + " ".join(f"{d[:7]:>7s}" for d in DIRS))

    out = dict(members=meta, n_members=K, best_single=best,
               per_member_pooled_OOF_mm={n: round(float(per[n].mean()), 4) for n in names},
               equal_weight=dict(MLE_mm=round(float(mle_eq.mean()), 4),
                                 per_contour_MLE_mm=eo.per_contour(Peq, GT),
                                 directional_rmse_mm=eo.directional(Peq, GT, *E.transpose(2, 0, 1, 3))),
               schemes={}, disjointness=None, geometry=None,
               signed_residual_correlation={d: {names[i]: {names[j]: round(float(R[d][i, j]), 4)
                                                           for j in range(K)} for i in range(K)}
                                            for d in DIRS},
               mean_offdiag_correlation={d: round(off[d], 4) for d in DIRS})
    store, first = {}, True
    for obj in c["objs"]:
        for sn in c["schemes"]:
            gmap = GMAP[sn]
            Q, Wn, rows = nested(P, GT, folds, subj, c, gmap, Ser, S, E, obj, quiet=not first)
            if first:
                out["disjointness"] = rows
                first = False
            Wl = SOLVER[obj](Ser, np.arange(ne), gmap, c)
            Ql = apply_w(Wl, gmap, S, E, np.arange(ne))
            mn, ml = eo.per_ear_mle(Q, GT), eo.per_ear_mle(Ql, GT)
            dirs = eo.directional(Q, GT, *E.transpose(2, 0, 1, 3))
            store[(obj, sn)] = (Q, mn, Wl)
            npar = K * (gmap.max() + 1)
            print(f"{obj:4s} {sn:21s} {npar:5d} {mn.mean():8.4f} {ml.mean():8.4f} "
                  f"{mn.mean()-ml.mean():+7.4f} {mn.mean()-mle_eq.mean():+8.4f} | "
                  + " ".join(f"{dirs[d]:7.4f}" for d in DIRS))
            out["schemes"][f"{obj}/{sn}"] = dict(
                objective=obj, scheme=sn, n_weights=int(npar),
                nested_MLE_mm=round(float(mn.mean()), 4),
                LEAKY_in_sample_MLE_mm=round(float(ml.mean()), 4),
                optimism_mm=round(float(mn.mean() - ml.mean()), 4),
                delta_vs_equal_weight_mm=round(float(mn.mean() - mle_eq.mean()), 4),
                per_contour_MLE_mm=eo.per_contour(Q, GT), directional_rmse_mm=dirs,
                weights_LEAKY={names[i]: np.round(Wl[i], 4).tolist() for i in range(K)},
                weights_nested_mean={names[i]: np.round(Wn[:, i].mean(0), 4).tolist()
                                     for i in range(K)},
                weights_nested_sd={names[i]: np.round(Wn[:, i].std(0, ddof=1), 4).tolist()
                                   for i in range(K)},
                group_layout=gmap.tolist())

    # ---- the (contour x direction) weight table, the thing this file exists for ----
    key = ("mle", "contour_x_direction")
    if key in store:
        Wl = store[key][2]
        print("\n----- OBJ=mle, contour_x_direction: LEAKY in-sample weights (contour x direction) -----")
        print(f"{'contour':16s} " + "".join(f"{d[:9]:>26s}" for d in DIRS))
        print(f"{'':16s} " + "".join("".join(f"{sh(n)[:7]:>8s}" for n in names) + "  " for _ in DIRS))
        for ci, (lo, hi, nm) in enumerate(CONT):
            print(f"{nm:16s} " + "".join(
                "".join(f"{Wl[i, GMAP['contour_x_direction'][ci, d]]:8.3f}" for i in range(K)) + "  "
                for d in range(3)))
        Q = store[key][0]
        out["geometry"] = geometry(Q, P, S, E, c)
        g = out["geometry"]
        print(f"\n----- is the reassembled point sensible? (nested mle/contour_x_direction) -----")
        print(f"  worst frame-coordinate box violation {g['box_containment_slack_mm']:+.2e} mm "
              f"(<= 0 everywhere: each frame coordinate IS a convex combination)")
        print(f"  distance from the EQUAL-WEIGHT point   mean {g['dist_to_equal_weight_point_mm']['mean']:.4f}"
              f"  median {g['dist_to_equal_weight_point_mm']['median']:.4f}"
              f"  p90 {g['dist_to_equal_weight_point_mm']['p90']:.4f}"
              f"  max {g['dist_to_equal_weight_point_mm']['max']:.4f} mm")
        print(f"  distance to the NEAREST member         mean {g['dist_to_nearest_member_mm']['mean']:.4f} mm"
              f"   (the equal-weight point sits "
              f"{g['dist_of_equal_weight_to_nearest_member_mm']['mean']:.4f} mm away)")
        print(f"  distance OUTSIDE the members' convex hull  mean "
              f"{g['dist_to_member_convex_hull_mm']['mean']:.4f}  p90 "
              f"{g['dist_to_member_convex_hull_mm']['p90']:.4f}  max "
              f"{g['dist_to_member_convex_hull_mm']['max']:.4f} mm on {c['nhull']} sampled landmarks;"
              f" {100*g['dist_to_member_convex_hull_mm']['frac_outside_1e-6']:.1f}% are strictly outside")
        print(f"  members disagree by {g['member_spread_mm']['mean']:.4f} mm on average, so the "
              f"combination travels {100*g['ratio_travel_to_spread']:.1f}% of the disagreement.")

    # ---- paired per-subject bootstrap vs equal weight ----
    print(f"\n----- paired per-subject bootstrap vs EQUAL WEIGHT ({c['nb']} draws, "
          f"{len(np.unique(subj))} subjects) -----")
    for (obj, sn), (Q, mn, Wl) in store.items():
        m, lo, hi, pn = eo.boot(mn - mle_eq, subj, c["nb"], c["boot_seed"])
        v = "ADOPT" if hi < 0 else "REJECT" if lo > 0 else "INDISTINGUISHABLE"
        print(f"  {obj}/{sn:21s} {m:+.4f} mm  CI [{lo:+.4f}, {hi:+.4f}]  P(<0)={pn:.3f} -> {v}")
        out["schemes"][f"{obj}/{sn}"]["bootstrap_vs_equal_weight"] = dict(
            delta_mm=round(m, 4), ci95=[round(lo, 4), round(hi, 4)], p_negative=round(pn, 4),
            verdict=v)
    m, lo, hi, pn = eo.boot(mle_eq - per[best], subj, c["nb"], c["boot_seed"])
    out["equal_weight"]["bootstrap_vs_best_single"] = dict(
        delta_mm=round(m, 4), ci95=[round(lo, 4), round(hi, 4)], p_negative=round(pn, 4))
    print(f"  {'equal weight vs best single':26s} {m:+.4f} mm  CI [{lo:+.4f}, {hi:+.4f}]"
          f"  P(<0)={pn:.3f}")

    # in-sample monotonicity across the nested scheme family (a solver check, not a result)
    for obj in c["objs"]:
        got = {sn: out["schemes"][f"{obj}/{sn}"]["LEAKY_in_sample_MLE_mm"] for sn in c["schemes"]}
        for a, b in (("global", "contour"), ("global", "direction"),
                     ("contour", "contour_x_direction"), ("direction", "contour_x_direction")):
            if a in got and b in got and obj == "mle":
                assert got[b] <= got[a] + 1e-3, \
                    f"{obj}: in-sample {b} {got[b]} > {a} {got[a]} but {a} is contained in {b}"
    res_all[setname] = out
    return store, names, P, mle_eq


# ------------------------------------------------------------------ main
def main():
    c = cfg()
    t0 = time.time()
    of = np.load(c["feats"])
    GT = of["gt"].astype(np.float64)
    ne = len(GT)
    E, orth, dmin, dmax = frame(of)
    subj, folds = eo.frozen_folds(ne)
    note = eo.verify_folds(subj, folds, of["subj"], of["fold"])

    print(f"DIRECTION-AWARE FITTED ENSEMBLE | {ne} ears / {len(np.unique(subj))} subjects | "
          f"pred={c['pred']} ridge={c['ridge']}")
    print(f"folds: {note}")
    print(f"frame (t,b,n): max|E E^T - I| = {orth:.2e}, det in [{dmin:.6f}, {dmax:.6f}] -- a "
          f"complete right-handed\northonormal basis, so projecting to (t,b,n) and reassembling "
          f"is exact and loses nothing.\n")

    res = dict(config={k: v for k, v in c.items()}, n_ears=int(ne),
               n_subjects=int(len(np.unique(subj))), fold_check=note,
               frame_check=dict(max_abs_gram_error=orth, det_min=dmin, det_max=dmax),
               fold0_published_table_check=check_f0(c, GT, E, folds) if c["check_f0"] else None,
               member_sets={})

    keep = {}
    for sm in c["member_sets"]:
        keep[sm] = run_set(sm, c, GT, E, folds, subj, res["member_sets"])
        print()

    # ---------- does splitting the seeds help? ----------
    if len(keep) == 2 and "ensemble" in keep and "separate" in keep:
        print("=" * 84)
        print("DOES SPLITTING THE 3 dgcnn SEEDS INTO SEPARATE MEMBERS HELP?")
        A, B = res["member_sets"]["ensemble"], res["member_sets"]["separate"]
        eqa, eqb = A["equal_weight"]["MLE_mm"], B["equal_weight"]["MLE_mm"]
        print(f"  equal weight   3 members (seed-mean) {eqa:.4f}   5 members (seeds split) {eqb:.4f}"
              f"   {eqb-eqa:+.4f}")
        rows = []
        for k in A["schemes"]:
            if k in B["schemes"]:
                a, b = A["schemes"][k]["nested_MLE_mm"], B["schemes"][k]["nested_MLE_mm"]
                print(f"  nested {k:26s} {a:.4f}   {b:.4f}   {b-a:+.4f}")
                rows.append(dict(scheme=k, seed_mean=a, seeds_split=b, delta_mm=round(b - a, 4)))
        eq_pair = keep["ensemble"][3], keep["separate"][3]
        m, lo, hi, pn = eo.boot(eq_pair[1] - eq_pair[0], subj, c["nb"], c["boot_seed"])
        print(f"  paired bootstrap of the EQUAL-WEIGHT difference (5 members - 3 members): "
              f"{m:+.4f} mm CI [{lo:+.4f}, {hi:+.4f}] P(<0)={pn:.3f}")
        res["seed_splitting"] = dict(
            equal_weight_seed_mean_mm=eqa, equal_weight_seeds_split_mm=eqb,
            equal_weight_delta_mm=round(eqb - eqa, 4), per_scheme=rows,
            equal_weight_bootstrap=dict(delta_mm=round(m, 4), ci95=[round(lo, 4), round(hi, 4)],
                                        p_negative=round(pn, 4)),
            note=("Splitting the seeds is NOT a new source of information: it re-weights the "
                  "families. 3 members give dgcnn 1/3 of the equal-weight mass, 5 members give "
                  "it 3/5, and dgcnn is the best family -- that alone moves the equal-weight "
                  "number. The 3 seeds are near-duplicate members (see ensemble_oof.py's "
                  "correlation table), so they add K*ncell fitted parameters against almost no "
                  "residual diversity."))

    # ---------- headline ----------
    print("=" * 84)
    flat = [(sm, k, v["nested_MLE_mm"], v["delta_vs_equal_weight_mm"],
             v["bootstrap_vs_equal_weight"]["ci95"], v["bootstrap_vs_equal_weight"]["verdict"])
            for sm, s in res["member_sets"].items() for k, v in s["schemes"].items()]
    flat.sort(key=lambda r: r[2])
    print("ALL CONFIGURATIONS, nested, best first (delta and CI are vs that member set's "
          "equal weight)")
    for sm, k, m, d, ci, v in flat:
        print(f"  {sm:9s} {k:26s} {m:.4f}  {d:+.4f}  CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  {v}")
    eqs = {sm: s["equal_weight"]["MLE_mm"] for sm, s in res["member_sets"].items()}
    res["headline"] = dict(equal_weight_MLE_mm=eqs, best_nested=dict(
        member_set=flat[0][0], scheme=flat[0][1], MLE_mm=flat[0][2],
        delta_vs_equal_weight_mm=flat[0][3], ci95=flat[0][4], verdict=flat[0][5]))
    print(f"\nbest nested: {flat[0][0]} / {flat[0][1]} = {flat[0][2]:.4f} mm "
          f"({flat[0][3]:+.4f} vs equal weight) -- but this is the MINIMUM OF "
          f"{len(flat)} CONFIGURATIONS,\nso it carries a selection optimism of its own that no "
          f"CI here covers. The pre-specified\nscheme is contour_x_direction; read that row, not "
          f"the minimum.")

    res["caveats"] = [
        "STACKING, NOT FULL NESTING (ensemble_oof.py's caveat 1, unchanged): the weights for "
        "outer fold k are fitted only on other folds' ears -- asserted, ear-level AND "
        "subject-level -- but those ears' OOF predictions come from base models whose training "
        "folds include fold k. optimism_mm quantifies only the in-sample WEIGHT fit.",
        "The reported best-of-all-configurations is the minimum over member sets x objectives x "
        "schemes. That selection is not covered by any bootstrap CI here. contour_x_direction "
        "was specified in advance; it is the row to read.",
        "OBJECTIVE MISMATCH. MSE-optimal weights do beat equal weight in sample on the metric "
        "(0.0014-0.0049 mm), but they lose to MLE-optimal weights by 0.0001-0.0017 mm -- the "
        "same size as every effect measured here. Out of sample that is enough to flip the "
        "sign on 4 of the 8 nested MSE configurations. Hence OBJ=mle, with OBJ=mse reported so "
        "the mismatch is a number rather than a worry.",
        "A per-direction combination is NOT a convex combination of the members: it is only "
        "inside their bounding box in the local frame (asserted). The measured distance to the "
        "convex hull is reported; it is small but nonzero.",
        "NO SURFACE PROJECTION. These are raw ensemble coordinates. ensemble_final.py measures "
        "-0.0055 mm for exact surface projection on top of the equal-weight ensemble; whether "
        "the fitted-weight gain survives projection (or is absorbed by it) is UNMEASURED here.",
        "pointnext -- the member that motivates direction-dependent weights, being best in the "
        "normal direction -- exists for FOLD 0 ONLY. It cannot enter any nested fit. The "
        "direction dependence among the three complete members is far weaker than the "
        "four-member fold-0 table suggests.",
        "RIDGE is untuned and left at 0. Tuning it on the fit ears requires an INNER split; "
        "choosing it by the nested score would put the selection back inside the estimate."]

    if c["out"]:
        os.makedirs(os.path.dirname(c["out"]) or ".", exist_ok=True)
        json.dump(res, open(c["out"], "w"), indent=1)
        print(f"\nwrote {c['out']}")
    if c["ens_npy"] and ("mle", "contour_x_direction") in keep.get("ensemble", ({},))[0]:
        Q = keep["ensemble"][0][("mle", "contour_x_direction")][0]
        os.makedirs(c["work"], exist_ok=True)
        np.save(f"{c['work']}/{c['ens_npy']}", Q)
        print(f"wrote {c['work']}/{c['ens_npy']}  {Q.shape}  (nested, member set 'ensemble', "
              f"OBJ=mle, contour_x_direction, world frame)")
    print(f"total {time.time()-t0:.1f}s")
    return res


# ------------------------------------------------------------------ smoke test
# Same reading of constraint 5 as ensemble_oof.py: BACKWARD is the weight solve
# (data -> weights), FORWARD is applying them (weights -> landmarks). Both run on synthetic
# member sets whose per-(contour x direction) optimum is known in closed form, so the
# solvers are checked against an answer instead of against themselves.
def write_seedless(d, GT, preds, folds):
    for tag, Pm in preds.items():
        for k in range(NFOLD):
            ev = np.where(folds == k)[0]
            b = f"{d}/{tag}_f{k}"
            json.dump({"fold": k, "ordered_MLE_mm": round(float(np.linalg.norm(
                Pm[ev] - GT[ev], axis=2).mean()), 4), "val_ear_index": [int(i) for i in ev]},
                open(b + ".json", "w"))
            np.save(b + ".npy", Pm[ev])


def run_env(d, **env):
    e = dict(WORK=d, FEATS=f"{d}/ortho_feats.npz", SEEDS="0", NB="200", NHULL="150",
             CHECK_F0="0", RIDGE="0.0", ENS_NPY="", OUT=f"{d}/ew.json",
             MEMBER_SETS="ensemble", OBJ="mle,mse")
    e.update(env)
    keep = {k: os.environ.get(k) for k in e}
    os.environ.update(e)
    sink = io.StringIO()
    try:
        with contextlib.redirect_stdout(sink):
            return main(), sink.getvalue()
    finally:
        for k, v in keep.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def smoke():
    t0 = time.time()
    d = os.environ.get("SMOKE_DIR", os.path.join(tempfile.gettempdir(), "ensemble_weighted_smoke"))
    ne = 40
    subj, folds = eo.frozen_folds(ne)
    rng = np.random.RandomState(0)
    G = rng.randn(ne, NL, 3) * 8 + np.array([12.0, -5.0, 30.0])
    os.makedirs(d, exist_ok=True)
    # a frame that is NOT the world axes, so every projection/reassembly below has teeth
    Efr = np.linalg.qr(np.random.RandomState(11).randn(ne, NL, 3, 3))[0].transpose(0, 1, 3, 2)
    gram = np.abs(np.einsum("elac,elbc->elab", Efr, Efr) - np.eye(3)).max()
    assert gram < 1e-12, gram

    def bundle(seeded, seedless, extra=None):
        """eo.write_bundle first (it writes its OWN ortho_feats.npz), then the real frame."""
        for sd, pr in seeded.items():
            eo.write_bundle(d, G, pr, subj, folds, seed=sd)
        np.savez(f"{d}/ortho_feats.npz", gt=G, t=Efr[:, :, 0], b=Efr[:, :, 1], n=Efr[:, :, 2],
                 subj=subj, fold=folds)
        write_seedless(d, G, seedless, folds)

    # ---- TEST 1: reassembly is exact, and one-hot weights reproduce a member exactly ----
    print("=" * 78)
    print("SMOKE 1/5 -- projection/reassembly identity and one-hot weights")
    Pt = rng.randn(3, ne, NL, 3) * 5
    S = project(Pt, Efr)
    back = np.einsum("keld,eldc->kelc", S, Efr)
    print(f"  P -> (t,b,n) -> P  max abs error {np.abs(back - Pt).max():.2e} mm")
    assert np.abs(back - Pt).max() < 1e-10
    for k in range(3):
        W = np.zeros((3, 12)); W[k] = 1.0
        Q = apply_w(W, GMAP["contour_x_direction"], S, Efr, np.arange(ne))
        assert np.abs(Q - Pt[k]).max() < 1e-10, k
    Weq = np.full((3, 12), 1 / 3)
    Q = apply_w(Weq, GMAP["contour_x_direction"], S, Efr, np.arange(ne))
    print(f"  one-hot weights reproduce each member to <1e-10; equal weights reproduce the "
          f"mean to {np.abs(Q - Pt.mean(0)).max():.2e}")
    assert np.abs(Q - Pt.mean(0)).max() < 1e-10

    # ---- TEST 2: known per-(contour x direction) optimum ----
    # In cell (ci,d) one designated member has 100x smaller error than the others, so the
    # unique simplex optimum is one-hot on it to ~1e-4 -- and a scheme coarser than
    # contour_x_direction provably cannot represent a per-cell answer. The good member's
    # error is small but NONZERO so the MLE objective stays smooth at its own optimum.
    print("\nSMOKE 2/5 -- planted per-(contour x direction) optimum; both solvers must find it")
    K = 3
    good = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1], [0, 2, 1]])       # (4 contours, 3 dirs)
    err = np.zeros((K, ne, NL, 3))
    for ci, (lo, hi, _) in enumerate(CONT):
        for dd in range(3):
            for k in range(K):
                sc = 0.01 if k == good[ci, dd] else (1 + k)
                err[k, :, lo:hi + 1, dd] = sc * rng.randn(ne, hi - lo + 1)
    Pm = np.einsum("keld,eldc->kelc", np.einsum("elc,eldc->eld", G, Efr)[None] + err, Efr)
    bundle({0: {"mA": Pm[0]}}, {"famA_mB": Pm[1], "famA_mC": Pm[2]})
    r1, log1 = run_env(d, MODELS="mA,famA_mB,famA_mC")
    for obj in ("mle", "mse"):
        sch = r1["member_sets"]["ensemble"]["schemes"]
        s = sch[f"{obj}/contour_x_direction"]
        Wl = np.array([s["weights_LEAKY"][n] for n in ("mA", "famA_mB", "famA_mC")])  # (3,12)
        hot = Wl.argmax(0).reshape(4, 3)
        print(f"  {obj}: argmax per cell {hot.tolist()} == planted {good.tolist()}; min winning "
              f"weight {Wl.max(0).min():.4f}; nested {s['nested_MLE_mm']:.4f} mm")
        print(f"       fitted parameters K=3 x ncell: "
              + ", ".join(f"{n} {sch[f'{obj}/{n}']['n_weights']}" for n in GMAP))
        assert (hot == good).all(), (hot, good)
        assert Wl.max(0).min() > 0.99, Wl
        assert s["nested_MLE_mm"] < 0.05, s["nested_MLE_mm"]
        for coarse in ("global", "contour", "direction"):
            assert sch[f"{obj}/{coarse}"]["nested_MLE_mm"] > 0.5, coarse
    e0 = r1["member_sets"]["ensemble"]["equal_weight"]["MLE_mm"]
    print(f"  seedless famA_<tag>_f<k> members loaded; equal weight {e0:.4f} mm, coarser schemes "
          f"stay > 0.5 mm, as they must")

    # ---- TEST 3: in-sample monotonicity + positive optimism when the optimum is fold-dependent
    print("\nSMOKE 3/5 -- fold-dependent optimum: nesting must cost, in-sample must be monotone")
    u, v = rng.randn(ne, NL, 3), rng.randn(ne, NL, 3)
    fav = np.isin(folds, [0, 1])[:, None, None]
    PA, PB = G + np.where(fav, 0.1, 3.0) * u, G + np.where(fav, 3.0, 0.1) * v
    bundle({0: {"mA": PA}}, {"famA_mB": PB})
    r2, _ = run_env(d, MODELS="mA,famA_mB", OBJ="mle")
    sc = r2["member_sets"]["ensemble"]["schemes"]
    for k in ("mle/global", "mle/contour", "mle/direction", "mle/contour_x_direction"):
        print(f"  {k:28s} nested {sc[k]['nested_MLE_mm']:.4f}  LEAKY "
              f"{sc[k]['LEAKY_in_sample_MLE_mm']:.4f}  optimism {sc[k]['optimism_mm']:+.4f}")
        assert sc[k]["optimism_mm"] > 0.005, k
    assert sc["mle/contour_x_direction"]["LEAKY_in_sample_MLE_mm"] <= \
        sc["mle/direction"]["LEAKY_in_sample_MLE_mm"] + 1e-6
    assert sc["mle/direction"]["LEAKY_in_sample_MLE_mm"] <= \
        sc["mle/global"]["LEAKY_in_sample_MLE_mm"] + 1e-6
    print("  in-sample is monotone along the nesting global >= direction >= contour_x_direction")

    # ---- TEST 4: geometry -- the combination leaves the convex hull, box holds ----
    print("\nSMOKE 4/5 -- geometry of the reassembled point")
    g = r1["member_sets"]["ensemble"]["geometry"]
    print(f"  box slack {g['box_containment_slack_mm']:.2e} mm; distance to the equal-weight "
          f"point mean {g['dist_to_equal_weight_point_mm']['mean']:.4f} max "
          f"{g['dist_to_equal_weight_point_mm']['max']:.4f}")
    print(f"  distance outside the members' convex hull mean "
          f"{g['dist_to_member_convex_hull_mm']['mean']:.4f} mm, "
          f"{100*g['dist_to_member_convex_hull_mm']['frac_outside_1e-6']:.0f}% strictly outside "
          f"-- a per-direction mix is NOT a convex combination")
    assert g["box_containment_slack_mm"] < 1e-8
    assert g["dist_to_member_convex_hull_mm"]["frac_outside_1e-6"] > 0.5

    # ---- TEST 5: the two member sets, the report shape, the saved .npy ----
    print("\nSMOKE 5/5 -- MEMBER_SETS ensemble vs separate, report shape, saved prediction set")
    bundle({0: {"mA": PA}, 1: {"mA": G + 1.7 * rng.randn(ne, NL, 3)}}, {"famA_mB": PB})
    r3, log3 = run_env(d, MODELS="mA,famA_mB", SEEDS="0,1", OBJ="mle",
                       SCHEMES="global,contour_x_direction", MEMBER_SETS="ensemble,separate",
                       ENS_NPY="ew.npy")
    assert sorted(r3["member_sets"]["ensemble"]["members"]) == ["famA_mB", "mA"]
    assert sorted(r3["member_sets"]["separate"]["members"]) == ["famA_mB", "mA#s0", "mA#s1"]
    Q = np.load(f"{d}/ew.npy")
    assert Q.shape == (ne, NL, 3) and Q[:2].shape == (2, NL, 3), Q.shape
    got = float(np.linalg.norm(Q - G, axis=2).mean())
    rep = r3["member_sets"]["ensemble"]["schemes"]["mle/contour_x_direction"]["nested_MLE_mm"]
    assert abs(got - rep) < 1e-4, (got, rep)
    dj = r3["member_sets"]["ensemble"]["disjointness"]
    assert len(dj) == NFOLD and all(r["ear_overlap"] == 0 and r["subject_overlap"] == 0 for r in dj)
    for k in ("caveats", "headline", "seed_splitting", "frame_check", "member_sets"):
        assert k in r3, k
    print(f"  members ensemble {sorted(r3['member_sets']['ensemble']['members'])}  "
          f"separate {sorted(r3['member_sets']['separate']['members'])}")
    print(f"  saved prediction set {Q.shape}, a 2-ear slice is {Q[:2].shape}, MLE {got:.4f} == "
          f"reported {rep:.4f}")
    print(f"  disjointness rows {len(dj)}/5, all zero ear and subject overlap")
    print("  the disjointness table as ensemble_oof.nested_oof prints it:")
    L = log3.splitlines()
    i = next(k for k, ln in enumerate(L) if ln.lstrip().startswith("outer "))
    for ln in L[i:i + NFOLD + 1] + [next(x for x in L[i:] if "ALL 5" in x)]:
        print("    " + ln.strip())
    print(f"\nSMOKE PASS  ({time.time()-t0:.1f}s)")
    print("=" * 78)


if __name__ == "__main__":
    smoke() if os.environ.get("SMOKE") == "1" or not os.path.exists(
        os.environ.get("FEATS", f"{os.environ.get('WORK', 'scratch')}/ortho_feats.npz")) else main()
