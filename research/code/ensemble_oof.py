"""
CROSS-FAMILY ENSEMBLER -- paired residual diversity, per-contour nonnegative weights, and
a NESTED-OOF weight fit, so the reported gain is not the gain of fitting on the very ears
the fit is then scored on.

Why this file exists, and in this order. Seven variants of the 2048-point DGCNN family
are null or harmful and the error decomposition is invariant across all of them, so the
only gain still available from models that ALREADY EXIST is combination -- and a
combination pays exactly as much as its members' residuals differ. Hence the first thing
printed is not a gain, it is the PAIRED RESIDUAL CORRELATION table: cross-family
diversity is the asset, and two models correlating at 0.98 cannot produce a gain no
matter how the weights are solved. A near-duplicate pair is flagged (DUP_R, default 0.95)
because it also makes the weight split between the duplicates ill-determined, which shows
up as a large across-fold weight sd in the table further down.

Second: at the effect sizes in play here a weight fitted in-sample is not a small sin.
The whole 2-seed prediction-ensemble gain is -0.032 mm and the seed main effect sd is
0.0165 mm, so in-sample optimism of a few hundredths of a millimetre is the same size as
everything being measured. The fit is therefore NESTED over the SAME frozen 5 folds: for
outer fold k the weights are solved on the OOF predictions of the other four folds only
and applied to fold k's ears. Ear-level AND subject-level disjointness of every
(fit, eval) pair is asserted and printed. The in-sample fit is computed as well and
reported as LEAKY_in_sample, purely to quantify the optimism -- it is never the headline.

    MODELS=base,fam_kpconv,fam_template SEEDS=0,1,2 python research/code/ensemble_oof.py
    MODELS=base SEEDS=0,1 SEED_MODE=separate python research/code/ensemble_oof.py
    python research/code/ensemble_oof.py        # <- no MODELS set: runs the smoke test

Writes research/results/ensemble.json (aggregates only -- no landmark coordinates) and
$WORK/$ENS_NPY, the nested-OOF ensemble prediction set: (E,85,3), WORLD frame, in the
train+val ear order the projection scripts assume. Surface-projecting it is one adapter
step, not a pipe -- proj_all.py hardcodes its input path (scratch/oof_tta.npz) and reads
the keys pred/gt/fold, so the .npy has to be wrapped into such an npz under a new name
and proj_all.py pointed at it.


ENVIRONMENT (all optional; every value is echoed into the report)
----------------------------------------------------------------
  MODELS      base       comma-separated model specs. A spec resolves to whichever of
                         $WORK/screen_<spec>_s<seed>_f<fold>  (gpu_screen.py, and
                         train_family.py's ALIAS) or
                         $WORK/<spec>_s<seed>_f<fold>         (train_family.py's own TAG,
                         e.g. fam_kpconv) is present, in that order.
  SEEDS       0,1,2      seeds to look for. A seed is used only if all 5 folds are there.
  SEED_MODE   ensemble   'ensemble': one member per spec, mean over its complete seeds
                         (the shippable quantity -- see cv_multiseed.py).
                         'separate': one member per (spec, seed).
  PRED        raw        'raw'  -> <base>.npy       + ordered_MLE_mm
                         'full' -> <base>_full.npy  + ordered_MLE_full_mm (train_family's
                         full-pipeline predictions, i.e. comparable to 1.3144 mm). PRED=full
                         needs train_family.py's OWN tag (MODELS=fam_kpconv, not kpconv):
                         its ALIAS block writes only screen_<variant>_*.{json,npy}, never
                         a _full.npy, so the screen_ name cannot serve PRED=full.
  SIMPLEX     0          1 -> constrain the weights of each contour to sum to 1.
  SIMPLEX_RHO 1e6        penalty scale for that constraint, relative to the data scale;
                         |sum(w)-1| is asserted below 1e-6 and reported.
  RIDGE       0.0        shrink the weights toward equal weighting, strength relative to
                         the data scale. Fitted inside the nested loop, so it is a
                         hyper-parameter of the FIT, not of the evaluation.
  DUP_R       0.95       flag a pair as near-duplicate above this residual correlation.
  NB          20000      per-subject bootstrap draws        BOOT_SEED 5
  WORK        scratch    FEATS $WORK/ortho_feats.npz (gt/t/b/n/subj/fold)
  ENS_NPY     ensemble_oof.npy   written into $WORK (gitignored)
  OUT         research/results/ensemble.json


WHAT THE WEIGHTS ARE
--------------------
One nonnegative weight per model PER CONTOUR (4 contours: 0-24 outer helix, 25-54 concha,
55-74 inner helix, 75-84 superior antihelix), solved by NNLS on the landmark COORDINATES:
a row of the design matrix is one (ear, landmark, coordinate) of that contour, a column is
one model, the target is ground truth. So the objective is the squared error whose mean
norm the MLE reports -- not a proxy for it. Per contour rather than global because the
per-contour MLEs differ by ~2x and a family can plausibly be better on the concha and
worse on the helix; per contour rather than per landmark because 85 x K weights on 272
training ears is where an ensembler starts fitting fold noise.

Three caveats a reader must have, all recorded in the JSON:

 1. STACKING ON OOF PREDICTIONS IS NOT FULLY NESTED, and no arrangement of these
    artefacts makes it so. The weights for outer fold k are fitted only on other folds'
    ears (that part is airtight, asserted below), but those ears' OOF predictions came
    from base models whose TRAINING folds include fold k. The base learners' parameters
    therefore are not independent of the evaluation ears. Full rigour needs the base
    models retrained inside each outer fold (25 trainings, not 5). What is reported here
    is the standard stacking estimate; the residual optimism it carries is NOT quantified
    by this script, and it is not the same thing as the in-sample optimism that IS.
 2. FRAME DEPENDENCE. The saved .npy files and ortho_feats.npz `gt` are in WORLD
    coordinates, and plain NNLS has no intercept, so a weight vector that does not sum to
    1 leaves a translation-proportional bias in the fit -- i.e. the solution depends on
    where the origin is. It is still consistent (the objective fitted is the objective
    scored), but it is not frame-invariant. SIMPLEX=1 makes it frame-invariant; sum(w) is
    printed for every contour so the size of the effect is visible either way.
 3. A WEIGHTED AVERAGE OF LANDMARKS IS NOT ON THE SURFACE and does not fix
    correspondence. The dominant error is ordered phase along the contour (along-contour
    RMSE 1.409 of a 1.4-ish total); averaging two models cancels the INDEPENDENT part of
    their phase error and cannot touch the shared part. The tangent/across/normal
    decomposition is printed for exactly this reason: if the gain is not in tangent_t, it
    is not a gain against the dominant term. With PRED=raw the numbers are raw-network
    OOF, NOT comparable to the shipped 1.273/1.3144 full-pipeline figures.
"""
import os, io, sys, json, tempfile, itertools
import numpy as np
from scipy.optimize import nnls

NL, NFOLD = 85, 5
CONT = [(0, 24, "outer_helix"), (25, 54, "concha"),
        (55, 74, "inner_helix"), (75, 84, "sup._antihelix")]


# ------------------------------------------------------------------ config
def cfg():
    e = os.environ.get
    c = dict(work=e("WORK", "scratch"),
             models=[s for s in e("MODELS", "base").split(",") if s],
             seeds=[int(s) for s in e("SEEDS", "0,1,2").split(",") if s.strip()],
             seed_mode=e("SEED_MODE", "ensemble"), pred=e("PRED", "raw"),
             simplex=e("SIMPLEX", "0") == "1", rho=float(e("SIMPLEX_RHO", "1e6")),
             ridge=float(e("RIDGE", "0.0")), dup_r=float(e("DUP_R", "0.95")),
             nb=int(e("NB", "20000")), boot_seed=int(e("BOOT_SEED", "5")),
             ens_npy=e("ENS_NPY", "ensemble_oof.npy"),
             out=e("OUT", "research/results/ensemble.json"))
    c["feats"] = e("FEATS", f"{c['work']}/ortho_feats.npz")
    assert c["seed_mode"] in ("ensemble", "separate"), c["seed_mode"]
    assert c["pred"] in ("raw", "full"), c["pred"]
    return c


# ------------------------------------------------------------------ frozen folds
def frozen_folds(ne):
    """Constraint 3, verbatim. subject = ear_index//2; array_split(RS(12345).perm, 5)."""
    subj = np.arange(ne) // 2
    parts = np.array_split(np.random.RandomState(12345).permutation(np.unique(subj)), NFOLD)
    of = np.full(ne, -1)
    for f, p in enumerate(parts):
        of[np.isin(subj, p)] = f
    assert (of >= 0).all()
    return subj, of


def verify_folds(subj, folds, feat_subj, feat_fold, path="research/results/folds.json"):
    """Three independent statements of the same split must agree before anything is fitted."""
    assert np.array_equal(subj, feat_subj), "ortho_feats.npz subj != ear_index//2"
    assert np.array_equal(folds, feat_fold), "ortho_feats.npz fold != the frozen rule"
    if not os.path.exists(path):
        return "frozen rule + ortho_feats.npz (folds.json absent)"
    a = json.load(open(path))["assignments"]
    if len(a) != len(folds):
        return f"frozen rule + ortho_feats.npz (folds.json has {len(a)} ears, data {len(folds)})"
    for r in a:
        assert folds[r["ear_index"]] == r["fold"], f"ear {r['ear_index']}: folds.json disagrees"
    return "frozen rule + ortho_feats.npz + research/results/folds.json"


# ------------------------------------------------------------------ loading
def load_seed(spec, seed, c, GT, folds):
    """Assemble one (spec, seed) pooled-OOF set. -> (P, per_fold_MLE) or (None, why)."""
    ne = len(GT)
    P, seen, per_fold = np.full((ne, NL, 3), np.nan), np.zeros(ne, bool), {}
    key = "ordered_MLE_full_mm" if c["pred"] == "full" else "ordered_MLE_mm"
    suf = "_full.npy" if c["pred"] == "full" else ".npy"
    styles = set()
    for f in range(NFOLD):
        hit = None
        for si, b in enumerate((f"{c['work']}/screen_{spec}_s{seed}_f{f}",
                                f"{c['work']}/{spec}_s{seed}_f{f}")):
            if all(os.path.exists(p) and os.path.getsize(p) > 0 for p in (b + ".json", b + suf)):
                hit = b
                styles.add(si)
                break
        if hit is None:
            return None, f"fold {f}: no complete artefact pair (*.json + *{suf})"
        try:
            j = json.load(open(hit + ".json"))
        except json.JSONDecodeError:
            return None, f"fold {f}: {os.path.basename(hit)}.json is corrupt"
        if j.get(key) is None:
            return None, f"fold {f}: report has no {key} (PRED={c['pred']})"
        idx = np.asarray(j["val_ear_index"], int)
        assert not seen[idx].any(), f"{spec} s{seed}: an ear is held out by two folds"
        assert (folds[idx] == f).all(), \
            f"{spec} s{seed} f{f}: val_ear_index disagrees with the frozen fold assignment"
        Q = np.load(hit + suf).astype(np.float64)
        assert Q.shape == (len(idx), NL, 3), f"{hit}{suf}: {Q.shape} != {(len(idx), NL, 3)}"
        d = float(np.linalg.norm(Q - GT[idx], axis=2).mean())
        # same contract screen_compare.py enforces: predictions must reproduce the reported
        # MLE against THIS gt, which is what proves .npy and ortho_feats share a frame.
        assert abs(d - j[key]) < 2e-3, \
            f"{hit}{suf}: recomputed MLE {d:.4f} != reported {j[key]} -- frame/order mismatch"
        seen[idx], P[idx], per_fold[f] = True, Q, float(j[key])
    if not seen.all():
        return None, f"{int((~seen).sum())} ears never held out"
    if len(styles) > 1:
        # both candidate names resolved for the same member: its five folds come from two
        # different writers (gpu_screen/ALIAS vs train_family's TAG), i.e. plausibly two
        # different trainings glued into one "member". Every fold still passes the fold and
        # MLE asserts, so nothing else catches this.
        print(f"  ! {spec} s{seed}: folds resolved from BOTH screen_{spec}_* and {spec}_* "
              f"-- one member assembled from two different runs? check the artefacts")
    return (P, per_fold), None


def load_members(c, GT, folds):
    """-> names, P (K,NE,85,3), meta. One member per spec (seed-mean) or per (spec,seed)."""
    names, mats, meta = [], [], {}
    for spec in c["models"]:
        got, why = {}, {}
        for s in c["seeds"]:
            r, err = load_seed(spec, s, c, GT, folds)
            if r is None:
                why[s] = err
            else:
                got[s] = r
        for s, e in why.items():
            print(f"  ! {spec} s{s}: {e} -- seed dropped")
        if not got:
            print(f"  ! {spec}: no complete seed -- MODEL DROPPED")
            continue
        if c["seed_mode"] == "separate":
            for s, (P, pf) in sorted(got.items()):
                nm = f"{spec}#s{s}"
                names.append(nm); mats.append(P)
                meta[nm] = dict(spec=spec, seeds=[s], per_fold_MLE=pf)
        else:
            ss = sorted(got)
            Pm = np.mean([got[s][0] for s in ss], axis=0)
            names.append(spec)
            mats.append(Pm)
            # per-fold MLE OF THE SEED-MEAN predictions, not the mean of the seeds' own
            # MLEs: averaging first is the entire point of this mode and the two differ by
            # the seed-ensemble gain (~0.032 mm), which would make the row contradict its
            # own pooled column.
            meta[spec] = dict(spec=spec, seeds=ss,
                              per_fold_MLE={f: round(float(per_ear_mle(Pm[folds == f],
                                                                      GT[folds == f]).mean()), 4)
                                            for f in range(NFOLD)})
    assert names, "no model could be loaded -- nothing to ensemble"
    return names, np.stack(mats), meta


# ------------------------------------------------------------------ metrics
def per_ear_mle(P, GT):
    return np.linalg.norm(P - GT, axis=2).mean(1)


def per_contour(P, GT):
    d = np.linalg.norm(P - GT, axis=2)
    return {nm: round(float(d[:, lo:hi + 1].mean()), 4) for lo, hi, nm in CONT}


def directional(P, GT, T, B, N):
    E = P - GT
    return {nm: round(float(np.sqrt((((E * V).sum(-1)) ** 2).mean())), 4)
            for nm, V in (("tangent_t", T), ("across_b", B), ("normal_n", N))}


def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def boot(diff, subj, nb, seed):
    """Paired PER-SUBJECT bootstrap: both ears of a subject move together (cv_verdict.py)."""
    us = np.unique(subj)
    per = {s: np.where(subj == s)[0] for s in us}
    rng = np.random.RandomState(seed)
    d = np.empty(nb)
    for k in range(nb):
        pick = np.concatenate([per[s] for s in rng.choice(us, len(us), replace=True)])
        d[k] = diff[pick].mean()
    return (float(diff.mean()), float(np.percentile(d, 2.5)),
            float(np.percentile(d, 97.5)), float((d < 0).mean()))


# ------------------------------------------------------------------ the solver
def solve_weights(P, GT, ears, c):
    """NNLS per contour on the landmark coordinates. -> W (K,4) >= 0, plus diagnostics.

    Rows are (ear, landmark, coordinate) triplets of one contour, columns are models. The
    optional simplex constraint is a heavily weighted extra row sum(w)=1 (its violation is
    asserted, not hoped for); the optional ridge is K extra rows pulling toward 1/K.
    """
    K = len(P)
    W, diag = np.zeros((K, len(CONT))), []
    for ci, (lo, hi, nm) in enumerate(CONT):
        A = P[:, ears, lo:hi + 1, :].reshape(K, -1).T
        b = GT[ears, lo:hi + 1, :].reshape(-1)
        sc = float(np.sqrt((A ** 2).mean())) * np.sqrt(len(b))     # column-norm scale
        A2, b2 = A, b
        if c["ridge"] > 0:
            r = np.sqrt(c["ridge"]) * sc
            A2 = np.vstack([A2, r * np.eye(K)])
            b2 = np.concatenate([b2, r * np.full(K, 1.0 / K)])
        if c["simplex"]:
            A2 = np.vstack([A2, c["rho"] * sc * np.ones((1, K))])
            b2 = np.concatenate([b2, [c["rho"] * sc]])
        w, _ = nnls(np.ascontiguousarray(A2), np.ascontiguousarray(b2))
        if c["simplex"]:
            assert abs(w.sum() - 1) < 1e-6, \
                f"simplex penalty failed on {nm}: sum(w)={w.sum():.9f}; raise SIMPLEX_RHO"
            w = w / w.sum()
        W[:, ci] = w
        res = float(np.sqrt(((A @ w - b) ** 2).mean()))
        diag.append(dict(contour=nm, rows=int(len(b)), sum_w=round(float(w.sum()), 6),
                         cond_A=round(float(np.linalg.cond(A)), 1),
                         fit_rms_mm=round(res, 4), active=int((w > 1e-9).sum())))
    return W, diag


def apply_weights(P, W, ears):
    out = np.zeros((len(ears), NL, 3))
    for ci, (lo, hi, nm) in enumerate(CONT):
        out[:, lo:hi + 1] = np.tensordot(W[:, ci], P[:, ears, lo:hi + 1, :], axes=(0, 0))
    return out


def nested_oof(P, GT, folds, subj, c):
    """Weights from the other four folds only, applied to this fold. The whole point."""
    ne = len(GT)
    out, times = np.full((ne, NL, 3), np.nan), np.zeros(ne, int)
    Ws, rows = [], []
    print("===== nested-OOF weight fit: disjointness check (constraint 2) =====")
    print(f"{'outer':>5s} {'fit_ear':>7s} {'ev_ear':>6s} {'ear_ovl':>7s} {'fit_sub':>7s} "
          f"{'ev_sub':>6s} {'sub_ovl':>7s} | {'fit_MLE':>7s} {'ev_MLE':>7s}")
    for k in range(NFOLD):
        fit, ev = np.where(folds != k)[0], np.where(folds == k)[0]
        eo = np.intersect1d(fit, ev)
        so = np.intersect1d(np.unique(subj[fit]), np.unique(subj[ev]))
        assert len(eo) == 0, f"outer {k}: {len(eo)} ears in BOTH the weight fit and its eval -- LEAK"
        assert len(so) == 0, f"outer {k}: {len(so)} subjects in BOTH fit and eval -- LEAK"
        W, _ = solve_weights(P, GT, fit, c)
        out[ev], times[ev] = apply_weights(P, W, ev), times[ev] + 1
        fm = float(per_ear_mle(apply_weights(P, W, fit), GT[fit]).mean())
        em = float(per_ear_mle(out[ev], GT[ev]).mean())
        print(f"{k:5d} {len(fit):7d} {len(ev):6d} {len(eo):7d} {len(np.unique(subj[fit])):7d} "
              f"{len(np.unique(subj[ev])):6d} {len(so):7d} | {fm:7.4f} {em:7.4f}")
        Ws.append(W)
        rows.append(dict(outer_fold=k, n_fit_ears=int(len(fit)), n_eval_ears=int(len(ev)),
                         ear_overlap=0, subject_overlap=0,
                         in_fit_MLE_mm=round(fm, 4), held_out_MLE_mm=round(em, 4)))
    assert (times == 1).all(), f"{int((times != 1).sum())} ears not scored exactly once"
    print(f"ALL {NFOLD} outer folds: fit and eval share ZERO ears and ZERO subjects; each of "
          f"the {ne} ears\nwas scored exactly once, by weights it did not contribute to. "
          f"(fit_MLE is in-sample\nfor that fold's weights, ev_MLE is held out -- the gap is "
          f"the optimism, per fold.)\n")
    return out, np.stack(Ws), rows


# ------------------------------------------------------------------ main
def main():
    c = cfg()
    of = np.load(c["feats"])
    GT, T, B, N = (of[k].astype(np.float64) for k in ("gt", "t", "b", "n"))
    ne = len(GT)
    subj, folds = frozen_folds(ne)
    note = verify_folds(subj, folds, of["subj"], of["fold"])

    print(f"CROSS-FAMILY ENSEMBLER | pred={c['pred']} seed_mode={c['seed_mode']} "
          f"seeds={c['seeds']} | {ne} ears / {len(np.unique(subj))} subjects")
    print(f"folds: {note}")
    print(f"weights: NNLS per contour, {'SIMPLEX (sum=1)' if c['simplex'] else 'nonnegative only'}"
          f", ridge={c['ridge']}\n")

    names, P, meta = load_members(c, GT, folds)
    K = len(names)
    mle = {nm: per_ear_mle(P[i], GT) for i, nm in enumerate(names)}
    sh = lambda s: s[:11]

    print(f"===== members ({K}) =====")
    print(f"{'model':12s} {'seeds':>7s} {'pooledOOF':>9s} | " +
          " ".join(f"{nm[:5]:>6s}" for _, _, nm in CONT) + " | " +
          " ".join(f"{'f'+str(f):>6s}" for f in range(NFOLD)))
    for i, nm in enumerate(names):
        pc = per_contour(P[i], GT)
        meta[nm]["pooled_OOF_MLE_mm"] = round(float(mle[nm].mean()), 4)
        meta[nm]["per_contour_MLE_mm"] = pc
        meta[nm]["directional_rmse_mm"] = directional(P[i], GT, T, B, N)
        print(f"{sh(nm):12s} {','.join(map(str, meta[nm]['seeds'])):>7s} "
              f"{mle[nm].mean():9.4f} | " + " ".join(f"{v:6.3f}" for v in pc.values()) + " | " +
              " ".join(f"{meta[nm]['per_fold_MLE'].get(f, float('nan')):6.3f}"
                       for f in range(NFOLD)))
    best = min(names, key=lambda n: mle[n].mean())
    print(f"best single model: {best}  {mle[best].mean():.4f} mm\n")
    if K < 2:
        print("Only one member loaded -- diversity and weights are undefined. Nothing to do.")
        return None

    # ---------- paired residual correlations: the thing worth having ----------
    E = P - GT[None]
    print("===== paired residual correlation: signed residual components (NE x 85 x 3) =====")
    Rm = np.eye(K)
    for i, j in itertools.combinations(range(K), 2):
        Rm[i, j] = Rm[j, i] = pearson(E[i].ravel(), E[j].ravel())
    print(f"{'':12s}" + "".join(f"{sh(n):>12s}" for n in names))
    for i, n in enumerate(names):
        print(f"{sh(n):12s}" + "".join(f"{Rm[i, j]:12.3f}" for j in range(K)))

    print(f"\n{'pair':26s} {'r_signed':>8s} {'r_perEar':>8s} {'r_perLM':>8s} {'r_MLE':>7s} "
          f"{'eqw_2':>7s} {'best_2':>7s}  flag")
    pairs, dups = [], []
    for i, j in itertools.combinations(range(K), 2):
        r_ear = float(np.mean([pearson(E[i, e].ravel(), E[j, e].ravel()) for e in range(ne)]))
        r_lm = float(np.mean([pearson(E[i, :, l].ravel(), E[j, :, l].ravel()) for l in range(NL)]))
        r_mle = pearson(mle[names[i]], mle[names[j]])
        e2 = float(per_ear_mle(0.5 * (P[i] + P[j]), GT).mean())
        b2 = min(mle[names[i]].mean(), mle[names[j]].mean())
        dup = Rm[i, j] > c["dup_r"] or r_mle > c["dup_r"]
        flag = "NEAR-DUPLICATE" if dup else ""
        print(f"{sh(names[i])+' + '+sh(names[j]):26s} {Rm[i, j]:8.3f} {r_ear:8.3f} {r_lm:8.3f} "
              f"{r_mle:7.3f} {e2:7.4f} {b2:7.4f}  {flag}")
        pc = {nm: round(pearson(E[i, :, lo:hi + 1].ravel(), E[j, :, lo:hi + 1].ravel()), 4)
              for lo, hi, nm in CONT}
        pairs.append(dict(a=names[i], b=names[j], r_signed_all=round(Rm[i, j], 4),
                          r_within_ear=round(r_ear, 4), r_within_landmark=round(r_lm, 4),
                          r_per_ear_MLE=round(r_mle, 4), r_signed_per_contour=pc,
                          equal_weight_pair_MLE_mm=round(e2, 4),
                          best_of_pair_MLE_mm=round(float(b2), 4),
                          pair_gain_mm=round(float(e2 - b2), 4), near_duplicate=bool(dup)))
        if dup:
            dups.append((names[i], names[j]))
    off = Rm[~np.eye(K, dtype=bool)]
    print(f"\nmean off-diagonal r_signed {off.mean():.3f} (min {off.min():.3f}, "
          f"max {off.max():.3f}) -- lower is the asset.")
    if dups:
        print(f"{len(dups)} NEAR-DUPLICATE pair(s) at r > {c['dup_r']}: "
              + ", ".join(f"{a}~{b}" for a, b in dups))
        print("  Duplicates add no diversity AND make their own weight split ill-determined:\n"
              "  expect a large across-fold sd on their weights below, and read the pair as one.")
    else:
        print(f"no pair exceeds r = {c['dup_r']}.")

    # ---------- ensembles ----------
    Peq = P.mean(0)
    mle_eq = per_ear_mle(Peq, GT)
    Pn, Wn, drows = nested_oof(P, GT, folds, subj, c)
    mle_n = per_ear_mle(Pn, GT)
    Wl, diagl = solve_weights(P, GT, np.arange(ne), c)     # in-sample == LEAKY
    mle_l = per_ear_mle(apply_weights(P, Wl, np.arange(ne)), GT)

    print("===== per-contour weights =====")
    for ci, (lo, hi, nm) in enumerate(CONT):
        print(f"{nm:15s} rows {diagl[ci]['rows']:6d}  cond(A) {diagl[ci]['cond_A']:9.1f}  "
              f"sum(w) {Wl[:, ci].sum():.4f}")
        for i, n in enumerate(names):
            print(f"    {sh(n):12s} nested {Wn[:, i, ci].mean():7.4f} "
                  f"+-{Wn[:, i, ci].std(ddof=1):.4f} (per fold "
                  + " ".join(f"{Wn[k, i, ci]:.3f}" for k in range(NFOLD))
                  + f")   leaky {Wl[i, ci]:7.4f}")

    print(f"\n===== pooled OOF over all {ne} ears =====")
    for n in names:
        print(f"  {sh(n):26s} {mle[n].mean():8.4f}")
    print(f"  {'equal weight (1/K)':26s} {mle_eq.mean():8.4f}  "
          f"({mle_eq.mean()-mle[best].mean():+.4f} vs best single)")
    print(f"  {'NESTED-OOF fitted':26s} {mle_n.mean():8.4f}  "
          f"({mle_n.mean()-mle[best].mean():+.4f} vs best single, "
          f"{mle_n.mean()-mle_eq.mean():+.4f} vs equal weight)   <-- the result")
    print(f"  {'LEAKY in-sample fit':26s} {mle_l.mean():8.4f}  "
          f"NOT A RESULT: weights fitted on the same {ne} ears they are scored on.")
    print(f"  optimism the nesting removes: {mle_n.mean()-mle_l.mean():+.4f} mm "
          f"(nested - leaky). Anything below this size, an in-sample ensembler invents.")

    print(f"\n===== paired per-subject bootstrap ({c['nb']} draws, {len(np.unique(subj))} "
          f"subjects) =====")
    cmps = {}
    for lbl, d in (("nested_vs_best_single", mle_n - mle[best]),
                   ("nested_vs_equal_weight", mle_n - mle_eq),
                   ("equal_weight_vs_best_single", mle_eq - mle[best])):
        m, lo, hi, pn = boot(d, subj, c["nb"], c["boot_seed"])
        v = "ADOPT" if hi < 0 else "REJECT" if lo > 0 else "INDISTINGUISHABLE"
        print(f"  {lbl:28s} {m:+.4f} mm  CI [{lo:+.4f}, {hi:+.4f}]  P(<0)={pn:.3f} -> {v}")
        cmps[lbl] = dict(delta_mm=round(m, 4), ci95=[round(lo, 4), round(hi, 4)],
                         p_negative=round(pn, 4), verdict=v)
    print(f"  reference for 'best single' is {best}. The interval covers SUBJECT sampling "
          f"only;\n  fold sd 0.0503 and seed sd 0.0165 mm are separate and are not in it.")

    print(f"\n{'model':22s} " + " ".join(f"{k:>10s}" for k in ("tangent_t", "across_b", "normal_n")))
    dirs = {}
    for lbl, Q in [(n, P[i]) for i, n in enumerate(names)] + \
                  [("equal_weight", Peq), ("nested_ensemble", Pn)]:
        dirs[lbl] = directional(Q, GT, T, B, N)
        print(f"{lbl[:21]:22s} " + " ".join(f"{dirs[lbl][k]:10.4f}" for k in dirs[lbl]))
    print("Along-contour (tangent_t) is 77% of the error energy. A gain that is not there is\n"
          "not a gain against the dominant term.")

    res = dict(
        config={k: v for k, v in c.items()},
        n_ears=int(ne), n_subjects=int(len(np.unique(subj))), fold_check=note,
        members=meta, best_single_model=best,
        diversity=dict(r_signed_matrix={names[i]: {names[j]: round(float(Rm[i, j]), 4)
                                                   for j in range(K)} for i in range(K)},
                       pairs=pairs, dup_threshold=c["dup_r"],
                       near_duplicate_pairs=[list(p) for p in dups],
                       mean_off_diagonal_r=round(float(off.mean()), 4)),
        weights=dict(contours=[nm for _, _, nm in CONT],
                     nested_per_outer_fold=[{names[i]: [round(float(Wn[k, i, ci]), 6)
                                                        for ci in range(len(CONT))]
                                             for i in range(K)} for k in range(NFOLD)],
                     nested_mean={names[i]: [round(float(Wn[:, i, ci].mean()), 6)
                                             for ci in range(len(CONT))] for i in range(K)},
                     nested_sd={names[i]: [round(float(Wn[:, i, ci].std(ddof=1)), 6)
                                           for ci in range(len(CONT))] for i in range(K)},
                     in_sample_LEAKY={names[i]: [round(float(Wl[i, ci]), 6)
                                                 for ci in range(len(CONT))] for i in range(K)},
                     solver_diagnostics_in_sample=diagl),
        nested_disjointness=drows,
        results=dict(
            per_model_pooled_OOF_mm={n: round(float(mle[n].mean()), 4) for n in names},
            equal_weight_MLE_mm=round(float(mle_eq.mean()), 4),
            nested_fitted_MLE_mm=round(float(mle_n.mean()), 4),
            LEAKY_in_sample_MLE_mm=round(float(mle_l.mean()), 4),
            optimism_mm=round(float(mle_n.mean() - mle_l.mean()), 4),
            per_contour_MLE_mm=dict(equal_weight=per_contour(Peq, GT),
                                    nested=per_contour(Pn, GT)),
            directional_rmse_mm=dirs),
        comparisons=cmps, n_bootstrap=c["nb"],
        caveats=[
            "STACKING, NOT FULL NESTING: the weights for outer fold k are fitted only on "
            "other folds' ears (asserted), but those ears' OOF predictions come from base "
            "models whose training folds include fold k. Full rigour needs the base models "
            "retrained inside each outer fold. The residual optimism from that is NOT "
            "quantified here; optimism_mm quantifies only the in-sample weight fit.",
            "LEAKY_in_sample_MLE_mm is reported to size the optimism and is not a result.",
            "Plain NNLS has no intercept and the coordinates are WORLD, so a non-simplex "
            "solution is not frame-invariant; sum(w) per contour is reported. SIMPLEX=1 "
            "removes this.",
            "A weighted average of landmarks is off-surface and does not change "
            "correspondence: it cancels only the INDEPENDENT part of the members' "
            "along-contour phase error. Check tangent_t, not the total.",
            ("PRED=raw: raw-network OOF, NOT comparable to the shipped 1.273 / 1.3144 mm "
             "full-pipeline numbers (no surface projection, no dense-SSM blend)."
             if c["pred"] == "raw" else
             "PRED=full: train_family.py full-pipeline predictions.")])

    if c["out"]:
        os.makedirs(os.path.dirname(c["out"]) or ".", exist_ok=True)
        json.dump(res, open(c["out"], "w"), indent=1)
        print(f"\nwrote {c['out']}")
    if c["ens_npy"]:
        os.makedirs(c["work"], exist_ok=True)
        np.save(f"{c['work']}/{c['ens_npy']}", Pn)
        print(f"wrote {c['work']}/{c['ens_npy']}  {Pn.shape}  (nested-OOF ensemble, world frame)")
    return res


# ------------------------------------------------------------------ smoke test
# There is no network here, so constraint 5's "forward AND backward" is read as: the
# BACKWARD direction is the weight SOLVE (data -> weights) and the FORWARD direction is
# applying them (weights -> landmarks). Both are exercised on synthetic prediction sets
# whose optimal weights are known in closed form, so the solver is checked against an
# answer rather than against itself.
def write_bundle(d, GT, preds, subj, folds, seed=0, full=None):
    """Synthesise the on-disk artefacts the real loader reads: screen_<tag>_s<seed>_f<k>.*
    (+ _full.npy for tags in `full`) plus an ortho_feats.npz stand-in with an orthonormal
    (t,b,n) triad per landmark."""
    os.makedirs(d, exist_ok=True)
    Q = np.linalg.qr(np.random.RandomState(11).randn(len(GT), NL, 3, 3))[0]
    np.savez(f"{d}/ortho_feats.npz", gt=GT, t=Q[..., 0], b=Q[..., 1], n=Q[..., 2],
             subj=subj, fold=folds)
    mle = lambda Q_, ev: round(float(np.linalg.norm(Q_[ev] - GT[ev], axis=2).mean()), 4)
    for tag, Pm in preds.items():
        for k in range(NFOLD):
            ev = np.where(folds == k)[0]
            b = f"{d}/screen_{tag}_s{seed}_f{k}"
            rec = {"variant": tag, "seed": seed, "fold": k, "ordered_MLE_mm": mle(Pm, ev),
                   "val_ear_index": [int(i) for i in ev]}
            if full and tag in full:
                rec["ordered_MLE_full_mm"] = mle(full[tag], ev)
                np.save(b + "_full.npy", full[tag][ev])
            json.dump(rec, open(b + ".json", "w"))
            np.save(b + ".npy", Pm[ev])


def run(d, models, quiet=True, **env):
    """One full main() on synthetic artefacts. quiet=False shows the report layout once."""
    e = dict(WORK=d, FEATS=f"{d}/ortho_feats.npz", MODELS=models, SEEDS="0", NB="200",
             SIMPLEX="0", RIDGE="0.0", SEED_MODE="ensemble", PRED="raw",
             ENS_NPY="", OUT=f"{d}/ens.json")
    e.update(env)
    keep = {k: os.environ.get(k) for k in e}
    os.environ.update(e)
    sink, real = io.StringIO(), sys.stdout
    if quiet:
        sys.stdout = sink
    try:
        return main()
    finally:
        sys.stdout = real
        for k, v in keep.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def wmat(res, key, names):
    return np.array([res["weights"][key][n] for n in names])          # (K,4)


def smoke():
    d = os.environ.get("SMOKE_DIR", os.path.join(tempfile.gettempdir(), "ensemble_oof_smoke"))
    ne = 40
    subj, folds = frozen_folds(ne)
    rng = np.random.RandomState(0)
    G = rng.randn(ne, NL, 3) * 8 + np.array([12.0, -5.0, 30.0])

    # ---- TEST 1: exactly anti-correlated members, ratio DIFFERENT PER CONTOUR ----
    # mA = G + e, mB = G - lam*e  =>  the unique zero-residual NNLS solution per contour is
    # w_A = lam/(1+lam), w_B = 1/(1+lam), and a contour-blind solver cannot produce it.
    print("=" * 78)
    print("SMOKE 1/5 -- anti-correlated members, per-contour analytic optimum; RIDGE")
    print("(this case prints the FULL report so the layout is visible; 2-5 are quiet)")
    e1 = rng.randn(ne, NL, 3)
    lam = np.ones((1, NL, 1))
    for ci, (lo, hi, nm) in enumerate(CONT):
        lam[0, lo:hi + 1, 0] = ci + 1
    write_bundle(d, G, {"mA": G + e1, "mB": G - lam * e1}, subj, folds)
    r1 = run(d, "mA,mB", quiet=False)
    L = np.array([float(lam[0, lo, 0]) for lo, hi, nm in CONT])
    want = np.stack([L / (1 + L), 1 / (1 + L)])                       # (2,4)
    Wl, Wn = wmat(r1, "in_sample_LEAKY", ["mA", "mB"]), np.array(
        [[r1["weights"]["nested_per_outer_fold"][k][n] for n in ("mA", "mB")]
         for k in range(NFOLD)])
    print(f"  analytic {np.round(want, 4).tolist()}")
    print(f"  leaky    {np.round(Wl, 4).tolist()}")
    print(f"  nested   max|W_k - analytic| over the 5 outer folds "
          f"{np.abs(Wn - want[None]).max():.2e}")
    assert np.allclose(Wl, want, atol=1e-5), Wl
    assert np.allclose(Wn, want[None], atol=1e-5), Wn
    assert r1["results"]["nested_fitted_MLE_mm"] < 1e-3, r1["results"]
    assert r1["results"]["equal_weight_MLE_mm"] > 0.5, "equal weight should be far from optimal"
    print(f"  nested MLE {r1['results']['nested_fitted_MLE_mm']:.4f} (exact solution exists) "
          f"vs equal-weight {r1['results']['equal_weight_MLE_mm']:.4f}")
    rr = run(d, "mA,mB", RIDGE="1.0")                  # ridge must pull toward 1/K and cost
    Wr = wmat(rr, "in_sample_LEAKY", ["mA", "mB"])
    assert np.abs(Wr - 0.5).max() < 0.5 * np.abs(Wl - 0.5).max(), Wr
    assert rr["results"]["nested_fitted_MLE_mm"] > r1["results"]["nested_fitted_MLE_mm"]
    print(f"  RIDGE=1.0 pulls the sup._antihelix weights {np.round(Wl[:, 3], 3).tolist()} -> "
          f"{np.round(Wr[:, 3], 3).tolist()} and costs "
          f"{rr['results']['nested_fitted_MLE_mm']:.4f} mm, as it should")

    print("\nSMOKE 2/5 -- same data with SIMPLEX=1 (weights already sum to 1)")
    r2 = run(d, "mA,mB", SIMPLEX="1")
    W2 = wmat(r2, "in_sample_LEAKY", ["mA", "mB"])
    assert np.allclose(W2, want, atol=1e-5), W2
    assert all(abs(sum(W2[:, ci]) - 1) < 1e-6 for ci in range(4))
    print(f"  simplex weights match the analytic optimum, sum(w) = "
          f"{[round(float(W2[:, ci].sum()), 9) for ci in range(4)]}")

    # ---- TEST 2: shared irreducible error -> sum(w) < 1 by a known amount ----
    # Per contour, orthogonalise a shared error `s` and a differential error `f` against the
    # coordinate block g. Members G+s+f and G+s-f give residual (S-1)g + S*s + D*f with
    # S = sum(w), D = w_A - w_B; orthogonality makes the optimum exactly
    # D = 0, S = |g|^2 / (|g|^2 + |s|^2), i.e. w_A = w_B = S/2 -- the frame-dependent
    # shrinkage that caveat 2 in the docstring is about, as a number.
    print("\nSMOKE 3/5 -- shared error: analytic sum(w) < 1 (the no-intercept shrinkage)")
    s, f = rng.randn(ne, NL, 3), rng.randn(ne, NL, 3)
    S_want = []
    for ci, (lo, hi, nm) in enumerate(CONT):
        g = G[:, lo:hi + 1].ravel()
        a = s[:, lo:hi + 1].ravel().copy(); a -= g * (a @ g) / (g @ g)
        b = f[:, lo:hi + 1].ravel().copy(); b -= g * (b @ g) / (g @ g); b -= a * (b @ a) / (a @ a)
        a *= [0.30, 0.20, 0.40, 0.10][ci] * np.linalg.norm(g) / np.linalg.norm(a)
        b *= 0.10 * np.linalg.norm(g) / np.linalg.norm(b)
        s[:, lo:hi + 1] = a.reshape(ne, hi - lo + 1, 3)
        f[:, lo:hi + 1] = b.reshape(ne, hi - lo + 1, 3)
        S_want.append((g @ g) / (g @ g + a @ a))
    write_bundle(d, G, {"mA": G + s + f, "mB": G + s - f}, subj, folds)
    r3 = run(d, "mA,mB")
    W3 = wmat(r3, "in_sample_LEAKY", ["mA", "mB"])
    want3 = np.stack([np.array(S_want) / 2, np.array(S_want) / 2])
    print(f"  analytic sum(w) {[round(float(x), 6) for x in S_want]}")
    print(f"  fitted   sum(w) {[round(float(W3[:, ci].sum()), 6) for ci in range(4)]}")
    assert np.allclose(W3, want3, atol=1e-5), (W3, want3)

    # ---- TEST 3: fold-dependent optimum + a near-duplicate member ----
    # mA is 20x better than mB on folds 0-1 and 20x worse on folds 2-4, so the optimal
    # weights genuinely differ per fold: an in-sample fit must look better than the nested
    # one. mC = mA + tiny noise must be flagged NEAR-DUPLICATE.
    print("\nSMOKE 4/5 -- fold-dependent optimum: optimism must be positive; dup flagging")
    u, v = rng.randn(ne, NL, 3), rng.randn(ne, NL, 3)
    fav = np.isin(folds, [0, 1])[:, None, None]
    PA = G + np.where(fav, 0.1, 2.0) * u
    PB = G + np.where(fav, 2.0, 0.1) * v
    write_bundle(d, G, {"mA": PA, "mB": PB, "mC": PA + 0.002 * rng.randn(ne, NL, 3)},
                 subj, folds)
    r4 = run(d, "mA,mB,mC")
    R = r4["results"]
    print(f"  nested {R['nested_fitted_MLE_mm']:.4f}  LEAKY {R['LEAKY_in_sample_MLE_mm']:.4f}  "
          f"optimism {R['optimism_mm']:+.4f} mm")
    assert R["optimism_mm"] > 0.01, f"nested is not paying for the fold-dependence: {R}"
    assert ["mA", "mC"] in r4["diversity"]["near_duplicate_pairs"], \
        r4["diversity"]["near_duplicate_pairs"]
    sd = np.array(r4["weights"]["nested_sd"]["mA"])
    print(f"  mA/mC flagged near-duplicate; across-fold sd of mA's weight {np.round(sd, 3).tolist()}")
    r5 = run(d, "mA,mB,mC", ENS_NPY="ens.npy")
    Pn = np.load(f"{d}/ens.npy")
    nk = len(r5["results"]["per_model_pooled_OOF_mm"])       # K as the run actually saw it
    print(f"  ensemble prediction set {Pn.shape}; a 2-ear slice is {Pn[:2].shape}; "
          f"fitted parameters {nk * len(CONT)} ({nk} models x {len(CONT)} contours)")
    assert Pn.shape == (ne, NL, 3) and Pn[:2].shape == (2, NL, 3), Pn.shape
    assert abs(np.linalg.norm(Pn - G, axis=2).mean(1).mean()
               - r5["results"]["nested_fitted_MLE_mm"]) < 1e-4, "saved .npy != reported MLE"
    for k in ("caveats", "nested_disjointness", "weights", "diversity", "results"):
        assert k in r5, k

    # ---- TEST 5: the two loader paths a real run depends on and 1-4 never touch ----
    print("\nSMOKE 5/5 -- loader: SEED_MODE ensemble vs separate, and PRED=full")
    PAf, PBf = G + 0.25 * u, G + 0.25 * v                 # 'full-pipeline' variants
    write_bundle(d, G, {"mA": PA, "mB": PB}, subj, folds, seed=0, full={"mA": PAf, "mB": PBf})
    # seed 1's error must be INDEPENDENT of seed 0's, or averaging the seeds gains nothing
    # and the per-fold-column check below has no teeth (collinear errors make the MLE of the
    # mean equal to the mean of the MLEs exactly).
    write_bundle(d, G, {"mA": G + 1.3 * rng.randn(ne, NL, 3),
                        "mB": G + 1.3 * rng.randn(ne, NL, 3)}, subj, folds, seed=1)
    sep = run(d, "mA,mB", SEEDS="0,1", SEED_MODE="separate")
    ens = run(d, "mA,mB", SEEDS="0,1", SEED_MODE="ensemble")
    assert sorted(sep["members"]) == ["mA#s0", "mA#s1", "mB#s0", "mB#s1"], list(sep["members"])
    assert sorted(ens["members"]) == ["mA", "mB"] and ens["members"]["mA"]["seeds"] == [0, 1]
    fl = run(d, "mA,mB", SEEDS="0", PRED="full")
    raw = run(d, "mA,mB", SEEDS="0", PRED="raw")
    assert fl["results"]["per_model_pooled_OOF_mm"]["mA"] < \
        raw["results"]["per_model_pooled_OOF_mm"]["mA"], "PRED=full read the raw .npy"
    # a seed-mean member's per-fold column must be the per-fold MLE OF THE MEAN prediction,
    # not the mean of the seeds' own MLEs -- the two differ by the seed-ensemble gain, which
    # would leave the row contradicting its own pooled column. Folds are equal-sized here,
    # so the plain mean of the column has to equal the pooled number.
    pf = [ens["members"]["mA"]["per_fold_MLE"][f] for f in range(NFOLD)]
    pooled = ens["members"]["mA"]["pooled_OOF_MLE_mm"]
    naive = float(np.mean([0.5 * (sep["members"]["mA#s0"]["per_fold_MLE"][f]
                                  + sep["members"]["mA#s1"]["per_fold_MLE"][f])
                           for f in range(NFOLD)]))
    assert abs(float(np.mean(pf)) - pooled) < 2e-4, (pf, pooled)
    assert naive > pooled + 0.01, f"seed errors are collinear -- this check proves nothing: {naive}"
    print(f"  separate -> {sorted(sep['members'])}, ensemble -> {sorted(ens['members'])} "
          f"(seeds {ens['members']['mA']['seeds']})")
    print(f"  seed-mean per-fold column means {np.mean(pf):.4f} = pooled {pooled:.4f}; "
          f"averaging the seeds' own MLEs instead would say {naive:.4f}")
    print(f"  PRED=raw mA {raw['results']['per_model_pooled_OOF_mm']['mA']:.4f} vs PRED=full "
          f"{fl['results']['per_model_pooled_OOF_mm']['mA']:.4f} -- distinct artefacts read")
    print("SMOKE PASS")
    print("=" * 78)


if __name__ == "__main__":
    main() if os.environ.get("MODELS") else smoke()
