"""
Is the per-contour parametrisation correction PREDICTABLE without ground truth?
(brief v2 §10.2 acceptance test)

Oracle (OOF, 340 ears): baseline 1.403 -> affine warp per contour 0.911 mm.
That oracle uses the GT. Here we ask whether the SAME correction can be regressed
from the model's OWN prediction (no GT at inference), evaluated with subject-grouped
5-fold CV so the regressor never sees the subject it corrects.

If predicted corrections recover a good share of the oracle gain, this is the lever.
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import RidgeCV

CONT = [(0, 24, "outer helix"), (25, 54, "concha"), (55, 74, "inner helix"),
        (75, 84, "sup. antihelix")]
z = np.load("scratch/cv_oof.npz")
P_all, G_all = z["oof"].astype(np.float64), z["gt"].astype(np.float64)
N = len(G_all)


def arc(P):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]


def sample_at(P, s_new):
    s = arc(P); out = np.empty((len(s_new), 3))
    for k, t in enumerate(s_new):
        if t <= s[0]:
            d = P[1] - P[0]; d = d / max(np.linalg.norm(d), 1e-9); out[k] = P[0] + d * (t - s[0])
        elif t >= s[-1]:
            d = P[-1] - P[-2]; d = d / max(np.linalg.norm(d), 1e-9); out[k] = P[-1] + d * (t - s[-1])
        else:
            j = np.searchsorted(s, t) - 1
            f = (t - s[j]) / max(s[j + 1] - s[j], 1e-12)
            out[k] = P[j] + f * (P[j + 1] - P[j])
    return out


def warp(P, a, b):
    """affine reparametrisation: s -> a*(s-s0)+s0 + b, then resample"""
    s = arc(P)
    return sample_at(P, s[0] + a * (s - s[0]) + b)


def err_of(P, G, a, b):
    return np.linalg.norm(warp(P, a, b) - G, axis=1).mean()


# ---- 1. oracle affine params per ear per contour (fast local optimiser) ------
print("computing oracle affine params ...", flush=True)
params = np.zeros((N, len(CONT), 2))
base_e = np.zeros((N, len(CONT)))
orac_e = np.zeros((N, len(CONT)))
for i in range(N):
    for c, (lo, hi, _) in enumerate(CONT):
        Pp, Gg = P_all[i, lo:hi + 1], G_all[i, lo:hi + 1]
        base_e[i, c] = err_of(Pp, Gg, 1.0, 0.0)
        best = (base_e[i, c], 1.0, 0.0)
        for a0, b0 in ((1.0, 0.0), (1.0, 0.8), (1.0, -0.8), (0.95, 0.0), (1.05, 0.0)):
            r = minimize(lambda x: err_of(Pp, Gg, x[0], x[1]), [a0, b0],
                         method="Nelder-Mead",
                         options=dict(xatol=1e-3, fatol=1e-4, maxiter=200))
            if r.fun < best[0]:
                best = (r.fun, r.x[0], r.x[1])
        orac_e[i, c] = best[0]; params[i, c] = best[1:]
    if (i + 1) % 60 == 0:
        print(f"  {i+1}/{N}", flush=True)
np.savez("scratch/phase_params.npz", params=params, base_e=base_e, orac_e=orac_e)

nw = np.array([hi - lo + 1 for lo, hi, _ in CONT])
print(f"\nbaseline {np.average(base_e.mean(0), weights=nw):.4f}  "
      f"oracle affine {np.average(orac_e.mean(0), weights=nw):.4f}")

# ---- 2. features from the PREDICTION only (no GT) ---------------------------
def features(P):
    f = []
    cen = P.mean(0); sc = np.linalg.norm(P - cen) / np.sqrt(len(P))
    Q = (P - cen) / sc                                   # pose/scale-normalised shape
    f.append(Q.ravel())
    for lo, hi, _ in CONT:
        seg = P[lo:hi + 1]
        L = arc(seg)[-1]
        g = np.linalg.norm(np.diff(seg, axis=0), axis=1)
        f.append([L, L / sc, g.std() / max(g.mean(), 1e-9), sc])
    return np.concatenate([np.atleast_1d(x).ravel() for x in f])


X = np.stack([features(P_all[i]) for i in range(N)])
subj = np.arange(N) // 2
print(f"features: {X.shape}")

# ---- 3. subject-grouped 5-fold CV: predict the params, apply, measure -------
rs = np.random.RandomState(7)
folds = np.array_split(rs.permutation(np.unique(subj)), 5)
pred_par = np.zeros_like(params)
for f in folds:
    te = np.isin(subj, f); tr = ~te
    for c in range(len(CONT)):
        for j in range(2):
            m = RidgeCV(alphas=np.logspace(-1, 4, 12)).fit(X[tr], params[tr, c, j])
            pred_par[te, c, j] = m.predict(X[te])

applied = np.zeros((N, len(CONT)))
for i in range(N):
    for c, (lo, hi, _) in enumerate(CONT):
        a, b = pred_par[i, c]
        a = float(np.clip(a, 0.85, 1.15)); b = float(np.clip(b, -3, 3))
        applied[i, c] = err_of(P_all[i, lo:hi + 1], G_all[i, lo:hi + 1], a, b)

print(f"\n{'contour':16s} {'baseline':>9s} {'oracle':>8s} {'PREDICTED':>10s} "
      f"{'gain':>7s} {'% of oracle':>12s}")
for c, (lo, hi, nm) in enumerate(CONT):
    b, o, p = base_e[:, c].mean(), orac_e[:, c].mean(), applied[:, c].mean()
    print(f"{nm:16s} {b:9.4f} {o:8.4f} {p:10.4f} {b-p:+7.4f} "
          f"{100*(b-p)/max(b-o,1e-9):11.0f}%")
B = np.average(base_e.mean(0), weights=nw); O = np.average(orac_e.mean(0), weights=nw)
A = np.average(applied.mean(0), weights=nw)
print(f"{'ALL 85':16s} {B:9.4f} {O:8.4f} {A:10.4f} {B-A:+7.4f} "
      f"{100*(B-A)/max(B-O,1e-9):11.0f}%")
# correlation between oracle and predicted params
for c, (lo, hi, nm) in enumerate(CONT):
    r_a = np.corrcoef(params[:, c, 0], pred_par[:, c, 0])[0, 1]
    r_b = np.corrcoef(params[:, c, 1], pred_par[:, c, 1])[0, 1]
    print(f"  {nm:16s} corr(oracle,pred): stretch {r_a:+.3f}  offset {r_b:+.3f}")
