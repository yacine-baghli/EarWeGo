"""
CORRESPONDENCE ORACLES v2 — corrected definitions.

AUDIT FIX. The v1 "perfect per-point" oracle measured the distance from each prediction
to the GT POLYLINE. That lets ground truth supply the *geometry* the corrected point
lies on, so it is NOT a valid geometry-vs-correspondence split (it is a lower bound that
mixes both). Here every candidate point lies on the PREDICTED polyline; ground truth is
used only to choose parameters.

EXACT RULES (identical for all oracles):
  * geometry      : the predicted contour polyline P_0..P_{n-1} (piecewise linear).
  * parameter     : arc length s along that polyline, s_k^pred = cumulative chord length.
  * interpolation : piecewise-linear between consecutive predicted points.
  * extrapolation : linear continuation of the first/last segment direction when the
                    parameter falls outside [0, L]. Range limited to [-PAD, L+PAD].
  * monotonicity  : enforced (non-decreasing). Scalar/affine are monotone by
                    construction (a > 0); the per-point oracle is solved by dynamic
                    programming over a non-decreasing parameter sequence.
  * clipping      : parameters are clipped to the padded range; no wrap-around.

ORACLE FAMILIES (increasing freedom, all on the predicted curve):
  1 scalar    s_k = s_k^pred + b                       (1 param / contour / ear)
  2 affine    s_k = a*(s_k^pred - s_0) + s_0 + b       (2 params, a>0)
  3 monotone  any non-decreasing (t_1..t_n)            (n params, DP-optimal)
"""
import numpy as np

CONT = [(0, 24, "outer helix"), (25, 54, "concha"),
        (55, 74, "inner helix"), (75, 84, "sup. antihelix")]
PAD = 3.0        # mm of allowed extrapolation beyond each end
STEP = 0.04      # mm parameter grid for the DP oracle


def arc(P):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]


def eval_poly(P, s_query):
    """piecewise-linear evaluation, linear extrapolation outside [0,L]. VECTORISED:
    s_query may have any shape; returns s_query.shape + (3,)."""
    s = arc(P); L = s[-1]
    sq = np.clip(np.asarray(s_query, float), -PAD, L + PAD)
    j = np.clip(np.searchsorted(s, sq) - 1, 0, len(P) - 2)
    f = (sq - s[j]) / np.maximum(s[j + 1] - s[j], 1e-12)
    out = P[j] + f[..., None] * (P[j + 1] - P[j])
    d0 = P[1] - P[0]; d0 = d0 / max(np.linalg.norm(d0), 1e-9)
    d1 = P[-1] - P[-2]; d1 = d1 / max(np.linalg.norm(d1), 1e-9)
    m0 = sq <= 0.0
    if m0.any():
        out[m0] = P[0] + sq[m0][..., None] * d0
    m1 = sq >= L
    if m1.any():
        out[m1] = P[-1] + (sq[m1] - L)[..., None] * d1
    return out


def err(P, G, s_query):
    return np.linalg.norm(eval_poly(P, s_query) - G, axis=-1)


def oracle_scalar(P, G):
    s = arc(P)
    grid = np.arange(-PAD, PAD + 1e-9, 0.01)                 # (B,)
    E = err(P, G[None, :, :], s[None, :] + grid[:, None]).mean(1)   # (B,)
    j = int(np.argmin(E))
    return float(E[j]), float(grid[j])


def oracle_affine(P, G):
    s = arc(P); s0 = s[0]
    A = np.arange(0.80, 1.2001, 0.0025)
    B = np.arange(-PAD, PAD + 1e-9, 0.02)
    # query params for every (a,b) pair at once: (nA,nB,n)
    sa = s0 + A[:, None] * (s - s0)[None, :]                 # (nA,n)
    Q = sa[:, None, :] + B[None, :, None]                    # (nA,nB,n)
    E = err(P, G[None, None, :, :], Q).mean(-1)              # (nA,nB)
    ia, ib = np.unravel_index(int(np.argmin(E)), E.shape)
    return float(E[ia, ib]), float(A[ia]), float(B[ib])


def oracle_monotone(P, G):
    """DP-optimal NON-DECREASING parameter sequence on the predicted polyline"""
    s = arc(P); L = s[-1]
    grid = np.arange(-PAD, L + PAD + 1e-9, STEP)
    Q = eval_poly(P, grid)                                  # (M,3) candidate points
    C = np.linalg.norm(Q[None, :, :] - G[:, None, :], axis=2)   # (n,M) cost
    dp = C[0].copy()
    for k in range(1, len(G)):
        dp = C[k] + np.minimum.accumulate(dp)               # non-decreasing constraint
    return dp.min() / len(G)


def run(P_all, G_all, tag):
    N = len(G_all)
    res = {}
    for lo, hi, name in CONT:
        n = hi - lo + 1
        b_, s_, a_, m_ = [], [], [], []
        ends, mids = [], []
        pars = []
        for i in range(N):
            P, G = P_all[i, lo:hi + 1], G_all[i, lo:hi + 1]
            e = np.linalg.norm(P - G, axis=1)
            b_.append(e.mean()); ends.append(0.5 * (e[0] + e[-1])); mids.append(e[1:-1].mean())
            v1, b1 = oracle_scalar(P, G); s_.append(v1)
            v2, a2, b2 = oracle_affine(P, G); a_.append(v2); pars.append((a2, b2))
            m_.append(oracle_monotone(P, G))
        res[name] = dict(n=n, base=np.mean(b_), scalar=np.mean(s_), affine=np.mean(a_),
                         mono=np.mean(m_), endpoint=np.mean(ends), interior=np.mean(mids),
                         params=np.array(pars))
        print(f"  {name:16s} done", flush=True)
    w = np.array([res[c[2]]["n"] for c in CONT])
    print(f"\n===== {tag} =====")
    print(f"{'contour':16s} {'n':>3s} {'baseline':>9s} {'scalar':>8s} {'affine':>8s} "
          f"{'monotone':>9s} | {'endpts':>7s} {'interior':>8s} | {'aff gap':>8s} {'mono gap':>8s}")
    for lo, hi, nm in CONT:
        r = res[nm]
        print(f"{nm:16s} {r['n']:3d} {r['base']:9.4f} {r['scalar']:8.4f} {r['affine']:8.4f} "
              f"{r['mono']:9.4f} | {r['endpoint']:7.4f} {r['interior']:8.4f} | "
              f"{r['base']-r['affine']:8.4f} {r['base']-r['mono']:8.4f}")
    agg = lambda k: np.average([res[c[2]][k] for c in CONT], weights=w)
    print(f"{'ALL 85':16s} {int(w.sum()):3d} {agg('base'):9.4f} {agg('scalar'):8.4f} "
          f"{agg('affine'):8.4f} {agg('mono'):9.4f} | {agg('endpoint'):7.4f} "
          f"{agg('interior'):8.4f} | {agg('base')-agg('affine'):8.4f} "
          f"{agg('base')-agg('mono'):8.4f}")
    return res


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "scratch/cv_oof.npz"
    z = np.load(src)
    key = "pred" if "pred" in z.files else "oof"
    P_all = z[key].astype(np.float64); G_all = z["gt"].astype(np.float64)
    if "have" in z.files:
        m = z["have"]; P_all, G_all = P_all[m], G_all[m]
    print(f"{src}: {len(G_all)} ears | overall MLE "
          f"{np.linalg.norm(P_all-G_all,axis=2).mean():.4f}")
    res = run(P_all, G_all, src)
    np.savez("scratch/oracles_v2_params.npz",
             **{f"{nm.replace(' ','_')}": res[nm]["params"] for _, _, nm in CONT})
