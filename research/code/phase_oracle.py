"""
PRIORITY-3 EXPERIMENT (brief v2 §10.2 / §11): how much of the error is CORRESPONDENCE
(parametrisation / phase along the contour) rather than GEOMETRY?

Hypothesis: the dominant residual is a coherent shift of the whole contour along its
own arc length. If true, a per-contour scalar (4 numbers per ear) recovers a large
fraction -- far easier to predict than 85x3 coordinates.

Oracles, in increasing power (all on OOF predictions, 340 ears):
  0 baseline
  1 d_perp        distance from each prediction to the GT contour POLYLINE
                  = the floor if the parametrisation were PERFECT per point
  2 scalar shift  one arc-length offset per contour per ear   (1 param)
  3 affine warp   offset + stretch of the parameter           (2 params)
Also reports the spread of the optimal shift, which decides whether it is a fixed bias
(free to fix) or per-ear information that must be predicted.
"""
import numpy as np

CONT = [(0, 24, "outer helix"), (25, 54, "concha"), (55, 74, "inner helix"),
        (75, 84, "sup. antihelix")]
z = np.load("scratch/cv_oof.npz")
P_all, G_all = z["oof"].astype(np.float64), z["gt"].astype(np.float64)
N = len(G_all)


def arc(P):
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    return d


def sample_at(P, s_new):
    """sample the polyline P at arc-length positions s_new (linear extrapolation at ends)"""
    s = arc(P)
    out = np.empty((len(s_new), 3))
    for k, t in enumerate(s_new):
        if t <= s[0]:
            d = P[1] - P[0]; d /= max(np.linalg.norm(d), 1e-9)
            out[k] = P[0] + d * (t - s[0])
        elif t >= s[-1]:
            d = P[-1] - P[-2]; d /= max(np.linalg.norm(d), 1e-9)
            out[k] = P[-1] + d * (t - s[-1])
        else:
            j = np.searchsorted(s, t) - 1
            f = (t - s[j]) / max(s[j + 1] - s[j], 1e-12)
            out[k] = P[j] + f * (P[j + 1] - P[j])
    return out


def dist_to_polyline(q, C):
    """min distance from point q to the polyline through C"""
    a, b = C[:-1], C[1:]
    ab = b - a
    t = np.clip(((q - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
    proj = a + t[:, None] * ab
    return np.linalg.norm(proj - q, axis=1).min()


rows = {}
shift_stats = {}
for lo, hi, name in CONT:
    n = hi - lo + 1
    e0, e1, e2, e3, best_d, best_ab = [], [], [], [], [], []
    for i in range(N):
        Pp, Gg = P_all[i, lo:hi + 1], G_all[i, lo:hi + 1]
        e0.append(np.linalg.norm(Pp - Gg, axis=1).mean())
        # 1 perfect per-point parametrisation = distance to the GT curve
        e1.append(np.mean([dist_to_polyline(p, Gg) for p in Pp]))
        s = arc(Pp)
        # 2 single scalar arc-length shift of our own predicted curve
        grid = np.linspace(-4.0, 4.0, 161)
        errs = [np.linalg.norm(sample_at(Pp, s + d) - Gg, axis=1).mean() for d in grid]
        j = int(np.argmin(errs)); e2.append(errs[j]); best_d.append(grid[j])
        # 3 affine warp of the parameter: s -> a*s + b
        bestv, besta = 1e9, None
        for a in np.linspace(0.85, 1.15, 31):
            sc = s * a
            sc = sc - sc[0] + s[0]                       # keep origin, stretch length
            for d in np.linspace(-3.0, 3.0, 61):
                v = np.linalg.norm(sample_at(Pp, sc + d) - Gg, axis=1).mean()
                if v < bestv:
                    bestv, besta = v, (a, d)
        e3.append(bestv); best_ab.append(besta)
    rows[name] = (np.mean(e0), np.mean(e1), np.mean(e2), np.mean(e3), n)
    bd = np.array(best_d)
    shift_stats[name] = (bd.mean(), bd.std(), np.abs(bd).mean(),
                         np.array([x[0] for x in best_ab]).mean())

print(f"{'contour':16s} {'n':>3s} {'baseline':>9s} {'perfect param':>14s} "
      f"{'scalar shift':>13s} {'affine warp':>12s}")
tot = np.zeros(4); cnt = 0
for lo, hi, name in CONT:
    b, p, s2, s3, n = rows[name]
    print(f"{name:16s} {n:3d} {b:9.3f} {p:14.3f} {s2:13.3f} {s3:12.3f}")
    tot += np.array([b, p, s2, s3]) * n; cnt += n
print(f"{'ALL 85':16s} {cnt:3d} {tot[0]/cnt:9.3f} {tot[1]/cnt:14.3f} "
      f"{tot[2]/cnt:13.3f} {tot[3]/cnt:12.3f}")

print("\noptimal scalar shift per contour (mm of arc length):")
for lo, hi, name in CONT:
    m, sd, am, a = shift_stats[name]
    print(f"  {name:16s} mean {m:+.3f}  sd {sd:.3f}  mean|shift| {am:.3f}  "
          f"mean stretch {a:.3f}")
print("\ninterpretation:")
print("  perfect param  = the floor if each point could slide freely to its best spot")
print("                   on the GT curve -> the part of the error that is GEOMETRY")
print("  scalar shift   = ONE number per contour per ear (the coherent-phase hypothesis)")
print("  a large mean|shift| with sd >> 0 means it is per-ear info that must be PREDICTED;")
print("  a consistent non-zero mean would be a free systematic fix.")
