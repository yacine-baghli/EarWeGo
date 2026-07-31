"""
VALID vector-level error decomposition (replaces the invalid Pythagorean estimate).

Two separate, exact analyses:

(1) LOCAL-FRAME ENERGY SPLIT. At each landmark build an orthonormal frame from the
    contour tangent t (neighbouring predicted landmarks), the oriented mesh normal n,
    and b = normalise(n x t) (re-orthogonalised). Because the frame is orthonormal,
        <|e|^2> = <e_t^2> + <e_b^2> + <e_n^2>
    holds EXACTLY, so RMSE components and their energy shares are well defined.
    Frame orthonormality is verified numerically and reported.

(2) ORACLE DISPLACEMENT IDENTITY. With d = p_oracle - p_base and e = p - gt,
        e_base = e_oracle - d   =>   <|e_base|^2> = <|e_oracle|^2> + <|d|^2> - 2<e_oracle . d>
    Every term is measured, INCLUDING the cross-term, so no orthogonality is assumed.
    The cross-term is reported as a normalised correlation so the reader can judge it.

No "floor" is asserted here: mean-of-Euclidean (the competition metric) and RMS energies
are different functionals, and both are reported side by side.
"""
import os, sys
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import oracles_v2 as O
from src.splits import get_split
from src.dataset import Dataset

MESH = "2026 Munich Tech Arena - Datas/2026 Munich Tech Arena - Datas/mesh"
LM = "2026 Munich Tech Arena - Datas/2026 Munich Tech Arena - Datas/landmarks"
MIRROR = np.array([1., -1., 1.])
CONT = [(0, 24, "outer helix"), (25, 54, "concha"),
        (55, 74, "inner helix"), (75, 84, "sup. antihelix")]
NE = int(os.environ.get("NE", "340"))

z = np.load("scratch/oof_final.npz")
P, G = z["pred"].astype(np.float64), z["gt"].astype(np.float64)
tr = get_split("train", mesh_dir=Path(MESH)); va = get_split("val", mesh_dir=Path(MESH))
order = [(p, s) for p in tr for s in ("left", "right")] + \
        [(p, s) for p in va for s in ("left", "right")]
ds = Dataset(MESH, LM); pid2idx = {q: i for i, q in enumerate(ds.subject_ids)}


def monotone_corrected(Pc, Gc):
    """DP-optimal non-decreasing reparametrisation ON the predicted curve -> points"""
    s = O.arc(Pc); L = s[-1]
    grid = np.arange(-O.PAD, L + O.PAD + 1e-9, O.STEP)
    Q = O.eval_poly(Pc, grid)
    C = np.linalg.norm(Q[None, :, :] - Gc[:, None, :], axis=2)      # (n,M)
    n = len(Gc); M = len(grid)
    dp = np.empty((n, M)); back = np.zeros((n, M), np.int32)
    dp[0] = C[0]
    ar = np.arange(M)
    for k in range(1, n):
        prev = dp[k - 1]
        run = np.minimum.accumulate(prev)                           # prefix min values
        # prefix ARGmin, vectorised: last index attaining the running minimum
        arg = np.maximum.accumulate(np.where(prev == run, ar, -1)).astype(np.int32)
        dp[k] = C[k] + run; back[k] = arg
    j = int(np.argmin(dp[n - 1])); idx = [j]
    for k in range(n - 1, 0, -1):
        j = back[k][j]; idx.append(j)
    idx = idx[::-1]
    return Q[idx]


# ---------- gather per-landmark quantities ----------
cache = {}
E_all, T_all, B_all, N_all = [], [], [], []
orth_err = []
D_all, EO_all = [], []
for i in range(NE):
    pid, side = order[i]
    if pid not in cache:
        m = ds[pid2idx[pid]][0]
        cache[pid] = (np.asarray(m.vertices), np.asarray(m.faces))
        if len(cache) > 25:
            cache.pop(next(iter(cache)))
    V, F = cache[pid]
    if side == "right":
        V = V * MIRROR; F = F[:, ::-1]
    lo_, hi_ = P[i].min(0) - 8, P[i].max(0) + 8
    vin = np.all((V >= lo_) & (V <= hi_), axis=1)
    fm = vin[F].all(axis=1); Fs = F[fm]
    keep = np.unique(Fs) if len(Fs) else np.arange(len(V))
    remap = -np.ones(len(V), int); remap[keep] = np.arange(len(keep))
    Vc = V[keep]; Fc = remap[Fs] if len(Fs) else None
    VN = np.zeros_like(Vc)
    if Fc is not None and len(Fc):
        fn = np.cross(Vc[Fc[:, 1]] - Vc[Fc[:, 0]], Vc[Fc[:, 2]] - Vc[Fc[:, 0]])
        for c in range(3):
            np.add.at(VN, Fc[:, c], fn)
    nr = np.linalg.norm(VN, axis=1, keepdims=True)
    VN = np.where(nr > 1e-9, VN / np.maximum(nr, 1e-9), np.array([0., 0., 1.]))
    tree = cKDTree(Vc)
    for lo, hi, _ in CONT:
        Pc, Gc = P[i, lo:hi + 1], G[i, lo:hi + 1]
        e = Pc - Gc
        # tangent from neighbouring PREDICTED landmarks (central differences)
        t = np.gradient(Pc, axis=0)
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
        nn = VN[tree.query(Pc)[1]]
        # re-orthogonalise n against t, then b = n x t  (orthonormal frame)
        nn = nn - (nn * t).sum(1, keepdims=True) * t
        nn /= np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-9)
        b = np.cross(nn, t)
        b /= np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-9)
        orth_err.append(np.abs([(t * nn).sum(1), (t * b).sum(1), (nn * b).sum(1)]).max())
        E_all.append(e)
        T_all.append((e * t).sum(1)); B_all.append((e * b).sum(1)); N_all.append((e * nn).sum(1))
        Po = monotone_corrected(Pc, Gc)
        D_all.append(Po - Pc); EO_all.append(Po - Gc)
    if (i + 1) % 40 == 0:
        print(f"  {i+1}/{NE}", flush=True)

E = np.concatenate(E_all); et = np.concatenate(T_all)
eb = np.concatenate(B_all); en = np.concatenate(N_all)
D = np.concatenate(D_all); EO = np.concatenate(EO_all)
e2 = (E ** 2).sum(1)

print(f"\nframe orthonormality: max |dot| between axes = {max(orth_err):.2e}")
print("\n(1) LOCAL-FRAME ENERGY SPLIT  (exact for an orthonormal frame)")
print(f"  total       RMSE {np.sqrt(e2.mean()):.4f} mm      mean|e| {np.linalg.norm(E,axis=1).mean():.4f} mm")
res = e2.mean() - (et ** 2 + eb ** 2 + en ** 2).mean()
print(f"  identity residual  {res:+.2e} mm^2 (should be ~0)")
for nm, c in (("along-contour t", et), ("across-contour b", eb), ("normal n", en)):
    print(f"  {nm:17s} RMSE {np.sqrt((c**2).mean()):.4f}   energy share "
          f"{(c**2).mean()/e2.mean()*100:5.1f}%   mean|comp| {np.abs(c).mean():.4f}")

print("\n(2) ORACLE DISPLACEMENT IDENTITY  (cross-term measured, nothing assumed)")
b2 = e2.mean(); o2 = (EO ** 2).sum(1).mean(); d2 = (D ** 2).sum(1).mean()
cross = (EO * D).sum(1).mean()
print(f"  <|e_base|^2>     {b2:.4f}")
print(f"  <|e_oracle|^2>   {o2:.4f}   (RMSE {np.sqrt(o2):.4f}, mean|e| {np.linalg.norm(EO,axis=1).mean():.4f})")
print(f"  <|d|^2>          {d2:.4f}   (RMSE {np.sqrt(d2):.4f}, mean|d| {np.linalg.norm(D,axis=1).mean():.4f})")
print(f"  -2<e_oracle.d>   {-2*cross:+.4f}")
print(f"  sum             {o2 + d2 - 2*cross:.4f}   vs <|e_base|^2> {b2:.4f}  "
      f"(residual {o2+d2-2*cross-b2:+.2e})")
cc = cross / np.sqrt(max(o2, 1e-12) * max(d2, 1e-12))
print(f"  normalised cross-term <e_o.d>/(rms|e_o| rms|d|) = {cc:+.3f}  "
      f"=> {'NOT orthogonal' if abs(cc) > 0.1 else 'near-orthogonal'}")
print("\nNOTE: mean-of-Euclidean (the competition metric) and RMS energy are different")
print("      functionals; both are reported above and must not be mixed in one identity.")
np.savez("scratch/decomp_valid.npz", et=et, eb=eb, en=en, D=D, EO=EO)
print("\nsaved scratch/decomp_valid.npz")
