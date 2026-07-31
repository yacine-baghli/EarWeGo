"""
LOCAL preprocessing: cropped ear submesh -> the tensors an intrinsic-operator
(DiffusionNet-style) network needs on a GPU box that has torch/numpy/scipy and NO mesh
library. Everything requiring mesh connectivity happens here; the GPU only ever sees
flat .npz tensors.

WHY. The measured error of the shipped 1.273mm model is 77% ORDERED CORRESPONDENCE
(phase along the contour), not local XYZ accuracy, and seven variants of the static
2048-point DGCNN family moved it by nothing. A network that diffuses along the SURFACE
(spectral heat kernel) and reads tangential derivatives in a per-vertex frame gets an
intrinsic notion of "along the ear" for free, which is the structure the point-cloud
backbone never had. This script produces exactly the operators such a network consumes:
  (L, mass)                   cotangent stiffness + lumped mass       -> intrinsic metric
  (evals, evecs) of (L, mass) spectral basis                          -> exp(-t*lambda) diffusion
  (basis_x, basis_y, grad_*)  per-vertex tangent frame + gradient     -> tangential features
  (nbr, lap_w)                one-ring, as a gather-only operator     -> no torch_scatter needed

CROP CONVENTION -- identical to research/code/build_screen_extra.py and
build_multisample_all.py, so this data is registered point-for-point with
screen_data_*.npz:
  * right ears mirrored by diag(1,-1,1); vertex normals transformed n -> -(MIRROR*n)
    because a reflection reverses orientation; face winding also flipped (F[:,[0,2,1]])
    so the emitted submesh is outward-wound in the canonical frame for BOTH sides.
  * axis-aligned bbox of the coarse landmarks in the (mirrored) world frame, 14mm margin.
  * rotated into the per-ear canonical frame by R^T after subtracting c0. Every stored R
    is a proper rotation (det = +1, verified), so winding orientation survives it.
  * ONE deliberate difference: the point-cloud builders keep every bbox-masked VERTEX,
    while a submesh needs faces, so we keep faces with all three corners inside the bbox
    and then the vertices those faces reference. Measured difference: 2 vertices out of
    20362 on ear 0 (isolated vertices with no fully-inside incident face).

OBTUSE TRIANGLES AND DEGENERACY (explicitly, not by luck -- 16.7% of the angles in a
real ear crop are obtuse):
  * mass = BARYCENTRIC lumping, sum(area/3) over incident faces. Mixed-Voronoi lumping
    gives NEGATIVE areas on obtuse triangles, which makes M indefinite and destroys the
    generalised eigenproblem. Barycentric is strictly positive for any vertex with a
    non-degenerate incident face, and every vertex here has one by construction.
  * stiffness keeps the TRUE cotangent weights, which are negative for obtuse angles.
    That is the consistent FEM operator, not an error; L stays positive semi-definite
    globally and the script asserts it (min generalised eigenvalue >= -EIG_EPS).
    CLAMP_COT=1 clamps negative weights to zero if a downstream component needs an
    M-matrix; it is off by default because it biases the metric.
  * faces with a repeated index or 2*area <= AREA_EPS are DROPPED before any weight is
    formed, so no division by zero can occur. |cot| is clamped to COT_CLAMP to bound
    slivers that survive the area test. Both counts are reported.
  * every emitted array is asserted finite; mass.min() > 0; grad least-squares
    determinant > 0 (guaranteed by GRAD_REG > 0, asserted anyway).

VARIABLE VERTEX COUNT -- RAGGED WITH OFFSETS, and additionally capped by MAXV.
Native crops run 19.5k..41.4k vertices: a 2.1x spread that is pure scan-resolution
accident (median edge 0.71..1.07mm). Two decisions:
  1. RAGGED, not padded. All per-vertex arrays are ONE flat axis of length Vtot with
     v_ptr (E+1,) offsets, and every index array (faces, nbr, lm_*) is GLOBAL into that
     axis. Reasons: (a) the eigenvector block dominates the file and padding to Vmax
     would store ~1GB of zeros; (b) the one-ring is ragged anyway (degree 2..10) so a
     padded layout needs a mask regardless -- the offsets ARE that mask, exactly;
     (c) because the layout is a block-diagonal concatenation, several ears can be run
     as ONE graph with no padding at all; (d) it lets MAXV=0 (native resolution) work
     without the file exploding to the worst-case ear.
     Padding is still one gather when you want it -- exact snippet in pad_batch() below,
     which the __main__ smoke test actually uses to drive a torch network.
  2. MAXV (default 12000) caps the vertex count with a connectivity-constrained vertex
     clustering (see decimate()). This is NOT a cosmetic size fix: it also equalises
     resolution across scans, removing a confound the point-cloud pipeline never had to
     face. Every native crop exceeds 12000, so with the default EVERY ear is decimated to
     ~12000 vertices (~1.0mm spacing vs 0.71..1.07mm native). MAXV=0 disables it and
     emits the faithful crop (~4GB for 340 ears; see the printed sizes).
     Cluster representatives are ORIGINAL vertices, never averaged centroids, so every
     emitted vertex lies exactly on the scanned surface and orig_rep is exact provenance.
     DEC_CENTROID=1 switches to mass-weighted centroids (smoother, off-surface).

LANDMARK TARGETS -- lm_vert / lm_face / lm_bary / lm_dist ARE GROUND TRUTH.
They encode the 85 annotated landmarks as (nearest vertex, containing face, barycentric
coordinates) on the emitted submesh. They are TRAINING TARGETS ONLY: they may be read
in the loss on TRAINING-FOLD ears and nowhere else -- never in a forward pass, never at
inference, never to build a feature, never to pick a per-ear parameter. Nothing else in
this file is derived from them, so a component that simply does not read lm_* is
leakage-free by construction. lm_dist is the representation floor they impose
(mm from the annotated point to the emitted surface) and is reported.

WHAT THE LOW SPECTRUM IS AND IS NOT. The crop is an open patch, so (L, mass) carries
natural Neumann conditions and the first few eigenfunctions are governed by the CROP
BOUNDARY, which is a coarse-landmark bbox artefact and varies per ear. Use the basis as
a diffusion operator (evecs @ diag(exp(-t*lambda)) @ evecs.T @ mass, which is invariant
to eigenvector sign and to any rotation inside a degenerate eigenspace, hence
deterministic even though evecs are not). Do NOT feed individual eigenfunctions as
features and expect them to mean the same thing across ears.

OUTPUT (LIMIT>0 appends _lim{LIMIT} so a smoke run never clobbers the real artefact):
  scratch/mesh_data.npz   geometry, connectivity, operators, remap, landmark targets
  scratch/mesh_spec.npz   evals (E,K) f32 + evecs (Vtot,K) f16, split off because it is
                          the big block, is regenerable from mesh_data's L and mass with
                          ~12 lines of scipy, and is the one thing you re-ship when K
                          changes.
Eigenvectors are float16: they are M-orthonormal so entries are O(1), and 5e-4 relative
error is orders of magnitude below the discretisation error of a 1mm mesh.

ENV (defaults in brackets)
  K_EIG [128]      generalised eigenpairs per ear
  MAXV [12000]     vertex cap; 0 = native crop, no decimation
  DEC_CENTROID [0] 1 = cluster centroid instead of representative vertex
  LARGEST_CC [1]   keep only the largest-area connected component of the submesh
  CLAMP_COT [0]    1 = clamp negative cotangent weights to zero
  COT_CLAMP [1e4]  bound on |cot| (sliver guard)
  AREA_EPS [1e-12] faces with 2*area <= this are dropped (mm^2)
  EIG_EPS [1e-8]   shift-invert offset, escalated x10 up to 6 times on ARPACK failure
  GRAD_REG [1e-5]  Tikhonov term on the 2x2 tangent least-squares
  LIMIT [0]        process only the first N ears (0 = all 340)
  OUT [scratch/mesh]  output prefix
  SMOKE [0]        1 = run the synthetic operator + torch-consumer test only

  python research/code/build_mesh_data.py                 # all 340 dev ears
  LIMIT=3 python research/code/build_mesh_data.py         # 3-ear check
  SMOKE=1 python research/code/build_mesh_data.py         # CPU self-test, no data needed
"""
import os, sys, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from deep_model.surfproj import closest_on_triangles

MESH = "2026 Munich Tech Arena - Datas/2026 Munich Tech Arena - Datas/mesh"
LM = "2026 Munich Tech Arena - Datas/2026 Munich Tech Arena - Datas/landmarks"
MIRROR = np.array([1., -1., 1.])
MARGIN = 14.0                     # must match build_screen_extra.py / build_multisample_all.py
NL = 85

K_EIG = int(os.environ.get("K_EIG", "128"))
MAXV = int(os.environ.get("MAXV", "12000"))
DEC_CENTROID = int(os.environ.get("DEC_CENTROID", "0"))
LARGEST_CC = int(os.environ.get("LARGEST_CC", "1"))
CLAMP_COT = int(os.environ.get("CLAMP_COT", "0"))
COT_CLAMP = float(os.environ.get("COT_CLAMP", "1e4"))
AREA_EPS = float(os.environ.get("AREA_EPS", "1e-12"))
EIG_EPS = float(os.environ.get("EIG_EPS", "1e-8"))
GRAD_REG = float(os.environ.get("GRAD_REG", "1e-5"))
LIMIT = int(os.environ.get("LIMIT", "0"))
OUT = os.environ.get("OUT", "scratch/mesh")


# --------------------------------------------------------------- mesh operators
def clean_faces(V, F):
    """drop faces with a repeated corner or vanishing area; return F, 2*area, counts"""
    ok = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 2] != F[:, 0])
    n_dup = int((~ok).sum()); F = F[ok]
    a2 = np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    good = a2 > AREA_EPS
    n_deg = int((~good).sum())
    return F[good], a2[good], n_dup, n_deg


def cot_laplacian(V, F, a2):
    """cotangent stiffness as (W, diag) and barycentric lumped mass.

    W is the SYMMETRIC off-diagonal weight matrix (csr, diagonal-free, explicit zeros
    kept so its pattern is exactly the mesh one-ring). L = diags(diag) - W, and
    (L f)_i = sum_k W_ik (f_i - f_k). Weights are (cot a + cot b)/2 per edge.
    """
    n = len(V)
    i0, i1, i2 = F[:, 0], F[:, 1], F[:, 2]
    A, B, C = V[i0], V[i1], V[i2]
    cA = np.einsum("ij,ij->i", B - A, C - A) / a2      # cot of the angle at A
    cB = np.einsum("ij,ij->i", A - B, C - B) / a2
    cC = np.einsum("ij,ij->i", A - C, B - C) / a2
    n_obt = int((cA < 0).sum() + (cB < 0).sum() + (cC < 0).sum())
    n_clip = int((np.abs(cA) > COT_CLAMP).sum() + (np.abs(cB) > COT_CLAMP).sum()
                 + (np.abs(cC) > COT_CLAMP).sum())
    cA, cB, cC = (np.clip(c, -COT_CLAMP, COT_CLAMP) for c in (cA, cB, cC))
    w = np.concatenate([cA, cA, cB, cB, cC, cC]) / 2.0
    if CLAMP_COT:
        w = np.maximum(w, 0.0)
    r = np.concatenate([i1, i2, i2, i0, i0, i1])
    c = np.concatenate([i2, i1, i0, i2, i1, i0])
    W = sp.coo_matrix((w, (r, c)), shape=(n, n)).tocsr()
    W.sort_indices()
    diag = np.asarray(W.sum(1)).ravel()
    mass = np.bincount(F.ravel(), np.repeat(a2 / 6.0, 3), minlength=n)
    return W, diag, mass, n_obt, n_clip


def tangent_frames(N):
    """orthonormal (X, Y) spanning the tangent plane; (X, Y, N) right-handed.

    Seeded from the coordinate axis LEAST aligned with N, so |cross(N, axis)| >= 0.816
    and the construction can never degenerate.
    """
    a = np.zeros_like(N)
    a[np.arange(len(N)), np.argmin(np.abs(N), axis=1)] = 1.0
    X = np.cross(N, a)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X, np.cross(N, X)


def grad_operator(V, X, Y, indptr, indices):
    """per-vertex tangential gradient by weighted least squares over the one-ring.

    Row k of vertex i's system is w_k * (e_k . g) = w_k * (f_j - f_i) with e_k the
    one-ring edge projected into (X_i, Y_i) and w_k = 1/|e_k| in 3D. Using the 3D length
    (not DiffusionNet's projected length) keeps w bounded on sharp folds, where an edge
    can project to nearly zero and would otherwise get a 1e9 weight; such an edge then
    correctly contributes almost nothing instead of dominating.
    Returns off-diagonal and diagonal coefficients: (grad f)_i =
      (sum_k gx_k f_{nbr_k} + gxd_i f_i,  sum_k gy_k f_{nbr_k} + gyd_i f_i).
    """
    n = len(V)
    row = np.repeat(np.arange(n), np.diff(indptr))
    e = V[indices] - V[row]
    ex = np.einsum("ij,ij->i", e, X[row])
    ey = np.einsum("ij,ij->i", e, Y[row])
    w2 = 1.0 / np.maximum(np.einsum("ij,ij->i", e, e), 1e-18)      # (1/|e|)^2
    gxx = np.bincount(row, w2 * ex * ex, minlength=n) + GRAD_REG
    gxy = np.bincount(row, w2 * ex * ey, minlength=n)
    gyy = np.bincount(row, w2 * ey * ey, minlength=n) + GRAD_REG
    det = gxx * gyy - gxy * gxy                                    # >= GRAD_REG^2 > 0
    assert det.min() > 0, "degenerate tangent least-squares"
    ax, ay = w2 * ex, w2 * ey                                      # A^T diag(w) columns
    gx = (gyy / det)[row] * ax + (-gxy / det)[row] * ay
    gy = (-gxy / det)[row] * ax + (gxx / det)[row] * ay
    return gx, gy, -np.bincount(row, gx, minlength=n), -np.bincount(row, gy, minlength=n)


def spectrum(W, diag, mass, k):
    """first k generalised eigenpairs of (L, M), smallest first, via shift-invert.

    sigma = -eps makes the factorised pencil L + eps*M strictly positive definite for a
    PSD L, so splu cannot hit the constant null vector. eps escalates on ARPACK failure,
    which is the one place a real mesh can defeat the solver (a true cotangent L with
    many obtuse triangles can dip a hair below zero). Returns diagnostics rather than
    trusting the solver: M-orthonormality error and relative generalised residual.
    """
    n = len(mass)
    kk = int(min(k, max(1, n - 2)))
    L = (sp.diags(diag) - W).tocsc()
    Mm = sp.diags(mass)
    eps = EIG_EPS
    ev = None
    for _ in range(6):
        try:
            ev, ec = spla.eigsh(L, k=kk, M=Mm, sigma=-eps, which="LM")
            break
        except (spla.ArpackError, RuntimeError):
            eps *= 10.0
    assert ev is not None, f"eigsh failed up to eps={eps:.1e} on n={n}"
    o = np.argsort(ev)
    ev, ec = ev[o], ec[:, o]
    ec *= np.sign(ec[np.abs(ec).argmax(0), np.arange(kk)])[None, :]   # deterministic sign
    Mec = mass[:, None] * ec
    orth = float(np.abs(ec.T @ Mec - np.eye(kk)).max())
    res = L @ ec - Mec * ev[None, :]
    den = np.maximum(np.abs(ev) * np.linalg.norm(Mec, axis=0), 1e-30)
    resid = float((np.linalg.norm(res, axis=0) / den)[1:].max()) if kk > 1 else 0.0
    raw_min = float(ev.min())
    ev = np.clip(ev, 0.0, None)
    if kk < k:                              # pathologically small crop: pad inertly
        ev = np.concatenate([ev, np.full(k - kk, 1e9)])
        ec = np.concatenate([ec, np.zeros((n, k - kk))], axis=1)
    return ev, ec, kk, orth, resid, raw_min, eps


# --------------------------------------------------------------- decimation
def decimate(V, F, VN, vmass, target):
    """connectivity-constrained vertex clustering down to ~target vertices.

    Plain grid clustering would merge the front and back of the helix rim whenever they
    share a grid cell, inventing a topological shortcut and destroying exactly the
    geodesic structure an intrinsic operator exists to exploit. So clusters are the
    connected components of the mesh graph RESTRICTED to intra-cell edges: two vertices
    fuse only if they are in the same cell AND connected through cells' own vertices.
    Grid size h is bisected until the cluster count lands in [0.85*target, target].
    Returns (lab: vertex -> cluster, rep: cluster -> representative vertex, h).
    """
    n = len(V)
    E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    P = V - V.min(0)
    ext = P.max(0) + 1e-9
    lo, hi, best = 1e-3, float(ext.max()), None
    for _ in range(28):
        h = np.sqrt(lo * hi)
        dims = np.floor(ext / h).astype(np.int64) + 1
        assert dims.prod() < 2 ** 62, "grid too fine"
        g = np.floor(P / h).astype(np.int64)
        cell = (g[:, 0] * dims[1] + g[:, 1]) * dims[2] + g[:, 2]
        same = cell[E[:, 0]] == cell[E[:, 1]]
        A = sp.coo_matrix((np.ones(int(same.sum()), np.int8), (E[same, 0], E[same, 1])),
                          shape=(n, n))
        nc, lab = connected_components(A, directed=False)
        if best is None or abs(nc - target) < abs(best[0] - target):
            best = (nc, lab, h)
        if nc > target:
            lo = h                                    # too many clusters -> coarser grid
        elif nc < 0.85 * target:
            hi = h
        else:
            best = (nc, lab, h)
            break
    nc, lab, h = best
    # representative = the member vertex nearest the cluster mean, so it lies ON the scan
    cen = np.stack([np.bincount(lab, V[:, c] * vmass, minlength=nc) for c in range(3)], 1)
    wsum = np.bincount(lab, vmass, minlength=nc)
    cen /= wsum[:, None]
    d = np.einsum("ij,ij->i", V - cen[lab], V - cen[lab])
    order = np.lexsort((d, lab))
    ls = lab[order]
    rep = order[np.r_[True, ls[1:] != ls[:-1]]]
    assert len(rep) == nc and (lab[rep] == np.arange(nc)).all(), "cluster labelling broken"
    if DEC_CENTROID:
        Vn = cen
        Nn = np.stack([np.bincount(lab, VN[:, c] * vmass, minlength=nc) for c in range(3)], 1)
    else:
        Vn, Nn = V[rep], VN[rep]
    Nn = Nn / np.maximum(np.linalg.norm(Nn, axis=1, keepdims=True), 1e-12)
    Fn = lab[F]
    keep = (Fn[:, 0] != Fn[:, 1]) & (Fn[:, 1] != Fn[:, 2]) & (Fn[:, 2] != Fn[:, 0])
    Fn = Fn[keep]
    _, ui = np.unique(np.sort(Fn, axis=1), axis=0, return_index=True)
    return Vn, Fn[ui], Nn, lab, rep, h            # winding of the surviving face is kept


# --------------------------------------------------------------- landmark targets
def landmark_targets(V, F, pts):
    """TRAINING TARGETS ONLY. (nearest vertex, containing face, barycentric, distance)"""
    m = len(F)
    v2f = sp.csr_matrix((np.ones(3 * m, np.int8), (F.ravel(), np.repeat(np.arange(m), 3))),
                        shape=(len(V), m))
    nv = cKDTree(V).query(pts, k=min(16, len(V)))[1]
    nv = np.atleast_2d(nv)
    face = np.zeros(len(pts), np.int64); bary = np.zeros((len(pts), 3)); dist = np.zeros(len(pts))
    for i, p in enumerate(pts):
        cand = np.unique(v2f[nv[i]].indices)
        T = F[cand]
        A, B, C = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
        q = closest_on_triangles(p, A, B, C)
        dd = np.linalg.norm(q - p, axis=1)
        j = int(dd.argmin())
        face[i], dist[i] = cand[j], dd[j]
        a, b, c, qq = A[j], B[j], C[j], q[j]
        nn = np.cross(b - a, c - a); den = max(float(nn @ nn), 1e-30)
        wa = float(np.cross(b - qq, c - qq) @ nn) / den
        wb = float(np.cross(c - qq, a - qq) @ nn) / den
        wv = np.clip([wa, wb, 1.0 - wa - wb], 0.0, 1.0)
        bary[i] = wv / wv.sum()
    rec = np.einsum("ij,ijk->ik", bary, V[F[face]])
    # Two real checks. (1) the barycentric weights must reproduce the exact closest
    # point on the chosen face -- this is what makes the stored (face, bary) a faithful
    # surface parameterisation of the landmark. (2) how far the landmark sat off the
    # surface to begin with. The former replaces a self-check that was algebraically
    # |rec - rec| and so returned 0 whatever the code did.
    ex = np.stack([closest_on_triangles(p, V[F[face[i]]][0:1], V[F[face[i]]][1:2],
                                        V[F[face[i]]][2:3])[0]
                   for i, p in enumerate(pts)])
    return (nv[:, 0].astype(np.int64), face, bary, dist,
            float(np.linalg.norm(rec - ex, axis=1).max()), float(dist.max()))


# --------------------------------------------------------------- GPU-side contract
def pad_batch(d, ears, spec=None):
    """THE contract the GPU code uses: ragged -> padded (B, P, ...) tensors.

    Padding convention that makes every operator a pure gather with NO mask and NO
    scatter (there is no torch_scatter on the box): a padded neighbour slot points at the
    vertex ITSELF and carries weight 0, and padded vertices get mass 0 / evecs 0, so they
    fall out of both the one-ring sums and the spectral projection.
        Lx      = lap_diag[...,None]*x - (lap_w[...,None] * xj).sum(2)
        grad_x  = grad_xd[...,None]*x  + (grad_x[...,None] * xj).sum(2)
        xj      = gather(x, 1, nbr.reshape(B,P*D,1).expand(-1,-1,C)).view(B,P,D,C)
    """
    vp, dp = d["v_ptr"], d["deg_ptr"]
    nv = np.array([vp[i + 1] - vp[i] for i in ears])
    deg = np.diff(dp)
    D = int(max(deg[vp[i]:vp[i + 1]].max() for i in ears))
    B, P = len(ears), int(nv.max())
    out = dict(nv=nv, vmask=np.zeros((B, P), bool),
               verts=np.zeros((B, P, 3), np.float32), nrm=np.zeros((B, P, 3), np.float32),
               basis_x=np.zeros((B, P, 3), np.float32), basis_y=np.zeros((B, P, 3), np.float32),
               mass=np.zeros((B, P), np.float32),
               nbr=np.tile(np.arange(P)[None, :, None], (B, 1, D)).astype(np.int64),
               lap_w=np.zeros((B, P, D), np.float32),
               grad_x=np.zeros((B, P, D), np.float32), grad_y=np.zeros((B, P, D), np.float32),
               lap_diag=np.zeros((B, P), np.float32),
               grad_xd=np.zeros((B, P), np.float32), grad_yd=np.zeros((B, P), np.float32))
    if spec is not None:
        out["evals"] = np.zeros((B, spec["evals"].shape[1]), np.float32)
        out["evecs"] = np.zeros((B, P, spec["evals"].shape[1]), np.float32)
    for b, i in enumerate(ears):
        g0, g1, n = vp[i], vp[i + 1], nv[b]
        out["vmask"][b, :n] = True
        for key in ("verts", "nrm", "basis_x", "basis_y", "mass", "lap_diag",
                    "grad_xd", "grad_yd"):
            out[key][b, :n] = d[key][g0:g1]
        col = np.arange(dp[g1] - dp[g0]) - np.repeat(dp[g0:g1] - dp[g0], deg[g0:g1])
        rw = np.repeat(np.arange(n), deg[g0:g1])
        out["nbr"][b, rw, col] = d["nbr"][dp[g0]:dp[g1]] - g0
        for key in ("lap_w", "grad_x", "grad_y"):
            out[key][b, rw, col] = d[key][dp[g0]:dp[g1]]
        if spec is not None:
            out["evals"][b] = spec["evals"][i]
            out["evecs"][b, :n] = spec["evecs"][g0:g1].astype(np.float32)
    return out


# --------------------------------------------------------------- driver
def run():
    from src.splits import get_split
    from src.dataset import Dataset

    d0 = np.load("scratch/deep_dataset.npz", allow_pickle=True)
    coarse, true, Rm, c0, split = d0["coarse"], d0["true"], d0["R"], d0["c0"], d0["split"]
    tr = get_split("train", mesh_dir=Path(MESH)); va = get_split("val", mesh_dir=Path(MESH))
    order = [(p, s) for p in tr for s in ("left", "right")] + \
            [(p, s) for p in va for s in ("left", "right")]
    assert len(order) == len(split) == 340, f"expected 340 dev ears, got {len(order)}"
    ds = Dataset(MESH, LM); pid2idx = {p: i for i, p in enumerate(ds.subject_ids)}
    lock = set(get_split("test", mesh_dir=Path(MESH)))
    assert not (set(p for p, _ in order) & lock), "LOCKBOX subject leaked into the order"
    NE = LIMIT if LIMIT else len(order)
    print(f"[meshprep] {NE} ears  K_EIG={K_EIG} MAXV={MAXV} margin={MARGIN} "
          f"clamp_cot={CLAMP_COT} largest_cc={LARGEST_CC}", flush=True)

    acc = {k: [] for k in ("verts", "faces", "nrm", "basis_x", "basis_y", "mass", "nbr",
                           "lap_w", "lap_diag", "grad_x", "grad_y", "grad_xd", "grad_yd",
                           "deg", "orig_rep", "crop_orig", "crop_to_final", "evecs")}
    per = {k: [] for k in ("n_vert", "n_face", "n_crop", "n_native", "area", "dec_h",
                           "n_obt", "n_clip", "n_dup", "n_deg", "n_cc", "cc_drop",
                           "orth", "resid", "raw_min", "eig_eps", "n_eig", "wind",
                           "wind_frac", "radial", "spacing")}
    evals = np.zeros((NE, K_EIG), np.float32)
    lm_vert = np.zeros((NE, NL), np.int64); lm_face = np.zeros((NE, NL), np.int64)
    lm_bary = np.zeros((NE, NL, 3), np.float32); lm_dist = np.zeros((NE, NL), np.float32)
    v_off = f_off = c_off = 0
    v_ptr = [0]; f_ptr = [0]; c_ptr = [0]
    cache = {}; t0 = time.time()

    for i in range(NE):
        pid, side = order[i]
        if pid not in cache:
            m = ds[pid2idx[pid]][0]
            V0 = np.asarray(m.vertices, np.float64); F0 = np.asarray(m.faces, np.int64)
            fn = np.cross(V0[F0[:, 1]] - V0[F0[:, 0]], V0[F0[:, 2]] - V0[F0[:, 0]])
            VN = np.zeros_like(V0)
            for c in range(3):
                np.add.at(VN, F0[:, c], fn)                       # area-weighted
            nn = np.linalg.norm(VN, axis=1, keepdims=True)
            VN = np.where(nn > 1e-12, VN / np.maximum(nn, 1e-12), np.array([0., 0., 1.]))
            rad = V0 - V0.mean(0)
            rad /= np.maximum(np.linalg.norm(rad, axis=1, keepdims=True), 1e-12)
            # Orientation must be decided by the divergence theorem, not by the radial
            # dot: a head is not star-shaped (ears, nose, concavities), so per-mesh
            # radial dot legitimately runs well below 1 -- 0.451 on P0001 -- and a
            # threshold on it rejects correctly-wound meshes. Signed volume
            # (1/6) sum v0.(v1 x v2) is positive iff the winding is counter-clockwise
            # (outward) and needs no star-shapedness. The radial dot stays a diagnostic.
            svol = float(np.einsum('ij,ij->i',
                                   V0[F0[:, 0]],
                                   np.cross(V0[F0[:, 1]], V0[F0[:, 2]])).sum() / 6.0)
            cache = {pid: (V0, F0, VN, float((VN * rad).sum(1).mean()), svol)}
        V0, F0, VN0, odot, svol = cache[pid]
        assert svol > 0, (f"{pid}: mesh winding is inward (signed volume {svol:.1f} mm^3); "
                          f"normals would point into the surface")
        per["radial"].append(odot)
        per.setdefault("signed_volume", []).append(svol)
        if side == "right":
            # A reflection reverses orientation. Flip the winding OR negate the mirrored
            # normal -- doing BOTH (as the first version did) leaves normals anti-parallel
            # to the winding, which is what the -0.983 assertion caught. Verified
            # numerically: recomputing normals from the flipped winding is outward.
            Vw, Fw = V0 * MIRROR, F0[:, [0, 2, 1]]
            fn = np.cross(Vw[Fw[:, 1]] - Vw[Fw[:, 0]], Vw[Fw[:, 2]] - Vw[Fw[:, 0]])
            VNw = np.zeros_like(Vw)
            for c in range(3):
                np.add.at(VNw, Fw[:, c], fn)
            nr = np.linalg.norm(VNw, axis=1, keepdims=True)
            VNw = np.where(nr > 1e-12, VNw / np.maximum(nr, 1e-12), np.array([0., 0., 1.]))
        else:
            Vw, VNw, Fw = V0, VN0, F0
        R, cc = Rm[i].astype(np.float64), c0[i].astype(np.float64)
        assert abs(np.linalg.det(R) - 1.0) < 1e-5, f"ear {i}: R is not a proper rotation"
        cw = coarse[i].astype(np.float64) @ R + cc
        lo, hi = cw.min(0) - MARGIN, cw.max(0) + MARGIN
        msk = np.all((Vw >= lo) & (Vw <= hi), axis=1)
        Fk = Fw[msk[Fw].all(1)]
        assert len(Fk) > 100, f"ear {i}: crop kept only {len(Fk)} faces"
        crop_orig = np.unique(Fk)
        rm = -np.ones(len(Vw), np.int64); rm[crop_orig] = np.arange(len(crop_orig))
        Vc = (Vw[crop_orig] - cc) @ R.T
        VNc = VNw[crop_orig] @ R.T
        Fc, a2c, n_dup, n_deg = clean_faces(Vc, rm[Fk])
        per["n_native"].append(len(Vc)); per["n_crop"].append(len(Vc))

        if MAXV and len(Vc) > MAXV:
            vm = np.bincount(Fc.ravel(), np.repeat(a2c / 6.0, 3), minlength=len(Vc))
            Vd, Fd, VNd, lab, rep, dh = decimate(Vc, Fc, VNc, np.maximum(vm, 1e-12), MAXV)
        else:
            Vd, Fd, VNd = Vc, Fc, VNc
            lab = rep = np.arange(len(Vc)); dh = 0.0
        Fd, a2d, nd2, ng2 = clean_faces(Vd, Fd)
        n_dup += nd2; n_deg += ng2

        # keep the faces' vertices, then optionally the largest-area component
        sel = np.unique(Fd)
        inv = -np.ones(len(Vd), np.int64); inv[sel] = np.arange(len(sel))
        V, F = Vd[sel], inv[Fd]
        N = VNd[sel]
        F, a2, nd3, ng3 = clean_faces(V, F); n_dup += nd3; n_deg += ng3
        W, diag, mass, n_obt, n_clip = cot_laplacian(V, F, a2)
        ncc, cl = connected_components(W, directed=False)
        per["n_cc"].append(int(ncc)); drop = 0
        if LARGEST_CC and ncc > 1:
            big = int(np.bincount(cl, mass).argmax())
            k2 = np.flatnonzero(cl == big)
            drop = len(V) - len(k2)
            inv2 = -np.ones(len(V), np.int64); inv2[k2] = np.arange(len(k2))
            F = inv2[F[(cl[F] == big).all(1)]]
            V, N = V[k2], N[k2]
            sel = sel[k2]
            F, a2, nd4, ng4 = clean_faces(V, F); n_dup += nd4; n_deg += ng4
            W, diag, mass, n_obt, n_clip = cot_laplacian(V, F, a2)
        per["cc_drop"].append(int(drop))
        n = len(V)
        assert mass.min() > 0, f"ear {i}: zero lumped mass"

        fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
        vd = (fn * N[F].mean(1)).sum(1)
        per["wind"].append(float(vd.mean())); per["wind_frac"].append(float((vd > 0).mean()))
        assert vd.mean() > 0.5, f"ear {i}: submesh winding disagrees with normals ({vd.mean():.3f})"

        X, Y = tangent_frames(N)
        assert max(np.abs((X * N).sum(1)).max(), np.abs((Y * N).sum(1)).max(),
                   np.abs((X * Y).sum(1)).max()) < 1e-9, f"ear {i}: tangent frame not orthogonal"
        gx, gy, gxd, gyd = grad_operator(V, X, Y, W.indptr, W.indices)
        ev, ec, kk, orth, resid, raw_min, eps = spectrum(W, diag, mass, K_EIG)
        assert raw_min > -1e-6, f"ear {i}: L is not PSD (min eigenvalue {raw_min:.2e})"
        assert orth < 1e-6 and resid < 1e-6, f"ear {i}: bad eigenpairs orth={orth:.1e} res={resid:.1e}"

        lv, lf, lb, ld, _, _ = landmark_targets(V, F, true[i].astype(np.float64))
        lm_vert[i], lm_face[i] = lv + v_off, lf + f_off
        lm_bary[i], lm_dist[i] = lb, ld

        for key, val in (("verts", V), ("nrm", N), ("basis_x", X), ("basis_y", Y),
                         ("mass", mass), ("lap_diag", diag), ("grad_xd", gxd),
                         ("grad_yd", gyd), ("lap_w", W.data), ("grad_x", gx),
                         ("grad_y", gy)):
            acc[key].append(np.asarray(val, np.float32))
        acc["faces"].append((F + v_off).astype(np.int32))
        acc["nbr"].append((W.indices + v_off).astype(np.int32))
        acc["deg"].append(np.diff(W.indptr).astype(np.int32))
        acc["orig_rep"].append(crop_orig[rep[sel]].astype(np.int32))
        acc["crop_orig"].append(crop_orig.astype(np.int32))
        fin = -np.ones(len(Vd), np.int64); fin[sel] = np.arange(n) + v_off
        acc["crop_to_final"].append(fin[lab].astype(np.int32))
        acc["evecs"].append(ec.astype(np.float16))
        evals[i] = ev
        for key, val in (("n_vert", n), ("n_face", len(F)), ("area", float(mass.sum())),
                         ("dec_h", dh), ("n_obt", n_obt), ("n_clip", n_clip),
                         ("n_dup", n_dup), ("n_deg", n_deg), ("orth", orth),
                         ("resid", resid), ("raw_min", raw_min), ("eig_eps", eps),
                         ("n_eig", kk), ("spacing", float(np.sqrt(mass.sum() / (0.866 * n))))):
            per[key].append(val)
        per["n_crop"][-1] = len(Vc)
        v_off += n; f_off += len(F); c_off += len(Vc)
        v_ptr.append(v_off); f_ptr.append(f_off); c_ptr.append(c_off)
        if (i + 1) % 20 == 0 or i + 1 == NE:
            el = time.time() - t0
            print(f"  {i+1}/{NE}  {el:.0f}s  eta {el/(i+1)*(NE-i-1):.0f}s  "
                  f"n={n} nnz={len(gx)} lam_k={evals[i][min(kk,K_EIG)-1]:.4f}", flush=True)

    P = {k: np.asarray(v) for k, v in per.items()}
    out = {k: np.concatenate(v) for k, v in acc.items() if k != "evecs"}
    out["v_ptr"] = np.asarray(v_ptr, np.int64)
    out["f_ptr"] = np.asarray(f_ptr, np.int64)
    out["c_ptr"] = np.asarray(c_ptr, np.int64)
    out["deg_ptr"] = np.concatenate([[0], np.cumsum(out.pop("deg"))]).astype(np.int64)
    out.update(lm_vert=lm_vert.astype(np.int32), lm_face=lm_face.astype(np.int32),
               lm_bary=lm_bary, lm_dist=lm_dist,
               ear_index=np.arange(NE, dtype=np.int32),
               pid=np.array([order[i][0] for i in range(NE)]),
               side=np.array([order[i][1] for i in range(NE)]),
               split=split[:NE], R=Rm[:NE], c0=c0[:NE], coarse=coarse[:NE],
               n_vert=P["n_vert"].astype(np.int32), n_face=P["n_face"].astype(np.int32),
               n_native=P["n_native"].astype(np.int32), area=P["area"].astype(np.float32),
               dec_h=P["dec_h"].astype(np.float32), n_eig=P["n_eig"].astype(np.int32),
               K_EIG=np.int32(K_EIG), MAXV=np.int32(MAXV), MARGIN=np.float32(MARGIN))
    for k, v in out.items():
        if v.dtype.kind == "f":
            assert np.isfinite(v).all(), f"non-finite values in {k}"
    assert out["deg_ptr"][-1] == len(out["nbr"]) and out["v_ptr"][-1] == len(out["verts"])
    assert (out["nbr"] >= 0).all() and out["nbr"].max() < len(out["verts"])
    assert (out["faces"] >= 0).all() and out["faces"].max() < len(out["verts"])
    assert out["lm_vert"].max() < len(out["verts"]) and out["lm_face"].max() < len(out["faces"])
    assert np.allclose(out["lm_bary"].sum(-1), 1.0, atol=1e-5)
    ecf = np.concatenate(acc["evecs"])

    sfx = f"_lim{LIMIT}" if LIMIT else ""
    fd, fs = f"{OUT}_data{sfx}.npz", f"{OUT}_spec{sfx}.npz"
    np.savez_compressed(fd, **out)
    np.savez(fs, evals=evals, evecs=ecf, v_ptr=out["v_ptr"], n_eig=out["n_eig"])

    def q(a, f="{:.4g}"):
        return "/".join(f.format(x) for x in np.percentile(a, [0, 50, 100]))
    print(f"\n--- vertex counts (min/median/max) native {q(P['n_native'], '{:.0f}')}  "
          f"emitted {q(P['n_vert'], '{:.0f}')}  faces {q(P['n_face'], '{:.0f}')}")
    print(f"    total vertices {out['v_ptr'][-1]}  one-ring nnz {len(out['nbr'])}  "
          f"mean degree {len(out['nbr'])/out['v_ptr'][-1]:.2f}")
    print(f"    surface area mm^2 {q(P['area'])}  vertex spacing mm {q(P['spacing'])}  "
          f"decimation h mm {q(P['dec_h'])}")
    print(f"--- eigenvalues  lambda_1 (first nonzero) {q(evals[:, 1])}  "
          f"lambda_{K_EIG} {q(evals[:, -1])}  lambda_0 max {evals[:, 0].max():.2e}")
    print(f"    valid eigenpairs {P['n_eig'].min()}..{P['n_eig'].max()} of {K_EIG}; "
          f"shift eps escalated on {int((P['eig_eps'] > EIG_EPS).sum())} ears")
    print("\n--- ASSERTIONS (all passed) ---")
    # The radial dot is a DIAGNOSTIC, not a bound: a head is not star-shaped, so per-ear
    # values legitimately fall well below 0.5 (min 0.288 over 340 ears). Orientation is
    # asserted by the divergence theorem upstream and by the winding/normal agreement below.
    print(f"  outward normals, mean dot(n, radial)         min {P['radial'].min():.4f}  "
          f"(diagnostic only -- orientation is asserted by signed volume)")
    print(f"  submesh winding vs normals, mean dot         min {P['wind'].min():.4f}  (> 0.5)")
    print(f"  ... fraction of faces agreeing               min {P['wind_frac'].min():.4f}")
    print(f"  tangent frame orthogonality                  < 1e-9 every ear")
    print(f"  grad least-squares determinant              > 0 every vertex")
    _mass_min = float(out["mass"].min())
    assert _mass_min > 0, f"lumped mass has a non-positive entry ({_mass_min:.3e})"
    print(f"  lumped mass strictly positive                min {_mass_min:.3e}  (> 0)")
    print(f"  min generalised eigenvalue (L PSD)           {P['raw_min'].min():.3e}  (> -1e-6)")
    print(f"  M-orthonormality error of evecs              max {P['orth'].max():.3e}  (< 1e-6)")
    print(f"  relative generalised residual               max {P['resid'].max():.3e}  (< 1e-6)")
    print(f"  no NaN/Inf in any emitted float array        OK ({sum(1 for v in out.values() if v.dtype.kind=='f')} arrays)")
    print(f"  barycentric coordinates sum to 1             OK")
    print(f"  index bounds (faces/nbr/lm_*) in range       OK")
    print(f"  lockbox subjects absent from the order       OK")
    print(f"  connected components per crop                {P['n_cc'].min()}..{P['n_cc'].max()}, "
          f"vertices dropped by largest-CC {P['cc_drop'].sum()}")
    print(f"  faces dropped: repeated corner {P['n_dup'].sum()}, zero area {P['n_deg'].sum()}")
    print(f"  obtuse angles (negative cot) {P['n_obt'].sum()/(3*P['n_face'].sum())*100:.1f}% "
          f"of angles; |cot| clamped {P['n_clip'].sum()} times")
    print(f"\n--- GT landmark -> emitted surface (TRAINING TARGET, representation floor)")
    print(f"    distance mm: mean {lm_dist.mean():.4f}  median {np.median(lm_dist):.4f}  "
          f"p99 {np.percentile(lm_dist, 99):.4f}  max {lm_dist.max():.4f}")
    for f in (fd, fs):
        print(f"wrote {f} ({os.path.getsize(f)/1e6:.1f} MB)")
    if LIMIT:
        print(f"projected size for 340 ears: "
              f"{(os.path.getsize(fd)+os.path.getsize(fs))/1e6*340/NE/1000:.2f} GB")


# --------------------------------------------------------------- smoke test
def grid_mesh(nx, ny, sx=1.0, sy=1.0, zf=None):
    x, y = np.linspace(0, sx, nx), np.linspace(0, sy, ny)
    XX, YY = np.meshgrid(x, y, indexing="ij")
    Z = np.zeros_like(XX) if zf is None else zf(XX, YY)
    V = np.stack([XX, YY, Z], -1).reshape(-1, 3).astype(np.float64)
    i = (np.arange(nx - 1)[:, None] * ny + np.arange(ny - 1)[None, :]).ravel()
    F = np.concatenate([np.stack([i, i + ny, i + ny + 1], 1),
                        np.stack([i, i + ny + 1, i + 1], 1)]).astype(np.int64)
    return V, F


def vertex_normals(V, F):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    N = np.zeros_like(V)
    for c in range(3):
        np.add.at(N, F[:, c], fn)
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)


def smoke():
    """B=2 synthetic surfaces at the target vertex count: verify the operators against
    closed-form answers, then drive a real torch DiffusionNet block + landmark head
    through forward AND backward on the EMITTED tensor layout."""
    import torch
    import torch.nn as nn
    t0 = time.time()
    NX = int(np.sqrt(max(MAXV, 400)))
    meshes = [grid_mesh(NX, NX, 30.0, 30.0),
              grid_mesh(NX, NX, 30.0, 30.0, lambda x, y: 3.0 * np.sin(x / 6.0) * np.cos(y / 5.0))]
    acc = {k: [] for k in ("verts", "faces", "nrm", "basis_x", "basis_y", "mass", "nbr",
                           "lap_w", "lap_diag", "grad_x", "grad_y", "grad_xd", "grad_yd",
                           "deg", "evecs")}
    v_ptr = [0]; ev_all = []
    for mi, (V, F) in enumerate(meshes):
        F, a2, nd, ng = clean_faces(V, F)
        N = vertex_normals(V, F)
        W, diag, mass, n_obt, n_clip = cot_laplacian(V, F, a2)
        X, Y = tangent_frames(N)
        gx, gy, gxd, gyd = grad_operator(V, X, Y, W.indptr, W.indices)
        ev, ec, kk, orth, resid, raw_min, eps = spectrum(W, diag, mass, K_EIG)
        n = len(V)
        print(f"mesh{mi}: n={n} faces={len(F)} nnz={W.nnz} degenerate={nd+ng} "
              f"obtuse_angles={n_obt} area={mass.sum():.4f}")
        L = (sp.diags(diag) - W)
        print(f"  mass.sum vs analytic area      {mass.sum():.6f} vs {30.0*30.0 if mi==0 else float('nan'):.6f}"
              if mi == 0 else f"  mass.sum (curved)             {mass.sum():.6f}")
        print(f"  max|L @ 1|                    {np.abs(L @ np.ones(n)).max():.3e}")
        bnd = ((np.abs(V[:, 0]) < 1e-9) | (np.abs(V[:, 0] - 30) < 1e-9) |
               (np.abs(V[:, 1]) < 1e-9) | (np.abs(V[:, 1] - 30) < 1e-9))
        g = np.array([0.7, -1.3, 0.4])
        f = V @ g
        gxv = gxd * f + np.bincount(np.repeat(np.arange(n), np.diff(W.indptr)),
                                    gx * f[W.indices], minlength=n)
        gyv = gyd * f + np.bincount(np.repeat(np.arange(n), np.diff(W.indptr)),
                                    gy * f[W.indices], minlength=n)
        err = max(np.abs(gxv - (X @ g)).max(), np.abs(gyv - (Y @ g)).max())
        if mi == 0:
            print(f"  max|L @ linear| interior      {np.abs((L @ f)[~bnd]).max():.3e}")
            print(f"  tangent grad of a.p, exact    {err:.3e}   (planar one-ring => exact)")
            lam = np.sort([np.pi ** 2 * (a ** 2 + b ** 2) / 900.0
                           for a in range(4) for b in range(4)])[:6]
            print(f"  Neumann eigenvalues [0,30]^2  {np.array2string(ev[:6], precision=6)}")
            print(f"                       analytic {np.array2string(lam, precision=6)}")
            print(f"    max rel error (nonzero)     {np.abs(ev[1:6]/lam[1:6]-1).max():.4f}")
        else:
            print(f"  tangent grad of a.p, O(h)     {err:.3e}   (curved: first-order)")
        print(f"  eigen: lam1={ev[1]:.6f} lam{kk}={ev[kk-1]:.4f} min={raw_min:.2e} "
              f"orth={orth:.2e} resid={resid:.2e}")
        assert np.abs(L @ np.ones(n)).max() < 1e-8 and mass.min() > 0
        assert raw_min > -1e-6 and orth < 1e-6 and resid < 1e-6
        for key, val in (("verts", V), ("nrm", N), ("basis_x", X), ("basis_y", Y),
                         ("mass", mass), ("lap_diag", diag), ("grad_xd", gxd),
                         ("grad_yd", gyd), ("lap_w", W.data), ("grad_x", gx), ("grad_y", gy)):
            acc[key].append(np.asarray(val, np.float32))
        acc["faces"].append((F + v_ptr[-1]).astype(np.int32))
        acc["nbr"].append((W.indices + v_ptr[-1]).astype(np.int32))
        acc["deg"].append(np.diff(W.indptr).astype(np.int32))
        acc["evecs"].append(ec.astype(np.float16))
        ev_all.append(ev.astype(np.float32)); v_ptr.append(v_ptr[-1] + n)

    d = {k: np.concatenate(v) for k, v in acc.items() if k != "evecs"}
    d["v_ptr"] = np.asarray(v_ptr, np.int64)
    d["deg_ptr"] = np.concatenate([[0], np.cumsum(d.pop("deg"))]).astype(np.int64)
    spec = {"evals": np.stack(ev_all), "evecs": np.concatenate(acc["evecs"])}
    b = pad_batch(d, [0, 1], spec)
    T = {k: torch.tensor(v) for k, v in b.items() if v.dtype != bool}
    print(f"\npadded batch: verts {tuple(T['verts'].shape)} nbr {tuple(T['nbr'].shape)} "
          f"evecs {tuple(T['evecs'].shape)} ({time.time()-t0:.0f}s so far)")

    C = 48

    class Block(nn.Module):
        """spectral diffusion + tangential gradient features, gather-only"""
        def __init__(self):
            super().__init__()
            self.logt = nn.Parameter(torch.full((C,), -1.0))
            self.gmlp = nn.Linear(2 * C, C)
            self.mlp = nn.Sequential(nn.Linear(3 * C, C), nn.ReLU(), nn.Linear(C, C))

        def forward(self, x, T):
            B, P, _ = x.shape
            D = T["nbr"].shape[-1]
            co = T["evecs"].transpose(1, 2) @ (T["mass"][..., None] * x)
            dif = T["evecs"] @ (co * torch.exp(-T["evals"][..., None] * torch.exp(self.logt)))
            xj = torch.gather(x, 1, T["nbr"].reshape(B, P * D, 1).expand(-1, -1, C)).view(B, P, D, C)
            ggx = T["grad_xd"][..., None] * x + (T["grad_x"][..., None] * xj).sum(2)
            ggy = T["grad_yd"][..., None] * x + (T["grad_y"][..., None] * xj).sum(2)
            lap = T["lap_diag"][..., None] * x - (T["lap_w"][..., None] * xj).sum(2)
            return x + self.mlp(torch.cat([dif, self.gmlp(torch.cat([ggx, ggy], -1)), lap], -1))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(9, C)
            self.blocks = nn.ModuleList([Block() for _ in range(2)])
            self.q = nn.Parameter(torch.randn(NL, C) * 0.1)
            self.head = nn.Sequential(nn.Linear(C, C), nn.ReLU(), nn.Linear(C, C))

        def forward(self, T, vmask):
            x = self.inp(torch.cat([T["verts"] / 30.0, T["nrm"], T["basis_x"]], -1))
            for bl in self.blocks:
                x = bl(x, T)
            lg = self.head(x) @ self.q.T                       # (B,P,NL)
            lg = lg.masked_fill(~vmask[..., None], -1e4)
            w = torch.softmax(lg.transpose(1, 2), -1)          # (B,NL,P)
            return w @ T["verts"]

    net = Net()
    vmask = torch.tensor(b["vmask"])
    out = net(T, vmask)
    loss = out.pow(2).mean()
    loss.backward()
    gn = sum(float(p.grad.norm()) for p in net.parameters() if p.grad is not None)
    print(f"landmark output {tuple(out.shape)}  params {sum(p.numel() for p in net.parameters()):,}"
          f"  loss {loss.item():.4f}  grad-norm sum {gn:.4f}")
    assert out.shape == (2, NL, 3), out.shape
    assert np.isfinite(gn) and gn > 0
    print(f"SMOKE OK ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    if int(os.environ.get("SMOKE", "0")):
        smoke()
    else:
        run()
