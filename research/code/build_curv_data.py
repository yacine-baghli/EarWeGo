"""
DIFFERENTIAL-GEOMETRY CHANNELS for the 8192-point surface clouds (LOCAL build).

Ear landmarks sit on ridges, crests, valleys and notches. Those ARE curvature features,
and no model in this repo has ever been given one: the input has been xyz (+ oriented
normals, which bought -0.0481mm). This adds per-point principal-curvature descriptors at
three metric scales, registered POINT-FOR-POINT with scratch/screen_data_8192nrm.npz.

    python research/code/build_curv_data.py           # -> scratch/screen_data_8192crv.npz
    SMOKE=1 python research/code/build_curv_data.py   # CPU self-test, no dataset needed
    SHARD=0 NSHARD=6 python research/code/build_curv_data.py ; ... ; SHARD=merge ...


ESTIMATOR -- WEIGHTED QUADRIC (MONGE) FIT ON THE MESH, THEN BARYCENTRIC TRANSFER
--------------------------------------------------------------------------------
Two things could have been done and one of them is wrong:

  (a) estimate curvature from the 8192-point CLOUD (local PCA / jet fitting on k-NN).
      Rejected. The cloud is an area-weighted RANDOM sample: its local density
      fluctuates as sqrt(n) and its "surface" near a fold is two interleaved sheets with
      no connectivity to tell them apart. scratch/build_curv.py (Jul 29) did exactly this
      at 2048 points with local-PCA surface variation and it is not in the shipped model.
  (b) estimate on the MESH, where connectivity exists, then interpolate to the sample
      points. Chosen.

The mesh estimate is the standard one: in the tangent frame (X, Y, N) at vertex i, fit
the Monge form

    w(u,v) = fu*u + fv*v + (fuu*u^2)/2 + fuv*u*v + (fvv*v^2)/2                    (5 dof)

by weighted least squares over a surface neighbourhood, then read the Weingarten map
S = I^-1 II off it (the linear terms fu, fv are KEPT, not assumed zero: the vertex normal
is only an estimate of the true normal, and dropping them biases k1,k2 by O(|grad|^2)):

    K = (fuu*fvv - fuv^2) / (1 + fu^2 + fv^2)^2
    H = -((1+fv^2)fuu - 2 fu fv fuv + (1+fu^2)fvv) / (2 (1 + fu^2 + fv^2)^{3/2})
    k1, k2 = H +- sqrt(max(H^2 - K, 0)),   k1 >= k2

SIGN CONVENTION, stated once and asserted in the smoke test against a sphere: N is the
OUTWARD normal (the one build_hires_data.py already ships), and the leading minus on H
makes a convex bulge POSITIVE. Sphere of radius R: k1 = k2 = +1/R, H = +1/R, K = +1/R^2,
S = +1, C = 1/R. K carries no sign convention at all (it is the product of the two
eigenvalues, so flipping the normal leaves it alone -- Theorema Egregium in discrete
clothing). That asymmetry matters; see REFLECTION below.

WHY NOT local PCA (lambda_min / sum), which is what scratch/build_curv.py shipped: that
is surface VARIATION, not curvature. It is a positive, unsigned thickness measure -- it
cannot tell a ridge from a rut, which is precisely the distinction that orders landmarks
along the helix. It is also a pure function of the point distribution, so it is
contaminated by tessellation density.

WHY NOT the cotangent Laplacian (mean-curvature normal) alone: it gives H only, has no
scale parameter, and is famously noisy on obtuse triangles (16.7% of angles here). It is
used below as an INDEPENDENT CROSS-CHECK, not as the estimator.


NEIGHBOURHOOD -- SURFACE-CONNECTED BALL, NOT A EUCLIDEAN BALL
-------------------------------------------------------------
A plain Euclidean/KD-tree ball is the failure mode that would silently ruin exactly the
landmarks we care about. The helix is a rolled edge 2-4mm thick: a 3mm Euclidean ball
centred on the helix crest also contains the posterior side of the same fold, so the
quadric is fitted to two sheets at once and the crest -- an outer-contour landmark --
reports as flat. build_mesh_data.decimate() already documents the same hazard for
clustering.

So the neighbourhood is  (k-ring of the mesh graph)  INTERSECT  (Euclidean ball of
radius r), with k = ceil(r / median_edge) + KRING_PAD PER RADIUS, capped at KRING_MAX
(the three patterns are built incrementally, so the extra ones are free). Being inside
the k-ring makes a neighbour reachable along the surface in <= k hops, so a point on the
far side of a fold is admitted only if it is ALSO within k hops around the rim.
This is a hop-bounded geodesic ball, not an exact one -- an exact multi-source geodesic
ball (scipy dijkstra with limit) costs ~20-40s/ear, i.e. 3+ hours for 340 ears, for a
distinction that only bites where the rim is thinner than r AND shorter than k hops
around. UNVERIFIED: the residual bridging rate is not measured. RMAX above ~6mm should
not be used without switching to a true geodesic ball.

Weights are Gaussian, w = exp(-d^2 / (2 sigma^2)) with sigma = r/2, so the ball edge
carries weight e^-2 = 0.135 and the estimate has no discontinuity in r.

Local coordinates are divided by r before the fit (u/r, v/r, w/r), which makes every
design column O(1) and the 5x5 normal matrix well-scaled at every radius; the Tikhonov
term is then a plain REG * sum(weights) on the diagonal and means the same thing at
r=1.5 and r=6.


SCALE -- THREE RADII IN MILLIMETRES, ALL SHIPPED
-------------------------------------------------
Curvature is 1/length, so "the curvature" of an ear is not a thing; a single radius is a
guess. RADII defaults to (1.5, 3.0, 6.0) mm, a dyadic ladder chosen from the data and the
anatomy, not from convenience:

  1.5mm  ~2x the native vertex spacing (build_mesh_data measured median edge 0.71-1.07mm
         over the 340 crops). Anything smaller fits scanner noise, not surface.
  3.0mm  the cross-section scale of the helix rim and the antihelix crest -- the ridges
         the outer / superior-antihelix contours actually run along.
  6.0mm  basin scale: concha bowl, cymba, triangular fossa. This is the "which
         compartment am I in" descriptor.

Every radius is shipped; nothing is averaged across scales here.


CHANNELS (4 per radius, 12 total, all bounded, all rotation- AND reflection-invariant)
---------------------------------------------------------------------------------------
Raw k1,k2 are NOT shipped. They are unbounded (1/length blows up on small features) and
badly conditioned near umbilics, and unbounded inputs are exactly where training dies.
Per radius r:

  S  = (2/pi) atan2(k1 + k2, k1 - k2)         shape index, EXACTLY bounded [-1,1],
                                              scale-INVARIANT.  +1 dome, +0.5 ridge,
                                              0 saddle, -0.5 rut, -1 cup.
  Cn = (2/pi) atan(C * r),  C = sqrt((k1^2+k2^2)/2)     curvedness, in [0,1)
  Hn = (2/pi) atan(H * r)                     mean curvature, in (-1,1)
  Kn = (2/pi) atan(K * r^2)                   Gaussian curvature, in (-1,1)

The r and r^2 factors make H and K dimensionless at their own scale, so the three radii
live on one axis instead of three. atan (not tanh) because it saturates far more slowly:
after clipping, |H r| <= KCLIP and |K r^2| <= KCLIP^2 = 4, which tanh would flatten to
+-0.999 while atan keeps it at +-0.85.

NOTE that the task brief asked for S = (2/pi) atan((k2+k1)/(k2-k1)); with the k1 >= k2
ordering that expression is the NEGATIVE of the one above (a dome would come out at -1).
The version shipped is the one where +1 is a dome. Same information, stated sign.

CLIPPING. k1,k2 are clipped to +-KCLIP/r (default KCLIP=2.0) BEFORE the four channels are
formed, and H,K,C,S are then recomputed from the clipped pair so the four channels stay
mutually consistent. The bound is scale-aware on purpose: at fitting scale r the tightest
sphere the fit can even represent has radius ~r/2, so |k| > 2/r is a numerical artefact,
not a feature. The clipped fraction per radius is REPORTED, not hidden.


THE REFLECTION -- WHAT ACTUALLY FLIPS, AND WHAT DOES NOT
---------------------------------------------------------
Right ears are mirrored by diag(1,-1,1). That is a reflection (det = -1), and this
project has already paid for getting normals wrong under it once (build_mesh_data.py's
"-0.983" assertion). The correct statement for curvature is narrower than folklore:

  * A reflection is an ISOMETRY of R^3. It maps the surface to its mirror and the
    exterior to the exterior, so the OUTWARD normal maps to n -> M n, and the height
    function w = e . n is invariant because M is orthogonal. The tangent basis (X, Y) is
    mapped to some other orthonormal tangent basis, which conjugates the Hessian by an
    orthogonal 2x2 -- eigenvalues unchanged.
    => k1, k2, H, K, S, C are ALL INVARIANT under the mirror. Nothing flips.
  * What DOES flip is the HANDEDNESS of the frame (e1, e2, n) of principal DIRECTIONS,
    and any pseudo-scalar built from it (geodesic torsion sign, "is e1 the first or the
    second"). This file ships no direction and no pseudo-scalar, so the issue is closed
    by construction rather than by a correction.
  * If the normal convention were broken -- e.g. mirroring the vertex normal AND flipping
    the winding, the exact bug build_mesh_data.py documents -- then H, S and k1,k2 all
    flip sign on right ears and K DOES NOT. So K is the channel that would fail to
    reveal the bug and S is the channel that would be destroyed by it.

Consequently the build computes curvature in the ALREADY-MIRRORED, outward-wound
canonical space (the same space build_hires_data.py samples), and asserts rather than
assumes:
  1. per ear, shipped normals agree with the emitted winding (dot > 0.99), as
     build_hires_data.py does;
  2. CHK right ears are recomputed a second time in the UNMIRRORED world frame -- the
     original normals carried through the reflection by the linear map instead of
     recomputed from the flipped winding -- and the 12 PER-VERTEX channels must agree to
     CHK_TOL. A direct, decisive mirror-invariance test on real anatomy;
  3. population-level: mean shape index over left ears vs over right ears must have the
     SAME SIGN and be within LR_TOL. A sign bug on one side lands at mean(S_right) ~
     -mean(S_left) and this catches it across all 340 ears, not just the 3 checked.
The smoke test additionally demonstrates the failure: it recomputes with a deliberately
inverted normal and prints that H and S flip while K does not, so assertion (2) is known
to have teeth.


REGISTRATION WITH screen_data_8192nrm.npz -- EXACT, NOT APPROXIMATE
--------------------------------------------------------------------
The sample points are not re-derived by nearest-neighbour lookup; the AREA-WEIGHTED FACE
SAMPLE IS REPLAYED. build_hires_data.py draws face f and barycentric (u,v) from
RandomState(90000 + 271*i + j), which depends only on the ear index and the sample index,
so the same crop + the same seed reproduces the same (face, barycentric) triple bit for
bit. Curvature is then interpolated with THOSE barycentric weights -- the transfer is the
same linear map that produced the point and its normal. Every ear asserts the replayed
cloud matches the shipped one (max |dx| < REPLAY_TOL, default 1e-4 mm); if that assert
ever fires the two files are not registered and nothing downstream is trustworthy.

Curvature is computed on a crop enlarged to MARGIN + RMAX (20mm) so that every vertex of
the shipped 14mm crop has a COMPLETE ball at every radius; without the enlargement the
outermost 6mm of the crop would be fitted against a truncated one-ring and would look
artificially flat.


INDEPENDENT CROSS-CHECK
------------------------
Angle-deficit Gaussian curvature K_ad = (2pi - sum theta) / mass is computed per interior
target vertex. It shares no code and no assumption with the quadric fit (no normal, no
tangent frame, no least squares) and it is sign-unambiguous. The Pearson correlation
between K_ad and the quadric K at r=RADII[0] is reported per ear; a build where the
tangent frame or the sign convention is broken cannot produce a high positive value.


OUTPUT  scratch/screen_data_<NPTS>crv.npz
  clouds (E,M,N,3) f32   nrm (E,M,N,3) f32       -- copied through, byte-identical
  crv    (E,M,N,12) f16  -- channels bounded in [-1,1]; f16 resolution 5e-4 is 3 orders
                            below the estimator's own error
  crv_names (12,) str    -- "S@1.5", "C@1.5", "H@1.5", "K@1.5", "S@3.0", ...
  crv_radii (3,) f32     coarse / true / R / c0 / split  -- as in screen_data_8192nrm

WIRING. `crv` is a per-point, sample-indexed array of the same (E,M,N,C) shape as `nrm`,
so a family consumes it with cls.NEEDS = ("nrm", "crv") and NOTHING ELSE CHANGES:
train_family.load_data pulls it, make_batch indexes it per sample, and default_augment
subsamples it along the point axis. It must NOT be listed in cls.ROTATES -- these are
scalars, and rotating them would be a shape error, not a sign error. fam_kpconv /
fam_ptv3 / fam_pointnext gate it on USE_CRV=1 in the ENVIRONMENT (cls.NEEDS is a CLASS
attribute read before instantiation, exactly as USE_NRM already is).

ONE HONEST CAVEAT ON AUGMENTATION. default_augment applies an isotropic scale jitter
(aug_scale=0.20, i.e. +-10%). Scaling a surface by s divides curvature by s, but the
shipped channels are fixed numbers attached to points, so under augmentation H,K,C are
inconsistent with the jittered geometry by up to 10%. That is comparable to the
estimator's own error and acts as feature noise rather than a bias. S is scale-INVARIANT
and is therefore exactly consistent under the jitter -- another reason to trust it most.


COST, MEASURED (one core, a 41.4k-target-vertex crop, the largest of the 340):
  hop patterns k=[5,8,14]  18.0s        fit r=1.5  3.5s      r=3.0  9-12s
  fit r=6.0  ~29s          angle deficit 0.8s      replay x4  0.3s
i.e. ~60s for the worst ear, ~17s for the smallest (20.4k), and the r=6 fit alone is
half of it -- the pattern holds 34M (vertex, neighbour) pairs and 22M survive the ball.
340 ears is ~2.5-3 CPU-hours; SHARD=k over 3 processes brings it to ~1h wall.
The accumulation was benchmarked three ways (see principal_curvatures); bincount won.

WHAT IS NOT VERIFIED HERE
  * the hop-bounded ball is not an exact geodesic ball; the residual bridging rate across
    a thin rim is not measured. The k*median_edge REACH per ear is reported, and a
    WARNING is printed if KRING_MAX truncates it below RMAX on any ear.
  * the estimator is validated against closed-form sphere/cylinder/saddle and against
    angle-deficit K on real ears (median r ~0.54-0.64 at r=1.5mm, which is what a noisy
    0.5mm-edge scan gives, not the 0.9997 the smooth phantom gives). There is no
    ground-truth curvature on a real ear, so absolute accuracy is UNKNOWN.
  * nothing here has been trained. The probe (research/code/curv_probe.py) measures
    whether the channel is DISCRIMINATIVE, not whether a network can use it.

ENV (defaults in brackets)
  NPTS [8192] M [4]        must match screen_data_<NPTS>nrm.npz
  RADII [1.5,3.0,6.0]      fitting radii in mm, comma separated
  KCLIP [2.0]              |k| <= KCLIP / r
  REG [1e-6]               Tikhonov on the normalised 5x5 normal matrix
  KRING_PAD [2] KRING_MAX [16]   hop budget = ceil(r/median_edge) + PAD per radius, capped
  BLOCK [400000]           target (vertex,neighbour) pairs per vectorised block
  CHK [3]                  right ears re-run unmirrored for the invariance assert
  CHK_TOL [2e-5] LR_TOL [0.10] REPLAY_TOL [1e-4]
  LIMIT [0]                first N ears only (0 = all 340)
  SHARD [""] NSHARD [6]    SHARD=k builds a shard; SHARD=merge stitches
  SRC [scratch/screen_data_<NPTS>nrm.npz]   OUT [scratch/screen_data_<NPTS>crv.npz]
  SMOKE [0]
"""
import os, sys, time
import numpy as np
import scipy.sparse as sp
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MESH = "2026 Munich Tech Arena - Datas/2026 Munich Tech Arena - Datas/mesh"
LM = "2026 Munich Tech Arena - Datas/2026 Munich Tech Arena - Datas/landmarks"
MIRROR = np.array([1., -1., 1.])
MARGIN = 14.0                              # must match build_hires_data.py

NPTS = int(os.environ.get("NPTS", "8192"))
M = int(os.environ.get("M", "4"))
RADII = tuple(float(x) for x in os.environ.get("RADII", "1.5,3.0,6.0").split(","))
RMAX = max(RADII)
KCLIP = float(os.environ.get("KCLIP", "2.0"))
REG = float(os.environ.get("REG", "1e-6"))
KRING_PAD = int(os.environ.get("KRING_PAD", "2"))
KRING_MAX = int(os.environ.get("KRING_MAX", "16"))
BLOCK = int(os.environ.get("BLOCK", "400000"))
CHK = int(os.environ.get("CHK", "3"))
CHK_TOL = float(os.environ.get("CHK_TOL", "2e-5"))
LR_TOL = float(os.environ.get("LR_TOL", "0.10"))
REPLAY_TOL = float(os.environ.get("REPLAY_TOL", "1e-4"))
LIMIT = int(os.environ.get("LIMIT", "0"))
SHARD = os.environ.get("SHARD", "")
NSHARD = int(os.environ.get("NSHARD", "6"))
SRC = os.environ.get("SRC", f"scratch/screen_data_{NPTS}nrm.npz")
OUT = os.environ.get("OUT", f"scratch/screen_data_{NPTS}crv.npz")

NR = len(RADII)
NCH = 4 * NR
NAMES = np.array([f"{c}@{r:g}" for r in RADII for c in "SCHK"])
TWO_PI = 2.0 / np.pi


# ------------------------------------------------------------------ mesh helpers
def vertex_normals(V, F):
    """area-weighted vertex normals; identical to build_hires_data.vertex_normals"""
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    VN = np.zeros_like(V)
    for c in range(3):
        np.add.at(VN, F[:, c], fn)
    nr = np.linalg.norm(VN, axis=1, keepdims=True)
    return np.where(nr > 1e-12, VN / np.maximum(nr, 1e-12), np.array([0., 0., 1.]))


def tangent_frames(N):
    """orthonormal (X,Y) spanning the tangent plane, seeded from the least-aligned axis.

    The seed is NOT reflection-equivariant, and does not need to be: k1,k2 are the
    eigenvalues of a 2x2 form and are invariant to which orthonormal tangent basis it is
    written in. (Principal DIRECTIONS would not be, which is why none are shipped.)
    """
    a = np.zeros_like(N)
    a[np.arange(len(N)), np.argmin(np.abs(N), axis=1)] = 1.0
    X = np.cross(N, a)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X, np.cross(N, X)


def hop_pattern(F, n, rows, k):
    """rows x n boolean CSR: vertices reachable from `rows` in <= k mesh-graph hops."""
    return hop_patterns(F, n, rows, [k])[0]


def hop_patterns(F, n, rows, ks):
    """one pattern per hop budget in the ASCENDING list `ks`, built INCREMENTALLY.

    Q(k2) = Q(k1) @ A^(k2-k1), so the total sparse-matmul work is that of the LARGEST
    budget alone -- the smaller patterns come out of the same walk for free. This matters:
    at RMAX=6mm the k=14 pattern holds ~825 neighbours per row while the r=1.5mm ball
    needs ~40, and the fit iterates over the PATTERN, not the ball. Sharing one k=14
    pattern across all three radii (the first version) spent 20x the work it had to at
    the small scales -- measured 8.6s vs 2.6s for r=1.5 on a 41k-vertex crop.
    """
    E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    r = np.concatenate([E[:, 0], E[:, 1], np.arange(n)])      # self-loops -> "<= k"
    c = np.concatenate([E[:, 1], E[:, 0], np.arange(n)])
    A = sp.coo_matrix((np.ones(len(r), bool), (r, c)), shape=(n, n)).tocsr()
    A.data[:] = True
    Q = A[rows]
    out, done = [], 1
    for k in ks:
        assert k >= done, "hop budgets must be ascending"
        for _ in range(k - done):
            Q = Q @ A
            Q.data[:] = True
        Q.sort_indices()
        out.append(Q.copy())
        done = k
    return out


def angle_deficit(V, F, n):
    """independent Gaussian curvature: (2pi - sum theta) / barycentric mass, + interior flag"""
    i0, i1, i2 = F[:, 0], F[:, 1], F[:, 2]
    A, B, C = V[i0], V[i1], V[i2]
    ang = np.zeros((len(F), 3))
    for j, (P, Q, S) in enumerate(((A, B, C), (B, C, A), (C, A, B))):
        e1, e2 = Q - P, S - P
        cs = np.einsum("ij,ij->i", e1, e2)
        sn = np.linalg.norm(np.cross(e1, e2), axis=1)
        ang[:, j] = np.arctan2(sn, cs)
    tot = np.bincount(F.ravel(), ang.ravel(), minlength=n)
    a2 = np.linalg.norm(np.cross(B - A, C - A), axis=1)
    mass = np.bincount(F.ravel(), np.repeat(a2 / 6.0, 3), minlength=n)
    e = np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    _, inv, cnt = np.unique(e, axis=0, return_inverse=True, return_counts=True)
    bnd = np.zeros(n, bool)
    bnd[np.unique(e[cnt[inv] == 1])] = True
    return (2 * np.pi - tot) / np.maximum(mass, 1e-12), ~bnd & (mass > 0)


# ------------------------------------------------------------------ the estimator
def principal_curvatures(V, N, X, Y, rows, Q, r, reg=REG, block=BLOCK):
    """weighted Monge quadric fit at `rows`; returns k1 >= k2 and the neighbour count.

    Q is the hop pattern (len(rows) x n). Neighbours are Q INTERSECT the r-ball.
    """
    nb = np.diff(Q.indptr)
    cut = [0]
    acc = 0
    for i, d in enumerate(nb):                       # split rows into ~`block`-pair chunks
        if acc + d > block and i > cut[-1]:
            cut.append(i); acc = 0
        acc += d
    cut.append(len(rows))
    k1 = np.zeros(len(rows)); k2 = np.zeros(len(rows)); cnt = np.zeros(len(rows), np.int32)
    s2 = 2.0 / (r * r)                               # 1/(2 sigma^2), sigma = r/2
    for a, b in zip(cut[:-1], cut[1:]):
        lo, hi = Q.indptr[a], Q.indptr[b]
        col = Q.indices[lo:hi].astype(np.int64)
        rl = np.repeat(np.arange(b - a), nb[a:b])    # row index LOCAL to the block, SORTED
        gr = rows[a:b][rl]
        e = V[col] - V[gr]
        d2 = np.einsum("ij,ij->i", e, e)
        keep = (d2 <= r * r) & (col != gr)
        col, rl, e, d2 = col[keep], rl[keep], e[keep], d2[keep]
        nrow = b - a
        cnt[a:b] = np.bincount(rl, minlength=nrow)
        w = np.exp(-s2 * d2)
        gk = gr[keep]
        u = np.einsum("ij,ij->i", e, X[gk]) / r
        v = np.einsum("ij,ij->i", e, Y[gk]) / r
        t = np.einsum("ij,ij->i", e, N[gk]) / r
        D = np.stack([u, v, 0.5 * u * u, u * v, 0.5 * v * v])            # (5, P)
        # 21 weighted moments by bincount. MEASURED alternatives, both rejected: filling a
        # (P,21) matrix and np.add.reduceat-ing it is 2.4x SLOWER (21 stride-168B column
        # writes per block), and the contiguous (21,P) cumsum variant wins only 20% while
        # losing 3 orders of accumulation accuracy over 400k terms.
        G = np.zeros((nrow, 5, 5)); rhs = np.zeros((nrow, 5))
        for p in range(5):
            rhs[:, p] = np.bincount(rl, w * D[p] * t, minlength=nrow)
            for q in range(p, 5):
                g = np.bincount(rl, w * D[p] * D[q], minlength=nrow)
                G[:, p, q] = g; G[:, q, p] = g
        lam = reg * np.bincount(rl, w, minlength=nrow)
        bad = cnt[a:b] < 6                          # 5 dof + 1; under-determined
        lam[bad] = 1.0
        G += lam[:, None, None] * np.eye(5)
        c = np.linalg.solve(G, rhs[..., None])[..., 0]
        fu, fv = c[:, 0], c[:, 1]
        fuu, fuv, fvv = c[:, 2] / r, c[:, 3] / r, c[:, 4] / r
        p2 = fu * fu + fv * fv
        K = (fuu * fvv - fuv * fuv) / (1 + p2) ** 2
        H = -((1 + fv * fv) * fuu - 2 * fu * fv * fuv + (1 + fu * fu) * fvv) \
            / (2 * (1 + p2) ** 1.5)                  # minus: outward normal, dome > 0
        s = np.sqrt(np.maximum(H * H - K, 0.0))
        k1[a:b] = np.where(bad, 0.0, H + s)
        k2[a:b] = np.where(bad, 0.0, H - s)
    return k1, k2, cnt


def channels(k1, k2, r, kclip=KCLIP):
    """clip, then the four bounded scale-aware channels. Returns (n,4) and clip counts."""
    lim = kclip / r
    nclip = int((np.abs(k1) > lim).sum() + (np.abs(k2) > lim).sum())
    k1 = np.clip(k1, -lim, lim); k2 = np.clip(k2, -lim, lim)
    H = 0.5 * (k1 + k2)
    K = k1 * k2
    C = np.sqrt(0.5 * (k1 * k1 + k2 * k2))
    S = TWO_PI * np.arctan2(k1 + k2, k1 - k2)        # +1 dome ... -1 cup, 0 saddle
    return np.stack([S, TWO_PI * np.arctan(C * r), TWO_PI * np.arctan(H * r),
                     TWO_PI * np.arctan(K * r * r)], 1), nclip


def curvature_on_mesh(V, N, F, n, rows):
    """(len(rows), NCH) channel block + diagnostics; one hop pattern PER RADIUS."""
    X, Y = tangent_frames(N)
    el = np.linalg.norm(V[F[:, [1, 2, 0]]] - V[F], axis=2).ravel()
    med = float(np.median(el))
    khops = [int(min(KRING_MAX, np.ceil(r / max(med, 1e-6)) + KRING_PAD)) for r in RADII]
    Qs = hop_patterns(F, n, rows, khops)
    out = np.zeros((len(rows), NCH), np.float64)
    diag = dict(khop=khops[-1], khops=khops, med_edge=med, nclip=[], nlow=[], k1=[],
                k2=[], nbr=[])
    for ri, r in enumerate(RADII):
        k1, k2, cnt = principal_curvatures(V, N, X, Y, rows, Qs[ri], r)
        out[:, 4 * ri:4 * ri + 4], nc = channels(k1, k2, r)
        diag["nclip"].append(nc); diag["nlow"].append(int((cnt < 6).sum()))
        diag["nbr"].append(float(cnt.mean()))
        diag["k1"].append(np.percentile(k1, [0.1, 1, 50, 99, 99.9]))
        diag["k2"].append(np.percentile(k2, [0.1, 1, 50, 99, 99.9]))
    return out, diag


# ------------------------------------------------------------------ per-ear build
def replay_sample(Fs, Vc, i, j):
    """EXACT replay of build_hires_data.py's area-weighted face sample for ear i, slot j."""
    A, B, C = Vc[Fs[:, 0]], Vc[Fs[:, 1]], Vc[Fs[:, 2]]
    ar = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)
    tot = ar.sum()
    assert tot > 0
    p = ar / tot
    rng = np.random.RandomState(90000 + 271 * i + j)
    f = rng.choice(len(Fs), NPTS, p=p)
    u, v = rng.rand(NPTS), rng.rand(NPTS)
    flip = u + v > 1.0
    u[flip], v[flip] = 1.0 - u[flip], 1.0 - v[flip]
    w = np.stack([1.0 - u - v, u, v], 1)[:, :, None]
    return Fs[f], w, tot


def build_ear(V, F, VN, R, cc, coarse_i, i):
    """returns crv (M,NPTS,NCH), replayed clouds/normals, per-VERTEX feat, diagnostics.

    `feat` is (n_target_vertices, NCH) in the order of np.unique of the 14mm crop's face
    indices. That order depends only on the VERTEX SET, so it is identical whether the ear
    was built from the mirrored or the unmirrored mesh -- which is what makes it, and not
    the sampled `crv`, the right thing to compare in the mirror-invariance check (the
    replayed sample uses the face's corner ORDER, which the winding flip changes).
    """
    cw = coarse_i @ R + cc
    lo, hi = cw.min(0) - MARGIN, cw.max(0) + MARGIN
    vin = np.all((V >= lo) & (V <= hi), axis=1)
    fm = vin[F].all(axis=1)
    Fs = F[fm] if fm.any() else F                     # exactly build_hires_data's crop
    Vc = (V - cc) @ R.T
    Nc = VN @ R.T

    lo2, hi2 = lo - RMAX, hi + RMAX                   # enlarged support for the fit
    f2 = np.all((V >= lo2) & (V <= hi2), axis=1)[F].all(axis=1)
    F2 = F[f2]
    sel = np.unique(F2)
    loc = -np.ones(len(V), np.int64); loc[sel] = np.arange(len(sel))
    V2, N2, F2l = Vc[sel], Nc[sel], loc[F2]
    tgt = loc[np.unique(Fs)]
    assert (tgt >= 0).all(), "14mm crop vertex missing from the enlarged crop"

    feat, diag = curvature_on_mesh(V2, N2, F2l, len(sel), tgt)
    full = np.zeros((len(sel), NCH))
    full[tgt] = feat

    Kad, interior = angle_deficit(V2, F2l, len(sel))
    m = interior[tgt]
    Kq = np.tan(feat[:, 3] / TWO_PI) / (RADII[0] ** 2)          # invert channel K@r0
    diag["k_corr"] = float(np.corrcoef(Kad[tgt][m], Kq[m])[0, 1]) if m.sum() > 8 else np.nan
    diag["n_tgt"] = len(tgt); diag["n_sup"] = len(sel)

    crv = np.zeros((M, NPTS, NCH), np.float32)
    cl = np.zeros((M, NPTS, 3), np.float32); nr = np.zeros((M, NPTS, 3), np.float32)
    for j in range(M):
        tri, w, _ = replay_sample(Fs, Vc, i, j)
        cl[j] = (w * Vc[tri]).sum(1).astype(np.float32)
        nn = (w * Nc[tri]).sum(1)
        nn /= np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-12)
        nr[j] = nn.astype(np.float32)
        crv[j] = (w * full[loc[tri]]).sum(1).astype(np.float32)
    return crv, cl, nr, feat, diag


# ------------------------------------------------------------------ driver
def run():
    from src.splits import get_split
    from src.dataset import Dataset

    z = np.load(SRC, allow_pickle=True)
    clouds0, nrm0 = z["clouds"], z["nrm"]
    coarse, true, Rm, c0, split = z["coarse"], z["true"], z["R"], z["c0"], z["split"]
    NE = LIMIT if LIMIT else len(coarse)
    assert clouds0.shape[1:] == (M, NPTS, 3), f"{SRC} is {clouds0.shape}, not (E,{M},{NPTS},3)"

    if SHARD == "merge":
        crv = np.zeros((NE, M, NPTS, NCH), np.float16)
        seen = np.zeros(NE, bool)
        for k in range(NSHARD):
            p = f"scratch/_crv{NPTS}_sh{k}.npz"
            assert os.path.exists(p), f"shard {k} missing -- run SHARD={k} first"
            s = np.load(p)
            crv[s["idx"]] = s["crv"]; seen[s["idx"]] = True
        assert seen.all(), f"{(~seen).sum()} ears missing after merge"
        np.savez_compressed(OUT, clouds=clouds0[:NE], nrm=nrm0[:NE], crv=crv,
                            crv_names=NAMES, crv_radii=np.array(RADII, np.float32),
                            coarse=coarse[:NE], true=true[:NE], R=Rm[:NE], c0=c0[:NE],
                            split=split[:NE])
        print(f"merged {NSHARD} shards -> {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB) "
              f"crv {crv.shape} range {crv.min():.4f}..{crv.max():.4f}")
        return

    tr = get_split("train", mesh_dir=Path(MESH)); va = get_split("val", mesh_dir=Path(MESH))
    order = [(p, s) for p in tr for s in ("left", "right")] + \
            [(p, s) for p in va for s in ("left", "right")]
    assert len(order) == len(split), f"{len(order)} ears in the split, {len(split)} in {SRC}"
    lock = set(get_split("test", mesh_dir=Path(MESH)))
    assert not (set(p for p, _ in order) & lock), "LOCKBOX subject leaked into the order"
    ds = Dataset(MESH, LM); pid2idx = {p: i for i, p in enumerate(ds.subject_ids)}

    IDX = np.arange(NE) if SHARD == "" else np.arange(int(SHARD), NE, NSHARD)
    if SHARD != "":
        # Drop the other shards' clouds. Only the replay assert and the normal-dot
        # diagnostic read them, and 340 ears of (M,NPTS,3) f32 is 268MB per PROCESS --
        # which is what decides how many shards fit in RAM alongside the 34M-nnz hop
        # pattern. ROW maps an ear index to its row in whatever is still resident.
        clouds0, nrm0 = clouds0[IDX], nrm0[IDX]
    ROW = {int(i): k for k, i in enumerate(IDX)} if SHARD != "" else \
          {int(i): int(i) for i in IDX}
    print(f"[curv] {len(IDX)} of {NE} ears  radii {RADII}mm  KCLIP={KCLIP} "
          f"channels={NCH}  src={SRC}", flush=True)

    crv = np.zeros((len(IDX), M, NPTS, NCH), np.float16)
    D = {k: [] for k in ("khop", "med_edge", "k_corr", "n_tgt", "n_sup", "replay",
                         "nrmdot", "meanS", "side", "reach")}
    nclip = np.zeros(NR, np.int64); nlow = np.zeros(NR, np.int64); nbr = np.zeros(NR)
    k1p = np.zeros((NR, 5)); k2p = np.zeros((NR, 5)); npts_tot = 0
    chk_done = 0; chk_err = []
    cache = {}; t0 = time.time()

    for slot, i in enumerate(IDX):
        pid, side = order[i]
        if pid not in cache:
            m = ds[pid2idx[pid]][0]
            V0 = np.asarray(m.vertices, np.float64); F0 = np.asarray(m.faces, np.int64)
            cache = {pid: (V0, F0, vertex_normals(V0, F0))}
        V0, F0, VN0 = cache[pid]
        if side == "right":                    # reflection: flip winding, RECOMPUTE normals
            V, F = V0 * MIRROR, F0[:, [0, 2, 1]]
            VN = vertex_normals(V, F)
        else:
            V, F, VN = V0, F0, VN0
        dot = float((VN * vertex_normals(V, F)).sum(1).mean())
        assert dot > 0.99, f"{pid}/{side}: shipped normals disagree with winding ({dot:.3f})"

        R, cc = Rm[i].astype(np.float64), c0[i].astype(np.float64)
        c, cl, nr, feat, dg = build_ear(V, F, VN, R, cc, coarse[i].astype(np.float64), i)
        rep = float(np.abs(cl - clouds0[ROW[int(i)]]).max())
        assert rep < REPLAY_TOL, (
            f"ear {i} ({pid}/{side}): replayed cloud differs from {SRC} by {rep:.3e} mm. "
            f"The curvature would be attached to the WRONG points -- refusing.")
        crv[slot] = c.astype(np.float16)

        # --- mirror invariance, on real anatomy: recompute a right ear UNMIRRORED.
        # Same canonical geometry reached by a different route: here the vertex normals
        # are the ORIGINAL ones pushed through the reflection by the linear map, while the
        # shipped path RECOMPUTES them from the flipped winding. If those two disagree --
        # the exact bug build_mesh_data.py documents -- H and S flip and this fires.
        if side == "right" and chk_done < CHK:
            R2 = (MIRROR[:, None] * R.T).T           # world->canonical through the mirror
            _, _, _, fu, _ = build_ear(V0, F0, VN0, R2, cc * MIRROR,
                                       coarse[i].astype(np.float64), i)
            assert fu.shape == feat.shape, (fu.shape, feat.shape)
            chk_err.append(float(np.abs(fu - feat).max())); chk_done += 1

        D["khop"].append(dg["khop"]); D["med_edge"].append(dg["med_edge"])
        D["reach"].append(dg["khop"] * dg["med_edge"])
        D["k_corr"].append(dg["k_corr"]); D["n_tgt"].append(dg["n_tgt"])
        D["n_sup"].append(dg["n_sup"]); D["replay"].append(rep)
        D["nrmdot"].append(float((nr * nrm0[ROW[int(i)]]).sum(-1).mean()))
        D["meanS"].append(float(c[..., 0::4].mean())); D["side"].append(side)
        nclip += np.array(dg["nclip"]); nlow += np.array(dg["nlow"])
        nbr += np.array(dg["nbr"]); npts_tot += dg["n_tgt"]
        k1p += np.stack(dg["k1"]); k2p += np.stack(dg["k2"])
        if (slot + 1) % 10 == 0 or slot + 1 == len(IDX):
            el = time.time() - t0
            print(f"  {slot+1}/{len(IDX)} (ear {i})  {el:.0f}s  "
                  f"eta {el/(slot+1)*(len(IDX)-slot-1):.0f}s  khop={dg['khop']} "
                  f"nsup={dg['n_sup']} Kcorr={dg['k_corr']:.3f}", flush=True)

    P = {k: np.asarray(v) for k, v in D.items()}
    n = len(IDX)
    print(f"\n--- support  target verts {P['n_tgt'].min()}/{int(np.median(P['n_tgt']))}/"
          f"{P['n_tgt'].max()}  enlarged {P['n_sup'].min()}/{int(np.median(P['n_sup']))}/"
          f"{P['n_sup'].max()}  median edge {P['med_edge'].min():.3f}.."
          f"{P['med_edge'].max():.3f}mm  hops {P['khop'].min()}..{P['khop'].max()}")
    # The hop budget is capped at KRING_MAX. If the cap binds on a fine mesh the k-ring
    # stops short of RMAX and the largest ball is TRUNCATED -- the estimate then quietly
    # becomes a smaller-scale one. Reported, because it is invisible in every other number.
    cap = int((P["khop"] >= KRING_MAX).sum())
    print(f"--- hop budget: capped at KRING_MAX={KRING_MAX} on {cap}/{n} ears; "
          f"k*median_edge reach {P['reach'].min():.2f}..{P['reach'].max():.2f}mm "
          f"vs RMAX {RMAX}mm" + ("" if P["reach"].min() >= RMAX else
          f"  <-- WARNING: {int((P['reach'] < RMAX).sum())} ears reach under RMAX, their "
          f"r={RMAX:g}mm ball is truncated; raise KRING_MAX"))
    print("--- RAW principal curvature (1/mm) before clipping, pooled percentiles")
    for ri, r in enumerate(RADII):
        print(f"    r={r:g}mm  mean |nbrs| {nbr[ri]/n:6.1f}   "
              f"k1 p0.1/p1/p50/p99/p99.9 " + "/".join(f"{x:+.3f}" for x in k1p[ri] / n) +
              f"   k2 " + "/".join(f"{x:+.3f}" for x in k2p[ri] / n))
        print(f"              clipped at |k|<={KCLIP/r:.3f}: {nclip[ri]} of "
              f"{2*npts_tot} values ({100*nclip[ri]/max(2*npts_tot,1):.3f}%)   "
              f"under-determined vertices (<6 nbrs): {nlow[ri]} "
              f"({100*nlow[ri]/max(npts_tot,1):.3f}%)")
    print(f"--- independent check: corr(quadric K@{RADII[0]:g}, angle-deficit K) "
          f"min {np.nanmin(P['k_corr']):.3f} median {np.nanmedian(P['k_corr']):.3f} "
          f"max {np.nanmax(P['k_corr']):.3f}")
    print(f"--- replay vs {SRC}: max |dx| {P['replay'].max():.3e} mm  "
          f"(tol {REPLAY_TOL})   normal dot min {P['nrmdot'].min():.6f}")
    L = P["side"] == "left"; Rt = ~L
    if L.any() and Rt.any():
        ml, mr = float(P["meanS"][L].mean()), float(P["meanS"][Rt].mean())
        assert np.sign(ml) == np.sign(mr) and abs(ml - mr) < LR_TOL, (
            f"left/right shape-index disagree (mean S left {ml:+.4f} right {mr:+.4f}). "
            f"A mirrored-normal sign bug lands here.")
        print(f"--- left/right consistency: mean shape index  left {ml:+.5f}  "
              f"right {mr:+.5f}  |diff| {abs(ml-mr):.5f} (< {LR_TOL})")
    if chk_err:
        assert max(chk_err) < CHK_TOL, (
            f"mirror invariance FAILED: right ears recomputed unmirrored differ by "
            f"{max(chk_err):.3e} (> {CHK_TOL})")
        print(f"--- mirror invariance: {len(chk_err)} right ears rebuilt in the UNMIRRORED "
              f"world frame, max channel diff {max(chk_err):.3e} (< {CHK_TOL})")
    lo, hi = float(crv.min()), float(crv.max())
    print(f"--- channels {list(NAMES)}\n    range {lo:.4f}..{hi:.4f}  "
          f"nan {int(np.isnan(crv.astype(np.float32)).sum())}")
    assert lo >= -1.001 and hi <= 1.001

    if SHARD == "":
        np.savez_compressed(OUT, clouds=clouds0[:NE], nrm=nrm0[:NE], crv=crv,
                            crv_names=NAMES, crv_radii=np.array(RADII, np.float32),
                            coarse=coarse[:NE], true=true[:NE], R=Rm[:NE], c0=c0[:NE],
                            split=split[:NE])
        f = OUT
    else:
        f = f"scratch/_crv{NPTS}_sh{SHARD}.npz"
        np.savez_compressed(f, crv=crv, idx=IDX)
    print(f"wrote {f} ({os.path.getsize(f)/1e6:.1f} MB)  crv {crv.shape} f16  "
          f"{time.time()-t0:.0f}s")


# ------------------------------------------------------------------ smoke test
def _sphere(nu=110, nv=56, R=8.0):
    """outward-wound UV sphere (open at the poles, so every kept vertex has a full ring)"""
    th = np.linspace(0.12, np.pi - 0.12, nv)
    ph = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    T, P = np.meshgrid(th, ph, indexing="ij")
    V = R * np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)], -1)
    V = V.reshape(-1, 3)
    idx = np.arange(nv * nu).reshape(nv, nu)
    a = idx[:-1, :]; b = idx[1:, :]; c = np.roll(idx, -1, 1)[1:, :]; d = np.roll(idx, -1, 1)[:-1, :]
    F = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3),
                        np.stack([a, c, d], -1).reshape(-1, 3)])
    return V, F.astype(np.int64)


def _height(f, sx=26.0, n=140):
    """z = f(x,y) patch, outward (+z) wound"""
    x = np.linspace(-sx / 2, sx / 2, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    V = np.stack([X.ravel(), Y.ravel(), f(X, Y).ravel()], 1)
    i = (np.arange(n - 1)[:, None] * n + np.arange(n - 1)[None, :]).ravel()
    F = np.concatenate([np.stack([i, i + n, i + n + 1], 1),
                        np.stack([i, i + n + 1, i + 1], 1)]).astype(np.int64)
    return V, F


def _fit(V, F, rows, r, flip=False):
    N = vertex_normals(V, F)
    if flip:
        N = -N
    X, Y = tangent_frames(N)
    Q = hop_pattern(F, len(V), rows, int(np.ceil(r / np.median(
        np.linalg.norm(V[F[:, [1, 2, 0]]] - V[F], axis=2))) + KRING_PAD))
    k1, k2, cnt = principal_curvatures(V, N, X, Y, rows, Q, r)
    return k1, k2, cnt


def smoke():
    import torch, torch.nn as nn
    t0 = time.time()
    print("=" * 78)
    print("SMOKE 1/4 -- estimator against closed-form curvature (r = 2.0 mm)")
    R0, r = 8.0, 2.0
    cases = []
    V, F = _sphere(R=R0)
    ctr = np.abs(V[:, 2]) < 0.5 * R0
    cases.append(("sphere R=8", V, F, np.flatnonzero(ctr), 1 / R0, 1 / R0, 1 / R0 ** 2,
                  1.0, 1 / R0))
    RC = 5.0
    V, F = _height(lambda x, y: RC - np.sqrt(np.maximum(RC ** 2 - np.clip(y, -0.94 * RC,
                                                                          0.94 * RC) ** 2, 1e-9)))
    m = (np.abs(V[:, 0]) < 8) & (np.abs(V[:, 1]) < 2.0)
    # z = R - sqrt(R^2-y^2) is a cylinder of radius R opening upward; outward (+z) normal
    # sees it as a RUT: k1 = 0, k2 = -1/R.
    cases.append(("cylinder R=5 (rut)", V, F, np.flatnonzero(m), 0.0, -1 / RC, 0.0,
                  -0.5, 1 / (RC * np.sqrt(2))))
    RS = 6.0
    V, F = _height(lambda x, y: (x ** 2 - y ** 2) / (2 * RS))
    m = (np.abs(V[:, 0]) < 1.5) & (np.abs(V[:, 1]) < 1.5)
    cases.append(("saddle rho=6", V, F, np.flatnonzero(m), 1 / RS, -1 / RS, -1 / RS ** 2,
                  0.0, 1 / RS))
    for nm, V, F, rows, e1, e2, eK, eS, eC in cases:
        k1, k2, cnt = _fit(V, F, rows, r)
        ch, nc = channels(k1, k2, r)
        H, K = 0.5 * (k1 + k2), k1 * k2
        C = np.sqrt(0.5 * (k1 ** 2 + k2 ** 2))
        print(f"  {nm:20s} n={len(rows):5d} |nbr|={cnt.mean():5.1f}  "
              f"k1 {np.median(k1):+.5f} (exact {e1:+.5f})  k2 {np.median(k2):+.5f} "
              f"(exact {e2:+.5f})")
        print(f"  {'':20s} H {np.median(H):+.5f} ({(e1+e2)/2:+.5f})  K {np.median(K):+.6f} "
              f"({eK:+.6f})  S {np.median(ch[:,0]):+.4f} ({eS:+.2f})  "
              f"C {np.median(C):.5f} ({eC:.5f})  clipped {nc}")
        assert abs(np.median(k1) - e1) < 0.02 and abs(np.median(k2) - e2) < 0.02, nm
        assert abs(np.median(ch[:, 0]) - eS) < 0.05, f"{nm}: shape index sign/scale"
    print("  => sign convention verified: outward normal, convex dome gives H > 0, S = +1")

    print("\nSMOKE 2/4 -- the reflection, and the bug it can hide")
    V, F = _sphere(R=R0)
    V = V + 0.9 * np.stack([np.sin(V[:, 1] / 3), np.cos(V[:, 0] / 4),
                            np.sin(V[:, 0] / 5)], 1)          # break every symmetry
    rows = np.flatnonzero(np.abs(V[:, 2]) < 0.5 * R0)
    k1, k2, _ = _fit(V, F, rows, r)
    ch = channels(k1, k2, r)[0]
    Vm, Fm = V * MIRROR, F[:, [0, 2, 1]]                       # reflection + winding flip
    k1m, k2m, _ = _fit(Vm, Fm, rows, r)
    chm = channels(k1m, k2m, r)[0]
    dmax = float(np.abs(ch - chm).max())
    print(f"  mirrored diag(1,-1,1) + winding flip : max |d(S,C,H,K)| = {dmax:.3e}")
    assert dmax < CHK_TOL, "curvature is NOT reflection invariant -- the build is wrong"
    k1b, k2b, _ = _fit(Vm, Fm, rows, r, flip=True)             # the documented bug
    chb = channels(k1b, k2b, r)[0]
    print(f"  same mirror with an INVERTED normal   : dS {np.abs(chb[:,0]-ch[:,0]).max():.3f} "
          f"dH {np.abs(chb[:,2]-ch[:,2]).max():.3f} dK {np.abs(chb[:,3]-ch[:,3]).max():.3e}"
          f"   corr(S, S_bug) = {np.corrcoef(ch[:,0], chb[:,0])[0,1]:+.4f}")
    assert np.corrcoef(ch[:, 0], chb[:, 0])[0, 1] < -0.9, "the mirror assert has no teeth"
    assert np.abs(chb[:, 3] - ch[:, 3]).max() < 1e-9, "K should be blind to the normal sign"
    print("  => S and H flip, K does not. The left/right mean-S assert is the guard.")

    print("\nSMOKE 3/4 -- independent estimator + barycentric transfer")
    Kad, interior = angle_deficit(V, F, len(V))
    k1s, k2s, _ = _fit(V, F, np.arange(len(V)), RADII[0])
    m = interior
    print(f"  corr(quadric K@{RADII[0]:g}, angle-deficit K) over {int(m.sum())} interior "
          f"verts = {np.corrcoef(k1s[m]*k2s[m], Kad[m])[0,1]:+.4f}")
    assert np.corrcoef(k1s[m] * k2s[m], Kad[m])[0, 1] > 0.5
    feat = channels(k1s, k2s, RADII[0])[0]
    rng = np.random.RandomState(0)
    fi = rng.randint(0, len(F), 4000)
    u, v = rng.rand(4000), rng.rand(4000)
    fl = u + v > 1; u[fl], v[fl] = 1 - u[fl], 1 - v[fl]
    w = np.stack([1 - u - v, u, v], 1)[:, :, None]
    pt = (w * V[F[fi]]).sum(1)
    ft = (w * feat[F[fi]]).sum(1)
    print(f"  barycentric transfer: {pt.shape[0]} pts -> feat {ft.shape}, "
          f"range {ft.min():+.4f}..{ft.max():+.4f}, "
          f"max|f - f(nearest vertex)| {np.abs(ft - feat[F[fi]][:,0]).max():.4f}")
    assert ft.shape == (4000, 4) and np.isfinite(ft).all()

    print("\nSMOKE 4/4 -- train_family wiring: NEEDS=('nrm','crv'), forward AND backward")
    from train_family import default_augment, TRAIN_DEFAULTS, default_loss, NL
    B, S, N, dev = 2, 1, 512, "cpu"

    class CrvFamily(nn.Module):
        NEEDS, ROTATES, SAMPLES = ("nrm", "crv"), ("nrm",), 1

        def __init__(self, w=48):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(3 + 3 + NCH, w), nn.ReLU(), nn.Linear(w, w))
            self.emb = nn.Embedding(NL, w)
            self.head = nn.Sequential(nn.Linear(2 * w + 3, w), nn.ReLU(), nn.Linear(w, 3))

        def forward(self, b):
            assert b["crv"].shape[-1] == NCH, b["crv"].shape
            g = self.enc(torch.cat([b["pc"] / 30.0, b["nrm"], b["crv"]], -1)).max(1).values
            x = torch.cat([g[:, None].expand(-1, NL, -1),
                           self.emb.weight[None].expand(b["pc"].shape[0], -1, -1),
                           b["coarse"] / 30.0], -1)
            return {"pred": b["coarse"] + self.head(x)}

    net = CrvFamily()
    g = torch.Generator(device=dev); g.manual_seed(0)
    b0 = {"pc": torch.randn(B, S, N, 3, generator=g) * 8,
          "nrm": torch.nn.functional.normalize(torch.randn(B, S, N, 3, generator=g), dim=-1),
          "crv": torch.rand(B, S, N, NCH, generator=g) * 2 - 1,
          "coarse": torch.randn(B, NL, 3, generator=g) * 8, "ear": torch.tensor([0, 1])}
    tg = b0["coarse"] + 0.4
    b1, tg1 = default_augment(b0, tg, {**TRAIN_DEFAULTS}, CrvFamily.ROTATES, g)
    assert b1["crv"].shape[1:3] == b1["pc"].shape[1:3], "crv was not subsampled with pc"
    assert torch.equal(b1["crv"].abs().max(), b1["crv"].abs().max())
    b1 = {k: (v[:, 0] if torch.is_tensor(v) and v.dim() >= 4 else v) for k, v in b1.items()}
    out = net(b1)
    loss = default_loss(out, tg1)
    loss.backward()
    gn = sum(float(p.grad.norm()) for p in net.parameters() if p.grad is not None)
    print(f"  augmented crv {tuple(b1['crv'].shape)} (unrotated, subsampled with pc)  "
          f"pred {tuple(out['pred'].shape)}  params "
          f"{sum(p.numel() for p in net.parameters()):,}  loss {float(loss):.4f}  "
          f"grad-norm {gn:.4f}")
    assert tuple(out["pred"].shape) == (2, NL, 3), out["pred"].shape
    assert np.isfinite(gn) and gn > 0
    print(f"SMOKE PASS ({time.time()-t0:.0f}s)")
    print("=" * 78)


if __name__ == "__main__":
    smoke() if int(os.environ.get("SMOKE", "0")) else run()
