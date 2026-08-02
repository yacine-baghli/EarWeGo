"""
HOW MUCH OF A LANDMARK'S POSITION IS IN THE LOCAL SURFACE GEOMETRY?  Training-free,
leave-one-subject-out, on the NATIVE (undecimated) crop.  Nothing here is fitted, so
nothing here can overfit; the only honesty requirement is that a descriptor taken from
the ear under test never appears in its own reference set, which is asserted.

WHY THIS AND NOT A MODEL.  Every number in this repo is "a model reached X mm".  None
tells us whether X is a modelling failure or the information content of the surface.
Huawei claim sub-0.5mm.  If the surface around landmark k does not distinguish the true
position from a position 1mm away, no architecture will, and the compute belongs on
context/targets instead of resolution.  So: measure the surface, not a network.

------------------------------------------------------------------ WHAT IS MEASURED

SURFACE.  Native cropped submesh, exactly the crop mesh_data.npz recorded
(crop_orig/c_ptr indices into the original .ply), so this is registered point-for-point
with every other artefact.  Median native vertex spacing ~0.70mm, crop area ~9300mm^2,
21.9k vertices (median).  Decimated mesh_data verts are NOT used -- the whole point is
resolution.  The surface is then sampled DETERMINISTICALLY at ~DENSE_H mm by a
barycentric lattice per face (m^2 sub-triangle centroids, m = ceil(sqrt(area)/DENSE_H)),
with barycentrically interpolated vertex normals.  Every such sample is exactly on the
piecewise-linear scanned surface.  Sub-face structure does not exist in the data, so
nothing below the ~0.70mm triangle scale is real geometry -- it is the linear
interpolant.  This bounds what the smallest descriptor scale can honestly say.

DESCRIPTOR (chosen, and why).  A multi-scale HEIGHT-OVER-TANGENT-PLANE QUADRIC.  At a
surface point p with unit normal n and tangent basis (t1,t2), for each scale
r in RADII, take the surface samples q with |q-p| <= r whose normal satisfies
n(q).n >= NDOT (drops the far sheet of a thin rim, which a Euclidean ball otherwise
swallows), set
    (u,v,h) = ((q-p).t1, (q-p).t2, (q-p).n) / r        [dimensionless]
and least-squares fit  h = c0 + c1 u + c2 v + c3 u^2 + c4 u v + c5 v^2.
Nine numbers per scale: c0..c5, the fit's RMS residual, the mean normal deviation
1 - n(q).n, and the fraction of samples with h > 0.  Four scales -> 36 dimensions.
  * Chosen over a spin image because it is DIRECTIONAL: c3,c4,c5 give both principal
    curvatures AND the principal direction, which is exactly what separates "on a crest"
    from "along a crest" -- the anisotropy that the 1.40mm tangent / 0.70mm across split
    is asking about.  A spin image is rotationally symmetric about n and throws that away.
  * Chosen over raw curvature because the residual and normal-deviation terms carry the
    roughness/anisotropy a curvature pair cannot, at no extra cost (same fit).
  * Chosen over a learned descriptor because a learned one would need folds, seeds and a
    training budget, and would confound "information present" with "model good".
  * Dimensionless by construction (u,v,h all divided by r), so the four scales are
    comparable and the concatenation is not dominated by the coarsest one.
  * t1 is the canonical-frame +Z axis projected into the tangent plane (fallback +X when
    that is under 0.1 in norm).  Legitimate here because every ear is ALREADY in the
    per-ear canonical frame that every model in this repo consumes.  It is NOT a
    rotation-invariant descriptor and does not claim to be; ROT_INV=1 zeroes the
    orientation-dependent part (c1,c2 and the off-diagonal c4, replacing c3,c5 by the two
    principal curvatures) as a sensitivity check.
Dimensions are standardised by the population mean/std of the reference descriptors
POOLED over ears and landmarks, with the test subject's two ears removed (exact rank-2
correction, asserted).  Distance is then plain L2.

(1) GEOMETRIC DISTINCTIVENESS -- can a nearest-neighbour matcher find the landmark?
For test ear e and landmark k: candidates = surface samples (subsampled to ~CAND_H mm)
within SEARCH_R of the search centre and with n(cand).n(gt) > CAND_NDOT.  References =
the descriptor at the TRUE landmark k of every ear whose subject differs from e's
(EXCL=subject; EXCL=fold additionally drops the whole frozen fold).  Two matchers:
    NN1   : pick the candidate minimising min_j ||D(cand) - D_k(j)||
    PROTO : pick the candidate minimising ||D(cand) - mean_j D_k(j)||
Reported: distance from the winning candidate to the true landmark.  Two search centres,
because they answer different questions:
    centre = model prediction (ensemble5_proj, the frozen 1.1776mm OOF) -> "could a
             descriptor matcher refine what we already have?"
    centre = true landmark                                              -> "is the
             information there at all, independent of where the model looks?"
Four reference numbers make the result readable, and NONE of them is optional:
    floor  : distance from gt to the NEAREST candidate      (candidate-grid resolution)
    chance : median distance from gt over all candidates    (what a coin flip gives)
    centre : distance from gt to the search centre          (= the model error, for pred)
    rank   : fraction of candidates whose descriptor distance beats the candidate nearest
             gt.  rank << 1 with a bad argmin means the information is present but the
             matcher's global minimum is elsewhere.
This BOUNDS NOTHING FORMALLY.  A better matcher (learned metric, context, priors) can
beat it; a worse descriptor would do worse.  It is an indicative localisation error for
one specific, honest, training-free matcher.

(2) SPATIAL FREQUENCY -- how fast does the descriptor change as you walk away?
Using the candidates within SHARP_R of the true landmark, least-squares slope through the
origin of ||D(c) - D(gt)|| against ||c - gt||:
    S       mm^-1, all directions
    S_along  restricted to |cos(dir, contour tangent)| > 0.866  (a 30 deg cone)
    S_across restricted to |cos(dir, contour tangent)| < 0.5    (a 60..90 deg cone)
and the cross-subject spread of the landmark's own descriptor,
    sigma_k = median over subject-disjoint pairs (i,j) of ||D_k(i) - D_k(j)||.
The ratio
    LAMBDA_k = sigma_k / S_k     [mm]
is the headline: HOW FAR YOU MUST MOVE ALONG THE SURFACE BEFORE THE DESCRIPTOR CHANGES
AS MUCH AS IT ALREADY DIFFERS BETWEEN TWO SUBJECTS AT THE SAME LANDMARK.  Below the
current error -> the geometry can in principle separate right from wrong at that scale.
Above it -> local geometry cannot, and context or better targets must supply the rest.
LAMBDA is DESCRIPTIVE, computed at the true landmark of the same ear; it makes no
held-out claim and none is needed, because it predicts nothing.  Its DIRECTIONAL split
is the weak part of this file and is reported as a secondary number only: in a direction
where the surface genuinely carries no signal, ||D(c)-D(gt)|| is driven by scan noise
rather than by shape, which makes S_along look larger (and LAMBDA_along smaller) than the
usable information warrants -- the synthetic self-test shows exactly this on a straight
ridge (lambda_along 0.20 vs lambda_across 0.16, a 1.25x gap, where the matcher shows
0.96mm vs 0.05mm, a 19x gap).  For direction, trust the NN1 split and the 1-D matchers,
which measure localisation itself rather than a descriptor gradient.
Then Spearman of LAMBDA (and S, sigma, and the (1) errors) against the CURRENT
per-landmark error, pooled and WITHIN CONTOUR (contour means removed), because contour
identity is a confounder for both.

(3) ANNOTATION CONSISTENCY -- what cannot be measured, and what can.
CANNOT: there are no repeat annotations in this dataset, by any annotator, on any ear.
Annotation noise is therefore NOT IDENTIFIABLE here and no number below is one.  Saying
otherwise would be inventing data.
CAN: for landmarks that sit on a genuine crest or valley, compare the GT position to the
exact extremum of the surface across the contour.  Walk the across-contour line through
gt (samples within CREST_HALF mm of gt and within CREST_BAND mm of the line), bin the
signed across-coordinate at CREST_BIN mm, and take the normal curvature in the across
direction at scale RADII[CREST_IDX],
    kappa_a(q) = (2 c3 al^2 + 2 c4 al be + 2 c5 be^2) / r,  (al,be) = (a_q.t1, a_q.t2),
with a_q the gt across-direction parallel-transported by projection into q's tangent
plane.  ridge = -kappa_a, so ridge > 0 is CONVEX (helix rim), ridge < 0 CONCAVE (fossa).
A crest is called well defined when the profile has an interior extremum of the sign of
ridge(0), with prominence >= CREST_PROM mm^-1 over the profile.  Reported per landmark:
the fraction of ears with a well-defined crest, |ridge(0)| (how much of a feature the
landmark sits on at all), and the MEAN (systematic offset) and STD (scatter) of the
signed gt->crest distance.  A mean well away from zero says the annotation protocol puts
the landmark off the crest on purpose -- a TARGET property, not a model error.  A scatter
far above the crest-detection precision says the target is loosely defined there.  This
proxy CONFOUNDS annotation scatter with crest-detection scatter and cannot separate them;
it applies only where a crest exists.
Also reported, from mesh_data: GT sits 0.021mm (mean) from the surface, p99 0.15mm, so
the annotations were placed on the surface and off-surface error is not a target issue.

------------------------------------------------------------------ WHAT IS NOT MEASURED
* No geodesics.  Distances near a landmark are Euclidean, which UNDERSTATES surface
  distance on a curved patch, so LAMBDA and the localisation errors are, if anything,
  slightly optimistic (a few % at 2-3mm on a 3mm-radius rim).
* No annotation noise (see (3)).
* No claim that a better matcher cannot beat (1).
* The lockbox is never loaded.

ENV (defaults in brackets)
  N_SCAN [80]        ears in the expensive scan; taken as whole subjects, deterministic.
                     0 = all 340.  References always use all 340 ears.
  RADII [1.5,3,4.5,6]  descriptor scales, mm
  KNB [160]          neighbours per scale (the per-scale sample spacing is r/SPACE_DIV,
                     so 160 covers 1.43 r; truncation is counted and reported)
  SPACE_DIV [5.0]    per-scale sample spacing = r / SPACE_DIV
  DENSE_H [0.28]     finest surface sample spacing, mm
  CAND_H [0.40]      candidate spacing, mm.  Sets the floor on any measured localisation
                     error at ~CAND_H/2.
  SEARCH_R [3.0]     candidate window radius, mm
  CAND_NDOT [0.3]    candidate normal gate against gt normal
  NDOT [-0.3]        descriptor-neighbour normal gate against the centre normal
  SHARP_R [2.0]      radius for the sharpness slope
  CREST_IDX [1]      which RADII entry the crest curvature uses
  CREST_HALF [3.0] CREST_BAND [0.5] CREST_BIN [0.25] CREST_PROM [0.03]
  EXCL [subject]     subject | fold  (fold additionally drops the whole frozen fold)
  ROT_INV [0]        1 = orientation-independent descriptor variant
  PRED [scratch/ensemble5_proj.npy]   frozen 1.1776mm OOF prediction, world frame
  OUT [research/results/info_limit.json]
  REF_CACHE [scratch/_info_limit_refs.npz]  pass-1 reference descriptors, reused when the
                     descriptor configuration signature matches (pass 1 costs ~10 min)
  SEED [777]  MMAX [4]  REG [1e-6]
  SMOKE [0]          1 = synthetic self-test, no dataset needed

  SMOKE=1 python research/code/info_limit.py     # ~60s CPU self-test
  N_SCAN=8 python research/code/info_limit.py    # quick real-data check
  python research/code/info_limit.py             # the real run
"""
import os, sys, json, time, warnings
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

warnings.filterwarnings("ignore", r"All-NaN|Mean of empty|Degrees of freedom")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from deep_model.surfproj import closest_on_triangles

MESH = "2026 Munich Tech Arena - Datas/2026 Munich Tech Arena - Datas/mesh"
MIRROR = np.array([1., -1., 1.])
NL = 85
CONT = [(0, 24, "outer_helix"), (25, 54, "concha"),
        (55, 74, "inner_helix"), (75, 84, "sup._antihelix")]

N_SCAN = int(os.environ.get("N_SCAN", "80"))
RADII = tuple(float(x) for x in os.environ.get("RADII", "1.5,3,4.5,6").split(","))
KNB = int(os.environ.get("KNB", "160"))
DENSE_H = float(os.environ.get("DENSE_H", "0.28"))
CAND_H = float(os.environ.get("CAND_H", "0.40"))
SEARCH_R = float(os.environ.get("SEARCH_R", "3.0"))
CAND_NDOT = float(os.environ.get("CAND_NDOT", "0.3"))
NDOT = float(os.environ.get("NDOT", "-0.3"))
SHARP_R = float(os.environ.get("SHARP_R", "2.0"))
CREST_IDX = int(os.environ.get("CREST_IDX", "1"))
CREST_HALF = float(os.environ.get("CREST_HALF", "3.0"))
CREST_BAND = float(os.environ.get("CREST_BAND", "0.5"))
CREST_BIN = float(os.environ.get("CREST_BIN", "0.25"))
CREST_PROM = float(os.environ.get("CREST_PROM", "0.03"))
EXCL = os.environ.get("EXCL", "subject")
ROT_INV = int(os.environ.get("ROT_INV", "0"))
PRED = os.environ.get("PRED", "scratch/ensemble5_proj.npy")
OUT = os.environ.get("OUT", "research/results/info_limit.json")
REF_CACHE = os.environ.get("REF_CACHE", "scratch/_info_limit_refs.npz")
SEED = int(os.environ.get("SEED", "777"))
MMAX = int(os.environ.get("MMAX", "4"))
SPACE_DIV = float(os.environ.get("SPACE_DIV", "5.0"))
REG = float(os.environ.get("REG", "1e-6"))
NS, NF = len(RADII), 9
ND = NS * NF


# ------------------------------------------------------------------ surface sampling
def face_lattice(m):
    """barycentric centroids of the m^2 sub-triangles of the order-m subdivision (m^2,3)"""
    i, j = np.meshgrid(np.arange(m), np.arange(m), indexing="ij")
    ij = np.stack([i.ravel(), j.ravel()], 1)
    up = ij[ij.sum(1) <= m - 1] + 1.0 / 3.0
    dn = ij[ij.sum(1) <= m - 2] + 2.0 / 3.0
    ab = np.concatenate([up, dn]) / m
    return np.concatenate([1.0 - ab.sum(1, keepdims=True), ab], 1)


def sample_faces(V, F, VN, h):
    """deterministic ~h-spaced samples on the piecewise-linear surface + interp normals"""
    T = V[F]
    A2 = np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    mf = np.clip(np.ceil(np.sqrt(np.maximum(A2 / 2, 1e-12)) / h).astype(np.int64), 1, MMAX)
    P, N = [], []
    for m in np.unique(mf):
        s = np.flatnonzero(mf == m)
        L = face_lattice(int(m))
        P.append(np.einsum("lb,nbc->nlc", L, V[F[s]]).reshape(-1, 3))
        N.append(np.einsum("lb,nbc->nlc", L, VN[F[s]]).reshape(-1, 3))
    P, N = np.concatenate(P), np.concatenate(N)
    return P, N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)


def tangent_basis(N):
    """t1 = canonical +Z projected into the tangent plane (fallback +X); t2 = N x t1"""
    z = np.zeros_like(N); z[:, 2] = 1.0
    t = z - N * (N[:, 2:3])
    bad = np.linalg.norm(t, axis=1) < 0.1
    if bad.any():
        x = np.zeros((int(bad.sum()), 3)); x[:, 0] = 1.0
        t[bad] = x - N[bad] * N[bad, 0:1]
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    return t, np.cross(N, t)


# ------------------------------------------------------------------ the descriptor
def describe(P, Pn, T1, T2, trees, SP, SN, chunk=6144):
    """(n,3) points with frames -> (n, NS*NF) descriptor; also returns truncation counts"""
    out = np.empty((len(P), ND))
    trunc = np.zeros(NS)
    eye = np.eye(6)
    for s, r in enumerate(RADII):
        Sp, Sn, tr = SP[s], SN[s], trees[s]
        for a in range(0, len(P), chunk):
            b = min(a + chunk, len(P))
            p, pn = P[a:b], Pn[a:b]
            d, ix = tr.query(p, k=min(KNB, len(Sp)), workers=-1)
            d, ix = np.atleast_2d(d), np.atleast_2d(ix)
            q, qn = Sp[ix], Sn[ix]
            w = ((d <= r) & (np.einsum("nkc,nc->nk", qn, pn) >= NDOT)).astype(np.float64)
            trunc[s] += float((d[:, -1] < r).sum())
            rel = (q - p[:, None]) / r
            u = np.einsum("nkc,nc->nk", rel, T1[a:b])
            v = np.einsum("nkc,nc->nk", rel, T2[a:b])
            h = np.einsum("nkc,nc->nk", rel, pn)
            X = np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], -1)
            ws = np.maximum(w.sum(1), 1.0)
            A = np.einsum("nki,nkj,nk->nij", X, X, w) + REG * ws[:, None, None] * eye
            c = np.linalg.solve(A, np.einsum("nki,nk,nk->ni", X, h, w)[..., None])[..., 0]
            res = h - np.einsum("nki,ni->nk", X, c)
            f = np.empty((b - a, NF))
            f[:, :6] = c
            f[:, 6] = np.sqrt((w * res * res).sum(1) / ws)
            f[:, 7] = (w * (1.0 - np.einsum("nkc,nc->nk", qn, pn))).sum(1) / ws
            f[:, 8] = (w * (h > 0)).sum(1) / ws
            out[a:b, s * NF:(s + 1) * NF] = f
    if ROT_INV:                                   # drop everything that needs t1's azimuth
        for s in range(NS):
            o = s * NF
            k1, k2 = principal(out[:, o + 3], out[:, o + 4], out[:, o + 5])
            out[:, o + 1] = out[:, o + 2] = out[:, o + 4] = 0.0
            out[:, o + 3], out[:, o + 5] = k1, k2
    assert np.isfinite(out).all(), "non-finite descriptor"
    return out, trunc


def principal(c3, c4, c5):
    """eigenvalues of [[2c3,c4],[c4,2c5]], larger first"""
    tr, dt = c3 + c5, np.sqrt(np.maximum((c3 - c5) ** 2 + c4 * c4, 0.0))
    return tr + dt, tr - dt


def kappa_dir(D, s, al, be):
    """normal curvature in the tangent direction (al,be) from scale-s coefficients, 1/mm"""
    o = s * NF
    return (2 * D[:, o + 3] * al * al + 2 * D[:, o + 4] * al * be
            + 2 * D[:, o + 5] * be * be) / RADII[s]


# ------------------------------------------------------------------ per-ear geometry
class Ear:
    """native crop of one ear in the canonical frame + its multi-scale sample sets"""

    def __init__(self, V, F, VN):
        F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 2] != F[:, 0])]
        T = V[F]
        a2 = np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
        self.V, self.F = V, F[a2 > 1e-12]
        self.VN = VN
        self.tree_v = cKDTree(V)
        m = len(self.F)
        self.v2f = sp.csr_matrix((np.ones(3 * m, np.int8),
                                  (self.F.ravel(), np.repeat(np.arange(m), 3))),
                                 shape=(len(V), m))
        P, N = sample_faces(V, self.F, VN, DENSE_H)
        rng = np.random.RandomState(SEED)
        perm = rng.permutation(len(P))
        self.h0 = float(np.sqrt(0.5 * a2.sum() / len(P)))
        self.SP, self.SN, self.trees = [], [], []
        for r in RADII:
            n = int(min(len(P), max(64, round(len(P) * min(1.0, (self.h0 / (r / SPACE_DIV)) ** 2)))))
            k = perm[:n]
            self.SP.append(P[k]); self.SN.append(N[k]); self.trees.append(cKDTree(P[k]))
        nc = int(min(len(P), max(64, round(len(P) * min(1.0, (self.h0 / CAND_H) ** 2)))))
        ck = perm[:nc]
        self.CP, self.CN = P[ck], N[ck]
        self.ctree = cKDTree(self.CP)

    def project(self, pts):
        """exact closest point on the crop + its interpolated normal"""
        nv = np.atleast_2d(self.tree_v.query(pts, k=min(16, len(self.V)))[1])
        Q = np.empty((len(pts), 3)); Nq = np.empty((len(pts), 3)); dd = np.empty(len(pts))
        for i, p in enumerate(pts):
            cand = np.unique(self.v2f[nv[i]].indices)
            T = self.F[cand]
            A, B, C = self.V[T[:, 0]], self.V[T[:, 1]], self.V[T[:, 2]]
            q = closest_on_triangles(p, A, B, C)
            e = np.linalg.norm(q - p, axis=1)
            j = int(e.argmin()); Q[i], dd[i] = q[j], e[j]
            a, b, c = A[j], B[j], C[j]
            nn = np.cross(b - a, c - a); den = max(float(nn @ nn), 1e-30)
            wa = float(np.cross(b - Q[i], c - Q[i]) @ nn) / den
            wb = float(np.cross(c - Q[i], a - Q[i]) @ nn) / den
            wv = np.clip([wa, wb, 1.0 - wa - wb], 0.0, 1.0); wv /= wv.sum()
            Nq[i] = wv @ self.VN[T[j]]
        return Q, Nq / np.maximum(np.linalg.norm(Nq, axis=1, keepdims=True), 1e-12), dd

    def desc_at(self, pts, nrm):
        t1, t2 = tangent_basis(nrm)
        return describe(pts, nrm, t1, t2, self.trees, self.SP, self.SN)


def load_ear(i, md, cache):
    import trimesh
    from pathlib import Path
    pid, side = str(md["pid"][i]), str(md["side"][i])
    if pid not in cache:
        cache.clear()
        m = trimesh.load(Path(MESH) / f"{pid}.ply")   # default processing, exactly
        # as src.dataset.Dataset does -- trimesh MERGES duplicate vertices there, so any
        # other load flag renumbers the vertices and breaks crop_orig (it does, on ear 123)
        V0 = np.asarray(m.vertices, np.float64); F0 = np.asarray(m.faces, np.int64)
        fn = np.cross(V0[F0[:, 1]] - V0[F0[:, 0]], V0[F0[:, 2]] - V0[F0[:, 0]])
        VN0 = np.zeros_like(V0)
        for c in range(3):
            for d in range(3):
                VN0[:, d] += np.bincount(F0[:, c], fn[:, d], minlength=len(V0))
        VN0 /= np.maximum(np.linalg.norm(VN0, axis=1, keepdims=True), 1e-12)
        cache[pid] = (V0, F0, VN0)
    V0, F0, VN0 = cache[pid]
    # a reflection reverses orientation; flipping the winding puts it back, and the
    # area-weighted normal of the flipped mirrored mesh is exactly MIRROR * VN0
    Vw, Fw, VN = ((V0 * MIRROR, F0[:, [0, 2, 1]], VN0 * MIRROR) if side == "right"
                  else (V0, F0, VN0))
    idx = md["crop_orig"][md["c_ptr"][i]:md["c_ptr"][i + 1]]
    inm = np.zeros(len(Vw), bool); inm[idx] = True
    Fk = Fw[inm[Fw].all(1)]
    keep = np.unique(Fk)
    assert np.array_equal(keep, np.sort(idx)), f"ear {i}: crop disagrees with mesh_data"
    rm = -np.ones(len(Vw), np.int64); rm[keep] = np.arange(len(keep))
    R, c0 = md["R"][i].astype(np.float64), md["c0"][i].astype(np.float64)
    return Ear((Vw[keep] - c0) @ R.T, rm[Fk], VN[keep] @ R.T)


def contour_tangent(G):
    """(85,3) unit tangent, np.gradient inside each contour"""
    T = np.zeros_like(G)
    for lo, hi, _ in CONT:
        t = np.gradient(G[lo:hi + 1], axis=0)
        T[lo:hi + 1] = t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    return T


KEYS = ["nn1_pred", "nn1_gt", "pro_pred", "pro_gt", "floor_pred", "floor_gt",
        "chance_pred", "chance_gt", "ctr_pred", "rank_pred", "rank_gt", "nn1_gt_fold",
        "nn1_gt_along", "nn1_gt_across", "nn1_pred_along", "nn1_pred_across",
        "nn1_1d_along", "nn1_1d_across", "chance_1d", "nn1_pair"]
KEYS += [f"nn1_s{j}" for j in range(NS)] + [
        "S", "S_along", "S_across", "ncand", "crest_ok", "crest_off", "ridge0"]


# ------------------------------------------------------------------ one landmark
def ear_candidates(e, G, P):
    """union over all 85 landmarks of the two SEARCH_R windows.

    The descriptor at a candidate does not depend on which landmark is being scanned
    (the frame is anchored to the canonical axes, not to the landmark), and adjacent
    landmarks on a contour sit ~2mm apart so their 3mm windows overlap heavily.  So the
    whole ear is described in ONE batched call and sliced per landmark.
    """
    loc = [np.unique(np.concatenate([e.ctree.query_ball_point(G[k], SEARCH_R),
                                     e.ctree.query_ball_point(P[k], SEARCH_R)]).astype(int))
           for k in range(len(G))]
    uni = np.unique(np.concatenate(loc))
    pos = np.full(len(e.CP), -1, np.int64); pos[uni] = np.arange(len(uni))
    D, _ = e.desc_at(e.CP[uni], e.CN[uni])
    return e.CP[uni], e.CN[uni], D, [pos[l] for l in loc]


def scan_landmark(e, g, gn, tc, pr, C0, CN0, DC0, Ref, RefF, Dg, mu, sd, res, Pair=None):
    """all three diagnostics at one true landmark of one ear.  Nothing is fitted.

    e    the Ear (native crop + sample sets)      g, gn  true landmark on the surface + normal
    tc   contour tangent (3,)                     pr     the frozen model prediction
    C0/CN0/DC0  this landmark's candidate positions, normals and RAW descriptors
    Ref  (m,ND) standardised reference descriptors from OTHER SUBJECTS at this landmark
    RefF the same with the whole frozen fold dropped (may be None)
    Dg   (ND,) standardised descriptor at the true landmark of THIS ear (descriptive only)
    Pair (1,ND) the CONTRALATERAL ear of the SAME subject, as a deliberately leaky control
    res  dict of scalars, filled in place
    """
    if len(C0) < 20:
        return
    ok = CN0 @ gn > CAND_NDOT
    if ok.sum() < 20:
        return
    C, CN, DC = C0[ok], CN0[ok], DC0[ok]
    ZC = (DC - mu) / sd
    dg = np.linalg.norm(C - g, axis=1)
    res["ncand"] = len(C)
    res["ctr_pred"] = float(np.linalg.norm(pr - g))

    tt = tc - gn * (tc @ gn)
    tt /= max(np.linalg.norm(tt), 1e-12)
    av = np.cross(gn, tt)                       # across-contour, in the tangent plane
    rel = C - g
    sa, st = rel @ av, rel @ tt                 # signed across / along coordinates

    def nn(R):
        gm = (ZC * ZC).sum(1)[:, None] + (R * R).sum(1)[None] - 2 * ZC @ R.T
        return np.sqrt(np.maximum(gm, 0)).min(1)

    dnn = nn(Ref)
    dpr = np.linalg.norm(ZC - Ref.mean(0), axis=1)
    j0 = int(dg.argmin())
    for nm, msk in (("gt", dg <= SEARCH_R),
                    ("pred", np.linalg.norm(C - pr, axis=1) <= SEARCH_R)):
        if msk.sum() < 20:
            continue
        ix = np.flatnonzero(msk)
        w = ix[dnn[ix].argmin()]
        res[f"nn1_{nm}"] = dg[w]
        res[f"nn1_{nm}_along"] = abs(st[w])
        res[f"nn1_{nm}_across"] = abs(sa[w])
        res[f"pro_{nm}"] = dg[ix[dpr[ix].argmin()]]
        res[f"floor_{nm}"] = dg[ix].min()
        res[f"chance_{nm}"] = float(np.median(dg[ix]))
        if msk[j0]:
            res[f"rank_{nm}"] = float((dnn[ix] < dnn[j0]).mean())
    ixg = np.flatnonzero(dg <= SEARCH_R)
    if len(ixg) >= 20:
        if RefF is not None:
            res["nn1_gt_fold"] = dg[ixg[nn(RefF)[ixg].argmin()]]
        # CONTROL A -- reference = the same subject's other ear.  Deliberately leaky, and
        # useless as a method; it separates "the surface is ambiguous" from "the surface
        # is fine but subjects differ too much", which nothing else here can do.
        if Pair is not None:
            res["nn1_pair"] = dg[ixg[nn(Pair)[ixg].argmin()]]
        # CONTROL B -- one scale at a time.  If the COARSE scale localises better than the
        # fine one, the usable signal is contextual (where in the ear you are), not
        # resolution, which is the whole question this workflow is asking.
        for j in range(NS):
            sl = slice(j * NF, (j + 1) * NF)
            R1, Z1 = Ref[:, sl], ZC[:, sl]
            gm = (Z1 * Z1).sum(1)[:, None] + (R1 * R1).sum(1)[None] - 2 * Z1 @ R1.T
            res[f"nn1_s{j}"] = dg[ixg[np.sqrt(np.maximum(gm, 0)).min(1)[ixg].argmin()]]

    # ---- 1-D matchers: how much information is there in ONE direction at a time?
    for nm, x, y in (("along", st, sa), ("across", sa, st)):
        m = (np.abs(y) <= CREST_BAND) & (np.abs(x) <= SEARCH_R)
        if m.sum() >= 12:
            ix = np.flatnonzero(m)
            res[f"nn1_1d_{nm}"] = abs(x[ix[dnn[ix].argmin()]])
            if nm == "across":
                res["chance_1d"] = float(np.median(np.abs(x[ix])))

    # ---- (2) how fast the descriptor changes with distance along the surface
    near = (dg <= SHARP_R) & (dg > 1e-9)
    if near.sum() >= 20:
        dd = np.linalg.norm(ZC - Dg, axis=1)
        res["S"] = (dg[near] * dd[near]).sum() / (dg[near] ** 2).sum()
        u = rel[near] - np.outer(rel[near] @ gn, gn)
        u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
        ct = np.abs(u @ tt)
        for nm, m2 in (("S_along", ct > 0.866), ("S_across", ct < 0.5)):
            if m2.sum() >= 8:
                x, y = dg[near][m2], dd[near][m2]
                res[nm] = (x * y).sum() / (x * x).sum()

    # ---- (3) crest across the contour
    band = (np.abs(st) <= CREST_BAND) & (np.abs(sa) <= CREST_HALF)
    if band.sum() < 12:
        return
    aq = av - CN[band] * (CN[band] @ av)[:, None]
    aq /= np.maximum(np.linalg.norm(aq, axis=1, keepdims=True), 1e-12)
    t1q, t2q = tangent_basis(CN[band])
    ridge = -kappa_dir(DC[band], CREST_IDX, np.einsum("nc,nc->n", aq, t1q),
                       np.einsum("nc,nc->n", aq, t2q))
    bi = np.round(sa[band] / CREST_BIN).astype(int)
    ub = np.unique(bi)
    prof = np.array([np.median(ridge[bi == u]) for u in ub])
    pos = ub * CREST_BIN
    r0 = float(prof[np.abs(pos).argmin()])
    res["ridge0"] = r0
    pv = prof if r0 >= 0 else -prof
    j = int(pv.argmax())
    res["crest_ok"] = float(0 < j < len(pv) - 1 and pv[j] - pv.min() >= CREST_PROM)
    if res["crest_ok"]:
        res["crest_off"] = pos[j]


# ------------------------------------------------------------------ the run
def run():
    t00 = time.time()
    md = np.load("scratch/mesh_data.npz")
    R, c0 = md["R"].astype(np.float64), md["c0"].astype(np.float64)
    NE = len(R)
    subj = np.arange(NE) // 2
    oz = np.load("scratch/ortho_feats.npz")
    gtw, fold = oz["gt"].astype(np.float64), oz["fold"]
    RT = np.transpose(R, (0, 2, 1))
    GT = np.einsum("ekc,ecd->ekd", gtw - c0[:, None], RT)
    PW = np.load(PRED).astype(np.float64)
    PR = np.einsum("ekc,ecd->ekd", PW - c0[:, None], RT)
    err_now = np.linalg.norm(PW - gtw, axis=2)
    rt = float(np.abs(np.einsum("ekc,ecd->ekd", GT, R) + c0[:, None] - gtw).max())
    assert rt < 1e-3, f"world<->canonical round trip off by {rt:.2e}mm"   # R is stored f32
    us = np.unique(subj)
    print(f"[info_limit] {NE} ears  desc {ND}d  scales {RADII}  KNB={KNB} "
          f"DENSE_H={DENSE_H} CAND_H={CAND_H} SEARCH_R={SEARCH_R} EXCL={EXCL} "
          f"ROT_INV={ROT_INV}", flush=True)
    print(f"             frozen prediction {PRED}: pooled {err_now.mean():.4f}mm", flush=True)

    scan_s = us if N_SCAN <= 0 else np.sort(
        np.random.RandomState(SEED).permutation(us)[:max(1, N_SCAN // 2)])
    scan = np.sort(np.concatenate([np.flatnonzero(subj == s) for s in scan_s]))
    print(f"             scan {len(scan)} ears / {len(scan_s)} subjects", flush=True)

    # ---------------- pass 1: descriptor at the true landmark of EVERY ear
    # Cached on a signature of everything that can change a reference descriptor, so a
    # rerun with a different N_SCAN / EXCL does not repay the 10 minutes.
    sigk = json.dumps([RADII, KNB, SPACE_DIV, DENSE_H, NDOT, MMAX, REG, ROT_INV, SEED, NE])
    D = GTP = None
    if os.path.exists(REF_CACHE):
        z = np.load(REF_CACHE, allow_pickle=True)
        if str(z["sig"]) == sigk:
            D, GTP, GTN, gtdist, h0, nsmp, trunc = (z["D"], z["GTP"], z["GTN"], z["gtdist"],
                                                    z["h0"], z["nsmp"], z["trunc"])
            print(f"  reusing reference descriptors from {REF_CACHE}", flush=True)
    cache, t0 = {}, time.time()
    if D is None:
        D = np.zeros((NE, NL, ND)); GTP = np.zeros((NE, NL, 3)); GTN = np.zeros((NE, NL, 3))
        gtdist = np.zeros((NE, NL)); h0 = np.zeros(NE); nsmp = np.zeros(NE)
        trunc = np.zeros(NS)
        for i in range(NE):
            e = load_ear(i, md, cache)
            q, n, d = e.project(GT[i])
            GTP[i], GTN[i], gtdist[i] = q, n, d
            D[i], tr = e.desc_at(q, n)
            trunc += tr
            h0[i], nsmp[i] = e.h0, len(e.SP[0])
            if (i + 1) % 20 == 0 or i + 1 == NE:
                el = time.time() - t0
                print(f"  refs {i+1}/{NE}  {el:.0f}s  eta {el/(i+1)*(NE-i-1):.0f}s  "
                      f"samples {len(e.SP[0])}  h0 {e.h0:.3f}mm", flush=True)
        np.savez(REF_CACHE, sig=sigk, D=D, GTP=GTP, GTN=GTN, gtdist=gtdist, h0=h0,
                 nsmp=nsmp, trunc=trunc)
    print(f"  GT -> native surface mm: mean {gtdist.mean():.4f}  p99 "
          f"{np.percentile(gtdist,99):.4f}  max {gtdist.max():.4f}  "
          f"over 0.5mm: {int((gtdist>0.5).sum())}/{NE*NL}")
    print(f"  descriptor support truncated per scale: "
          f"{np.round(trunc/(NE*NL),4).tolist()}", flush=True)

    # cross-subject descriptor spread per landmark, in standardised units
    mu, sd = D.reshape(-1, ND).mean(0), D.reshape(-1, ND).std(0) + 1e-12
    Z = (D - mu) / sd
    diff = subj[:, None] != subj[None, :]
    sig = np.zeros(NL)
    for k in range(NL):
        A = Z[:, k]
        gm = A @ A.T
        dm = np.sqrt(np.maximum(np.diag(gm)[:, None] + np.diag(gm)[None, :] - 2 * gm, 0))
        sig[k] = float(np.median(dm[diff]))

    # ---------------- pass 2: candidate scan on the held-out ears
    O = {k: np.full((len(scan), NL), np.nan) for k in KEYS}
    tot, s1 = NE * NL, D.reshape(-1, ND).sum(0)
    s2 = (D.reshape(-1, ND) ** 2).sum(0)
    t0 = time.time()
    for a, i in enumerate(scan):
        e = load_ear(i, md, cache)          # one Ear at a time: a held ear is ~25MB
        Tc = contour_tangent(GTP[i])
        keep = subj != subj[i] if EXCL == "subject" else fold != fold[i]
        keepF = fold != fold[i]
        assert not keep[i] and not keep[i ^ 1] and not keepF[i], "test ear leaked into refs"
        drop = np.flatnonzero(subj == subj[i]); dn = len(drop) * NL
        mu_e = (s1 - D[drop].reshape(-1, ND).sum(0)) / (tot - dn)
        sd_e = np.sqrt(np.maximum((s2 - (D[drop].reshape(-1, ND) ** 2).sum(0))
                                  / (tot - dn) - mu_e ** 2, 0)) + 1e-12
        Ze = (D - mu_e) / sd_e
        CP, CNn, DCa, loc = ear_candidates(e, GTP[i], PR[i])
        for k in range(NL):
            res = {}
            scan_landmark(e, GTP[i, k], GTN[i, k], Tc[k], PR[i, k], CP[loc[k]],
                          CNn[loc[k]], DCa[loc[k]], Ze[keep, k], Ze[keepF, k],
                          Ze[i, k], mu_e, sd_e, res, Ze[i ^ 1:(i ^ 1) + 1, k])
            for kk, v in res.items():
                O[kk][a, k] = v
        el = time.time() - t0
        print(f"  scan {a+1}/{len(scan)} (ear {i})  {el:.0f}s  "
              f"eta {el/(a+1)*(len(scan)-a-1):.0f}s", flush=True)

    report(O, sig, err_now, gtdist, h0, nsmp, trunc, scan, subj, NE, time.time() - t00)


# ------------------------------------------------------------------ reporting
def report(O, sig, err_now, gtdist, h0, nsmp, trunc, scan, subj, NE, secs):
    nmed = lambda a: np.nan if not np.isfinite(a).any() else float(np.nanmedian(a))
    med = {k: np.array([nmed(O[k][:, j]) for j in range(NL)]) for k in O}
    med["crest_ok"] = np.array([np.nan if not np.isfinite(O["crest_ok"][:, j]).any()
                                else float(np.nanmean(O["crest_ok"][:, j]))
                                for j in range(NL)])          # a FRACTION, not a median
    S, ek = med["S"], err_now.mean(0)
    lam = sig / np.maximum(S, 1e-12)
    lam_al = sig / np.maximum(med["S_along"], 1e-12)
    lam_ac = sig / np.maximum(med["S_across"], 1e-12)
    off_sd = np.array([np.nanstd(O["crest_off"][:, j])
                       if np.isfinite(O["crest_off"][:, j]).sum() > 3 else np.nan
                       for j in range(NL)])
    cname = np.array([nm for lo, hi, nm in CONT for _ in range(lo, hi + 1)])

    def dm(x):
        y = np.asarray(x, float).copy()
        for lo, hi, _ in CONT:
            y[lo:hi + 1] -= np.nanmean(y[lo:hi + 1])
        return y

    def cor(x, y):
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 8:
            return [None, None]
        r = spearmanr(x[m], y[m])
        return [round(float(r.statistic), 3), round(float(r.pvalue), 6)]

    print("\n" + "=" * 126)
    print("PER-LANDMARK TABLE   (mm unless stated; per-column median over the scanned ears)")
    print("  err       current frozen per-landmark error, all 340 ears")
    print("  NN1p/NN1g nearest-neighbour descriptor match, window on the prediction / on "
          "the truth")
    print("  al/ac     the NN1g error split along / across the contour")
    print("  1Dal/1Dac 1-D matcher restricted to the along- / across-contour line")
    print("  flr/chn   nearest candidate to gt (grid floor) / median candidate distance "
          "(chance)")
    print("  rnk       fraction of candidates whose descriptor beats the one nearest gt")
    print("  lam       sigma/S mm: distance you must move for the descriptor to change as "
          "much as it")
    print("            already differs between subjects.  lam_al/ac the same along/across")
    print("  |r0|      |across-contour normal curvature| at gt, 1/mm")
    print("  crst/off/sd  fraction of ears with a well-defined crest / median signed "
          "gt->crest offset / its sd")
    print("=" * 126)
    print(f"{'k':>3} {'contour':<14}{'err':>6}{'NN1p':>6}{'NN1g':>6}{'al':>5}{'ac':>5}"
          f"{'1Dal':>6}{'1Dac':>6}{'flr':>5}{'chn':>5}{'rnk':>6}{'S':>6}{'sig':>5}"
          f"{'lam':>6}{'lam_al':>7}{'lam_ac':>7}{'|r0|':>6}{'crst':>5}{'off':>6}{'sd':>5}")
    print("-" * 126)
    f6 = lambda v, w=6, p=2: (" " * w if not np.isfinite(v) else f"{v:>{w}.{p}f}")
    for k in range(NL):
        print(f"{k:>3} {cname[k]:<14}{ek[k]:>6.2f}{f6(med['nn1_pred'][k])}"
              f"{f6(med['nn1_gt'][k])}{f6(med['nn1_gt_along'][k],5)}"
              f"{f6(med['nn1_gt_across'][k],5)}{f6(med['nn1_1d_along'][k])}"
              f"{f6(med['nn1_1d_across'][k])}{f6(med['floor_gt'][k],5)}"
              f"{f6(med['chance_gt'][k],5)}{f6(med['rank_gt'][k],6,3)}{f6(S[k])}"
              f"{f6(sig[k],5)}{f6(lam[k])}{f6(lam_al[k],7)}{f6(lam_ac[k],7)}"
              f"{f6(abs(med['ridge0'][k]),6,3)}{f6(med['crest_ok'][k],5)}"
              f"{f6(med['crest_off'][k])}{f6(off_sd[k],5)}")

    print("\n" + "=" * 126)
    print("CONTOUR SUMMARY (median over the landmarks of the contour)")
    cols = ["err", "NN1p", "NN1g", "al", "ac", "1Dal", "1Dac", "flr", "chn", "rnk",
            "lam", "lam_al", "lam_ac", "|r0|", "crst", "off"]
    src = [ek, med["nn1_pred"], med["nn1_gt"], med["nn1_gt_along"], med["nn1_gt_across"],
           med["nn1_1d_along"], med["nn1_1d_across"], med["floor_gt"], med["chance_gt"],
           med["rank_gt"], lam, lam_al, lam_ac, np.abs(med["ridge0"]), med["crest_ok"],
           med["crest_off"]]
    print(f"{'contour':<16}" + "".join(f"{c:>8}" for c in cols))
    rows = {}
    for lo, hi, nmm in CONT + [(0, 84, "ALL")]:
        v = [float(np.nanmedian(x[lo:hi + 1])) for x in src]
        rows[nmm] = {c: round(w, 4) for c, w in zip(cols, v)}
        print(f"{nmm:<16}" + "".join(f"{w:>8.3f}" for w in v))

    print("\nPOOLED over every scanned (ear, landmark) -- not medians of medians")
    pool = {}
    for k in (["ctr_pred", "nn1_pred", "pro_pred", "nn1_gt", "pro_gt", "nn1_gt_fold",
               "nn1_pair", "nn1_gt_along", "nn1_gt_across", "nn1_1d_along",
               "nn1_1d_across", "chance_1d"]
              + [f"nn1_s{j}" for j in range(NS)] + ["floor_gt", "chance_gt", "rank_gt"]):
        v = O[k][np.isfinite(O[k])]
        pool[k] = dict(mean=round(float(v.mean()), 4), median=round(float(np.median(v)), 4),
                       p90=round(float(np.percentile(v, 90)), 4), n=int(len(v)))
        print(f"  {k:<15} mean {v.mean():.4f}  median {np.median(v):.4f}  "
              f"p90 {np.percentile(v,90):.4f}  n {len(v)}")
    w = np.isfinite(O["nn1_pred"]) & np.isfinite(O["ctr_pred"])
    d = O["nn1_pred"][w] - O["ctr_pred"][w]
    pool["refine_vs_frozen"] = dict(frac_improved=round(float((d < 0).mean()), 4),
                                    delta_mm=round(float(d.mean()), 4))
    print(f"  NN1 as a refiner of the frozen prediction: {(d<0).mean()*100:.1f}% of "
          f"(ear,landmark) improved, mean delta {d.mean():+.4f} mm")

    print("\nSPEARMAN AGAINST THE CURRENT PER-LANDMARK ERROR (n=85)")
    C = {}
    for nmm, x in (("lambda", lam), ("lambda_along", lam_al), ("lambda_across", lam_ac),
                   ("sharpness_S", S), ("sigma", sig), ("NN1_gt", med["nn1_gt"]),
                   ("NN1_pred", med["nn1_pred"]), ("NN1_1d_along", med["nn1_1d_along"]),
                   ("NN1_1d_across", med["nn1_1d_across"]), ("NN1_pair", med["nn1_pair"]),
                   ("rank_gt", med["rank_gt"]),
                   ("abs_ridge0", np.abs(med["ridge0"])), ("crest_ok", med["crest_ok"]),
                   ("crest_off_sd", off_sd)):
        C[nmm] = {"pooled": cor(x, ek), "within_contour": cor(dm(x), dm(ek))}
        print(f"  {nmm:<15} pooled rho {str(C[nmm]['pooled'][0]):>7} "
              f"(p={C[nmm]['pooled'][1]})   within-contour rho "
              f"{str(C[nmm]['within_contour'][0]):>7} (p={C[nmm]['within_contour'][1]})")
    rat = lam_al / lam_ac
    print(f"  lambda_along / lambda_across: median {np.nanmedian(rat):.3f}  "
          f"(>1 = less informative ALONG the contour)")
    print(f"  NN1g along / across:          median "
          f"{np.nanmedian(med['nn1_gt_along']/med['nn1_gt_across']):.3f}")

    det = np.isfinite(lam) & (lam < ek)
    und = np.isfinite(lam) & (lam > 2 * ek)
    print(f"\nGEOMETRICALLY DETERMINED (lambda below the current error)  "
          f"{int(det.sum())}/85: {np.flatnonzero(det).tolist()}")
    print(f"NOT DETERMINED (lambda above twice the current error)      "
          f"{int(und.sum())}/85: {np.flatnonzero(und).tolist()}")
    beats = np.isfinite(med["nn1_gt"]) & (med["nn1_gt"] < 0.5 * med["chance_gt"])
    print(f"NN1 beats chance by 2x at                                 "
          f"{int(beats.sum())}/85: {np.flatnonzero(beats).tolist()}")

    out = dict(
        config=dict(radii=list(RADII), knb=KNB, space_div=SPACE_DIV, dense_h=DENSE_H,
                    cand_h=CAND_H, search_r=SEARCH_R, ndot=NDOT, cand_ndot=CAND_NDOT,
                    sharp_r=SHARP_R, excl=EXCL, rot_inv=ROT_INV, pred=PRED, seed=SEED,
                    n_desc_dims=ND, crest=dict(idx=CREST_IDX, r=RADII[CREST_IDX],
                                               half=CREST_HALF, band=CREST_BAND,
                                               bin=CREST_BIN, prom=CREST_PROM)),
        n_ears=int(NE), n_scan_ears=int(len(scan)),
        scan_subjects=sorted(set(int(subj[i]) for i in scan)),
        surface=dict(sample_spacing_mm=round(float(np.median(h0)), 4),
                     samples_per_ear=int(np.median(nsmp)),
                     gt_to_surface_mm=dict(mean=round(float(gtdist.mean()), 5),
                                           p99=round(float(np.percentile(gtdist, 99)), 5),
                                           max=round(float(gtdist.max()), 5),
                                           n_over_0p5=int((gtdist > 0.5).sum())),
                     support_truncated_frac=[round(float(t / (NE * NL)), 4) for t in trunc]),
        per_landmark={k: [None if not np.isfinite(v) else round(float(v), 4) for v in arr]
                      for k, arr in dict(
                          err_now=ek, nn1_pred=med["nn1_pred"], nn1_gt=med["nn1_gt"],
                          nn1_gt_along=med["nn1_gt_along"], nn1_gt_across=med["nn1_gt_across"],
                          nn1_1d_along=med["nn1_1d_along"], nn1_1d_across=med["nn1_1d_across"],
                          pro_gt=med["pro_gt"], floor_gt=med["floor_gt"],
                          chance_gt=med["chance_gt"], rank_gt=med["rank_gt"], S=S,
                          nn1_pair=med["nn1_pair"],
                          **{f"nn1_s{j}": med[f"nn1_s{j}"] for j in range(NS)},
                          S_along=med["S_along"], S_across=med["S_across"], sigma=sig,
                          lam=lam, lam_along=lam_al, lam_across=lam_ac,
                          ridge0=med["ridge0"], crest_ok=med["crest_ok"],
                          crest_off=med["crest_off"], crest_off_sd=off_sd,
                          ncand=med["ncand"]).items()},
        per_contour=rows, pooled=pool, correlations=C,
        geometrically_determined=np.flatnonzero(det).tolist(),
        not_determined=np.flatnonzero(und).tolist(),
        caveats=[
            "Annotation noise is NOT measured and is NOT measurable here: the dataset has "
            "no repeat annotations. Section (3) is a crest-offset proxy that confounds "
            "annotation scatter with crest-detection scatter, and applies only where a "
            "crest exists.",
            "The localisation errors bound nothing formally. They are what ONE "
            "training-free nearest-neighbour matcher on ONE descriptor achieves; a learned "
            "metric, a prior or a context model can beat them.",
            "Distances near a landmark are Euclidean, not geodesic, so lambda and the "
            "localisation errors are slightly optimistic on strongly curved patches.",
            "Sub-triangle structure is the linear interpolant, not measured geometry; the "
            "smallest descriptor scale sits close to the ~0.70mm native triangle size.",
            "lambda, S and the crest statistics are DESCRIPTIVE, computed at the true "
            "landmark of the same ear. Only the section-(1) matcher is held out.",
            "The DIRECTIONAL split of lambda is noise-contaminated in the uninformative "
            "direction and understates anisotropy badly (verified on the synthetic "
            "self-test). Use nn1_gt_along/across and the 1-D matchers for direction.",
            "The scan uses a subset of ears (N_SCAN); per-landmark entries are medians "
            "over that subset, so a single per-landmark value carries the sampling error "
            "of that many observations.",
        ],
        runtime_s=round(secs, 1))
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}  ({secs/60:.1f} min)")


# ------------------------------------------------------------------ smoke test
def synth(seed):
    """a 56x56mm patch with four regions of KNOWN distinctiveness: a straight convex ridge
    (sharp across, featureless along), a genuinely empty plane, a field of isolated bumps
    (sharp in every direction), and a second ridge.  The 85 synthetic 'landmarks' sit
    exactly on the geometric feature of their region, so the ordering of every statistic
    this file computes is known in advance and can be asserted."""
    rng = np.random.RandomState(seed)
    h, half = 0.75, 28.0
    n = int(2 * half / h) + 1
    x = np.linspace(-half, half, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    p = 1.0 + 0.05 * rng.randn(4)                       # per-subject shape variation
    bumps = np.array([(a, b) for a in np.linspace(-18, 18, 5)
                      for b in (-16.0, -11.0, -6.0, -1.0)])[:20]

    def zf(XX, YY):
        Z = 3.0 * p[0] * np.exp(-((YY - 24.0) ** 2) / (2 * (1.6 * p[1]) ** 2))
        Z = Z + 2.6 * p[0] * np.exp(-((YY + 24.0) ** 2) / (2 * (1.4 * p[1]) ** 2))
        for bx, by in bumps:
            Z = Z + 2.2 * p[2] * np.exp(-(((XX - bx) ** 2 + (YY - by) ** 2)
                                          / (2 * (1.5 * p[3]) ** 2)))
        return Z

    Z = zf(X, Y) + 0.02 * rng.randn(*X.shape)           # scanner-scale noise
    V = np.stack([X, Y, Z], -1).reshape(-1, 3)
    i = (np.arange(n - 1)[:, None] * n + np.arange(n - 1)[None, :]).ravel()
    F = np.concatenate([np.stack([i, i + n, i + n + 1], 1),
                        np.stack([i, i + n + 1, i + 1], 1)]).astype(np.int64)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    VN = np.zeros_like(V)
    for c in range(3):
        for d in range(3):
            VN[:, d] += np.bincount(F[:, c], fn[:, d], minlength=len(V))
    VN /= np.maximum(np.linalg.norm(VN, axis=1, keepdims=True), 1e-12)
    G = np.zeros((NL, 3))
    G[0:25, 0] = np.linspace(-18, 18, 25); G[0:25, 1] = 24.0         # crest of ridge A
    G[25:55, 0] = np.linspace(-18, 18, 30); G[25:55, 1] = 12.0       # empty plane
    G[55:75, :2] = bumps                                             # bump apices
    G[75:85, 0] = np.linspace(-16, 16, 10); G[75:85, 1] = -24.0      # crest of ridge D
    G[:, 2] = zf(G[:, 0], G[:, 1])
    return V, F, VN, G


def smoke():
    global SEARCH_R, CREST_HALF
    t0 = time.time()
    SEARCH_R, CREST_HALF = 2.0, 2.0
    NEs = 4                                              # 2 subjects x 2 ears
    E, GTP, GTN = [], np.zeros((NEs, NL, 3)), np.zeros((NEs, NL, 3))
    D = np.zeros((NEs, NL, ND))
    for i in range(NEs):
        V, F, VN, G = synth(100 + (i // 2) * 7 + (i % 2))
        e = Ear(V, F, VN)
        q, nq, dd = e.project(G)
        E.append(e); GTP[i], GTN[i] = q, nq
        D[i], tr = e.desc_at(q, nq)
        if i == 0:
            print(f"mesh {len(V)} verts {len(F)} faces | samples {len(e.SP[0])} at "
                  f"h0={e.h0:.3f}mm | candidates {len(e.CP)} | truncated/pt "
                  f"{np.round(tr/NL,4).tolist()}")
            print(f"  GT -> surface mm max {dd.max():.4f}   descriptor {D[i].shape}")
            fl, rg = D[i, 25:55], D[i, 0:25]
            print(f"  mean |quadratic coeffs| at r={RADII[1]}mm: plane "
                  f"{np.abs(fl[:,NF+3:NF+6]).mean():.4f}   ridge "
                  f"{np.abs(rg[:,NF+3:NF+6]).mean():.4f}")
            assert np.abs(fl[:, NF + 3:NF + 6]).mean() < 0.15 * np.abs(rg[:, NF + 3:NF + 6]).mean(), \
                "the empty plane is not flat -- the synthetic layout is wrong"
    subj = np.arange(NEs) // 2
    mu, sd = D.reshape(-1, ND).mean(0), D.reshape(-1, ND).std(0) + 1e-12
    Z = (D - mu) / sd
    sig = np.array([float(np.median(np.linalg.norm(Z[subj == 0, k][:, None]
                                                   - Z[subj == 1, k][None], axis=-1)))
                    for k in range(NL)])
    i = 0
    keep = subj != subj[i]
    assert not keep[i] and not keep[i ^ 1], "test ear leaked into its own references"
    Tc = contour_tangent(GTP[i])
    O = {k: np.full((1, NL), np.nan) for k in KEYS}
    PRs = GTP[i] + 0.8 * Tc
    CP, CNn, DCa, loc = ear_candidates(E[i], GTP[i], PRs)
    print(f"  candidates: {sum(len(l) for l in loc)} landmark-slots -> {len(CP)} unique")
    for k in range(NL):
        res = {}
        scan_landmark(E[i], GTP[i, k], GTN[i, k], Tc[k], PRs[k], CP[loc[k]], CNn[loc[k]],
                      DCa[loc[k]], Z[keep, k], None, Z[i, k], mu, sd, res,
                      Z[i ^ 1:(i ^ 1) + 1, k])
        for kk, v in res.items():
            O[kk][0, k] = v
    g = lambda k: O[k][0]
    lam = sig / np.maximum(g("S"), 1e-12)
    lam_al = sig / np.maximum(g("S_along"), 1e-12)
    lam_ac = sig / np.maximum(g("S_across"), 1e-12)
    q = lambda a: "  ".join(f"{np.nanmedian(a[lo:hi+1]):6.2f}" for lo, hi in
                            ((0, 24), (25, 54), (55, 74), (75, 84)))
    print(f"\n{'':<22}{'ridgeA':>6}  {'plane':>6}  {'bumps':>6}  {'ridgeD':>6}")
    for nm, a in (("NN1 match error", g("nn1_gt")), ("  along the contour", g("nn1_gt_along")),
                  ("  across the contour", g("nn1_gt_across")),
                  ("1-D along matcher", g("nn1_1d_along")),
                  ("1-D across matcher", g("nn1_1d_across")),
                  ("grid floor", g("floor_gt")), ("chance", g("chance_gt")),
                  ("lambda", lam), ("lambda_along", lam_al), ("lambda_across", lam_ac),
                  ("NN1 vs paired ear", g("nn1_pair")),
                  *[(f"NN1 at r={RADII[j]}mm only", g(f"nn1_s{j}")) for j in range(NS)],
                  ("|ridge curvature|", np.abs(g("ridge0"))), ("crest found", g("crest_ok")),
                  ("crest offset", g("crest_off"))):
        print(f"{nm:<22}{q(a)}")
    A, P, B = slice(0, 25), slice(25, 55), slice(55, 75)
    assert np.nanmedian(g("nn1_gt")[B]) < 0.5, "a bump apex must be found to under 0.5mm"
    assert np.nanmedian(g("nn1_gt")[B]) < 0.4 * np.nanmedian(g("nn1_gt")[P]), \
        "a bump must localise far better than an empty plane"
    assert np.nanmedian(g("nn1_gt")[P]) > 0.8 * np.nanmedian(g("chance_gt")[P]), \
        "an empty plane must be no better than chance"
    assert np.nanmedian(g("nn1_gt_along")[A]) > 2 * np.nanmedian(g("nn1_gt_across")[A]), \
        "a straight ridge must localise across but not along"
    assert np.nanmedian(g("nn1_1d_across")[A]) < 0.4, "1-D across a ridge must be sharp"
    assert np.nanmedian(lam[B]) < np.nanmedian(lam[P]), "bump must be sharper than plane"
    assert np.nanmedian(g("crest_ok")[A]) == 1.0 and np.nanmedian(g("crest_ok")[P]) == 0.0, \
        "the crest test must fire on a ridge and not on a plane"
    assert abs(np.nanmedian(g("crest_off")[A])) <= CREST_BIN, \
        "a landmark placed on the crest must show ~zero crest offset"
    assert np.nanmedian(g("ridge0")[A]) > 0 and np.nanmedian(g("ridge0")[B]) > 0, \
        "convex features must give positive ridgeness"
    assert np.nanmedian(g("nn1_pair")[B]) < 0.5, "the paired-ear control must find a bump"
    assert np.isfinite(np.stack([g(f"nn1_s{j}") for j in range(NS)])).all(),         "single-scale matchers must all be defined"
    assert GTP.shape == (NEs, NL, 3) and D.shape == (NEs, NL, ND)
    print(f"\nshapes: landmarks {GTP.shape}  descriptors {D.shape}  sigma {sig.shape}  "
          f"lambda {lam.shape}  stats {len(KEYS)} x {NL}")
    print(f"SMOKE OK ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    if int(os.environ.get("SMOKE", "0")):
        smoke()
    else:
        run()
