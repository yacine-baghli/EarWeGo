"""
LOCAL preprocessing for the CASCADE stage: one high-resolution surface patch per
(ear, landmark), centred on the CURRENT PREDICTION and expressed in a per-landmark
local frame. Runs LOCALLY (needs the meshes); ships flat .npz tensors to the GPU box.

WHY THIS EXISTS
---------------
Every model in this repo predicts all 85 landmarks in one pass over the whole ear:
dgcnn at 2048 points, kpconv/ptv3 at 8192 points over a ~14 mm-margin crop. Measured
mean point spacing there is 1.09 mm (8192 pts) to 2.19 mm (2048 pts), while the native
scan has ~0.67 mm vertex spacing and the annotated landmarks lie 0.021 mm from that
surface. Precise landmark localisation is normally a CASCADE: a global stage proposes,
then each landmark is refined from a tight local crop at a resolution the global stage
cannot afford. This file builds that crop.

ACHIEVED RESOLUTION -- all numbers below are MEASURED over the 28900 patches of the
shipped scratch/local_crops.npz, not estimated.
The surface inside an 8 mm ball around a predicted landmark has median area 238.7 mm^2:
1.19x the flat disc pi*R^2 = 201 mm^2, because the ear surface is crumpled. 1024 points
spread uniformly over that would be sqrt(238.7/1024) = 0.48 mm -- 2.3x finer than the
8192-point clouds and 4.5x finer than the 2048-point ones, but NOT the 0.25 mm a
flat-disc calculation suggests. Reaching 0.25 mm uniformly needs ~3840 points per patch
and a 1.3 GB artefact.
So the sampling is STRATIFIED: CORE_FRAC of the points come from the inner CORE_R ball
and the rest from the full ball. Measured at the defaults (R=8, core 3 mm, half the
points): core spacing 0.2115 mm median (0.1132 min, 0.3074 max), surround 0.6829 mm
median (0.5578-1.0339). The fine sampling is where the accuracy is needed -- 55% of the
residuals this stage must fix are under 1 mm, 95% under 3 mm. The model is told which
stratum a point is in by the derived indicator 1[|q| < core_r] (no extra storage),
because a heatmap over a non-uniformly sampled set is otherwise biased toward the dense
region.
The number that matters most: GT lies 0.1026 mm (median) / 0.1217 mm (mean) from the
NEAREST SAMPLED PATCH POINT, p99 0.5036 mm. That is the floor of a hard argmax over
these patches, against a 1.1776 mm baseline and a 0.5 mm goal -- so the representation
is not the binding constraint, which is the whole claim this artefact exists to test.

CENTRES ARE PREDICTIONS, NOT GROUND TRUTH  (constraint 2)
---------------------------------------------------------
Each patch is centred on PRED[ear, landmark], by default scratch/ensemble5_proj.npy:
the pooled out-of-fold prediction set of the current best pipeline (equal-weight
ensemble of dgcnn+normals 3-seed / kpconv 2-seed / ptv3 2-seed, then exact surface
projection; 1.1776 mm pooled OOF over the 340 development ears, see
research/results/best_current.json). It is fold-safe BY CONSTRUCTION: ear e's entry was
produced by the base models of fold(e), i.e. by models that never trained on ear e.
Centring on GT would be a leak that no shape check catches; centring on an IN-FOLD
prediction would be worse than a leak, it would be a silent train/test mismatch -- the
refiner would learn to fix centres that are better than the ones it will meet at
inference. OOF centres give the refiner exactly the input distribution it will see.

  RESIDUAL STACKING CAVEAT, stated because it is real. For outer fold j the refiner
  trains on ears outside fold j, whose OOF centres came from base models that DID train
  on fold j. Fold j's own ears keep clean centres (their base models never saw them), so
  the evaluation ear is not leaked into its own input -- but fold j's ground truth did
  influence the centres of the TRAINING ears. This is exactly caveat 1 of
  ensemble_oof.py, the standard stacking caveat this whole repo already lives with;
  removing it needs the base models retrained inside each outer fold (25 trainings).
  It is NOT quantified here.

THE LOCAL FRAME, and why the tangent is repeatable
--------------------------------------------------
Rows of `frame[e,l]` are (t, b, n), a proper rotation (det +1, asserted):
  t  contour tangent, np.gradient over the PREDICTED landmarks of that contour,
     normalised. The 85 landmarks are 4 ordered contours and the tangent direction is
     where 77% of the remaining error energy lives (tangent RMSE 1.40 vs across 0.70,
     normal 0.27), so this axis is the one the refiner most needs aligned.
  n  the mesh normal at the surface point nearest the centre, re-orthogonalised
     against t and renormalised.
  b  n x t, so (t,b,n) is right-handed.
This is EXACTLY the convention research/code/build_ortho_feats.py already uses, so the
residuals shipped here are directly comparable to the repo's tangent/across/normal
decomposition. It is derived only from predictions and mesh geometry -- no GT.
A random tangent would force the network to be rotation-invariant about the normal for
no reason; a global reference axis projected into the tangent plane would be repeatable
but arbitrary, and would need a degeneracy fallback wherever the normal aligns with it.
The contour tangent has neither problem: measured over all 28900 patches, |dot(n_raw,t)|
is 0.031 median and 0.343 at p99, and it exceeds 0.9 on 4 patches (0.014%, all at
concha landmarks 50 and 54) with a max of 0.994. The frames stay exactly orthonormal
regardless (max |F F^T - I| = 1.2e-7, det = 1.0000000 +- 1e-7); on those 4 the b and n
axes are merely ill-CONDITIONED, meaning a small change in the predicted tangent swings
them. 4 in 28900 is not worth a special case, but it is counted and printed rather than
assumed away. FRAME=global switches to the alternative (canonical +z projected into the
tangent plane, +x fallback) for an ablation, and reports its own degeneracy rate.

PATCH EXTRACTION
----------------
Euclidean ball of radius CROP_R around the centre, in the per-ear CANONICAL frame (the
same frame as screen_data_*.npz / mesh_data.npz / train_family.py's `true`; right ears
already mirrored, so no reflection is ever seen). Faces with all three corners inside
the ball are kept, then only the CONNECTED COMPONENT containing the vertex nearest the
centre. That last step matters: an 8 mm Euclidean ball on the helix routinely swallows
the far side of the rim, which is a topological shortcut -- median 1 component but up
to 4, and in the pilot one patch had 72% of its in-ball faces on a disconnected sheet.
CC=0 disables it, and the kept-face fraction is reported either way.
Points are drawn by AREA-WEIGHTED FACE SAMPLING with a uniform barycentric point inside
the chosen face, identical to build_hires_data.py, so density is set by area and not by
tessellation. Normals are barycentrically interpolated from vertex normals and
renormalised.

PRECISION.  Coordinates and normals ship as float16. In the local frame every coordinate
is bounded by CROP_R = 8 mm; measured over the shipped file the quantisation step is
9.8e-4 mm at the median stored magnitude and 3.9e-3 mm at the largest (|q| max 7.98 mm),
and the local->canonical round trip preserves |q| to 7.5e-7 mm. Compare: the GT
landmarks sit 0.021 mm from the scanned surface (mean; 0.15 mm at p99), the patch
spacing is 0.21-0.68 mm, and the target accuracy is 0.5 mm. fp16 therefore costs about
0.001 mm -- roughly 1% of the representation floor and 0.1% of the target -- and halves
a 715 MB artefact to 358 MB. FP16=0 emits float32 if you want to check that yourself.

TARGETS ARE PER-FOLD AND ARE NOT WHAT THE TRAINER READS
--------------------------------------------------------
`local_targets_f<k>.npz` carries resid (E,85,3) = the GT-minus-centre residual expressed
in the local frame, with every VALIDATION ear of fold k set to NaN, plus `fold` and
`train_ear_mask` so train_family.py's ARTEFACTS loader can prove it fold-safe. It is
asserted here that no validation ear's residual is finite.
  These files are DIAGNOSTIC / for standalone experiments. fam_local.py does NOT read
  them, and the training path does not need them: train_family.py hands the target to
  loss() as a separate argument (`true`, canonical frame) and never puts it in the batch,
  so a family that outputs absolute canonical landmarks is leakage-free by construction
  and gets the identical gradient. Shipping the residuals as well would create a second,
  weaker path to the same numbers; they exist so the residual distribution can be
  analysed and so a non-train_family experiment has a fold-safe target file.

INFERENCE.  The crop depends on the prediction, so at test time this script must run
AGAIN on the LOCKBOX ears with PRED pointing at that stage's output. It is local mesh
work either way. Cost is 2.9 s/ear single-process (measured), so 340 ears is ~16 min.

OUTPUT (LIMIT>0 appends _lim<LIMIT> so a smoke run never clobbers the real artefact)
  scratch/local_crops.npz        pts/nrm (E,85,N,3) f16 LOCAL frame; centre (E,85,3) and
                                 frame (E,85,3,3) f32 CANONICAL; per-patch diagnostics
  scratch/local_targets_f<k>.npz resid (E,85,3) local frame, NaN on fold-k val ears
Sizes as built: crops 357.7 MB (fp16; ~715 MB at FP16=0), targets 1.7 MB each. Build
time 12.9 min wall for 340 ears in 4 sequential shards of 89.3 MB each.

ENV (defaults in brackets)
  PRED [scratch/ensemble5_proj.npy]  WORLD-frame (E,85,3) prediction set to centre on
  CROP_R [8.0]     ball radius mm. 8.0 puts the true landmark inside the patch for
                   99.955% of the 28900 (ear,landmark) pairs (measured on PRED); 6.0
                   gives 99.72%, 5.0 gives 99.32%, 4.0 gives 98.2%.
  NPTS [1024]      points per patch
  CORE_R [3.0]     inner stratum radius mm      CORE_FRAC [0.5]  share of points in it
  CC [1]           keep only the seed-connected component of the in-ball faces
  FRAME [contour]  contour | global            FP16 [1]
  MARGIN [12.0]    per-ear pre-crop margin around the prediction, mm
  SEED [90210]     sampler seed (per-ear stream = SEED + 9973*ear)
  SHARD ['']       k | merge                   NSHARD [4]
  LIMIT [0]        first N ears only           OUT [scratch/local]
  TARGETS [1]      also write the 5 per-fold residual target files
  ANALYSE [1]      print the reachability / representation-floor analysis (uses GT for
                   REPORTING only, over all ears; nothing GT-derived is written outside
                   the per-fold masked target files)

  python research/code/build_local_crops.py                 # all 340 ears
  SHARD=0 NSHARD=4 python research/code/build_local_crops.py; ... ; SHARD=merge ...
  LIMIT=3 python research/code/build_local_crops.py         # 3-ear check
  SMOKE=1 python research/code/build_local_crops.py         # CPU self-test, no data
"""
import os, sys, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MESH = "2026 Munich Tech Arena - Datas/2026 Munich Tech Arena - Datas/mesh"
LM = "2026 Munich Tech Arena - Datas/2026 Munich Tech Arena - Datas/landmarks"
MIRROR = np.array([1., -1., 1.])
NL, NFOLD = 85, 5
CONT = [(0, 24), (25, 54), (55, 74), (75, 84)]

E = os.environ.get
PRED = E("PRED", "scratch/ensemble5_proj.npy")
CROP_R = float(E("CROP_R", "8.0"))
NPTS = int(E("NPTS", "1024"))
CORE_R = float(E("CORE_R", "3.0"))
CORE_FRAC = float(E("CORE_FRAC", "0.5"))
CC = int(E("CC", "1"))
FRAME = E("FRAME", "contour")
FP16 = int(E("FP16", "1"))
MARGIN = float(E("MARGIN", "12.0"))
SEED = int(E("SEED", "90210"))
SHARD = E("SHARD", "")
NSHARD = int(E("NSHARD", "4"))
LIMIT = int(E("LIMIT", "0"))
OUT = E("OUT", "scratch/local")
TARGETS = int(E("TARGETS", "1"))
ANALYSE = int(E("ANALYSE", "1"))
assert FRAME in ("contour", "global"), FRAME
assert 0.0 <= CORE_FRAC <= 1.0 and 0 < CORE_R <= CROP_R
DT = np.float16 if FP16 else np.float32
NCORE = int(round(NPTS * CORE_FRAC))


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


# ------------------------------------------------------------------ geometry
def vertex_normals(V, F):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    N = np.zeros_like(V)
    for c in range(3):
        np.add.at(N, F[:, c], fn)
    n = np.linalg.norm(N, axis=1, keepdims=True)
    return np.where(n > 1e-12, N / np.maximum(n, 1e-12), np.array([0., 0., 1.]))


def local_frames(P, N_at, mode=FRAME):
    """(85,3,3) rows (t,b,n). P = predicted landmarks, N_at = mesh normal at each.

    contour: t from np.gradient along the predicted contour, n = Gram-Schmidt of the mesh
             normal against t, b = n x t.
    global : t = canonical +z projected into the tangent plane (+x where that degenerates),
             n = the mesh normal, b = n x t. Kept only as the ablation the docstring names.
    Returns the frames plus diagnostics (|dot(n_raw,t)| and the count of degenerate t).
    """
    Fr = np.zeros((NL, 3, 3))
    raw = np.zeros(NL)
    ndeg = 0
    n0 = N_at / np.maximum(np.linalg.norm(N_at, axis=1, keepdims=True), 1e-12)
    if mode == "contour":
        for a, b in CONT:
            t = np.gradient(P[a:b + 1], axis=0)
            ln = np.linalg.norm(t, axis=1, keepdims=True)
            ndeg += int((ln < 1e-9).sum())
            t = t / np.maximum(ln, 1e-12)
            n = n0[a:b + 1]
            raw[a:b + 1] = np.abs((n * t).sum(1))
            n = n - (n * t).sum(1, keepdims=True) * t
            n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
            Fr[a:b + 1, 0], Fr[a:b + 1, 2] = t, n
            Fr[a:b + 1, 1] = np.cross(n, t)
    else:
        z = np.zeros((NL, 3)); z[:, 2] = 1.0
        raw = np.abs(n0[:, 2])
        bad = raw > 0.99
        ndeg = int(bad.sum())
        z[bad] = np.array([1.0, 0.0, 0.0])
        t = z - (z * n0).sum(1, keepdims=True) * n0
        t = t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
        Fr[:, 0], Fr[:, 2], Fr[:, 1] = t, n0, np.cross(n0, t)
    det = np.linalg.det(Fr)
    assert np.abs(det - 1.0).max() < 1e-6, f"frame is not a proper rotation (det {det.min()}..{det.max()})"
    gram = np.einsum("lij,likj->lik", Fr, Fr[:, None])
    assert np.abs(gram - np.eye(3)).max() < 1e-9, "frame is not orthonormal"
    return Fr, raw, ndeg


def sample_faces(V, Fs, npts, rng):
    """area-weighted face choice + uniform barycentric point. -> (npts,3) weights, faces"""
    A, B, C = V[Fs[:, 0]], V[Fs[:, 1]], V[Fs[:, 2]]
    ar = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)
    tot = float(ar.sum())
    f = rng.choice(len(Fs), npts, p=ar / tot)
    u, v = rng.rand(npts), rng.rand(npts)
    fl = u + v > 1.0
    u[fl], v[fl] = 1.0 - u[fl], 1.0 - v[fl]
    return np.stack([1.0 - u - v, u, v], 1)[:, :, None], Fs[f], tot


def patch(V, F, VN, tree, c, rng):
    """One patch around canonical-frame point c. -> pts, nrms (world-canonical), stats."""
    idx = tree.query_ball_point(c, CROP_R)
    if len(idx) < 8:
        return None
    inb = np.zeros(len(V), bool); inb[np.asarray(idx)] = True
    Fs = F[inb[F].all(1)]
    if len(Fs) < 8:
        return None
    n_in = len(Fs)
    ncc = 1
    if CC:
        vs = np.unique(Fs)
        loc = -np.ones(len(V), np.int64); loc[vs] = np.arange(len(vs))
        ed = np.vstack([Fs[:, [0, 1]], Fs[:, [1, 2]], Fs[:, [2, 0]]])
        A = sp.coo_matrix((np.ones(len(ed), np.int8), (loc[ed[:, 0]], loc[ed[:, 1]])),
                          shape=(len(vs), len(vs)))
        ncc, lab = connected_components(A, directed=False)
        if ncc > 1:
            seed = int(vs[np.argmin(np.linalg.norm(V[vs] - c, axis=1))])
            Fs2 = Fs[lab[loc[Fs[:, 0]]] == lab[loc[seed]]]
            if len(Fs2) >= 8:
                Fs = Fs2
    w, tri, area = sample_faces(V, Fs, NPTS - NCORE, rng)
    pts = [(w * V[tri]).sum(1)]
    nrs = [(w * VN[tri]).sum(1)]
    core_area = 0.0
    if NCORE:
        Fc = Fs[(np.linalg.norm(V[Fs] - c, axis=2) <= CORE_R).all(1)]
        if len(Fc) < 8:                      # core degenerate: draw from the full ball
            Fc = Fs
        w, tri, core_area = sample_faces(V, Fc, NCORE, rng)
        pts.append((w * V[tri]).sum(1))
        nrs.append((w * VN[tri]).sum(1))
    p = np.concatenate(pts); n = np.concatenate(nrs)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    return p, n, dict(area=area, core_area=core_area, n_face=len(Fs),
                      n_cc=ncc, keep_frac=len(Fs) / n_in)


# ------------------------------------------------------------------ driver
def run():
    from pathlib import Path
    from src.splits import get_split
    from src.dataset import Dataset

    d0 = np.load("scratch/deep_dataset.npz", allow_pickle=True)
    Rm, c0, TRUE = (d0["R"].astype(np.float64), d0["c0"].astype(np.float64),
                    d0["true"].astype(np.float64))
    Pw = np.load(PRED).astype(np.float64)
    NE_ALL = len(Pw)
    assert Pw.shape == (NE_ALL, NL, 3) and len(Rm) == NE_ALL, Pw.shape
    # WORLD -> CANONICAL. world = canon @ R + c0 with R a proper rotation, so the inverse
    # is (world - c0) @ R.T and the map is an isometry: every mm quoted here is a mm.
    Pc = np.stack([(Pw[i] - c0[i]) @ Rm[i].T for i in range(NE_ALL)])
    NE = LIMIT if LIMIT else NE_ALL

    tr = get_split("train", mesh_dir=Path(MESH)); va = get_split("val", mesh_dir=Path(MESH))
    order = [(p, s) for p in tr for s in ("left", "right")] + \
            [(p, s) for p in va for s in ("left", "right")]
    assert len(order) == NE_ALL, f"{len(order)} ears in the split order, {NE_ALL} in {PRED}"
    lock = set(get_split("test", mesh_dir=Path(MESH)))
    assert not (set(p for p, _ in order) & lock), "LOCKBOX subject leaked into the order"
    ds = Dataset(MESH, LM); pid2idx = {p: i for i, p in enumerate(ds.subject_ids)}

    IDX = np.arange(NE) if SHARD == "" else np.arange(int(SHARD), NE, NSHARD)
    print(f"[local_crops] {len(IDX)} of {NE} ears | R={CROP_R} NPTS={NPTS} "
          f"core {NCORE}@{CORE_R}mm | frame={FRAME} cc={CC} fp16={FP16} | centres {PRED}",
          flush=True)

    pts = np.zeros((len(IDX), NL, NPTS, 3), DT)
    nrm = np.zeros((len(IDX), NL, NPTS, 3), DT)
    cen = np.zeros((len(IDX), NL, 3), np.float32)
    frm = np.zeros((len(IDX), NL, 3, 3), np.float32)
    cdist = np.zeros((len(IDX), NL), np.float32)
    st = {k: np.zeros((len(IDX), NL), np.float32)
          for k in ("area", "core_area", "n_face", "n_cc", "keep_frac")}
    fdeg = np.zeros(len(IDX), np.int32)
    fraw = np.zeros((len(IDX), NL), np.float32)
    nfail = 0
    cache, t0 = {}, time.time()

    for slot, i in enumerate(IDX):
        pid, side = order[i]
        if pid not in cache:
            m = ds[pid2idx[pid]][0]
            cache = {pid: (np.asarray(m.vertices, np.float64),
                           np.asarray(m.faces, np.int64))}
        V0, F0 = cache[pid]
        if side == "right":                  # reflection: flip winding, recompute normals
            Vw, Fw = V0 * MIRROR, F0[:, [0, 2, 1]]
        else:
            Vw, Fw = V0, F0
        R, cc = Rm[i], c0[i]
        assert abs(np.linalg.det(R) - 1.0) < 1e-5, f"ear {i}: R is not a proper rotation"
        Vc = (Vw - cc) @ R.T                 # canonical frame; rotation preserves area
        P = Pc[i]
        lo, hi = P.min(0) - MARGIN, P.max(0) + MARGIN
        vin = np.all((Vc >= lo) & (Vc <= hi), axis=1)
        Fk = Fw[vin[Fw].all(1)]
        assert len(Fk) > 100, f"ear {i}: pre-crop kept only {len(Fk)} faces"
        keep = np.unique(Fk)
        rmv = -np.ones(len(Vc), np.int64); rmv[keep] = np.arange(len(keep))
        V, F = Vc[keep], rmv[Fk]
        VN = vertex_normals(V, F)            # recomputed in the shipped space, both sides
        fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
        wd = float((fn * VN[F].mean(1)).sum(1).mean())
        assert wd > 0.5, f"ear {i}: crop winding disagrees with its normals ({wd:.3f})"

        tree = cKDTree(V)
        dnn, jnn = tree.query(P)
        cdist[slot] = dnn
        Fr, raw, nd = local_frames(P, VN[jnn])
        frm[slot], fraw[slot], fdeg[slot], cen[slot] = Fr, raw, nd, P
        for l in range(NL):
            rng = np.random.RandomState(SEED + 9973 * int(i) + 101 * l)
            r = patch(V, F, VN, tree, P[l], rng)
            if r is None:
                nfail += 1
                continue
            p, n, s = r
            # LOCAL frame: q = (x - c) @ F^T, rows of F are (t,b,n). Pure rotation, so the
            # normals take the same map and stay unit.
            pts[slot, l] = ((p - P[l]) @ Fr[l].T).astype(DT)
            nrm[slot, l] = (n @ Fr[l].T).astype(DT)
            for k in st:
                st[k][slot, l] = s[k]
        if (slot + 1) % 20 == 0 or slot + 1 == len(IDX):
            el = time.time() - t0
            print(f"  {slot+1}/{len(IDX)} (ear {i})  {el:.0f}s  "
                  f"eta {el/(slot+1)*(len(IDX)-slot-1):.0f}s", flush=True)

    assert nfail == 0, f"{nfail} patches had too little surface -- inspect before shipping"
    sfx = f"_lim{LIMIT}" if LIMIT else ""
    if SHARD != "":
        f = f"scratch/_lcrop{sfx}_sh{SHARD}.npz"
        np.savez(f, idx=IDX, pts=pts, nrm=nrm, centre=cen, frame=frm, cdist=cdist,
                 fdeg=fdeg, fraw=fraw, **{f"st_{k}": v for k, v in st.items()})
        print(f"wrote {f} ({os.path.getsize(f)/1e6:.1f} MB)  shard {SHARD}/{NSHARD}")
        return
    finish(NE, pts, nrm, cen, frm, cdist, fdeg, fraw, st, TRUE, sfx)


def merge():
    sfx = f"_lim{LIMIT}" if LIMIT else ""
    d0 = np.load("scratch/deep_dataset.npz", allow_pickle=True)
    Rm, c0, TRUE = (d0["R"].astype(np.float64), d0["c0"].astype(np.float64),
                    d0["true"].astype(np.float64))
    NE = LIMIT if LIMIT else len(TRUE)
    pts = np.zeros((NE, NL, NPTS, 3), DT); nrm = np.zeros((NE, NL, NPTS, 3), DT)
    cen = np.zeros((NE, NL, 3), np.float32); frm = np.zeros((NE, NL, 3, 3), np.float32)
    cdist = np.zeros((NE, NL), np.float32); fdeg = np.zeros(NE, np.int32)
    fraw = np.zeros((NE, NL), np.float32)
    st = {k: np.zeros((NE, NL), np.float32)
          for k in ("area", "core_area", "n_face", "n_cc", "keep_frac")}
    seen = np.zeros(NE, bool)
    for k in range(NSHARD):
        p = f"scratch/_lcrop{sfx}_sh{k}.npz"
        assert os.path.exists(p), f"shard {k} missing -- run SHARD={k} first"
        z = np.load(p); ix = z["idx"]
        assert not seen[ix].any(), f"shard {k} overlaps an earlier shard"
        assert z["pts"].dtype == DT and z["pts"].shape[1:] == (NL, NPTS, 3), (
            f"shard {k} holds {z['pts'].dtype} {z['pts'].shape[1:]} but this merge expects "
            f"{DT.__name__} {(NL, NPTS, 3)} -- the shards and the merge were run with "
            f"different NPTS/FP16 settings")
        pts[ix], nrm[ix], cen[ix], frm[ix] = z["pts"], z["nrm"], z["centre"], z["frame"]
        cdist[ix], fdeg[ix], fraw[ix] = z["cdist"], z["fdeg"], z["fraw"]
        for q in st:
            st[q][ix] = z[f"st_{q}"]
        seen[ix] = True
    assert seen.all(), f"{int((~seen).sum())} ears missing after merge"
    finish(NE, pts, nrm, cen, frm, cdist, fdeg, fraw, st, TRUE, sfx)


def finish(NE, pts, nrm, cen, frm, cdist, fdeg, fraw, st, TRUE, sfx):
    """Write the crop artefact + the per-fold target files, and print the analysis."""
    subj, folds = frozen_folds(NE)
    if LIMIT:
        print(f"  ! LIMIT={LIMIT}: frozen_folds({NE}) is NOT the 340-ear split. The target "
              f"files from a LIMIT run are for shape checks only.")
    # The coarse landmarks are bit-identical across deep_dataset.npz and every
    # screen_data_*.npz (verified). Shipping a copy lets fam_local.py prove, by exact
    # equality on the first forward, that the crop artefact and the DATA npz agree on ear
    # ORDER and on the canonical FRAME. A near-match threshold could not: the per-ear
    # median |centre - coarse| already reaches 11.9 mm on a correctly matched pair.
    COARSE = np.load("scratch/deep_dataset.npz", allow_pickle=True)["coarse"][:NE]
    meta = dict(coarse=COARSE.astype(np.float32),
                crop_r=np.float32(CROP_R), npts=np.int32(NPTS), core_r=np.float32(CORE_R),
                core_frac=np.float32(CORE_FRAC), n_core=np.int32(NCORE),
                frame_mode=np.array(FRAME), cc=np.int32(CC), pred_path=np.array(PRED),
                seed=np.int32(SEED), margin=np.float32(MARGIN),
                ear_index=np.arange(NE, dtype=np.int32), fold=folds.astype(np.int32),
                subj=subj.astype(np.int32))
    fc = f"{OUT}_crops{sfx}.npz"
    np.savez(fc, pts=pts, nrm=nrm, centre=cen, frame=frm, centre_dist=cdist,
             frame_ndeg=fdeg, frame_raw_dot=fraw,
             **{f"patch_{k}": v for k, v in st.items()}, **meta)
    print(f"\nwrote {fc} ({os.path.getsize(fc)/1e6:.1f} MB)  pts {pts.shape} {pts.dtype}")

    # ---- residual targets, per fold, validation ears NaN'd and asserted ----
    G = TRUE[:NE]
    resid = np.einsum("elij,elj->eli", frm.astype(np.float64),
                      (G - cen.astype(np.float64)))          # (E,85,3) local frame
    if TARGETS:
        for k in range(NFOLD):
            m = folds != k                                   # training ears of fold k
            r = resid.astype(np.float32).copy()
            r[~m] = np.nan
            assert np.isnan(r[~m]).all() and np.isfinite(r[m]).all(), \
                f"fold {k}: a validation ear's residual is finite -- LEAK"
            f = f"{OUT}_targets_f{k}{sfx}.npz"
            np.savez(f, resid=r, fold=np.int64(k), train_ear_mask=m,
                     centre=cen, frame=frm, **{q: meta[q] for q in
                                               ("crop_r", "npts", "pred_path")})
            assert not np.isfinite(np.load(f)["resid"][~m]).any()
        print(f"wrote {NFOLD} x {OUT}_targets_f<k>{sfx}.npz "
              f"({os.path.getsize(f)/1e3:.0f} kB each; val ears NaN, asserted)")

    # ---- diagnostics ----
    q = lambda a, f="{:.4f}": "/".join(f.format(x) for x in np.percentile(a, [0, 50, 99, 100]))
    A, CA = st["area"], st["core_area"]
    sp_out = np.sqrt(A / max(NPTS - NCORE, 1))
    sp_in = np.sqrt(np.where(CA > 0, CA, A) / max(NCORE, 1))
    print(f"\n--- patch geometry (min/median/p99/max over {NE*NL} patches)")
    print(f"    surface area mm^2            {q(A, '{:.1f}')}   (flat disc pi*R^2 = "
          f"{np.pi*CROP_R**2:.1f}; ratio median {np.median(A)/(np.pi*CROP_R**2):.2f})")
    print(f"    faces per patch              {q(st['n_face'], '{:.0f}')}")
    print(f"    in-ball components           {q(st['n_cc'], '{:.0f}')}  kept-face fraction "
          f"{q(st['keep_frac'], '{:.3f}')}")
    print(f"    ACHIEVED SPACING mm  outer   {q(sp_out)}")
    print(f"                         core    {q(sp_in)}   (core area mm^2 {q(CA, '{:.1f}')})")
    print(f"    centre-to-surface mm         {q(cdist)}")
    print(f"    |dot(n_raw,t)| (frame Gram-Schmidt) {q(fraw)}   degenerate tangents "
          f"{int(fdeg.sum())}")
    # Conditioning, not correctness: the frame stays exactly orthonormal (asserted per ear
    # in local_frames), but where |dot| -> 1 the b and n axes are poorly determined, so a
    # small change in the predicted tangent swings them. Count it rather than trust it.
    print(f"    ... |dot| > 0.9 on {int((fraw > 0.9).sum())} of {fraw.size} patches "
          f"({(fraw > 0.9).mean()*100:.4f}%), > 0.7 on {int((fraw > 0.7).sum())}; "
          f"max |F F^T - I| {np.abs(np.einsum('elij,elkj->elik', frm, frm) - np.eye(3)).max():.2e}")

    if ANALYSE:
        # GT is used HERE FOR REPORTING ONLY, over all ears. Nothing GT-derived leaves
        # this block except the per-fold masked target files written above.
        R = np.linalg.norm(resid, axis=2)
        print(f"\n--- residual this stage must fix (GT - centre), REPORTING ONLY")
        print(f"    |resid| mm                   {q(R)}   mean {R.mean():.4f}")
        for r in (CORE_R, 4.0, 6.0, CROP_R):
            print(f"    reachable within {r:4.1f} mm      {(R < r).mean()*100:.3f}%  "
                  f"({int((R >= r).sum())} of {R.size} outside)")
        lab = ("along-contour t", "across b", "normal n") if FRAME == "contour" else \
              ("tangent1 t", "tangent2 b", "normal n")
        for j, nm in enumerate(lab):
            print(f"    RMSE {nm:16s}        {np.sqrt((resid[..., j]**2).mean()):.4f} mm")
        P32 = pts.astype(np.float32)
        d = np.linalg.norm(P32 - resid.astype(np.float32)[:, :, None, :], axis=3).min(2)
        print(f"\n--- HEATMAP REPRESENTATION FLOOR: GT to nearest PATCH POINT")
        print(f"    mm                           {q(d)}   mean {d.mean():.4f}")
        print(f"    over 0.5 mm on {int((d > 0.5).sum())} patches ({(d > 0.5).mean()*100:.3f}%), "
              f"over 1.0 mm on {int((d > 1.0).sum())} ({(d > 1.0).mean()*100:.3f}%) -- the tail "
              f"is the {int((R >= CROP_R).sum())} patches whose GT is outside the ball at all")
        print(f"    This is the floor of a HARD argmax over the sampled points. Soft-argmax\n"
              f"    interpolates between them, so it is a ceiling on the discretisation cost,\n"
              f"    not on the achievable error; compare it against the 0.5 mm target and\n"
              f"    against the 1.1776 mm the cascade is refining.")
        # Precision. The stored array IS the quantised one, so re-quantising it measures
        # nothing (that mistake returns a hard 0.00e+00). What is meaningful is the ulp at
        # the magnitudes actually stored, plus the round-trip through the frame, which is
        # where a transposed rotation would show up as a gross error rather than a rounding.
        qmax = float(np.abs(P32).max())
        aq = np.abs(P32)
        ulp = np.where(aq > 0, 2.0 ** (np.floor(np.log2(np.maximum(aq, 1e-8))) - 10), 0.0)
        back = cen.astype(np.float64)[:, :, None, :] + np.einsum(
            "elpi,elij->elpj", pts.astype(np.float64), frm.astype(np.float64))
        dc = np.linalg.norm(back - cen.astype(np.float64)[:, :, None, :], axis=3)
        print(f"\n--- precision ({DT.__name__} storage): |q| max {qmax:.3f} mm; "
              f"quantisation step at the stored magnitudes: median {np.median(ulp):.2e} "
              f"max {ulp.max():.2e} mm")
        print(f"    local->canonical round trip preserves |q|: max |{'|q|'}-|x-c||  "
              f"{np.abs(dc - np.linalg.norm(P32, axis=3)).max():.2e} mm; "
              f"reconstructed canonical range {back.min():.2f}..{back.max():.2f}")


# ------------------------------------------------------------------ smoke test
def smoke():
    """Synthetic surfaces with a KNOWN area and a known answer, then a torch
    forward+backward over the emitted layout via fam_local.MODEL."""
    import torch
    t0 = time.time()
    print("=" * 78)
    print("SMOKE 1/4 -- patch extraction on a synthetic surface with analytic area")
    g = 220
    u = np.linspace(-14, 14, g)
    UU, VV = np.meshgrid(u, u, indexing="ij")
    ZZ = np.zeros_like(UU)
    V = np.stack([UU.ravel(), VV.ravel(), ZZ.ravel()], 1).astype(np.float64)
    ii = (np.arange(g - 1)[:, None] * g + np.arange(g - 1)[None, :]).ravel()
    F = np.concatenate([np.stack([ii, ii + g, ii + g + 1], 1),
                        np.stack([ii, ii + g + 1, ii + 1], 1)]).astype(np.int64)
    VN = vertex_normals(V, F)
    tree = cKDTree(V)
    rng = np.random.RandomState(0)
    p, n, s = patch(V, F, VN, tree, np.array([0.0, 0.0, 0.0]), rng)
    # a plane: the in-ball faces are those with all 3 corners inside, so the area is
    # slightly under pi*R^2, and every sampled point must satisfy |q| <= R.
    print(f"  patch area {s['area']:.2f} mm^2 vs pi*R^2 {np.pi*CROP_R**2:.2f} "
          f"(ratio {s['area']/(np.pi*CROP_R**2):.3f})  faces {s['n_face']}  cc {s['n_cc']}")
    print(f"  |q| max {np.linalg.norm(p, axis=1).max():.4f} <= R {CROP_R};  core points "
          f"within {CORE_R}mm: {int((np.linalg.norm(p[NPTS-NCORE:], axis=1) <= CORE_R).sum())}"
          f"/{NCORE}")
    assert 0.90 < s["area"] / (np.pi * CROP_R ** 2) < 1.001, s["area"]
    assert np.linalg.norm(p, axis=1).max() <= CROP_R + 1e-9
    assert (np.linalg.norm(p[NPTS - NCORE:], axis=1) <= CORE_R + 1e-9).all()
    assert np.abs(np.linalg.norm(n, axis=1) - 1).max() < 1e-6
    assert np.abs(np.abs(n[:, 2]) - 1).max() < 1e-9, "plane normals must be +-z"
    # two parallel sheets 1mm apart: the Euclidean ball catches both, CC must keep one
    V2 = np.concatenate([V, V + np.array([0., 0., 1.0])])
    F2 = np.concatenate([F, F + len(V)])
    r2 = patch(V2, F2, vertex_normals(V2, F2), cKDTree(V2), np.zeros(3), rng)
    print(f"  two sheets 1mm apart: components {r2[2]['n_cc']}, kept-face fraction "
          f"{r2[2]['keep_frac']:.3f}, area {r2[2]['area']:.2f} (one sheet)")
    assert r2[2]["n_cc"] == 2 and abs(r2[2]["keep_frac"] - 0.5) < 0.02
    assert abs(r2[2]["area"] - s["area"]) < 1e-6, "CC pruning did not isolate one sheet"

    print("\nSMOKE 2/4 -- local frame: proper rotation, exact round-trip, fp16 cost")
    P = np.stack([np.linspace(-6, 6, NL), np.sin(np.linspace(0, 6, NL)) * 3,
                  np.zeros(NL)], 1)
    Nq = VN[tree.query(P)[1]]
    Fr, raw, nd = local_frames(P, Nq)
    print(f"  det {np.linalg.det(Fr).min():.9f}..{np.linalg.det(Fr).max():.9f}  "
          f"|dot(n_raw,t)| max {raw.max():.4f}  degenerate tangents {nd}")
    x = V[rng.choice(len(V), 5000)]
    q = (x - P[0]) @ Fr[0].T
    back = P[0] + q @ Fr[0]
    print(f"  round-trip fp64 max err {np.abs(back - x).max():.2e} mm")
    q16 = q.astype(np.float16).astype(np.float64)
    inb = np.linalg.norm(q, axis=1) <= CROP_R                 # what actually gets stored
    print(f"  fp16 storage cost, |q| <= R={CROP_R}: max {np.abs(q16[inb]-q[inb]).max():.2e} mm "
          f"(ulp at {CROP_R}mm = {CROP_R*2**-11:.5f})")
    print(f"  fp16 cost out to |q| = {np.abs(q).max():.1f} mm (not stored): max "
          f"{np.abs(q16 - q).max():.2e} mm -- the bound is proportional to |q|")
    assert np.abs(back - x).max() < 1e-9
    assert np.abs(q16[inb] - q[inb]).max() < CROP_R * 2 ** -10

    print("\nSMOKE 3/4 -- per-fold targets: a validation ear's GT must never be written")
    ne = 20
    subj, folds = frozen_folds(ne)
    cen_ = rng.randn(ne, NL, 3)
    frm_ = np.tile(np.eye(3), (ne, NL, 1, 1))
    G = cen_ + rng.randn(ne, NL, 3) * 0.5
    res = np.einsum("elij,elj->eli", frm_, G - cen_)
    for k in range(NFOLD):
        m = folds != k
        r = res.astype(np.float32).copy(); r[~m] = np.nan
        assert np.isnan(r[~m]).all() and np.isfinite(r[m]).all()
        assert not np.isfinite(r[folds == k]).any()
    bad = res.astype(np.float32).copy()                  # the mistake this guards against
    try:
        m = folds != 0
        assert np.isnan(bad[~m]).all()
        raise SystemExit("an unmasked target file was ACCEPTED -- LEAK")
    except AssertionError:
        pass
    print(f"  {NFOLD} folds x {ne} ears: every held-out ear NaN, every training ear finite;"
          f"\n  an unmasked array is refused by the same assertion")

    print("\nSMOKE 4/4 -- torch forward+backward over the emitted layout (fam_local.MODEL)")
    import importlib
    fl = importlib.import_module("fam_local")
    B, N = 2, 64
    meta = dict(nl=NL, contours=CONT, scale=30.0, npts=2048, fold=0, dev="cpu",
                n_train_ears=16, artefacts={})
    net = fl.MODEL({**fl.MODEL.DEFAULTS, "lm_per_step": 0}, meta)
    batch = {"pc": torch.randn(B, 128, 3), "coarse": torch.randn(B, NL, 3),
             "ear": torch.arange(B),
             "crop": torch.randn(B, NL * N, 3), "cnrm": torch.randn(B, NL * N, 3),
             "qc": torch.randn(B, NL, 3), "qf": torch.eye(3).repeat(B, NL, 1),
             "npatch": torch.tensor(N)}
    out = net(batch)
    tg = torch.randn(B, NL, 3)
    loss = net.loss(out, tg, batch)
    loss.backward()
    gn = sum(float(p.grad.norm()) for p in net.parameters() if p.grad is not None)
    print(f"  pred {tuple(out['pred'].shape)}  params "
          f"{sum(p.numel() for p in net.parameters()):,}  loss {float(loss):.4f}  "
          f"grad-norm-sum {gn:.3f}")
    assert tuple(out["pred"].shape) == (B, NL, 3), out["pred"].shape
    assert np.isfinite(gn) and gn > 0
    print(f"SMOKE PASS ({time.time()-t0:.0f}s)")
    print("=" * 78)


if __name__ == "__main__":
    if int(os.environ.get("SMOKE", "0")):
        smoke()
    elif SHARD == "merge":
        merge()
    else:
        run()
