"""
LOCAL preprocessing for FAMILY E (research/code/fam_bilateral.py): the LOW-RESOLUTION
FULL-HEAD point cloud, plus the partner-ear columns that let train_family.py's per-ear
batching see both ears of a subject. Emits ONE self-contained DATA npz; the GPU box needs
numpy + torch only and never touches a mesh.

    python research/code/build_bilateral_data.py            # all 340 dev ears
    LIMIT=4 python research/code/build_bilateral_data.py    # 4-ear check
    SMOKE=1 python research/code/build_bilateral_data.py    # CPU self-test, no data needed

WHY EAR-INDEXED PARTNER COLUMNS AND NOT A PAIRED BATCH. train_family.make_batch indexes
every extra array by the TARGET ear index e, so a family cannot reach index e^1 from
inside a forward pass. Rather than teach the trainer a second indexing mode (which every
other family would then have to be re-validated against), the partner's arrays are
PERMUTED HERE: pcp[e] = clouds[e^1], headp[e] = head[e^1]. Constraint 3 fixes
subject = ear_index // 2 with the two ears stored as consecutive (left, right) rows, so
partner(e) = e ^ 1 -- and that is ASSERTED against the pid/side arrays rather than
assumed. Cost: `clouds`/`nrm`/`head`/`head_nrm` are each stored twice (+116 MB), which is
the price of not modifying shared trainer plumbing.

THE HEAD CLOUD IS PER EAR, NOT PER SUBJECT, and that is the point. head[e] is the
SUBJECT'S WHOLE HEAD expressed in EAR e's canonical frame. So the same head appears twice
per subject, in the two ears' frames, and the difference between those two views IS the
ear-to-ear relative pose -- the only place fam_bilateral's subject token can see it, since
the two ear crops are each canonicalised independently. Both views are members of one
order-invariantly pooled set, which is what keeps the subject token identical for the two
ears (see fam_bilateral's docstring).

MIRRORING AND NORMALS -- THE BUG THIS FILE REFUSES TO REPEAT. A right ear is mirrored by
M = diag(1,-1,1) into the left-ear canonical frame. M is a REFLECTION, so:
  * the OUTWARD normal of the reflected surface is M*n. It is NOT -(M*n).
  * but the WINDING-derived normal of the reflected mesh with UNCHANGED winding is
    -(M*n), because cross(M a, M b) = det(M) * M * cross(a, b).
So "mirror the normal vector and negate it" and "recompute from the unchanged winding"
both ship INWARD normals. The safe route, used here, is to flip the winding
(F[:, [0, 2, 1]]) and recompute the vertex normals in the frame we actually emit.
MEASURED on P0001, and asserted per ear:
    dot(flip-and-recompute , +M*n) = +1.0000
    dot(flip-and-recompute , -M*n) = -1.0000
    dot(same-winding recompute, -M*n) = +1.0000        <- the two wrong routes agree
and against the SHIPPED scratch/screen_data_2048nrm.npz normals for ear 1 (a right ear),
dot = +1.0000 for flip-and-recompute and -1.0000 for -(M*n). NB the DOCSTRINGS of
build_screen_extra.py and build_mesh_data.py both state the rule as "n -> -(MIRROR*n)";
their CODE flips the winding and recomputes. The code is right and the docstrings are
stale. Do not copy them.

THE FRAME IS PROVEN, NOT CLAIMED (VERIFY=1, the default). For every ear this script
rebuilds the 14 mm coarse-landmark bbox crop and REGENERATES the base file's 2048-point
sample indices from their original RNG stream (RandomState(1000 + 97*i + j), exactly
build_multisample_all.py / build_screen_extra.py), then asserts the resulting coordinates
AND normals equal the ones in BASE. That is a point-for-point proof that the head cloud
shares BASE's canonical frame, ear order, mirroring convention and normal orientation --
the four things that fail silently. The measured worst-case discrepancy is reported (0.0 on
the full dev set).
  The regeneration only reproduces BASE in the ORIGINAL float32. The bbox mask is a
  >= / <= test, so promoting coarse/R/c0 to float64 moves the bounds by ~1e-6 mm and flips
  vertices sitting exactly on them: MEASURED on ear 10 (P0008/left), 22866 kept vertices in
  float32 against 22865 in float64. One vertex reindexes the entire RandomState draw and
  the check reports a 63.21 mm "frame error" for a crop that is in fact correct. The head
  cloud itself is built in float64 and does not care; only VERIFY has to be bit-faithful.

POINT BUDGET: HPTS = 3072 POINTS OVER A 140 mm CROP. Two measurements decide both numbers.
  1. HRADIUS. The raw scans are head + neck + shoulders and their TOTAL surface area
     varies 2.25x across subjects (measured: 159,746 / 173,714 / 349,643 / 359,003 mm^2 on
     4 sampled subjects) purely by how much torso the scan happens to include. A fixed
     point budget on the full scan therefore gives a per-subject RESOLUTION that varies by
     ~1.5x -- the same scan-resolution confound build_mesh_data.py removed with MAXV, and
     one that a subject-level descriptor would happily latch onto. Cropping to 140 mm
     around the ear centre (c0) cuts that to 1.53x: MEASURED over ALL 340 dev ears, the
     retained area is 63,301 / 73,533 / 96,842 mm^2 (min/median/max) and the emitted mean
     nearest-neighbour spacing 3.411 / 3.719 / 4.206 mm, a 1.23x resolution spread. The
     residual is genuine head-size variation plus how far down the neck the 140 mm ball
     reaches; it is NOT removed, only reduced, and `head_spacing` is stored per ear so a
     later analysis can condition on it.
     The crop still covers the WHOLE HEAD: extents are 199 / 216 / 249 mm (x), 116 / 144 /
     224 (y), 229 / 265 / 277 (z), i.e. it reaches the crown, the chin and the nape and is
     cut off in y only by the head's own width -- so the CONTRALATERAL ear falls outside.
     That is fine: the other ear is already its own context member at full resolution.
     HRADIUS=0 emits the whole scan and accepts the 2.25x confound.
  2. HPTS. At 3072 points over ~73,500 mm^2 the emitted cloud has a MEASURED mean
     nearest-neighbour spacing of 3.708 mm over the 340 ears (median 3.701, p99 5.572).
     The ear crop already owns the sub-millimetre scale (2048 points over ~7,240 mm^2 =
     0.995 mm), so the head member's job is strictly the >3.5 mm scale: where the ear sits
     on the skull, its protrusion, the mastoid / jaw / cheek, gross head shape. Spending
     more points buys detail the crop already has, at linear cost in the file and in the
     forward pass. 2048 would give ~4.5 mm and 4096 ~3.2 mm; 3072 sits mid-range of the
     2-4k this representation was specified at. fam_bilateral.HEAD_SPACING_MM is set to the
     3.708 mm number and HJIT is derived from it, so a rebuild at another HPTS/HRADIUS must
     feed the printed mean back.

VOXEL PICK, NOT RANDOM. Points are the member vertices nearest each occupied grid cell's
centroid, so every emitted point lies EXACTLY on the scan and the density is equalised
(scanners oversample the ear and nose). Measured on P0001-left at 3072 points:
    voxel   mean NN 3.678 mm   p99 NN 5.474 mm
    random  mean NN 2.372 mm   p99 NN 6.404 mm
i.e. a uniform-random pick of the same size both clumps (smaller typical spacing) and
leaves bigger holes (larger tail). Cell size is bisected to land the occupied-cell count
in [HPTS, OVER*HPTS] and the surplus is dropped with a fixed-seed RNG. Over the 340 ears
the chosen cell is 4.93 / 5.62 / 6.62 mm and no ear ever needed padding.

NO GROUND TRUTH IS USED. Inputs are mesh vertices, mesh faces, and the per-ear coarse
alignment (R, c0, coarse) that every existing component already consumes. `true` is copied
through from BASE unchanged because train_family.py needs it as the loss target; nothing in
this script reads it, and no emitted array is derived from it. The 30-subject LOCKBOX is
asserted absent from the ear order.

OUTPUT: $OUT (default scratch/bilat_data_2048.npz), a DROP-IN replacement for
scratch/screen_data_2048nrm.npz -- it carries every key that file has, byte-identical,
plus:
  pcp        (E,M,N,3) f32  partner ear's crop, in the PARTNER's own canonical frame
  pcp_nrm    (E,M,N,3) f32  ... and its outward normals
  head       (E,Nh,3)  f32  subject's head, low-res, in THIS ear's canonical frame
  head_nrm   (E,Nh,3)  f32
  headp      (E,Nh,3)  f32  the same head in the PARTNER's canonical frame (= head[e^1])
  headp_nrm  (E,Nh,3)  f32
  has_partner (E,) f32      1.0 iff ear e^1 exists (all 1.0 in this dataset)
  has_head    (E,) f32      1.0 iff a head cloud was emitted
  partner (E,) i32, pid (E,) U, side (E,) U, head_spacing (E,) f32,
  head_area (E,) f32, head_src_verts (E,) i32, head_extent (E,3) f32,
  head_cell_mm (E,) f32, hpts (), hradius (), margin (), frame_err_mm ()

ENV (defaults in brackets)
  BASE   [scratch/screen_data_2048nrm.npz]  the 2048-point clouds + normals to extend
  OUT    [scratch/bilat_data_2048.npz]
  HPTS   [3072]    points in the head cloud
  HRADIUS [140.0]  mm around c0; 0 = the whole scan (accepts the 2.25x area confound)
  OVER   [1.30]    upper bound on occupied cells before trimming, as a multiple of HPTS
  MARGIN [14.0]    the 14 mm bbox of build_screen_extra.py, used only by VERIFY
  VERIFY [1]       regenerate BASE's clouds/normals and assert equality
  SEED   [0]       RNG for the surplus-cell trim (per ear: SEED + 7919*i)
  LIMIT  [0]       first N ears only (0 = all); appends _lim{N} to OUT
  SMOKE  [0]       1 = synthetic self-test, no dataset needed
"""
import os, sys, time
import numpy as np
from scipy.spatial import cKDTree
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

MESH = "2026 Munich Tech Arena - Datas/2026 Munich Tech Arena - Datas/mesh"
LM = "2026 Munich Tech Arena - Datas/2026 Munich Tech Arena - Datas/landmarks"
MIRROR = np.array([1., -1., 1.])
NL = 85

BASE = os.environ.get("BASE", "scratch/screen_data_2048nrm.npz")
OUT = os.environ.get("OUT", "scratch/bilat_data_2048.npz")
HPTS = int(os.environ.get("HPTS", "3072"))
HRADIUS = float(os.environ.get("HRADIUS", "140.0"))
OVER = float(os.environ.get("OVER", "1.30"))
MARGIN = float(os.environ.get("MARGIN", "14.0"))
VERIFY = int(os.environ.get("VERIFY", "1"))
SEED = int(os.environ.get("SEED", "0"))
LIMIT = int(os.environ.get("LIMIT", "0"))
FB_SPACING = 3.708              # fam_bilateral.HEAD_SPACING_MM; cross-checked at the end


# --------------------------------------------------------------- geometry
def vertex_normals(V, F):
    """area-weighted vertex normals from the winding of F (outward iff F is CCW).

    bincount, not np.add.at: same result, ~20x faster on a 730k-face scan, and this runs
    twice per subject (once per side).
    """
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    N = np.stack([np.bincount(F.ravel(), np.repeat(fn[:, c], 3), minlength=len(V))
                  for c in range(3)], 1)
    nr = np.linalg.norm(N, axis=1, keepdims=True)
    return np.where(nr > 1e-12, N / np.maximum(nr, 1e-12), np.array([0., 0., 1.]))


def signed_volume(V, F):
    """(1/6) sum v0 . (v1 x v2); positive iff the winding is outward. The radial dot is
    NOT usable here -- a head is not star-shaped, so it runs as low as 0.45 on a correctly
    wound scan (measured on P0001 by build_mesh_data.py)."""
    return float(np.einsum("ij,ij->i", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum() / 6.0)


def mirror_side(V, F, VN, side):
    """(V, F, VN) for `side`, in the world frame the canonical R/c0 were fitted in.

    Right: reflect the vertices, FLIP THE WINDING, and RECOMPUTE the normals. The
    assertion is deliberately non-vacuous -- it checks the recomputed field against +M*n
    (correct) and would fire on -(M*n) (the classic wrong route). See the module docstring.
    """
    if side != "right":
        return V, F, VN
    Vw, Fw = V * MIRROR, F[:, [0, 2, 1]]
    VNw = vertex_normals(Vw, Fw)
    d = float((VNw * (VN * MIRROR)).sum(1).mean())
    assert d > 0.99, (
        f"mirrored normals are not +M*n (mean dot {d:.4f}). Under a reflection the outward "
        f"normal is M*n; -(M*n) and a same-winding recompute both give ~-1 and ship INWARD "
        f"normals, which is the bug this assertion exists to catch.")
    return Vw, Fw, VNw


def voxel_pick(P, target, seed, over=OVER):
    """~uniform subsample of P to EXACTLY `target` ORIGINAL points.

    Bisect the cell size until the occupied-cell count lands in [target, over*target],
    keep the member vertex nearest each cell's centroid (so every point is ON the scan,
    never an averaged centroid), then drop the surplus with a fixed-seed RNG. A random
    drop from an already near-uniform set stays near-uniform; picking the most-populated
    cells instead would re-introduce the density bias voxelisation just removed.
    Returns (idx into P, cell size mm, occupied cells, n_padded).
    """
    Q = P - P.min(0)
    ext = Q.max(0) + 1e-9
    lo, hi, best = 0.02, float(ext.max()), None
    for _ in range(40):
        h = float(np.sqrt(lo * hi))
        g = np.floor(Q / h).astype(np.int64)
        dims = g.max(0) + 1
        assert dims.prod() < 2 ** 62, f"grid too fine: {dims} cells at h={h}"
        cell = (g[:, 0] * dims[1] + g[:, 1]) * dims[2] + g[:, 2]
        _, inv = np.unique(cell, return_inverse=True)
        nc = int(inv.max()) + 1
        if nc >= target and (best is None or nc < best[0]):
            best = (nc, h, inv)
        if nc > over * target:
            lo = h                                  # too many cells -> coarser grid
        elif nc < target:
            hi = h
        else:
            break
    if best is None:                                # fewer distinct cells than target ever
        best = (nc, h, inv)
    nc, h, inv = best
    cen = np.stack([np.bincount(inv, P[:, c], minlength=nc) for c in range(3)], 1)
    cen /= np.bincount(inv, minlength=nc)[:, None]
    d = np.einsum("ij,ij->i", P - cen[inv], P - cen[inv])
    order = np.lexsort((d, inv))
    ls = inv[order]
    rep = order[np.r_[True, ls[1:] != ls[:-1]]]
    assert len(rep) == nc
    rng = np.random.RandomState(seed)
    pad = 0
    if len(rep) > target:
        rep = rep[np.sort(rng.choice(len(rep), target, replace=False))]
    elif len(rep) < target:
        pad = target - len(rep)
        rep = np.concatenate([rep, rep[rng.randint(0, len(rep), pad)]])
    return rep, h, nc, pad


def head_cloud(Pc, Nc, a2c, Fc, hpts=HPTS, hradius=HRADIUS, seed=0):
    """canonical-frame head vertices+normals -> (pts, nrm, diagnostics).

    `Pc`/`Nc` are ALL the canonicalised vertices; `Fc`/`a2c` are only used to report the
    retained surface AREA (which is what the point budget is calibrated against).
    """
    r = np.linalg.norm(Pc, axis=1)
    keep = np.ones(len(Pc), bool) if hradius <= 0 else (r <= hradius)
    assert keep.sum() >= hpts, \
        f"only {int(keep.sum())} vertices within {hradius} mm of the ear centre, need {hpts}"
    fin = keep[Fc].all(1)
    idx, h, nc, pad = voxel_pick(Pc[keep], hpts, seed)
    P, N = Pc[keep][idx], Nc[keep][idx]
    nn = cKDTree(P).query(P, k=2)[0][:, 1]
    return P.astype(np.float32), N.astype(np.float32), dict(
        area=float(a2c[fin].sum() / 2.0), src=int(keep.sum()), cell=h, cells=nc, pad=pad,
        spacing=float(nn.mean()), sp_med=float(np.median(nn)), sp_p99=float(np.percentile(nn, 99)),
        extent=(P.max(0) - P.min(0)).astype(np.float32))


# --------------------------------------------------------------- driver
def ear_order():
    """the (pid, side) order every artefact in this repo is registered against, plus the
    partner map derived from it -- asserted, not assumed."""
    from src.splits import get_split
    tr = get_split("train", mesh_dir=Path(MESH))
    va = get_split("val", mesh_dir=Path(MESH))
    lock = set(get_split("test", mesh_dir=Path(MESH)))
    order = [(p, s) for p in tr for s in ("left", "right")] + \
            [(p, s) for p in va for s in ("left", "right")]
    assert not (set(p for p, _ in order) & lock), "LOCKBOX subject leaked into the ear order"
    ne = len(order)
    for e in range(0, ne, 2):
        assert order[e][0] == order[e + 1][0], \
            f"ears {e},{e+1} are not the same subject ({order[e][0]} / {order[e+1][0]}); " \
            f"partner(e) = e^1 and subject = e//2 (constraint 3) do not hold for this order"
        assert (order[e][1], order[e + 1][1]) == ("left", "right"), \
            f"ears {e},{e+1} are not (left, right) but {order[e][1]}, {order[e+1][1]}"
    return order, np.arange(ne) ^ 1


def run():
    from src.dataset import Dataset
    t0 = time.time()
    b = np.load(BASE, allow_pickle=True)
    clouds, nrm = b["clouds"], b["nrm"]
    coarse, true, Rm, c0, split = b["coarse"], b["true"], b["R"], b["c0"], b["split"]
    order, partner = ear_order()
    NEA, M, NPTS = clouds.shape[0], clouds.shape[1], clouds.shape[2]
    assert len(order) == NEA, f"{BASE} has {NEA} ears, the split order has {len(order)}"
    NE = LIMIT if LIMIT else NEA
    print(f"[bilatprep] {NE}/{NEA} ears  HPTS={HPTS} HRADIUS={HRADIUS} OVER={OVER} "
          f"VERIFY={VERIFY}  base {BASE} ({NPTS} pts x {M} samples)", flush=True)

    head = np.zeros((NEA, HPTS, 3), np.float32)
    hnrm = np.zeros((NEA, HPTS, 3), np.float32)
    have = np.zeros(NEA, np.float32)
    per = {k: [] for k in ("area", "src", "cell", "cells", "pad", "spacing", "sp_med",
                           "sp_p99", "svol", "mirr_dot", "nvert", "nface")}
    extent = np.zeros((NEA, 3), np.float32)
    ds = Dataset(MESH, LM)
    pid2idx = {p: i for i, p in enumerate(ds.subject_ids)}
    cache, ferr = {}, 0.0

    for i in range(NE):
        pid, side = order[i]
        if pid not in cache:
            m = ds[pid2idx[pid]][0]
            V0 = np.asarray(m.vertices, np.float64)
            F0 = np.asarray(m.faces, np.int64)
            cache = {pid: (V0, F0, vertex_normals(V0, F0), signed_volume(V0, F0))}
        V0, F0, VN0, svol = cache[pid]
        assert svol > 0, (f"{pid}: mesh winding is inward (signed volume {svol:.1f} mm^3), "
                          f"so every normal in this file would point into the surface")
        Vw, Fw, VNw = mirror_side(V0, F0, VN0, side)
        per["svol"].append(svol)
        per["mirr_dot"].append(float((VNw * (VN0 * MIRROR)).sum(1).mean())
                               if side == "right" else 1.0)
        R, cc = Rm[i].astype(np.float64), c0[i].astype(np.float64)
        assert abs(np.linalg.det(R) - 1.0) < 1e-5, f"ear {i}: R is not a proper rotation"

        # --- frame proof against BASE, point for point (see the docstring)
        if VERIFY:
            # IN THE ORIGINAL float32, deliberately. build_screen_extra.py computed the
            # bbox from the float32 coarse/R/c0, and the mask is a >= / <= test: promoting
            # to float64 moves the bounds by ~1e-6 mm and flips vertices that sit exactly
            # on them. MEASURED on ear 10 (P0008/left) -- 22866 vertices in float32 vs
            # 22865 in float64, one vertex, which reindexes the whole RandomState draw and
            # shows up as a 63.21 mm "frame error". The crop itself is then computed in
            # mixed precision exactly as the original did.
            cw = coarse[i] @ Rm[i] + c0[i]
            lo, hi = cw.min(0) - np.float32(MARGIN), cw.max(0) + np.float32(MARGIN)
            msk = np.all((Vw >= lo) & (Vw <= hi), axis=1)
            sel = msk if msk.any() else np.ones(len(Vw), bool)
            cl = (Vw[sel] - c0[i]) @ Rm[i].T
            cn = VNw[sel] @ Rm[i].T
            for j in range(M):
                k = np.random.RandomState(1000 + 97 * i + j).randint(0, len(cl), NPTS)
                ec = float(np.abs(cl[k].astype(np.float32) - clouds[i, j]).max())
                en = float(np.abs(cn[k].astype(np.float32) - nrm[i, j]).max())
                assert max(ec, en) < 2e-4, (
                    f"ear {i} sample {j}: regenerated crop differs from {BASE} by "
                    f"{ec:.2e} mm / normals by {en:.2e}. The head cloud would then be in a "
                    f"different frame, ear order or mirroring convention than the clouds "
                    f"the network trains on, and every context member would be wrong.")
                ferr = max(ferr, ec, en)

        # --- the head cloud, in THIS ear's canonical frame
        Pc = (Vw - cc) @ R.T
        Nc = VNw @ R.T
        a2 = np.linalg.norm(np.cross(Vw[Fw[:, 1]] - Vw[Fw[:, 0]],
                                     Vw[Fw[:, 2]] - Vw[Fw[:, 0]]), axis=1)
        head[i], hnrm[i], dg = head_cloud(Pc, Nc, a2, Fw, HPTS, HRADIUS, SEED + 7919 * i)
        have[i] = 1.0
        extent[i] = dg.pop("extent")
        for k, v in dg.items():
            per[k].append(v)
        per["nvert"].append(len(V0)); per["nface"].append(len(F0))
        if (i + 1) % 20 == 0 or i + 1 == NE:
            el = time.time() - t0
            print(f"  {i+1}/{NE}  {el:.0f}s  eta {el/(i+1)*(NE-i-1):.0f}s  "
                  f"{pid}/{side} src={dg['src']} cells={dg['cells']} "
                  f"h={dg['cell']:.2f}mm spacing={dg['spacing']:.3f}mm", flush=True)

    # --- partner columns. `have_partner` is a property of the PAIR, so it is symmetric.
    ok = have > 0
    hasp = (ok & ok[partner] & (partner != np.arange(NEA))).astype(np.float32)
    P = {k: np.asarray(v, np.float64) for k, v in per.items()}
    out = dict(clouds=clouds, nrm=nrm, coarse=coarse, true=true, R=Rm, c0=c0, split=split,
               pcp=clouds[partner], pcp_nrm=nrm[partner],
               head=head, head_nrm=hnrm, headp=head[partner], headp_nrm=hnrm[partner],
               has_partner=hasp, has_head=have,
               partner=partner.astype(np.int32),
               pid=np.array([p for p, _ in order]), side=np.array([s for _, s in order]),
               head_spacing=np.zeros(NEA, np.float32), head_area=np.zeros(NEA, np.float32),
               head_src_verts=np.zeros(NEA, np.int32), head_cell_mm=np.zeros(NEA, np.float32),
               head_extent=extent, hpts=np.int32(HPTS), hradius=np.float32(HRADIUS),
               margin=np.float32(MARGIN),
               # -1 means "not checked", so a VERIFY=0 build can never be mistaken for a
               # build whose frame was proven to match BASE exactly
               frame_err_mm=np.float32(ferr if VERIFY else -1.0))
    for key, src in (("head_spacing", "spacing"), ("head_area", "area"),
                     ("head_src_verts", "src"), ("head_cell_mm", "cell")):
        out[key][:NE] = P[src].astype(out[key].dtype)
    for k, v in out.items():
        if getattr(v, "dtype", None) is not None and v.dtype.kind == "f":
            assert np.isfinite(v).all(), f"non-finite values in {k}"
    assert np.allclose(np.linalg.norm(out["head_nrm"][:NE].reshape(-1, 3), axis=1), 1.0,
                       atol=1e-3), "head normals are not unit length"
    # the pairing contract, stated as an executable identity rather than a comment
    for i in range(NE):
        j = int(partner[i])
        if j < NE:
            assert np.array_equal(out["headp"][i], out["head"][j]), f"headp[{i}] != head[{j}]"
            assert np.array_equal(out["pcp"][i], clouds[j]), f"pcp[{i}] != clouds[{j}]"

    sfx = f"_lim{LIMIT}" if LIMIT else ""
    path = OUT.replace(".npz", f"{sfx}.npz")
    np.savez(path, **out)

    def q(a, f="{:.4g}"):
        return "/".join(f.format(x) for x in np.percentile(a, [0, 50, 100]))
    print(f"\n--- scan (min/median/max over {NE} ears)")
    print(f"    source vertices {q(P['nvert'], '{:.0f}')}  faces {q(P['nface'], '{:.0f}')}  "
          f"kept within {HRADIUS}mm {q(P['src'], '{:.0f}')}")
    print(f"    retained surface area mm^2 {q(P['area'], '{:.0f}')}  "
          f"(spread {P['area'].max()/P['area'].min():.2f}x)")
    print(f"--- emitted head cloud: {HPTS} points")
    print(f"    mean NN spacing mm {q(P['spacing'])}  median {q(P['sp_med'])}  "
          f"p99 {q(P['sp_p99'])}")
    print(f"    voxel cell mm {q(P['cell'])}  occupied cells {q(P['cells'], '{:.0f}')}  "
          f"padded points {int(P['pad'].sum())}")
    print(f"    extent mm  x {q(extent[:NE, 0])}  y {q(extent[:NE, 1])}  z {q(extent[:NE, 2])}")
    print("\n--- ASSERTIONS (all passed) ---")
    print(f"  ear order is (left, right) pairs, partner = e^1        {NE} ears")
    print(f"  lockbox subjects absent from the ear order             OK")
    print(f"  source winding outward, signed volume > 0              min {P['svol'].min():.3e} mm^3")
    print(f"  mirrored normals are +M*n (NOT -(M*n))                 min dot "
          f"{P['mirr_dot'].min():.4f}")
    print(f"  R is a proper rotation, |det - 1| < 1e-5               OK")
    print("  regenerated crop+normals == %-26s max err %.2e (< 2e-4)"
          % (os.path.basename(BASE), ferr) if VERIFY else
          "  VERIFY=0 -- the frame was NOT proven against BASE       SKIPPED")
    print(f"  head normals unit length                               OK")
    print(f"  headp[e] == head[e^1] and pcp[e] == clouds[e^1]        OK")
    print(f"  no NaN/Inf in any emitted float array                  OK")
    print(f"\nhas_partner {int(hasp.sum())}/{NEA}   has_head {int(have.sum())}/{NEA}")
    print(f"wrote {path} ({os.path.getsize(path)/1e6:.1f} MB)  in {time.time()-t0:.0f}s")
    print(f"\nfam_bilateral.HJIT is calibrated on a {FB_SPACING} mm head spacing; this build "
          f"measured {P['spacing'].mean():.3f} mm mean. "
          + ("consistent." if abs(P['spacing'].mean() - FB_SPACING) < 0.5 else
             f"RE-EXPORT HEAD_SPACING_MM={P['spacing'].mean():.3f} to the GPU box."))
    if LIMIT:
        # every per-ear array is allocated at full E, so a LIMIT run writes a FULL-SIZE
        # file whose head columns are zero past LIMIT -- the size is already final
        print(f"LIMIT={LIMIT}: ears {NE}..{NEA-1} carry has_head=0 and zeroed head columns. "
              f"The file size ({os.path.getsize(path)/1e6:.0f} MB) is already the full-run size.")
    return path


# --------------------------------------------------------------- smoke test
def uv_ellipsoid(nu=96, nv=48, ax=(95., 75., 85.)):
    """a closed triangulated ellipsoid standing in for a head scan (CPU smoke test only)"""
    u = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    v = np.linspace(0.03, np.pi - 0.03, nv)
    UU, VV = np.meshgrid(u, v, indexing="ij")
    V = (np.stack([np.sin(VV) * np.cos(UU), np.sin(VV) * np.sin(UU),
                   np.cos(VV) * np.ones_like(UU)], -1).reshape(-1, 3) * np.array(ax))
    ij = np.arange(nu * nv).reshape(nu, nv)
    a, d = ij[:, :-1], ij[:, 1:]
    b, c = np.roll(ij, -1, 0)[:, :-1], np.roll(ij, -1, 0)[:, 1:]
    F = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3),
                        np.stack([a, c, d], -1).reshape(-1, 3)]).astype(np.int64)
    if signed_volume(V, F) < 0:
        F = F[:, [0, 2, 1]]
    return V, F


def smoke():
    """Synthetic 2-subject dataset driven through the REAL per-ear code path, then through
    fam_bilateral's model on the emitted layout. Asserts the landmark output is (2,85,3)."""
    import torch
    import train_family as T
    import fam_bilateral as FB
    t0 = time.time()
    torch.manual_seed(0)            # the printed losses are quoted as exact output, so the
                                    # model init has to be reproducible run to run
    hpts, npts, M, NEA = 512, 256, 2, 4
    V0, F0 = uv_ellipsoid()
    VN0 = vertex_normals(V0, F0)
    print(f"synthetic head: {len(V0)} verts {len(F0)} faces  signed volume "
          f"{signed_volume(V0, F0):.3e} mm^3   HPTS={hpts} HRADIUS={HRADIUS}")

    # the mirroring rule, measured on the synthetic mesh: the two wrong routes must both
    # land on the INWARD normal
    Vw, Fw, VNw = mirror_side(V0, F0, VN0, "right")
    for nm, ref, want in (("+M*n", VN0 * MIRROR, +1.0), ("-(M*n)", -VN0 * MIRROR, -1.0),
                          ("same-winding recompute", vertex_normals(V0 * MIRROR, F0), -1.0)):
        d = float((VNw * ref).sum(1).mean())
        print(f"  dot(flip-and-recompute, {nm:22s}) = {d:+.4f}  (expect {want:+.0f})")
        assert abs(d - want) < 0.02, f"{nm}: {d}"

    rs = np.random.RandomState(0)
    order = [("S0", "left"), ("S0", "right"), ("S1", "left"), ("S1", "right")]
    Rm = np.zeros((NEA, 3, 3), np.float32); c0 = np.zeros((NEA, 3), np.float32)
    clouds = np.zeros((NEA, M, npts, 3), np.float32)
    nrm = np.zeros((NEA, M, npts, 3), np.float32)
    coarse = np.zeros((NEA, NL, 3), np.float32)
    head = np.zeros((NEA, hpts, 3), np.float32); hnrm = np.zeros_like(head)
    for i, (pid, side) in enumerate(order):
        Vs, Fs, VNs = mirror_side(V0 * (1.0 + 0.05 * (i // 2)), F0, VN0, side)
        A = np.linalg.qr(rs.randn(3, 3))[0]
        A *= np.sign(np.linalg.det(A))
        cen = Vs[rs.randint(len(Vs))]
        Rm[i], c0[i] = A, cen
        Pc = (Vs - cen) @ A.T
        Nc = VNs @ A.T
        a2 = np.linalg.norm(np.cross(Vs[Fs[:, 1]] - Vs[Fs[:, 0]],
                                     Vs[Fs[:, 2]] - Vs[Fs[:, 0]]), axis=1)
        head[i], hnrm[i], dg = head_cloud(Pc, Nc, a2, Fs, hpts, HRADIUS, 7919 * i)
        print(f"  ear {i} {pid}/{side:5s}: src {dg['src']:6d} -> {hpts} pts, cell "
              f"{dg['cell']:.2f}mm, spacing mean {dg['spacing']:.3f} p99 {dg['sp_p99']:.3f} mm, "
              f"area {dg['area']:.0f} mm^2")
        # an "ear crop": the 40 mm neighbourhood of c0, sampled like build_screen_extra
        near = np.flatnonzero(np.linalg.norm(Pc, axis=1) < 40.0)
        coarse[i] = Pc[near[rs.choice(len(near), NL, replace=False)]] * 0.6
        for j in range(M):
            k = np.random.RandomState(1000 + 97 * i + j).randint(0, len(near), npts)
            clouds[i, j] = Pc[near[k]].astype(np.float32)
            nrm[i, j] = Nc[near[k]].astype(np.float32)
    sw = np.arange(NEA) ^ 1
    d = dict(clouds=clouds, nrm=nrm, coarse=coarse, true=coarse + 0.5, R=Rm, c0=c0,
             split=np.array(["train"] * NEA), pcp=clouds[sw], pcp_nrm=nrm[sw],
             head=head, head_nrm=hnrm, headp=head[sw], headp_nrm=hnrm[sw],
             has_partner=np.ones(NEA, np.float32), has_head=np.ones(NEA, np.float32))
    for i in range(NEA):
        assert np.array_equal(d["headp"][i], d["head"][i ^ 1])
        assert np.array_equal(d["pcp"][i], d["clouds"][i ^ 1])
    print(f"  emitted layout OK: headp[e] == head[e^1], pcp[e] == clouds[e^1] for all "
          f"{NEA} ears; keys {sorted(d)}")

    # --- drive fam_bilateral on the EMITTED layout, through the trainer's own plumbing
    import tempfile
    tmp = os.path.join(tempfile.gettempdir(), "bilatprep_smoke")
    os.makedirs(tmp, exist_ok=True)
    dp = f"{tmp}/bilat_smoke.npz"
    np.savez(dp, **d)
    npar = None
    for mode in FB.MODES:
        os.environ["MODE"] = mode
        import importlib
        fb = importlib.reload(FB)
        data = T.load_data(dp, fb.MODEL.NEEDS, "cpu")
        meta = dict(nl=NL, contours=T.CONTOURS, scale=30.0, npts=npts, fold=0, dev="cpu",
                    n_train_ears=NEA, artefacts={})
        cfg = {**T.TRAIN_DEFAULTS, **fb.MODEL.DEFAULTS}
        m = fb.MODEL(cfg, meta)
        b = T.make_batch(data, [0, 1], [[0], [1]], 1)
        tg = data["true"][torch.tensor([0, 1])]
        b, tg = fb.MODEL.AUGMENT(b, tg, cfg, fb.MODEL.ROTATES,
                                torch.Generator().manual_seed(0))
        o = m(T._flatten_samples(b, 1))
        L = T.default_loss(o, tg, m, b)
        L.backward()
        gn = sum(float(p.grad.norm()) for p in m.parameters() if p.grad is not None)
        npar = npar or sum(p.numel() for p in m.parameters())
        assert o["pred"].shape == (2, NL, 3), o["pred"].shape
        assert sum(p.numel() for p in m.parameters()) == npar and gn > 0
        print(f"  [{mode:15s}] pred {tuple(o['pred'].shape)}  params {npar:,}  "
              f"loss {float(L):.3f}  grad-norm {gn:.2f}  needs {fb.MODEL.NEEDS}")
    print(f"SMOKE OK ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    smoke() if int(os.environ.get("SMOKE", "0")) else run()
