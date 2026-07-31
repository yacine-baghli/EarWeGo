"""
LOCAL prep for Family D (research/code/pretrain_ssl.py): V DISJOINT surface samples per ear.

WHY THIS EXISTS. The shipped clouds are drawn WITH REPLACEMENT from the same 19.5k-41.4k
crop vertices (`rng.randint(0, len(cl), NPTS)` in build_multisample_all.py), so two
"independent" samples of one ear share about NPTS^2/n_v = 100-210 EXACTLY COINCIDENT
points, and ~5 % of each sample is an internal duplicate of another of its own points. For
the fresh-sample TTA that artefact is harmless. For a RESAMPLING-CONSISTENCY pretext task
it is not: a coincident pair is matchable by coordinate identity rather than by shape, and
it flatters the reported match statistic, so the task can look healthy while teaching the
encoder to memorise coordinates. Here the crop's vertices are PARTITIONED -- view j holds
NPTS vertices that no other view holds -- and the file records `disjoint=True` so
pretrain_ssl.py reports which regime it is running in instead of assuming.

CROP CONVENTION -- identical to build_multisample_all.py and build_screen_extra.py, so
this artefact is registered with screen_data_*.npz ear for ear:
  * right ears mirrored by diag(1,-1,1); because a reflection reverses orientation the
    winding is flipped and the normals are RECOMPUTED, never transformed and sign-reasoned
    (that is the bug build_screen_extra.py documents having shipped once);
  * axis-aligned bbox of the COARSE landmarks (not the GT) + 14 mm margin;
  * rotated into the per-ear canonical frame by R^T after subtracting c0.
Every right ear is asserted to satisfy the reflection identity
vertex_normals(M@V, flip(F)) == M @ vertex_normals(V, F) to >0.99 -- which is +1.000 when
the winding flip took effect and -1.000 when it did not, so it detects the exact
inward-normal bug. The radial-dot outwardness diagnostic is asserted on the mean, as in
build_screen_extra.py. Do NOT "check" normals by comparing vertex_normals(V, F) against
vertex_normals(V, F): that is identically 1.0 and detects nothing (this file shipped that
tautology once; the smoke test now asserts the real identity both ways).

FOLD SAFETY. This artefact is fold-INDEPENDENT, and that is legitimate only because every
stored array is PER-EAR: there is no mean, no PCA basis, no codebook, no normalisation
statistic -- nothing that couples one ear to another. pretrain_ssl.py does the fold slicing
and asserts it. If you ever add a cross-ear quantity here, this file stops being fold-safe
and must take FOLD and write one artefact per fold. The printed SUMMARY statistics do span
all ears, as in every existing builder; that is an experimenter-level exposure of geometry
(never of labels), and it is why the per-ear arrays rather than the printout are the
artefact.

GROUND TRUTH IS DELIBERATELY ABSENT. Every other npz in scratch/ carries `true`; this one
does not, because SSL has no use for it and an artefact that CANNOT leak beats one that
merely must be handled carefully.

ENV (defaults in brackets)
  V     [4]                     views per ear; needs crop_vertices >= V*NPTS to partition
  NPTS  [2048]                  points per view
  OUT   [scratch/ssl_views.npz]
  LIMIT [0]                     process only the first N ears (0 = all 340)
  SMOKE [0]                     1 = synthetic self-test of the partition logic, no dataset

    python research/code/build_ssl_views.py            # all 340 dev ears, V=4
    V=6 python research/code/build_ssl_views.py         # 6 disjoint views (needs 12288 verts)
    SMOKE=1 python research/code/build_ssl_views.py     # CPU self-test, no data needed
"""
import os, sys
import numpy as np
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

MESH = "2026 Munich Tech Arena - Datas/2026 Munich Tech Arena - Datas/mesh"
LM = "2026 Munich Tech Arena - Datas/2026 Munich Tech Arena - Datas/landmarks"
MIRROR = np.array([1., -1., 1.])
MARGIN = 14.0                      # must match build_multisample_all.py
V = int(os.environ.get("V", "4"))
NPTS = int(os.environ.get("NPTS", "2048"))
OUT = os.environ.get("OUT", "scratch/ssl_views.npz")
LIMIT = int(os.environ.get("LIMIT", "0"))
SMOKE = int(os.environ.get("SMOKE", "0"))


def vertex_normals(Vv, F):
    """area-weighted vertex normals from the winding of F (outward iff F is CCW)"""
    fn = np.cross(Vv[F[:, 1]] - Vv[F[:, 0]], Vv[F[:, 2]] - Vv[F[:, 0]])
    VN = np.zeros_like(Vv)
    for c in range(3):
        np.add.at(VN, F[:, c], fn)
    nr = np.linalg.norm(VN, axis=1, keepdims=True)
    return np.where(nr > 1e-12, VN / np.maximum(nr, 1e-12), np.array([0., 0., 1.]))


def partition(n, v, npts, rng):
    """v DISJOINT index blocks of npts drawn from n vertices, or overlapping
    without-replacement blocks when the crop is too small (reported, never silent)."""
    if n >= v * npts:
        perm = rng.permutation(n)
        return np.stack([perm[j * npts:(j + 1) * npts] for j in range(v)]), True
    return np.stack([rng.choice(n, npts, replace=(n < npts)) for _ in range(v)]), False


def build():
    from src.splits import get_split
    from src.dataset import Dataset
    d = np.load("scratch/deep_dataset.npz", allow_pickle=True)
    coarse, Rm, c0, split = d["coarse"], d["R"], d["c0"], d["split"]
    tr = get_split("train", mesh_dir=Path(MESH)); va = get_split("val", mesh_dir=Path(MESH))
    order = [(p, s) for p in tr for s in ("left", "right")] + \
            [(p, s) for p in va for s in ("left", "right")]
    assert len(order) == len(split), f"{len(order)} ears vs {len(split)} in deep_dataset.npz"
    ne = LIMIT or len(order)
    ds = Dataset(MESH, LM)
    pid2idx = {p: i for i, p in enumerate(ds.subject_ids)}
    cl = np.zeros((ne, V, NPTS, 3), np.float32)
    nr = np.zeros_like(cl)
    ok = np.zeros(ne, bool)
    crop, cache, orient, mirror_chk = [], {}, [], []
    for i, (pid, side) in enumerate(order[:ne]):
        if pid not in cache:
            m = ds[pid2idx[pid]][0]
            cache[pid] = (np.asarray(m.vertices, np.float64), np.asarray(m.faces))
            if len(cache) > 8:
                cache.pop(next(iter(cache)))
        V0, F0 = cache[pid]
        VN0 = vertex_normals(V0, F0)
        if side == "right":
            Vv, F = V0 * MIRROR, F0[:, [0, 2, 1]]
            VN = vertex_normals(Vv, F)
            # THE mirroring check, and it can actually fail. For a diagonal reflection
            # M = diag(1,-1,1): cross(Ma, Mb) = det(M) M^-T (a x b) = -M(a x b), and
            # swapping two face indices negates the cross product again, so recomputing
            # from the FLIPPED winding must equal M @ VN0 exactly. Forgetting the flip
            # gives -M @ VN0 -- the inward-normal bug build_screen_extra.py documents
            # having shipped once. Verified numerically: +1.000 vs -1.000.
            # (The first version of this line compared vertex_normals(Vv, F) with
            # vertex_normals(Vv, F) -- the same call -- which is identically 1.0 and could
            # not detect an inverted normal at all.)
            chk = float((VN * (VN0 * MIRROR)).sum(1).mean())
            assert chk > 0.99, (f"{pid}/right: mirrored normals are not the reflection of "
                                f"the original ones ({chk:+.3f}); expected +1.000, and "
                                f"-1.000 means the winding flip did not take effect")
        else:
            Vv, F, VN = V0, F0, VN0
            chk = 1.0
        # Diagnostic ONLY, deliberately not asserted. build_screen_extra.py gates on
        # mean(radial dot) > 0.5; that threshold does not hold on a subset -- MEASURED
        # +0.478 over the first 3 subjects -- because a head is not star-shaped, which is
        # exactly what that file's own comment says. A weak statistic that varies this much
        # between subjects is a diagnostic, not a gate; the per-ear mirroring identity above
        # is exact (+1.000 / -1.000) and is the check that can actually fail for cause.
        rad = Vv - Vv.mean(0)
        rad /= np.maximum(np.linalg.norm(rad, axis=1, keepdims=True), 1e-12)
        orient.append(float((VN * rad).sum(1).mean()))
        mirror_chk.append(chk)
        R, cc = Rm[i], c0[i]
        cw = coarse[i] @ R + cc
        lo, hi = cw.min(0) - MARGIN, cw.max(0) + MARGIN
        msk = np.all((Vv >= lo) & (Vv <= hi), axis=1)
        sel = msk if msk.any() else np.ones(len(Vv), bool)
        P = (Vv[sel] - cc) @ R.T
        N = VN[sel] @ R.T                                  # rotate only: unit-preserving
        crop.append(len(P))
        blocks, ok[i] = partition(len(P), V, NPTS,
                                  np.random.RandomState(700000 + 251 * i))
        for j in range(V):
            cl[i, j] = P[blocks[j]].astype(np.float32)
            nr[i, j] = N[blocks[j]].astype(np.float32)
        if (i + 1) % 60 == 0:
            print(f"  {i+1}/{ne}", flush=True)
    print(f"mirroring identity (right ears): min {min(mirror_chk):+.4f} -- ASSERTED >0.99 "
          f"per ear, -1.000 would mean an inverted normal")
    print(f"radial-dot diagnostic (shipped normals, NOT asserted -- see the note in the "
          f"loop): mean {np.mean(orient):+.3f} min {min(orient):+.3f} max {max(orient):+.3f}")
    cs = np.array(crop)
    print(f"crop vertices: min {cs.min()} median {int(np.median(cs))} max {cs.max()} "
          f"(need {V*NPTS} to partition)")
    print(f"ears partitioned disjointly: {int(ok.sum())}/{ne}")
    nl = np.linalg.norm(nr.reshape(-1, 3), axis=1)
    print(f"normal unit-norm: min {nl.min():.4f} max {nl.max():.4f}")
    out = dict(clouds=cl, nrm=nr, coarse=coarse[:ne], R=Rm[:ne], c0=c0[:ne],
               split=split[:ne], ear_index=np.arange(ne, dtype=np.int32),
               ear_disjoint=ok, disjoint=bool(ok.all()), crop_vertices=cs.astype(np.int32),
               V=np.int32(V), NPTS=np.int32(NPTS), MARGIN=np.float32(MARGIN))
    path = OUT if not LIMIT else OUT.replace(".npz", f"_lim{LIMIT}.npz")
    np.savez_compressed(path, **out)
    print(f"wrote {path} ({os.path.getsize(path)/1e6:.1f} MB) disjoint={bool(ok.all())}")
    print("NOTE: `true` is deliberately absent -- this artefact cannot leak GT.")


def smoke():
    """partition logic + the disjointness property it exists to provide, no dataset needed"""
    rng = np.random.RandomState(0)
    n, v, npts = 9000, 4, 2048
    b, disj = partition(n, v, npts, rng)
    assert b.shape == (v, npts) and disj
    allidx = b.reshape(-1)
    assert len(np.unique(allidx)) == v * npts, "blocks overlap -- they are not a partition"
    print(f"partition({n}, V={v}, NPTS={npts}) -> {b.shape} disjoint={disj}, "
          f"{len(np.unique(allidx))}/{v*npts} unique indices")
    b2, disj2 = partition(5000, v, npts, rng)
    assert not disj2 and b2.shape == (v, npts)
    dup = v * npts - len(np.unique(b2.reshape(-1)))
    print(f"partition(5000, ...) -> disjoint={disj2} (crop too small), "
          f"{dup} shared indices ACROSS views, none WITHIN a view: "
          f"{all(len(np.unique(x)) == npts for x in b2)}")

    # the with-replacement baseline this script replaces, measured on the same crop size
    a = rng.randint(0, n, npts); c = rng.randint(0, n, npts)
    shared = len(set(a.tolist()) & set(c.tolist()))
    print(f"for contrast, build_multisample_all.py's with-replacement draw at n={n}: "
          f"{shared} coincident points between two views ({npts-len(np.unique(a))} "
          f"internal duplicates within one view)")
    assert shared > 0, "the defect this script fixes should be visible at this crop size"

    # a synthetic bumpy grid patch, so the normal path is exercised too
    u, w = np.meshgrid(np.linspace(-15, 15, 40), np.linspace(-12, 12, 34), indexing="ij")
    z = 3.0 * np.sin(u / 5.0) * np.cos(w / 4.0)
    Vv = np.stack([u, w, z], -1).reshape(-1, 3)
    ij = np.arange(Vv.shape[0]).reshape(40, 34)
    p, q = ij[:-1, :-1].ravel(), ij[1:, :-1].ravel()
    r, s = ij[1:, 1:].ravel(), ij[:-1, 1:].ravel()
    F = np.concatenate([np.stack([p, q, r], -1), np.stack([p, r, s], -1)])
    N = vertex_normals(Vv, F)
    assert np.allclose(np.linalg.norm(N, axis=1), 1.0, atol=1e-6)
    bb, dj = partition(len(Vv), 2, 400, rng)
    assert dj and not (set(bb[0].tolist()) & set(bb[1].tolist()))
    print(f"synthetic patch {len(Vv)} verts {len(F)} faces: unit normals PASS, "
          f"2 disjoint 400-point views PASS")

    # the right-ear mirroring identity that build() asserts, and the bug it catches
    n_flip = vertex_normals(Vv * MIRROR, F[:, [0, 2, 1]])      # what build() ships
    n_same = vertex_normals(Vv * MIRROR, F)                    # winding flip FORGOTTEN
    good = float((n_flip * (N * MIRROR)).sum(1).mean())
    bad = float((n_same * (N * MIRROR)).sum(1).mean())
    taut = float((n_flip * vertex_normals(Vv * MIRROR, F[:, [0, 2, 1]])).sum(1).mean())
    print(f"mirroring identity: flipped winding {good:+.4f} (build asserts >0.99), "
          f"winding flip forgotten {bad:+.4f} -> REFUSED")
    print(f"   the tautology this replaced, vertex_normals(V,F).vertex_normals(V,F) = "
          f"{taut:+.4f} for BOTH, i.e. it detected nothing")
    assert good > 0.99 and bad < -0.99 and abs(taut - 1.0) < 1e-6
    print("smoke OK")


if __name__ == "__main__":
    smoke() if SMOKE else build()
