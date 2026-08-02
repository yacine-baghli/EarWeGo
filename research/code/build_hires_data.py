"""
HIGH-RESOLUTION SURFACE CLOUDS for Family A (8k / 16k / 32k points).

Every cloud so far was built by sampling mesh VERTICES. That caps the honest resolution:
crop vertex counts run 8582 (min) to 52955 (median 21859), so 8192 without replacement is
only just feasible and 16384/32768 are impossible for most ears -- you would be sampling
the same vertex several times and calling it a denser cloud.

This samples the SURFACE instead: pick a face with probability proportional to its area,
then a uniform barycentric point inside it. Density is then unbounded and independent of
the tessellation, which also removes a bias the vertex sampler had -- vertex sampling
over-represents finely-tessellated regions (high curvature) relative to their area.

Normals are barycentrically interpolated from the face's vertex normals and renormalised,
so they vary smoothly rather than snapping to vertex values.

Orientation is handled exactly as build_screen_extra.py does, and for the same reason: a
right ear is mirrored by diag(1,-1,1), which is a reflection, so the winding is flipped
and normals are RECOMPUTED in the shipped space rather than sign-flipped by hand. Shipped
normals are asserted per ear against their own winding.

    NPTS=8192 M=4 python research/code/build_hires_data.py     # -> screen_data_8192nrm.npz

Writes scratch/screen_data_<NPTS>nrm.npz with clouds (E,M,NPTS,3) and nrm (E,M,NPTS,3).

DTYPE=fp16 halves the file. train_family.load_data does `torch.tensor(d[k]).float()`, so
the network still sees float32 and NOTHING downstream changes. The cost is a storage
quantisation of the point positions: fp16 has an 11-bit significand, so the step at a
coordinate of magnitude |x| in [16,32)mm is 2^-11*16 = 0.0156mm and in [32,64)mm it is
0.03125mm. Measured on the real 8192 clouds the round-trip displacement is RMS 0.0043mm /
max 0.0206mm (see res_sweep_prep.py), i.e. 0.4% of the 1.09mm point spacing and 0.4% of
the 1.1776mm error the sweep is trying to move. Normals cost 1.6e-4 rad. Use it whenever
disk or upload bandwidth binds; do NOT use it for anything that stores GROUND TRUTH.
"""
import os, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.splits import get_split
from src.dataset import Dataset

MESH = "2026 Munich Tech Arena - Datas/2026 Munich Tech Arena - Datas/mesh"
LM = "2026 Munich Tech Arena - Datas/2026 Munich Tech Arena - Datas/landmarks"
MIRROR = np.array([1., -1., 1.])
MARGIN = 14.0
NPTS = int(os.environ.get("NPTS", "8192"))
M = int(os.environ.get("M", "4"))
LIMIT = int(os.environ.get("LIMIT", "0"))
# Shard so a killed run costs one shard, not the whole build. The first attempt died
# silently at ~160/340 with an empty stderr (external termination, not a traceback), and
# losing 40 minutes of work to that is avoidable. Shards merge with SHARD=merge.
SHARD = os.environ.get("SHARD", "")
NSHARD = int(os.environ.get("NSHARD", "4"))
DT = np.float16 if os.environ.get("DTYPE", "fp32") == "fp16" else np.float32


def vertex_normals(V, F):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    VN = np.zeros_like(V)
    for c in range(3):
        np.add.at(VN, F[:, c], fn)
    nr = np.linalg.norm(VN, axis=1, keepdims=True)
    return np.where(nr > 1e-12, VN / np.maximum(nr, 1e-12), np.array([0., 0., 1.]))


d = np.load("scratch/deep_dataset.npz", allow_pickle=True)
coarse, true, Rm, c0, split = d["coarse"], d["true"], d["R"], d["c0"], d["split"]
tr = get_split("train", mesh_dir=Path(MESH)); va = get_split("val", mesh_dir=Path(MESH))
order = [(p, s) for p in tr for s in ("left", "right")] + \
        [(p, s) for p in va for s in ("left", "right")]
assert len(order) == len(split)
ds = Dataset(MESH, LM); pid2idx = {p: i for i, p in enumerate(ds.subject_ids)}
NE = LIMIT if LIMIT else len(order)

if SHARD == "merge":                     # stitch the shards into the final file
    cl = np.zeros((NE, M, NPTS, 3), DT)
    nr = np.zeros((NE, M, NPTS, 3), DT)
    seen = np.zeros(NE, bool)
    for k in range(NSHARD):
        p = f"scratch/_hires{NPTS}_sh{k}.npz"
        assert os.path.exists(p), f"shard {k} missing -- run SHARD={k} first"
        z = np.load(p)
        assert z["clouds"].dtype == DT, \
            f"shard {k} is {z['clouds'].dtype} but DTYPE says {np.dtype(DT)}"
        assert z["clouds"].shape[1] == M, f"shard {k} has M={z['clouds'].shape[1]}, not {M}"
        idx = z["idx"]
        cl[idx] = z["clouds"]; nr[idx] = z["nrm"]; seen[idx] = True
        # DELSHARD=1 drops each shard the moment it is in RAM, so the peak DISK need is
        # max(all shards, the output) rather than their sum. At 32768/M=4 that is 0.49GB
        # instead of 0.97GB, which is the difference between fitting and not on a machine
        # with ~1.2GB free. It also means a failed merge costs the whole rebuild, so it is
        # off by default.
        if os.environ.get("DELSHARD") == "1":
            del z
            os.remove(p)
    assert seen.all(), f"{(~seen).sum()} ears missing after merge"
    out = f"scratch/screen_data_{NPTS}nrm.npz"
    np.savez_compressed(out, clouds=cl, nrm=nr, coarse=coarse[:NE], true=true[:NE],
                        R=Rm[:NE], c0=c0[:NE], split=split[:NE])
    nl = np.linalg.norm(nr.reshape(-1, 3).astype(np.float32), axis=1)
    print(f"merged {NSHARD} shards -> {out} ({os.path.getsize(out)/1e6:.1f} MB) "
          f"clouds {cl.shape} {cl.dtype} | normal norm {nl.min():.5f}..{nl.max():.5f}")
    sys.exit(0)

RANGE = range(NE) if SHARD == "" else range(int(SHARD), NE, NSHARD)

IDX = np.array(list(RANGE))
cl = np.zeros((len(IDX), M, NPTS, 3), DT)
nr = np.zeros((len(IDX), M, NPTS, 3), DT)
cache = {}
areas, nface, spacing = [], [], []
for slot, i in enumerate(IDX):
    pid, side = order[i]
    if pid not in cache:
        m = ds[pid2idx[pid]][0]
        V0 = np.asarray(m.vertices, np.float64); F0 = np.asarray(m.faces, np.int64)
        cache = {pid: (V0, F0, vertex_normals(V0, F0))}
    V0, F0, VN0 = cache[pid]
    if side == "right":                      # reflection: flip winding, recompute normals
        V, F = V0 * MIRROR, F0[:, [0, 2, 1]]
        VN = vertex_normals(V, F)
    else:
        V, F, VN = V0, F0, VN0
    chk = float((VN * vertex_normals(V, F)).sum(1).mean())
    assert chk > 0.99, f"{pid}/{side}: shipped normals disagree with winding ({chk:.3f})"

    R, cc = Rm[i].astype(np.float64), c0[i].astype(np.float64)
    cw = coarse[i] @ R + cc
    lo, hi = cw.min(0) - MARGIN, cw.max(0) + MARGIN
    vin = np.all((V >= lo) & (V <= hi), axis=1)
    fm = vin[F].all(axis=1)                  # keep faces fully inside the crop
    Fs = F[fm] if fm.any() else F
    Vc = (V - cc) @ R.T                      # canonical frame (rotation preserves area)
    Nc = VN @ R.T
    A, B, C = Vc[Fs[:, 0]], Vc[Fs[:, 1]], Vc[Fs[:, 2]]
    ar = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)
    tot = ar.sum()
    assert tot > 0, f"ear {i}: crop has zero area"
    p = ar / tot
    areas.append(tot); nface.append(len(Fs))
    spacing.append(np.sqrt(tot / NPTS))      # mean point spacing for this NPTS
    for j in range(M):
        rng = np.random.RandomState(90000 + 271 * i + j)
        f = rng.choice(len(Fs), NPTS, p=p)   # area-weighted face choice
        u, v = rng.rand(NPTS), rng.rand(NPTS)
        flip = u + v > 1.0                   # fold into the triangle
        u[flip], v[flip] = 1.0 - u[flip], 1.0 - v[flip]
        w = np.stack([1.0 - u - v, u, v], 1)[:, :, None]
        tri = Fs[f]
        cl[slot, j] = (w * Vc[tri]).sum(1).astype(DT)
        n = (w * Nc[tri]).sum(1)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        nr[slot, j] = n.astype(DT)
    if (slot + 1) % 20 == 0:
        print(f"  {slot+1}/{len(IDX)} (ear {i})", flush=True)

sp = np.array(spacing)
print(f"\ncrop faces  min {min(nface)} median {int(np.median(nface))} max {max(nface)}")
print(f"crop area mm^2  min {min(areas):.0f} median {np.median(areas):.0f} max {max(areas):.0f}")
print(f"mean point spacing at {NPTS} pts: {sp.mean():.4f}mm "
      f"(min {sp.min():.4f} max {sp.max():.4f})")
nl = np.linalg.norm(nr.reshape(-1, 3).astype(np.float32), axis=1)
print(f"normal unit-norm ({np.dtype(DT)}): min {nl.min():.5f} max {nl.max():.5f}")
assert abs(nl.min() - 1) < 2e-3 and abs(nl.max() - 1) < 2e-3

if SHARD == "":
    out = f"scratch/screen_data_{NPTS}nrm.npz" if not LIMIT else \
          f"scratch/screen_data_{NPTS}nrm_lim{LIMIT}.npz"
    np.savez_compressed(out, clouds=cl, nrm=nr, coarse=coarse[:NE], true=true[:NE],
                        R=Rm[:NE], c0=c0[:NE], split=split[:NE])
else:
    # a shard stores only its own ears plus the index map; SHARD=merge stitches them
    out = f"scratch/_hires{NPTS}_sh{SHARD}.npz"
    np.savez_compressed(out, clouds=cl, nrm=nr, idx=IDX)
print(f"wrote {out} ({os.path.getsize(out)/1e6:.1f} MB)  clouds {cl.shape}")
