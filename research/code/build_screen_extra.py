"""
Extra screening inputs for the two variants that cannot reuse the baseline clouds.

  screen_data_4096.npz   M independent 4096-point samples per ear, sampled WITHOUT
                         replacement (falls back to with-replacement only for crops
                         holding fewer than 4096 vertices; the count is reported).
                         For variant `pts4096`, which also rescales K=96 / GK=40 so the
                         physical neighbourhood window is held constant, not the index count.

  screen_norm_2048.npz   consistently oriented per-point normals for the EXISTING
                         2048-point clouds, so variant `normals` changes the feature
                         vector and nothing else.

Orientation. Vertex normals are accumulated from triangle normals on the ORIGINAL mesh
winding, where they are outward-consistent; the script asserts this by checking the mean
dot product with the outward radial direction. A right ear is mirrored by
diag(1,-1,1), a reflection, which reverses orientation -- so its normals are transformed
as n -> -(MIRROR * n), not simply mirrored. Normals are then rotated into the per-ear
canonical frame by R^T (rotation only, no translation).

The 2048 sample indices are regenerated with the identical RNG stream used by
build_multisample_all.py and the resulting coordinates are asserted equal to
all_multisample.npz, so the normals are guaranteed to line up point-for-point.

    python research/code/build_screen_extra.py
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
M = int(os.environ.get("M", "4"))

d = np.load("scratch/deep_dataset.npz", allow_pickle=True)
coarse, true, Rm, c0, split = d["coarse"], d["true"], d["R"], d["c0"], d["split"]
ref = np.load("scratch/all_multisample.npz")["clouds"]

tr = get_split("train", mesh_dir=Path(MESH)); va = get_split("val", mesh_dir=Path(MESH))
order = [(p, s) for p in tr for s in ("left", "right")] + \
        [(p, s) for p in va for s in ("left", "right")]
assert len(order) == len(split) == len(ref)
ds = Dataset(MESH, LM); pid2idx = {p: i for i, p in enumerate(ds.subject_ids)}
NE = len(order)

cl4 = np.zeros((NE, M, 4096, 3), np.float32)
nr2 = np.zeros((NE, M, 2048, 3), np.float32)
cache = {}
short, crop_sizes, orient_checks = 0, [], []
for i, (pid, side) in enumerate(order):
    if pid not in cache:
        m = ds[pid2idx[pid]][0]
        V0 = np.asarray(m.vertices, np.float64); F = np.asarray(m.faces)
        # outward-consistent vertex normals from the original winding
        fn = np.cross(V0[F[:, 1]] - V0[F[:, 0]], V0[F[:, 2]] - V0[F[:, 0]])
        VN = np.zeros_like(V0)
        for c in range(3):
            np.add.at(VN, F[:, c], fn)                 # area-weighted
        nn = np.linalg.norm(VN, axis=1, keepdims=True)
        VN = np.where(nn > 1e-12, VN / np.maximum(nn, 1e-12), np.array([0., 0., 1.]))
        rad = V0 - V0.mean(0)
        rad /= np.maximum(np.linalg.norm(rad, axis=1, keepdims=True), 1e-12)
        cache[pid] = (V0, VN, float((VN * rad).sum(1).mean()))
        if len(cache) > 12:
            cache.pop(next(iter(cache)))
    V0, VN0, odot = cache[pid]
    orient_checks.append(odot)
    if side == "right":
        V, VN = V0 * MIRROR, -(VN0 * MIRROR)           # reflection reverses orientation
    else:
        V, VN = V0, VN0
    R, cc = Rm[i], c0[i]
    cw = coarse[i] @ R + cc
    lo, hi = cw.min(0) - MARGIN, cw.max(0) + MARGIN
    msk = np.all((V >= lo) & (V <= hi), axis=1)
    sel = msk if msk.any() else np.ones(len(V), bool)
    cl = (V[sel] - cc) @ R.T
    cn = VN[sel] @ R.T                                 # rotate normals (unit-preserving)
    crop_sizes.append(len(cl))
    for j in range(M):
        # --- 2048 normals: regenerate the EXACT baseline sample indices
        idx = np.random.RandomState(1000 + 97 * i + j).randint(0, len(cl), 2048)
        assert np.allclose(cl[idx].astype(np.float32), ref[i, j], atol=1e-5), \
            f"ear {i} sample {j}: regenerated cloud differs from all_multisample"
        nr2[i, j] = cn[idx].astype(np.float32)
        # --- 4096 clouds: independent stream, without replacement where possible
        rng = np.random.RandomState(50000 + 131 * i + j)
        if len(cl) >= 4096:
            k = rng.choice(len(cl), 4096, replace=False)
        else:
            k = rng.randint(0, len(cl), 4096)
            short += 1
        cl4[i, j] = cl[k].astype(np.float32)
    if (i + 1) % 60 == 0:
        print(f"  {i+1}/{NE}", flush=True)

om = float(np.mean(orient_checks))
assert om > 0.5, f"vertex normals are not outward-consistent (mean radial dot {om:.3f})"
print(f"\noutward-orientation check: mean dot(normal, radial) = {om:.3f}")
cs = np.array(crop_sizes)
print(f"crop vertices: min {cs.min()} median {int(np.median(cs))} max {cs.max()}")
print(f"4096 samples needing replacement: {short}/{NE*M}")
nl = np.linalg.norm(nr2.reshape(-1, 3), axis=1)
print(f"normal unit-norm: min {nl.min():.4f} max {nl.max():.4f}")

np.savez_compressed("scratch/screen_data_4096.npz", clouds=cl4, coarse=coarse,
                    true=true, R=Rm, c0=c0, split=split)
# gpu_screen.py reads normals from a `nrm` key INSIDE the data file, so ship a combined
# file; run the variant with DATA=<work>/screen_data_2048nrm.npz. Without this the
# `normals` variant would silently train on XYZ only and measure nothing.
np.savez_compressed("scratch/screen_data_2048nrm.npz", clouds=np.load(
    "scratch/screen_data_2048.npz")["clouds"], nrm=nr2, coarse=coarse, true=true,
    R=Rm, c0=c0, split=split)
for f in ("screen_data_4096.npz", "screen_data_2048nrm.npz"):
    print(f"wrote scratch/{f} ({os.path.getsize('scratch/'+f)/1e6:.1f} MB)")
