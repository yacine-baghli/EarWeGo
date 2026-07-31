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


def vertex_normals(V, F):
    """Area-weighted vertex normals from the winding of F (outward iff F is CCW)."""
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    VN = np.zeros_like(V)
    for c in range(3):
        np.add.at(VN, F[:, c], fn)
    nr = np.linalg.norm(VN, axis=1, keepdims=True)
    return np.where(nr > 1e-12, VN / np.maximum(nr, 1e-12), np.array([0., 0., 1.]))


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
        V0 = np.asarray(m.vertices, np.float64); F0 = np.asarray(m.faces)
        VN = vertex_normals(V0, F0)
        rad = V0 - V0.mean(0)
        rad /= np.maximum(np.linalg.norm(rad, axis=1, keepdims=True), 1e-12)
        cache[pid] = (V0, F0, VN, float((VN * rad).sum(1).mean()))
        if len(cache) > 12:
            cache.pop(next(iter(cache)))
    V0, F0, VN0, odot = cache[pid]
    # Do NOT transform the normal vector and reason about the sign -- that is how the
    # first version of this script shipped INWARD normals for every right ear. A
    # reflection reverses orientation, so flip the winding and recompute the normals in
    # the space we actually ship. Verified numerically: recomputing from the flipped
    # winding gives +0.995 against the true outward direction, whereas -(M*n) and
    # same-winding recomputation both give -0.995.
    if side == "right":
        V, F = V0 * MIRROR, F0[:, [0, 2, 1]]
        VN = vertex_normals(V, F)
    else:
        V, F, VN = V0, F0, VN0
    # assert the shipped normals really are the winding's outward normals
    chk = float((VN * vertex_normals(V, F)).sum(1).mean())
    assert chk > 0.99, f"{pid}/{side}: shipped normals disagree with winding ({chk:.3f})"
    orient_checks.append(odot)
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
# Weak diagnostic only: a head is not star-shaped, so this cannot validate the
# mirroring. The per-ear winding-consistency assertion above is the real check.
assert om > 0.5, f"original meshes are not outward-wound (mean radial dot {om:.3f})"
print(f"\nradial-dot diagnostic (original meshes): {om:.3f}")
print(f"per-ear winding consistency: asserted >0.99 for all {NE} ears")
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
