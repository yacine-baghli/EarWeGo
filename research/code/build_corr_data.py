"""
Data prep for the dense-correspondence / dense-SSM architecture.

Outputs ONE npz containing:
  template_V/F        the template ear surface (train ear closest to the GPA mean)
  bary_f, bary_w      the 85 landmarks as barycentric points ON the template
  clouds  (340,P,3)   per-ear cropped full-res surface point clouds
  gt_lms  (340,85,3)  ground-truth landmarks (evaluation; anchors for TRAIN only)
  init_lms(340,85,3)  init for the fit: GT for train ears, DEEP PREDICTIONS for val
                      (val crops/inits never use GT -> no leakage)
  split   (340,)
All in the mirrored-left world frame (right ears mirrored), consistent with the
rest of the pipeline.
"""
import os, sys, time
import numpy as np
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_all_landmarks
from src.splits import get_split
from src.dataset import Dataset
from src.geometry import procrustes_align
from nicp import barycentric_of, transport

MESH = "2026 Munich Tech Arena - Datas/2026 Munich Tech Arena - Datas/mesh"
LM = "2026 Munich Tech Arena - Datas/2026 Munich Tech Arena - Datas/landmarks"
MIRROR = np.array([1., -1., 1.])
CROP = 12.0
NPTS = 16384
OUT = "scratch/corr_data.npz"

allm = load_all_landmarks()
tr = get_split("train", mesh_dir=Path(MESH)); va = get_split("val", mesh_dir=Path(MESH))
ds = Dataset(MESH, LM); pid2idx = {p: i for i, p in enumerate(ds.subject_ids)}
rng = np.random.RandomState(0)
_cache = {}


def mesh_of(pid, side):
    if pid not in _cache:
        m = ds[pid2idx[pid]][0]
        _cache[pid] = (np.asarray(m.vertices), np.asarray(m.faces))
    V, F = _cache[pid]
    if side == "right":
        V = V * MIRROR
        F = F[:, ::-1]
    return V, F


def lms_of(pid, side):
    L = allm[pid][side].copy()
    if side == "right":
        L[:, 1] *= -1
    return L


def crop(V, F, around, margin=CROP, want_faces=False):
    lo, hi = around.min(0) - margin, around.max(0) + margin
    vin = np.all((V >= lo) & (V <= hi), axis=1)
    fm = vin[F].all(axis=1)
    Fs = F[fm]
    keep = np.unique(Fs)
    remap = -np.ones(len(V), int); remap[keep] = np.arange(len(keep))
    return (V[keep], remap[Fs]) if want_faces else V[keep]


# ---------------- template = train ear closest to the GPA mean ----------------
shapes, ids = [], []
for p in tr:
    for s in ("left", "right"):
        shapes.append(lms_of(p, s)); ids.append((p, s))
shapes = np.array(shapes)
mean_shape = shapes.mean(0)
al = np.array([procrustes_align(S, mean_shape, allow_scale=True)[0] for S in shapes])
d2m = np.linalg.norm(al - mean_shape, axis=2).mean(1)
ti = int(np.argmin(d2m)); tpid, tside = ids[ti]
TL = lms_of(tpid, tside)
TV, TF = crop(*mesh_of(tpid, tside), around=TL, want_faces=True)
print(f"template {tpid} {tside}: {len(TV)} verts {len(TF)} faces (mean-shape dist {d2m[ti]:.3f}mm)")
t0 = time.time()
bf, bw = barycentric_of(TL, TV, TF)
err = np.linalg.norm(transport(TF[bf], bw, TV) - TL, axis=1).mean()
print(f"barycentric landmark map: err {err:.4f}mm ({time.time()-t0:.0f}s)")

# ---------------- per-ear clouds ---------------------------------------------
seeds = [np.load(f"scratch/gpu_cont_s{s}_valpred.npz") for s in range(4)]
PRED = np.mean([z["raw"] for z in seeds], axis=0)          # val preds, order va (L,R)
order = [("train", p, s) for p in tr for s in ("left", "right")] + \
        [("val", p, s) for p in va for s in ("left", "right")]
clouds = np.zeros((len(order), NPTS, 3), np.float32)
gt_lms = np.zeros((len(order), 85, 3), np.float32)
init_lms = np.zeros((len(order), 85, 3), np.float32)
splits = []
vi = 0
for i, (sp, pid, side) in enumerate(order):
    V, F = mesh_of(pid, side)
    gt = lms_of(pid, side)
    if sp == "train":
        init = gt                                   # anchors available for training ears
        around = gt
    else:
        init = PRED[vi]; vi += 1                    # NO GT at test time
        around = init
    pts = crop(V, F, around=around)
    idx = rng.randint(0, len(pts), NPTS) if len(pts) < NPTS else \
        rng.choice(len(pts), NPTS, replace=False)
    clouds[i] = pts[idx]; gt_lms[i] = gt; init_lms[i] = init; splits.append(sp)
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(order)} ears", flush=True)

np.savez_compressed(OUT, template_V=TV.astype(np.float32), template_F=TF.astype(np.int32),
                    bary_f=bf.astype(np.int32), bary_w=bw.astype(np.float32),
                    clouds=clouds, gt_lms=gt_lms, init_lms=init_lms,
                    split=np.array(splits), template_pid=f"{tpid}_{tside}")
print(f"saved {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB) "
      f"[{splits.count('train')} train / {splits.count('val')} val]")
