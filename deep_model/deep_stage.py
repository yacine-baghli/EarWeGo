"""
Deep refinement STAGE for the pipeline: mesh + coarse -> deep-refined 85 landmarks.

Frames exactly like scripts/preprocess_deep.py (rotation-align to the SSM mean
shape, center on the coarse centroid, mm-scale, crop, sample), runs the torch-free
DeepEnsemble, maps back to world. Right ears are handled by mirroring in/out.

Ship: this module + deep_infer_v2.py + deep_predict.py + the gpu_cont_s*.npz weights
+ ssm_mean/ssm_comp (from deep_dataset or the bundle). No torch needed.
"""
import numpy as np
from src.geometry import procrustes_align
from .deep_predict import DeepEnsemble

MIRROR = np.array([1., -1., 1.])


def _frame(mesh_verts, coarse_world, mean_shape, npts=2048, margin=14.0, seed=0):
    tf = procrustes_align(mean_shape, coarse_world, allow_scale=True)[1]
    R, c0 = tf["R"], tf["t_tgt"]                       # rotation, coarse centroid
    lo, hi = coarse_world.min(0) - margin, coarse_world.max(0) + margin
    m = np.all((mesh_verts >= lo) & (mesh_verts <= hi), axis=1)
    cl = (mesh_verts[m] - c0) @ R.T
    if len(cl) == 0:
        cl = (mesh_verts - c0) @ R.T
    idx = np.random.RandomState(seed).randint(0, len(cl), npts)
    return cl[idx], (coarse_world - c0) @ R.T, R, c0


def deep_refine(mesh_verts, coarse_world, ensemble, mean_shape, side="left", npts=2048):
    """Return deep-refined (85,3) landmarks in world frame. `side` handles mirroring."""
    mv, cw = mesh_verts, coarse_world
    if side == "right":
        mv = mesh_verts * MIRROR
        cw = coarse_world * MIRROR
    cloud, coarse_canon, R, c0 = _frame(mv, cw, mean_shape, npts)
    pred_canon = ensemble.predict(cloud, coarse_canon)
    world = pred_canon @ R + c0
    if side == "right":
        world = world * MIRROR
    return world


def load_ensemble(weight_paths, ssm_mean, ssm_comp, blend=0.3, tta=False):
    return DeepEnsemble(weight_paths, ssm_mean, ssm_comp, blend=blend, tta=tta)
