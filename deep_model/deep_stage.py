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
from .surfproj import SurfaceProjector

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


def deep_refine(mesh_verts, coarse_world, ensemble, mean_shape, side="left", npts=2048,
                mesh_faces=None, dense_ssm=None, ssm_alpha=0.3):
    """Return deep-refined (85,3) landmarks in world frame. `side` handles mirroring.

    If `mesh_faces` is given, the predictions are projected onto the mesh SURFACE
    (exact point-to-triangle). The ground truth lies 0.006mm from the surface while
    raw predictions sit ~0.17mm off it, so this is a systematic gain: measured
    1.329 -> 1.309mm on validation, improving 100% of ears (paired t-test p=2e-29).
    """
    mv, cw = mesh_verts, coarse_world
    if side == "right":
        mv = mesh_verts * MIRROR
        cw = coarse_world * MIRROR
    cloud, coarse_canon, R, c0 = _frame(mv, cw, mean_shape, npts)
    pred_canon = ensemble.predict(cloud, coarse_canon)
    world = pred_canon @ R + c0                       # mirrored frame if side==right
    if mesh_faces is not None:
        F = np.asarray(mesh_faces)
        world = project_to_surface(world, mv, F)
        if dense_ssm is not None:
            # dense-SSM hybrid fit (surface + these landmarks), blended, then re-projected
            cl = _surface_points(world, mv, margin=12.0)
            world = dense_ssm.refine(cl, world, alpha=ssm_alpha)
            world = project_to_surface(world, mv, F)
    if side == "right":
        world = world * MIRROR
    return world


def _surface_points(around, verts, margin=12.0, npts=16384, seed=0):
    """crop the mesh vertices around `around` and subsample (target for the SSM fit)"""
    lo, hi = around.min(0) - margin, around.max(0) + margin
    m = np.all((verts >= lo) & (verts <= hi), axis=1)
    pts = verts[m] if m.any() else verts
    if len(pts) > npts:
        pts = pts[np.random.RandomState(seed).choice(len(pts), npts, replace=False)]
    return pts


def project_to_surface(pts, verts, faces, margin=8.0):
    """snap points onto the mesh surface (normal direction, minimal tangential motion)"""
    lo, hi = pts.min(0) - margin, pts.max(0) + margin
    vin = np.all((verts >= lo) & (verts <= hi), axis=1)
    fmask = vin[faces].any(axis=1)
    Fs = faces[fmask]
    if len(Fs) == 0:
        return pts
    keep = np.unique(Fs)
    remap = -np.ones(len(verts), int); remap[keep] = np.arange(len(keep))
    out, _ = SurfaceProjector(verts[keep], remap[Fs]).project(pts)
    return out


def load_ensemble(weight_paths, ssm_mean, ssm_comp, blend=0.3, tta=False):
    return DeepEnsemble(weight_paths, ssm_mean, ssm_comp, blend=blend, tta=tta)
