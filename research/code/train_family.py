"""
UNIVERSAL FAMILY TRAINER — the one training/evaluation harness every new model family
plugs into, so that a family contributes ONLY its architecture and its loss.

Why this file exists. Seven variants of the 2048-point static-DGCNN refinement family
were screened and all were null or harmful, and the error decomposition (along-contour
RMSE 1.4456 / across 0.7233 / normal 0.2822) was INVARIANT across all seven. The next
programme is therefore structural: template transfer, intrinsic surface operators,
dense coordinate fields. Those families differ in their inputs, their artefacts and
their losses, but they must be compared on exactly the same folds, with exactly the same
evaluation, against exactly the same reference. Re-deriving the harness per family is
how that comparability dies. Hence: one trainer, a registry, and a fixed report schema.

What this adds over gpu_screen.py (which it otherwise deliberately mirrors):
  * a family REGISTRY, so a new architecture needs zero trainer edits;
  * arbitrary config from the environment (CFG_*), so a search driver can sweep it;
  * an EXPLICIT-SPLIT mode used by search_driver.py's inner CV loop, kept in a
    separate output namespace so an inner-loop run can never be mistaken for an
    outer-fold report;
  * evaluation through the COMPLETE FINAL INFERENCE PIPELINE — fresh-sample TTA, exact
    point-to-triangle surface projection, dense-SSM MAP blend, re-projection — so the
    number a family is judged on is the number the pipeline would actually ship.
    Reported as ordered_MLE_full_mm alongside the raw-network ordered_MLE_mm.

    FAMILY=dgcnn SEED=0 FOLD=0 EPOCHS=1200 python3 train_family.py
    python research/code/train_family.py          # <- no FAMILY set: runs the smoke test


ENVIRONMENT (all optional except FAMILY; every value is echoed into the report)
------------------------------------------------------------------------------
  FAMILY           (required for a real run) key into REGISTRY below
  FAMILY_MODULE    override: import from this module instead of consulting REGISTRY
  FAMILY_CLASS     override: this class name (default: the module's `MODEL`)
  FOLD        0    outer fold 0..4 of the FROZEN subject-grouped split
  SEED        0    seeds torch (incl. cuda), numpy, and the batch sampler
  EPOCHS      1200
  VARIANT     $FAMILY   label written to the report (screen_compare.py keys on it)
  TAG         fam_<FAMILY>_s<SEED>_f<FOLD>   output basename in $WORK
  WORK        scratch          output directory
  DATA        $WORK/screen_data_2048.npz
  EVAL_EVERY  max(1, EPOCHS//12)  epochs between val evaluations
  TTA         4    fresh surface samples averaged at the FINAL evaluation
  ALIAS       1    also write screen_<VARIANT>_s<SEED>_f<FOLD>.{json,npy} so the
                   existing screen_compare.py / cv_verdict.py read the run unchanged.
                   Forced to 0 in explicit-split mode.
  CFG_JSON    {}   whole family config as one JSON object
  CFG_<NAME>       one config entry, auto-typed (int/float/bool/JSON/str).
                   CFG_LR=3e-4 CFG_KNN=32 CFG_RADII='[11,9,7.4]'
  FULL_EVAL   1    run the full-pipeline evaluation if its artefacts are present
  TRIS        $WORK/mesh_data.npz         packed per-ear triangles (contract below)
  SSM         $WORK/dense_ssm_f<FOLD>.npz per-FOLD dense SSM (contract below)
  ARTEFACTS   ""   optional per-FOLD fold-safe artefact npz handed to the family
  SSM_ALPHA   0.3  SSM weight in the blend        SSM_KUSE 120
  PROJ_MARGIN 8.0  bbox margin for candidate faces PROJ_K 12  candidate vertices/point
  SSM_CROP    12.0 crop margin for the SSM target SSM_NPTS 16384
  FRAME_TOL   5.0  max median coarse-to-mesh distance (mm) before the mesh is rejected
  TRAIN_EARS / VAL_EARS   JSON lists of ear indices. EXPLICIT-SPLIT mode; asserted
                   subject-disjoint. Only search_driver.py's inner loop sets these.


FAMILY CONTRACT
---------------
A family module exposes a class (named `MODEL`, or named in REGISTRY):

  cls.DEFAULTS      dict   config defaults, merged UNDER the CFG_* environment
  cls.SEARCH_SPACE  dict   name -> list of candidate values; search_driver.py samples
                           this. Documented grid, never exhaustive enumeration.
  cls.NEEDS         tuple  extra keys to pull out of DATA and hand to forward().
                           A key of shape (E,M,...) is sample-indexed like `clouds`;
                           a key of shape (E,...) is ear-indexed.
  cls.ROTATES       tuple  subset of NEEDS holding DIRECTION vectors, rotated (not
                           translated or scaled) by the augmentation. Default ('nrm',).
  cls.SAMPLES       int    simultaneous independent fresh samples per forward. 1 ->
                           batch['pc'] is (B,N,3); K>1 -> (B,K,N,3).
  cls.BATCH         callable(ears, samples, meta) -> dict, MERGED into the batch after
                           the default build. NEEDS only covers dense per-ear arrays
                           (leading dim E, or (E,M,...) for sample-indexed). A RAGGED
                           offset-packed mesh -- v_ptr/f_ptr, one-ring, spectral basis --
                           does not fit that shape and must be gathered and padded per
                           batch; this is the hook for it, and it is where a family's own
                           pack_batch/pad_batch belongs.
  cls.AUGMENT       callable(batch, tg, cfg, rotates, gen) -> (batch, tg), or None to
                           disable augmentation entirely. Default: default_augment, which
                           subsamples + rotates + scales + jitters the cloud and leaves
                           anything that is not shaped (B,S,N,C) alone. A family whose
                           BATCH adds mesh tensors MUST set this, because the default
                           augmenter would rotate the cloud and not the mesh.

  cls(cfg, meta)           meta = dict(nl, contours, scale, npts, fold, dev,
                                       n_train_ears, artefacts)
  model.forward(batch) -> dict
        batch = {'pc', 'coarse' (B,85,3), 'ear' (B,) long, **extras}
        return  {'pred': (B,85,3)}  optionally 'aux': [(B,85,3), ...] for deep
                supervision (geometric weights, last = strongest) and 'reg': scalar.
  model.loss(out, tg) or model.loss(out, tg, batch) -> scalar   OPTIONAL; default_loss
                is used if absent. The arity is detected from the signature, so a
                structural family whose loss lives on the mesh can take the batch.

LEAKAGE (constraint 2) IS STRUCTURAL HERE. `true` is never put into the batch dict —
the target is a separate argument that only reaches loss(). A family therefore cannot
read ground truth in a forward pass even by accident. Any artefact npz (ARTEFACTS, SSM)
must carry `fold` and `train_ear_mask`; the loader asserts the fold matches and that no
validation ear is in the mask, and REFUSES to load an artefact that cannot prove it.


NPZ CONTRACT 1 — TRIS, packed per-ear triangles (the GPU box has no mesh library)
--------------------------------------------------------------------------------
Meshes are ragged, so they ship flat-packed. Built LOCALLY (needs the mesh dataset),
consumed on the GPU box by numpy+scipy only. TWO layouts are accepted and the loader
sniffs which one it has:

  A. `verts`/`faces` + `v_ptr`/`f_ptr`, face indices GLOBAL into the flat vertex axis.
     This is exactly what research/code/build_mesh_data.py already emits
     (scratch/mesh_data.npz), so the intrinsic-operator families and this projector
     share ONE mesh artefact instead of each carrying their own.
  B. `V`/`F` + `v_off`/`f_off`, face indices LOCAL to their ear block. The minimal
     contract when all you can ship is triangles.

  verts / V    (sum_e n_e, 3) float32  vertices, PER-EAR CANONICAL FRAME
  faces / F    (sum_e m_e, 3) int32    triangles
  v_ptr/v_off  (E+1,) int64            ear e's block is [ptr[e] : ptr[e+1]]
  f_ptr/f_off  (E+1,) int64

CANONICAL FRAME, not world, deliberately: the network emits canonical coordinates, and
canonical->world is `p @ R + c0` with R a pure ROTATION (no scale), so projecting before
that map is metrically identical and needs no extra transform. Right-ear mirroring is
already baked into the canonical frame upstream, so the projector never sees a
reflection and never has to reverse a winding. Crop each ear generously
(build_mesh_data.py uses 14 mm around the coarse landmarks) — the projector's own bbox
pass narrows it further.

The frame is NOT taken on trust from a string. For every evaluated ear the loader
measures the distance from that ear's COARSE landmarks to the nearest mesh vertex and
asserts the median is under FRAME_TOL (default 5 mm). Calibration, measured on
scratch/screen_data_2048.npz against the 8192 pooled surface samples it already carries:
the coarse landmarks sit 0.457 mm from the surface (median over the 340 ears, 0.601 max)
while the point-sampling gap alone is 0.432 mm — i.e. the coarse init is essentially ON
the surface, and its 3.766 mm mean error against GT is almost entirely TANGENTIAL. So the
healthy value of this statistic is ~0.5 mm; 5 mm leaves ~10x headroom for a coarser
decimation, and a frame, mirroring or ear-ordering mismatch lands at 10-40 mm. The
measured value is reported as full_pipeline.coarse_to_mesh_mm, so it is that REPORTED
number, not the loose threshold, that catches a silent regression.

CAVEAT: build_mesh_data.py decimates to MAXV=12000 vertices by default while the native
crops hold 19.5k-41.4k. Projecting onto a decimated surface is NOT identical to
projecting onto the native mesh that produced the shipped 1.3144 mm number. Build it
with MAXV=0 for a projection-grade mesh, or treat the full-pipeline delta as indicative
and say which mesh produced it.

Projection runs ON THE GPU BOX rather than as a local post-pass on the saved .npy.
Reason: surfproj.py is already numpy+scipy only, so nothing new is needed, and keeping
it in-process means the raw and full-pipeline numbers come from the SAME best-val
checkpoint inside one run and land in one JSON. A local post-pass would put the shipped
number in a different file produced at a different time — precisely the gap where a
silent mismatch hides — and the search driver could never see it. If TRIS is absent the
run still completes: ordered_MLE_full_mm is null and full_pipeline.status says why.

Ship to the GPU box: train_family.py + the family module + surfproj.py + dense_fit.py
+ the data npz + (optionally) mesh_data.npz and dense_ssm_f<FOLD>.npz.


NPZ CONTRACT 2 — SSM, the per-FOLD dense shape model
----------------------------------------------------
Everything DenseSSMFit needs, plus the two fields that prove it is fold-safe:

  mean (3n,) / comps (K,3n) / eig (K,)  float   dense PCA basis
  template_F (m,3) int / bary_f (85,) int / bary_w (85,3) float
  fold () int                 the outer fold this model belongs to
  train_ear_mask (E,) bool    ears used to BUILD it

The shipped scratch/dense_ssm.npz is NOT usable here: it was fitted on the fixed
train/val split, so on four of the five folds it has seen ears that are now validation
ears. It also lacks template_F/bary_f/bary_w. Rebuilding it per fold is a local NICP
job and is NOT part of this harness. Until dense_ssm_f<FOLD>.npz exists the blend stage
is SKIPPED and the report says so — it is never silently substituted.
"""
import os, re, sys, json, time, inspect, tempfile, importlib
import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
# surfproj.py / dense_fit.py live in deep_model/ locally and are SHIPPED NEXT TO THIS
# FILE on the GPU box, so both locations are on the path.
for _p in (_HERE, _ROOT, os.path.join(_ROOT, "deep_model")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

NL, SCALE = 85, 30.0
CONTOURS = [(0, 24), (25, 54), (55, 74), (75, 84)]
NFOLD = 5

# A family needs no trainer edit: add one line here, or set FAMILY_MODULE/FAMILY_CLASS.
# "__self__" means "a class defined in this file" (used by the smoke test only).
REGISTRY = {
    "fake":        ("__self__", "FakeFamily"),      # smoke test only
    "kpconv":      ("fam_kpconv", None),            # None -> the module's `MODEL`
    "pointnext":   ("fam_pointnext", None),
    "ptv3":        ("fam_ptv3", None),
    "diffusionnet": ("fam_diffusionnet", None),
    "template":    ("fam_template", None),
    "bilateral":   ("fam_bilateral", None),         # MODE=single|bilateral|bilateral_head
}

TRAIN_DEFAULTS = dict(lr=1.5e-3, bs=16, wd=5e-4, sub_frac=0.625,
                      aug_rot=1.2, aug_scale=0.20, aug_jit=0.25, aug_qjit=0.9)


# ------------------------------------------------------------------ env / config
def _autotype(v):
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    if re.fullmatch(r"[-+]?\d+", v):
        return int(v)
    if re.fullmatch(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?", v):
        return float(v)
    if v[:1] in "[{\"":
        return json.loads(v)
    return v


def cfg_from_env():
    """CFG_JSON first, then individual CFG_<NAME> vars (which win)."""
    out = json.loads(os.environ.get("CFG_JSON", "{}"))
    for k, v in os.environ.items():
        if k.startswith("CFG_") and k != "CFG_JSON":
            out[k[4:].lower()] = _autotype(v)
    return out


def resolve_family(name):
    mod, cls = REGISTRY.get(name, (os.environ.get("FAMILY_MODULE", name), None))
    mod = os.environ.get("FAMILY_MODULE", mod)
    cls = os.environ.get("FAMILY_CLASS", cls)
    if mod == "__self__":
        return globals()[cls]
    m = importlib.import_module(mod)
    if cls:
        return getattr(m, cls)
    assert hasattr(m, "MODEL"), (
        f"{mod} exposes no MODEL symbol, so REGISTRY['{name}'] cannot be constructed. The "
        f"FAMILY CONTRACT is `MODEL = <class>` at module level with `cls(cfg, meta)`; point "
        f"FAMILY_CLASS=<class name> at it instead if it is named something else. Classes "
        f"found: {[k for k, v in vars(m).items() if isinstance(v, type) and issubclass(v, nn.Module)]}")
    return m.MODEL


# ------------------------------------------------------------------ frozen folds
def frozen_folds(ne):
    """Constraint 3, verbatim. subject = ear_index//2; array_split(RS(12345).perm, 5)."""
    subj = np.arange(ne) // 2
    parts = np.array_split(np.random.RandomState(12345).permutation(np.unique(subj)), NFOLD)
    return subj, [np.asarray(p) for p in parts]


FOLDS_JSON = os.path.join(os.path.dirname(_HERE), "results", "folds.json")


def verify_folds(subj, parts, path=FOLDS_JSON):
    """Cross-check the derived split against the committed file when it is present."""
    if not os.path.exists(path):
        return "folds.json absent (remote box) -- split derived from the frozen rule only"
    a = json.load(open(path))["assignments"]
    of = np.full(len(subj), -1)
    for f, p in enumerate(parts):
        of[np.isin(subj, p)] = f
    if len(a) != len(subj):
        return f"folds.json has {len(a)} ears, data has {len(subj)} -- NOT cross-checked"
    for r in a:
        assert of[r["ear_index"]] == r["fold"], \
            f"ear {r['ear_index']}: derived fold {of[r['ear_index']]} != folds.json {r['fold']}"
    return "verified against research/results/folds.json"


# ------------------------------------------------------------------ data
# cls.NEEDS pulls arbitrary keys out of DATA and hands them to forward(). A key holding
# ground truth would be a leak that passes every shape check, so the names are blocked
# outright: the target reaches loss() as a separate argument and nothing else.
GT_KEYS = frozenset({"true", "true_lms", "gt", "gt_lms", "landmarks", "lms",
                     "target", "targets", "lm_bary", "lm_vert", "lm_face"})


def load_data(path, needs, dev):
    bad = sorted(GT_KEYS & set(needs))
    assert not bad, (f"cls.NEEDS requests {bad}, which hold GROUND TRUTH. The target is a "
                     f"separate argument that reaches loss() only (constraint 2); it must "
                     f"never enter a batch dict.")
    d = np.load(path, allow_pickle=True)
    clouds = torch.tensor(d["clouds"]).float()
    assert clouds.dim() == 4, "clouds must be (E,M,N,3) fresh samples"
    data = dict(clouds=clouds.to(dev),
                coarse=torch.tensor(d["coarse"]).float().to(dev),
                true=torch.tensor(d["true"]).float().to(dev),
                R=d["R"].astype(np.float64), c0=d["c0"].astype(np.float64),
                E=clouds.shape[0], M=clouds.shape[1], N=clouds.shape[2], extra={})
    for k in needs:
        assert k in d.files, f"family needs '{k}' but {path} has {d.files}"
        data["extra"][k] = torch.tensor(d[k]).float().to(dev)
    return data


def make_batch(data, ears, samples, nsamp, extra_fn=None, meta=None):
    """ears (B,) int, samples (B,nsamp) int -> batch dict. `true` is NEVER included."""
    e = torch.as_tensor(ears, dtype=torch.long, device=data["clouds"].device)
    s = torch.as_tensor(samples, dtype=torch.long, device=e.device)
    pc = data["clouds"][e[:, None], s]                              # (B,nsamp,N,3)
    b = {"pc": pc, "coarse": data["coarse"][e], "ear": e}
    for k, v in data["extra"].items():
        b[k] = v[e[:, None], s] if v.dim() >= 4 else v[e]
    if extra_fn is not None:
        got = extra_fn(np.asarray(ears), np.asarray(samples), meta)
        assert "true" not in got, "cls.BATCH must never return ground truth (constraint 2)"
        b.update(got)
    return b


def _flatten_samples(b, nsamp):
    if nsamp == 1:
        b = dict(b)
        for k, v in b.items():
            if torch.is_tensor(v) and v.dim() >= 4:
                b[k] = v[:, 0]
    return b


# ------------------------------------------------------------------ augmentation
def rand_rot(B, maxang, gen, dev):
    ax = torch.randn(B, 3, device=dev, generator=gen)
    ax = ax / ax.norm(dim=1, keepdim=True)
    ang = (torch.rand(B, device=dev, generator=gen) - .5) * maxang
    c, s = ang.cos(), ang.sin(); x, y, z = ax[:, 0], ax[:, 1], ax[:, 2]; C = 1 - c
    return torch.stack([
        torch.stack([c + x*x*C, x*y*C - z*s, x*z*C + y*s], -1),
        torch.stack([y*x*C + z*s, c + y*y*C, y*z*C - x*s], -1),
        torch.stack([z*x*C - y*s, z*y*C + x*s, c + z*z*C], -1)], 1)


def default_augment(b, tg, cfg, rotates, gen):
    """Same augmentation as gpu_screen.py, generalised over the sample and extra dims.

    Point-valued tensors get rotation + isotropic scale + jitter; DIRECTION-valued
    extras (cls.ROTATES) get the rotation only.
    """
    pc = b["pc"]; B, S, N, _ = pc.shape; dev = pc.device
    R = rand_rot(B, cfg["aug_rot"], gen, dev)
    sc = 1 + (torch.rand(B, 1, 1, device=dev, generator=gen) - .5) * cfg["aug_scale"]
    nsub = max(8, min(N, int(round(N * cfg["sub_frac"]))))
    sub = torch.rand(B, S, N, device=dev, generator=gen).argsort(-1)[..., :nsub]
    out = dict(b)

    def rot(t):                                     # row vectors: t @ R^T, per batch item
        return torch.einsum("bnj,bij->bni", t, R) if t.dim() == 3 else \
            torch.einsum("bsnj,bij->bsni", t, R)

    p = torch.gather(pc, 2, sub[..., None].expand(-1, -1, -1, 3))
    p = rot(p) * sc[:, None] + torch.randn(p.shape, device=dev, generator=gen) * cfg["aug_jit"]
    out["pc"] = p
    done = set()
    for k in b:
        if k in ("pc", "coarse", "ear"):
            continue
        v = b[k]
        if torch.is_tensor(v) and v.dim() == 4 and v.shape[:3] == pc.shape[:3]:
            v = torch.gather(v, 2, sub[..., None].expand(-1, -1, -1, v.shape[-1]))
            out[k] = rot(v) if k in rotates else v
            done.add(k)
    # a ROTATES key whose shape is not (B,S,N,C) would silently go UNROTATED while the
    # cloud and the target are rotated -- every shape check passes and the family trains
    # against an inconsistent frame. Refuse instead.
    miss = [k for k in rotates if k in b and k not in done]
    assert not miss, (f"cls.ROTATES names {miss}, but default_augment cannot rotate them: "
                      f"shapes {[tuple(b[k].shape) for k in miss]} are not (B,S,N,C) like "
                      f"pc {tuple(pc.shape)}. Provide cls.AUGMENT for this family.")
    out["coarse"] = rot(b["coarse"]) * sc + \
        torch.randn(b["coarse"].shape, device=dev, generator=gen) * cfg["aug_qjit"]
    return out, rot(tg) * sc


# ------------------------------------------------------------------ loss
def default_loss(out, tg, model=None, batch=None):
    """Deep supervision with geometric weights (last aux strongest), then the final.

    A family may define its own loss as either `loss(out, tg)` or `loss(out, tg, batch)`;
    a structural family (vertex heatmaps, coordinate fields) generally needs the batch to
    reach the mesh operators it was built from, so both arities are supported and the
    required-positional count decides which is called.
    """
    if model is not None and hasattr(model, "loss"):
        n = sum(1 for p in inspect.signature(model.loss).parameters.values()
                if p.default is p.empty and p.kind in
                (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD))
        return model.loss(out, tg, batch) if n >= 3 else model.loss(out, tg)
    aux = out.get("aux") or []
    L = 0.0
    if aux:
        w = np.array([0.5 ** (len(aux) - 1 - t) for t in range(len(aux))]); w /= w.sum()
        for t, a in enumerate(aux):
            L = L + float(w[t]) * ((a - tg) ** 2).sum(-1).mean()
    return L + ((out["pred"] - tg) ** 2).sum(-1).mean() + out.get("reg", 0.0)


# ------------------------------------------------------------------ full pipeline
def load_tris(path, ne):
    """Accept layout A (verts/faces + v_ptr/f_ptr, GLOBAL indices) or B (V/F + *_off)."""
    z = np.load(path, allow_pickle=True)
    if "verts" in z.files:
        vp, fp, V, F, glob = z["v_ptr"], z["f_ptr"], z["verts"], z["faces"], True
    else:
        vp, fp, V, F, glob = z["v_off"], z["f_off"], z["V"], z["F"], False
    vp, fp, F = vp.astype(np.int64), fp.astype(np.int64), F.astype(np.int64)
    assert len(vp) == ne + 1 and len(fp) == ne + 1, \
        f"{path}: offsets describe {len(vp)-1} ears, the data npz has {ne}"
    assert vp[-1] == len(V) and fp[-1] == len(F), f"{path}: offsets do not close"
    return dict(v_ptr=vp, f_ptr=fp, V=V, F=F, glob=glob, path=path,
                layout="A_global" if glob else "B_local")


def ear_mesh(tris, i):
    Ve = tris["V"][tris["v_ptr"][i]:tris["v_ptr"][i + 1]].astype(np.float64)
    Fe = tris["F"][tris["f_ptr"][i]:tris["f_ptr"][i + 1]]
    return Ve, (Fe - tris["v_ptr"][i]) if tris["glob"] else Fe


def check_frame(tris, coarse_canon, ear_idx, tol):
    """Prove the mesh is in the SAME frame and ear order as the predictions.

    A string field claiming "canonical" proves nothing. This measures it: the coarse
    landmarks must sit on the ear. A frame, mirroring or ear-index mismatch multiplies
    this distance, and a decimated mesh only nudges it.
    """
    from scipy.spatial import cKDTree
    per = []
    for i in ear_idx:
        Ve, Fe = ear_mesh(tris, i)
        assert Fe.min() >= 0 and Fe.max() < len(Ve), \
            f"{tris['path']}: ear {i} face indices out of range for its vertex block " \
            f"(layout sniffed as {tris['layout']})"
        per.append(float(np.median(cKDTree(Ve).query(coarse_canon[i])[0])))
    med = float(np.median(per))
    assert med < tol, \
        (f"{tris['path']}: median coarse-landmark-to-mesh distance is {med:.2f} mm "
         f"(> FRAME_TOL {tol}) on {len(ear_idx)} ears -- the mesh is not in the same "
         f"canonical frame / ear order as the data npz, or right ears are not mirrored "
         f"the same way. Projecting would silently corrupt every prediction.")
    return round(med, 4), round(float(np.max(per)), 4)


def load_ssm(path, fold, va_idx, kuse):
    from dense_fit import DenseSSMFit
    z = np.load(path)
    assert "fold" in z.files and "train_ear_mask" in z.files, \
        (f"{path} carries no fold/train_ear_mask, so it cannot prove it was built from "
         f"this fold's TRAINING ears only -- refusing to load it (constraint 2)")
    assert int(z["fold"]) == fold, f"{path} was built for fold {int(z['fold'])}, not {fold}"
    m = z["train_ear_mask"].astype(bool)
    assert not m[va_idx].any(), \
        f"{path}: train_ear_mask contains {int(m[va_idx].sum())} validation ears -- LEAK"
    return DenseSSMFit(path, kuse=kuse)


def project(pts, V, F, margin, k):
    """Exact point-to-triangle snap, identical to deep_stage.project_to_surface."""
    from surfproj import SurfaceProjector
    lo, hi = pts.min(0) - margin, pts.max(0) + margin
    vin = np.all((V >= lo) & (V <= hi), axis=1)
    Fs = F[vin[F].any(axis=1)]
    if len(Fs) == 0:
        return pts
    keep = np.unique(Fs)
    remap = -np.ones(len(V), np.int64); remap[keep] = np.arange(len(keep))
    return SurfaceProjector(V[keep].astype(np.float64), remap[Fs], k=k).project(pts)[0]


def surface_points(around, V, margin, npts, seed=0):
    """deep_stage._surface_points: crop the vertices around the prediction, subsample."""
    lo, hi = around.min(0) - margin, around.max(0) + margin
    m = np.all((V >= lo) & (V <= hi), axis=1)
    pts = V[m] if m.any() else V
    if len(pts) > npts:
        pts = pts[np.random.RandomState(seed).choice(len(pts), npts, replace=False)]
    return pts.astype(np.float64)


def full_pipeline(Praw, ear_idx, tris, ssm, opt):
    """TTA-mean -> surface projection -> dense-SSM blend -> re-projection, per ear.

    Exactly deep_stage.deep_refine's order. Runs in the CANONICAL frame (see the module
    docstring); the caller maps to world afterwards.
    """
    out = Praw.copy()
    for k, i in enumerate(ear_idx):
        Ve, Fe = ear_mesh(tris, i)
        p = project(Praw[k], Ve, Fe, opt["proj_margin"], opt["proj_k"])
        if ssm is not None:
            cl = surface_points(p, Ve, opt["ssm_crop"], opt["ssm_npts"])
            p = ssm.refine(cl, p, alpha=opt["ssm_alpha"])
            p = project(p, Ve, Fe, opt["proj_margin"], opt["proj_k"])
        out[k] = p
    return out


# ------------------------------------------------------------------ evaluation
def evaluate(model, data, ear_idx, nsamp_family, tta, extra_fn=None, meta=None):
    """Returns canonical TTA-mean predictions, the per-sample stack, and its spread."""
    model.eval()
    per_ear = []
    with torch.no_grad():
        for i in ear_idx:
            per = []
            for s in range(tta):
                # each TTA repeat is one fresh surface sample; a family with SAMPLES>1
                # gets DISTINCT samples in its slots, as it does during training. Handing
                # it the same cloud K times (which is what gpu_screen.py's fusion2 eval
                # did) is a train/test mismatch for exactly the fusion the family exists
                # to do. Identical to `s % M` when SAMPLES == 1, i.e. for every family
                # screened so far.
                js = (s + np.arange(nsamp_family))[None] % data["M"]
                b = _flatten_samples(make_batch(data, [i], js, nsamp_family, extra_fn, meta),
                                     nsamp_family)
                per.append(model(b)["pred"][0].cpu().numpy().astype(np.float64))
            per_ear.append(np.stack(per))
    PS = np.stack(per_ear)                                          # (n,tta,85,3)
    var = float(np.linalg.norm(PS - PS.mean(1, keepdims=True), axis=3).mean())
    return PS.mean(1), PS, var


def to_world(P, ear_idx, R, c0):
    return np.stack([P[k] @ R[i] + c0[i] for k, i in enumerate(ear_idx)])


def stats(P, G):
    E = np.linalg.norm(P - G, axis=2)
    return dict(ordered=float(E.mean()), median=float(np.median(E)),
                p90=float(np.percentile(E, 90)),
                per_contour={f"{lo}-{hi}": round(float(E[:, lo:hi + 1].mean()), 4)
                             for lo, hi in CONTOURS},
                per_ear=[round(float(x), 5) for x in E.mean(1)])


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    FAMILY = os.environ["FAMILY"]
    FOLD = int(os.environ.get("FOLD", "0"))
    SEED = int(os.environ.get("SEED", "0"))
    EPOCHS = int(os.environ.get("EPOCHS", "1200"))
    WORK = os.environ.get("WORK", "scratch")
    VARIANT = os.environ.get("VARIANT", FAMILY)
    TAG = os.environ.get("TAG", f"fam_{FAMILY}_s{SEED}_f{FOLD}")
    DATA = os.environ.get("DATA", f"{WORK}/screen_data_2048.npz")
    TTA = int(os.environ.get("TTA", "4"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # seeding covers torch (incl. cuda), numpy, and the batch/augmentation streams
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); np.random.seed(SEED)
    SAMPLER = np.random.RandomState(1_000_003 * SEED + 17)
    GEN = torch.Generator(device=dev); GEN.manual_seed(SEED * 7919 + 13)

    cls = resolve_family(FAMILY)
    cfg = {**TRAIN_DEFAULTS, **getattr(cls, "DEFAULTS", {}), **cfg_from_env()}
    NEEDS = tuple(getattr(cls, "NEEDS", ()))
    ROTATES = tuple(getattr(cls, "ROTATES", ("nrm",)))
    NSAMP = int(getattr(cls, "SAMPLES", 1))
    BATCH_FN = getattr(cls, "BATCH", None)
    AUG = getattr(cls, "AUGMENT", default_augment)

    data = load_data(DATA, NEEDS, dev)
    NE, M = data["E"], data["M"]
    subj, parts = frozen_folds(NE)
    fold_note = verify_folds(subj, parts)

    if os.environ.get("VAL_EARS"):
        va_idx = np.asarray(json.loads(os.environ["VAL_EARS"]), int)
        tr_idx = np.asarray(json.loads(os.environ["TRAIN_EARS"]), int)
        vs, ts = set(subj[va_idx].tolist()), set(subj[tr_idx].tolist())
        assert not (vs & ts), f"explicit split shares {len(vs & ts)} SUBJECT(S) -- grouping broken"
        assert not (set(va_idx.tolist()) & set(tr_idx.tolist())), "explicit split shares ears"
        split_mode, ALIAS = "explicit_inner", False
        print(f"EXPLICIT-SPLIT (inner CV of outer fold {FOLD}): {len(ts)} train / {len(vs)} "
              f"val SUBJECTS, {len(tr_idx)} / {len(va_idx)} ears. NOT an outer-fold report.",
              flush=True)
    else:
        val_s = set(parts[FOLD].tolist())
        va_idx = np.array([i for i in range(NE) if subj[i] in val_s])
        tr_idx = np.array([i for i in range(NE) if subj[i] not in val_s])
        vs, ts = val_s, set(np.unique(subj).tolist()) - val_s
        split_mode = "frozen_fold"
        ALIAS = os.environ.get("ALIAS", "1") == "1"
        print(f"FOLD {FOLD}/{NFOLD}: {len(ts)} train / {len(vs)} val SUBJECTS "
              f"({len(tr_idx)} / {len(va_idx)} ears)  [{fold_note}]", flush=True)

    artefacts = {}
    if os.environ.get("ARTEFACTS"):
        z = np.load(os.environ["ARTEFACTS"], allow_pickle=True)
        assert "fold" in z.files and "train_ear_mask" in z.files, \
            f"{os.environ['ARTEFACTS']} cannot prove it is fold-safe (constraint 2)"
        assert int(z["fold"]) == FOLD, "artefact npz was built for a different fold"
        assert not z["train_ear_mask"].astype(bool)[va_idx].any(), \
            "artefact npz was built using validation ears -- LEAK"
        artefacts = {k: z[k] for k in z.files}

    meta = dict(nl=NL, contours=CONTOURS, scale=SCALE, npts=data["N"], fold=FOLD,
                dev=dev, n_train_ears=len(tr_idx), artefacts=artefacts)
    model = cls(cfg, meta).to(dev)
    NPARAM = sum(p.numel() for p in model.parameters())
    print(f"[{VARIANT} seed{SEED} fold{FOLD}] {FAMILY}: {NPARAM:,} params | "
          f"{data['N']} pts x {M} samples | cfg {json.dumps(cfg, sort_keys=True)}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    EVERY = int(os.environ.get("EVAL_EVERY", str(max(1, EPOCHS // 12))))
    BS = int(cfg["bs"])
    TRUE = data["true"].cpu().numpy().astype(np.float64)
    tr_probe = tr_idx[:40]

    def world_MLE(P, idx):
        return float(np.linalg.norm(to_world(P, idx, data["R"], data["c0"])
                                    - to_world(TRUE[idx], idx, data["R"], data["c0"]),
                                    axis=2).mean())

    curve, best = [], (9e9, None)
    for ep in range(EPOCHS):
        model.train()
        perm = SAMPLER.permutation(tr_idx)
        for b in range(0, len(perm), BS):
            bi = perm[b:b + BS]
            js = SAMPLER.randint(0, M, (len(bi), NSAMP))
            batch = make_batch(data, bi, js, NSAMP, BATCH_FN, meta)
            tg = data["true"][torch.as_tensor(bi, dtype=torch.long, device=dev)]
            if AUG is not None:
                batch, tg = AUG(batch, tg, cfg, ROTATES, GEN)
            batch = _flatten_samples(batch, NSAMP)
            opt.zero_grad()
            loss = default_loss(model(batch), tg, model, batch)
            loss.backward(); opt.step()
        sch.step()
        if (ep + 1) % EVERY == 0 or ep + 1 == EPOCHS:
            Pv, _, var = evaluate(model, data, va_idx, NSAMP, 2, BATCH_FN, meta)
            Pt, _, _ = evaluate(model, data, tr_probe, NSAMP, 1, BATCH_FN, meta)
            vm, tm = world_MLE(Pv, va_idx), world_MLE(Pt, tr_probe)
            curve.append({"epoch": ep + 1, "train_MLE": round(tm, 4), "val_MLE": round(vm, 4)})
            print(f"  ep{ep+1:4d} train {tm:.4f} val {vm:.4f} sampvar {var:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            if vm < best[0]:
                best = (vm, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

    if best[1] is not None:
        model.load_state_dict(best[1])
    Pv, _, var = evaluate(model, data, va_idx, NSAMP, TTA, BATCH_FN, meta)
    Gw = to_world(TRUE[va_idx], va_idx, data["R"], data["c0"])
    Pw = to_world(Pv, va_idx, data["R"], data["c0"])
    raw = stats(Pw, Gw)

    # ---- full pipeline: TTA mean -> projection -> dense-SSM blend -> re-projection
    TRIS = os.environ.get("TRIS", f"{WORK}/mesh_data.npz")
    SSMP = os.environ.get("SSM", f"{WORK}/dense_ssm_f{FOLD}.npz")
    fp = {"stages": ["tta_mean"], "tta": TTA, "tris": None, "ssm": None,
          "frame": "canonical", "status": None}
    full, Pfw = None, None
    if os.environ.get("FULL_EVAL", "1") != "1":
        fp["status"] = "disabled by FULL_EVAL=0"
    elif not os.path.exists(TRIS):
        fp["status"] = (f"{TRIS} absent -> no surface projection and no SSM blend; "
                        f"ordered_MLE_full_mm is null. Build it locally with "
                        f"research/code/build_mesh_data.py (see NPZ CONTRACT 1).")
    else:
        tris = load_tris(TRIS, NE)
        cc = data["coarse"].cpu().numpy().astype(np.float64)
        fp["layout"] = tris["layout"]
        fp["coarse_to_mesh_mm"], fp["coarse_to_mesh_max_mm"] = check_frame(
            tris, cc, va_idx, float(os.environ.get("FRAME_TOL", "5.0")))
        ssm = None
        if os.path.exists(SSMP):
            ssm = load_ssm(SSMP, FOLD, va_idx, int(os.environ.get("SSM_KUSE", "120")))
            fp["ssm"] = SSMP
        else:
            fp["status"] = (f"{SSMP} absent -> surface projection only, dense-SSM blend "
                            f"SKIPPED. scratch/dense_ssm.npz is deliberately NOT "
                            f"substituted: it was fitted on the fixed-split training "
                            f"ears and would leak on 4 of the 5 folds.")
        opt_fp = dict(proj_margin=float(os.environ.get("PROJ_MARGIN", "8.0")),
                      proj_k=int(os.environ.get("PROJ_K", "12")),
                      ssm_alpha=float(os.environ.get("SSM_ALPHA", "0.3")),
                      ssm_crop=float(os.environ.get("SSM_CROP", "12.0")),
                      ssm_npts=int(os.environ.get("SSM_NPTS", "16384")))
        fp.update(opt_fp); fp["tris"] = TRIS
        fp["stages"] += ["surface_projection"] + (
            ["dense_ssm_blend", "reprojection"] if ssm is not None else [])
        Pf = full_pipeline(Pv, va_idx, tris, ssm, opt_fp)
        Pfw = to_world(Pf, va_idx, data["R"], data["c0"])
        full = stats(Pfw, Gw)
        fp["status"] = fp["status"] or "complete"

    res = {"variant": VARIANT, "seed": SEED, "fold": FOLD, "params": int(NPARAM),
           "runtime_s": round(time.time() - t0, 1), "epochs": EPOCHS,
           "config": {**cfg, "_family": FAMILY, "_data": DATA, "_samples": NSAMP,
                      "_needs": list(NEEDS), "_eval_every": EVERY,
                      "_batch_hook": BATCH_FN is not None,
                      "_augment": None if AUG is None else AUG.__name__},
           "ordered_MLE_mm": round(raw["ordered"], 4),
           "median_mm": round(raw["median"], 4),
           "P90_mm": round(raw["p90"], 4),
           "per_contour_MLE_mm": raw["per_contour"],
           "fresh_sample_pred_variance_mm": round(var, 4),
           "train_val_curve": curve,
           "per_ear_MLE": raw["per_ear"],
           "val_ear_index": [int(i) for i in va_idx],
           "ordered_MLE_full_mm": None if full is None else round(full["ordered"], 4),
           "median_full_mm": None if full is None else round(full["median"], 4),
           "P90_full_mm": None if full is None else round(full["p90"], 4),
           "per_contour_MLE_full_mm": None if full is None else full["per_contour"],
           "per_ear_MLE_full": None if full is None else full["per_ear"],
           "full_pipeline": fp,
           "split_mode": split_mode, "fold_check": fold_note,
           "n_train_subjects": len(ts), "n_val_subjects": len(vs),
           "best_val_MLE_mm": round(best[0], 4) if best[1] is not None else None}
    os.makedirs(WORK, exist_ok=True)
    json.dump(res, open(f"{WORK}/{TAG}.json", "w"), indent=1)
    np.save(f"{WORK}/{TAG}.npy", Pw)                    # RAW, matches ordered_MLE_mm
    if Pfw is not None:
        np.save(f"{WORK}/{TAG}_full.npy", Pfw)
    if ALIAS:                                           # what screen_compare/cv_verdict glob
        json.dump(res, open(f"{WORK}/screen_{VARIANT}_s{SEED}_f{FOLD}.json", "w"), indent=1)
        np.save(f"{WORK}/screen_{VARIANT}_s{SEED}_f{FOLD}.npy", Pw)
    fm = "n/a" if full is None else f"{full['ordered']:.4f}"
    print(f"\n[{VARIANT} s{SEED} f{FOLD}] raw MLE {raw['ordered']:.4f} | full-pipeline {fm} | "
          + "per-contour " + "/".join(f"{v:.3f}" for v in raw["per_contour"].values())
          + f" | sampvar {var:.4f} | {NPARAM/1e3:.0f}k params | {time.time()-t0:.0f}s"
          + f"\n  pipeline: {'+'.join(fp['stages'])} -- {fp['status']}"
          + f"\n  wrote {WORK}/{TAG}.json/.npy" + (f" (+ screen_{VARIANT} alias)" if ALIAS else ""),
          flush=True)
    return res


# ------------------------------------------------------------------ smoke test
class FakeFamily(nn.Module):
    """Smoke-test family: global cloud context + landmark embedding -> offset from coarse.

    Exists only to exercise the harness end-to-end on CPU. It is intentionally too weak
    to be a baseline for anything.
    """
    DEFAULTS = dict(width=32)
    SEARCH_SPACE = dict(width=[16, 32, 64], lr=[1e-3, 3e-3], sub_frac=[0.5, 0.625])
    NEEDS, ROTATES, SAMPLES = (), ("nrm",), 1

    @staticmethod
    def BATCH(ears, samples, meta):
        """Exercises the ragged-artefact hook: a per-ear tensor the DATA npz never held."""
        return {"probe": torch.full((len(ears), 1), float(meta["fold"]))}

    def __init__(self, cfg, meta):
        super().__init__()
        w = int(cfg["width"]); self.nl = meta["nl"]
        self.enc = nn.Sequential(nn.Linear(3, w), nn.ReLU(), nn.Linear(w, w), nn.ReLU())
        self.emb = nn.Embedding(self.nl, w)
        self.mid = nn.Linear(2 * w + 3, 3)
        self.head = nn.Sequential(nn.Linear(2 * w + 3, w), nn.ReLU(), nn.Linear(w, 3))

    def forward(self, batch):
        pc, q = batch["pc"], batch["coarse"]
        assert "probe" in batch, "cls.BATCH hook did not reach forward()"
        assert "true" not in batch, "ground truth reached a forward pass"
        g = self.enc(pc / SCALE).max(1).values[:, None].expand(-1, self.nl, -1)
        e = self.emb.weight[None].expand(pc.shape[0], -1, -1)
        x = torch.cat([g, e, q / SCALE], -1)
        a = q + self.mid(x)
        return {"pred": a + self.head(x), "aux": [a]}


def fake_bundle(dirpath, ne=20, m=2, npts=256, grid=24, seed=0):
    """Synthetic ear-shaped surfaces + the two full-pipeline artefacts, for smoke tests.

    A per-ear triangulated height-field stands in for the mesh. The 85 'true' landmarks
    are fixed BARYCENTRIC points on it, so ground truth lies exactly on the surface and
    the surface-projection stage is genuinely exercised, not just shape-checked.
    """
    os.makedirs(dirpath, exist_ok=True)
    rng = np.random.RandomState(seed)
    u = np.linspace(-15, 15, grid)
    UU, VV = np.meshgrid(u, u, indexing="ij")
    idx = np.arange(grid * grid).reshape(grid, grid)
    a, b, c, d = idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]
    F = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3),
                        np.stack([a, c, d], -1).reshape(-1, 3)]).astype(np.int32)
    bf = rng.choice(len(F), 85, replace=False).astype(np.int32)
    bw = rng.dirichlet(np.ones(3) * 3.0, 85).astype(np.float32)

    def verts(k):
        z = 1.2 * np.sin(UU / 6.0 + 0.1 * k) * np.cos(VV / 5.0 - 0.07 * k) + 0.15 * k / ne
        return np.stack([UU.ravel(), VV.ravel(), z.ravel()], 1).astype(np.float32)

    Vs = [verts(k) for k in range(ne)]
    tr = lambda V: (bw[..., None] * V[F[bf]]).sum(1)
    clouds = np.zeros((ne, m, npts, 3), np.float32)
    true = np.stack([tr(V) for V in Vs]).astype(np.float32)
    for e in range(ne):
        for j in range(m):
            s = np.random.RandomState(100 * e + j).randint(0, len(Vs[e]), npts)
            clouds[e, j] = Vs[e][s] + np.random.RandomState(7 * e + j).randn(npts, 3) * 0.05
    coarse = (true + rng.randn(ne, 85, 3).astype(np.float32) * 0.8).astype(np.float32)
    dp = f"{dirpath}/fake_data.npz"
    np.savez(dp, clouds=clouds, coarse=coarse, true=true,
             R=np.tile(np.eye(3, dtype=np.float32), (ne, 1, 1)),
             c0=np.zeros((ne, 3), np.float32), split=np.array(["train"] * ne))

    vp = np.arange(ne + 1, dtype=np.int64) * len(Vs[0])
    fp_ = np.arange(ne + 1, dtype=np.int64) * len(F)
    tp = f"{dirpath}/fake_tris.npz"                        # layout B: LOCAL face indices
    np.savez(tp, V=np.concatenate(Vs), F=np.tile(F, (ne, 1)), v_off=vp, f_off=fp_,
             frame=np.array("canonical"), margin=np.float32(14.0))
    tp2 = f"{dirpath}/fake_meshdata.npz"                   # layout A: GLOBAL, build_mesh_data
    np.savez(tp2, verts=np.concatenate(Vs), v_ptr=vp, f_ptr=fp_,
             faces=np.concatenate([F + vp[e] for e in range(ne)]).astype(np.int32))

    n3 = 3 * len(Vs[0])
    Q = np.linalg.qr(rng.randn(n3, 4))[0].T.astype(np.float32)
    subj, parts = frozen_folds(ne)
    mask = ~np.isin(subj, parts[0])
    sp = f"{dirpath}/fake_ssm_f0.npz"
    np.savez(sp, mean=np.stack(Vs).mean(0).reshape(-1), comps=Q * 3.0,
             eig=np.array([9., 4., 1., .25], np.float32), template_F=F,
             bary_f=bf, bary_w=bw, fold=np.int64(0), train_ear_mask=mask)
    return dp, tp, sp, tp2


def smoke():
    print("=" * 78)
    print("SMOKE 1/2 -- module-level forward/backward, B=2")
    dev = "cpu"
    torch.manual_seed(0)            # the printed numbers are a regression baseline
    meta = dict(nl=NL, contours=CONTOURS, scale=SCALE, npts=256, fold=0, dev=dev,
                n_train_ears=16, artefacts={})
    net = FakeFamily({**TRAIN_DEFAULTS, **FakeFamily.DEFAULTS}, meta)
    npar = sum(p.numel() for p in net.parameters())
    batch = {"pc": torch.randn(2, 256, 3) * 8, "coarse": torch.randn(2, NL, 3) * 8,
             "ear": torch.tensor([0, 1]), **FakeFamily.BATCH([0, 1], None, meta)}
    out = net(batch)
    loss = default_loss(out, torch.randn(2, NL, 3), net)
    loss.backward()
    gnorm = sum(float(p.grad.norm()) for p in net.parameters() if p.grad is not None)
    print(f"  pred {tuple(out['pred'].shape)}  aux {len(out['aux'])}x"
          f"{tuple(out['aux'][0].shape)}  params {npar:,}  loss {float(loss):.4f}  "
          f"grad-norm-sum {gnorm:.3f}")
    assert tuple(out["pred"].shape) == (2, NL, 3), out["pred"].shape
    assert gnorm > 0, "no gradient reached the parameters"

    # the augmentation must be a per-ear SIMILARITY: identical scale for cloud, coarse
    # and target, and pairwise distances preserved up to that scale. A per-batch mix-up
    # of the rotation matrices passes every shape check and silently destroys training.
    acfg = {**TRAIN_DEFAULTS, "aug_jit": 0.0, "aug_qjit": 0.0, "sub_frac": 1.0}
    g = torch.Generator(device=dev); g.manual_seed(3)
    b0 = {"pc": torch.randn(2, 1, 64, 3, generator=g) * 8,
          "coarse": torch.randn(2, NL, 3, generator=g) * 8, "ear": torch.tensor([0, 1]),
          "probe": torch.zeros(2, 1)}
    tg0 = b0["coarse"] + 0.3
    b1, tg1 = default_augment(b0, tg0, acfg, (), g)

    def dists(t):                     # float64, no cdist matmul shortcut
        t = t.double()
        return (t[:, :, None] - t[:, None]).norm(dim=-1)

    scales = {}
    for nm, a, c in (("coarse", b0["coarse"], b1["coarse"]), ("tg", tg0, tg1)):
        r0, r1 = dists(a), dists(c)                       # landmarks keep their order
        s = (r1.sum((1, 2)) / r0.sum((1, 2)))[:, None, None]
        err = float((r1 - s * r0).abs().max())
        assert err < 1e-4, \
            f"{nm} augmentation is not a per-ear similarity (max dist err {err:.3e}) -- " \
            f"the rotation matrices are being mixed across the batch"
        scales[nm] = s.flatten()
    # the cloud is also SUBSAMPLED (a permutation here), so compare its distance
    # SPECTRUM, which is permutation-invariant
    d0 = dists(b0["pc"][:, 0]).flatten(1).sort(-1).values
    d1 = dists(b1["pc"][:, 0]).flatten(1).sort(-1).values
    scales["pc"] = d1.sum(-1) / d0.sum(-1)
    assert float((d1 - scales["pc"][:, None] * d0).abs().max()) < 1e-4, \
        "cloud augmentation is not a per-ear similarity"
    assert torch.allclose(scales["pc"], scales["tg"], atol=1e-6) and \
        torch.allclose(scales["coarse"], scales["tg"], atol=1e-6), \
        f"cloud/coarse/target scales disagree: {scales}"
    print("  aug is a per-ear similarity, one shared scale per ear: "
          f"{[round(float(x), 6) for x in scales['tg']]}")

    print("\nSMOKE 2/2 -- trainer end-to-end incl. the FULL inference pipeline")
    tmp = os.environ.get("SMOKE_DIR",
                         os.path.join(tempfile.gettempdir(), "train_family_smoke"))
    dp, tp, sp, tp2 = fake_bundle(tmp)

    # constraint 2, as an executable test rather than a claim: the two ways a family
    # could pull ground truth into a forward pass must both be REFUSED, and a ROTATES
    # key the default augmenter cannot rotate must be refused rather than left unrotated.
    for needs, why in ((("true",), "NEEDS=('true',)"), (("lm_bary",), "NEEDS=('lm_bary',)")):
        try:
            load_data(dp, needs, dev); raise SystemExit(f"{why} was ACCEPTED -- LEAK")
        except AssertionError as e:
            assert "GROUND TRUTH" in str(e), str(e)
    try:
        default_augment({"pc": torch.zeros(2, 1, 8, 3), "coarse": torch.zeros(2, NL, 3),
                         "ear": torch.tensor([0, 1]), "nrm": torch.zeros(2, NL, 3)},
                        torch.zeros(2, NL, 3), acfg, ("nrm",), g)
        raise SystemExit("an unrotatable ROTATES key was silently left unrotated")
    except AssertionError as e:
        assert "cls.ROTATES" in str(e), str(e)
    print("  refused: NEEDS=('true',) / NEEDS=('lm_bary',) / an unrotatable ROTATES key")
    env = dict(FAMILY="fake", FOLD="0", SEED="0", EPOCHS="4", WORK=tmp, DATA=dp,
               TRIS=tp, SSM=sp, TTA="2", EVAL_EVERY="2", ALIAS="1",
               CFG_BS="8", CFG_WIDTH="16", TAG="fam_fake_s0_f0")
    keep = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    res = main()
    # the same run against the build_mesh_data.py layout (GLOBAL face indices): the
    # loader must produce bit-identical geometry, otherwise the sniffing is wrong
    os.environ.update(TRIS=tp2, TAG="fam_fake_s0_f0_A", ALIAS="0")
    resA = main()
    for k, v in keep.items():
        os.environ.pop(k) if v is None else os.environ.__setitem__(k, v)
    assert res["full_pipeline"]["layout"] == "B_local"
    assert resA["full_pipeline"]["layout"] == "A_global"
    assert res["ordered_MLE_full_mm"] == resA["ordered_MLE_full_mm"], \
        (f"mesh layouts disagree: B_local {res['ordered_MLE_full_mm']} vs A_global "
         f"{resA['ordered_MLE_full_mm']}")
    print(f"  both mesh layouts agree exactly ({res['ordered_MLE_full_mm']:.4f} mm); "
          f"coarse-to-mesh {res['full_pipeline']['coarse_to_mesh_mm']} mm median / "
          f"{res['full_pipeline']['coarse_to_mesh_max_mm']} max")
    need = ["variant", "seed", "fold", "params", "runtime_s", "epochs", "config",
            "ordered_MLE_mm", "median_mm", "P90_mm", "per_contour_MLE_mm",
            "fresh_sample_pred_variance_mm", "train_val_curve", "per_ear_MLE",
            "val_ear_index", "ordered_MLE_full_mm"]
    miss = [k for k in need if k not in res]
    assert not miss, f"report is missing {miss}"
    assert len(res["per_contour_MLE_mm"]) == 4
    assert res["ordered_MLE_full_mm"] is not None, "full pipeline did not run"
    assert res["config"]["_batch_hook"] and res["config"]["_augment"] == "default_augment"
    assert res["full_pipeline"]["stages"] == ["tta_mean", "surface_projection",
                                             "dense_ssm_blend", "reprojection"]
    P = np.load(f"{tmp}/fam_fake_s0_f0.npy")
    G = np.load(dp)["true"][np.array(res["val_ear_index"])].astype(np.float64)
    assert abs(np.linalg.norm(P - G, axis=2).mean() - res["ordered_MLE_mm"]) < 2e-3, \
        "saved .npy disagrees with ordered_MLE_mm (screen_compare.py would assert)"
    assert os.path.exists(f"{tmp}/screen_fake_s0_f0.json"), "alias not written"
    print(f"  report keys OK | raw {res['ordered_MLE_mm']:.4f} -> full "
          f"{res['ordered_MLE_full_mm']:.4f} mm | .npy consistent with ordered_MLE_mm")
    print("SMOKE PASS")
    print("=" * 78)


if __name__ == "__main__":
    main() if os.environ.get("FAMILY") else smoke()
