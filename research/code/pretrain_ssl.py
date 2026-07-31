"""
FAMILY D -- FOLD-CLEAN self-supervised pretraining of a pluggable point encoder, plus the
fine-tuning entry point that loads the result into a landmark model.

    FOLD=0 ENCODER=dgcnn EPOCHS=300 python3 pretrain_ssl.py        # pretrain fold 0
    FOLD=0 MASKED=1 CONSIST=0 NORMAL=0 python3 pretrain_ssl.py     # one task at a time
    FAMILY=ssl FAMILY_MODULE=pretrain_ssl FOLD=0 EPOCHS=1200 \
        CFG_SSL_CKPT=scratch/ssl_dgcnn_MCN_f0_s0.pt CFG_FREEZE_EPOCHS=100 \
        python3 train_family.py                                    # fine-tune
    python research/code/pretrain_ssl.py                           # <- no FOLD: smoke test

WHY THIS EXISTS, AND WHAT IT IS NOT EXPECTED TO FIX. The shipped 1.273 mm model's error is
77 % ORDERED CORRESPONDENCE along the contour and only 2 % normal-direction, and seven
variants of the same 2048-point DGCNN family moved none of it. Pretraining is therefore
NOT proposed as another local-XYZ refinement: of the three pretext tasks below only
CONSIST directly builds a per-point field that must *distinguish* one surface location
from another on the SAME ear (its negatives are in-ear), which is the closest
self-supervised proxy to "phase along the contour" available without labels. MASKED and
NORMAL build local shape representation, which the decomposition says is already nearly
saturated -- they are included because they are cheap, because they are the two tasks the
literature reports as most transferable, and because a null result from them is
INFORMATIVE given the error decomposition. Expect CONSIST to carry any gain; report the
three separately (that is why each is its own env switch) and do not bundle them.

--------------------------------------------------------------------------------------
LEAKAGE (constraint 2) -- designed first, because it is the whole point of this component
--------------------------------------------------------------------------------------
Pretraining for outer fold f may use ONLY the meshes/clouds of fold f's TRAINING subjects.
Mechanisms, not promises:

  * FOLD is REQUIRED (`main()` refuses to run without it). There is no default fold, so a
    run cannot silently pretrain on the wrong split.
  * `FoldClean` slices the data npz to the training ears with a fancy index -- which
    numpy materialises as a FRESH array -- drops the full array, and then ASSERTS the
    resulting torch tensor's storage is exactly its own numel. A validation row is
    therefore not merely unused, it is not present in any byte the model can reach. The
    assertion is printed.
  * The ground-truth keys are never read. SSL reads `clouds` and (optionally) `nrm` and
    nothing else -- not `true`, not even `coarse` (the ~3.7 mm initial estimate). The
    reader takes an explicit allow-list and reports which GT keys existed in the file and
    were skipped.
  * NO cross-ear statistic of any kind is computed. All normalisation is by FIXED
    constants shared with the rest of the pipeline (SCALE = 30 mm, NRM_R = 3 mm), so there
    is no normalisation statistic, no codebook and no PCA basis that could carry
    validation geometry. If you add one, it must be fitted inside `FoldClean`'s slice.
  * The pretext-loss MONITOR curve is an inner split of the TRAINING subjects
    (MON_FRAC, subject-grouped and asserted disjoint), so even the early-stopping decision
    never touches validation geometry.
  * The checkpoint filename carries the fold (`ssl_<enc>_<tasks>_f<FOLD>_s<SEED>.pt`) AND
    the payload carries `fold` + the train-ear list. `check_ckpt_fold(path, fold)` --
    the function the trainer calls -- asserts the label matches AND independently
    re-derives the fold's validation ears from the frozen rule and asserts the
    checkpoint's training ears are disjoint from them. A mislabelled file cannot pass.
  * The crop bbox of every cloud in the shipped npz derives from `coarse`, not from GT
    (build_multisample_all.py), so the input geometry itself is label-free.

A TRANSDUCTIVE variant -- pretraining on the unlabelled validation (or lockbox) geometry
as well -- is a legitimately different experiment with a legitimately different claim, and
it is deliberately NOT implemented here, not even behind a flag, because a flag is how it
ends up switched on by a copy-pasted command line. If it is run, it must be a separate
script, a separate report and a separate row in the results table, and it must never be
compared against a shipped number that was tuned inductively.

--------------------------------------------------------------------------------------
THE THREE PRETEXT TASKS -- and whether a constant/degenerate solution is possible
--------------------------------------------------------------------------------------
MASKED  masked multi-scale surface reconstruction.
  A CONTIGUOUS region is hidden: the `nmask` points nearest the seed in GEODESIC distance,
  approximated as shortest-path distance on the cloud's kNN graph with edges longer than
  GEO_MAXE dropped, and (when normals are available) edges whose endpoint normals disagree
  by more than GEO_NDOT dropped as well.
    - NOT random scatter: at ~1 mm spacing a scattered hole is filled by interpolating its
      own surviving neighbours, so the task collapses to local smoothing and teaches
      nothing about shape.
    - NOT a Euclidean ball: the ear is exactly the shape where Euclidean != geodesic. A
      6 mm Euclidean ball on the antihelix also deletes part of the facing scapha/helix
      sheet, so the "hole" is two disconnected patches and the rim the model must
      extrapolate from is contaminated by unrelated geometry. The smoke test constructs a
      folded sheet and asserts the two masks genuinely differ.
    - CAVEAT, stated rather than hidden: a kNN-graph geodesic BRIDGES any air gap smaller
      than GEO_MAXE, which must exceed the point spacing (~1.26 mm at SUB_FRAC*2048) to
      keep the graph connected. Real crevices narrower than GEO_MAXE (tragus notch) can
      still be bridged; the normal gate removes the facing-sheet case, which is the common
      one. Exact upgrade path: use mesh connectivity (`nbr`/`v_ptr` in
      scratch/mesh_data.npz) plus a per-(ear,sample) nearest-vertex map, which is a LOCAL
      preprocessing addition, not a GPU-box dependency. Not done here.
  Multi-scale = the masked FRACTION is drawn per step from MASK_SCALES and the loss is
  reported per scale, so scales are attributable instead of blended. One region per step,
  deliberately: several simultaneous balls make the per-scale attribution ambiguous.
  Target: for query points near the hidden region, the vector to the nearest point of the
  CLEAN cloud (a local closest-point / SDF field). Queries are the hidden points displaced
  by an isotropic random offset of length U(QMIN, QOFF).
  DEGENERACY: a constant prediction is NOT a solution -- the offsets are isotropic, so the
  target's mean is ~0 and the constant predictor's loss is the target's full variance,
  which is COMPUTED AND PRINTED every eval as `const` next to the model's loss. If the
  ratio is ~1 the task learnt nothing, and you will see it. The residual shortcut is that
  the target norm is bounded by QOFF, so predicting a shrunken vector is a safe hedge;
  the reported RMS error in mm makes that visible too.
  FLOOR: the target is the nearest point of a FINITE cloud, ~0.43 mm from the true
  surface (measured, README), so the loss cannot reach zero.

CONSIST  independent-resampling consistency.
  Two views of the SAME ear from two DIFFERENT stored surface samples (j != k, asserted),
  sharing one rigid augmentation and jittered independently.
    - Correspondence is MUTUAL nearest neighbour in space, computed on the clean
      coordinates before the per-view jitter, kept only when the distance is <= NN_MAX and
      (with normals) the endpoint normals agree. Justification: the measured
      point-sampling gap between two independent samples is 0.432 mm (README), i.e. two
      orders below the encoder's ~5-7 mm receptive field, so a mutual-NN pair really is
      the same surface location. What mutual-NN gets WRONG is a fold: two points 0.4 mm
      apart across a crevice are nearest neighbours and are not the same location -- hence
      the normal gate, and hence the reported retained fraction and matched distance so a
      degraded match is visible rather than assumed.
    - The shipped clouds are drawn WITH REPLACEMENT from the same 19.5k-41.4k crop
      vertices, so two samples share ~2048^2/n_v ~= 100-210 EXACTLY COINCIDENT points.
      Those are not two independent samples of one location, they are one sample counted
      twice; they are dropped (distance <= DUP_EPS) because keeping them flatters the
      match statistic and lets the encoder match on coordinate identity. The count is
      reported. build_ssl_views.py exists to remove this defect at the source (disjoint,
      without-replacement views) and sets `disjoint=True` in its npz, which this script
      reports.
  DEGENERACY: "the same embedding at matched points" is EXACTLY satisfied by a constant,
  so that objective is not used. The loss is InfoNCE with IN-EAR negatives: matched pairs
  must be more similar to each other than to other points OF THE SAME EAR. A collapsed
  encoder scores log(n_candidates) ~ 5.5 nats, which is COMPUTED AND PRINTED as `collapse`
  next to the loss every eval. Points within EXCL mm of the anchor are excluded from the
  negatives (neither positive nor negative): at 1 mm spacing they are almost the same
  surface location, so pushing them apart is a false negative fighting the smoothness of
  the field -- and with-replacement sampling puts exact copies of the anchor in its own
  view, which the same rule removes. If TAU <= 0 the loss switches to plain MSE
  consistency, and the script REFUSES to run unless VC > 0, because MSE-only admits the
  constant solution.
  We do NOT ask for rotation invariance (both views share one rigid transform). The
  downstream head predicts coordinates in the per-ear canonical frame; an invariance to a
  transform the task depends on is a capacity tax, not a prior.

NORMAL  oriented-normal + local-patch reconstruction from a masked/perturbed input.
  Targets at the hidden points' positions, computed on the CLEAN cloud: the oriented
  per-point normal from `nrm` (triangle-derived and consistently oriented upstream --
  geometry, not annotation), and the local covariance tensor of the k-NN patch normalised
  by a FIXED radius NRM_R (6 upper-triangular entries), which carries the tangent plane,
  the local anisotropy and, through its smallest eigenvalue, the curvature.
  NORMAL_MASK=1 (default) hides a region so this is EXTRAPOLATION from the rim; with
  NORMAL_MASK=0 it degrades to denoising the jittered input, which is much easier -- the
  mode is reported. By default NORMAL builds its OWN masked view (independently seeded)
  so that "NORMAL alone" and "NORMAL with MASKED on" are the same task; SHARE_VIEW=1
  saves one encoder forward per step by sharing MASKED's view and CONFOUNDS the ablation.
  If the data npz has no `nrm`, the oriented target is unavailable and the task degrades
  to a sign-free unoriented-normal loss (1 - cos^2); this is printed, not silent.
  DEGENERACY: a constant is again not a solution and again not merely asserted -- the
  batch-mean predictor's loss is printed as `const`. It is however the WEAKEST of the
  three tasks in this respect: normals in the canonical ear frame have a strong mean
  (the ear is a bowl), so the constant baseline is already decent and the achievable
  margin is small. Judge NORMAL by loss/const, not by loss.

--------------------------------------------------------------------------------------
MEASURED, fold 0 of scratch/screen_data_2048nrm.npz, 2026-07-31 -- read this before
tuning anything, because several of these numbers are not what the geometry looks like
--------------------------------------------------------------------------------------
  * point spacing: median nearest-neighbour 0.88mm at 2048 points, 1.10mm at 1280
    (SUB_FRAC). Median 8th-neighbour 2.90mm / 3.65mm -- for a RANDOM (not lattice) sample
    those differ by ~3.3x, which is why the graph limit is measured against the k-th
    neighbour and not the first.
  * the crop is BIG: back out the area from the spacing and it is ~7000mm^2, several times
    the ear proper, because the coarse-landmark bbox + 14mm margin brings in surrounding
    head. CONSEQUENCE FOR MASK_SCALES: a physically small hole contains almost no points
    (a 3mm-radius patch holds ~5 points at 1280), so the useful mask range is a FRACTION
    range, and the defaults 0.05/0.15/0.35 correspond to geodesic radii of roughly
    11/20/37mm (37mm measured directly, the others from the area). Nobody should read
    "masked surface reconstruction" here as inpainting a 3mm dimple.
  * 2-4% of an ear crop's points sit in small disconnected islands (min graph degree 0 on
    every ear tested), so seed re-drawing is required, not optional -- see mask_view.
  * mutual-NN correspondence between two stored samples: 31-33% of points get a partner,
    mean matched distance 0.73mm at 1280 points, and 59-62 pairs PER EAR are exactly
    coincident and dropped -- the with-replacement artefact, at the size predicted by
    NPTS^2/n_v. build_ssl_views.py removes it at the source.
  * cost: 158s per epoch for all three tasks on 232 fit ears at BS=8 on this CPU (4 encoder
    forwards per step). One CPU epoch already takes CONSIST to 0.70 of its collapse value.
  * the fine-tune model is 813,232 parameters -- identical to the shipped baseline, which
    is the check that LandmarkHead + DGCNNEncoder + the contour stage really is the shipped
    architecture and not a lookalike.

--------------------------------------------------------------------------------------
WHAT THE FINE-TUNE INHERITS
--------------------------------------------------------------------------------------
ONLY the encoder. Every pretext decoder is discarded, deliberately: they are conditioned
on query points that do not exist downstream, so keeping them would smuggle task-specific
capacity into the comparison against a from-scratch control. The control arm is the SAME
architecture with CFG_SSL_CKPT="" CFG_REQUIRE_CKPT=false, which is the only comparison
that isolates pretraining.

FREEZE_EPOCHS: the encoder is frozen (requires_grad=False AND eval() mode, so a sibling
encoder's norm statistics also stop moving) for the first FREEZE_EPOCHS epochs, then
unfrozen. train_family.py has no per-epoch model hook, so the epoch counter increments on
each `model.train(True)` call, which that trainer makes exactly once per epoch. A trainer
that calls train() more often would unfreeze early; `model.set_epoch(ep)` overrides the
counter and takes precedence, and calling it is the CONTRACT a trainer should honour.

--------------------------------------------------------------------------------------
ENVIRONMENT (defaults in brackets; every value is echoed into the report)
--------------------------------------------------------------------------------------
  FOLD             REQUIRED for pretraining. Outer fold 0..4 of the frozen split.
  SEED        [0]  seeds torch + numpy + the sampler streams
  EPOCHS      [300]
  DATA        [$WORK/screen_data_2048nrm.npz]   needs `clouds` (E,M,N,3); `nrm` optional
  WORK        [scratch]      checkpoint + report directory
  ENCODER     [dgcnn]  dgcnn|pointnext|kpconv|ptv3|diffusionnet, via
                       fam_template.make_encoder, which falls back to dgcnn LOUDLY (and
                       today always does -- see that module's measured note)
  WIDTH [256]  GK [20]  SCALE [30.0]  SUB_FRAC [0.625]  (SUB_FRAC matches the trainer's
                       augmentation, so SSL sees the same point density the fine-tune does)
  MASKED [1] CONSIST [1] NORMAL [1]         the three switches
  W_MASKED [1.0] W_CONSIST [1.0] W_NORMAL [1.0]   comparable, because each loss is
                       divided by its OWN degeneracy baseline (NORM_BY_CONST [1], see
                       _ratio); a logged value of 1.0 therefore means "no better than a
                       constant predictor". Vary ONE at a time regardless.
  BS [8] LR [1e-3] WD [1e-4] MON_FRAC [0.15] EVAL_EVERY [max(1,EPOCHS//12)]
  AUG_ROT [1.2] AUG_SCALE [0.20] AUG_JIT [0.25]     as TRAIN_DEFAULTS
  MASK_SCALES [0.05,0.15,0.35]  masked fraction of the input points, one per step
  GEO_K [8]       surface-graph degree
  GEO_FAC [1.2]   edge-length limit as a MULTIPLE of that cloud's median k-th-neighbour
                  distance -- the real knob, adaptive to the point count. MEASURED on real
                  fold-0 crops: 4.35mm at 1280 points (11.8% of kNN edges dropped), 3.46mm
                  at 2048 (12.4% dropped)
  GEO_MAXE [6.0]  absolute ceiling on that limit (mm); non-binding at the defaults
  GEO_RETRY [4]   seed re-draws for an item whose ball is not reachable (see mask_view)
  GEO_NDOT [0.0]  drop graph edges whose endpoint normals dot below this (needs `nrm`)
  GEO_ITERS [0]   min-plus sweeps; 0 = auto = clamp(3*sqrt(nmask), 24, 200)
  NQ [256] QMIN [0.3] QOFF [3.0] MASK_K [24] MASK_CLIP [6.0]
  NPAIR [256] TAU [0.1] EXCL [2.0] NN_MAX [1.2] DUP_EPS [1e-3] PROJ [64] VC [0.0]
  KNRM [16] NRM_R [3.0] NORM_JIT [0.5] W_NRM_VEC [1.0] W_COV [1.0] NORMAL_MASK [1]
  SHARE_VIEW [0]
  fine-tune only: CFG_SSL_CKPT [""] CFG_FREEZE_EPOCHS [0] CFG_REQUIRE_CKPT [true]
                  CFG_K [48] CFG_NPASS [4] CFG_MAX_OFF [0.0=unbounded] CFG_USE_NRM [false]

OUTPUT
  $WORK/ssl_<encoder>_<tasks>_f<FOLD>_s<SEED>.pt     encoder weights + fold proof
  $WORK/ssl_<encoder>_<tasks>_f<FOLD>_s<SEED>.json   config, curves, diagnostics
<tasks> is a subset of the fixed string "MCN", so the file name states the task set and
the fold, and no fine-tune can load a checkpoint whose provenance is ambiguous.
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(os.path.dirname(_HERE))):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# reuse, do not re-implement: make_encoder already carries the sibling-encoder probe and
# the honest dgcnn fallback, and the DGCNNEncoder it returns is the shipped model's
# backbone verbatim (which is why the fine-tune's parameter count lands on the shipped
# 813,232 to the unit).
from fam_template import make_encoder, gather_pts, knn   # noqa: E402

NL, NFOLD = 85, 5
CONTOURS = [(0, 24), (25, 54), (55, 74), (75, 84)]


def _flag(name, default):
    return os.environ.get(name, default) not in ("0", "false", "False", "no", "")


def _floats(name, default):
    return [float(x) for x in os.environ.get(name, default).split(",")]


SEED = int(os.environ.get("SEED", "0"))
EPOCHS = int(os.environ.get("EPOCHS", "300"))
WORK = os.environ.get("WORK", "scratch")
DATA = os.environ.get("DATA", f"{WORK}/screen_data_2048nrm.npz")
ENCODER = os.environ.get("ENCODER", "dgcnn")
WIDTH = int(os.environ.get("WIDTH", "256"))
GK = int(os.environ.get("GK", "20"))
SCALE = float(os.environ.get("SCALE", "30.0"))
SUB_FRAC = float(os.environ.get("SUB_FRAC", "0.625"))
BS = int(os.environ.get("BS", "8"))
LR = float(os.environ.get("LR", "1e-3"))
WD = float(os.environ.get("WD", "1e-4"))
MON_FRAC = float(os.environ.get("MON_FRAC", "0.15"))
AUG_ROT = float(os.environ.get("AUG_ROT", "1.2"))
AUG_SCALE = float(os.environ.get("AUG_SCALE", "0.20"))
AUG_JIT = float(os.environ.get("AUG_JIT", "0.25"))

MASKED = _flag("MASKED", "1")
CONSIST = _flag("CONSIST", "1")
NORMAL = _flag("NORMAL", "1")
W_MASKED = float(os.environ.get("W_MASKED", "1.0"))
W_CONSIST = float(os.environ.get("W_CONSIST", "1.0"))
W_NORMAL = float(os.environ.get("W_NORMAL", "1.0"))
SHARE_VIEW = _flag("SHARE_VIEW", "0")

MASK_SCALES = _floats("MASK_SCALES", "0.05,0.15,0.35")
GEO_K = int(os.environ.get("GEO_K", "8"))
GEO_FAC = float(os.environ.get("GEO_FAC", "1.2"))
GEO_MAXE = float(os.environ.get("GEO_MAXE", "6.0"))
GEO_RETRY = int(os.environ.get("GEO_RETRY", "4"))
GEO_NDOT = float(os.environ.get("GEO_NDOT", "0.0"))
GEO_ITERS = int(os.environ.get("GEO_ITERS", "0"))
NQ = int(os.environ.get("NQ", "256"))
QMIN = float(os.environ.get("QMIN", "0.3"))
QOFF = float(os.environ.get("QOFF", "3.0"))
MASK_K = int(os.environ.get("MASK_K", "24"))
MASK_CLIP = float(os.environ.get("MASK_CLIP", "6.0"))

NPAIR = int(os.environ.get("NPAIR", "256"))
TAU = float(os.environ.get("TAU", "0.1"))
EXCL = float(os.environ.get("EXCL", "2.0"))
NN_MAX = float(os.environ.get("NN_MAX", "1.2"))
DUP_EPS = float(os.environ.get("DUP_EPS", "1e-3"))
PROJ = int(os.environ.get("PROJ", "64"))
VC = float(os.environ.get("VC", "0.0"))

KNRM = int(os.environ.get("KNRM", "16"))
NRM_R = float(os.environ.get("NRM_R", "3.0"))
NORM_JIT = float(os.environ.get("NORM_JIT", "0.5"))
W_NRM_VEC = float(os.environ.get("W_NRM_VEC", "1.0"))
W_COV = float(os.environ.get("W_COV", "1.0"))
# Every pretext loss is divided by its own degeneracy baseline (see _ratio): with raw
# losses the covariance term needed W_COV=20 to balance on the smoke geometry and 0.7 on
# real crops, i.e. the weight was absorbing a units mismatch. 0 = raw losses, and then the
# magnitudes in the report are the only guide to the weights.
NORM_BY_CONST = _flag("NORM_BY_CONST", "1")
NORMAL_MASK = _flag("NORMAL_MASK", "1")

BIG = 1e9                      # stands in for inf: keeps min-plus arithmetic nan-free
GT_KEYS = frozenset({"true", "true_lms", "gt", "gt_lms", "landmarks", "lms",
                     "target", "targets", "lm_bary", "lm_vert", "lm_face", "lm_dist"})
READ_KEYS = ("clouds", "nrm")          # the ONLY keys this file ever reads from DATA


# ============================================================ fold-clean data access
def frozen_folds(ne):
    """Constraint 3, verbatim. subject = ear_index//2; array_split(RS(12345).perm, 5)."""
    subj = np.arange(ne) // 2
    parts = np.array_split(np.random.RandomState(12345).permutation(np.unique(subj)), NFOLD)
    return subj, [np.asarray(p) for p in parts]


def fold_ears(ne, fold):
    subj, parts = frozen_folds(ne)
    val_s = set(parts[fold].tolist())
    tr = np.array([i for i in range(ne) if subj[i] not in val_s], np.int64)
    va = np.array([i for i in range(ne) if subj[i] in val_s], np.int64)
    return subj, tr, va


def _storage_exact(t):
    """True iff the tensor owns exactly its own bytes -- i.e. it is not a view into a
    larger buffer that still holds the rows we sliced away."""
    return t.untyped_storage().nbytes() == t.numel() * t.element_size()


class FoldClean:
    """Fold f's TRAINING geometry, and nothing else, with the assertions as the API.

    The full arrays exist only inside __init__, as the transient right-hand side of a
    fancy index; what survives is a fresh contiguous copy of the training rows whose
    storage size is asserted equal to its numel. `report()` prints the proof.
    """

    def __init__(self, path, fold, dev, want_nrm=True):
        z = np.load(path, allow_pickle=True)
        keys = set(z.files)
        assert "clouds" in keys, f"{path} has {sorted(keys)}, needs `clouds` (E,M,N,3)"
        raw = z["clouds"]
        assert raw.ndim == 4 and raw.shape[-1] == 3, f"clouds must be (E,M,N,3), got {raw.shape}"
        ne, shp = raw.shape[0], raw.shape
        subj, tr, va = fold_ears(ne, fold)
        cl = np.ascontiguousarray(raw[tr])        # fancy index -> fresh array, no val rows
        del raw
        nr = None
        if want_nrm and "nrm" in keys:
            rawn = z["nrm"]
            assert rawn.shape == shp, f"nrm {rawn.shape} does not match clouds {shp}"
            nr = np.ascontiguousarray(rawn[tr])
            del rawn
        self.path, self.fold, self.dev = path, int(fold), dev
        self.ne_file, self.subj = int(ne), subj
        self.ear = tr.copy()
        self.val_ear = va.copy()
        self.disjoint = bool(z["disjoint"]) if "disjoint" in keys else False
        self.skipped_gt = sorted(GT_KEYS & keys)
        self.skipped_other = sorted(keys - GT_KEYS - set(READ_KEYS))
        z.close()

        self.clouds = torch.tensor(cl).float().to(dev)
        self.nrm = torch.tensor(nr).float().to(dev) if nr is not None else None
        self.E, self.M, self.N = self.clouds.shape[:3]

        # --- the leakage assertions (constraint 2), as executable statements
        assert self.E == len(tr), f"loaded {self.E} rows for {len(tr)} training ears"
        assert not (set(self.ear.tolist()) & set(va.tolist())), "train/val ear overlap"
        assert not np.isin(subj[self.ear], subj[va]).any(), \
            "a loaded ear shares a SUBJECT with a validation ear -- grouping broken"
        assert _storage_exact(self.clouds), \
            "clouds tensor is a view into a buffer that still holds validation rows"
        assert self.nrm is None or _storage_exact(self.nrm), "nrm tensor is a view"
        assert torch.isfinite(self.clouds).all(), "non-finite point in the training clouds"
        self.checks_passed = True

    def report(self):
        mb = self.clouds.numel() * 4 / 1e6 + (self.nrm.numel() * 4 / 1e6 if self.nrm is not None else 0)
        return (f"[fold-clean f{self.fold}] {self.path}: {self.ne_file} ears in file -> "
                f"{self.E} TRAIN ears ({len(np.unique(self.subj[self.ear]))} subjects) loaded, "
                f"{len(self.val_ear)} VAL ears never materialised\n"
                f"   tensors {tuple(self.clouds.shape)}"
                f"{' + nrm' if self.nrm is not None else ' (no nrm)'} = {mb:.1f} MB, "
                f"storage == numel: PASS | no val ear reachable: PASS | "
                f"subject-disjoint: PASS\n"
                f"   GT keys present in file and NOT read: {self.skipped_gt} | "
                f"other keys not read: {self.skipped_other} | "
                f"views {'DISJOINT (build_ssl_views.py)' if self.disjoint else 'with-replacement (shipped npz)'}")

    def inner_split(self, mon_frac, rs):
        """subject-grouped inner split of the TRAINING ears -- the SSL monitor set."""
        s = np.unique(self.subj[self.ear])
        nm = max(1, int(round(len(s) * mon_frac)))
        mon_s = set(rs.permutation(s)[:nm].tolist())
        loc = np.arange(self.E)
        mon = loc[np.isin(self.subj[self.ear], list(mon_s))]
        fit = loc[~np.isin(self.subj[self.ear], list(mon_s))]
        assert not (set(mon.tolist()) & set(fit.tolist()))
        assert not np.isin(self.subj[self.ear[mon]], self.subj[self.ear[fit]]).any(), \
            "SSL monitor split is not subject-grouped"
        return fit, mon


# ============================================================ geometry (pure torch)
def rand_rot(B, maxang, gen, dev):
    ax = torch.randn(B, 3, device=dev, generator=gen)
    ax = ax / ax.norm(dim=1, keepdim=True)
    ang = (torch.rand(B, device=dev, generator=gen) - .5) * maxang
    c, s = ang.cos(), ang.sin(); x, y, z = ax[:, 0], ax[:, 1], ax[:, 2]; C = 1 - c
    return torch.stack([
        torch.stack([c + x*x*C, x*y*C - z*s, x*z*C + y*s], -1),
        torch.stack([y*x*C + z*s, c + y*y*C, y*z*C - x*s], -1),
        torch.stack([z*x*C - y*s, z*y*C + x*s, c + z*z*C], -1)], 1)


def rot(t, R):
    return torch.einsum("bnj,bij->bni", t, R)


def knn_graph(pc, k, fac, cap, nrm=None, ndot=None):
    """Surface graph of a point cloud: kNN edges, with edges longer than
    `fac` x (that cloud's median k-th-neighbour distance) -- capped at `cap` mm -- removed,
    plus (optionally) edges whose endpoint normals disagree.

    The length limit is ADAPTIVE because a fixed millimetre cap is wrong for the point
    counts this file mixes: 2048 stored points, SUB_FRAC*2048 after subsampling, and
    whatever a caller passes. It is measured against the k-th neighbour, not the first,
    because for a random (not lattice) sample the two differ by ~3.2x at k=8 -- the first
    version used the nearest-neighbour distance with fac=2.2 and threw away 60% of the
    edges, which is how the smoke test caught it. This is a WITHIN-CLOUD quantity computed
    at run time from the input itself, so it is not a cross-ear statistic and cannot carry
    fold information (constraint 2).

    Calibration to be aware of: at 1280 points on a ~2030mm^2 ear crop the k=8 radius is
    ~2.0mm, so the default limit is ~3.0mm and a real crevice narrower than that CAN still
    be bridged. GEO_NDOT (facing sheets) is the cheap defence; mesh connectivity is the
    exact one and is not implemented here.
    """
    nd, idx = torch.cdist(pc, pc).topk(k + 1, largest=False, dim=-1)
    nd, idx = nd[..., 1:], idx[..., 1:]
    nn1 = nd[..., 0].median(-1).values                        # (B,) nearest-neighbour
    knn_r = nd[..., -1].median(-1).values                     # (B,) k-th neighbour
    maxe = torch.clamp(fac * knn_r, max=cap)[:, None, None]
    keep = nd <= maxe
    n_cap = int((~keep).sum())
    n_gate = 0
    if nrm is not None and ndot is not None:
        agree = (gather_pts(nrm, idx) * nrm[:, :, None, :]).sum(-1) >= ndot
        n_gate = int((agree.logical_not() & keep).sum())
        keep = keep & agree
    w = torch.where(keep, nd, torch.full_like(nd, BIG))
    return idx, w, keep.sum(-1), dict(
        edges_capped=n_cap, edges_normal_gated=n_gate, edge_total=int(w.numel()),
        nn1_mm=float(nn1.mean()), knn_r_mm=float(knn_r.mean()),
        geo_maxe_mm=float(maxe.mean()))


def geodesic(idx, w, seed, iters):
    """min-plus relaxation on a SYMMETRISED kNN graph -> geodesic distance from `seed`.

    Symmetrised deliberately. Relaxing only i from its own kNN list makes the graph
    directed, and a seed sitting in a locally sparse patch then has no OUTGOING edge and
    reaches nothing -- which is exactly how this failed on the first run of the
    learnability probe, not a hypothetical. The reverse sweep is one scatter_reduce.
    `iters` sweeps give the exact shortest path using at most `iters` edges, so an
    under-sized `iters` OVERestimates far distances; the caller asserts the selected
    frontier is finite.
    """
    B, N, k = w.shape
    g = torch.full((B, N), BIG, device=w.device).scatter_(1, seed[:, None], 0.0)
    flat, wf = idx.reshape(B, -1), w.reshape(B, -1)
    for _ in range(iters):
        g = torch.minimum(g, (g.gather(1, flat).view(B, N, k) + w).amin(-1))     # into i
        g = g.scatter_reduce(1, flat, g.repeat_interleave(k, 1) + wf, reduce="amin")
    return g


def euclid_frontier(pc, seed, nmask):
    """the same-cardinality EUCLIDEAN ball, for the geodesic-vs-Euclidean diagnostic"""
    d = (pc - pc.gather(1, seed[:, None, None].expand(-1, 1, 3))).norm(dim=-1)
    return d.argsort(-1)[:, :nmask]


def local_cov(pc, k, ref_r):
    """normalised local covariance of the k-NN patch (6 upper-triangular entries) and the
    unoriented PCA normal. Normalisation is by the FIXED ref_r, never by a data statistic."""
    idx = knn(pc, pc, k + 1)[..., 1:]
    Q = gather_pts(pc, idx) - pc[:, :, None, :]
    C = torch.einsum("bnki,bnkj->bnij", Q, Q) / (Q.shape[2] * ref_r ** 2)
    ev, evec = torch.linalg.eigh(C)
    iu = torch.triu_indices(3, 3, device=pc.device)
    return C[..., iu[0], iu[1]], evec[..., 0], ev


def mutual_nn(pa, pb, nn_max, dup_eps, na=None, nb=None, ndot=0.0):
    """MUTUAL nearest-neighbour correspondence a -> b on CLEAN coordinates.

    Returns (j_of_a, ok, dist). `ok` drops non-mutual pairs, pairs beyond nn_max, EXACTLY
    coincident pairs (the with-replacement artefact: one sample counted twice, not two
    samples of one location) and, when normals are given, pairs across a fold.

    The reported distance is RECOMPUTED by subtraction, not read off cdist: cdist uses the
    expanded ||a||^2+||b||^2-2ab form, whose float32 error at ear-scale coordinates is
    ~0.01mm -- two hundred times DUP_EPS -- so the duplicate test would silently never
    fire if it trusted cdist. Measured while writing the smoke test, not in theory.
    """
    d = torch.cdist(pa, pb)
    ja = d.argmin(-1)
    ib = d.argmin(-2)                                    # (B,Nb) best a for each b
    pb_sel = gather_pts(pb, ja[..., None])[:, :, 0]
    dist = (pa - pb_sel).norm(dim=-1)
    back = ib.gather(1, ja)
    ok = (back == torch.arange(pa.shape[1], device=pa.device)[None]) \
        & (dist <= nn_max) & (dist > dup_eps)
    n_dup = int((dist <= dup_eps).sum())
    if na is not None and nb is not None:
        ok = ok & ((na * gather_pts(nb, ja[..., None])[:, :, 0]).sum(-1) >= ndot)
    return ja, ok, dist, n_dup


# ============================================================ view construction
def make_views(data, ears, samples, gen, cfg, need):
    """ears/samples: numpy (B,) / (B,S). One rigid transform per item shared by all views;
    independent jitter per view. Returns dict of (B,NSUB,3) clouds (+ normals)."""
    dev = data.clouds.device
    e = torch.as_tensor(ears, dtype=torch.long, device=dev)
    B = len(ears)
    N = data.N
    nsub = max(16, int(round(N * cfg["sub_frac"])))
    R = rand_rot(B, cfg["aug_rot"], gen, dev)
    sc = 1 + (torch.rand(B, 1, 1, device=dev, generator=gen) - .5) * cfg["aug_scale"]
    out = {}
    for v, col in enumerate(need):
        s = torch.as_tensor(samples[:, v], dtype=torch.long, device=dev)
        pc = data.clouds[e, s]
        sub = torch.rand(B, N, device=dev, generator=gen).argsort(-1)[:, :nsub]
        p = rot(torch.gather(pc, 1, sub[..., None].expand(-1, -1, 3)), R) * sc
        out[col] = p
        if data.nrm is not None:
            nn_ = torch.gather(data.nrm[e, s], 1, sub[..., None].expand(-1, -1, 3))
            out[col + "_n"] = Fn.normalize(rot(nn_, R), dim=-1)
        else:
            out[col + "_n"] = None
    out["nsub"] = nsub
    return out


def jitter(p, sd, gen):
    return p + torch.randn(p.shape, device=p.device, generator=gen) * sd


def mask_view(pc, nrm, frac, gen, cfg):
    """Hide the `nmask` points nearest a random seed in graph-geodesic distance, then pad
    the visible set back to N so the batch stays static-shaped.

    Padding: the visible points are used once each in random order and the first
    (N - nvis) of that order are used a second time, so the density in the visible region
    rises by N/nvis (<= 1.54 at the largest default scale). Documented rather than hidden;
    SUB_FRAC is set to the trainer's own subsample fraction so this straddles, rather than
    departs from, the density the fine-tuned model sees.
    """
    B, N, _ = pc.shape
    dev = pc.device
    nmask = max(8, min(N - 16, int(round(N * frac))))
    idx, w, deg, gstat = knn_graph(pc, cfg["geo_k"], cfg["geo_fac"], cfg["geo_maxe"],
                                   nrm, cfg["geo_ndot"])
    it = cfg["geo_iters"] or int(min(200, max(24, 3.0 * np.sqrt(nmask))))
    # MEASURED on real fold-0 crops: 2-4% of the points of an ear crop sit in small
    # disconnected islands (min degree 0 on every ear), so a uniformly drawn seed lands in
    # an island a few percent of the time and its ball then holds 3-19 points instead of
    # nmask. Hence: draw the seed only among well-connected points, and RE-DRAW for the
    # items that still cannot reach nmask. One attempt suffices almost always; the
    # assertion below is the last resort, not the mechanism.
    g, bad, tries = None, None, 0
    for tries in range(1, int(cfg["geo_retry"]) + 1):
        elig = deg >= max(1, cfg["geo_k"] // 2)
        seed = torch.where(elig, torch.rand(B, N, device=dev, generator=gen),
                           torch.full((B, N), 2.0, device=dev)).argmin(-1)
        gi = geodesic(idx, w, seed, it)
        short = (gi < BIG / 2).sum(-1) < nmask
        g = gi if g is None else torch.where(bad[:, None], gi, g)
        bad = short if bad is None else (bad & short)
        if not bool(bad.any()):
            break
    order = g.argsort(-1)
    m_idx = order[:, :nmask]
    frontier = g.gather(1, m_idx[:, -1:])
    assert not bool(bad.any()), (
        f"{int(bad.sum())}/{B} items have no connected {nmask}-point region after {tries} "
        f"seed draws (graph limit {gstat['geo_maxe_mm']:.2f}mm at k-th-nb "
        f"{gstat['knn_r_mm']:.2f}mm, min degree {int(deg.min())}, {it} sweeps). Raise "
        f"GEO_FAC/GEO_K/GEO_ITERS or lower the mask scale.")
    keep = torch.ones(B, N, dtype=torch.bool, device=dev).scatter_(1, m_idx, False)
    nvis = N - nmask
    rank = torch.where(keep, torch.rand(B, N, device=dev, generator=gen),
                       torch.full((B, N), 2.0, device=dev))
    vis = rank.argsort(-1)[:, :nvis]
    sel = vis[:, torch.arange(N, device=dev) % nvis]
    sel = sel.gather(1, torch.rand(B, N, device=dev, generator=gen).argsort(-1))
    gstat.update(nmask=nmask, nvis=nvis, geo_radius_mm=float(frontier.mean()),
                 sweeps=it, seed_draws=tries)
    return sel, m_idx, gstat


# ============================================================ heads
class QueryDecoder(nn.Module):
    """query-conditioned decoder: read the K nearest INPUT points' features + relative
    positions and predict `nout` numbers at an arbitrary query location."""

    def __init__(self, C, nout, k=MASK_K, hid=128):
        super().__init__()
        self.k, self.nout = k, nout
        self.pt = nn.Sequential(nn.Linear(C + 4, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU())
        self.out = nn.Sequential(nn.Linear(2 * hid + 1, hid), nn.ReLU(), nn.Linear(hid, nout))

    def forward(self, q, pc, h):
        idx = knn(q, pc, self.k)
        pK, fK = gather_pts(pc, idx), gather_pts(h, idx)
        rel = (pK - q[:, :, None, :]) / SCALE
        dn = rel.norm(dim=-1, keepdim=True)
        x = self.pt(torch.cat([fK, rel, dn], -1))
        return self.out(torch.cat([x.mean(2), x.max(2).values, dn.min(2).values], -1))


class SSLNet(nn.Module):
    """the pluggable encoder plus one head per enabled pretext task"""

    def __init__(self, encoder=ENCODER, cin=3, width=WIDTH, gk=GK, npts=1280,
                 tasks=("M", "C", "N")):
        super().__init__()
        self.enc, self.C = make_encoder(encoder, cin=cin, C=width, gk=gk, npts=min(npts, 512))
        self.enc_name, self.cin, self.width, self.gk = encoder, cin, width, gk
        self.tasks = tuple(tasks)
        self.head_m = QueryDecoder(self.C, 3) if "M" in tasks else None
        self.head_n = QueryDecoder(self.C, 9) if "N" in tasks else None
        self.proj = nn.Sequential(nn.Linear(self.C, self.C), nn.ReLU(),
                                  nn.Linear(self.C, PROJ)) if "C" in tasks else None

    def encode(self, pc, nrm=None):
        return self.enc(pc, nrm if self.cin > 3 else None)


# ============================================================ pretext losses
def _ratio(L, const):
    """Every pretext loss is divided by its OWN degeneracy baseline (the best constant
    predictor on this batch, or log(n_candidates) for InfoNCE).

    Why, and what it costs. The three tasks have incommensurable units (mm^2 / nats /
    dimensionless second moments) and their raw magnitudes are not even stable across
    datasets: W_COV=20 balanced the covariance term against the normal term on the smoke
    geometry and OVER-weighted it by 28x on real ear crops, where the covariance target is
    28x larger. Dividing by the baseline makes every W_* dimensionless and comparable, and
    makes the logged number BE the degeneracy ratio -- 1.0 means "no better than a
    constant", which is the quantity this component has to be judged on. The divisor is a
    function of the TARGET only (no parameter appears in it), so this is exactly a per-batch
    reweighting and nothing subtler; the cost is that a batch with an unusually small
    baseline gets a proportionally larger step. Set NORM_BY_CONST=0 for raw losses.
    """
    return L / const if (NORM_BY_CONST and float(const) > 0) else L


def masked_loss(net, pc_in, h, q, tgt):
    """closest-point field at the queries. `const` is the batch-mean predictor's loss."""
    pred = MASK_CLIP * torch.tanh(net.head_m(q, pc_in, h) / MASK_CLIP)
    L = ((pred - tgt) ** 2).sum(-1).mean()
    const = ((tgt - tgt.mean((0, 1), keepdim=True)) ** 2).sum(-1).mean()
    return _ratio(L, const), dict(
        masked=float(L / const), masked_raw=float(L), masked_const=float(const),
        masked_rms_mm=float((pred - tgt).norm(dim=-1).pow(2).mean().sqrt()),
        masked_tgt_rms_mm=float(tgt.norm(dim=-1).pow(2).mean().sqrt()))


def infonce(za, zb, pa, pb, valid, tau, excl):
    """anchors za (B,P,D) vs candidates zb; positive = the diagonal.

    Points within `excl` mm of the anchor are neither positive nor negative: at ~1mm
    spacing they are nearly the same surface location (and with-replacement sampling puts
    exact copies of the anchor in its own view), so a repulsion there is a false negative.
    """
    B, P, _ = za.shape
    dev = za.device
    logits = torch.einsum("bpd,bqd->bpq", za, zb) / tau
    eye = torch.eye(P, dtype=torch.bool, device=dev)[None]
    near = (torch.cdist(pa, pb) < excl) & ~eye
    bad = near | (~valid[:, None, :] & ~eye)
    logits = logits.masked_fill(bad, -1e4)
    lab = torch.arange(P, device=dev)[None].expand(B, -1)
    ce = Fn.cross_entropy(logits.reshape(B * P, P), lab.reshape(B * P), reduction="none").view(B, P)
    w = valid.float()
    L = (ce * w).sum() / w.sum().clamp(min=1)
    ncand = (~bad).float().sum(-1)
    return L, float((ncand * w).sum() / w.sum().clamp(min=1)), float(near.float().sum(-1).mean())


def consist_loss(net, hA, hB, pA, pB, pairs):
    ia, jb, valid = pairs["ia"], pairs["jb"], pairs["valid"]
    zA = Fn.normalize(net.proj(hA.gather(1, ia[..., None].expand(-1, -1, hA.shape[-1]))), dim=-1)
    zB = Fn.normalize(net.proj(hB.gather(1, jb[..., None].expand(-1, -1, hB.shape[-1]))), dim=-1)
    qa = pA.gather(1, ia[..., None].expand(-1, -1, 3))
    qb = pB.gather(1, jb[..., None].expand(-1, -1, 3))
    if TAU > 0:
        L1, nc, nex = infonce(zA, zB, qa, qb, valid, TAU, EXCL)
        L2, _, _ = infonce(zB, zA, qb, qa, valid, TAU, EXCL)
        L = 0.5 * (L1 + L2)
        coll = float(np.log(max(nc, 1.0)))               # the collapsed encoder's score
        log = dict(consist=float(L) / max(coll, 1e-9), consist_raw=float(L),
                   consist_collapse=coll, consist_cand=nc, consist_excluded=nex)
        L = _ratio(L, coll)
    else:
        w = valid.float()[..., None]
        L = (((zA - zB) ** 2).sum(-1) * valid.float()).sum() / valid.float().sum().clamp(min=1)
        log = dict(consist=float(L), consist_collapse=0.0, consist_cand=0.0, consist_excluded=0.0)
    if VC > 0:                          # VICReg-style guard: only meaningful for TAU<=0
        z = torch.cat([zA, zB], 1).reshape(-1, zA.shape[-1])
        sd = z.var(0, unbiased=False).clamp(min=1e-8).sqrt()
        L = L + VC * Fn.relu(1.0 - sd).mean()
        log["consist_z_sd"] = float(sd.mean())
    log["consist_pairs"] = float(valid.float().sum(-1).mean())
    return L, log


def normal_loss(net, pc_in, h, q, n_tgt, cov_tgt, oriented):
    o = net.head_n(q, pc_in, h)
    nhat = Fn.normalize(o[..., :3], dim=-1)
    cos = (nhat * n_tgt).sum(-1)
    Ln = (1 - cos).mean() if oriented else (1 - cos ** 2).mean()
    Lc = ((o[..., 3:] - cov_tgt) ** 2).mean()
    cconst = ((cov_tgt - cov_tgt.mean((0, 1), keepdim=True)) ** 2).mean()
    nbar = Fn.normalize(n_tgt.mean((0, 1), keepdim=True), dim=-1)
    nconst = (1 - (nbar * n_tgt).sum(-1)).mean() if oriented \
        else (1 - (nbar * n_tgt).sum(-1) ** 2).mean()
    L = W_NRM_VEC * _ratio(Ln, nconst) + W_COV * _ratio(Lc, cconst)
    return L, dict(normal=float(L), normal_nrm=float(Ln / nconst),
                   normal_nrm_raw=float(Ln), normal_nrm_const=float(nconst),
                   normal_cov=float(Lc / cconst), normal_cov_raw=float(Lc),
                   normal_cov_const=float(cconst),
                   normal_cos=float(cos.mean()), normal_oriented=float(oriented))


# ============================================================ one SSL step
def masked_encode(net, pc, nr, frac, gen, cfg):
    """hide a geodesic region, jitter what is left, encode it. One hole per call, so
    MASKED and NORMAL can each own an independently seeded hole (SHARE_VIEW=0)."""
    sel, m_idx, gstat = mask_view(pc, nr, frac, gen, cfg)
    g3 = sel[..., None].expand(-1, -1, 3)
    pc_in = jitter(pc.gather(1, g3), cfg["aug_jit"], gen)
    nr_in = nr.gather(1, g3) if nr is not None else None
    return dict(pc=pc, nr=nr, pc_in=pc_in, h=net.encode(pc_in, nr_in), m_idx=m_idx,
                stat=gstat)


def ssl_step(net, data, ears, gen, sampler, cfg, tasks):
    """all enabled pretext tasks on one batch of ears. Returns (loss, logs)."""
    B = len(ears)
    dev = data.clouds.device
    own_n = "N" in tasks and not (SHARE_VIEW and "M" in tasks)
    need = [t for t in ("m", "a", "b", "n") if
            (t == "m" and "M" in tasks) or (t in "ab" and "C" in tasks) or
            (t == "n" and own_n)]
    if not need:
        raise AssertionError("no pretext task enabled")
    # distinct stored samples per view: CONSIST is meaningless if A and B are the same
    # sample (identical points make the correspondence exact and coordinate-matchable)
    cols = np.stack([sampler.permutation(data.M)[:len(need)] if data.M >= len(need)
                     else sampler.randint(0, data.M, len(need)) for _ in range(B)])
    if "C" in tasks:
        ia, ib = need.index("a"), need.index("b")
        assert (cols[:, ia] != cols[:, ib]).all(), "CONSIST drew the same sample twice"
    V = make_views(data, ears, cols, gen, cfg, need)
    nsub = V["nsub"]
    L = 0.0
    logs = {}

    scale_frac = float(cfg["mask_scales"][sampler.randint(len(cfg["mask_scales"]))])
    if "M" in tasks or ("N" in tasks and NORMAL_MASK):
        logs["mask_frac"] = scale_frac          # only meaningful when a hole is made
    mv = masked_encode(net, V["m"], V["m_n"], scale_frac, gen, cfg) if "M" in tasks else None
    if mv is not None:
        logs.update(mv["stat"])

    if "M" in tasks:
        nq = min(NQ, mv["m_idx"].shape[1])
        pick = mv["m_idx"][:, torch.randperm(mv["m_idx"].shape[1], device=dev,
                                             generator=gen)[:nq]]
        ph = mv["pc"].gather(1, pick[..., None].expand(-1, -1, 3))
        u = Fn.normalize(torch.randn(B, nq, 3, device=dev, generator=gen), dim=-1)
        r = QMIN + (QOFF - QMIN) * torch.rand(B, nq, 1, device=dev, generator=gen)
        q = ph + u * r
        nnq = knn(q, mv["pc"], 1)                          # nearest on the CLEAN cloud
        tgt = gather_pts(mv["pc"], nnq)[:, :, 0] - q
        Lm, lg = masked_loss(net, mv["pc_in"], mv["h"], q, tgt)
        L = L + W_MASKED * Lm
        logs.update(lg)
        logs[f"masked_s{scale_frac:g}"] = lg["masked"]

    if "C" in tasks:
        pA, pB = V["a"], V["b"]
        ja, ok, dist, n_dup = mutual_nn(pA, pB, NN_MAX, DUP_EPS, V["a_n"], V["b_n"], GEO_NDOT)
        npair = min(NPAIR, nsub)
        rank = torch.where(ok, torch.rand(B, nsub, device=dev, generator=gen),
                           torch.full((B, nsub), 2.0, device=dev))
        ia = rank.argsort(-1)[:, :npair]
        pairs = dict(ia=ia, jb=ja.gather(1, ia), valid=ok.gather(1, ia))
        hA = net.encode(jitter(pA, cfg["aug_jit"], gen), V["a_n"])
        hB = net.encode(jitter(pB, cfg["aug_jit"], gen), V["b_n"])
        Lc, lg = consist_loss(net, hA, hB, pA, pB, pairs)
        L = L + W_CONSIST * Lc
        logs.update(lg)
        logs["consist_match_frac"] = float(ok.float().mean())
        logs["consist_match_mm"] = float((dist * ok).sum() / ok.float().sum().clamp(min=1))
        logs["consist_dup_dropped"] = float(n_dup) / B

    if "N" in tasks:
        if not NORMAL_MASK:                       # denoising mode: no hole, heavier jitter
            nv = dict(pc=V["n"], nr=V["n_n"], m_idx=None)
            nv["pc_in"] = jitter(nv["pc"], NORM_JIT, gen)
            nv["h"] = net.encode(nv["pc_in"], nv["nr"])
            npool = nv["pc"].shape[1]
            pick = torch.rand(B, npool, device=dev, generator=gen).argsort(-1)[:, :min(NQ, npool)]
            mode = "denoise_visible"
        else:                                     # extrapolate into a hole it cannot see
            nv = mv if (SHARE_VIEW and "M" in tasks) else \
                masked_encode(net, V["n"], V["n_n"], scale_frac, gen, cfg)
            m_idx = nv["m_idx"]
            nq = min(NQ, m_idx.shape[1])
            pick = m_idx[:, torch.randperm(m_idx.shape[1], device=dev, generator=gen)[:nq]]
            mode = "extrapolate_hidden" + ("_shared" if nv is mv else "")
        cov, n_un, _ = local_cov(nv["pc"], KNRM, NRM_R)           # targets: CLEAN cloud
        p3 = pick[..., None].expand(-1, -1, 3)
        q = nv["pc"].gather(1, p3)
        cov_t = cov.gather(1, pick[..., None].expand(-1, -1, cov.shape[-1]))
        oriented = nv["nr"] is not None
        n_t = (nv["nr"] if oriented else n_un).gather(1, p3)
        Ln, lg = normal_loss(net, nv["pc_in"], nv["h"], q, n_t, cov_t, oriented)
        L = L + W_NORMAL * Ln
        logs.update(lg)
        logs[f"normal_s{scale_frac:g}"] = lg["normal"]
        logs["normal_mode_" + mode] = 1.0
    logs["total"] = float(L)
    return L, logs


# ============================================================ checkpoint contract
def task_code(masked=MASKED, consist=CONSIST, normal=NORMAL):
    code = "".join(c for c, on in zip("MCN", (masked, consist, normal)) if on)
    assert code, "at least one of MASKED/CONSIST/NORMAL must be enabled"
    return code


def pretrain_path(fold, encoder=ENCODER, tasks=None, seed=SEED, work=WORK):
    """the ONE naming rule. The fold is in the filename so a wrong-fold load is visible
    before the file is even opened."""
    return os.path.join(work, f"ssl_{encoder}_{tasks or task_code()}_f{int(fold)}_s{int(seed)}.pt")


def save_ckpt(path, net, fold, data, cfg, extra):
    payload = dict(kind="family_d_ssl_encoder", fold=int(fold), encoder=net.enc_name,
                   tasks="".join(net.tasks), cin=int(net.cin), width=int(net.width),
                   enc_width=int(net.C), gk=int(net.gk), scale=float(SCALE),
                   seed=int(SEED), data=str(data.path), npts=int(data.N),
                   sub_frac=float(cfg["sub_frac"]),
                   train_ears=[int(x) for x in data.ear],
                   val_ears_excluded=[int(x) for x in data.val_ear],
                   ne_file=int(data.ne_file), torch_version=str(torch.__version__),
                   created=time.strftime("%Y-%m-%dT%H:%M:%S"),
                   cfg={k: v for k, v in cfg.items()}, **extra,
                   enc_state={k: v.detach().cpu() for k, v in net.enc.state_dict().items()})
    torch.save(payload, path)
    return path


def check_ckpt_fold(path, fold, ne=None):
    """THE function a fine-tune must call. Raises unless the checkpoint provably belongs
    to `fold`: the label must match AND the stored training ears must be disjoint from the
    validation ears re-derived independently from the frozen rule."""
    assert os.path.exists(path), f"SSL checkpoint {path} does not exist"
    # weights_only=True: the payload is deliberately built from tensors and plain python
    # types only, so it loads under the safe unpickler (verified). A checkpoint that needs
    # the unsafe one is not one of ours.
    p = torch.load(path, map_location="cpu", weights_only=True)
    assert p.get("kind") == "family_d_ssl_encoder", f"{path} is not a Family D SSL checkpoint"
    assert int(p["fold"]) == int(fold), (
        f"FOLD MISMATCH: {os.path.basename(path)} was pretrained on fold {int(p['fold'])} "
        f"but is being loaded for fold {fold}. Its training ears are validation ears here "
        f"-- loading it would leak (constraint 2).")
    ne = int(ne or p["ne_file"])
    _, tr, va = fold_ears(ne, int(fold))
    got = set(int(x) for x in p["train_ears"])
    assert not (got & set(va.tolist())), (
        f"{path} claims fold {fold} but its train_ears intersect that fold's VALIDATION "
        f"ears in {len(got & set(va.tolist()))} places -- the label is wrong, not the rule.")
    assert got <= set(tr.tolist()), \
        f"{path} used {len(got - set(tr.tolist()))} ears outside fold {fold}'s training set"
    p["_subset_of_train"] = len(got) < len(tr)
    return p


def init_encoder(enc, payload, min_frac=0.9):
    """load pretrained weights into `enc` and PROVE something was loaded: silently loading
    nothing (renamed keys, an adapter prefix) is the failure mode this guards."""
    before = float(sum(p.detach().abs().sum() for p in enc.parameters()))
    r = enc.load_state_dict(payload["enc_state"], strict=False)
    own = set(enc.state_dict().keys())
    matched = own - set(r.missing_keys)
    frac = len(matched) / max(1, len(own))
    after = float(sum(p.detach().abs().sum() for p in enc.parameters()))
    assert frac >= min_frac, (
        f"only {len(matched)}/{len(own)} encoder tensors matched the checkpoint "
        f"({frac:.0%} < {min_frac:.0%}). missing={r.missing_keys[:5]} "
        f"unexpected={r.unexpected_keys[:5]}. The fine-tune would have trained from "
        f"scratch while reporting a pretrained run.")
    assert abs(after - before) > 0, "load_state_dict changed nothing"
    return dict(matched=len(matched), total=len(own), frac=round(frac, 4),
                missing=list(r.missing_keys), unexpected=list(r.unexpected_keys),
                param_absum_before=round(before, 3), param_absum_after=round(after, 3))


class FreezeSchedule:
    """freeze for the first `epochs` epochs. eval() as well as requires_grad=False, so a
    sibling encoder's norm running statistics also stop moving while frozen."""

    def __init__(self, module, epochs):
        self.m, self.epochs, self.state = module, int(epochs), None

    def apply(self, epoch):
        frozen = epoch < self.epochs
        if frozen != self.state:
            self.m.requires_grad_(not frozen)
            self.state = frozen
        if frozen:
            self.m.eval()
        return frozen


# ============================================================ fine-tuning entry point
class LandmarkHead(nn.Module):
    """one refinement pass: bounded offset, then a soft-argmax snap onto the input points.

    This is the shipped 1.273mm model's Head. It is RE-IMPLEMENTED rather than imported
    because gpu_screen.py loads its dataset at module scope and therefore cannot be
    imported at all (see CONTRACTS in the report: an importable dgcnn family module would
    remove this duplication).
    """

    def __init__(self, C, k=48, max_off=None):
        super().__init__()
        self.emb, self.embO = nn.Embedding(NL, 32), nn.Embedding(NL, 32)
        self.offset = nn.Sequential(nn.Linear(2 * C + 32, 256), nn.ReLU(), nn.Dropout(0.1),
                                    nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 3))
        self.attn = nn.Sequential(nn.Linear(C + 3 + 32, 128), nn.ReLU(),
                                  nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.C, self.k, self.max_off = C, k, max_off

    def forward(self, pc, h, q):
        B = pc.shape[0]
        ar = torch.arange(NL, device=pc.device)
        idx = knn(q, pc, self.k)
        ctx = gather_pts(h, idx)
        eo = self.embO(ar)[None].expand(B, -1, -1)
        off = self.offset(torch.cat([ctx.mean(2), ctx.max(2).values, eo], -1))
        if self.max_off:
            off = self.max_off * torch.tanh(off / self.max_off)
        q1 = q + off
        idx2 = knn(q1, pc, self.k)
        fK, pK = gather_pts(h, idx2), gather_pts(pc, idx2)
        rel = (pK - q1[:, :, None, :]) / SCALE
        e = self.emb(ar)[None, :, None, :].expand(B, NL, self.k, 32)
        w = torch.softmax(self.attn(torch.cat([fK, rel, e], -1)).squeeze(-1), -1)
        return q1, (w[..., None] * pK).sum(2)


class MODEL(nn.Module):
    """Family D fine-tune, as train_family.py resolves it via FAMILY_MODULE=pretrain_ssl.

    The encoder is the pretrained one (fold-checked at construction); the head is the
    shipped refinement head. The from-scratch control is the identical architecture with
    CFG_SSL_CKPT="" CFG_REQUIRE_CKPT=false -- the only comparison that isolates
    pretraining rather than the head re-implementation.
    """
    DEFAULTS = dict(encoder=ENCODER, width=WIDTH, gk=GK, k=48, npass=4, max_off=0.0,
                    ssl_ckpt="", freeze_epochs=int(os.environ.get("FREEZE_EPOCHS", "0")),
                    require_ckpt=True, use_nrm=False, contour=True)
    SEARCH_SPACE = dict(freeze_epochs=[0, 50, 200, 400], k=[32, 48, 96], npass=[2, 4, 6],
                        width=[128, 256], max_off=[0.0, 3.0, 7.0], lr=[3e-4, 1.5e-3])
    NEEDS = ("nrm",) if _flag("CFG_USE_NRM", "0") else ()
    ROTATES = ("nrm",)
    SAMPLES = 1

    def __init__(self, cfg, meta):
        super().__init__()
        self.use_nrm = bool(cfg["use_nrm"])
        cin = 3 + 3 * self.use_nrm
        self.enc, C = make_encoder(cfg["encoder"], cin=cin, C=int(cfg["width"]),
                                   gk=int(cfg["gk"]), npts=min(meta["npts"], 512))
        self.npass = int(cfg["npass"])
        self.head = LandmarkHead(C, int(cfg["k"]), float(cfg["max_off"]) or None)
        self.contour_nets = nn.ModuleList([
            nn.Sequential(nn.Conv1d(3 + 64 + 32, 96, 5, padding=2), nn.ReLU(),
                          nn.Conv1d(96, 96, 3, padding=1), nn.ReLU(), nn.Conv1d(96, 3, 1))
            for _ in CONTOURS]) if cfg["contour"] else None
        self.lmfeat = nn.Sequential(nn.Linear(C, 64), nn.ReLU())
        self.embC = nn.Embedding(NL, 32)
        self.C = C
        self.ckpt_report = None
        path = str(cfg["ssl_ckpt"] or "")
        if path:
            p = check_ckpt_fold(path, meta["fold"])
            # check_ckpt_fold with ne=None re-derives the frozen split from the CHECKPOINT's
            # OWN `ne_file`, so it is self-consistent and therefore VACUOUS whenever the
            # pretraining npz had a different ear count from the one being fine-tuned. That
            # is not hypothetical: pretraining on a 100-ear file and fine-tuning on 340
            # passes with 14 of the checkpoint's TRAINING ears being validation ears here
            # (measured). meta exposes the trainer's own train-ear count, which pins the
            # split down; assert the checkpoint's ear count induces exactly that split.
            _, tr_ck, _ = fold_ears(int(p["ne_file"]), int(meta["fold"]))
            assert len(tr_ck) == int(meta["n_train_ears"]), (
                f"EAR-COUNT MISMATCH: {os.path.basename(path)} was pretrained on a "
                f"{p['ne_file']}-ear file, whose fold {meta['fold']} has {len(tr_ck)} "
                f"training ears, but this fine-tune has {meta['n_train_ears']}. The fold "
                f"check re-derived the WRONG split, so it proved nothing -- the checkpoint's "
                f"training ears may be validation ears here (constraint 2). Pretrain on the "
                f"same ear set you fine-tune on.")
            assert p["encoder"] == cfg["encoder"] and int(p["width"]) == int(cfg["width"]) \
                and int(p["cin"]) == cin, (
                f"checkpoint is encoder={p['encoder']} width={p['width']} cin={p['cin']}, "
                f"the fine-tune builds {cfg['encoder']}/{cfg['width']}/{cin}")
            self.ckpt_report = dict(path=path, fold=int(p["fold"]), tasks=p["tasks"],
                                    subset_of_train=bool(p["_subset_of_train"]),
                                    **init_encoder(self.enc, p))
            print(f"  SSL init: {os.path.basename(path)} fold {p['fold']} tasks "
                  f"{p['tasks']} -> {self.ckpt_report['matched']}/"
                  f"{self.ckpt_report['total']} encoder tensors", flush=True)
        else:
            assert not cfg["require_ckpt"], (
                "CFG_SSL_CKPT is empty. Set it, or set CFG_REQUIRE_CKPT=false to run the "
                "explicit from-scratch CONTROL arm (which is a different experiment).")
            print("  SSL init: NONE (from-scratch control arm)", flush=True)
        self.freezer = FreezeSchedule(self.enc, int(cfg["freeze_epochs"]))
        self.register_buffer("_epoch", torch.zeros((), dtype=torch.long), persistent=False)
        self.freezer.apply(0)

    def set_epoch(self, ep):
        """explicit epoch hook -- takes precedence over the train() counter"""
        self._epoch.fill_(int(ep))
        return self.freezer.apply(int(ep))

    def train(self, mode=True):
        super().train(mode)
        if mode:                     # train_family.py calls this exactly once per epoch
            self.freezer.apply(int(self._epoch))
            self._epoch += 1
        return self

    def forward(self, b):
        pc = b["pc"]
        h = self.enc(pc, b["nrm"] if self.use_nrm else None)
        aux, q = [], b["coarse"]
        for _ in range(self.npass):
            q1, q2 = self.head(pc, h, q)
            aux += [q1, q2]
            q = q2
        if self.contour_nets is not None:
            f = self.lmfeat(gather_pts(h, knn(q, pc, 1))[:, :, 0])
            e = self.embC(torch.arange(NL, device=pc.device))[None].expand(pc.shape[0], -1, -1)
            inp = torch.cat([q / SCALE, f, e], -1)
            out = torch.zeros_like(q)
            for (lo, hi), net in zip(CONTOURS, self.contour_nets):
                out[:, lo:hi + 1] = net(inp[:, lo:hi + 1].transpose(1, 2)).transpose(1, 2)
            q = q + out
        return dict(pred=q, aux=aux)


# ============================================================ pretraining driver
def build_cfg():
    return dict(sub_frac=SUB_FRAC, aug_rot=AUG_ROT, aug_scale=AUG_SCALE, aug_jit=AUG_JIT,
                mask_scales=MASK_SCALES, geo_k=GEO_K, geo_fac=GEO_FAC, geo_maxe=GEO_MAXE,
                geo_ndot=GEO_NDOT, geo_iters=GEO_ITERS, geo_retry=GEO_RETRY, nq=NQ, qmin=QMIN, qoff=QOFF,
                mask_k=MASK_K, npair=NPAIR, tau=TAU, excl=EXCL, nn_max=NN_MAX,
                proj=PROJ, vc=VC, knrm=KNRM, nrm_r=NRM_R, norm_jit=NORM_JIT,
                normal_mask=NORMAL_MASK, share_view=SHARE_VIEW, lr=LR, wd=WD, bs=BS,
                w_masked=W_MASKED, w_consist=W_CONSIST, w_normal=W_NORMAL,
                encoder=ENCODER, width=WIDTH, gk=GK, scale=SCALE, epochs=EPOCHS, seed=SEED)


def mean_logs(rows):
    out = {}
    for k in rows[0]:
        v = [r[k] for r in rows if k in r and isinstance(r[k], (int, float))]
        if v:
            out[k] = float(np.mean(v))
    return out


def main():
    t0 = time.time()
    assert "FOLD" in os.environ, (
        "FOLD is REQUIRED. Fold-clean pretraining has no meaningful default: a wrong fold "
        "pretrains on ears that are validation ears downstream (constraint 2).")
    fold = int(os.environ["FOLD"])
    assert 0 <= fold < NFOLD, f"FOLD must be 0..{NFOLD-1}"
    assert TAU > 0 or VC > 0, (
        "TAU<=0 selects the MSE consistency objective, whose global optimum is a CONSTANT "
        "embedding. Set VC>0 (variance guard) or keep TAU>0 (InfoNCE).")
    tasks = tuple(task_code())
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); np.random.seed(SEED)
    sampler = np.random.RandomState(1_000_003 * SEED + 17)
    gen = torch.Generator(device=dev); gen.manual_seed(SEED * 7919 + 13)
    cfg = build_cfg()

    data = FoldClean(DATA, fold, dev, want_nrm=True)
    print(data.report(), flush=True)
    if "C" in tasks:
        assert data.M >= 2, "CONSIST needs M>=2 independent samples per ear"
    fit, mon = data.inner_split(MON_FRAC, np.random.RandomState(4242 + SEED))
    print(f"   SSL inner split (train-fold only): {len(fit)} fit / {len(mon)} monitor ears, "
          f"subject-grouped: PASS", flush=True)

    # cin=3: SSL pretrains the XYZ-only encoder. `nrm` is still used -- for the graph
    # normal gate, the correspondence gate and the NORMAL target -- but never as an input
    # channel, so a normals-input encoder is a separate experiment. CFG_USE_NRM=true at
    # fine-tune time then fails the cin assertion in MODEL rather than loading a mismatch.
    net = SSLNet(ENCODER, 3, WIDTH, GK, int(round(data.N * SUB_FRAC)), tasks).to(dev)
    npar = sum(p.numel() for p in net.parameters())
    nenc = sum(p.numel() for p in net.enc.parameters())
    print(f"[ssl f{fold} {ENCODER} tasks={''.join(tasks)}] {npar:,} params "
          f"({nenc:,} encoder / {npar-nenc:,} discarded heads) | per-point width {net.C} | "
          f"{data.N} pts x {data.M} samples | {dev}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    every = int(os.environ.get("EVAL_EVERY", str(max(1, EPOCHS // 12))))

    def monitor():
        net.eval()
        rows = []
        with torch.no_grad():
            for b in range(0, len(mon), BS):
                idx = mon[b:b + BS]
                if len(idx) < 2:
                    continue
                rows.append(ssl_step(net, data, idx, gen, sampler, cfg, tasks)[1])
        return mean_logs(rows) if rows else {}

    curve, best = [], (9e9, None, -1)
    for ep in range(EPOCHS):
        net.train()
        rows = []
        perm = sampler.permutation(fit)
        for b in range(0, len(perm), BS):
            idx = perm[b:b + BS]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            L, lg = ssl_step(net, data, idx, gen, sampler, cfg, tasks)
            L.backward(); opt.step()
            rows.append(lg)
        sch.step()
        if (ep + 1) % every == 0 or ep + 1 == EPOCHS:
            tr_l, mo_l = mean_logs(rows), monitor()
            curve.append({"epoch": ep + 1, "train": {k: round(v, 5) for k, v in tr_l.items()},
                          "monitor": {k: round(v, 5) for k, v in mo_l.items()}})
            # every number after `mon` is a DEGENERACY RATIO: <1 beats the constant /
            # collapsed predictor, >=1 does not. That is the only reading that matters.
            msg = f"  ep{ep+1:4d} train {tr_l['total']:.4f} mon {mo_l.get('total', float('nan')):.4f}"
            if "M" in tasks:
                msg += f" | masked {mo_l['masked']:.3f} ({mo_l['masked_rms_mm']:.2f}mm)"
            if "C" in tasks:
                msg += f" | consist {mo_l['consist']:.3f} ({mo_l['consist_raw']:.2f}/" \
                       f"{mo_l['consist_collapse']:.2f} nats)"
            if "N" in tasks:
                msg += f" | nrm {mo_l['normal_nrm']:.3f} cov {mo_l['normal_cov']:.3f}"
            print(msg + f" ({time.time()-t0:.0f}s)", flush=True)
            if mo_l.get("total", 9e9) < best[0]:
                best = (mo_l["total"], {k: v.detach().cpu().clone()
                                        for k, v in net.state_dict().items()}, ep + 1)

    if best[1] is not None:
        net.load_state_dict(best[1])
    path = pretrain_path(fold, ENCODER, "".join(tasks), SEED, WORK)
    save_ckpt(path, net, fold, data, cfg,
              dict(monitor_total=float(best[0]), best_epoch=int(best[2]),
                   n_params_encoder=int(nenc)))
    rep = dict(kind="family_d_ssl", fold=fold, encoder=ENCODER, tasks="".join(tasks),
               seed=SEED, epochs=EPOCHS, params=int(npar), params_encoder=int(nenc),
               enc_width=int(net.C), data=DATA, disjoint_views=data.disjoint,
               n_train_ears=int(data.E), n_fit=int(len(fit)), n_monitor=int(len(mon)),
               val_ears_excluded=int(len(data.val_ear)), cfg=cfg,
               best_monitor_total=float(best[0]), best_epoch=int(best[2]),
               ckpt=path, runtime_s=round(time.time() - t0, 1), curve=curve)
    jpath = path.replace(".pt", ".json")
    json.dump(rep, open(jpath, "w"), indent=1)
    print(f"\nsaved {path}\n      {jpath}  ({time.time()-t0:.0f}s, best monitor "
          f"{best[0]:.4f} @ep{best[2]})", flush=True)
    print(f"CHECK: check_ckpt_fold({os.path.basename(path)}, {fold}) -> "
          f"{'PASS' if check_ckpt_fold(path, fold) else ''}", flush=True)


# ============================================================ smoke test (CPU, <90s)
def synth_ear(rs, npts, gap=4.0, r=7.0, half_w=10.0):
    """A folded 'horseshoe' sheet with radial bumps: the two arms come within `gap` mm of
    each other in SPACE while being ~40mm apart ALONG the surface. Analytic oriented
    normals. This is the property that separates a geodesic ball from a Euclidean one."""
    tmax = np.pi - np.arcsin(gap / (2 * r))
    th = rs.uniform(-tmax, tmax, npts)
    v = rs.uniform(-half_w, half_w, npts)
    A, f1, f2 = 0.9 + 0.4 * rs.rand(), 1.5 + rs.rand(), 1.0 + rs.rand()
    a = A * np.sin(f1 * th) * np.cos(f2 * v / 10)
    at = A * f1 * np.cos(f1 * th) * np.cos(f2 * v / 10)
    av = -A * f2 / 10 * np.sin(f1 * th) * np.sin(f2 * v / 10)
    ct, st = np.cos(th), np.sin(th)
    P = np.stack([(r + a) * ct, v, (r + a) * st], -1)
    Pt = np.stack([at * ct - (r + a) * st, np.zeros(npts), at * st + (r + a) * ct], -1)
    Pv = np.stack([av * ct, np.ones(npts), av * st], -1)
    N = np.cross(Pt, Pv)
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    N *= np.sign((N * np.stack([ct, np.zeros(npts), st], -1)).sum(1))[:, None]   # outward
    return P.astype(np.float32), N.astype(np.float32), th


def smoke():
    import tempfile
    t0 = time.time()
    dev = "cpu"
    torch.manual_seed(0); np.random.seed(0)
    rs = np.random.RandomState(0)
    NE, M, NP = 8, 4, 768
    print(f"== Family D smoke test (CPU) ==  torch {torch.__version__}")

    # ---------- 1. a data npz with the real key names, INCLUDING GT that must not be read
    cl = np.zeros((NE, M, NP, 3), np.float32); nr = np.zeros_like(cl)
    for i in range(NE):
        for j in range(M):
            cl[i, j], nr[i, j], _ = synth_ear(np.random.RandomState(100 * i + j), NP)
    tmp = os.path.join(tempfile.gettempdir(), "pretrain_ssl_smoke")
    os.makedirs(tmp, exist_ok=True)
    dpath = os.path.join(tmp, "ssl_smoke_data.npz")
    np.savez(dpath, clouds=cl, nrm=nr, coarse=rs.randn(NE, NL, 3).astype(np.float32),
             true=rs.randn(NE, NL, 3).astype(np.float32),
             R=np.tile(np.eye(3, dtype=np.float32), (NE, 1, 1)),
             c0=np.zeros((NE, 3), np.float32))

    # ---------- 2. fold-clean loading
    data = FoldClean(dpath, 0, dev)
    print(data.report())
    _, tr0, va0 = fold_ears(NE, 0)
    assert list(data.ear) == list(tr0) and data.E == len(tr0)
    assert _storage_exact(data.clouds) and "true" in data.skipped_gt
    d1 = FoldClean(dpath, 1, dev)
    assert not (set(data.ear.tolist()) & set(d1.val_ear.tolist()) & set(va0.tolist())), "sanity"
    assert set(data.ear.tolist()) != set(d1.ear.tolist()), "folds must differ"
    fit, mon = data.inner_split(0.3, np.random.RandomState(1))
    print(f"   inner split {len(fit)} fit / {len(mon)} monitor (subject-grouped)")

    # ---------- 3. geodesic vs Euclidean, on the folded sheet
    gen = torch.Generator(device=dev); gen.manual_seed(7)
    pc = data.clouds[:2, 0]; nrmv = data.nrm[:2, 0]
    th = synth_ear(np.random.RandomState(100 * int(data.ear[0])), NP)[2]
    seed_i = torch.tensor([int(np.argmax(th)), int(np.argmax(th))])       # near one arm tip
    gidx, gw, gdeg, gstat = knn_graph(pc, GEO_K, GEO_FAC, GEO_MAXE, nrmv, GEO_NDOT)
    g = geodesic(gidx, gw, seed_i, 120)
    nm = 96
    gmask = g.argsort(-1)[:, :nm]
    emask = euclid_frontier(pc, seed_i, nm)
    inter = len(set(gmask[0].tolist()) & set(emask[0].tolist())) / nm
    gsp = float(g.gather(1, gmask[:, -1:]).mean())
    esp = float((pc - pc[:, seed_i[0]][:, None]).norm(dim=-1).gather(1, emask[:, -1:]).mean())
    print(f"   geodesic ball {nm} pts: radius {gsp:.2f}mm (geodesic) vs {esp:.2f}mm "
          f"(Euclidean) | overlap {inter:.0%} | graph limit {gstat['geo_maxe_mm']:.2f}mm "
          f"(nn1 {gstat['nn1_mm']:.2f}mm, k-th {gstat['knn_r_mm']:.2f}mm), edges capped "
          f"{gstat['edges_capped']}/{gstat['edge_total']}, normal-gated "
          f"{gstat['edges_normal_gated']}, min degree {int(gdeg.min())}")
    assert inter < 0.9, "geodesic and Euclidean masks are indistinguishable on a folded sheet"
    assert gsp < esp * 6 and gsp > 0
    cross = set(emask[0].tolist()) - set(gmask[0].tolist())
    assert len(cross) > 0, "the Euclidean ball must reach the facing arm; it did not"
    print(f"   -> {len(cross)}/{nm} Euclidean-ball points are NOT in the geodesic ball "
          f"(they sit on the facing arm, {gap_note()})")

    # ---------- 4. mutual-NN matching statistics + the duplicate guard
    pa, pb = data.clouds[:2, 0], data.clouds[:2, 1]
    ja, ok, dist, ndup = mutual_nn(pa, pb, NN_MAX, DUP_EPS, data.nrm[:2, 0], data.nrm[:2, 1], 0.0)
    print(f"   mutual-NN A<->B: retained {float(ok.float().mean()):.1%}, matched distance "
          f"{float((dist*ok).sum()/ok.float().sum()):.3f}mm, exact duplicates dropped {ndup}")
    pb2 = pb.clone(); pb2[:, :20] = pa[:, :20]                    # inject coincident points
    _, ok2, _, ndup2 = mutual_nn(pa, pb2, NN_MAX, DUP_EPS)
    assert ndup2 >= 20 and not ok2[:, :20].any(), "coincident pairs were not dropped"
    print(f"   duplicate guard: injected 20/ear coincident points -> {ndup2} dropped, "
          f"none retained as positives")

    # ---------- 5. every task subset: forward AND backward
    cfg = build_cfg(); cfg["sub_frac"] = 0.625
    sampler = np.random.RandomState(3)
    ears = fit[:4] if len(fit) >= 4 else data.ear[:4]
    mags = {}
    for tset in (("M",), ("C",), ("N",), ("M", "C", "N")):
        net = SSLNet("dgcnn", 3, 96, 12, int(NP * cfg["sub_frac"]), tset).to(dev)
        npar = sum(p.numel() for p in net.parameters())
        net.train()
        L, lg = ssl_step(net, data, ears, gen, sampler, cfg, tset)
        L.backward()
        gn = sum(float(p.grad.norm()) for p in net.parameters() if p.grad is not None)
        genc = sum(float(p.grad.norm()) for p in net.enc.parameters() if p.grad is not None)
        assert torch.isfinite(L) and gn > 0 and genc > 0, "no gradient reached the encoder"
        keys = {k: round(v, 4) for k, v in lg.items() if isinstance(v, float)}
        print(f"\n   tasks={''.join(tset)} params {npar:,} loss {float(L):.4f} "
              f"grad-norm {gn:.3g} (encoder {genc:.3g})")
        for k in sorted(keys):
            if k.startswith(("masked", "consist", "normal", "geo_", "mask_", "total")):
                print(f"      {k:24s} {keys[k]}")
        if len(tset) == 1:
            mags[tset[0]] = keys
    # degeneracy references, printed as ratios so a null task is visible
    if "M" in mags:
        assert mags["M"]["masked_const"] > 0
        print(f"\n   UNTRAINED degeneracy ratios (a fresh net sits AT or above its "
              f"degenerate optimum -- that is the point of printing them): masked "
              f"{mags['M']['masked']:.3f}, consist {mags['C']['consist']:.3f}, normal "
              f"{mags['N']['normal_nrm']:.3f} (nrm) / {mags['N']['normal_cov']:.3f} (cov). "
              f"1.0 = no better than a constant.")

    # ---------- 5b. does the objective ESCAPE its degenerate optimum? (not just run)
    # CONSIST is checked by default because its degenerate solution is the dangerous one (a
    # constant embedding) and its reference value, log(n_candidates), is exact. MASKED and
    # NORMAL need ~300 steps to separate from their constant baselines, which does not fit
    # a 90s CPU budget -- LEARN=1 runs them and prints the numbers quoted in the report.
    lc = int(os.environ.get("LEARN_C", "40"))
    net = SSLNet("dgcnn", 3, 96, 12, int(NP * cfg["sub_frac"]), ("C",))
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3)
    g2 = torch.Generator(device=dev); g2.manual_seed(11); s2 = np.random.RandomState(5)
    hist = []
    for _ in range(lc):
        opt.zero_grad()
        L, lg = ssl_step(net, data, ears, g2, s2, cfg, ("C",))
        L.backward(); opt.step(); hist.append(lg)
    a, b = mean_logs(hist[:8]), mean_logs(hist[-8:])
    print(f"\n   CONSIST escapes collapse over {lc} steps: ratio-to-collapse "
          f"{a['consist']:.3f} -> {b['consist']:.3f} ({b['consist_raw']:.2f} of "
          f"{b['consist_collapse']:.2f} nats, {b['consist_pairs']:.0f} pairs/ear)")
    assert b["consist"] < 0.75, (
        "CONSIST did not separate from the constant-embedding optimum: it is either "
        "collapsing or the correspondence is wrong")
    if _flag("LEARN", "0"):
        for tset, key, ref in ((("M",), "masked", "masked_const"),
                               (("N",), "normal_nrm", "normal_nrm_const")):
            n2 = SSLNet("dgcnn", 3, 96, 12, int(NP * cfg["sub_frac"]), tset)
            o2 = torch.optim.AdamW(n2.parameters(), lr=3e-3)
            h2 = []
            for _ in range(int(os.environ.get("LEARN_N", "300"))):
                o2.zero_grad()
                L, lg = ssl_step(n2, data, ears, g2, s2, cfg, tset)
                L.backward(); o2.step(); h2.append(lg)
            p, q = mean_logs(h2[:20]), mean_logs(h2[-20:])
            print(f"   {''.join(tset)}: {key} ratio-to-const {p[key]:.3f} -> {q[key]:.3f} "
                  f"(raw {q[key.replace('normal_nrm', 'normal_nrm_raw').replace('masked', 'masked_raw')]:.4f}"
                  f" vs const {q[ref]:.4f})")
            assert q[key] < 0.95, f"{key} never beat its constant baseline"

    # ---------- 6. checkpoint: save, fold-check, wrong-fold refusal
    net = SSLNet("dgcnn", 3, 96, 12, int(NP * cfg["sub_frac"]), ("M", "C", "N")).to(dev)
    cpath = save_ckpt(pretrain_path(0, "dgcnn", "MCN", 0, tmp), net, 0, data, cfg,
                      dict(monitor_total=0.0, best_epoch=0, n_params_encoder=0))
    p = check_ckpt_fold(cpath, 0, NE)
    print(f"\n   checkpoint {os.path.basename(cpath)}: fold {p['fold']} tasks {p['tasks']} "
          f"train_ears {len(p['train_ears'])} val_excluded {len(p['val_ears_excluded'])} "
          f"-> check_ckpt_fold(fold=0) PASS")
    for wrong in (1, 2):
        try:
            check_ckpt_fold(cpath, wrong, NE)
            raise SystemExit(f"FAIL: a fold-{wrong} load of a fold-0 checkpoint was allowed")
        except AssertionError as e:
            assert "FOLD MISMATCH" in str(e)
    print("   wrong-fold load refused for folds 1 and 2: PASS")
    bad = dict(torch.load(cpath, map_location="cpu", weights_only=True))
    bad["fold"] = 1                                  # relabelled, but the ears betray it
    bpath = os.path.join(tmp, "ssl_dgcnn_MCN_f1_s0.pt"); torch.save(bad, bpath)
    try:
        check_ckpt_fold(bpath, 1, NE)
        raise SystemExit("FAIL: a RELABELLED checkpoint passed the fold check")
    except AssertionError as e:
        assert "VALIDATION" in str(e) or "outside fold" in str(e)
    print("   relabelled checkpoint (fold field edited to 1) refused by the ear-set "
          "re-derivation: PASS")

    # ---------- 7. fine-tune entry point: (2,85,3), backward, freeze schedule
    meta = dict(nl=NL, contours=CONTOURS, scale=SCALE, npts=NP, fold=0, dev=dev,
                n_train_ears=len(data.ear), artefacts={})
    fcfg = dict(MODEL.DEFAULTS, encoder="dgcnn", width=96, gk=12, k=24, npass=2,
                ssl_ckpt=cpath, freeze_epochs=2, contour=True)
    m = MODEL(fcfg, meta)
    nptot = sum(p.numel() for p in m.parameters())
    b = {"pc": data.clouds[:2, 0], "coarse": torch.randn(2, NL, 3) * 6,
         "ear": torch.tensor([0, 1])}
    m.train()                                          # epoch 0 -> frozen
    out = m(b)
    assert out["pred"].shape == (2, NL, 3), out["pred"].shape
    loss = ((out["pred"] - torch.randn(2, NL, 3)) ** 2).sum(-1).mean() + \
        sum(((a - torch.zeros(2, NL, 3)) ** 2).sum(-1).mean() for a in out["aux"]) * 0.0
    loss.backward()
    ge = [p.grad for p in m.enc.parameters() if p.grad is not None]
    print(f"\n   MODEL(fine-tune) {nptot:,} params | pred {tuple(out['pred'].shape)} "
          f"aux {len(out['aux'])} | ckpt {m.ckpt_report['matched']}/"
          f"{m.ckpt_report['total']} encoder tensors matched "
          f"({m.ckpt_report['frac']:.0%})")
    assert out["pred"].shape == (2, NL, 3)
    assert len(ge) == 0, "FREEZE_EPOCHS=2 but the encoder received gradients at epoch 0"
    frozen_hist = []
    for ep in range(4):
        m.zero_grad(set_to_none=True)
        m.train()
        o = m(b)
        (((o["pred"]) ** 2).sum(-1).mean()).backward()
        got = any(p.grad is not None for p in m.enc.parameters())
        frozen_hist.append(not got)
    head_g = sum(float(p.grad.norm()) for p in m.head.parameters() if p.grad is not None)
    print(f"   freeze schedule (FREEZE_EPOCHS=2), encoder frozen per epoch 1..4: "
          f"{frozen_hist} | head grad-norm {head_g:.3f} throughout")
    assert frozen_hist == [True, False, False, False], frozen_hist
    assert head_g > 0, "the head must train while the encoder is frozen"
    m.set_epoch(0)
    assert m.freezer.apply(0) and not m.freezer.apply(9), "set_epoch must drive the freeze"
    print("   set_epoch() overrides the train() counter: PASS")

    # ---------- 8. control arm + require_ckpt guard
    try:
        MODEL(dict(fcfg, ssl_ckpt=""), meta)
        raise SystemExit("FAIL: an empty CFG_SSL_CKPT was accepted with require_ckpt=True")
    except AssertionError as e:
        assert "CONTROL" in str(e)
    mc = MODEL(dict(fcfg, ssl_ckpt="", require_ckpt=False), meta)
    assert mc(b)["pred"].shape == (2, NL, 3) and mc.ckpt_report is None
    print("   from-scratch control arm builds only with CFG_REQUIRE_CKPT=false: PASS")

    # ---------- 9. a checkpoint pretrained on a DIFFERENT-SIZED ear set must be refused.
    # check_ckpt_fold(path, fold) alone re-derives the split from the checkpoint's own
    # ne_file and so cannot see this: with ne_file=100 against a 340-ear fine-tune, 14 of
    # the checkpoint's TRAINING ears are validation ears of the fold it is loaded for, and
    # the fold assertion still passes. MODEL cross-checks against the trainer's train-ear
    # count, which is the only thing that pins the split down.
    _, tr100, _ = fold_ears(100, 0)
    _, _, va340 = fold_ears(340, 0)
    leak = sorted(set(tr100.tolist()) & set(va340.tolist()))
    xp = dict(torch.load(cpath, map_location="cpu", weights_only=True))
    xp["ne_file"], xp["train_ears"] = 100, [int(x) for x in tr100]
    xpath = os.path.join(tmp, "ssl_dgcnn_MCN_f0_s0_ne100.pt"); torch.save(xp, xpath)
    check_ckpt_fold(xpath, 0)                       # passes: self-consistent, hence vacuous
    m340 = dict(meta, n_train_ears=272)             # a 340-ear fine-tune of fold 0
    try:
        MODEL(dict(fcfg, ssl_ckpt=xpath), m340)
        raise SystemExit("FAIL: a checkpoint from a 100-ear file was loaded into a 340-ear "
                         f"fold-0 fine-tune, leaking {len(leak)} validation ears")
    except AssertionError as e:
        assert "EAR-COUNT MISMATCH" in str(e), str(e)
    print(f"   ne_file=100 checkpoint passes check_ckpt_fold's own re-derivation but is "
          f"REFUSED by MODEL for a 340-ear fold-0 fine-tune\n      (it would have leaked "
          f"{len(leak)} validation ears, e.g. {leak[:6]}): PASS")
    print(f"\n== smoke OK in {time.time()-t0:.1f}s ==")


def gap_note():
    return "the 4.0mm air gap exceeds the graph edge limit, so it cannot be bridged"


if __name__ == "__main__":
    main() if "FOLD" in os.environ else smoke()
