"""
ENDPOINT SPECIALIST -- high-resolution local refinement of the 8 CONTOUR ENDPOINTS.

WHY ENDPOINTS. The normalised cumulative arc-length profile of the GT landmarks is
nearly constant across subjects (sd 0.0063 on inner_helix, 0.0073 on sup._antihelix ->
0.33mm and 0.14mm against those contours' lengths), so for 30 of the 85 landmarks PHASE
IS DETERMINED by the curve. An oracle given ONLY the two endpoints of a contour, placing
the interior equidistantly along the rigidly-repositioned predicted curve, takes
sup._antihelix 1.2051 -> 0.3336 and inner_helix 1.4859 -> 1.0724. The endpoints GATE that
constraint, and they are measurably no better localised than the average landmark. So a
specialist that only has to solve 8 localisation problems, at a resolution the 2048-point
backbones never see, is worth about -0.20mm overall if it reaches that oracle.

MEASURED, on the 340-ear pooled OOF ensemble (dgcnn3+kpconv+ptv3, equal weight, BEFORE
surface projection; pooled 1.1952mm, non-endpoint landmarks 1.1821mm):

    landmark        mean    med     p90     p95     p99     max
    0  outer_hi_a  1.3007  1.0818  2.2091  2.7716  4.8624 10.9182
    24 outer_hi_b  1.1994  0.9961  2.2673  2.6714  4.4357  5.5444
    25 concha_a    0.9785  0.8234  1.6121  2.1084  3.0745 12.2398
    54 concha_b    0.7546  0.6381  1.3386  1.5874  2.2159  3.2971
    55 inner_hi_a  1.0981  0.9280  2.0580  2.4964  3.3921  6.6223
    74 inner_hi_b  2.5329  2.0312  5.1725  6.6457  7.8313 10.5598
    75 sup_anti_a  1.3555  1.1313  2.4470  3.1564  4.1426  7.1258
    84 sup_anti_b  1.3539  1.0311  2.7494  3.2257  4.4295  5.9064
    ALL 8         1.3217  0.9969  2.5412  3.4541  5.9923 12.2398

CAVEAT ON THE BRIEF'S NUMBERS: the brief quotes 1.35-1.82mm for the endpoints. The table
above is what this repo's artefacts actually give (0.75-2.53mm, mean 1.3217). It is the
same ordering of magnitude and the same conclusion -- endpoints are worse than average
and lm 74 is catastrophic -- but the per-landmark values differ, so the brief's figures
were computed on something else (a different ensemble, or after projection). Everything
sized below is sized against the table above, which reproduces from
`python research/code/fam_endpoint.py` (the analysis block prints it).

CROP RADIUS = 7.0 mm, JUSTIFIED. The crop must contain the true endpoint. Measured
P(|GT - estimate| < R) over 340x8:

    R      3      4      5      6      7      8     10     12   mm
    all  92.94  96.32  97.98  98.97  99.52  99.82  99.89  99.96  %
    lm74 67.1   79.4   87.9   93.5   97.1   99.1   99.7   99.7   %

7.0mm is the smallest radius that keeps the WORST landmark above 97%. Going to 8mm buys
0.3 points overall at 1.4x the points. Going to 6mm loses 3.6 points on lm 74, the one
landmark that most needs the reach.

GATHER CAP = 448, JUSTIFIED, AND THE TRUNCATION IS REPORTED. Points of the 8192-point
cloud (spacing 1.09mm) inside a 7mm ball around the estimate: min 108, p05 148, median
203, p95 312, max 431. KMAX=448 therefore NEVER truncates -- at evaluation (full cloud)
or during training (sub_frac=0.625, so 0.625x those counts). That matters because the
trainer evaluates on the full cloud and trains on 62.5% of it: with KMAX=256 the ball is
truncated on 14.4% of EVAL crops and 0.2% of TRAIN crops, i.e. the physical support
silently shrinks at evaluation only. `nt` in the forward output is the UNCAPPED ball size
and the audit (min/median/max/frac_truncated/frac_empty) is printed once per DISTINCT
cloud size -- training's sub_frac*N AND evaluation's full N, each labelled -- so a radius
or point-count change cannot start truncating unnoticed. Keying it on the point count is
load-bearing, not cosmetic: the eval crop is the denser one and truncates FIRST, so an
audit that only reported the first few forwards would report training twice and never
show the failure. At the smoke's deliberately undersized cap the two lines read
`truncated 0.3906` (train, 640 pts) against `truncated 0.8750` (EVAL, 1024 pts).
    For EP_SET=all the same cap over 85 landmarks is 10x the memory; use
    CFG_KMAX=192 CFG_CROP_R=5.5 there and read the printed truncation line.

HEATMAP SOFT-ARGMAX, NOT FREE REGRESSION -- and why. The GT landmarks lie exactly on the
mesh, and the distance from a GT endpoint to the nearest 8192-cloud point is mean 0.5387,
median 0.5062, p90 0.9208, max 2.0404 mm. So a HARD argmax over the crop has a 0.54mm
floor against the 1.32mm the endpoints currently carry, and a SOFT argmax -- a convex
combination of crop points -- interpolates below it while staying inside the hull of a
locally planar surface patch, i.e. essentially on the surface. Free XYZ regression has no
such constraint and would have to re-learn the surface. The head therefore emits
    p_j = q_j + sum_i w_i (p_i - q_j) + resid * tanh(...)
The residual is bounded at RESID = 1.0mm, sized as ~2x the 0.54mm mean nearest-point
distance: it exists only to correct the soft-argmax's hull bias and any off-surface
offset in the annotation convention, never to relocate the landmark. HEAD=reg replaces
the whole thing with a free displacement bounded at BOUND = 4.0mm (96.3% of endpoint
errors are under 4mm; a bound at the p99 of 6mm would spend most of the tanh's useful
range on 1% of cases). Run both -- the point of the switch is that the heatmap claim gets
measured rather than assumed.

PER-LANDMARK LOCAL CONDITIONING -- THE THING fam_phase.py GOT WRONG. fam_phase regressed
a whole contour from one pooled global vector and scored 1.81. Here every landmark has
its OWN crop, its own per-point features and its own conditioning token, and NOTHING is
pooled over the ear. The token is built from the landmark's identity embedding plus the
offsets to its 2*NBR in-contour neighbours in the current estimate, clamped at the
contour boundary -- so an endpoint's token contains literal zeros on the side where the
contour stops, which is exactly the signal "you are an end". The per-point features carry
the along-contour coordinate t . (p - q), so the head can express "slide 0.8mm further
along the curve" directly rather than having to discover the curve direction.

HOW THE CURRENT ESTIMATE REACHES THE MODEL -- AND ITS LEAKAGE STATUS, HONESTLY.
This family REFINES an existing prediction. That prediction arrives through the trainer's
per-FOLD ARTEFACTS hook as `est` (E,85,3) float32 in the CANONICAL frame, and cls.BATCH
slices the batch's rows out of it. train_family.py already refuses an artefact that
cannot prove itself: it asserts `fold` matches and that `train_ear_mask` contains no
validation ear. Build it with `BUILD_EST=1 python research/code/fam_endpoint.py`.

  CLEAN: a VALIDATION ear's estimate is the out-of-fold prediction of the base models for
  the fold that ear belongs to, which is THIS fold. No model that produced it ever saw
  that ear's GT. The number this family reports is therefore honest as a refinement of a
  genuinely held-out estimate.

  NOT CLEAN, AND NOT PAPERED OVER: a TRAINING ear's estimate is the OOF prediction of some
  OTHER fold G's base models. Those models were trained on every ear outside fold G --
  which INCLUDES this fold's validation ears. So this fold's validation GT influenced,
  very diffusely (one ear in 272, through a base model's weights, on a different ear's
  prediction), the INPUTS the refiner was trained on. This is ordinary non-nested stacking
  contamination. It does not put validation GT in the loss, and it does not make the
  validation ears' inputs better than held-out -- which is the failure mode that inflates
  a score -- but it is a leak channel and it is stated rather than hidden. The artefact
  carries `nested=False` and a `leak_note` string, and this module PRINTS the note at
  construction. The strictly-clean version needs nested OOF: for each outer fold, retrain
  the base models with an inner 5-fold over that fold's training ears only, use the inner
  OOF for the training ears and the outer models for the validation ears, and stamp
  `nested=True`. That is 5x the base training and is NOT done here.

  FALLBACK: with no artefact (or EST_SRC=coarse) the estimate is batch['coarse'], the
  3.766mm coarse init. That mode is a much harder problem than this head was sized for --
  a 7mm crop covers only ~78% of coarse errors -- and exists so the trainer smoke test
  runs without artefacts. It warns loudly.

WHAT IT OUTPUTS. `pred` is (B,85,3): the refined landmarks at the selected indices and the
INCOMING ESTIMATE passed through everywhere else, because train_family.py's contract and
its whole report schema are all-85. With EP_SET=ends the trainer's ordered_MLE_mm is
therefore 77/85 frozen ensemble and only 8/85 under test; the number to read is the
endpoint subset, which `per_endpoint()` computes from the saved .npy and which the smoke
test prints. With EP_SET=all every landmark is refined and ordered_MLE_mm is meaningful
directly. Set EP_SET to measure the specialisation instead of assuming it.

KNOWN LIMITS, STATED SO THEY ARE NOT DISCOVERED LATER.
  * THE SPECIALISATION IS THE EXPERIMENT, SO RUN BOTH ARMS. EP_SET=ends and EP_SET=all are
    the SAME weights (the landmark set is a buffer, not a width); the only difference is
    what gets refined and what the loss sees. If EP_SET=all matches EP_SET=ends on the 8
    endpoints there is no specialisation, only a local-crop refiner, and the claim in this
    docstring is wrong.
  * TRAIN/EVAL DENSITY MISMATCH, INHERITED FROM THE HARNESS. sub_frac=0.625 means the crop
    holds ~128 points while training and ~203 while evaluating. The soft-argmax is a
    density-weighted average over a UNIFORM surface sample, so it is consistent under that
    change, but the max-pooled crop context is not. CFG_SUB_FRAC=1.0 removes the mismatch
    for this family at the cost of one augmentation; it is worth an arm.
  * SCALE AUGMENTATION MOVES THE CROP. aug_scale=0.20 means crop_r is +-10% of its stated
    physical size during training. Same trade the rest of the repo makes; flagged because
    every radius here is justified in millimetres.
  * WITH EP_SET=ends THE TRAINER'S ordered_MLE_mm IS 77/85 FROZEN. best_val_MLE_mm, which
    selects the checkpoint, is that same 85-mean, so checkpoint selection is driven by a
    quantity that is ~91% constant. It still ranks correctly (the constant cancels), but
    its resolution is 8/85 of what the printed number suggests -- read the deltas, not the
    absolute value, and score the run with per_endpoint() on the saved .npy.
  * NOTHING HERE IMPOSES THE ARC-LENGTH PROFILE. This family only localises the endpoints.
    Turning better endpoints into the -0.20mm the oracle promises needs the profile step
    (equidistant placement along the repositioned predicted curve for inner_helix and
    sup._antihelix only -- it DESTROYS outer_helix and concha), which lives elsewhere.

    ARTEFACTS=scratch/endpoint_est_f0.npz DATA=scratch/screen_data_8192nrm.npz \
      FAMILY=endpoint FOLD=0 SEED=0 EPOCHS=400 python research/code/train_family.py
    BUILD_EST=1 python research/code/fam_endpoint.py     # write the per-fold artefacts
    python research/code/fam_endpoint.py                 # CPU smoke + analysis, <90s
    SMOKE_TRAIN=1 python research/code/fam_endpoint.py   # + train_family.py end-to-end

ENVIRONMENT (each is also reachable as CFG_<NAME> for search_driver.py, EXCEPT USE_NRM
and EST_SRC, which cls.NEEDS/cls.BATCH are read off the CLASS before any instance exists)
  EP_SET     ends   ends | all | a JSON list of landmark indices
  CROP_R     7.0    crop radius, mm            KMAX    448  gather cap per crop
  HEAD       heat   heat | reg                 RESID   1.0  mm, tanh residual on the heat head
  BOUND      4.0    mm, tanh bound for HEAD=reg
  WIDTH      96     point-feature width        ROUNDS  2    local set-conv rounds
  NBR        3      in-contour neighbours per side in the conditioning token
  EMB        32     landmark embedding width   DROPOUT 0.1
  TAU        1.0    initial softmax temperature (a learned gain multiplies the logits)
  W_HEAT     0.3    weight of the Gaussian heatmap cross-entropy
  HEAT_SIGMA 1.0    mm, sigma of that Gaussian target
  W_AUX      0.3    weight on the pre-residual prediction
  LOSSK      l2     l2 | huber                 HUBER   2.0  mm, huber knee
  USE_NRM    1      oriented normals in the crop (ENVIRONMENT ONLY -- sets cls.NEEDS, and
                    cls.ROTATES tracks it: the augmenter refuses a ROTATES key it was not given)
  EST_SRC    artefact  artefact | coarse       (ENVIRONMENT ONLY -- sets cls.BATCH)
  EST_JIT    0.0    mm/coord jitter on the estimate during training
  NB_STATS   2      instrument this many forwards, then stop
  BUILD_EST  --     run the artefact builder instead of the smoke test
"""
import os, json, math, time
import numpy as np
import torch
import torch.nn as nn

NL, SCALE = 85, 30.0
CONTOURS = [(0, 24), (25, 54), (55, 74), (75, 84)]
CNAMES = ["outer_helix", "concha", "inner_helix", "sup._antihelix"]
NFOLD = 5
ENDS = sorted([lo for lo, _ in CONTOURS] + [hi for _, hi in CONTOURS])

# NEEDS and BATCH are read off the CLASS before it is instantiated, so the two switches
# that decide what the trainer LOADS have to come from the environment.
ENV_USE_NRM = int(os.environ.get("USE_NRM", "1"))
EST_SRC = os.environ.get("EST_SRC", "artefact")

ENV_DEFAULTS = dict(
    ep_set=os.environ.get("EP_SET", "ends"),
    crop_r=float(os.environ.get("CROP_R", "7.0")),
    kmax=int(os.environ.get("KMAX", "448")),
    head=os.environ.get("HEAD", "heat"),
    resid=float(os.environ.get("RESID", "1.0")),
    bound=float(os.environ.get("BOUND", "4.0")),
    width=int(os.environ.get("WIDTH", "96")),
    rounds=int(os.environ.get("ROUNDS", "2")),
    nbr=int(os.environ.get("NBR", "3")),
    emb=int(os.environ.get("EMB", "32")),
    dropout=float(os.environ.get("DROPOUT", "0.1")),
    tau=float(os.environ.get("TAU", "1.0")),
    w_heat=float(os.environ.get("W_HEAT", "0.3")),
    heat_sigma=float(os.environ.get("HEAT_SIGMA", "1.0")),
    w_aux=float(os.environ.get("W_AUX", "0.3")),
    lossk=os.environ.get("LOSSK", "l2"),
    huber=float(os.environ.get("HUBER", "2.0")),
    use_nrm=ENV_USE_NRM,
    est_jit=float(os.environ.get("EST_JIT", "0.0")),
    nb_stats=int(os.environ.get("NB_STATS", "2")),
)


def landmark_set(spec):
    if spec == "ends":
        return list(ENDS)
    if spec == "all":
        return list(range(NL))
    v = json.loads(spec) if isinstance(spec, str) else list(spec)
    assert v and all(0 <= int(i) < NL for i in v), f"EP_SET={spec!r} out of range"
    return sorted(int(i) for i in v)


def contour_of(j):
    for lo, hi in CONTOURS:
        if lo <= j <= hi:
            return lo, hi
    raise AssertionError(j)


# ------------------------------------------------------------------ frozen folds
def frozen_folds(ne):
    """Constraint 3, verbatim -- duplicated so the builder does not import the trainer."""
    subj = np.arange(ne) // 2
    parts = np.array_split(np.random.RandomState(12345).permutation(np.unique(subj)), NFOLD)
    return subj, [np.asarray(p) for p in parts]


# ------------------------------------------------------------------ crop geometry
def crop_idx(q, pc, radius, kmax, keep=8):
    """TRUE metric ball around each query, gathered nearest-first up to `kmax`.

    q (B,J,3) mm, pc (B,N,3) mm -> idx (B,J,K), mask (B,J,K), ntrue (B,J). `ntrue` is the
    UNCAPPED ball size so the caller can prove the cap is not truncating; the nearest
    `keep` members are kept regardless of the radius so a landmark is never blind.
    """
    K = min(int(kmax), pc.shape[1])
    d = torch.cdist(q, pc)
    nt = (d <= radius).sum(-1)
    dv, ii = d.topk(K, largest=False, dim=-1)          # sorted ascending
    m = dv <= radius
    if keep:
        m = m | (torch.arange(K, device=q.device) < min(keep, K))
    return ii, m, nt


def gather_nb(x, idx):
    B, J, K = idx.shape
    C = x.shape[-1]
    return torch.gather(x, 1, idx.reshape(B, J * K, 1).expand(-1, -1, C)).view(B, J, K, C)


def mmax(x, m):
    return x.masked_fill(~m[..., None], -1e4).amax(2)


def mmean(x, m):
    w = m[..., None].to(x.dtype)
    return (x * w).sum(2) / w.sum(2).clamp(min=1)


def nb_report(nt, K):
    v = nt.reshape(-1).float()
    return dict(min=int(v.amin()), median=int(v.median()), max=int(v.amax()),
                mean=round(float(v.mean()), 1), cap=int(K),
                frac_truncated=round(float((v > K).float().mean()), 4),
                frac_empty=round(float((v == 0).float().mean()), 4))


# ------------------------------------------------------------------ augmentation
def rand_rot(B, maxang, gen, dev):
    """train_family.rand_rot verbatim, so both augmenters draw the same distribution."""
    ax = torch.randn(B, 3, device=dev, generator=gen)
    ax = ax / ax.norm(dim=1, keepdim=True)
    ang = (torch.rand(B, device=dev, generator=gen) - .5) * maxang
    c, s = ang.cos(), ang.sin(); x, y, z = ax[:, 0], ax[:, 1], ax[:, 2]; C = 1 - c
    return torch.stack([
        torch.stack([c + x*x*C, x*y*C - z*s, x*z*C + y*s], -1),
        torch.stack([y*x*C + z*s, c + y*y*C, y*z*C - x*s], -1),
        torch.stack([z*x*C - y*s, z*y*C + x*s, c + z*z*C], -1)], 1)


def _rot(t, R):
    return torch.einsum("b...j,bij->b...i", t, R)


ALLOWED = {"pc", "nrm", "coarse", "ear", "est"}


def endpoint_augment(b, tg, cfg, rotates, gen):
    """ONE per-ear similarity for the cloud, the coarse init, the ESTIMATE and the target.

    default_augment moves only tensors shaped exactly like the cloud, so `est` -- (B,85,3),
    the thing every crop is centred on -- would be left in the ORIGINAL frame while pc,
    coarse and the target are rotated and scaled. Every shape check passes and the crops
    are taken in the wrong place. Hence our own, with a stray-key assert so a future extra
    batch tensor fails loudly instead of silently staying put.

    `est` gets its own jitter (cfg['est_jit'], default 0) and NOT aug_qjit: qjit is 0.9
    mm/coord, i.e. 1.56mm rms in 3D, which would swamp a 1.3mm estimate and destroy the
    error distribution the bounded head is calibrated against.
    """
    pc = b["pc"]; B, S, N, _ = pc.shape; dev = pc.device
    stray = [k for k, v in b.items() if torch.is_tensor(v) and k not in ALLOWED]
    assert not stray, (f"endpoint_augment does not know how to move {stray}; an unclassified "
                       f"tensor would stay in the ORIGINAL frame while everything else is "
                       f"rotated. Add it to ALLOWED and handle it here.")
    assert set(rotates) <= set(b), f"cls.ROTATES names {sorted(set(rotates) - set(b))}"
    R = rand_rot(B, cfg["aug_rot"], gen, dev)
    sc = 1 + (torch.rand(B, 1, 1, device=dev, generator=gen) - .5) * cfg["aug_scale"]
    nsub = max(8, min(N, int(round(N * cfg["sub_frac"]))))
    sub = torch.rand(B, S, N, device=dev, generator=gen).argsort(-1)[..., :nsub]
    g = sub[..., None].expand(-1, -1, -1, 3)
    out = dict(b)
    p = torch.gather(pc, 2, g)
    out["pc"] = _rot(p, R) * sc[:, None] + \
        torch.randn(p.shape, device=dev, generator=gen) * cfg["aug_jit"]
    if "nrm" in b:
        out["nrm"] = _rot(torch.gather(b["nrm"], 2, g), R)          # a direction: never scaled
    out["coarse"] = _rot(b["coarse"], R) * sc + \
        torch.randn(b["coarse"].shape, device=dev, generator=gen) * cfg["aug_qjit"]
    if "est" in b:
        out["est"] = _rot(b["est"], R) * sc + \
            torch.randn(b["est"].shape, device=dev, generator=gen) * float(cfg["est_jit"])
    return out, _rot(tg, R) * sc


# ------------------------------------------------------------------ the estimate
def est_tensor(meta):
    """The per-ear current estimate (E,85,3) CANONICAL, or None. Cached on `meta`."""
    a = meta.get("artefacts") or {}
    if EST_SRC != "artefact" or "est" not in a:
        return None
    if "_est_t" not in meta:
        e = np.asarray(a["est"], np.float32)
        assert e.ndim == 3 and e.shape[1:] == (NL, 3), \
            f"artefact 'est' must be (E,85,3) CANONICAL, got {e.shape}"
        # the trainer proves fold-safety by indexing train_ear_mask with va_idx; if `est` is
        # not the same length, that proof certifies a DIFFERENT set of ears than the one the
        # crops are actually centred on, and the ear axis is silently misaligned.
        assert len(e) == len(a["train_ear_mask"]), \
            (f"artefact 'est' has {len(e)} ears but train_ear_mask has "
             f"{len(a['train_ear_mask'])} -- the leakage assertion would not cover `est`")
        meta["_est_t"] = torch.tensor(e, device=meta["dev"])
    return meta["_est_t"]


# ------------------------------------------------------------------ the model
class MODEL(nn.Module):
    """train_family.py's REGISTRY['endpoint'].

    One independent localisation problem per selected landmark. No tensor is ever pooled
    over the ear or over the contour: the only global object in the graph is the shared
    weight set, which is what stops the head memorising a per-ear constant (the exact
    failure mode of the affine-regression gate, which hit its oracle on training ears and
    gained nothing held out).
    """
    DEFAULTS = ENV_DEFAULTS
    SEARCH_SPACE = dict(crop_r=[5.5, 7.0, 8.5], kmax=[256, 448], width=[64, 96, 128],
                        rounds=[1, 2, 3], nbr=[2, 3, 5], emb=[16, 32, 64],
                        resid=[0.0, 0.5, 1.0, 2.0], bound=[3.0, 4.0, 6.0],
                        tau=[0.5, 1.0, 2.0], w_heat=[0.0, 0.1, 0.3, 1.0],
                        heat_sigma=[0.6, 1.0, 1.6], w_aux=[0.0, 0.3],
                        est_jit=[0.0, 0.25, 0.5], dropout=[0.0, 0.1, 0.2],
                        lossk=["l2", "huber"], lr=[3e-4, 7e-4, 1.5e-3])
    NEEDS = ("nrm",) if ENV_USE_NRM else ()
    # ROTATES must track NEEDS: endpoint_augment asserts every ROTATES key is present in the
    # batch (so a direction tensor can never be left unrotated), and with USE_NRM=0 'nrm' is
    # never loaded. A constant ("nrm",) crashed the first training step of that arm.
    ROTATES = NEEDS
    SAMPLES = 1
    AUGMENT = staticmethod(endpoint_augment)

    @staticmethod
    def BATCH(ears, samples, meta):
        t = est_tensor(meta)
        if t is None:
            return {}
        e = torch.as_tensor(np.asarray(ears), dtype=torch.long, device=t.device)
        return {"est": t[e]}

    def __init__(self, cfg, meta):
        super().__init__()
        c = self.c = {**ENV_DEFAULTS, **(cfg or {})}
        for k in ("kmax", "width", "rounds", "nbr", "emb", "use_nrm", "nb_stats"):
            c[k] = int(c[k])
        for k in ("crop_r", "resid", "bound", "dropout", "tau", "w_heat", "heat_sigma",
                  "w_aux", "huber", "est_jit"):
            c[k] = float(c[k])
        assert c["head"] in ("heat", "reg"), c["head"]
        assert c["lossk"] in ("l2", "huber"), c["lossk"]
        assert not (c["use_nrm"] and "nrm" not in self.NEEDS), \
            "use_nrm is on but NEEDS is empty: set USE_NRM=1 in the ENVIRONMENT (not only " \
            "CFG_USE_NRM) so the trainer loads 'nrm', and point DATA at a *nrm.npz"
        assert meta["nl"] == NL and meta["contours"] == CONTOURS, "landmark geometry contract broken"

        idx = landmark_set(c["ep_set"])
        self.register_buffer("idx", torch.tensor(idx, dtype=torch.long))
        # per-landmark in-contour neighbours, CLAMPED at the contour boundary: an endpoint's
        # token therefore carries literal zero offsets on the side where the contour stops.
        nb, ta, tb = [], [], []
        for j in idx:
            lo, hi = contour_of(j)
            nb.append([min(max(j + o, lo), hi) for o in range(-c["nbr"], c["nbr"] + 1) if o])
            ta.append(max(j - 1, lo)); tb.append(min(j + 1, hi))
        self.register_buffer("nb", torch.tensor(nb, dtype=torch.long))
        self.register_buffer("ta", torch.tensor(ta, dtype=torch.long))
        self.register_buffer("tb", torch.tensor(tb, dtype=torch.long))

        W, E = c["width"], c["emb"]
        cin = 5 + (5 if c["use_nrm"] else 0)     # rel3 r1 tdotrel1 (+ n3 ndotu1 ndott1)
        self.emb = nn.Embedding(NL, E)
        self.ctx = nn.Sequential(nn.Linear(E + 3 * 2 * c["nbr"], W), nn.ReLU(),
                                 nn.Linear(W, W))
        self.inp = nn.Sequential(nn.Linear(cin, W), nn.ReLU(), nn.Linear(W, W))
        self.rnd = nn.ModuleList([nn.Sequential(nn.Linear(2 * W, W), nn.ReLU(),
                                                nn.Linear(W, W), nn.ReLU())
                                  for _ in range(c["rounds"])])
        self.drop = nn.Dropout(c["dropout"])
        cctx = 3 * W
        if c["head"] == "heat":
            self.out = nn.Sequential(nn.Linear(2 * W, W), nn.ReLU(), nn.Linear(W, 1))
            self.log_gain = nn.Parameter(torch.tensor(math.log(1.0 / max(c["tau"], 1e-6))))
            if c["resid"] > 0:
                self.res = nn.Sequential(nn.Linear(cctx, W), nn.ReLU(), nn.Linear(W, 3))
        else:
            self.reg = nn.Sequential(nn.Linear(cctx, W), nn.ReLU(), nn.Linear(W, 3))
        self.left = c["nb_stats"]
        self.stats, self.seen = {}, set()

        art = meta.get("artefacts") or {}
        if EST_SRC == "artefact" and "est" in art:
            self.note = (f"est artefact: {int(art.get('n_members', 0))} members, "
                         f"nested={bool(np.asarray(art.get('nested', False)))} -- "
                         f"{str(np.asarray(art.get('leak_note', 'no leak_note field')))}")
        else:
            self.note = ("NO 'est' ARTEFACT (or EST_SRC=coarse) -> refining batch['coarse'], the "
                         "3.766mm init, which a 7mm crop reaches for only ~78% of endpoints. NOT "
                         "the configuration this head is sized for. BUILD_EST=1 python "
                         "research/code/fam_endpoint.py, then ARTEFACTS=scratch/endpoint_est_f<F>.npz")
        print(f"[fam_endpoint] {len(idx)} landmarks ({c['ep_set']}) head={c['head']} "
              f"crop_r={c['crop_r']}mm kmax={c['kmax']} resid={c['resid']}mm "
              f"bound={c['bound']}mm use_nrm={c['use_nrm']}\n  {self.note}", flush=True)

    def forward(self, b):
        c = self.c
        pc, est = b["pc"], b.get("est", b["coarse"])
        B = pc.shape[0]
        q = est[:, self.idx]                                            # (B,J,3)
        t = est[:, self.tb] - est[:, self.ta]
        t = t / t.norm(dim=-1, keepdim=True).clamp(min=1e-6)            # (B,J,3) along-contour
        off = (est[:, self.nb] - q[:, :, None]) / c["crop_r"]           # (B,J,2*nbr,3)

        ii, m, nt = crop_idx(q, pc, c["crop_r"], c["kmax"])
        P = gather_nb(pc, ii)                                           # (B,J,K,3)
        rel = P - q[:, :, None]
        dist = rel.norm(dim=-1, keepdim=True)
        f = [rel / c["crop_r"], dist / c["crop_r"],
             (rel * t[:, :, None]).sum(-1, keepdim=True) / c["crop_r"]]
        if c["use_nrm"]:
            n = gather_nb(b["nrm"], ii)
            f += [n, (n * rel).sum(-1, keepdim=True) / dist.clamp(min=1e-6),
                  (n * t[:, :, None]).sum(-1, keepdim=True)]
        tok = self.ctx(torch.cat([self.emb(self.idx)[None].expand(B, -1, -1),
                                  off.flatten(2)], -1))                 # (B,J,W)
        x = self.inp(torch.cat(f, -1)) + tok[:, :, None]
        for lyr in self.rnd:
            g = mmax(x, m)
            x = lyr(torch.cat([x, g[:, :, None].expand_as(x)], -1))
        g, gm = mmax(x, m), mmean(x, m)
        ctx = self.drop(torch.cat([g, gm, tok], -1))

        logit = None
        if c["head"] == "heat":
            lg = self.out(torch.cat([x, g[:, :, None].expand_as(x)], -1)).squeeze(-1)
            logit = (lg * self.log_gain.exp()).masked_fill(~m, -1e4)
            w = torch.softmax(logit, -1)
            pre = q + (w[..., None] * rel).sum(2)
            fin = pre + c["resid"] * torch.tanh(self.res(ctx)) if c["resid"] > 0 else pre
        else:
            pre = fin = q + c["bound"] * torch.tanh(self.reg(ctx))

        # audit once per DISTINCT cloud size. Training runs at sub_frac*N and evaluation at
        # the full N, and it is the denser EVAL crop that truncates first; keying on the
        # point count is what makes eval truncation visible instead of only the benign
        # training figure (a plain first-N-forwards counter reports training twice).
        npt = int(pc.shape[1])
        if self.left > 0 and npt not in self.seen:
            self.seen.add(npt); self.left -= 1
            self.stats = dict(landmarks=int(self.idx.numel()), crop_r_mm=c["crop_r"],
                              points=npt, **nb_report(nt, ii.shape[-1]))
            print(f"  crop audit ({self.stats['landmarks']} landmarks x {B} ears, "
                  f"r={c['crop_r']}mm, {npt}-pt cloud, {'train' if self.training else 'EVAL'}): "
                  f"ball size before the cap min {self.stats['min']} / med "
                  f"{self.stats['median']} / max {self.stats['max']} / mean "
                  f"{self.stats['mean']}, cap {self.stats['cap']}, truncated "
                  f"{self.stats['frac_truncated']:.4f}, empty "
                  f"{self.stats['frac_empty']:.4f}", flush=True)

        pred, aux = est.clone(), est.clone()
        pred[:, self.idx] = fin
        aux[:, self.idx] = pre
        return {"pred": pred, "aux": [aux], "fin": fin, "pre": pre, "q": q,
                "logit": logit, "crop": P, "mask": m, "nt": nt}

    def loss(self, out, tg):
        """Only the REFINED landmarks contribute. The 77 pass-through landmarks are the
        incoming estimate and carry no gradient; including them would just add a constant
        that makes the printed loss unreadable."""
        c = self.c
        gt = tg[:, self.idx]

        def term(p):
            d = p - gt
            if c["lossk"] == "l2":
                return (d ** 2).sum(-1).mean()
            r = d.norm(dim=-1)
            k = c["huber"]
            return torch.where(r < k, 0.5 * r ** 2, k * (r - 0.5 * k)).mean()

        L = term(out["fin"]) + (c["w_aux"] * term(out["pre"]) if c["w_aux"] > 0 else 0.0)
        if c["head"] == "heat" and c["w_heat"] > 0:
            d2 = ((out["crop"] - gt[:, :, None]) ** 2).sum(-1)
            tp = torch.softmax((-d2 / (2 * c["heat_sigma"] ** 2)).masked_fill(~out["mask"], -1e4), -1)
            L = L + c["w_heat"] * -(tp * torch.log_softmax(out["logit"], -1)).sum(-1).mean()
        return L

    def diag(self):
        return self.stats


# ------------------------------------------------------------------ reporting
def per_endpoint(P, G, idx=None):
    """Per-landmark error table over a matched (n,85,3) prediction/GT pair."""
    idx = ENDS if idx is None else list(idx)
    E = np.linalg.norm(np.asarray(P, float) - np.asarray(G, float), axis=2)
    rows = [(j, float(E[:, j].mean()), float(np.median(E[:, j])),
             float(np.percentile(E[:, j], 90))) for j in idx]
    return rows, float(E[:, idx].mean()), float(E.mean())


def print_table(before, after, G, idx=None, tag=""):
    idx = ENDS if idx is None else list(idx)
    rb, mb, ab = per_endpoint(before, G, idx)
    ra, ma, aa = per_endpoint(after, G, idx)
    print(f"  {'lm':>4s} {'contour':>15s} {'before':>8s} {'after':>8s} {'delta':>8s}  {tag}")
    for (j, b, _, _), (_, a, _, _) in zip(rb, ra):
        lo, _ = contour_of(j)
        nm = CNAMES[[c[0] for c in CONTOURS].index(lo)]
        print(f"  {j:4d} {nm:>15s} {b:8.4f} {a:8.4f} {a-b:+8.4f}")
    print(f"  {'':4s} {'SUBSET mean':>15s} {mb:8.4f} {ma:8.4f} {ma-mb:+8.4f}")
    print(f"  {'':4s} {'all 85 mean':>15s} {ab:8.4f} {aa:8.4f} {aa-ab:+8.4f}")
    return mb, ma


# ------------------------------------------------------------------ artefact builder
MEMBERS = [("screen_normalsfix", (0, 1, 2)), ("famA_kpconv", ()), ("famA_ptv3", ())]

LEAK_NOTE = (
    "OOF ensemble. A VALIDATION ear's estimate comes from the base models of ITS OWN fold, "
    "which never saw it -- clean. A TRAINING ear's estimate comes from another fold's base "
    "models, which WERE trained on this fold's validation ears, so this fold's validation "
    "GT diffusely influenced the refiner's training INPUTS (non-nested stacking "
    "contamination). No validation GT reaches the loss and the validation inputs are not "
    "better than held out. nested=True requires an inner 5-fold retrain of the base models "
    "per outer fold; not done here.")


def build_est(work="scratch", data=None, members=MEMBERS, out_tpl=None, nested=False):
    """Assemble the OOF ensemble into one fold-safe `est` artefact per outer fold.

    Predictions on disk are WORLD-frame (train_family saves to_world(...)); the trainer's
    batch is CANONICAL, so they are mapped back with p_canon = (p_world - c0) @ R^T, R
    being a pure rotation. The round trip is asserted against the npz's own `true`.
    """
    data = data or f"{work}/screen_data_8192nrm.npz"
    out_tpl = out_tpl or f"{work}/endpoint_est_f{{}}.npz"
    z = np.load(data)
    R, c0 = z["R"].astype(np.float64), z["c0"].astype(np.float64)
    TRUE = z["true"].astype(np.float64)
    NE = len(R)
    subj, parts = frozen_folds(NE)

    got, src = [], np.full(NE, -1)
    for tag, seeds in members:
        P = np.full((NE, NL, 3), np.nan)
        for f in range(NFOLD):
            js = f"{work}/{tag}_s{seeds[0]}_f{f}.json" if seeds else f"{work}/{tag}_f{f}.json"
            assert os.path.exists(js), f"missing {js}"
            ix = np.array(json.load(open(js))["val_ear_index"])
            ps = [f"{work}/{tag}_s{s}_f{f}.npy" for s in seeds] or [f"{work}/{tag}_f{f}.npy"]
            P[ix] = np.mean([np.load(p) for p in ps], 0).astype(np.float64)
            src[ix] = f
        assert not np.isnan(P).any(), f"{tag}: an ear was never held out"
        got.append(P)
        e = np.stack([(P[i] - c0[i]) @ R[i].T for i in range(NE)])
        print(f"  {tag:20s} {NE} ears, pooled OOF {np.linalg.norm(e - TRUE, axis=2).mean():.4f} mm")
    W = np.mean(got, 0)
    E = np.stack([(W[i] - c0[i]) @ R[i].T for i in range(NE)])          # world -> canonical
    err = np.linalg.norm(E - TRUE, axis=2)
    print(f"  ensemble of {len(got)}: pooled OOF {err.mean():.4f} mm | "
          f"endpoints {err[:, ENDS].mean():.4f} mm")

    for f in range(NFOLD):
        mask = ~np.isin(subj, parts[f])
        p = out_tpl.format(f)
        np.savez(p, est=E.astype(np.float32), fold=np.int64(f), train_ear_mask=mask,
                 src_fold=src.astype(np.int64), n_members=np.int64(len(got)),
                 members=np.array([t for t, _ in members]), nested=np.bool_(nested),
                 leak_note=np.array(LEAK_NOTE))
        print(f"  wrote {p}  ({int(mask.sum())} train / {int((~mask).sum())} val ears)")
    return E, err


# ------------------------------------------------------------------ smoke test
def synth(B, npts, seed=0):
    """A pinna-scale curved sheet (50 x 36 mm) sampled at ~1.09mm spacing, with the 85
    landmarks on FOUR curves lying exactly on it -- so the crops, the along-contour
    tangent and the neighbour offsets all mean on this cloud what they mean on real ears."""
    g = torch.Generator().manual_seed(seed)
    z = lambda u, v: 0.006 * u ** 2 - 0.010 * v ** 2 + 3.0 * torch.sin(u / 6.0)
    nrm_of = lambda u, v: torch.nn.functional.normalize(
        torch.stack([-(0.012 * u + 0.5 * torch.cos(u / 6.0)), 0.020 * v, torch.ones_like(u)], -1), dim=-1)
    u = (torch.rand(B, npts, generator=g) - .5) * 50.0
    v = (torch.rand(B, npts, generator=g) - .5) * 36.0
    pc = torch.stack([u, v, z(u, v)], -1)
    nrm = nrm_of(u, v)
    lu, lv = [], []
    for ci, (lo, hi) in enumerate(CONTOURS):
        s = torch.linspace(0, 1, hi - lo + 1)[None].expand(B, -1)
        a = 0.35 + 0.18 * ci
        lu.append((s - .5) * 42.0 * (1 - 0.15 * ci))
        lv.append(14.0 * torch.sin(math.pi * s * a + ci) - 4.0 * ci)
    lu, lv = torch.cat(lu, 1), torch.cat(lv, 1)
    tgt = torch.stack([lu, lv, z(lu, lv)], -1)
    est = tgt + torch.randn(B, NL, 3, generator=g) * 0.75
    return pc, nrm, est, tgt


def trainer_smoke():
    """End-to-end through train_family.py, proving the FAMILY CONTRACT and, crucially, that
    the per-fold `est` ARTEFACT survives the trainer's leakage assertions."""
    import tempfile, train_family as TF
    tmp = os.path.join(tempfile.gettempdir(), "fam_endpoint_smoke")
    dp, tp, sp, _ = TF.fake_bundle(tmp, npts=1024)
    d = dict(np.load(dp))
    ne = len(d["coarse"])
    # fake_bundle emits no normals and NEEDS is fixed at import, so add them. Its surface
    # is the height field z(u,v,k) below, so the normal is analytic; the assert catches the
    # coupling if fake_bundle's geometry ever changes.
    u, v = d["clouds"][..., 0], d["clouds"][..., 1]
    k = np.arange(ne, dtype=np.float32)[:, None, None]
    a, b_ = u / 6.0 + 0.1 * k, v / 5.0 - 0.07 * k
    assert np.abs(1.2 * np.sin(a) * np.cos(b_) + 0.15 * k / ne - d["clouds"][..., 2]).max() < 0.4, \
        "train_family.fake_bundle's height field changed -- the analytic normals are stale"
    n = np.stack([-1.2 * np.cos(a) * np.cos(b_) / 6.0,
                  1.2 * np.sin(a) * np.sin(b_) / 5.0, np.ones_like(u)], -1)
    d["nrm"] = (n / np.linalg.norm(n, axis=-1, keepdims=True)).astype(np.float32)
    np.savez(dp, **d)
    subj, parts = frozen_folds(ne)
    ap = f"{tmp}/est_f0.npz"
    np.savez(ap, est=(d["coarse"] + 0.2).astype(np.float32), fold=np.int64(0),
             train_ear_mask=~np.isin(subj, parts[0]), n_members=np.int64(3),
             nested=np.bool_(False), leak_note=np.array(LEAK_NOTE))
    # the trainer must REFUSE an artefact that claims validation ears as training ears
    bad = f"{tmp}/est_bad.npz"
    np.savez(bad, est=d["coarse"], fold=np.int64(0), train_ear_mask=np.ones(ne, bool))
    env = dict(FAMILY="endpoint", FAMILY_MODULE="fam_endpoint", FAMILY_CLASS="MODEL",
               FOLD="0", SEED="0", EPOCHS="2", WORK=tmp, DATA=dp, TRIS=tp, SSM=sp,
               TTA="1", EVAL_EVERY="2", ALIAS="0", TAG="fam_endpoint_smoke",
               ARTEFACTS=bad, CFG_BS="8", CFG_WIDTH="32", CFG_KMAX="64", CFG_CROP_R="6.0")
    keep = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        try:
            TF.main(); raise SystemExit("a leaky artefact was ACCEPTED")
        except AssertionError as e:
            assert "LEAK" in str(e), str(e)
        print("  refused a train_ear_mask covering validation ears (LEAK)")
        os.environ["ARTEFACTS"] = ap
        res = TF.main()
    finally:
        for k, v in keep.items():
            os.environ.pop(k) if v is None else os.environ.__setitem__(k, v)
    assert res["ordered_MLE_full_mm"] is not None, "full pipeline did not run"
    assert res["config"]["_batch_hook"] and res["config"]["_augment"] == "endpoint_augment"
    print(f"  trainer report OK | raw {res['ordered_MLE_mm']:.4f} -> full "
          f"{res['ordered_MLE_full_mm']:.4f} mm | {res['params']:,} params")


def real_smoke(work="scratch", nears=12, steps=200, bs=6, lr=3e-3, seed=0):
    """Per-endpoint error BEFORE and AFTER on TRAINING ears -- an OVERFIT capacity check.

    This is deliberately NOT a generalisation claim: it is 12 training ears of fold 0 and
    200 CPU gradient steps, with GT in the loss for exactly those ears and no held-out
    measurement anywhere. It answers one question only -- can the crop + heatmap head
    express the correction at all, or is the target outside its reach? A number here says
    nothing whatever about what the family will score on a fold.
    """
    dp, ap = f"{work}/screen_data_8192nrm.npz", f"{work}/endpoint_est_f0.npz"
    if not (os.path.exists(dp) and os.path.exists(ap)):
        print(f"  SKIPPED: need {dp} and {ap} (BUILD_EST=1 python research/code/fam_endpoint.py)")
        return
    t0 = time.time()
    z, a = np.load(dp), np.load(ap)
    ne = len(z["coarse"])
    subj, parts = frozen_folds(ne)
    tr = np.array([i for i in range(ne) if subj[i] not in set(parts[0].tolist())])[:nears]
    pc = torch.tensor(z["clouds"][tr, 0]).float()
    nr = torch.tensor(z["nrm"][tr, 0]).float()
    est = torch.tensor(a["est"][tr]).float()
    tg = torch.tensor(z["true"][tr]).float()
    print(f"  {len(tr)} fold-0 TRAINING ears, {pc.shape[1]}-pt clouds, loaded in "
          f"{time.time()-t0:.1f}s")

    torch.manual_seed(seed)
    meta = dict(nl=NL, contours=CONTOURS, scale=SCALE, npts=pc.shape[1], fold=0,
                dev="cpu", n_train_ears=len(tr), artefacts={k: a[k] for k in a.files})
    net = MODEL(dict(nb_stats=1), meta)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    rng = np.random.RandomState(seed)
    for s in range(steps):
        bi = rng.choice(len(tr), bs, replace=False)
        b = {"pc": pc[bi], "nrm": nr[bi], "est": est[bi], "coarse": est[bi],
             "ear": torch.tensor(bi)}
        L = net.loss(net(b), tg[bi])
        opt.zero_grad(); L.backward(); opt.step(); sch.step()
    net.eval()
    with torch.no_grad():
        P = torch.cat([net({"pc": pc[i:i+4], "nrm": nr[i:i+4], "est": est[i:i+4],
                            "coarse": est[i:i+4], "ear": torch.arange(i, min(i+4, len(tr)))})["pred"]
                       for i in range(0, len(tr), 4)]).numpy()
    ix = net.idx.tolist()
    print(f"\n  --- OVERFIT capacity check: {steps} steps, loss {float(L):.4f}, "
          f"{time.time()-t0:.0f}s ---")
    mb, ma = print_table(est.numpy(), P, tg.numpy(), ix, tag="(TRAINING ears -- NOT held out)")
    print(f"  refined subset {mb:.4f} -> {ma:.4f} mm, {100*(mb-ma)/mb:+.1f}%. CAPACITY ONLY: "
          f"the head can reach the target through the crop. Whether it GENERALISES is the "
          f"5-fold run, not this.")


def analysis(work="scratch"):
    """The measured facts this module is sized against, recomputed from the artefacts."""
    ap = f"{work}/endpoint_est_f0.npz"
    dp = f"{work}/screen_data_8192nrm.npz"
    if not (os.path.exists(ap) and os.path.exists(dp)):
        print("  SKIPPED (no artefacts on this box)")
        return
    a, z = np.load(ap), np.load(dp)
    E, T = a["est"].astype(np.float64), z["true"].astype(np.float64)
    d = np.linalg.norm(E - T, axis=2)
    print(f"  pooled OOF estimate {d.mean():.4f} mm | endpoints {d[:, ENDS].mean():.4f} | "
          f"others {d[:, np.setdiff1d(np.arange(NL), ENDS)].mean():.4f}")
    print(f"  {'lm':>4s} {'contour':>15s} {'mean':>7s} {'med':>7s} {'p90':>7s} {'p99':>7s} {'max':>7s}")
    for j in ENDS:
        lo, _ = contour_of(j)
        print(f"  {j:4d} {CNAMES[[c[0] for c in CONTOURS].index(lo)]:>15s} {d[:, j].mean():7.4f} "
              f"{np.median(d[:, j]):7.4f} {np.percentile(d[:, j], 90):7.4f} "
              f"{np.percentile(d[:, j], 99):7.4f} {d[:, j].max():7.4f}")
    de = d[:, ENDS]
    print("  P(|GT-est| < R) over the 8 endpoints: " +
          "  ".join(f"{r}mm {100*(de < r).mean():.2f}%" for r in (4, 6, 7, 8)))
    print(f"       worst landmark at 7mm: {100*(de < 7).mean(0).min():.1f}%")
    P = z["clouds"][:, 0].astype(np.float64)
    nn_, cnt = [], []
    for i in range(len(P)):
        dg = np.linalg.norm(P[i][None] - T[i, ENDS][:, None], axis=2)
        dq = np.linalg.norm(P[i][None] - E[i, ENDS][:, None], axis=2)
        nn_.append(dg.min(1)); cnt.append((dq <= 7.0).sum(1))
    nn_, cnt = np.concatenate(nn_), np.concatenate(cnt)
    print(f"  GT endpoint -> nearest 8192-cloud point: mean {nn_.mean():.4f} med "
          f"{np.median(nn_):.4f} p90 {np.percentile(nn_, 90):.4f} max {nn_.max():.4f} mm"
          f"   <- the HARD-argmax floor")
    print(f"  points inside the 7mm ball: min {cnt.min()} p05 {np.percentile(cnt,5):.0f} "
          f"med {np.median(cnt):.0f} p95 {np.percentile(cnt,95):.0f} max {cnt.max()}"
          f"   -> KMAX=448 truncates {100*(cnt>448).mean():.2f}% of EVAL crops")


def _smoke():
    t0 = time.time()
    torch.manual_seed(0)
    B, NPTS = 2, 1536                       # 50x36mm sheet at ~1.09mm spacing
    pc, nrm, est, tgt = synth(B, NPTS)
    meta = dict(nl=NL, contours=CONTOURS, scale=SCALE, npts=NPTS, fold=0, dev="cpu",
                n_train_ears=272, artefacts={})

    print("=" * 78)
    print("SMOKE 1/4 -- forward/backward, both heads, B=2")
    for head in ("heat", "reg"):
        net = MODEL(dict(head=head, nb_stats=1), meta)
        b = {"pc": pc, "nrm": nrm, "est": est, "coarse": est, "ear": torch.arange(B)}
        out = net(b)
        L = net.loss(out, tgt)
        L.backward()
        gn = sum(float(p.grad.norm()) ** 2 for p in net.parameters() if p.grad is not None) ** .5
        nog = [n for n, p in net.named_parameters() if p.grad is None]
        assert out["pred"].shape == (2, NL, 3), out["pred"].shape
        assert torch.isfinite(out["pred"]).all() and not nog, nog
        # the 77 unselected landmarks must be the incoming estimate, bit for bit
        keep = np.setdiff1d(np.arange(NL), net.idx.numpy())
        assert torch.equal(out["pred"][:, keep], est[:, keep]), "a non-selected landmark moved"
        # the displacement must respect its stated bound
        disp = (out["fin"] - out["q"]).norm(dim=-1).amax()
        lim = net.c["crop_r"] + net.c["resid"] if head == "heat" else net.c["bound"]
        assert float(disp) <= lim + 1e-4, (float(disp), lim)
        print(f"  head={head:4s} pred {tuple(out['pred'].shape)}  refined {net.idx.numel()}  "
              f"params {sum(p.numel() for p in net.parameters()):,}  loss {float(L):.4f}  "
              f"grad-norm {gn:.3f}  max|disp| {float(disp):.3f} <= {lim} mm")

    print("\nSMOKE 2/4 -- EP_SET switch, conditioning locality, augmentation")
    n8 = MODEL(dict(nb_stats=0), meta)
    nall = MODEL(dict(ep_set="all", kmax=96, crop_r=5.5, nb_stats=0), meta)
    assert n8.idx.tolist() == ENDS and nall.idx.numel() == NL
    print(f"  EP_SET=ends -> {n8.idx.tolist()}\n  EP_SET=all  -> 85 landmarks, and the SAME "
          f"{sum(p.numel() for p in nall.parameters()):,} parameters: the landmark set is data "
          f"(a buffer), not width, so the two arms differ only in what they refine")
    # an endpoint's neighbour token is CLAMPED, so it carries zero offsets on the dead side
    off = (est[:, n8.nb] - est[:, n8.idx][:, :, None]).norm(dim=-1)
    print(f"  clamped-neighbour offsets that are exactly 0 (the 'you are an end' signal): "
          f"{int((off == 0).sum())}/{off.numel()} = {n8.c['nbr']} per landmark, as designed")
    assert int((off == 0).sum()) == off.numel() // 2
    # perturbing the cloud OUTSIDE one landmark's crop must not move that landmark. The
    # perturbation pushes points RADIALLY OUTWARD by 1.5x, so it cannot smuggle a point
    # into the ball and the check is about locality rather than about the crop's edge.
    with torch.no_grad():
        n8.eval()
        b_ = {"nrm": nrm, "est": est, "coarse": est, "ear": torch.arange(B)}
        base = n8({"pc": pc, **b_})["fin"]
        c_ = est[:, ENDS[0]][:, None]
        far = ((pc - c_).norm(dim=-1) > n8.c["crop_r"] + 0.5)[..., None]
        alt = n8({"pc": torch.where(far, c_ + (pc - c_) * 1.5, pc), **b_})["fin"]
    mv = (alt[:, 0] - base[:, 0]).norm(dim=-1).amax()
    print(f"  pushing every point >{n8.c['crop_r']+0.5}mm from landmark 0 a further 50% away "
          f"moves its prediction by {float(mv):.2e} mm (locality: nothing is pooled over the ear)")
    assert float(mv) < 1e-5
    g = torch.Generator().manual_seed(3)
    cfg = dict(aug_rot=1.2, aug_scale=0.20, aug_jit=0.0, aug_qjit=0.0, sub_frac=1.0, est_jit=0.0)
    b0 = {"pc": pc[:, None], "nrm": nrm[:, None], "coarse": est, "est": est,
          "ear": torch.arange(B)}
    b1, tg1 = endpoint_augment(b0, tgt, cfg, ("nrm",), g)
    dd = lambda t: (t[:, :, None].double() - t[:, None].double()).norm(dim=-1)
    s = (dd(tg1).sum((1, 2)) / dd(tgt).sum((1, 2)))
    for nm, a_, c_ in (("est", est, b1["est"]), ("coarse", est, b1["coarse"])):
        e = float((dd(c_) - s[:, None, None] * dd(a_)).abs().max())
        assert e < 1e-4, f"{nm} is not on the same per-ear similarity as the target ({e:.2e})"
    d0 = dd(pc).flatten(1).sort(-1).values; d1 = dd(b1["pc"][:, 0]).flatten(1).sort(-1).values
    assert float((d1 - s[:, None] * d0).abs().max()) < 1e-4
    assert float((b1["nrm"].norm(dim=-1) - 1).abs().max()) < 1e-5, "normals were scaled"
    print(f"  aug: cloud, coarse, est and target share one per-ear similarity "
          f"{[round(float(x), 6) for x in s]}; normals rotated only")
    try:
        endpoint_augment({**b0, "junk": torch.zeros(B, 3)}, tgt, cfg, ("nrm",), g)
        raise SystemExit("a stray batch tensor was silently left in the original frame")
    except AssertionError as e:
        assert "does not know how to move" in str(e), str(e)
    print("  refused a stray batch tensor the augmenter cannot move")
    # USE_NRM=0 has to survive the AUGMENTER, not merely forward/backward: endpoint_augment
    # asserts every ROTATES key is in the batch, so a hard-coded ("nrm",) asserted on the
    # first training step of that arm. NEEDS/ROTATES freeze at import, so the coupling is
    # asserted here and the batch that arm builds is augmented for real.
    assert MODEL.ROTATES == MODEL.NEEDS, f"ROTATES {MODEL.ROTATES} != NEEDS {MODEL.NEEDS}"
    endpoint_augment({k: v for k, v in b0.items() if k != "nrm"}, tgt, cfg, (), g)
    print(f"  ROTATES {MODEL.ROTATES} tracks NEEDS; the USE_NRM=0 batch (no 'nrm') augments")

    print("\nSMOKE 3/4 -- measured facts on the real artefacts")
    analysis()

    print("\nSMOKE 4/4 -- per-endpoint error before/after on TRAINING ears")
    real_smoke()
    print(f"\nSMOKE PASS ({time.time()-t0:.0f}s)")
    print("=" * 78)
    if os.environ.get("SMOKE_TRAIN"):
        print("\n--- train_family.py end-to-end (SMOKE_TRAIN=1) ---")
        trainer_smoke()


if __name__ == "__main__":
    if os.environ.get("BUILD_EST"):
        print("building per-fold `est` artefacts from the OOF ensemble")
        build_est(os.environ.get("WORK", "scratch"))
    else:
        _smoke()
