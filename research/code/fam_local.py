"""
FAMILY `local` -- the CASCADE refinement stage. A small, weight-shared, per-landmark
network that reads one high-resolution local patch (build_local_crops.py) and moves the
current prediction by a bounded residual.

WHAT IS DIFFERENT ABOUT THIS FAMILY
-----------------------------------
Every other family in train_family.py's REGISTRY sees the whole ear once, at 1.09-2.19 mm
point spacing, and emits 85 landmarks. This one sees 85 separate patches at a measured
0.2115 mm (core) / 0.6829 mm (surround) spacing and emits 85 corrections. It does not
compete with the global stage, it composes with it: `pred = centre + residual`, and
`centre` is the existing pipeline's own output. Turning it off (residual = 0) reproduces
the 1.1776 mm baseline exactly, which is the property that makes it safe to ship.
VERIFIED END TO END on the real artefact and the real split, two runs, both reproducible:
  FAMILY=local FOLD=0 EPOCHS=1 TTA=1 FULL_EVAL=0 CFG_LM_PER_STEP=16 CFG_BS=8
      -> raw MLE 1.2066 mm on fold 0's 68 validation ears (baseline 1.1872 on those ears)
  ... CFG_HEAD=offset CFG_LR=0.0        -> raw MLE 1.1872 mm, EXACTLY the baseline.
The second is the decisive one: an identity refiner driven through the whole harness --
crop artefact, BATCH hook, forward, canonical->world, evaluate -- reproduces the ensemble
to 4 decimals, so ear order, local frame and composition are all correct. The 0.019 mm in
the first is the heatmap head's initialisation drift (mean move 0.17 mm, see the prior
table in the constructor); a frame, ear-order or pipeline mismatch lands at 3-20 mm.

INPUT CONTRACT, and how the current estimate enters
---------------------------------------------------
Everything comes through cls.BATCH from $LOCAL_CROPS (default scratch/local_crops.npz),
never through the DATA npz, because a (E,85,N,3) tensor is neither ear-indexed nor
sample-indexed and train_family.py's make_batch would slice its landmark axis as if it
were the fresh-sample axis. For the same reason every key handed back is at most 3-D:
`_flatten_samples` slices [:,0] off anything with dim >= 4.

  crop   (B, 85*N, 3)  patch points, LOCAL frame, mm      -> view(B,85,N,3)
  cnrm   (B, 85*N, 3)  patch normals, LOCAL frame, unit
  qc     (B, 85, 3)    THE CURRENT ESTIMATE, canonical frame -- the patch centre
  qf     (B, 85*3, 3)  rows (t,b,n) of the local frame, canonical -> view(B,85,3,3)
  npatch ()            N, so the views survive augmentation subsampling

The current estimate enters TWICE and in exactly two ways, both of them explicit:
 1. as the ORIGIN of the patch coordinates -- every point is (x - qc) @ frame^T, so the
    network sees the surface *relative to where the global stage thinks the landmark is*,
    which is the whole reason a local crop is informative;
 2. as the base of the output, pred = qc + rho @ frame.
It is never a free-floating feature, so the network cannot learn to ignore the centre and
regress an absolute position -- it has no absolute position to regress to.

OUTPUT: A HEATMAP, NOT A FREE OFFSET (HEAD=heatmap, the default)
----------------------------------------------------------------
The head scores every patch point and soft-argmaxes:
    rho = sum_i softmax(logit_i / temp) * q_i
Three reasons, in order of weight:
 * the answer is constrained to the convex hull of points that were sampled FROM THE
   SCANNED SURFACE, so the prediction cannot drift off the ear. The measured
   representation floor -- GT to the nearest sampled patch point -- is printed by
   build_local_crops.py --analyse and is the honest cost of that constraint;
 * it is bounded by construction, |rho| <= CROP_R, with no tanh to saturate;
 * the gradient is a competition between points rather than a regression to a mean, which
   is what makes heatmaps beat coordinate regression in landmark detection generally.
Caveat, stated rather than hidden: a convex combination of surface points is INSIDE the
hull, not on the surface. On a patch this small the sag is sub-micron where the surface is
flat and grows with curvature; the existing exact surface projection removes it for free
(the composition below re-projects), so this is not a reason to prefer a free offset.
HEAD=offset is the ABLATION the docstring owes you: rho = rmax * tanh(Linear(global)).
rmax defaults to 6.0 mm, from the measured residual distribution of the 1.1776 mm
ensemble over 28900 (ear,landmark) pairs -- p90 2.33, p99 4.57, p99.9 7.13, max 12.28 --
so 6.0 mm can express 99.72% of the corrections while keeping |rho|/rmax < 0.4 for 95% of
them, i.e. well inside tanh's linear region. It is an ablation, not the default: the
comparison heatmap-vs-offset at equal width and equal epochs is what decides whether the
surface constraint is worth its representation floor, and IT HAS NOT BEEN RUN.

CONDITIONING: SHARED WEIGHTS + LANDMARK IDENTITY (COND=id, the default)
-----------------------------------------------------------------------
One network serves all 85 landmarks -- 85 separate networks on 272 training ears would
be 272 examples each, and the local appearance of neighbouring landmarks on one contour
is nearly the same problem. Specialisation comes from a learned embedding per landmark
index, concatenated to every point feature (so it can steer the heatmap, not just bias the
output). COND=contour uses a 4-way contour embedding, COND=both uses their sum, and
HEADS=percontour additionally gives each contour its own final scoring layer.
  IS THE EMBEDDING ENOUGH?  UNKNOWN -- this model has never been trained. What is known
  from this repo: the four contours differ by 1.6x in error (concha 0.92, inner helix
  1.46) and the per-landmark floor is 0.566 mm in the concha versus 0.986-1.087 elsewhere,
  so the contours are genuinely different problems and pure weight sharing is optimistic.
  The decisive experiment is cheap and is one fold: COND=id vs COND=both vs
  HEADS=percontour at fixed width/epochs/seed. Anything under ~0.05 mm apart on one fold
  is noise (the measured fold sd is 0.0503 mm), so this needs the 5-fold runs to settle.

COMPOSITION WITH THE EXISTING PIPELINE
--------------------------------------
train_family.py evaluates `pred` in the canonical frame, maps to world, and then runs
tta_mean -> surface projection -> (dense SSM, absent) -> reprojection. Since `pred` here
is `centre + correction` and `centre` is exactly ensemble5_proj's canonical form,
ordered_MLE_mm is directly comparable to 1.1776 mm and ordered_MLE_full_mm is the same
thing re-projected. The full-fold picture composes the same way: run this family on all
5 folds, assemble the OOF set, and the number is a drop-in replacement for
scratch/ensemble5_proj.npy. Nothing downstream changes.
  ONE HONEST DEPENDENCE: the crops are built around a FIXED prediction set. If the global
  stage changes, the crops must be rebuilt. This is a 16-minute local job (measured), but
  it means this family is not seed-ensemblable for free -- five refiner seeds all read the
  same centres, so they share the centre's error and only the correction's variance
  averages down.

COST -- STATED UP FRONT BECAUSE IT IS THE MAIN OBJECTION
--------------------------------------------------------
This is 85 forward passes per ear. At N=1024 that is 87k points per ear versus 8192 for
kpconv/ptv3 and 2048 for dgcnn: 10.6x and 42.5x more points. The network is far cheaper
per point (a plain shared MLP; no kNN graph, no attention, no kernel points), so the
honest estimate is NOT 85x, but it is not free either.
  measured baselines on the A6000, 272 training ears, 1200 epochs:
      dgcnn 2048      2.01 s/epoch      kpconv 8192  5.14 s/epoch   ptv3 8192  3.93 s/epoch
  arithmetic for this family at width 64, N=1024, all 85 landmarks per step:
      ~26 kFLOP/point forward -> 2.2 GFLOP/ear -> 0.61 TFLOP/epoch forward,
      ~1.8 TFLOP/epoch with the backward pass.
  MEASURED, real fold-0 epochs through train_family.py on 272 training ears, bs=8, 8 CPU
  threads (differencing a 3-epoch run against a 1-epoch run, so the data load and the
  evaluations cancel):
      lm_per_step=16   10.5 s/epoch          lm_per_step=0 (all 85)   74.5 s/epoch
  The ratio is 7.1x, not 85/16 = 5.3x, because BATCH gathers all 85 patches whatever K is;
  that gather is the fixed floor. An earlier draft of this docstring quoted 160 s/epoch at
  K=16 by scaling a bs=4/K=4 timing linearly in K -- wrong by 15x, and it is the
  measurement above that stands.
  CFG_LM_PER_STEP is still the lever: sample K of the 85 landmarks per training step
  (default 16). The weights are shared, so this is ordinary SGD over the landmark axis,
  not a restriction of the model. Evaluation always uses all 85.
  NOTHING IN THIS FAMILY HAS BEEN TIMED ON THE GPU BOX. But CPU is the pessimistic bound
  and even a 10x speedup puts K=85 at 7.5 s/epoch and K=16 at 1.1 s/epoch, i.e. between
  dgcnn (2.01) and kpconv (5.14) rather than 2-4x above them. A 5-fold 2-seed 1200-epoch
  sweep is therefore hours, not the 33-67 h an earlier draft feared.
  The cost of the lm_per_step trade is more epochs to reach the same number of
  landmark-gradient-steps; treat EPOCHS as needing to rise, not stay at 1200.

ENV
  LOCAL_CROPS  scratch/local_crops.npz   the artefact from build_local_crops.py
  CFG_* as usual. Family config:
    width 64            shared MLP width
    emb 32              landmark/contour embedding width
    head heatmap|offset      cond id|contour|both      heads shared|percontour
    temp 1.0            softmax temperature on the heatmap logits (mm^-1 scale free)
    rmax 6.0            tanh bound, HEAD=offset only (justified above)
    prior 1.0           Gaussian locality prior sigma, mm, learnable; 0 = off. Sets how
                        close to the identity the heatmap head starts -- measured table
                        in the constructor. HEAD=offset starts at the identity exactly.
    lm_per_step 16      landmarks sampled per training step; 0 = all 85
    core_r 3.0          must match the crop artefact; asserted against it
    sub_local 0.75      fraction of patch points kept per training step
    jit_local 0.05      per-point jitter mm         cjit 0.35   centre jitter mm
    delta_w 0.0         optional L2 pull on the correction, mm^-2 (0 = off)

  FAMILY=local FOLD=0 SEED=0 EPOCHS=2000 CFG_LM_PER_STEP=16 python3 train_family.py
  python research/code/fam_local.py        # CPU smoke test
"""
import os, time
import numpy as np
import torch
import torch.nn as nn

NL, NFOLD = 85, 5
CONTOURS = [(0, 24), (25, 54), (55, 74), (75, 84)]
CID = np.zeros(NL, np.int64)
for _c, (_a, _b) in enumerate(CONTOURS):
    CID[_a:_b + 1] = _c

LOCAL_CROPS = os.environ.get("LOCAL_CROPS", "scratch/local_crops.npz")
_CACHE = {}


def crops(dev):
    """Load the crop artefact once, resident on `dev`. fp16 on GPU, fp32 on CPU.

    Kept in fp16 because that is how it is stored and 355 MB fits; the per-batch gather
    casts to fp32, so nothing downstream sees a half-precision activation.
    """
    if dev in _CACHE:
        return _CACHE[dev]
    z = np.load(LOCAL_CROPS)
    n = int(z["npts"])
    half = dev != "cpu"
    d = dict(pts=torch.tensor(z["pts"]).to(dev, torch.float16 if half else torch.float32),
             nrm=torch.tensor(z["nrm"]).to(dev, torch.float16 if half else torch.float32),
             centre=torch.tensor(z["centre"]).float().to(dev),
             frame=torch.tensor(z["frame"]).float().to(dev),
             coarse=torch.tensor(z["coarse"]).float().to(dev),
             npts=n, crop_r=float(z["crop_r"]), core_r=float(z["core_r"]),
             pred_path=str(z["pred_path"]), frame_mode=str(z["frame_mode"]))
    assert d["pts"].shape[1] == NL and d["pts"].shape[2] == n, d["pts"].shape
    # a landmark-axis / sample-axis mix-up is the one bug that passes every shape check
    assert d["centre"].shape[1] == NL and d["frame"].shape[1:] == (NL, 3, 3)
    det = torch.linalg.det(d["frame"])
    assert float((det - 1).abs().max()) < 1e-4, f"crop frames are not proper rotations {det.min()}"
    print(f"[fam_local] {LOCAL_CROPS}: {tuple(d['pts'].shape)} patches, R={d['crop_r']} "
          f"core={d['core_r']} frame={d['frame_mode']} centres={d['pred_path']}", flush=True)
    _CACHE[dev] = d
    return d


# ------------------------------------------------------------------ batch hook
def BATCH(ears, samples, meta):
    """cls.BATCH: gather the (ear, 85) patches. Every value is at most 3-D on purpose."""
    d = crops(meta["dev"])
    e = torch.as_tensor(np.asarray(ears), dtype=torch.long, device=d["centre"].device)
    B, N = len(e), d["npts"]
    return {"crop": d["pts"][e].float().reshape(B, NL * N, 3),
            "cnrm": d["nrm"][e].float().reshape(B, NL * N, 3),
            "qc": d["centre"][e],
            "qf": d["frame"][e].reshape(B, NL * 3, 3),
            "qcoarse": d["coarse"][e],
            "npatch": torch.tensor(N)}


# ------------------------------------------------------------------ augmentation
def local_augment(b, tg, cfg, rotates, gen):
    """Augment IN THE LOCAL FRAME. No global rotation or scale -- and that is deliberate.

    The patch is already expressed in a frame the builder derives deterministically at
    inference time too, so rotating it would teach invariance the deployed model never
    needs; and a global isotropic scale cannot be applied consistently (it would have to
    rescale the frame, the centre, the target and the patch, which is just a change of
    units the network already sees through q/crop_r).

    What DOES vary between training and deployment is how good the centre is, so that is
    what is augmented: displace the centre by delta in local coordinates and translate the
    patch by -delta. The pair (patch, centre) stays exactly consistent -- the same surface,
    described from a slightly different origin -- and the absolute target `tg` is
    untouched, so this simulates a worse global stage with no bookkeeping. Cost: the patch
    loses a delta-wide rim of its far side, which at cjit=0.35 mm against R=8 mm is 4%.
    Plus the usual point subsample and a small per-point jitter.
    """
    B = b["qc"].shape[0]
    N = int(b["npatch"])
    dev = b["qc"].device
    q = b["crop"].view(B, NL, N, 3)
    n = b["cnrm"].view(B, NL, N, 3)
    Fr = b["qf"].view(B, NL, 3, 3)
    out = dict(b)
    dl = torch.randn(B, NL, 1, 3, device=dev, generator=gen) * cfg.get("cjit", 0.35)
    q = q - dl
    ns = max(16, int(round(N * cfg.get("sub_local", 0.75))))
    if ns < N:
        sub = torch.rand(B, NL, N, device=dev, generator=gen).argsort(-1)[..., :ns]
        g3 = sub[..., None].expand(-1, -1, -1, 3)
        q, n = torch.gather(q, 2, g3), torch.gather(n, 2, g3)
    j = cfg.get("jit_local", 0.05)
    if j:
        q = q + torch.randn(q.shape, device=dev, generator=gen) * j
    out["crop"] = q.reshape(B, NL * ns, 3)
    out["cnrm"] = n.reshape(B, NL * ns, 3)
    # local -> canonical is delta @ frame (rows of `frame` are t,b,n), NOT frame @ delta.
    # Both shapes typecheck and the transposed version is a silent frame corruption, which
    # is what SMOKE 5 exists to catch.
    out["qc"] = b["qc"] + torch.einsum("blij,bli->blj", Fr, dl[:, :, 0])
    out["npatch"] = torch.tensor(ns)
    return out, tg


# ------------------------------------------------------------------ model
def mlp(*w):
    L = []
    for a, b in zip(w[:-1], w[1:]):
        L += [nn.Linear(a, b), nn.GELU()]
    return nn.Sequential(*L[:-1])


class LocalRefiner(nn.Module):
    DEFAULTS = dict(width=64, emb=32, head="heatmap", cond="id", heads="shared",
                    temp=1.0, rmax=6.0, prior=1.0, lm_per_step=16, core_r=3.0,
                    crop_r=8.0, sub_local=0.75, jit_local=0.05, cjit=0.35,
                    delta_w=0.0, lr=1e-3, bs=8)
    SEARCH_SPACE = dict(width=[48, 64, 96], emb=[16, 32], temp=[0.5, 1.0, 2.0],
                        head=["heatmap", "offset"], cond=["id", "both"],
                        heads=["shared", "percontour"], lm_per_step=[8, 16, 32],
                        prior=[1.0, 2.0, 4.0, 0.0], cjit=[0.0, 0.35, 0.7],
                        lr=[5e-4, 1e-3, 2e-3])
    NEEDS, ROTATES, SAMPLES = (), (), 1
    BATCH = staticmethod(BATCH)
    AUGMENT = staticmethod(local_augment)

    # 8 per-point channels: q/R (3), unit normal in the local frame (3), |q|/R (1),
    # and the stratum indicator 1[|q| < core_r] (1). The last one is not cosmetic: the
    # sampler is deliberately denser inside core_r, so a heatmap that did not know where
    # the density step is would read it as evidence.
    CIN = 8

    def __init__(self, cfg, meta):
        super().__init__()
        c = {**self.DEFAULTS, **cfg}
        self.cfg = c
        w, e = int(c["width"]), int(c["emb"])
        self.head_mode, self.cond = c["head"], c["cond"]
        self.per_contour = c["heads"] == "percontour"
        self.temp, self.rmax = float(c["temp"]), float(c["rmax"])
        # crop_r / core_r are NORMALISERS and the core indicator's threshold, so a config
        # value that disagrees with the artefact silently rescales every input channel.
        # The artefact wins; the config value only supplies a default for the smoke test.
        self.core_r, self.crop_r = float(c["core_r"]), float(c["crop_r"])
        if os.path.exists(LOCAL_CROPS):
            z = np.load(LOCAL_CROPS, mmap_mode="r")
            self.crop_r, self.core_r = float(z["crop_r"]), float(z["core_r"])
            if (abs(self.crop_r - float(c["crop_r"])) > 1e-6 or
                    abs(self.core_r - float(c["core_r"])) > 1e-6):
                print(f"[fam_local] crop_r/core_r taken from {LOCAL_CROPS}: "
                      f"{self.crop_r}/{self.core_r} (config said "
                      f"{c['crop_r']}/{c['core_r']})", flush=True)
        self.lm_per_step, self.delta_w = int(c["lm_per_step"]), float(c["delta_w"])
        self.register_buffer("cid", torch.tensor(CID))
        # LOCALITY PRIOR on the heatmap: logit -= 0.5*(|q|/sigma)^2, sigma learnable.
        # Without it a zero-initialised score layer gives a uniform softmax, i.e. rho = the
        # patch centroid, and the refiner STARTS by making the 1.1776 mm baseline worse.
        # Measured |rho| at init on 510 real patches (scratch/local_crops_lim6.npz):
        #     sigma   0.25    0.5    1.0    2.0    4.0   uniform
        #     mean   0.060  0.071  0.178  0.356  0.673   1.109  mm
        #     p90    0.097  0.113  0.288  0.531  0.946   1.557  mm
        # sigma = 1.0 mm is the default: 0.18 mm of init drift against a 1.1776 mm baseline,
        # while a p90 correction of 2.33 mm costs only 0.5*(2.33/1)^2 = 2.7 logits for the
        # score layer to overcome, which is nothing. Tighter is a better warm start and a
        # stiffer model; sigma is a PARAMETER, so this only sets where the search begins.
        # prior=0 disables it and reproduces the 1.1 mm uniform-softmax start.
        self.prior = float(c["prior"])
        self.log_sigma = nn.Parameter(torch.tensor(float(np.log(max(self.prior, 1e-6)))))
        self._checked = False
        if c["cond"] in ("id", "both"):
            self.emb_id = nn.Embedding(NL, e)
        if c["cond"] in ("contour", "both"):
            self.emb_ct = nn.Embedding(len(CONTOURS), e)
        self.enc = mlp(self.CIN + e, w, w)
        self.mix = mlp(2 * w, w, w)
        nh = len(CONTOURS) if self.per_contour else 1
        self.score = nn.ModuleList([nn.Linear(w, 1) for _ in range(nh)])
        self.off = nn.ModuleList([nn.Linear(w, 3) for _ in range(nh)])
        # Start as the identity refiner. MEASURED consequence at step 1 with head=heatmap:
        # only score.weight (64 params) and log_sigma receive gradient -- enc, mix and the
        # embedding get exactly 0.0, because dL/dh is score.weight^T * (...) = 0. It
        # unblocks after one optimiser step. score.bias is permanently dead (softmax is
        # shift-invariant); harmless, 1 parameter.
        for m in list(self.score) + list(self.off):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def embed(self, ids):
        if self.cond == "id":
            return self.emb_id(ids)
        if self.cond == "contour":
            return self.emb_ct(self.cid[ids])
        return self.emb_id(ids) + self.emb_ct(self.cid[ids])

    def forward(self, batch):
        qc, Fr = batch["qc"], batch["qf"]
        B = qc.shape[0]
        if "qcoarse" in batch and not self._checked:
            # EAR ORDER AND FRAME, proved rather than assumed. `coarse` is bit-identical in
            # deep_dataset.npz and every screen_data_*.npz, and the crop artefact ships a
            # copy, so exact equality here means the crop npz and the DATA npz index the
            # same ears in the same canonical frame. A wrong ear order otherwise passes
            # every shape check and just trains against the wrong surface.
            e = float((batch["qcoarse"] - batch["coarse"]).abs().max())
            assert e < 1e-4, (
                f"crop artefact and DATA npz disagree on the coarse landmarks (max {e:.4g} "
                f"mm) -- different ear order, different canonical frame, or a crop built "
                f"from a different dataset. Refusing to train.")
            self._checked = True
        N = int(batch["npatch"])
        Fr = Fr.view(B, NL, 3, 3)
        q = batch["crop"].view(B, NL, N, 3)
        nq = batch["cnrm"].view(B, NL, N, 3)
        ids = torch.arange(NL, device=qc.device)
        if self.training and 0 < self.lm_per_step < NL:
            ids = torch.randperm(NL, device=qc.device)[:self.lm_per_step].sort().values
            q, nq = q[:, ids], nq[:, ids]
        L = len(ids)
        r = q.norm(dim=-1, keepdim=True)
        x = torch.cat([q / self.crop_r, nq, r / self.crop_r,
                       (r < self.core_r).to(q.dtype)], -1)
        em = self.embed(ids)[None, :, None].expand(B, L, N, -1)
        h = self.enc(torch.cat([x, em], -1))
        h = self.mix(torch.cat([h, h.amax(2, keepdim=True).expand_as(h)], -1))
        ci = self.cid[ids]
        if self.head_mode == "heatmap":
            lg = torch.stack([m(h)[..., 0] for m in self.score], -1)     # (B,L,N,nh)
            lg = lg.gather(3, ci[None, :, None, None].expand(B, L, N, 1))[..., 0] \
                if self.per_contour else lg[..., 0]
            if self.prior:
                lg = lg - 0.5 * (r[..., 0] / self.log_sigma.exp()) ** 2
            wgt = torch.softmax(lg / self.temp, -1)
            rho = torch.einsum("bln,blnc->blc", wgt, q)
        else:
            g = h.amax(2)                                                # (B,L,w)
            o = torch.stack([m(g) for m in self.off], -1)                # (B,L,3,nh)
            o = o.gather(3, ci[None, :, None, None].expand(B, L, 3, 1))[..., 0] \
                if self.per_contour else o[..., 0]
            rho = self.rmax * torch.tanh(o)
        pred = qc.clone()
        pred[:, ids] = qc[:, ids] + torch.einsum("blij,bli->blj", Fr[:, ids], rho)
        return {"pred": pred, "lm_sub": ids, "rho": rho}

    def loss(self, out, tg, batch):
        """Squared error on the REFINED landmarks only.

        The unrefined ones are pred == centre by construction: including them would add a
        constant with no gradient and silently rescale the learning rate by 85/K whenever
        lm_per_step changes. Same functional form as train_family.default_loss so the
        numbers are comparable.
        """
        i = out["lm_sub"]
        L = ((out["pred"][:, i] - tg[:, i]) ** 2).sum(-1).mean()
        return L + self.delta_w * (out["rho"] ** 2).sum(-1).mean() if self.delta_w else L


MODEL = LocalRefiner


# ------------------------------------------------------------------ smoke test
def _fake_batch(B, N, dev="cpu", seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    Fr = torch.linalg.qr(torch.randn(B, NL, 3, 3, generator=g))[0]
    Fr = Fr * torch.sign(torch.linalg.det(Fr))[..., None, None]      # proper rotation
    q = torch.randn(B, NL, N, 3, generator=g) * 2.5
    q = q.clamp(-8, 8)
    n = torch.randn(B, NL, N, 3, generator=g)
    n = n / n.norm(dim=-1, keepdim=True)
    co = torch.randn(B, NL, 3, generator=g)
    return {"pc": torch.randn(B, 128, 3, generator=g),
            "coarse": co, "qcoarse": co.clone(),
            "ear": torch.arange(B), "qc": torch.randn(B, NL, 3, generator=g) * 10,
            "crop": q.reshape(B, NL * N, 3), "cnrm": n.reshape(B, NL * N, 3),
            "qf": Fr.reshape(B, NL * 3, 3), "npatch": torch.tensor(N)}, Fr, q


def smoke():
    t0 = time.time()
    print("=" * 78)
    B, N = 2, 256
    meta = dict(nl=NL, contours=CONTOURS, scale=30.0, npts=2048, fold=0, dev="cpu",
                n_train_ears=16, artefacts={})
    batch, Fr, q = _fake_batch(B, N)
    tg = batch["qc"] + torch.randn(B, NL, 3) * 0.8

    print("SMOKE 1/7 -- forward/backward, both heads, both conditionings")
    for head in ("heatmap", "offset"):
        for cond in ("id", "contour", "both"):
            for hd in ("shared", "percontour"):
                net = MODEL({**MODEL.DEFAULTS, "head": head, "cond": cond, "heads": hd,
                             "lm_per_step": 0}, meta)
                out = net(batch)
                assert tuple(out["pred"].shape) == (B, NL, 3), out["pred"].shape
                loss = net.loss(out, tg, batch)
                loss.backward()
                gn = sum(float(p.grad.norm()) for p in net.parameters()
                         if p.grad is not None)
                assert np.isfinite(gn) and gn > 0, (head, cond, hd, gn)
                if cond == "id" and hd == "shared":
                    print(f"  head={head:8s} cond={cond:8s} {hd:11s} pred "
                          f"{tuple(out['pred'].shape)}  params "
                          f"{sum(p.numel() for p in net.parameters()):,}  "
                          f"loss {float(loss):.4f}  grad {gn:.3f}")
    print("  all 12 (head x cond x heads) combinations pass forward AND backward")

    print("\nSMOKE 2/7 -- at init the refiner must be ~the identity (= the 1.1776 baseline)")
    net = MODEL({**MODEL.DEFAULTS, "lm_per_step": 0}, meta)
    net.eval()
    with torch.no_grad():
        o = net(batch)
    d = float((o["pred"] - batch["qc"]).norm(dim=-1).max())
    n0 = MODEL({**MODEL.DEFAULTS, "lm_per_step": 0, "prior": 0.0}, meta)
    n0.eval()
    with torch.no_grad():
        d0 = float((n0(batch)["pred"] - batch["qc"]).norm(dim=-1).max())
    b1k, _, _ = _fake_batch(B, 1024, seed=1)
    with torch.no_grad():
        d1 = float((net(b1k)["pred"] - b1k["qc"]).norm(dim=-1).max())
        dm = float((net(b1k)["pred"] - b1k["qc"]).norm(dim=-1).mean())
    print(f"  heatmap, prior={MODEL.DEFAULTS['prior']}mm : max |pred - centre| {d:.4f} mm "
          f"at N=256, {d1:.4f} mm at N=1024 (mean {dm:.4f})")
    print(f"  heatmap, prior OFF        : max |pred - centre| {d0:.4f} mm at N=256  <- a "
          f"uniform softmax lands on the patch\n    centroid. The residue with the prior on "
          f"is Monte-Carlo noise of the prior-weighted\n    centroid and falls as 1/sqrt(N), "
          f"which the two N above show; it is NOT a bias.")
    assert d < d0 and d1 < d, (d, d0, d1)
    netf = MODEL({**MODEL.DEFAULTS, "head": "offset", "lm_per_step": 0}, meta)
    netf.eval()
    with torch.no_grad():
        of = netf(batch)
    df = float((of["pred"] - batch["qc"]).abs().max())
    print(f"  HEAD=offset  max |pred - centre| at init {df:.3e} mm  (tanh(0) = 0, so this "
          f"one starts EXACTLY at the 1.1776mm baseline)")
    assert df < 1e-6, df

    print("\nSMOKE 3/7 -- the residual is bounded, and by what")
    for _ in range(30):                       # push the heatmap to an extreme
        with torch.no_grad():
            for m in net.score:
                nn.init.normal_(m.weight, std=8.0); nn.init.normal_(m.bias, std=8.0)
        with torch.no_grad():
            rho = net(batch)["rho"]
    rmax_seen = float(rho.norm(dim=-1).max())
    qmax = float(q.norm(dim=-1).max())
    print(f"  heatmap: max |rho| {rmax_seen:.4f} mm <= max |q| {qmax:.4f} mm "
          f"(convex hull of the sampled surface -- no tanh needed)")
    assert rmax_seen <= qmax + 1e-4
    with torch.no_grad():
        for m in netf.off:
            nn.init.normal_(m.weight, std=50.0); nn.init.normal_(m.bias, std=50.0)
        rf = netf(batch)["rho"]
    print(f"  offset : max |rho| {float(rf.norm(dim=-1).max()):.4f} mm <= sqrt(3)*rmax "
          f"{np.sqrt(3)*MODEL.DEFAULTS['rmax']:.4f} mm (tanh bound, rmax="
          f"{MODEL.DEFAULTS['rmax']})")
    assert float(rf.abs().max()) <= MODEL.DEFAULTS["rmax"] + 1e-4

    print("\nSMOKE 4/7 -- composition: pred = centre + rho @ frame, in the canonical frame")
    net2 = MODEL({**MODEL.DEFAULTS, "lm_per_step": 0}, meta)
    with torch.no_grad():
        for m in net2.score:
            nn.init.normal_(m.weight, std=2.0)
        o2 = net2(batch)
    rec = batch["qc"] + torch.einsum("blij,bli->blj", Fr, o2["rho"])
    print(f"  max |pred - (centre + rho @ frame)| {float((o2['pred'] - rec).abs().max()):.3e}")
    assert float((o2["pred"] - rec).abs().max()) < 1e-4
    # and the heatmap answer must be a convex combination of the patch points
    dmin = (o2["rho"][:, :, None] - q).norm(dim=-1).min(-1).values
    print(f"  distance from rho to the nearest patch point: max {float(dmin.max()):.4f} mm "
          f"(0 would mean hard argmax; this is the soft interpolation)")

    print("\nSMOKE 5/7 -- augmentation keeps (patch, centre) consistent")
    g = torch.Generator(device="cpu").manual_seed(7)
    cfg = {**MODEL.DEFAULTS, "sub_local": 1.0, "jit_local": 0.0, "cjit": 0.5}
    b2, tg2 = local_augment(batch, tg, cfg, (), g)
    # a point's ABSOLUTE canonical position must be unchanged by the centre jitter
    N2 = int(b2["npatch"])
    a0 = batch["qc"][:, :, None] + torch.einsum(
        "blij,blni->blnj", Fr, batch["crop"].view(B, NL, N, 3))
    a1 = b2["qc"][:, :, None] + torch.einsum(
        "blij,blni->blnj", Fr, b2["crop"].view(B, NL, N2, 3))
    print(f"  centre moved by up to {float((b2['qc']-batch['qc']).norm(dim=-1).max()):.4f} mm; "
          f"max change in a patch point's ABSOLUTE position {float((a1-a0).abs().max()):.2e} mm")
    assert float((a1 - a0).abs().max()) < 1e-4, "augmentation broke patch/centre consistency"
    assert torch.equal(tg2, tg), "the absolute target must not move"
    cfg2 = {**cfg, "sub_local": 0.5}
    b3, _ = local_augment(batch, tg, cfg2, (), g)
    print(f"  sub_local=0.5 -> npatch {int(b3['npatch'])} of {N}; shapes "
          f"{tuple(b3['crop'].shape)}; forward still fine: "
          f"{tuple(net(b3)['pred'].shape)}")
    assert int(b3["npatch"]) == N // 2 and tuple(net(b3)["pred"].shape) == (B, NL, 3)

    print("\nSMOKE 6/7 -- a crop artefact that disagrees on the ear order is REFUSED")
    bad = dict(batch); bad["qcoarse"] = batch["qcoarse"].flip(0)     # two ears swapped
    try:
        MODEL({**MODEL.DEFAULTS, "lm_per_step": 0}, meta)(bad)
        raise SystemExit("a mismatched crop artefact was ACCEPTED -- wrong-surface training")
    except AssertionError as ex:
        msg = str(ex)
    assert "ear order" in msg, msg
    print(f"  swapping two ears' coarse landmarks raises: {msg[:88]}...")

    print("\nSMOKE 7/7 -- lm_per_step, and a CPU cost measurement")
    nk = MODEL({**MODEL.DEFAULTS, "lm_per_step": 16}, meta)
    nk.train()
    o = nk(batch)
    assert len(o["lm_sub"]) == 16 and tuple(o["pred"].shape) == (B, NL, 3)
    keep = torch.ones(NL, dtype=torch.bool); keep[o["lm_sub"]] = False
    assert float((o["pred"][:, keep] - batch["qc"][:, keep]).abs().max()) == 0.0, \
        "unrefined landmarks must pass the centre through untouched"
    nk.eval()
    assert len(nk(batch)["lm_sub"]) == NL, "evaluation must refine all 85"
    print(f"  train: {len(o['lm_sub'])} of {NL} landmarks refined, the other "
          f"{int(keep.sum())} pass the centre through unchanged; eval: all {NL}")

    b1024, _, _ = _fake_batch(1, 1024)
    net.train()
    torch.set_num_threads(min(8, os.cpu_count() or 4))
    for lm in (85, 16):
        m = MODEL({**MODEL.DEFAULTS, "lm_per_step": 0 if lm == 85 else lm}, meta)
        m.train()
        m(b1024)["pred"].sum().backward()             # warm up
        t = time.time()
        for _ in range(3):
            m.zero_grad()
            o = m(b1024)
            m.loss(o, torch.randn(1, NL, 3), b1024).backward()
        dt = (time.time() - t) / 3
        print(f"  CPU fwd+bwd, 1 ear x {lm:2d} landmarks x 1024 pts: {dt*1000:7.1f} ms  "
              f"-> {dt*272:6.1f} s/epoch on {torch.get_num_threads()} CPU threads "
              f"(GPU will be 30-100x faster; NOT a GPU measurement)")
    p = sum(x.numel() for x in net.parameters())
    print(f"  params {p:,} (vs dgcnn 813,616 / kpconv 1,044,196 / ptv3 1,857,520)")
    print(f"SMOKE PASS ({time.time()-t0:.0f}s)")
    print("=" * 78)


if __name__ == "__main__":
    smoke()
