"""
FAMILY: PROFILE DECODER — landmarks placed by ARC-LENGTH POSITION, per contour.

WHY. The normalised cumulative chord-length profile of the GT landmarks is nearly
CONSTANT across subjects on two of the four contours (measured on the 340 development
ears; `arc_profile` below is the exact definition used):

    contour          n    length    sd(profile)    sd in mm    gap CV
    outer_helix     25   113.4mm      0.0153         1.74       0.075
    concha          30    82.4mm      0.0162         1.33       0.105
    inner_helix     20    50.5mm      0.0064         0.32       0.035
    sup._antihelix  10    16.8mm      0.0073         0.12       0.037

Those two contours (30 of the 85 landmarks) currently carry 1.4859 and 1.2051 mm of
error while their PHASE is determined by the curve to 0.32 and 0.12 mm. So the phase of
those landmarks is not something to be regressed — it is something to be IMPOSED. This
family imposes it: the network predicts a curve, and the landmarks are the points at the
population arc-length positions along that curve.

    landmark_i = point at normalised arc length s_i along the predicted polyline

The other two contours keep the ordinary free-XYZ head, because their profile is NOT
constant: an oracle that forces the uniform profile on them costs -146% and -224%.
PROFILE_CONTOURS therefore defaults to "2,3" and a run that sets it to "0,1,2,3" is
expected to be much worse.

MEASURED BEFORE YOU SPEND A GPU RUN ON THIS — research/results/profile_apply.json applies
the placement operator alone, post hoc and fold-safely, to the 1.1952 mm equal-weight OOF
ensemble (dgcnn3 + kpconv + ptv3, pre-projection). Mean error of the 30 profile landmarks:

    baseline                                             1.3923
    fold-mean profile, the model's OWN endpoints         1.4086   (+0.0163, i.e. HARMFUL)
    LEAKY  GT endpoint landmarks substituted             1.1809   (-0.2114)
    LEAKY  + interior carried along (reposition)         0.8769   (-0.3040)
    LEAKY  + population profile replaces the phase       0.8266   (-0.0503)

Two things follow, and the second is the uncomfortable one.

  * The operator is correct: the last row reproduces the independently measured oracle
    (inner_helix 1.4859 -> 1.0724, sup._antihelix 1.2051 -> 0.3336) to 4 decimals.
  * Of the 0.566 mm the oracle recovers on those 30 landmarks, the ARC-LENGTH PROFILE
    contributes 0.050 mm — 8.9%. The other 91% is the ENDPOINTS. And with the ensemble's
    own endpoints (off by 1.585 mm) the placement is worth +0.0057 mm pooled over 85,
    per-subject bootstrap CI [+0.0005, +0.0110], i.e. significantly harmful.
    The endpoint-noise sweep in the same file puts the break-even for repositioning at
    about 1.0 mm of MEAN EUCLIDEAN independent endpoint error, i.e. 1.6x better than the
    ensemble manages. (The sweep is parameterised by the per-axis Gaussian sd sigma;
    break-even sits near sigma=0.63, and E||N(0,sigma^2 I_3)|| = 1.5958*sigma, so reading
    sigma against the 1.585 mm mean-norm figure would overstate the bar by 1.6x. The
    sweep prints both columns.)

So this family is worth training, but it is not the lever on its own; it is the consumer
of one. Its value is conditional on fam_endpoint.py, and the interface below exists for
exactly that. Caveat on the caveat: those numbers are post hoc on predictions trained
WITHOUT the placement, and the sweep's noise is independent of the curve's own error,
which is right for an external endpoint predictor and pessimistic for one sharing this
backbone. A model trained end to end with the placement in the graph can put its curve
somewhere else, which is the experiment run_profile_probe.sh runs.

WHAT THIS IS NOT — read fam_phase.py before changing anything here. Family F predicted a
16-control-point centripetal Catmull-Rom per contour from ONE pooled vector (global max-
pool + contour-pooled features + the flattened coarse curve) and scored 1.8096 mm against
the 1.2652 mm baseline on the same fold. research/results/family_F.json falsified the
"reduced-rank curve floor" reading of that number: raising NCTRL 16 -> 24 -> 32 made it
WORSE (1.8096 -> 1.8559 -> 1.8883) and the damage was uniform across contours (1.40-1.56x)
rather than tracking the measured floor. The cause was UNDER-CONDITIONING. Two consequences
are load-bearing here:

  1. PER-LANDMARK LOCAL CONDITIONING. The curve is the baseline's own output: the shipped
     813,232-param DGCNN with 4 tied offset-and-snap passes, each of which does a K=48 kNN
     gather AT EVERY LANDMARK's current position, plus the per-contour Conv1d smoother.
     Nothing is predicted from a pooled vector. With PROFILE_CONTOURS="" this module IS
     the baseline, parameter for parameter (813,232 at cin=3), so the one thing under test
     is the output operator.

  2. FULL DEGREES OF FREEDOM. The curve is the POLYLINE THROUGH THE n PREDICTED LANDMARK
     POSITIONS of the contour: K = n control points, 3n dof, the same as free XYZ.
     research/results/curve_floor.json measures the price of anything less — a cubic
     B-spline at ~60% dof already costs 0.7448 mm mean on inner_helix and 0.9351/0.9758 on
     outer_helix/concha, against a total budget of ~1.19 mm — and 0.0000 mm at full dof.
     A polyline through all n points pays exactly that 0.0000: `place(P, arc_profile(P))`
     reproduces P to floating-point accuracy, asserted in the smoke test. Smoothness is
     NOT the structure being imposed; arc-length position is, and it is imposed alone.

  The placement removes exactly (n-2) dof — the interior phases — which is precisely the
  quantity the population profile pins down to 0.32/0.12 mm.

ENDPOINTS ARE THE ANCHORS. s_0 = 0 and s_{n-1} = 1, so the two endpoint landmarks are the
polyline's ends and the placement never moves them; every interior landmark is measured
from them. They are also the worst-localised landmarks (1.35-1.82 mm). The interface for
an external endpoint predictor (fam_endpoint.py) is therefore first-class:

    batch['endpoints']   (B, 2*4, 3) float32, SAME canonical frame as `coarse`/`pc`
                         row 2*ci is contour ci's FIRST landmark (global index CONTOURS[ci][0]),
                         row 2*ci+1 its LAST (global index CONTOURS[ci][1]) — that is
                         [0,24,25,54,55,74,75,84] = EP_IDX = fam_endpoint.ENDS exactly, so
                         an (E,85,3) array from that family is also accepted and sliced.
                         Rows for contours not in PROFILE_CONTOURS are ignored.
                         Must be finite; there is no per-ear opt-out sentinel.

  * shape (B,P,3) deliberately: train_family._flatten_samples collapses axis 1 of ANY
    tensor with dim>=4 when SAMPLES==1, so a (B,4,2,3) tensor would be silently sliced to
    (B,2,3). Keep it 3-D.
  * supply it through cls.BATCH, which reads `endpoints` (E,8,3) out of the ARTEFACTS npz
    — so it inherits the trainer's fold/train_ear_mask proof. Ship it from fam_endpoint.py
    as `endpoints` in that npz and nothing here changes.
  * cls.AUGMENT is profile_augment, which applies the per-ear rotation and scale to those
    rows. default_augment would leave them in the un-augmented frame.
  * ENDPOINT_FIT says what is done with them: the polyline is repositioned so its ends land
    on the given endpoints (`similarity` = rotation+scale+translation from the 2-point
    correspondence, `rigid` = no scale, `translate` = shift onto the first endpoint only,
    `none` = ignore). This is the oracle's "rigidly repositioned predicted curve", which
    given GT endpoints and the uniform profile recovers sup._antihelix 1.2051 -> 0.3336
    and inner_helix 1.4859 -> 1.0724.

THE PROFILE ITSELF IS A POPULATION STATISTIC, so it is fold-scoped (constraint 2).
build_profile.py writes scratch/profile_f<FOLD>.npz from that fold's TRAINING ears only,
carrying `fold` and `train_ear_mask`; train_family.py refuses an artefact that cannot
prove it, and load_profile() adds the check the trainer cannot make — that the mask is
the WHOLE training fold, |mask| == meta['n_train_ears']. Without ARTEFACTS the UNIFORM
profile is used, which is fold-safe by construction (it reads no ear at all) and is
within 0.0007 / 0.0014 normalised = 0.035 / 0.024 mm of the measured population mean on
the two profile contours. PROFILE_FALLBACK=error refuses instead.

PROFILE_MODE=learned adds a small per-ear deviation, predicted from the SAME per-landmark
local features the contour smoother uses, bounded to the measured per-position sd
(DEV_BOUND scales that bound) and strictly monotone by construction. `fixed` adds ZERO
parameters.

    FAMILY=profile FOLD=0 SEED=0 DATA=scratch/screen_data_2048nrm.npz USE_NRM=1 \
        ARTEFACTS=scratch/profile_f0.npz python research/code/train_family.py
    python research/code/fam_profile.py                # CPU smoke test
    SMOKE_TRAIN=1 python research/code/fam_profile.py  # + train_family.py end-to-end

ENV (each is also reachable as CFG_<NAME>, which is how search_driver.py sweeps it):
  PROFILE_CONTOURS  "2,3"        contour indices placed by profile; "" = pure baseline
  PROFILE_MODE      fixed        fixed | learned
  PROFILE_FALLBACK  uniform      uniform | error, when no profile artefact is given
  DEV_BOUND         1.0          learned deviation bound, as a multiple of the measured sd
  DEV_LOGB          0.5          max |log| gap multiplier before the sd bound is applied
  ENDPOINT_FIT      similarity   similarity | rigid | translate | none
  EP_JIT            0.0          mm of augmentation jitter on supplied endpoints
  W_RAW             1.0          weight on the ordered MSE of the PRE-placement curve
  W_END             0.0          extra weight on the profile contours' endpoint landmarks.
                                 Default 0 keeps the arm a one-change experiment, but the
                                 measurement above says this is the knob with the most
                                 behind it: those 4 landmarks are 91% of the recoverable
                                 error on the contours they gate.
  WIDTH 256  K 48  GK 20  NPASS 4  UNTIED 0  DROPOUT 0.1  USE_NRM 0   (baseline backbone)
"""
import os
import numpy as np
import torch
import torch.nn as nn

NL, SCALE = 85, 30.0
CONTOURS = [(0, 24), (25, 54), (55, 74), (75, 84)]
CNAMES = ["outer_helix", "concha", "inner_helix", "sup._antihelix"]
NC = len(CONTOURS)
# population sd of the normalised profile, measured over the 340 development ears. Used
# ONLY as the default deviation bound when the artefact carries no per-position sd; it is
# an aggregate over all ears, so it is never allowed to set a POSITION (see load_profile).
PROFILE_SD = (0.0153, 0.0162, 0.0064, 0.0073)
# global landmark index of each contour's (first, last). Identical to fam_endpoint.py's
# ENDS, element for element, so its artefacts need no reordering here.
EP_IDX = [i for lo, hi in CONTOURS for i in (lo, hi)]


def _ilist(v):
    """"2,3" | [2,3] | 2 | "" | "none" -> list[int]"""
    if v is None or v == "" or v == [] or str(v).lower() in ("none", "-"):
        return []
    if isinstance(v, int):
        return [v]
    return [int(x) for x in (v.split(",") if isinstance(v, str) else v)]


ENV_DEFAULTS = dict(
    profile_contours=os.environ.get("PROFILE_CONTOURS", "2,3"),
    profile_mode=os.environ.get("PROFILE_MODE", "fixed"),
    profile_fallback=os.environ.get("PROFILE_FALLBACK", "uniform"),
    dev_bound=float(os.environ.get("DEV_BOUND", "1.0")),
    dev_logb=float(os.environ.get("DEV_LOGB", "0.5")),
    endpoint_fit=os.environ.get("ENDPOINT_FIT", "similarity"),
    ep_jit=float(os.environ.get("EP_JIT", "0.0")),
    w_raw=float(os.environ.get("W_RAW", "1.0")),
    w_end=float(os.environ.get("W_END", "0.0")),
    width=int(os.environ.get("WIDTH", "256")),
    k=int(os.environ.get("K", "48")),
    gk=int(os.environ.get("GK", "20")),
    npass=int(os.environ.get("NPASS", "4")),
    untied=int(os.environ.get("UNTIED", "0")),
    dropout=float(os.environ.get("DROPOUT", "0.1")),
    use_nrm=int(os.environ.get("USE_NRM", "0")),
)


def derive(cfg):
    c = dict(ENV_DEFAULTS, **(cfg or {}))
    for k in ("width", "k", "gk", "npass", "untied", "use_nrm"):
        c[k] = int(c[k])
    for k in ("dev_bound", "dev_logb", "ep_jit", "w_raw", "w_end", "dropout"):
        c[k] = float(c[k])
    c["prof"] = _ilist(c["profile_contours"])
    assert all(0 <= i < NC for i in c["prof"]), c["prof"]
    assert c["profile_mode"] in ("fixed", "learned"), c["profile_mode"]
    assert c["endpoint_fit"] in ("similarity", "rigid", "translate", "none"), c["endpoint_fit"]
    return c


# ------------------------------------------------------------------ the arc-length ops
def arc_profile(P):
    """Normalised cumulative CHORD length of an ordered polyline P (B,n,3) -> (B,n).

    THE definition: the population profile, the placement and every check in this file use
    this one function, so `place_on_polyline(P, arc_profile(P)) == P` exactly. Chord, not
    a spline's arc length — the GT landmarks are the polyline's own vertices, so the two
    agree at the vertices by construction and nothing has to be integrated.
    """
    seg = (P[:, 1:] - P[:, :-1]).norm(dim=-1)
    cum = torch.cat([torch.zeros_like(seg[:, :1]), seg.cumsum(-1)], -1)
    return cum / cum[:, -1:].clamp(min=1e-9)


def place_on_polyline(C, s):
    """The point at normalised arc length s (B,n) along the polyline C (B,K,3) -> (B,n,3).

    Piecewise-linear, differentiable in C and in s. s must be non-decreasing and in [0,1];
    the output is then ordered along C by construction, so the placement cannot reorder
    landmarks — the property fam_phase.py needed a softplus cumsum to obtain.
    """
    seg = (C[:, 1:] - C[:, :-1]).norm(dim=-1)
    cum = torch.cat([torch.zeros_like(seg[:, :1]), seg.cumsum(-1)], -1)
    q = s * cum[:, -1:]
    j = (torch.searchsorted(cum.contiguous(), q.contiguous()) - 1).clamp(0, C.shape[1] - 2)
    a, b = torch.gather(cum, 1, j), torch.gather(cum, 1, j + 1)
    f = ((q - a) / (b - a).clamp(min=1e-9))[..., None]
    g = lambda A, i: torch.gather(A, 1, i[..., None].expand(-1, -1, 3))
    return g(C, j) + f * (g(C, j + 1) - g(C, j))


def _skew(w):
    z = torch.zeros_like(w[..., 0])
    return torch.stack([torch.stack([z, -w[..., 2], w[..., 1]], -1),
                        torch.stack([w[..., 2], z, -w[..., 0]], -1),
                        torch.stack([-w[..., 1], w[..., 0], z], -1)], -2)


def reposition(C, e0, e1, mode="similarity"):
    """Move the polyline C (B,K,3) so that its two ENDS land on e0 / e1 (B,3).

    A 2-point correspondence fixes translation and scale but leaves the roll about the
    chord free; the minimal (Rodrigues) rotation taking C's chord direction to the target
    chord direction is used, which changes the curve's shape as little as a rotation can.
    Degenerate only at an exactly ANTI-parallel chord (1+cos -> 0, clamped); the endpoint
    predictor would have to reverse the contour for that, which the ordering forbids.
    """
    a, b = C[:, -1] - C[:, 0], e1 - e0
    na, nb = a.norm(dim=-1, keepdim=True).clamp(min=1e-9), b.norm(dim=-1, keepdim=True)
    X = C - C[:, :1]
    if mode != "translate":
        u, v = a / na, b / nb.clamp(min=1e-9)
        K = _skew(torch.cross(u, v, dim=-1))
        R = torch.eye(3, device=C.device, dtype=C.dtype) + K \
            + K @ K / (1 + (u * v).sum(-1)).clamp(min=1e-6)[:, None, None]
        X = torch.einsum("bkj,bij->bki", X, R)
    if mode == "similarity":
        X = X * (nb / na)[:, None]
    return X + e0[:, None]


# ------------------------------------------------------------------ the profile itself
def load_profile(meta, cfg):
    """Per-contour (mean profile, per-position sd) for THIS fold, and where it came from.

    LEAKAGE (constraint 2). train_family.py has already asserted that the ARTEFACTS npz
    carries `fold` and `train_ear_mask`, that the fold matches, and that no validation ear
    is in the mask. That still permits a mask built from a SUBSET of the training fold —
    or an old fixed-split artefact that happens to miss this fold's validation ears — so
    the check the trainer cannot make is made here: the mask must be the whole training
    fold. Together the two pin the mask to exactly this fold's training ears.

    The uniform fallback reads no ear at all, so it cannot leak; it is also within 0.035 mm
    (inner_helix) and 0.024 mm (sup._antihelix) of the measured population mean.
    """
    art = meta.get("artefacts") or {}
    if all(f"prof_c{ci}" in art for ci in range(NC)):
        assert int(art["fold"]) == meta["fold"], \
            f"profile artefact is for fold {int(art['fold'])}, this run is fold {meta['fold']}"
        m = np.asarray(art["train_ear_mask"]).astype(bool)
        assert int(m.sum()) == meta["n_train_ears"], (
            f"profile artefact was built from {int(m.sum())} ears but this fold trains on "
            f"{meta['n_train_ears']} -- it is not THIS fold's training set, so the profile "
            f"cannot be shown to be leakage-free. Rebuild it with build_profile.py.")
        prof = [np.asarray(art[f"prof_c{ci}"], dtype=np.float64) for ci in range(NC)]
        sd = [np.asarray(art[f"prof_sd_c{ci}"], dtype=np.float64) for ci in range(NC)]
        src = f"fold {meta['fold']} TRAINING ears only ({int(m.sum())} ears)"
    else:
        assert cfg["profile_fallback"] == "uniform", (
            "no ARTEFACTS npz with prof_c0..prof_c3 and PROFILE_FALLBACK=error: build one "
            "with research/code/build_profile.py and pass ARTEFACTS=scratch/profile_f<FOLD>.npz")
        prof = [np.linspace(0.0, 1.0, hi - lo + 1) for lo, hi in CONTOURS]
        sd = [np.full(hi - lo + 1, PROFILE_SD[ci]) for ci, (lo, hi) in enumerate(CONTOURS)]
        for v in sd:
            v[0] = v[-1] = 0.0
        src = "UNIFORM (no profile artefact; reads no ear, so fold-safe by construction)"
        if cfg["profile_mode"] == "learned":
            # CONSTRAINT-2 NOTE, kept in the run log rather than in a comment. The POSITIONS
            # here are uniform and read no ear, but the learned-deviation BOUND falls back to
            # PROFILE_SD, which is pooled over all 340 dev ears and so includes this fold's
            # validation ears. It sets the WIDTH of a permitted deviation, never a position,
            # and the fold-safe sds span it to within 0.046 mm; still, pass
            # ARTEFACTS=scratch/profile_f<FOLD>.npz and this path is not taken at all.
            src += " + learned bound from the POOLED all-340 sd (NOT fold-safe; <=0.046mm wide)"
    for ci, (lo, hi) in enumerate(CONTOURS):
        n = hi - lo + 1
        assert prof[ci].shape == (n,) and sd[ci].shape == (n,), \
            f"profile for contour {ci} has shape {prof[ci].shape}, expected ({n},)"
        assert abs(prof[ci][0]) < 1e-9 and abs(prof[ci][-1] - 1) < 1e-9, \
            f"profile for contour {ci} must run from 0 to 1, got {prof[ci][[0, -1]]}"
        assert (np.diff(prof[ci]) > 0).all(), f"profile for contour {ci} is not increasing"
    return prof, sd, src


class ProfileDecoder(nn.Module):
    """One contour's placement operator. ZERO parameters unless PROFILE_MODE=learned."""

    def __init__(self, s_mean, sd, cfg, fin):
        super().__init__()
        self.register_buffer("s_mean", torch.tensor(s_mean, dtype=torch.float32))
        self.register_buffer("sd", torch.tensor(sd * cfg["dev_bound"], dtype=torch.float32))
        self.learned = cfg["profile_mode"] == "learned"
        self.logb = cfg["dev_logb"]
        if self.learned:
            # per-landmark local features in, one gap multiplier per landmark out. A Conv1d
            # over the contour sequence, exactly like the baseline's contour smoother: each
            # output still sees its OWN landmark's kNN-gathered features, which is the
            # conditioning fam_phase.py did not have.
            self.head = nn.Sequential(nn.Conv1d(fin, 64, 5, padding=2), nn.ReLU(),
                                      nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
                                      nn.Conv1d(64, 1, 1))

    def phases(self, feat):
        """feat (B,n,F) -> s (B,n) strictly increasing, s[0]=0, s[-1]=1, |s-s_mean| <= sd."""
        s0 = self.s_mean[None].expand(feat.shape[0], -1)
        if not self.learned:
            return s0
        m = self.logb * torch.tanh(self.head(feat.transpose(1, 2))[:, 0, :-1])
        g = (s0[:, 1:] - s0[:, :-1]) * torch.exp(m)
        g = g / g.sum(-1, keepdim=True)
        s = torch.cat([torch.zeros_like(g[:, :1]), g.cumsum(-1)], -1)
        # bound the deviation to the measured sd. Scaling the WHOLE deviation by 1/r keeps
        # every gap a convex combination of two positive gaps, so monotonicity survives the
        # bound; clipping per position would not.
        d = s - s0
        r = (d[:, 1:-1].abs() / self.sd[None, 1:-1].clamp(min=1e-9)).amax(-1).clamp(min=1.0)
        return s0 + d / r[:, None]


# ------------------------------------------------------------------ baseline backbone
def knn(q, pc, k):
    return torch.cdist(q, pc).topk(k, largest=False, dim=-1).indices


def edge_conv(gidx, feat, mlp):
    B, P, C = feat.shape
    fj = torch.gather(feat, 1, gidx.reshape(B, -1, 1).expand(-1, -1, C)).view(B, P, -1, C)
    return mlp(torch.cat([feat[:, :, None].expand_as(fj), fj - feat[:, :, None]], -1)).amax(2)


class Head(nn.Module):
    """gpu_screen.py's refinement pass, verbatim: bounded-free offset then a surface snap,
    both driven by a K-nearest gather AT EACH LANDMARK's current position."""

    def __init__(self, C, K, dropout):
        super().__init__()
        self.emb, self.embO = nn.Embedding(NL, 32), nn.Embedding(NL, 32)
        self.offset = nn.Sequential(nn.Linear(2 * C + 32, 256), nn.ReLU(), nn.Dropout(dropout),
                                    nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 3))
        self.attn = nn.Sequential(nn.Linear(C + 3 + 32, 128), nn.ReLU(),
                                  nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.C, self.K = C, K

    def gather(self, pc, h, q):
        B = pc.shape[0]
        idx = knn(q, pc, self.K).reshape(B, NL * self.K)
        fK = torch.gather(h, 1, idx[..., None].expand(-1, -1, self.C)).view(B, NL, self.K, self.C)
        pK = torch.gather(pc, 1, idx[..., None].expand(-1, -1, 3)).view(B, NL, self.K, 3)
        return fK, pK

    def forward(self, pc, h, q):
        B = pc.shape[0]
        ar = torch.arange(NL, device=pc.device)
        fK, _ = self.gather(pc, h, q)
        off = self.offset(torch.cat([fK.mean(2), fK.amax(2),
                                     self.embO(ar)[None].expand(B, -1, -1)], -1))
        q1 = q + off
        fK2, pK2 = self.gather(pc, h, q1)
        rel = (pK2 - q1[:, :, None, :]) / SCALE
        e = self.emb(ar)[None, :, None, :].expand(B, NL, self.K, 32)
        w = torch.softmax(self.attn(torch.cat([fK2, rel, e], -1)).squeeze(-1), -1)
        return q1, (w[..., None] * pK2).sum(2)


class Net(nn.Module):
    """The shipped 813,232-param base model (gpu_screen.py VARIANT=base), unchanged, with
    the per-landmark feature tensor exposed so the profile decoder can condition on it."""

    def __init__(self, c, cin):
        super().__init__()
        C = c["width"]
        self.ec1 = nn.Sequential(nn.Linear(2 * cin, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.ec2 = nn.Sequential(nn.Linear(2 * 64, 128), nn.ReLU())
        self.ec3 = nn.Sequential(nn.Linear(2 * 128, 128), nn.ReLU())
        self.fuse = nn.Sequential(nn.Linear(320, C), nn.ReLU())
        self.mix = nn.Sequential(nn.Linear(2 * C, C), nn.ReLU())
        self.heads = nn.ModuleList([Head(C, c["k"], c["dropout"])
                                    for _ in range(c["npass"] if c["untied"] else 1)])
        self.lmfeat = nn.Sequential(nn.Linear(C, 64), nn.ReLU())
        self.contour_nets = nn.ModuleList([
            nn.Sequential(nn.Conv1d(3 + 64 + 32, 96, 5, padding=2), nn.ReLU(),
                          nn.Conv1d(96, 96, 3, padding=1), nn.ReLU(), nn.Conv1d(96, 3, 1))
            for _ in CONTOURS])
        self.embC = nn.Embedding(NL, 32)
        self.c = c

    def backbone(self, pc, ft):
        pos = pc / SCALE
        gidx = knn(pos, pos, self.c["gk"])
        x = pos if ft is None else torch.cat([pos, ft], -1)
        h1 = edge_conv(gidx, x, self.ec1)
        h2 = edge_conv(gidx, h1, self.ec2)
        h3 = edge_conv(gidx, h2, self.ec3)
        h = self.fuse(torch.cat([h1, h2, h3], -1))
        return self.mix(torch.cat([h, h.amax(1, keepdim=True).expand(-1, pc.shape[1], -1)], -1))

    def forward(self, pc, q0, ft=None):
        B, c = pc.shape[0], self.c
        h = self.backbone(pc, ft)
        outs, q = [], q0
        for i in range(c["npass"]):
            q1, q2 = self.heads[i if c["untied"] else 0](pc, h, q)
            outs.append((q1, q2)); q = q2
        f = self.lmfeat(torch.gather(h, 1, knn(q, pc, 1).expand(-1, -1, c["width"])))
        e = self.embC(torch.arange(NL, device=pc.device))[None].expand(B, -1, -1)
        inp = torch.cat([q / SCALE, f, e], -1)                    # (B,85,99) per-landmark
        d = torch.zeros_like(q)
        for (lo, hi), net in zip(CONTOURS, self.contour_nets):
            d[:, lo:hi + 1] = net(inp[:, lo:hi + 1].transpose(1, 2)).transpose(1, 2)
        return outs, q + d, inp


# ------------------------------------------------------------------ trainer plumbing
def profile_augment(b, tg, cfg, rotates, gen):
    """train_family.default_augment, extended to the 'endpoints' rows.

    default_augment transforms `pc`, `coarse`, the target and any (B,S,N,C) extra. A
    (B,8,3) endpoint tensor matches none of those, so it would stay in the UN-augmented
    frame while everything else is rotated and scaled, and the decoder would reposition
    its curve onto endpoints from a different frame. The random stream is consumed in
    default_augment's exact order (rand_rot, scale, subsample, cloud jitter, coarse
    jitter) and the endpoints are transformed afterwards, so with EP_JIT=0 this is
    bit-identical to default_augment on every key default_augment handles. The smoke test
    asserts that identity rather than trusting this comment.
    """
    from train_family import rand_rot
    pc = b["pc"]; B, S, N, _ = pc.shape; dev = pc.device
    R = rand_rot(B, cfg["aug_rot"], gen, dev)
    sc = 1 + (torch.rand(B, 1, 1, device=dev, generator=gen) - .5) * cfg["aug_scale"]
    nsub = max(8, min(N, int(round(N * cfg["sub_frac"]))))
    sub = torch.rand(B, S, N, device=dev, generator=gen).argsort(-1)[..., :nsub]
    rot = lambda t: torch.einsum("bnj,bij->bni", t, R) if t.dim() == 3 else \
        torch.einsum("bsnj,bij->bsni", t, R)
    out = dict(b)
    p = torch.gather(pc, 2, sub[..., None].expand(-1, -1, -1, 3))
    out["pc"] = rot(p) * sc[:, None] + torch.randn(p.shape, device=dev, generator=gen) * cfg["aug_jit"]
    done = set()
    for k, v in b.items():
        if k not in ("pc", "coarse", "ear") and torch.is_tensor(v) and v.dim() == 4 \
                and v.shape[:3] == pc.shape[:3]:
            v = torch.gather(v, 2, sub[..., None].expand(-1, -1, -1, v.shape[-1]))
            out[k] = rot(v) if k in rotates else v
            done.add(k)
    miss = [k for k in rotates if k in b and k not in done]
    assert not miss, f"cls.ROTATES names {miss}, which profile_augment cannot rotate: " \
                     f"shapes {[tuple(b[k].shape) for k in miss]}"
    out["coarse"] = rot(b["coarse"]) * sc + \
        torch.randn(b["coarse"].shape, device=dev, generator=gen) * cfg["aug_qjit"]
    if "endpoints" in b:
        ep = rot(b["endpoints"]) * sc
        j = cfg.get("ep_jit", 0.0)
        if j:
            ep = ep + torch.randn(ep.shape, device=dev, generator=gen) * j
        out["endpoints"] = ep
    return out, rot(tg) * sc


class FamProfile(nn.Module):
    """train_family.py's REGISTRY['profile'].

    NEEDS and AUGMENT are CLASS attributes the trainer reads BEFORE any instance exists,
    so normals must be requested through the ENVIRONMENT (USE_NRM=1); CFG_USE_NRM alone
    cannot make the trainer load 'nrm' and is rejected in __init__.
    """

    DEFAULTS = ENV_DEFAULTS
    SEARCH_SPACE = dict(profile_mode=["fixed", "learned"], dev_bound=[0.5, 1.0, 2.0],
                        dev_logb=[0.25, 0.5, 1.0], endpoint_fit=["similarity", "rigid", "none"],
                        w_raw=[0.0, 0.5, 1.0, 2.0], w_end=[0.0, 1.0, 3.0],
                        npass=[3, 4, 5], k=[32, 48, 64], gk=[16, 20, 28],
                        dropout=[0.0, 0.1, 0.2], untied=[0, 1], lr=[7e-4, 1.5e-3, 3e-3])
    NEEDS = ("nrm",) if ENV_DEFAULTS["use_nrm"] else ()
    ROTATES = ("nrm",)
    SAMPLES = 1
    AUGMENT = staticmethod(profile_augment)

    @staticmethod
    def BATCH(ears, samples, meta):
        """Hand the per-ear endpoints to forward(), if the fold-safe artefact carries them.

        This is the whole fam_endpoint.py interface: write `endpoints` into the same npz
        that carries `fold` / `train_ear_mask`, point ARTEFACTS at it, and the decoder
        repositions each profile contour onto those endpoints. CANONICAL frame, as
        fam_endpoint.build_est already produces (p_canon = (p_world - c0) @ R^T).

        (E,2*NC,3) in CONTOURS order, or a full (E,85,3) landmark array from which the
        endpoint rows are taken — fam_endpoint.ENDS is EP_IDX element for element, so its
        own layout drops straight in. It must be that family's IMPROVED predictions; its
        `est` key is the un-refined ensemble it takes as input and would gain nothing.
        """
        ep = (meta.get("artefacts") or {}).get("endpoints")
        if ep is None:
            return {}
        ep = np.asarray(ep)
        if ep.ndim == 3 and ep.shape[1] == NL:
            ep = ep[:, EP_IDX]
        assert ep.ndim == 3 and ep.shape[1:] == (2 * NC, 3), \
            f"artefact 'endpoints' must be (E,{2 * NC},3) or (E,{NL},3), got {ep.shape}"
        return {"endpoints": torch.tensor(ep[np.asarray(ears)], dtype=torch.float32,
                                          device=meta["dev"])}

    def __init__(self, cfg, meta):
        super().__init__()
        c = self.c = derive(cfg)
        assert meta["nl"] == NL and meta["contours"] == CONTOURS, "head geometry contract broken"
        assert not (c["use_nrm"] and "nrm" not in self.NEEDS), \
            "use_nrm is on but NEEDS is empty: set USE_NRM=1 in the ENVIRONMENT (not only " \
            "CFG_USE_NRM) so the trainer loads 'nrm', and point DATA at screen_data_2048nrm.npz"
        self.net = Net(c, 3 + 3 * c["use_nrm"])
        prof, sd, self.src = load_profile(meta, c)
        self.dec = nn.ModuleList([ProfileDecoder(prof[ci], sd[ci], c, 99) for ci in c["prof"]])
        self.sup = np.array([0.5 ** (c["npass"] - 1 - t) for t in range(c["npass"])])
        self.sup /= self.sup.sum()
        self.end_idx = [i for ci in c["prof"] for i in CONTOURS[ci]]
        print(f"fam_profile: profile on {[CNAMES[i] for i in c['prof']] or 'NOTHING (baseline)'}"
              f" | mode {c['profile_mode']} | endpoint_fit {c['endpoint_fit']} | profile from "
              f"{self.src}", flush=True)

    def forward(self, b):
        outs, raw, feat = self.net(b["pc"], b["coarse"],
                                   b.get("nrm") if self.c["use_nrm"] else None)
        ep = b.get("endpoints") if self.c["endpoint_fit"] != "none" else None
        if ep is not None:
            assert torch.isfinite(ep).all(), "batch['endpoints'] contains non-finite rows"
        pred, phase = raw.clone(), {}
        for k, ci in enumerate(self.c["prof"]):
            lo, hi = CONTOURS[ci]
            C = raw[:, lo:hi + 1]
            if ep is not None:
                C = reposition(C, ep[:, 2 * ci], ep[:, 2 * ci + 1], self.c["endpoint_fit"])
            s = self.dec[k].phases(feat[:, lo:hi + 1])
            pred[:, lo:hi + 1] = place_on_polyline(C, s)
            phase[ci] = s
        return {"pred": pred, "raw": raw, "phase": phase,
                "aux": [q2 for _, q2 in outs], "pre": [q1 for q1, _ in outs]}

    def loss(self, out, tg):
        """The baseline objective on the PRE-placement curve, plus the ordered metric on the
        PLACED landmarks. Both terms are needed: the placement only ever moves a landmark
        ALONG the curve, so without the raw term the curve's shape has no gradient of its
        own where the profile happens to be insensitive to it."""
        L = 0.0
        for t, (q1, q2) in enumerate(zip(out["pre"], out["aux"])):
            L = L + float(self.sup[t]) * (0.4 * ((q1 - tg) ** 2).sum(-1).mean()
                                          + ((q2 - tg) ** 2).sum(-1).mean())
        L = L + self.c["w_raw"] * ((out["raw"] - tg) ** 2).sum(-1).mean() \
            + ((out["pred"] - tg) ** 2).sum(-1).mean()
        if self.c["w_end"] and self.end_idx:
            i = self.end_idx
            L = L + self.c["w_end"] * ((out["raw"][:, i] - tg[:, i]) ** 2).sum(-1).mean()
        return L


MODEL = FamProfile


# ------------------------------------------------------------------ smoke test
def synth(B=2, npts=1024, seed=0):
    """A pinna-scale curved sheet with 4 ORDERED contours lying on it. The landmarks are
    placed at a deliberately NON-uniform arc-length profile, so the placement operator has
    something to do and the profile checks are not vacuous."""
    g = torch.Generator().manual_seed(seed)
    u = (torch.rand(B, npts, generator=g) - .5) * 50.0
    v = (torch.rand(B, npts, generator=g) - .5) * 36.0
    sheet = lambda a, b_: .006 * a ** 2 - .010 * b_ ** 2 + 3.0 * torch.sin(a / 6.0)
    pc = torch.stack([u, v, sheet(u, v)], -1) + torch.randn(B, npts, 3, generator=g) * .05
    nrm = torch.stack([-(.012 * u + .5 * torch.cos(u / 6.0)), .020 * v, torch.ones_like(u)], -1)
    nrm = nrm / nrm.norm(dim=-1, keepdim=True)
    geo = [(22., 15., .2, 5.4), (13., 9., 3.4, 8.6), (16., 11., .5, 4.3), (6., 4., 1.0, 3.0)]
    tgt = []
    for ci, ((lo, hi), (rx, ry, a0, a1)) in enumerate(zip(CONTOURS, geo)):
        n = hi - lo + 1
        w = torch.linspace(0, 1, n)
        w = w + .10 * torch.sin(2 * np.pi * w) * (1 + .3 * ci)          # non-uniform phase
        w = (w - w[0]) / (w[-1] - w[0])
        th = (a0 + (a1 - a0) * w)[None] + .05 * torch.randn(B, 1, generator=g)
        a, b_ = rx * th.cos(), ry * th.sin()
        tgt.append(torch.stack([a, b_, sheet(a, b_)], -1))
    tgt = torch.cat(tgt, 1)
    return pc, nrm, tgt + torch.randn(B, NL, 3, generator=g) * .9, tgt


def trainer_smoke():
    """End-to-end through train_family.py: proves the FAMILY CONTRACT (cls(cfg, meta),
    forward(batch), loss, the ARTEFACTS fold proof, the report schema, the full pipeline)."""
    import tempfile, train_family as TF
    tmp = os.path.join(tempfile.gettempdir(), "fam_profile_smoke")
    dp, tp, sp = TF.fake_bundle(tmp, npts=384)[:3]
    ne = len(np.load(dp)["true"])
    subj, parts = TF.frozen_folds(ne)
    mask = ~np.isin(subj, parts[0])
    gt = np.load(dp)["true"][mask].astype(np.float64)
    art = dict(fold=np.int64(0), train_ear_mask=mask,
               endpoints=np.stack([np.load(dp)["coarse"][:, [i for lo, hi in CONTOURS
                                                             for i in (lo, hi)]]], 0)[0])
    for ci, (lo, hi) in enumerate(CONTOURS):
        s = arc_profile(torch.tensor(gt[:, lo:hi + 1])).numpy()
        art[f"prof_c{ci}"], art[f"prof_sd_c{ci}"] = s.mean(0), s.std(0, ddof=1)
    ap = f"{tmp}/profile_f0.npz"
    np.savez(ap, **art)
    env = dict(FAMILY="profile", FAMILY_MODULE="fam_profile", FAMILY_CLASS="FamProfile",
               FOLD="0", SEED="0", EPOCHS="2", WORK=tmp, DATA=dp, TRIS=tp, SSM=sp,
               ARTEFACTS=ap, TTA="1", EVAL_EVERY="2", ALIAS="0", TAG="fam_profile_smoke",
               CFG_BS="8", CFG_WIDTH="32", CFG_NPASS="2", CFG_K="12", CFG_GK="8",
               CFG_PROFILE_MODE="learned")
    keep = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    res = TF.main()
    for k, v in keep.items():
        os.environ.pop(k) if v is None else os.environ.__setitem__(k, v)
    assert res["ordered_MLE_full_mm"] is not None, "full pipeline did not run"
    assert res["config"]["_augment"] == "profile_augment" and res["config"]["_batch_hook"]
    print(f"  trainer report OK | raw {res['ordered_MLE_mm']:.4f} -> full "
          f"{res['ordered_MLE_full_mm']:.4f} mm | {res['params']:,} params | "
          f"augment {res['config']['_augment']} | endpoints reached forward()")


if __name__ == "__main__":
    import time
    t0 = time.time()
    torch.manual_seed(0)
    B, NPTS = 2, int(os.environ.get("SMOKE_NPTS", "1024"))
    pc, nrm, q0, tgt = synth(B, NPTS)
    meta = dict(nl=NL, contours=CONTOURS, scale=SCALE, npts=NPTS, fold=0, dev="cpu",
                n_train_ears=272, artefacts={})

    print("=" * 78)
    print("1/8  arc-length ops are exact")
    P = tgt[:, 55:75].double()
    s_gt = arc_profile(P)
    rt = (place_on_polyline(P, s_gt) - P).norm(dim=-1).max()
    print(f"  place(P, arc_profile(P)) == P : max err {float(rt):.3e} mm  "
          f"-> ZERO representation floor at full dof (curve_floor.json: 0.7448 mm for "
          f"inner_helix at ~60% dof)")
    assert float(rt) < 1e-9
    e0, e1 = P[:, 0] + 1.0, P[:, -1] - 0.5
    for mode in ("similarity", "rigid", "translate"):
        Q = reposition(P, e0, e1, mode)
        d0, d1 = float((Q[:, 0] - e0).norm(dim=-1).max()), float((Q[:, -1] - e1).norm(dim=-1).max())
        print(f"  reposition({mode:10s}): |C0-e0| {d0:.2e}  |C1-e1| {d1:.2e}  "
              f"len {float(arc_profile(Q)[0, 1]):.6f} vs {float(s_gt[0, 1]):.6f} (profile preserved)")
        assert d0 < 1e-9 and (mode != "similarity" or d1 < 1e-9)
        assert float((arc_profile(Q) - s_gt).abs().max()) < 1e-9

    print("\n2/8  model, forward + backward")
    fam = FamProfile({}, meta)
    npar = sum(p.numel() for p in fam.parameters())
    base = sum(p.numel() for p in FamProfile({"profile_contours": ""}, meta).parameters())
    lrn = sum(p.numel() for p in FamProfile({"profile_mode": "learned"}, meta).parameters())
    print(f"  params fixed {npar:,} | baseline (PROFILE_CONTOURS='') {base:,} | "
          f"learned {lrn:,}  (+{lrn - base:,} for 2 deviation heads)")
    assert npar == base, "PROFILE_MODE=fixed must add no parameters"
    out = fam({"pc": pc, "coarse": q0, "ear": torch.arange(B)})
    L = fam.loss(out, tgt)
    L.backward()
    gn = sum(float(p.grad.norm()) ** 2 for p in fam.parameters() if p.grad is not None) ** .5
    print(f"  pred {tuple(out['pred'].shape)}  raw {tuple(out['raw'].shape)}  "
          f"passes {len(out['aux'])}  loss {float(L):.4f}  grad-norm {gn:.3e}")
    assert tuple(out["pred"].shape) == (2, NL, 3), out["pred"].shape
    assert torch.isfinite(out["pred"]).all() and gn > 0
    nog = [n for n, p in fam.named_parameters() if p.grad is None]
    assert not nog, nog

    print("\n3/8  the placed landmarks reproduce the intended profile")

    def locate(P, C):
        """Arc length of each point of P along the polyline C, normalised. INDEPENDENT of
        place_on_polyline: closest point on each segment, then the winning segment."""
        d = C[:, 1:] - C[:, :-1]
        seg = d.norm(dim=-1)
        cum = torch.cat([torch.zeros_like(seg[:, :1]), seg.cumsum(-1)], -1)
        t = (((P[:, :, None] - C[:, None, :-1]) * d[:, None]).sum(-1)
             / (seg ** 2)[:, None].clamp(min=1e-12)).clamp(0, 1)
        foot = C[:, None, :-1] + t[..., None] * d[:, None]
        j = (P[:, :, None] - foot).norm(dim=-1).argmin(-1)
        return (torch.gather(cum[:, :-1], 1, j) + torch.gather(t, 2, j[..., None])[..., 0]
                * torch.gather(seg, 1, j)) / cum[:, -1:]

    print(f"  {'contour':16s} {'n':>3s} {'|arc(placed)-s|':>16s} {'chord dev, net':>15s} "
          f"{'chord dev, GT curve':>21s}")
    for ci in fam.c["prof"]:
        lo, hi = CONTOURS[ci]
        C, s = out["raw"][:, lo:hi + 1].double(), out["phase"][ci].double()
        got = place_on_polyline(C, s)
        ea = float((locate(got, C) - s).abs().max())
        # the placed points' OWN chord profile equals s only up to CORNER CUTTING, so it is
        # reported, not asserted -- on the untrained net the polyline is a zigzag, on the GT
        # contour (a real ear curve) it is the honest number.
        G = tgt[:, lo:hi + 1].double()
        ec = float((arc_profile(got) - s).abs().max())
        eg = float((arc_profile(place_on_polyline(G, s)) - s).abs().max())
        print(f"  {CNAMES[ci]:16s} {hi-lo+1:3d} {ea:16.3e} {ec:15.4f} {eg:21.5f}")
        assert ea < 1e-6, "placed landmark is not at its intended arc length"
    kept = [i for ci in range(NC) if ci not in fam.c["prof"]
            for i in range(CONTOURS[ci][0], CONTOURS[ci][1] + 1)]   # +1: hi is INCLUSIVE
    assert torch.equal(out["pred"][:, kept], out["raw"][:, kept]), \
        "a contour outside PROFILE_CONTOURS was modified"
    print(f"  contours outside PROFILE_CONTOURS are bit-identical to the free-XYZ head "
          f"({len(kept)} landmarks)")

    print("\n4/8  learned deviation: bounded by the measured sd, strictly monotone")
    lm = FamProfile({"profile_mode": "learned", "dev_bound": 1.0}, meta)
    o2 = lm({"pc": pc, "coarse": q0, "ear": torch.arange(B)})
    o2["pred"].sum().backward()
    assert all(p.grad is not None and float(p.grad.norm()) > 0 for p in lm.dec.parameters()), \
        "no gradient reached the deviation head"
    print(f"  gradient reaches both deviation heads at the default init; max|s-s_mean| there "
          f"{max(float((o2['phase'][ci] - lm.dec[k].s_mean[None]).abs().max()) for k, ci in enumerate(lm.c['prof'])):.2e}")

    class _Stub(nn.Module):
        """Deterministic EXTREME alternating gap multipliers, so the sd clamp is actually
        exercised rather than trivially satisfied by a small default-init deviation."""
        def forward(self, x):
            v = 3.0 * (-1.0) ** torch.arange(x.shape[-1], device=x.device, dtype=x.dtype)
            return v[None, None].expand(x.shape[0], 1, -1)

    for d in lm.dec:
        d.head = _Stub()
    with torch.no_grad():
        o2 = lm({"pc": pc, "coarse": q0, "ear": torch.arange(B)})
    for k, ci in enumerate(lm.c["prof"]):
        d, s = lm.dec[k], o2["phase"][ci]
        g = (d.s_mean[1:] - d.s_mean[:-1]) * torch.exp(
            d.logb * torch.tanh(_Stub()(torch.zeros(1, 1, len(d.s_mean)))[:, 0, :-1]))
        raw = torch.cat([torch.zeros(1, 1), (g / g.sum()).cumsum(-1)], -1) - d.s_mean[None]
        dev, step = (s - d.s_mean[None]).abs(), (s[:, 1:] - s[:, :-1]).min()
        print(f"  {CNAMES[ci]:16s} unbounded max|dev| {float(raw.abs().max()):.5f} -> bounded "
              f"{float(dev.max()):.5f} (sd {float(d.sd.max()):.5f}) | min gap {float(step):.5f}"
              f" | s0 {float(s[0, 0]):.1e} s_end {float(s[0, -1]):.6f}")
        assert float(raw.abs().max()) > float(d.sd.max()), "the clamp was never exercised"
        assert (dev <= d.sd[None] + 1e-6).all(), "deviation escaped the measured sd bound"
        assert float(step) > 0, "phases are not strictly increasing"

    print("\n5/8  external endpoints (the fam_endpoint.py interface)")
    ends = torch.stack([tgt[:, i] for lo, hi in CONTOURS for i in (lo, hi)], 1)
    o3 = fam({"pc": pc, "coarse": q0, "ear": torch.arange(B), "endpoints": ends})
    for ci in fam.c["prof"]:
        lo, hi = CONTOURS[ci]
        d0 = float((o3["pred"][:, lo] - ends[:, 2 * ci]).norm(dim=-1).max())
        d1 = float((o3["pred"][:, hi] - ends[:, 2 * ci + 1]).norm(dim=-1).max())
        moved = float((o3["pred"][:, lo:hi + 1] - out["pred"][:, lo:hi + 1]).norm(dim=-1).mean())
        print(f"  {CNAMES[ci]:16s} endpoint residual {d0:.2e} / {d1:.2e} mm | interior moved "
              f"{moved:.3f} mm")
        assert d0 < 1e-4 and d1 < 1e-4, "supplied endpoints were not honoured exactly"
    assert torch.equal(o3["pred"][:, kept], o3["raw"][:, kept])
    full = np.arange(4 * NL * 3, dtype=np.float32).reshape(4, NL, 3)
    got = FamProfile.BATCH(np.array([1, 3]), None, dict(dev="cpu", artefacts={"endpoints": full}))
    g8 = FamProfile.BATCH(np.array([1, 3]), None,
                          dict(dev="cpu", artefacts={"endpoints": full[:, EP_IDX]}))
    assert tuple(got["endpoints"].shape) == (2, 2 * NC, 3) and torch.equal(
        got["endpoints"], g8["endpoints"]), "the (E,85,3) and (E,8,3) layouts disagree"
    assert FamProfile.BATCH(np.array([0]), None, dict(dev="cpu", artefacts={})) == {}
    print(f"  cls.BATCH pulls endpoints {tuple(got['endpoints'].shape)} out of ARTEFACTS; "
          f"(E,85,3) and (E,8,3) layouts agree; absent -> {{}}")
    print(f"  EP_IDX {EP_IDX} == fam_endpoint.ENDS")

    print("\n6/8  profile_augment == default_augment, plus the endpoint rows")
    import train_family as TF
    acfg = {**TF.TRAIN_DEFAULTS, "ep_jit": 0.0}
    b0 = {"pc": pc[:, None, :64], "coarse": q0, "ear": torch.arange(B)}
    ga = torch.Generator(); ga.manual_seed(11)
    gb = torch.Generator(); gb.manual_seed(11)
    ra, ta = TF.default_augment(b0, tgt, acfg, (), ga)
    rb, tb = profile_augment({**b0, "endpoints": ends}, tgt, acfg, (), gb)
    for k in ("pc", "coarse"):
        assert torch.equal(ra[k], rb[k]), f"profile_augment diverged from default_augment on {k}"
    assert torch.equal(ta, tb)
    # the endpoints must undergo the SAME similarity as the target
    dt = (tb[:, 0] - tb[:, 24]).norm(dim=-1) / (tgt[:, 0] - tgt[:, 24]).norm(dim=-1)
    de = (rb["endpoints"][:, 0] - rb["endpoints"][:, 1]).norm(dim=-1) / \
         (ends[:, 0] - ends[:, 1]).norm(dim=-1)
    print(f"  pc/coarse/target bit-identical to default_augment | scale on target "
          f"{[round(float(x), 6) for x in dt]} vs on endpoints {[round(float(x), 6) for x in de]}")
    assert torch.allclose(dt, de, atol=1e-5)
    assert float((rb["endpoints"][:, 0] - tb[:, 0]).norm(dim=-1).max()) < 1e-4, \
        "endpoints and the target are no longer in the same frame after augmentation"

    print("\n7/8  fold safety of the profile artefact")
    good = {"fold": np.int64(0), "train_ear_mask": np.ones(340, bool)}
    for ci, (lo, hi) in enumerate(CONTOURS):
        good[f"prof_c{ci}"] = np.linspace(0, 1, hi - lo + 1)
        good[f"prof_sd_c{ci}"] = np.full(hi - lo + 1, .01)
    for bad, why, msg in (
            ({**good, "fold": np.int64(3)}, "artefact from another fold", "fold 3"),
            ({**good, "train_ear_mask": np.zeros(340, bool)}, "mask is not the whole "
             "training fold", "not THIS fold's training set"),
            ({**good, "prof_c2": np.linspace(0, 1, 20)[[0, 1, 2, 3, 4, 6, 5, 7, 8, 9, 10, 11,
                                                        12, 13, 14, 15, 16, 17, 18, 19]]},
             "non-monotone profile", "not increasing")):
        try:
            load_profile(dict(fold=0, n_train_ears=340, artefacts=bad), derive({}))
            raise SystemExit(f"ACCEPTED {why}")
        except AssertionError as e:
            assert msg in str(e), str(e)
            print(f"  refused: {why:40s} -> {' '.join(str(e).split())[:78]}")
    try:
        load_profile(dict(fold=0, n_train_ears=272, artefacts={}),
                     derive({"profile_fallback": "error"}))
        raise SystemExit("ACCEPTED a missing profile under PROFILE_FALLBACK=error")
    except AssertionError as e:
        print(f"  refused: {'no artefact under PROFILE_FALLBACK=error':44s} -> "
              f"{str(e)[:64]}")
    p, sd, src = load_profile(dict(fold=0, n_train_ears=272, artefacts={}), derive({}))
    print(f"  fallback accepted: {src}")

    print("\n8/8  the operator on GT: what the profile costs when the curve is perfect")
    print(f"  {'contour':16s} {'uniform':>9s} {'popmean':>9s}   (mm error of GT landmarks "
          f"replaced by their placement on the GT polyline)")
    gtd = tgt.double()
    for ci, (lo, hi) in enumerate(CONTOURS):
        C = gtd[:, lo:hi + 1]
        n = hi - lo + 1
        su = torch.linspace(0, 1, n, dtype=torch.float64)[None].expand(B, -1)
        sm = arc_profile(C).mean(0, keepdim=True).expand(B, -1)
        eu = float((place_on_polyline(C, su) - C).norm(dim=-1).mean())
        em = float((place_on_polyline(C, sm) - C).norm(dim=-1).mean())
        print(f"  {CNAMES[ci]:16s} {eu:9.4f} {em:9.4f}")
    print(f"\n  (synthetic curves -- the REAL numbers are in research/results/"
          f"profile_apply.json, produced by profile_apply.py on the OOF ensemble)")
    print(f"SMOKE PASS  ({time.time() - t0:.1f}s)")
    print("=" * 78)

    if os.environ.get("SMOKE_TRAIN"):
        print("\n--- train_family.py end-to-end (SMOKE_TRAIN=1) ---")
        trainer_smoke()
        print(f"OK  ({time.time() - t0:.1f}s total)")
