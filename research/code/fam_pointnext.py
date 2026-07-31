"""
POINTNEXT-STYLE HIERARCHICAL ENCODER for high-resolution ear clouds (8k/16k/32k),
feeding the existing offset/snap refinement head of gpu_screen.py.

WHY. Every variant of the 2048-point static-DGCNN family is null: the error is 77%
along-contour phase, and a flat 2048-point graph simply cannot see the sub-millimetre
ridge detail that fixes phase. This module changes the *backbone* -- multi-resolution,
so a landmark can be positioned by 11 mm of context and then snapped against a
0.25 mm-spaced cloud -- while leaving the head that already works alone.

GEOMETRY, IN MILLIMETRES. The one measured anchor is the baseline: at 2048 points the
mean nearest-neighbour spacing is 0.995 mm and the K=48 landmark window spans 7.35 mm.
Clouds are iid draws from mesh vertices, so the area per point is A = pi*7.35^2/48 =
3.535 mm^2 = 3.57 s^2 (a Poisson process gives 4 s^2 -- the cloud is essentially
Poisson, which is the consistency check that lets the anchor be extrapolated).
Hence  r(K, s) = K2R * s * sqrt(K)  with  K2R = 7.35/(0.995*sqrt(48)) = 1.0662,
and spacing scales as 1/sqrt(N) (area per point ~ 1/N on a fixed surface):

    N        spacing s        r for NSAMPLE=32
    2048     0.9950 mm        6.00 mm   (the baseline resolution, for reference)
    8192     0.4975 mm        3.00 mm
    16384    0.3518 mm        2.12 mm
    32768    0.2487 mm        1.50 mm

Every set-abstraction level k gets ONE radius r_k = K2R * s(N_k) * sqrt(NSAMPLE),
i.e. the radius that holds NSAMPLE points at that level's own spacing. Level point
counts interpolate geometrically from NPTS to NFINAL over STAGES stages, so
STAGES=4/6/8 all reach the same global bottleneck and only the per-stage ratio changes.
`plan_report()` prints the table for any config.

WHICH CONSTRAINT ACTUALLY BINDS -- measured, because the whole family is justified on
millimetre bookkeeping and the two grouping calls behave differently:
  * SAME-LEVEL InvResMLP grouping (query and source both at level k) is radius-bound as
    intended: r_k holds ~NSAMPLE points, so the mask bites (True fraction 0.99) and the
    extent is r_k to within a few percent.
  * STRIDED downsampling (query at level k+1, source at the FINER level k, radius r_k+1)
    is NOT radius-bound. The finer source packs ~NSAMPLE*ratio candidates inside r_k+1,
    so the NSAMPLE nearest all sit well inside it and the mask is True EVERYWHERE
    (measured: exactly 1.0000 at every stage, at both 8192 and 32768). The binding
    constraint is the k-NN, so the real extent of the level-k -> k+1 aggregation is
    r_k (the FINER level's radius), not the r_k+1 printed beside it -- one stage, a
    factor sqrt(ratio) ~ 1.83, finer than the table's radius column suggests. Nothing is
    broken by this (the hierarchy is still geometric and monotone) but the receptive
    field is smaller than a naive read of the table, so `plan_report()` prints the
    strided extent explicitly. Making r_k+1 bind would need PointNet++'s take-any-point-
    inside-the-ball, which discards the nearest neighbours and is strictly worse.

MULTI-RESOLUTION HEAD. This is the part that does not survive naive porting. The head
gathers KHEAD=48 points; at 32768 points those 48 span 1.83 mm, but the coarse
residual reaches 10.6 mm at p99, so a fixed-K head at high resolution could never
reach its target. Instead each refinement pass declares a PHYSICAL radius and is served
by the decoder level whose own KHEAD-window most nearly matches it. The passes therefore
walk coarse-to-fine in resolution as well as radius, gather cost stays 85*48 per pass,
and the LAST pass is forced onto level 0 so the final snap is a convex combination of
full-resolution surface points. Weights are UNTIED per pass.

The ladder is a GEOMETRIC interpolation R_FIRST -> R_LAST (11.0 -> 2.5 mm) between
gpu_screen untied6's two calibration ANCHORS -- coarse residual p99 = 10.6 mm and final
residual p90 = 2.5 mm. It is not untied6's radius LIST, which is [11.0, 9.0, 7.4, 5.5,
4.0, 3.0]: that one is hand-spaced and stops at 3.0. The offsets O_FIRST -> O_LAST
(7.0 -> 0.7) do reproduce untied6's endpoints, and the interior values land within
0.25 mm of its [7.0, 4.5, 3.0, 2.0, 1.2, 0.7].

Snapping a continuous radius ladder onto a discrete level ladder is approximate, so a
pass's declared radius binds only when it is SMALLER than that level's KHEAD-window.
Measured at 8192/STAGES=4: passes 0/2/4/5 are radius-bound (mask True fraction 0.97,
0.97, 0.99, 0.67) while passes 1 and 3 are k-NN-bound, seeing 6.53 mm where 8.18 mm was
declared and 3.66 mm where 4.52 mm was declared. The error is always in the SAFE
direction -- a pass never sees more context than it declared -- and the final pass, the
one that sets the output, is properly radius-limited to R_LAST.

The snap is unchanged: unconstrained (optionally tanh-bounded) offset, then a softmax
over the KHEAD gathered surface points, output = sum w_j p_j with w >= 0, sum w = 1.
That convex-combination property is asserted numerically by the smoke test.

SAMPLING. Exact FPS is a sequential loop of length m, so at 32768 points it costs tens
of thousands of tiny kernel launches per forward. `chunkfps` (the default) sorts by
Z-ORDER code, splits into contiguous ~FPS_CHUNK-point blocks and runs exact FPS inside
all blocks at once, cutting the loop to m/G. Measured at 8192 -> 2435 points on an
ear-sized synthetic shell (mean/p99 distance from any point to the nearest selected
point, and mean spacing between selected points; lower cover / higher spacing = better):

    sampler     cover_mean  cover_p99  sel_spacing   CPU
    rand          0.5483     1.8318      0.7886     0.00s
    chunkfps      0.4261     1.0081      1.2037     0.32s
    fps (exact)   0.4241     0.9830      1.2307     3.39s

So chunkfps is within 0.5% of exact FPS on mean coverage and 2.6% on the tail, 10x
cheaper, and DETERMINISTIC (verified: repeated eval forwards on one cloud differ by
0.0000 mm). The Z-order partition is load-bearing -- with random blocks instead, every
block spans the whole object, their picks collide, and coverage collapses to roughly
uniform random sampling (0.5277 / 1.6438 / 0.7894 when measured the same way).

COST. Two things dominate and both are instrumented (`PEAK`, printed by the smoke test).
(a) The distance blocks inside knn_dist carry no gradient, so they are transient; their
size is capped by QBYTES (default 500 MB) by shrinking the query block, not by a fixed
row count -- otherwise a large batch at 32768 points would allocate B*2048*32768*4.
(b) The grouped (B, N_k, NSAMPLE, C) activations ARE kept for backward and set the
batch size. Measured on CPU at WIDTH=256 / STAGES=4 / NPASS=6, grouped bytes per ear:
8192 -> 368 MB, 16384 -> 528 MB, 32768 -> 787 MB; the worst reachable config
(32768 / STAGES=8 / WIDTH=384 / NPASS=8, 37.7M params) -> 4978 MB per ear. These count
grouped tensors only, so they are a LOWER bound on the true footprint. Expected
feasible batch on 48 GB: ~16 at 32768/STAGES=4/WIDTH=256, ~4 at 32768/STAGES=8/WIDTH=384.
Unverified on GPU -- the development machine has 15.7 GB of RAM and segfaults on the
worst config at B=2, which is why that config is reported at B=1.

AUGMENTATION JITTER MUST FOLLOW THE DENSITY, or this family cannot test its own
hypothesis. train_family.py's TRAIN_DEFAULTS carry aug_jit=0.25 mm per coordinate, i.e.
0.25*sqrt(3) = 0.433 mm of RMS 3-D displacement per point -- calibrated for the 2048-point
baseline, where the mean spacing is 0.995 mm and the jitter is a harmless 0.44 spacings.
Inherited unchanged it becomes 0.87 spacings at 8192, 1.23 at 16384 and 1.74 at 32768:
the augmenter would displace every point by more than the distance to its neighbour,
destroying exactly the sub-millimetre detail the extra points were added to resolve, and
it would do so at TRAIN time only (evaluate() does not augment) -- a train/test mismatch
that would return null for a reason having nothing to do with resolution. So AUG_JIT
defaults to 0.25*sqrt(2048/NPTS), holding the baseline's jitter-to-spacing ratio fixed,
and is exported in DEFAULTS (which train_family merges OVER TRAIN_DEFAULTS) and in
SEARCH_SPACE. plan_report() prints the ratio so it can never be silently wrong again.

NO GROUND TRUTH IS TOUCHED. This module sees only (cloud, coarse, optional normals).

TWO ENTRY POINTS. `PointNeXtLandmark(...).forward(pc, q0, ft)` is gpu_screen.py's idiom
and takes its config from the environment variables below. `MODEL(cfg, meta)` is the
train_family.py FAMILY CONTRACT adapter and takes the same knobs as CFG_<NAME> (it
rebinds the module globals the layers read, once, before building any submodule), so
search_driver.py can sweep SEARCH_SPACE without editing code. USE_NRM is the one knob
that MUST come from the environment, because the driver reads cls.NEEDS off the class
before instantiating it.

    python research/code/fam_pointnext.py                    # CPU smoke test, B=2, 8192 pts
    NPTS=32768 STAGES=6 WIDTH=384 python research/code/fam_pointnext.py
    SMOKE_B=1 NPTS=32768 STAGES=8 WIDTH=384 NPASS=8 python research/code/fam_pointnext.py
"""
import os, math, time
import numpy as np
import torch
import torch.nn as nn

NL, SCALE = 85, 30.0
CONTOURS = [(0, 24), (25, 54), (55, 74), (75, 84)]

NPTS = int(os.environ.get("NPTS", "8192"))            # 8192 | 16384 | 32768
WIDTH = int(os.environ.get("WIDTH", "256"))           # 128 | 256 | 384 -- decoder/head width
STAGES = int(os.environ.get("STAGES", "4"))           # 4 | 6 | 8 set-abstraction stages
NPASS = int(os.environ.get("NPASS", "6"))             # 4 | 6 | 8 refinement passes
DROPOUT = float(os.environ.get("DROPOUT", "0.1"))
USE_NRM = int(os.environ.get("USE_NRM", "0"))         # 1 -> concat per-point normals
NSAMPLE = int(os.environ.get("NSAMPLE", "32"))        # ball-query neighbours per centroid
NFINAL = int(os.environ.get("NFINAL", "64"))          # centroids at the bottleneck
BLOCKS = int(os.environ.get("BLOCKS", "1"))           # InvResMLP blocks per stage
EXPANSION = int(os.environ.get("EXPANSION", "2"))     # InvResMLP inverted-bottleneck factor
NORM = os.environ.get("NORM", "bn")                   # bn | ln | none
KHEAD = int(os.environ.get("KHEAD", "48"))            # head window, in points
KEEPMIN = int(os.environ.get("KEEPMIN", "8"))         # nearest points the head always keeps
R_FIRST = float(os.environ.get("R_FIRST", "11.0"))    # pass-0 physical radius (coarse p99)
R_LAST = float(os.environ.get("R_LAST", "2.5"))       # last-pass radius (final p90)
O_FIRST = float(os.environ.get("O_FIRST", "7.0"))     # pass-0 offset bound, mm
O_LAST = float(os.environ.get("O_LAST", "0.7"))
SAMPLER = os.environ.get("SAMPLER", "chunkfps")       # chunkfps | fps | rand
FPS_CHUNK = int(os.environ.get("FPS_CHUNK", "1024"))  # points per parallel FPS partition
QCHUNK = int(os.environ.get("QCHUNK", "2048"))        # max query rows per cdist block
QBYTES = float(os.environ.get("QBYTES", "5e8"))       # byte ceiling for one cdist block
CONTOUR_NET = int(os.environ.get("CONTOUR_NET", "1"))
# per-coordinate cloud jitter, mm. Scaled so the jitter-to-spacing ratio matches the
# 2048-point baseline's 0.25/0.995 -- see AUGMENTATION JITTER above.
AUG_JIT = float(os.environ.get("AUG_JIT", 0.25 * math.sqrt(2048.0 / NPTS)))
SPACING_2048 = float(os.environ.get("SPACING_2048", "0.995"))
WINDOW_48 = float(os.environ.get("WINDOW_48", "7.35"))
K2R = WINDOW_48 / (SPACING_2048 * math.sqrt(48.0))

PEAK = {"bytes": 0, "what": "", "retained": 0}         # memory instrumentation, see COST


def _peak(t, what, retained=True):
    """record the largest tensor allocated, and accumulate the ones autograd keeps."""
    n = t.numel() * t.element_size()
    if n > PEAK["bytes"]:
        PEAK.update(bytes=n, what=f"{what} {tuple(t.shape)}")
    if retained:
        PEAK["retained"] += n
    return t


def spacing(n):
    return SPACING_2048 * math.sqrt(2048.0 / n)


def level_plan(npts=NPTS, stages=STAGES, nfinal=NFINAL, nsample=NSAMPLE):
    """point count, mean NN spacing and ball radius for levels 0..stages."""
    ratio = (npts / nfinal) ** (1.0 / stages)
    N = [npts] + [max(nfinal, int(round(npts / ratio ** k))) for k in range(1, stages + 1)]
    S = [spacing(n) for n in N]
    R = [K2R * s * math.sqrt(nsample) for s in S]
    return N, S, R


def pass_plan(npass=NPASS, npts=NPTS, stages=STAGES, nfinal=NFINAL, nsample=NSAMPLE):
    """per-pass (radius mm, offset bound mm, decoder level) -- coarse to fine in both."""
    N, S, _ = level_plan(npts, stages, nfinal, nsample)
    f = (lambda a, b, i: a * (b / a) ** (i / (npass - 1)) if npass > 1 else b)
    RP = [f(R_FIRST, R_LAST, i) for i in range(npass)]
    OP = [f(O_FIRST, O_LAST, i) for i in range(npass)]
    lev, cur = [], stages
    for r in RP:
        want = r / (K2R * math.sqrt(KHEAD))            # spacing whose KHEAD-window is r
        k = min(range(stages + 1), key=lambda j: abs(math.log(S[j] / want)))
        k = min(k, cur)                                # never coarsen again
        while N[k] < KHEAD and k > 0:
            k -= 1
        lev.append(k); cur = k
    lev[-1] = 0                                        # final snap on the full-resolution cloud
    return RP, OP, lev


# --------------------------------------------------------------- torch-only sampling/grouping
def fps_exact(xyz, m):
    """farthest-point sampling, batched, deterministic start at index 0. O(m*N)."""
    B, N, _ = xyz.shape
    m = min(m, N)
    out = torch.empty(B, m, dtype=torch.long, device=xyz.device)
    d = xyz.new_full((B, N), 1e10)
    far = torch.zeros(B, dtype=torch.long, device=xyz.device)
    ar = torch.arange(B, device=xyz.device)
    for i in range(m):
        out[:, i] = far
        d = torch.minimum(d, ((xyz - xyz[ar, far][:, None, :]) ** 2).sum(-1))
        far = d.argmax(1)
    return out


def morton(p):
    """Z-order code of each point on a 10-bit-per-axis grid. Sorting by it makes any
    contiguous run of points a spatially COMPACT cluster, which is the whole reason
    block-FPS works: FPS inside a compact block spreads inside that block, so the union
    over blocks is a stratified spread. With RANDOM blocks it does not -- every block
    then covers the whole object and their picks collide (measured: random-block FPS is
    barely better than uniform random sampling)."""
    lo = p.amin(1, keepdim=True)
    ex = (p.amax(1, keepdim=True) - lo).clamp(min=1e-9)
    c = ((p - lo) / ex * 1023).long().clamp(0, 1023)

    def sp(v):                                         # part1by2: insert 2 zeros per bit
        v = (v | (v << 16)) & 0x030000FF
        v = (v | (v << 8)) & 0x0300F00F
        v = (v | (v << 4)) & 0x030C30C3
        v = (v | (v << 2)) & 0x09249249
        return v
    return sp(c[..., 0]) | (sp(c[..., 1]) << 1) | (sp(c[..., 2]) << 2)


@torch.no_grad()
def fps_sample(xyz, m):
    """FPS index set (B,m). `chunkfps` runs exact FPS in parallel on a Z-ORDER partition
    into ~FPS_CHUNK-point blocks, cutting the sequential loop from m to m/G steps. Ranks
    are interleaved before trimming, so dropping down to exactly m removes each block's
    lowest-priority picks rather than a whole block."""
    B, N, _ = xyz.shape
    if m >= N:
        return torch.arange(N, device=xyz.device)[None].expand(B, N)
    if SAMPLER == "rand":
        return torch.rand(B, N, device=xyz.device).argsort(1)[:, :m]
    if SAMPLER == "fps":
        return fps_exact(xyz, m)
    G = max(1, N // FPS_CHUNK)
    L = N // G                                         # < G points are dropped from candidacy
    per = min(L, -(-m // G))
    order = morton(xyz).argsort(1)[:, :G * L]
    sub = xyz.gather(1, order[..., None].expand(-1, -1, 3)).reshape(B * G, L, 3)
    loc = fps_exact(sub, per).reshape(B, G, per)
    gi = order.reshape(B, G, L).gather(2, loc)
    return gi.transpose(1, 2).reshape(B, G * per)[:, :m]


@torch.no_grad()
def knn_dist(q, pc, k):
    """(idx, dist) of the k nearest, ascending. Blocked over queries; no autograd path
    exists through indices or the radius mask, so the distance block is transient."""
    B, M = q.shape[:2]
    N = pc.shape[1]
    k = min(k, N)
    di = torch.empty(B, M, k, dtype=torch.long, device=q.device)
    dd = torch.empty(B, M, k, device=q.device)
    step = max(1, min(QCHUNK, int(QBYTES // (B * N * 4))))   # hold the block under QBYTES
    for a in range(0, M, step):
        b = min(a + step, M)
        v, i = _peak(torch.cdist(q[:, a:b], pc), "cdist", False).topk(k, largest=False, dim=-1)
        dd[:, a:b], di[:, a:b] = v, i
    return di, dd


@torch.no_grad()
def ball_group(q, pc, radius, k, keepmin):
    """ball query as the k NEAREST points masked to the radius (PointNet++ takes k
    arbitrary points inside it, which is strictly worse), with the keepmin nearest
    always kept so no neighbourhood is empty."""
    idx, d = knn_dist(q, pc, k)
    ins = d <= radius
    ins[..., :min(keepmin, ins.shape[-1])] = True      # distances are sorted ascending
    return idx, ins


def gather_nb(x, idx):
    B, M, k = idx.shape
    C = x.shape[-1]
    return _peak(x.gather(1, idx.reshape(B, M * k, 1).expand(-1, -1, C)).view(B, M, k, C),
                 "grouped")


class Norm(nn.Module):
    """BatchNorm over channels of an arbitrarily shaped (..., C) tensor, PointNeXt-style."""
    def __init__(self, c):
        super().__init__()
        self.kind = NORM                               # frozen at construction, not read later
        self.n = (nn.BatchNorm1d(c) if NORM == "bn" else
                  nn.LayerNorm(c) if NORM == "ln" else nn.Identity())

    def forward(self, x):
        if self.kind != "bn":
            return self.n(x)
        return self.n(x.reshape(-1, x.shape[-1])).view(x.shape)


class LocalAgg(nn.Module):
    """grouped MLP + masked max-pool. Relative positions normalised by the ball radius."""
    def __init__(self, cin, cout):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(cin + 3, cout), Norm(cout), nn.ReLU())

    def forward(self, pc, f, q, idx, ins, radius):
        rel = (gather_nb(pc, idx) - q[:, :, None, :]) / radius
        x = rel if f is None else torch.cat([rel, gather_nb(f, idx)], -1)
        return _peak(self.mlp(x), "grouped-mlp").masked_fill(~ins[..., None], -1e4).max(2).values


class InvRes(nn.Module):
    """PointNeXt inverted-residual: local aggregation, expand, project, residual, act."""
    def __init__(self, c, exp):
        super().__init__()
        self.la = LocalAgg(c, c)
        self.pw = nn.Sequential(nn.Linear(c, exp * c), Norm(exp * c), nn.ReLU(),
                                nn.Linear(exp * c, c), Norm(c))
        self.act = nn.ReLU()

    def forward(self, pc, x, idx, ins, radius):
        return self.act(x + self.pw(self.la(pc, x, pc, idx, ins, radius)))


class SA(nn.Module):
    """set abstraction: FPS -> ball query -> grouped MLP -> BLOCKS InvResMLP at the new level."""
    def __init__(self, cin, cout, nblk, exp):
        super().__init__()
        self.down = LocalAgg(cin, cout)
        self.blocks = nn.ModuleList([InvRes(cout, exp) for _ in range(nblk)])

    def forward(self, pc, x, npoint, radius):
        sub = fps_sample(pc, npoint)
        q = pc.gather(1, sub[..., None].expand(-1, -1, 3))
        idx, ins = ball_group(q, pc, radius, NSAMPLE, 1)
        y = self.down(pc, x, q, idx, ins, radius)
        if len(self.blocks):
            sidx, sins = ball_group(q, q, radius, NSAMPLE, 1)
            for b in self.blocks:
                y = b(q, y, sidx, sins, radius)
        return q, y


class FP(nn.Module):
    """feature propagation: 3-NN inverse-distance interpolation + skip + MLP."""
    def __init__(self, c_lo, c_skip, cout):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(c_lo + c_skip, cout), Norm(cout), nn.ReLU(),
                                 nn.Linear(cout, cout), Norm(cout), nn.ReLU())

    def forward(self, p_lo, f_lo, p_hi, f_hi):
        idx, d = knn_dist(p_hi, p_lo, 3)
        w = 1.0 / d.clamp(min=1e-6)
        w = w / w.sum(-1, keepdim=True)
        up = (gather_nb(f_lo, idx) * w[..., None]).sum(2)
        return self.mlp(torch.cat([up, f_hi], -1))


# --------------------------------------------------------------- head (from gpu_screen.py)
class Head(nn.Module):
    """one refinement pass: bounded offset, then a softmax convex combination of the
    KHEAD gathered surface points. Output stays on the surface by construction."""
    def __init__(self, C, max_off):
        super().__init__()
        self.emb, self.embO = nn.Embedding(NL, 32), nn.Embedding(NL, 32)
        self.offset = nn.Sequential(nn.Linear(2 * C + 32, 256), nn.ReLU(), nn.Dropout(DROPOUT),
                                    nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 3))
        self.attn = nn.Sequential(nn.Linear(C + 3 + 32, 128), nn.ReLU(),
                                  nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        # read the globals in the BODY, never as a default argument: Python binds defaults
        # at import, which would silently ignore a cfg that rebinds KHEAD before building
        self.C, self.max_off, self.k = C, max_off, KHEAD

    def gather(self, pc, h, q, radius):
        idx, ins = ball_group(q, pc, radius, self.k, KEEPMIN)
        return gather_nb(h, idx), gather_nb(pc, idx), ins

    def forward(self, pc, h, q, radius, diag=False):
        fK, _, mask = self.gather(pc, h, q, radius)
        w = mask.float()[..., None]
        ctx = torch.cat([(fK * w).sum(2) / w.sum(2).clamp(min=1),
                         fK.masked_fill(~mask[..., None], -1e4).max(2).values], -1)
        ar = torch.arange(NL, device=pc.device)
        eo = self.embO(ar)[None].expand(pc.shape[0], -1, -1)
        off = self.offset(torch.cat([ctx, eo], -1))
        if self.max_off is not None:
            off = self.max_off * torch.tanh(off / self.max_off)
        q1 = q + off
        fK2, pK2, mask2 = self.gather(pc, h, q1, radius)
        rel = (pK2 - q1[:, :, None, :]) / SCALE
        e = self.emb(ar)[None, :, None, :].expand(pc.shape[0], NL, pK2.shape[2], 32)
        logit = self.attn(torch.cat([fK2, rel, e], -1)).squeeze(-1).masked_fill(~mask2, -1e4)
        wa = torch.softmax(logit, -1)
        q2 = (wa[..., None] * pK2).sum(2)
        return q1, q2, ({"w": wa.detach(), "pK": pK2.detach()} if diag else None)


class PointNeXtLandmark(nn.Module):
    """PointNeXt encoder/decoder + NPASS untied offset/snap passes + per-contour smoothing.

    forward(pc, q0, ft=None, diag=False) -> (outs, final, info)
      pc    (B,NPTS,3) canonical-frame surface samples, millimetres
      q0    (B,85,3)   coarse landmarks in the same frame
      ft    (B,NPTS,F) optional per-point features (normals when USE_NRM)
      outs  list of NPASS (q1, q2) pairs for deep supervision, ordered coarse->fine
      final (B,85,3)
      info  per-pass {"w","pK"} when diag, else [None]*NPASS
    """
    def __init__(self, npts=NPTS, stages=STAGES, width=WIDTH, npass=NPASS,
                 nfinal=NFINAL, nsample=NSAMPLE, blocks=BLOCKS, exp=EXPANSION,
                 cin=3 + (3 if USE_NRM else 0)):
        super().__init__()
        self.N, self.S, self.R = level_plan(npts, stages, nfinal, nsample)
        self.RP, self.OP, self.LEV = pass_plan(npass, npts, stages, nfinal, nsample)
        self.L, self.npass = stages, npass
        w0 = max(32, width // 4)
        enc = [w0] + [min(2 * width, w0 * 2 ** k) for k in range(1, stages + 1)]
        dec = [width] + enc[1:]
        self.stem = nn.Sequential(nn.Linear(cin, w0), Norm(w0), nn.ReLU(),
                                  nn.Linear(w0, w0), Norm(w0), nn.ReLU())
        self.sas = nn.ModuleList([SA(enc[k], enc[k + 1], blocks, exp) for k in range(stages)])
        self.ups = nn.ModuleList([FP(dec[k + 1], enc[k], dec[k]) for k in range(stages)])
        # only the levels some pass (or the contour net) actually reads get a head projection,
        # so no parameter is left without a gradient path
        self.used = sorted(set(self.LEV) | {0})
        self.proj = nn.ModuleDict({str(k): nn.Sequential(nn.Linear(dec[k] + enc[stages], width),
                                                        nn.ReLU()) for k in self.used})
        self.heads = nn.ModuleList([Head(width, self.OP[i]) for i in range(npass)])
        if CONTOUR_NET:
            self.lmfeat = nn.Sequential(nn.Linear(width, 64), nn.ReLU())
            self.embC = nn.Embedding(NL, 32)
            self.cnet = nn.ModuleList([
                nn.Sequential(nn.Conv1d(3 + 64 + 32, 96, 5, padding=2), nn.ReLU(),
                              nn.Conv1d(96, 96, 3, padding=1), nn.ReLU(), nn.Conv1d(96, 3, 1))
                for _ in CONTOURS])
        else:
            self.cnet = None
        self.enc_ch, self.dec_ch = enc, dec

    def contour(self, pc, h, q):
        B = pc.shape[0]
        idx, _ = knn_dist(q, pc, 1)
        f = self.lmfeat(gather_nb(h, idx).squeeze(2))
        e = self.embC(torch.arange(NL, device=pc.device))[None].expand(B, -1, -1)
        inp = torch.cat([q / SCALE, f, e], -1)
        out = torch.zeros(B, NL, 3, device=pc.device, dtype=q.dtype)
        for (lo, hi), net in zip(CONTOURS, self.cnet):
            out[:, lo:hi + 1] = net(inp[:, lo:hi + 1].transpose(1, 2)).transpose(1, 2)
        return q + out

    def forward(self, pc, q0, ft=None, diag=False):
        # train_family.py's augmentation feeds sub_frac*N points, evaluation feeds all N.
        # Hold the LADDER (ratios, channels, pass->level map) fixed and move the physical
        # quantities with the actual density: N_k scales by f, spacing by 1/sqrt(f).
        dens = pc.shape[1] / self.N[0]
        NK = [max(4, int(round(n * dens))) for n in self.N]
        RK = [r / math.sqrt(dens) for r in self.R]
        x = self.stem(pc / SCALE if ft is None else torch.cat([pc / SCALE, ft], -1))
        ps, xs = [pc], [x]
        for k, sa in enumerate(self.sas):
            p, x = sa(ps[-1], xs[-1], min(NK[k + 1], ps[-1].shape[1]), RK[k + 1])
            ps.append(p); xs.append(x)
        g = xs[-1].max(1).values                        # global token, PointNeXt bottleneck
        dec = [None] * (self.L + 1)
        dec[self.L] = xs[self.L]
        for k in range(self.L, 0, -1):
            dec[k - 1] = self.ups[k - 1](ps[k], dec[k], ps[k - 1], xs[k - 1])
        feats = {k: self.proj[str(k)](torch.cat(
            [dec[k], g[:, None, :].expand(-1, dec[k].shape[1], -1)], -1)) for k in self.used}
        outs, info, q = [], [], q0
        for i, hd in enumerate(self.heads):
            k = self.LEV[i]
            q1, q2, dg = hd(ps[k], feats[k], q, self.RP[i], diag)
            outs.append((q1, q2)); info.append(dg); q = q2
        return outs, (self.contour(ps[0], feats[0], q) if self.cnet is not None else q), info


class MODEL(nn.Module):
    """train_family.py FAMILY CONTRACT adapter: cls(cfg, meta), forward(batch) -> dict.

    The layers read NSAMPLE / KHEAD / NORM / SAMPLER / ... as module globals (gpu_screen's
    idiom), so a cfg that overrides them rebinds those globals ONCE here, before any
    submodule is constructed. One model per process, which is how the driver runs.
    """
    DEFAULTS = dict(npts=NPTS, width=WIDTH, stages=STAGES, npass=NPASS, dropout=DROPOUT,
                    use_nrm=USE_NRM, nsample=NSAMPLE, nfinal=NFINAL, blocks=BLOCKS,
                    expansion=EXPANSION, norm=NORM, khead=KHEAD, keepmin=KEEPMIN,
                    r_first=R_FIRST, r_last=R_LAST, o_first=O_FIRST, o_last=O_LAST,
                    sampler=SAMPLER, fps_chunk=FPS_CHUNK, qbytes=QBYTES,
                    contour_net=CONTOUR_NET, aug_jit=round(AUG_JIT, 4))
    SEARCH_SPACE = dict(width=[128, 256, 384], stages=[4, 6, 8], npass=[4, 6, 8],
                        dropout=[0.0, 0.1, 0.2], nsample=[16, 32, 48], blocks=[1, 2],
                        expansion=[2, 4], norm=["bn", "ln"], khead=[32, 48, 64],
                        r_first=[9.0, 11.0, 13.0], r_last=[2.0, 2.5, 3.5],
                        nfinal=[32, 64, 128],
                        aug_jit=[round(f * AUG_JIT, 4) for f in (0.5, 1.0, 2.0)])
    NEEDS = ("nrm",) if USE_NRM else ()      # class-level, so USE_NRM must come from the env
    ROTATES = ("nrm",)
    SAMPLES = 1

    def __init__(self, cfg, meta):
        super().__init__()
        assert int(cfg.get("use_nrm", USE_NRM)) == USE_NRM, (
            "set USE_NRM in the environment, not CFG_USE_NRM -- train_family.py reads "
            "cls.NEEDS before instantiating, so the normals channel is fixed by then")
        assert int(cfg["npts"]) == meta["npts"], (
            f"cfg npts={cfg['npts']} but DATA has {meta['npts']} points per cloud; the "
            "radius ladder is in millimetres and depends on the density, so point the "
            "run at matching DATA or set CFG_NPTS to the file's N")
        # AUG_JIT is derived from the IMPORT-time NPTS, the data density from meta, and
        # train_family's inherited 0.25 mm would pass every shape check while displacing
        # each point further than its neighbour -- see AUGMENTATION JITTER in the
        # docstring. Tie it to the density that is actually being trained on.
        assert NPTS == meta["npts"], (
            f"the ENVIRONMENT NPTS={NPTS} but DATA has {meta['npts']} points. NPTS is what "
            f"calibrates the density-derived DEFAULTS (aug_jit={AUG_JIT:.4f}), and CFG_NPTS "
            f"cannot re-derive them because DEFAULTS is built at import. Set NPTS={meta['npts']} "
            f"in the environment (CFG_NPTS alone is not enough).")
        s0 = spacing(meta["npts"])
        assert float(cfg["aug_jit"]) <= s0, (
            f"aug_jit={cfg['aug_jit']} mm/coord is {float(cfg['aug_jit'])*math.sqrt(3)/s0:.2f}x "
            f"the {s0:.4f} mm mean point spacing at {meta['npts']} points, so the augmenter "
            f"erases the sub-millimetre detail this family exists to resolve, at train time "
            f"only. Set CFG_AUG_JIT<={s0:.4f} (the density-scaled default is "
            f"{0.25*math.sqrt(2048.0/meta['npts']):.4f}) or run at a lower NPTS.")
        global NSAMPLE, KHEAD, KEEPMIN, NORM, DROPOUT, SAMPLER, FPS_CHUNK, QBYTES
        global R_FIRST, R_LAST, O_FIRST, O_LAST, CONTOUR_NET
        NSAMPLE, KHEAD, KEEPMIN = int(cfg["nsample"]), int(cfg["khead"]), int(cfg["keepmin"])
        NORM, DROPOUT = str(cfg["norm"]), float(cfg["dropout"])
        SAMPLER, FPS_CHUNK, QBYTES = str(cfg["sampler"]), int(cfg["fps_chunk"]), float(cfg["qbytes"])
        R_FIRST, R_LAST = float(cfg["r_first"]), float(cfg["r_last"])
        O_FIRST, O_LAST = float(cfg["o_first"]), float(cfg["o_last"])
        CONTOUR_NET = int(cfg["contour_net"])
        self.net = PointNeXtLandmark(npts=meta["npts"], stages=int(cfg["stages"]),
                                     width=int(cfg["width"]), npass=int(cfg["npass"]),
                                     nfinal=int(cfg["nfinal"]), nsample=NSAMPLE,
                                     blocks=int(cfg["blocks"]), exp=int(cfg["expansion"]),
                                     cin=3 + (3 if USE_NRM else 0))

    def forward(self, b):
        outs, final, _ = self.net(b["pc"], b["coarse"], b.get("nrm"))
        return {"pred": final, "aux": [q for pair in outs for q in pair]}

    def loss(self, out, tg):
        """gpu_screen.py's schedule verbatim: within a pass the pre-snap offset gets 0.4
        and the snapped point 1.0; across passes the weight halves backwards from the last."""
        a = out["aux"]
        n = len(a) // 2
        w = [0.5 ** (n - 1 - t) for t in range(n)]
        s = sum(w)
        L = sum((w[t] / s) * (0.4 * ((a[2 * t] - tg) ** 2).sum(-1).mean()
                              + ((a[2 * t + 1] - tg) ** 2).sum(-1).mean()) for t in range(n))
        return L + ((out["pred"] - tg) ** 2).sum(-1).mean()


def plan_report(npts=NPTS, stages=STAGES, width=WIDTH, npass=NPASS, nsample=NSAMPLE,
                nfinal=NFINAL, batch=1):
    """the millimetre plan plus the analytic size of the largest activation."""
    N, S, R = level_plan(npts, stages, nfinal, nsample)
    RP, OP, LEV = pass_plan(npass, npts, stages, nfinal, nsample)
    w0 = max(32, width // 4)
    enc = [w0] + [min(2 * width, w0 * 2 ** k) for k in range(1, stages + 1)]
    print(f"  levels (ratio {(npts/nfinal)**(1/stages):.2f}x per stage)")
    print(f"    {'k':>2s} {'N':>7s} {'spacing':>8s} {'radius':>7s} {'strided':>8s} {'ch':>5s}")
    for k in range(stages + 1):
        # 'radius' is the mask radius AND the same-level InvResMLP extent; 'strided' is
        # the real extent of the k-1 -> k downsampling, which is k-NN-bound at the finer
        # level's radius and so ignores the mask entirely (see the docstring).
        st = f"{R[k-1]:6.2f}mm" if k else f"{'--':>8s}"
        print(f"    {k:2d} {N[k]:7d} {S[k]:7.4f}mm {R[k]:6.2f}mm {st:>8s} {enc[k]:5d}")
    print(f"  aug_jit {AUG_JIT:.4f}mm/coord = {AUG_JIT*math.sqrt(3)/S[0]:.2f}x the level-0 "
          f"spacing ({S[0]:.4f}mm); train_family's inherited 0.25 would be "
          f"{0.25*math.sqrt(3)/S[0]:.2f}x")
    print(f"  passes  radius/offset mm -> level")
    print("    " + "  ".join(f"{RP[i]:.2f}/{OP[i]:.2f}->L{LEV[i]}" for i in range(npass)))
    big = max((batch * N[k + 1] * nsample * enc[k + 1] * 4, f"grouped L{k+1}")
              for k in range(stages))
    cd = (batch * max(1, min(QCHUNK, int(QBYTES // (batch * N[0] * 4)))) * N[0] * 4, "cdist block")
    for nb, what in (big, cd):
        print(f"  analytic {what:14s} {nb/1e6:8.1f} MB at batch {batch}")
    return N, S, R, RP, OP, LEV


if __name__ == "__main__":
    torch.manual_seed(0); np.random.seed(0)
    B, P = int(os.environ.get("SMOKE_B", "2")), NPTS
    print(f"config NPTS={NPTS} WIDTH={WIDTH} STAGES={STAGES} NPASS={NPASS} "
          f"NSAMPLE={NSAMPLE} BLOCKS={BLOCKS} EXPANSION={EXPANSION} NORM={NORM} "
          f"DROPOUT={DROPOUT} USE_NRM={USE_NRM} SAMPLER={SAMPLER} K2R={K2R:.4f}", flush=True)
    plan_report(batch=B)

    # synthetic ear-sized surface: an ellipsoid whose area matches the real crop
    # (2048 pts x 3.535 mm^2/pt = 7240 mm^2), so ball masks are populated realistically.
    def shell(n, gen):
        v = torch.randn(n, 3, generator=gen)
        return (v / v.norm(dim=1, keepdim=True)) * torch.tensor([30.0, 22.0, 14.0])
    gen = torch.Generator().manual_seed(1)
    pc = torch.stack([shell(P, gen) for _ in range(B)])
    nrm = pc / pc.norm(dim=-1, keepdim=True)
    ang = torch.linspace(0, 2 * math.pi, NL)
    ring = torch.stack([30 * ang.cos(), 22 * ang.sin(), 14 * (2 * ang).sin() * 0.3], -1)
    q0 = ring[None].expand(B, -1, -1) + torch.randn(B, NL, 3, generator=gen) * 1.5

    net = PointNeXtLandmark()
    npar = sum(p.numel() for p in net.parameters())
    print(f"params: {npar:,}   enc {net.enc_ch}  dec {net.dec_ch}  head levels {net.LEV}",
          flush=True)

    t0 = time.time()
    outs, final, info = net(pc, q0, nrm if USE_NRM else None, diag=True)
    tf = time.time() - t0
    loss = sum(((q1 - q0) ** 2).sum(-1).mean() + ((q2 - q0) ** 2).sum(-1).mean()
               for q1, q2 in outs) + ((final - q0) ** 2).sum(-1).mean()
    t1 = time.time(); loss.backward(); tb = time.time() - t1
    gn = sum(float(p.grad.norm()) for p in net.parameters() if p.grad is not None)

    print(f"forward {tf:.1f}s  backward {tb:.1f}s  loss {float(loss):.3f}  grad-norm-sum {gn:.1f}")
    print(f"passes {len(outs)}  q1 {tuple(outs[0][0].shape)}  q2 {tuple(outs[-1][1].shape)}")
    print(f"final  {tuple(final.shape)}")
    print(f"largest tensor allocated: {PEAK['bytes']/1e6:.1f} MB  ({PEAK['what']})")
    ret = PEAK["retained"] / B
    print(f"grouped activations autograd keeps: {PEAK['retained']/1e6:.0f} MB at B={B} "
          f"({ret/1e6:.0f} MB/ear) -> {int((44e9 - 12 * npar * 4) / ret)} ears fit in 48GB "
          f"(44GB usable, AdamW states, grouped tensors only -- a lower bound, so halve it)")

    # the snap must be a convex combination of gathered SURFACE points, every pass
    for i, (dg, (_, q2)) in enumerate(zip(info, outs)):
        w, pK = dg["w"], dg["pK"]
        assert float((w.sum(-1) - 1).abs().max()) < 1e-5, f"pass {i}: weights do not sum to 1"
        assert float(w.min()) >= 0.0, f"pass {i}: negative weight"
        rec = (w[..., None] * pK).sum(2)
        assert float((rec - q2).abs().max()) < 1e-4, f"pass {i}: q2 is not the convex sum"
        lo, hi = pK.amin(2), pK.amax(2)
        assert bool(((q2 >= lo - 1e-4) & (q2 <= hi + 1e-4)).all()), f"pass {i}: q2 outside hull box"
    # report the WORST landmark, per ear against ITS OWN cloud. A min over all 85*B
    # points (or a cross-ear comparison) flatters this by an order of magnitude: the snap
    # is a convex combination of <=KHEAD neighbours, so it lands INTERIOR to them, not on
    # one of them, and interior is what the numbers below have to show.
    def to_cloud(P):
        return torch.stack([torch.cdist(P[b], pc[b]).min(1).values for b in range(B)])
    d0, dF = to_cloud(outs[-1][1]), to_cloud(final)
    print(f"convex-combination property holds for all {len(outs)} passes; distance from a "
          f"level-0 cloud point, over all {B}x{NL} landmarks:"
          f"\n    last-pass snap q2       mean {float(d0.mean()):.4f}mm  WORST "
          f"{float(d0.max()):.4f}mm  (<= level-0 spacing {net.S[0]:.4f}mm x a small factor)"
          f"\n    final (post contour-net) mean {float(dF.mean()):.4f}mm  WORST "
          f"{float(dF.max()):.4f}mm  -- the contour net is a FREE offset after the snap, "
          f"so the network OUTPUT is not on the surface (gpu_screen.py behaves the same "
          f"way; surface projection happens in train_family's full pipeline)")

    if B == 2:
        assert final.shape == (2, NL, 3), final.shape
    assert final.shape == (B, NL, 3), final.shape
    assert torch.isfinite(final).all()
    for n, p in net.named_parameters():
        assert p.grad is not None, f"no gradient reached {n}"

    # train_family.py's augmentation subsamples to sub_frac*N -- the ladder must follow
    keep = torch.randperm(P, generator=gen)[:int(P * 0.625)]
    sub = pc[:, keep]
    _, fsub, _ = net(sub, q0, nrm[:, keep] if USE_NRM else None)
    assert fsub.shape == (B, NL, 3) and torch.isfinite(fsub).all()
    print(f"sub_frac=0.625 path ({sub.shape[1]} pts): {tuple(fsub.shape)}, "
          f"level-0 radius rescaled {net.R[0]:.2f} -> {net.R[0]/math.sqrt(0.625):.2f}mm")

    # the shared-driver contract
    m = MODEL({**MODEL.DEFAULTS}, dict(nl=NL, contours=CONTOURS, scale=SCALE, npts=P,
                                       fold=0, dev="cpu", n_train_ears=272, artefacts={}))
    b = {"pc": pc, "coarse": q0, "ear": torch.zeros(B, dtype=torch.long)}
    o = m({**b, **({"nrm": nrm} if MODEL.NEEDS else {})})
    ls = m.loss(o, q0)
    ls.backward()
    assert o["pred"].shape == (B, NL, 3) and len(o["aux"]) == 2 * NPASS
    print(f"MODEL adapter: pred {tuple(o['pred'].shape)}  aux {len(o['aux'])}  "
          f"loss {float(ls):.3f}  NEEDS {MODEL.NEEDS}  SAMPLES {MODEL.SAMPLES}  "
          f"aug_jit {MODEL.DEFAULTS['aug_jit']}mm (train_family default 0.25)")

    # the three footguns that silently invalidate a run must be HARD errors, not comments.
    # Every case is expressed RELATIVE to the running NPTS, so the checks hold at any
    # resolution the smoke test is invoked with (an earlier version hard-coded 32768 and
    # quietly tested nothing at all when NPTS was already 32768).
    def meta_at(n):
        return dict(nl=NL, contours=CONTOURS, scale=SCALE, npts=n, fold=0, dev="cpu",
                    n_train_ears=272, artefacts={})
    for bad, mt, want in (
            (dict(npts=P // 2), meta_at(P), "points per cloud"),          # cfg vs DATA N
            (dict(npts=2 * P), meta_at(2 * P), "ENVIRONMENT NPTS"),       # env vs DATA N
            (dict(aug_jit=spacing(P) * 1.01), meta_at(P), "mean point spacing")):
        try:
            MODEL({**MODEL.DEFAULTS, **bad}, mt)
            raise SystemExit(f"MODEL accepted {bad} -- the '{want}' guard is dead")
        except AssertionError as e:
            assert want in str(e), f"wrong guard fired for {bad}: {e}"
    print(f"refused: cfg npts != DATA npts | env NPTS != DATA npts | aug_jit > the "
          f"{spacing(P):.4f}mm spacing (train_family's inherited 0.25 is "
          f"{0.25*math.sqrt(3)/spacing(32768):.2f}x it at 32768 pts)")
    print(f"OK  landmark output {tuple(final.shape)}  {npar:,} params  {tf+tb:.1f}s total")
