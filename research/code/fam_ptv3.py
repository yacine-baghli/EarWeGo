"""
PTv3 BACKBONE for the offset/snap head — serialization-based attention instead of kNN.

Why this family exists. The 813k-param static-DGCNN refinement family is exhausted:
seven single-change variants were all null, and the error decomposition is invariant
across them (along-contour RMSE 1.4456 = 77% of the energy, across-contour 0.7233,
normal 0.2822). The dominant error is ORDERED CORRESPONDENCE — phase along the
contour — not local XYZ accuracy, and a kNN EdgeConv stack cannot see phase because
its receptive field is an unordered ball that grows only ~5mm over three layers. The
bet here is that a backbone whose feature at every point comes from a LONG, ORDERED,
GLOBALLY CONSISTENT sweep of the surface (a space-filling curve through the whole crop,
attention over windows of that sweep, the window offset and the curve itself reshuffled
every block) exposes the along-surface position information phase needs. Point
Transformer V3, Wu et al. CVPR 2024.

The mechanism, all implemented here from scratch in torch (no torch_geometric, no
sparse-conv library, no external Hilbert package):
  * quantise to a voxel grid, encode each cell with a space-filling curve
    (Z-order/Morton, or Hilbert via Skilling's AxestoTranspose), sort by the code;
  * attention over CONTIGUOUS FIXED-SIZE PATCHES of that order — O(N*patch) not
    O(N^2), and no neighbour search anywhere in the backbone;
  * xCPE position encoding: a depthwise conv along the serialized order, which is also
    the only thing carrying information ACROSS patch boundaries inside a block;
  * the serialization is re-planned every block (4 axis permutations x curve x reversal
    x a half-patch roll of the window boundaries);
  * U-Net stages: grid coarsening (the voxel grows by VOXGROW per stage) with a
    fine->coarse index map for exact unpooling, plus a skip connection.

MEASURED, not assumed (smoke test step 4): the receptive field is already GLOBAL.
Perturbing one input point moves all 8192 output features — with n/patch = 32 windows a
single axis permutation scatters one contiguous window across every other window, so two
blocks suffice. Do not read the growth story as gradual. `patch` controls how much
surface is compared IN ONE ATTENTION, not whether distant points interact; that is the
knob to sweep, and patch=1024 is the interesting end of it.

The head is the SHIPPED offset/snap head and contour smoother, copied from
gpu_screen.py, so a run of this file is a backbone swap and nothing else.

    NPTS=8192 CURVE=hilbert WIDTH=48 PATCH=256 python research/code/fam_ptv3.py
    FAMILY=ptv3 FOLD=0 SEED=0 DATA=screen_data_4096.npz \
      CFG_NPTS=4096 CFG_POOLR=2 CFG_VOXEL=0.9 python3 train_family.py     # the real runs

VOXEL/VOXGROW must match the point spacing, and the defaults are MEASURED on the real
340-ear canonical clouds rather than guessed. Occupied cells per ear at voxel v (grid
anchored at the origin, mean over 40 ears, 4096-POINT clouds): 1.4mm 2702, 1.7mm 2274,
2.0mm 1923, 2.8mm 1230, 3.4mm 903, 4.0mm 679, 5.3mm 401. This is a fold-over-fold SHEET,
so the count falls far more slowly than 1/v^2 near the sampling limit: doubling the voxel
per stage does NOT quarter it, and VOXGROW=2 leaves stage 2 at 1.76x its slot budget
(which the merge then absorbs). VOXGROW=2.5 brings both poolings to ~0.8-1.1x budget, so
grid coarsening genuinely decides the pooling. The smoke test prints occupied -> slots for
every pooling; if occupied greatly exceeds the slots, raise VOXGROW.

Occupancy depends on BOTH the voxel and N — at these voxels the SAMPLING, not the
geometry, is the binding limit (2.125mm: 1214 cells at 2048 points, 1783 at 4096, 2194 at
8192) — so POOLR has to follow the cloud size. Stage-1/stage-2 occupancy as a fraction of
the slot budget, each row measured on clouds of ITS OWN size (the 8192 row on two
independent 4096-samples of one ear concatenated, since screen_data_8192.npz does not
exist; the stage-2 figure is taken on the raw cloud and is therefore a slight OVER-
estimate, because that pooling actually runs on stage-1 centroids):

    NPTS=8192  POOLR=4  VOXEL=0.85   1.07 / 0.82    <- the defaults. Stage 1 sits just
                                       OVER budget: 2194 occupied voxels into 2048 slots,
                                       so the uniform along-curve merge absorbs ~150 of
                                       them. Raise VOXGROW to ~2.8 to clear it.
    NPTS=4096  POOLR=2  VOXEL=0.9    0.81 / 0.35
    NPTS=2048  POOLR=2  VOXEL=1.3    0.76 / 0.32

NOTE: scratch/ holds 2048- and 4096-point clouds only. An 8192-point run needs a
screen_data_8192.npz that does not exist yet; until it does, use the 4096 row. The family
REFUSES a cfg npts that disagrees with the file it was handed, because that mismatch is
invisible in every shape check and only shows up as a pooling ladder that no longer
coarsens by the grid — so CFG_NPTS travels with DATA, POOLR and VOXEL. Under
search_driver.py use the UNPREFIXED names (NPTS=4096 POOLR=2, which reach DEFAULTS at
import): trial_env deliberately drops every CFG_* it did not itself sample, so a
CFG_NPTS exported for a sweep would vanish and every trial would refuse.

Warmup matters — this is a pre-norm transformer, not the baseline CNN. At the baseline's
flat lr=1.5e-3 the loss bounces; with 30 steps of linear warmup to 1e-3 it descends
cleanly. Trainability evidence, independently reproduced: the first 8 fold-0 TRAIN ears at
NPTS=2048 POOLR=2 VOXEL=1.3, full-batch AdamW 1e-3 behind that warmup, with the pass terms
left unnormalised, overfit 4.344 -> 2.048 (step 100) -> 1.224 (200) -> 0.639 mm (300),
loss 138 -> 7.1 and still falling; GT enters the loss and nothing else. It says NOTHING
about generalisation, and 300 steps on 8 ears is not a converged overfit — it only shows
gradients reach every part of the architecture and the objective descends.

Two entry points, one implementation:
  MODEL = PTv3Family    the train_family.py contract — cfg/meta in, {'pred','aux'} out.
                        Config comes from CFG_* (which override the env defaults below).
  Net                   the gpu_screen.Net signature, for standalone use:
                        outs, final, per = Net(cfg=None)([pc], q0, [ft])
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

# Env defaults. train_family.py merges these UNDER its CFG_* variables, so a search
# driver sweeps CFG_PATCH=1024 while a bare `python fam_ptv3.py` reads PATCH=1024.
DEFAULTS = dict(
    npts=int(os.environ.get("NPTS", "8192")),        # smoke-test cloud size; N is per-call
    width=int(os.environ.get("WIDTH", "48")),        # stage-0 channels; doubles per stage
    stages=int(os.environ.get("STAGES", "3")),       # encoder stages (stages-1 poolings)
    depth=int(os.environ.get("DEPTH", "2")),         # attention blocks per stage
    patch=int(os.environ.get("PATCH", "256")),       # attention window along the curve
    curve=os.environ.get("CURVE", "hilbert"),        # morton | hilbert | both (alternates)
    npass=int(os.environ.get("NPASS", "4")),         # offset/snap refinement passes
    untied=int(os.environ.get("UNTIED", "0")),       # 1 = per-pass head weights
    dropout=float(os.environ.get("DROPOUT", "0.1")),
    use_nrm=int(os.environ.get("USE_NRM", "0")),     # 1 = concat oriented normals
    use_crv=int(os.environ.get("USE_CRV", "0")),     # 1 = concat curvature (build_curv_data)
    n_crv=int(os.environ.get("N_CRV", "12")),        # its channel count, len(crv_names)
    heads=int(os.environ.get("HEADS", "4")),         # must divide every stage width
    mlp=int(os.environ.get("MLP", "2")),             # feed-forward expansion ratio
    voxel=float(os.environ.get("VOXEL", "0.85")),    # stage-0 voxel edge, mm
    voxgrow=float(os.environ.get("VOXGROW", "2.5")), # voxel growth per stage (see below)
    bits=int(os.environ.get("BITS", "9")),           # grid half-extent = voxel*2**(bits-1)
    poolr=int(os.environ.get("POOLR", "4")),         # nominal point ratio per pooling
    headc=int(os.environ.get("HEADC", "256")),       # channels handed to the head
    k=int(os.environ.get("K", "48")),                # head's landmark window, as shipped
)
SCALE, NL = 30.0, 85
CONTOURS = [(0, 24), (25, 54), (55, 74), (75, 84)]
PERMS = [(0, 1, 2), (1, 2, 0), (2, 0, 1), (0, 2, 1)]
BIG = torch.iinfo(torch.int64).max


class Cfg(dict):
    """attribute access, so the blocks read c.patch and stay close to gpu_screen's idiom"""
    __getattr__ = dict.__getitem__


def config(cfg=None):
    """Fill anything the caller left out from the env defaults, then check consistency.
    train_family.py hands over its whole cfg, training keys included; extras are kept."""
    c = Cfg({**DEFAULTS, **(cfg or {})})
    c["widths"] = [c.width * 2 ** s for s in range(c.stages)]
    assert all(w % c.heads == 0 for w in c.widths), \
        f"heads {c.heads} must divide every stage width {c.widths}"
    assert c.bits - (c.stages - 1) >= 2, "not enough grid bits left for the coarsest stage"
    assert c.voxel * (1 << (c.bits - 1)) > 90.0, \
        (f"grid half-extent {c.voxel * (1 << (c.bits - 1)):.0f}mm cannot hold an ear crop: "
         f"real ears reach 57.6mm along an axis and 65.5mm from the canonical origin, and "
         f"since rotation can put that whole radius on one axis, aug_scale=0.20 takes the "
         f"worst case to 72mm. Raise bits.")
    assert c.curve in ("morton", "hilbert", "both"), c.curve
    return c


# --------------------------------------------------------- space-filling curves
def _interleave(parts, bits):
    """Bit k of parts[i] -> code bit k*n + (n-1-i): Skilling's transpose layout."""
    n = len(parts)
    code = torch.zeros_like(parts[0])
    for k in range(bits):
        for i in range(n):
            code |= ((parts[i] >> k) & 1) << (k * n + (n - 1 - i))
    return code


def morton_code(g, bits):
    """Z-order: plain bit interleave of the integer grid coordinates."""
    return _interleave([g[..., 0], g[..., 1], g[..., 2]], bits)


def hilbert_code(g, bits):
    """3-D Hilbert index, Skilling 2004 AxestoTranspose, vectorised over any shape.

    Consecutive codes are grid cells one step apart — the property Morton lacks, and the
    reason the curve choice can matter. Verified by brute force in the smoke test.
    """
    X = [g[..., 0].clone(), g[..., 1].clone(), g[..., 2].clone()]
    n = len(X)
    Q = 1 << (bits - 1)
    while Q > 1:                                   # inverse undo
        P = Q - 1
        for i in range(n):
            on = (X[i] & Q) != 0
            if i == 0:                             # X[0] aliases X[i]: exchange is a no-op
                X[0] = torch.where(on, X[0] ^ P, X[0])
            else:
                t = (X[0] ^ X[i]) & P
                X[0] = torch.where(on, X[0] ^ P, X[0] ^ t)
                X[i] = torch.where(on, X[i], X[i] ^ t)
        Q >>= 1
    for i in range(1, n):                          # Gray encode
        X[i] = X[i] ^ X[i - 1]
    t = torch.zeros_like(X[0])
    Q = 1 << (bits - 1)
    while Q > 1:
        t = torch.where((X[n - 1] & Q) != 0, t ^ (Q - 1), t)
        Q >>= 1
    return _interleave([x ^ t for x in X], bits)


def grid_coord(pos, voxel, bits):
    """Quantise to an integer grid ANCHORED AT THE CANONICAL ORIGIN, half-extent
    voxel*2**(bits-1) per axis.

    Deliberately not the cloud's own bounding box. The clouds are already in a per-ear
    canonical frame, so a fixed anchor makes cell indices — and therefore the whole
    serialization — comparable between ears and between fresh surface samples of the
    same ear, which is the entire premise of using an ordered sweep to fix phase. A
    bbox anchor would instead make every point's cell depend on one extreme point.
    """
    return (torch.floor(pos / voxel).long() + (1 << (bits - 1))).clamp_(0, (1 << bits) - 1)


def serialize(pos, valid, voxel, bits, curve, perm):
    """Order (B,n) sorting the points along the curve; invalid slots sort last."""
    g = grid_coord(pos, voxel, bits)[..., list(perm)]
    code = (hilbert_code if curve == "hilbert" else morton_code)(g, bits)
    return torch.where(valid, code, torch.full_like(code, BIG)).argsort(1)


def plan(g, curve):
    """Serialization pattern for global block index g: cycle the axis permutation, the
    curve, the direction and the patch phase so a point sits with different neighbours in
    every block. With only n/patch windows this reaches every point in two blocks, not
    gradually — see the receptive-field measurement in the smoke test."""
    cu = curve if curve != "both" else ("morton", "hilbert")[(g // len(PERMS)) % 2]
    return cu, g % len(PERMS), bool((g // 2) % 2), bool(g % 2)


def counts(n, c):
    """points per stage: n/poolr each pooling, rounded up to a whole patch"""
    out = [n]
    for _ in range(c.stages - 1):
        out.append(min(out[-1], max(c.patch, -(-(out[-1] // c.poolr) // c.patch) * c.patch)))
    return out


def _inv(o):
    ar = torch.arange(o.shape[1], device=o.device).expand_as(o)
    return torch.empty_like(o).scatter_(1, o, ar)


# ------------------------------------------------------------ grid pool / unpool
def grid_pool(x, pos, code, valid, n2):
    """Coarsen to n2 slots by grouping on the COARSE voxel code (equal codes are the same
    voxel). Features max-pool, positions mean-pool. Returns the fine->coarse index map
    for exact unpooling, and the occupied-voxel count.

    The occupied-cell count falls even more slowly than 1/v^2 on a real ear (measured in
    the module docstring), so it is never exactly n/poolr. When the coarse grid
    leaves MORE occupied voxels than the n2 slots the tensor budget allows, adjacent
    voxels ALONG THE CURVE are merged uniformly (rank*n2//occupied) instead of dumping the
    tail into the last slot: either way every group is a contiguous run of the coarse
    curve, so locality survives. Slot n2 collects invalid input points and is discarded.
    """
    B, n, C = x.shape
    key = torch.where(valid, code, torch.full_like(code, BIG))
    o = key.argsort(1)                                     # equal voxels become contiguous
    ks, vs = torch.gather(key, 1, o), torch.gather(valid, 1, o)
    new = torch.ones_like(ks, dtype=torch.bool)
    new[:, 1:] = ks[:, 1:] != ks[:, :-1]
    rank = (new & vs).long().cumsum(1) - 1                 # voxel rank along the curve
    occ = (rank.amax(1, keepdim=True) + 1).clamp(min=1)    # occupied coarse voxels
    slot = torch.where(occ > n2, rank * n2 // occ, rank)
    slot = torch.where(vs, slot.clamp(max=n2 - 1), torch.full_like(rank, n2))
    xs = torch.gather(x, 1, o[..., None].expand(-1, -1, C))
    ps = torch.gather(pos, 1, o[..., None].expand(-1, -1, 3))
    si = slot[..., None]
    px = x.new_zeros(B, n2 + 1, C).scatter_reduce_(
        1, si.expand(-1, -1, C), xs, "amax", include_self=False)
    cnt = x.new_zeros(B, n2 + 1, 1).scatter_add_(1, si, torch.ones_like(si, dtype=x.dtype))
    pp = pos.new_zeros(B, n2 + 1, 3).scatter_add_(1, si.expand(-1, -1, 3), ps)
    up = torch.zeros_like(slot).scatter_(1, o, slot).clamp(max=n2 - 1)
    return (px[:, :n2], pp[:, :n2] / cnt[:, :n2].clamp(min=1),
            cnt[:, :n2, 0] > 0, up, float(occ.float().mean()))


def grid_unpool(xc, up):
    return torch.gather(xc, 1, up[..., None].expand(-1, -1, xc.shape[-1]))


# ------------------------------------------------------------------ transformer
class Block(nn.Module):
    """pre-norm patch attention + MLP, with xCPE along the serialized order"""

    def __init__(self, C, c):
        super().__init__()
        self.cpe = nn.Conv1d(C, C, 5, padding=2, groups=C)   # depthwise, along the curve
        self.cpe_lin = nn.Linear(C, C)
        self.n1 = nn.LayerNorm(C); self.qkv = nn.Linear(C, 3 * C); self.pr = nn.Linear(C, C)
        self.n2 = nn.LayerNorm(C)
        self.mlp = nn.Sequential(nn.Linear(C, c.mlp * C), nn.GELU(), nn.Dropout(c.dropout),
                                 nn.Linear(c.mlp * C, C))
        self.dp = nn.Dropout(c.dropout)
        self.heads = c.heads

    def forward(self, x, valid, o, inv, P):
        """valid=None means every slot is occupied, which lets SDPA take the flash path
        (an additive mask forces the far more memory-hungry fallback).

        n need not be a multiple of P: the trainer's augmentation subsamples the cloud to
        sub_frac*npts, so the point count reaching a forward pass differs between training
        and evaluation. The serialized order is padded with masked slots instead.
        """
        B, n, C = x.shape
        xs = torch.gather(x, 1, o[..., None].expand(-1, -1, C))
        w = None if valid is None else torch.gather(valid, 1, o)[..., None].to(x.dtype)
        pad = (-n) % P
        if pad:
            w = x.new_ones(B, n, 1) if w is None else w
            xs, w = Fn.pad(xs, (0, 0, 0, pad)), Fn.pad(w, (0, 0, 0, pad))
        if w is not None:
            xs = xs * w
        xs = xs + self.cpe_lin(self.cpe(xs.transpose(1, 2)).transpose(1, 2))
        npat, hd = (n + pad) // P, C // self.heads
        h = self.n1(xs).reshape(B * npat, P, C)
        q, k, v = self.qkv(h).view(B * npat, P, 3, self.heads, hd).permute(2, 0, 3, 1, 4)
        m = None if w is None else (1.0 - w.view(B * npat, 1, 1, P)) * -1e4   # never -inf
        a = Fn.scaled_dot_product_attention(q, k, v, attn_mask=m)
        xs = xs + self.dp(self.pr(a.transpose(1, 2).reshape(B, n + pad, C)))
        xs = xs + self.dp(self.mlp(self.n2(xs)))
        xs = xs[:, :n] if pad else xs
        if w is not None:                            # padding contributes nothing upward
            xs = xs * w[:, :n]
        return torch.gather(xs, 1, inv[..., None].expand(-1, -1, C))


class PTv3(nn.Module):
    """serialized-attention U-Net: per-point features at full resolution + global mix"""

    def __init__(self, cin, c):
        super().__init__()
        W = c.widths
        self.stem = nn.Sequential(nn.Linear(cin, W[0]), nn.GELU(), nn.Linear(W[0], W[0]))
        self.enc = nn.ModuleList([nn.ModuleList([Block(W[s], c) for _ in range(c.depth)])
                                  for s in range(c.stages)])
        self.dec = nn.ModuleList([nn.ModuleList([Block(W[s], c) for _ in range(c.depth)])
                                  for s in range(c.stages - 1)])
        self.down = nn.ModuleList([nn.Linear(W[s], W[s + 1]) for s in range(c.stages - 1)])
        self.up = nn.ModuleList([nn.Linear(W[s + 1], W[s]) for s in range(c.stages - 1)])
        self.skip = nn.ModuleList([nn.Linear(2 * W[s], W[s]) for s in range(c.stages - 1)])
        self.out = nn.Sequential(nn.Linear(W[0], c.headc), nn.GELU())
        self.mix = nn.Sequential(nn.Linear(2 * c.headc, c.headc), nn.ReLU())
        self.c = c
        self.pool_stats = []                         # (occupied coarse voxels, slots)

    def vox(self, s):
        return self.c.voxel * self.c.voxgrow ** s

    def order(self, s, g, pos, valid, cache, P):
        c = self.c
        curve, pi, rev, roll = plan(g, c.curve)
        ck = (s, curve, pi)
        if ck not in cache:
            with torch.no_grad():
                cache[ck] = serialize(pos, valid, self.vox(s), c.bits - s, curve, PERMS[pi])
        o = cache[ck]
        if rev:
            o = o.flip(1)
        if roll:                                     # move the window boundaries
            o = torch.roll(o, P // 2, 1)
        return o, _inv(o)

    def features(self, pc, ft=None):
        """per-point features at full resolution, BEFORE any global pooling — so a
        receptive field measured on this output is purely the serialized attention"""
        c = self.c
        B, N, _ = pc.shape
        ns = counts(N, c)
        x = self.stem(pc / SCALE if ft is None else torch.cat([pc / SCALE, ft], -1))
        pos = pc
        valid = torch.ones(B, N, dtype=torch.bool, device=pc.device)
        full = True                                  # no padded slots yet -> no mask
        cache, keep, stats, g = {}, [], [], 0
        for s in range(c.stages):
            P = min(c.patch, ns[s])
            for blk in self.enc[s]:
                o, inv = self.order(s, g, pos, valid, cache, P)
                x = blk(x, None if full else valid, o, inv, P); g += 1
            if s == c.stages - 1:
                break
            keep.append([x, pos, valid, full, None])
            with torch.no_grad():
                cc = (hilbert_code if c.curve != "morton" else morton_code)(
                    grid_coord(pos, self.vox(s + 1), c.bits - s - 1), c.bits - s - 1)
            x, pos, valid, up, occ = grid_pool(x, pos, cc, valid, ns[s + 1])
            keep[-1][4] = up
            full = bool(valid.all())
            x = self.down[s](x); stats.append((round(occ, 1), ns[s + 1]))
        for s in reversed(range(c.stages - 1)):
            xf, pos, valid, full, up = keep[s]
            x = self.skip[s](torch.cat([grid_unpool(self.up[s](x), up), xf], -1))
            P = min(c.patch, ns[s])
            for blk in self.dec[s]:
                o, inv = self.order(s, g, pos, valid, cache, P)
                x = blk(x, None if full else valid, o, inv, P); g += 1
        self.pool_stats = stats
        return self.out(x)

    def forward(self, pc, ft=None):
        h = self.features(pc, ft)
        g = h.max(1, keepdim=True).values.expand(-1, h.shape[1], -1)
        return self.mix(torch.cat([h, g], -1))


# ------------------------------------------------------- shipped offset/snap head
def knn(q, pc, k):
    return torch.cdist(q, pc).topk(k, largest=False, dim=-1).indices


class Head(nn.Module):
    """one refinement pass: offset then surface snap — as shipped in gpu_screen.py"""

    def __init__(self, C, c):
        super().__init__()
        self.emb = nn.Embedding(NL, 32); self.embO = nn.Embedding(NL, 32)
        self.offset = nn.Sequential(nn.Linear(2 * C + 32, 256), nn.ReLU(), nn.Dropout(c.dropout),
                                    nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 3))
        self.attn = nn.Sequential(nn.Linear(C + 3 + 32, 128), nn.ReLU(),
                                  nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.C, self.k = C, c.k

    def gather(self, pc, h, q):
        B = pc.shape[0]
        idx = knn(q, pc, self.k).reshape(B, NL * self.k)
        fK = torch.gather(h, 1, idx[..., None].expand(-1, -1, self.C)).view(B, NL, self.k, self.C)
        pK = torch.gather(pc, 1, idx[..., None].expand(-1, -1, 3)).view(B, NL, self.k, 3)
        return fK, pK

    def forward(self, pc, h, q):
        fK, pK = self.gather(pc, h, q)
        ctx = torch.cat([fK.mean(2), fK.max(2).values], -1)
        eo = self.embO(torch.arange(NL, device=pc.device))[None].expand(pc.shape[0], -1, -1)
        q1 = q + self.offset(torch.cat([ctx, eo], -1))
        fK2, pK2 = self.gather(pc, h, q1)
        rel = (pK2 - q1[:, :, None, :]) / SCALE
        e = self.emb(torch.arange(NL, device=pc.device))[None, :, None, :].expand(
            pc.shape[0], NL, self.k, 32)
        w = torch.softmax(self.attn(torch.cat([fK2, rel, e], -1)).squeeze(-1), -1)
        return q1, (w[..., None] * pK2).sum(2)


class Net(nn.Module):
    """gpu_screen.Net's signature on the PTv3 backbone, so its loop is reusable as-is."""

    def __init__(self, cin=None, cfg=None):
        super().__init__()
        c = cfg if isinstance(cfg, Cfg) and "widths" in cfg else config(cfg)
        self.c = c
        self.enc = PTv3(3 + 3 * bool(c.use_nrm) + c.n_crv * bool(c.use_crv)
                        if cin is None else cin, c)
        self.heads = nn.ModuleList([Head(c.headc, c) for _ in range(c.npass if c.untied else 1)])
        self.lmfeat = nn.Sequential(nn.Linear(c.headc, 64), nn.ReLU())
        self.contour_nets = nn.ModuleList([
            nn.Sequential(nn.Conv1d(3 + 64 + 32, 96, 5, padding=2), nn.ReLU(),
                          nn.Conv1d(96, 96, 3, padding=1), nn.ReLU(), nn.Conv1d(96, 3, 1))
            for _ in CONTOURS])
        self.embC = nn.Embedding(NL, 32)

    def contour(self, pc, h, q):
        B = pc.shape[0]
        idx = knn(q, pc, 1).squeeze(-1)
        f = self.lmfeat(torch.gather(h, 1, idx[..., None].expand(-1, -1, self.c.headc)))
        e = self.embC(torch.arange(NL, device=pc.device))[None].expand(B, -1, -1)
        inp = torch.cat([q / SCALE, f, e], -1)
        out = torch.zeros(B, NL, 3, device=pc.device, dtype=q.dtype)
        for (lo, hi), net in zip(CONTOURS, self.contour_nets):
            out[:, lo:hi + 1] = net(inp[:, lo:hi + 1].transpose(1, 2)).transpose(1, 2)
        return q + out

    def one(self, pc, q0, ft=None):
        h = self.enc(pc, ft)
        outs, q = [], q0
        for i in range(self.c.npass):
            q1, q2 = self.heads[i if self.c.untied else 0](pc, h, q)
            outs.append((q1, q2)); q = q2
        return outs, self.contour(pc, h, q)

    def forward(self, pcs, q0, fts=None):
        """pcs: list of fresh surface samples. Returns (outs_last, final, per_sample).

        Multiple samples are plain-averaged; the learned fusion of gpu_screen's `fusion2`
        is not reimplemented because that variant screened null.
        """
        finals, outs = [], None
        for s, pc in enumerate(pcs):
            outs, fin = self.one(pc, q0, None if fts is None else fts[s])
            finals.append(fin)
        return outs, torch.stack(finals, 0).mean(0), finals


def _env_flag(name):
    """NEEDS is a class attribute the trainer reads BEFORE any cfg exists, so it has to
    peek at CFG_<NAME> the same way train_family.py would resolve it."""
    v = str(os.environ.get(f"CFG_{name}", os.environ.get(name, "0"))).strip('"')
    return v.lower() not in ("0", "false", "")


def _nrm_wanted():
    return _env_flag("USE_NRM")


def _crv_wanted():
    return _env_flag("USE_CRV")


class PTv3Family(nn.Module):
    """train_family.py adapter, with gpu_screen.py's objective so a run is a BACKBONE swap.

    `aux` carries the (pre-snap, snapped) pair of every pass, exactly as fam_pointnext
    does, and `loss` reproduces the shipped schedule. Leaving it to the harness's
    default_loss would NOT be neutral: default_loss reads the interleaved list as 8
    successive depths and halves the weight every HALF pass, so pass 0's snapped point
    gets 0.0078 instead of 0.0667 (8.5x less) and the decay per pass becomes 4x instead
    of 2x. fam_kpconv and fam_pointnext both carry the same private loss, so this is what
    keeps the family comparable to its siblings rather than what makes it different.
    """

    DEFAULTS = DEFAULTS
    SEARCH_SPACE = dict(patch=[128, 256, 1024], curve=["hilbert", "morton", "both"],
                        width=[32, 48, 64], stages=[2, 3, 4], depth=[2, 3],
                        voxel=[0.7, 0.85, 1.0], voxgrow=[2.0, 2.5, 3.0], npass=[2, 4],
                        dropout=[0.0, 0.1, 0.2], mlp=[2, 4], heads=[4, 8],
                        lr=[3e-4, 1e-3, 1.5e-3])
    # curvature is SCALAR: it belongs in NEEDS but must never appear in ROTATES
    NEEDS = (("nrm",) if _nrm_wanted() else ()) + (("crv",) if _crv_wanted() else ())
    ROTATES = ("nrm",)
    SAMPLES = 1

    def __init__(self, cfg, meta):
        super().__init__()
        c = config(cfg)
        assert meta["nl"] == NL and abs(meta["scale"] - SCALE) < 1e-9, \
            f"this family hardcodes NL={NL}/SCALE={SCALE} with the shipped head"
        assert bool(c.use_nrm) == ("nrm" in self.NEEDS), \
            ("use_nrm and NEEDS disagree: NEEDS is fixed at import from CFG_USE_NRM/"
             "USE_NRM, so set it in the environment, not only in a cfg dict")
        assert bool(c.use_crv) == ("crv" in self.NEEDS), \
            ("use_crv and NEEDS disagree: set USE_CRV/CFG_USE_CRV in the ENVIRONMENT and "
             "point DATA at scratch/screen_data_8192crv.npz (N_CRV = its channel count)")
        # VOXEL/POOLR are a millimetre ladder against a given DENSITY (see the table in
        # the module docstring): at 2048 points VOXEL=0.85/POOLR=4 leaves stage 1 with
        # 1214 occupied voxels for 512 slots, so the uniform merge -- not the grid --
        # would be doing the coarsening, silently. Refuse rather than run that.
        assert int(c.npts) == meta["npts"], (
            f"cfg npts={c.npts} but DATA has {meta['npts']} points per cloud. Set CFG_NPTS "
            f"to the file's N together with the POOLR/VOXEL row documented for it: "
            f"8192->4/0.85, 4096->2/0.9, 2048->2/1.3.")
        self.net = Net(cfg=c)

    def forward(self, batch):
        f = [batch[k] for k in ("nrm", "crv") if batch.get(k) is not None]
        ft = [torch.cat(f, -1)] if f else None
        outs, fin, _ = self.net([batch["pc"]], batch["coarse"], ft)
        return {"pred": fin, "aux": [q for pair in outs for q in pair]}

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


MODEL = PTv3Family


# ------------------------------------------------------------------- smoke test
if __name__ == "__main__":
    import itertools, time

    torch.manual_seed(0); np.random.seed(0)
    # resolve use_nrm the way NEEDS and train_family.py's CFG_* merge do, so that both
    # USE_NRM=1 and CFG_USE_NRM=1 run this file consistently
    C = config({"use_nrm": int(_nrm_wanted()), "use_crv": int(_crv_wanted())})

    # 1. is the Hilbert encoding actually a Hilbert curve?
    for bits in (3, 4):
        S = 1 << bits
        g = torch.tensor(list(itertools.product(range(S), repeat=3)))
        for nm, fn in (("morton", morton_code), ("hilbert", hilbert_code)):
            code = fn(g, bits)
            bij = len(set(code.tolist())) == len(code) and int(code.max()) == S ** 3 - 1
            step = (g[code.argsort()][1:] - g[code.argsort()][:-1]).abs().sum(1)
            print(f"curve check bits={bits} {nm:8s} cells={S**3:5d} bijective={bij} "
                  f"consecutive_L1: max={int(step.max())} all_one={bool((step == 1).all())}")
            assert bij, f"{nm} is not a bijection onto [0,{S**3-1}]"
        assert bool((step == 1).all()), "hilbert_code: consecutive codes are not adjacent"
    print("HILBERT VERIFIED: every consecutive pair of codes is one grid step apart\n")

    # 2. forward + backward on a synthetic B=2 batch of SURFACE points. The phantom is
    # scaled so its VOXEL OCCUPANCY matches the real ears (1974 cells at 2.125mm and 360 at
    # 5.31mm, against 1783/400 on real 4096-point clouds) — an area match is not enough,
    # because an oblique folded sheet crosses ~1.8x the cells a compact surface of the same
    # area does, and occupancy is exactly what the pooling budget below is checked against.
    # At NPTS=8192 the phantom is ~10% BELOW a real 8192-point ear (2194 at 2.125mm), so
    # the printed stage-1 stat understates the real load by about that much.
    B, N = 2, C.npts
    t = torch.rand(B, N); th = torch.rand(B, N) * 6.283
    pc = torch.stack([27 * th.cos() * (1 + 0.1 * (3 * t).sin()),
                      19.5 * th.sin(), 45 * t - 22.5], -1) + torch.randn(B, N, 3) * 0.15
    ftp = ([Fn.normalize(pc, dim=-1)] if C.use_nrm else []) + \
          ([torch.rand(B, N, C.n_crv) * 2 - 1] if C.use_crv else [])
    ft = torch.cat(ftp, -1) if ftp else None
    q0 = pc[:, torch.linspace(0, N - 1, NL).long()] + torch.randn(B, NL, 3) * 0.8
    net = Net(cfg=C)
    npar = sum(p.numel() for p in net.parameters())
    print(f"stage widths {C.widths}  points/stage {counts(N, C)}  patch {C.patch}  "
          f"curve {C.curve}  blocks {(2 * C.stages - 1) * C.depth}")
    print(f"params: {npar:,}")
    half = C.voxel * (1 << (C.bits - 1))
    gc = grid_coord(pc, C.voxel, C.bits)
    sat = float(((gc == 0) | (gc == (1 << C.bits) - 1)).float().mean())
    print(f"grid half-extent {half:.0f}mm (real ears reach 57.6mm on an axis, 65.5mm from "
          f"the origin, 72mm augmented); cloud max |coord| {float(pc.abs().max()):.0f}mm; "
          f"saturated coords {sat:.1%}")

    t0 = time.time()
    outs, fin, per = net([pc], q0, None if ft is None else [ft])
    fwd = time.time() - t0
    loss = ((fin - q0) ** 2).sum(-1).mean() + sum(((a - q0) ** 2).sum(-1).mean()
                                                 for a, b in outs)
    loss.backward()
    gn = sum(float(p.grad.norm()) ** 2 for p in net.parameters() if p.grad is not None) ** 0.5
    nog = [n for n, p in net.named_parameters() if p.grad is None]
    print(f"in {tuple(pc.shape)} -> outs {len(outs)}x{tuple(outs[0][0].shape)} "
          f"final {tuple(fin.shape)}")
    print("grid pooling  " + "  ".join(f"{o:.0f} occupied voxels -> {s} slots"
                                       for o, s in net.enc.pool_stats))
    print(f"grad norm {gn:.3f} | params without grad {len(nog)} | "
          f"fwd {fwd:.1f}s fwd+bwd {time.time()-t0:.1f}s")
    assert fin.shape == (2, NL, 3), fin.shape
    assert torch.isfinite(fin).all() and gn > 0 and not nog

    # 3. pool/unpool index map: a per-voxel-constant signal must survive the round trip
    #    exactly while the coarse grid fits the slot budget (a max of equal values)
    n0, n2 = 512, 128
    vv = torch.ones(1, n0, dtype=torch.bool); ps = torch.randn(1, n0, 3)
    for hi in (100, 400):
        cd = torch.randint(0, hi, (1, n0))
        sig = cd[..., None].float().expand(-1, -1, 4).contiguous()
        px, pp, v2, up, occ = grid_pool(sig, ps, cd, vv, n2)
        err = float((grid_unpool(px, up) - sig).abs().max())
        print(f"pool check  {occ:.0f} occupied -> {n2} slots, {int(v2.sum())} filled, "
              f"round-trip max|err| {err:.1e}" + ("" if occ <= n2 else "  (merged)"))
        if occ <= n2:
            assert err == 0.0 and int(v2.sum()) == round(occ)
        else:
            assert int(v2.sum()) == n2

    # 4. does re-shuffling the serialization actually grow the receptive field? Perturb
    #    ONE input point by 1e-3 mm (far too small to change a voxel) and count how many
    #    of the N per-point features move. patch is the single-block bound.
    net.eval()
    with torch.no_grad():
        pc2 = pc.clone(); pc2[0, 0] += 1e-3
        ones = torch.ones(B, N, dtype=torch.bool)
        o1 = serialize(pc, ones, C.voxel, C.bits, C.curve, PERMS[0])
        o2 = serialize(pc2, ones, C.voxel, C.bits, C.curve, PERMS[0])
        h1, h2 = net.enc.features(pc, ft), net.enc.features(pc2, ft)
    moved = int(((h1 - h2).abs().max(-1).values[0] > 1e-7).sum())
    print(f"receptive field: 1 point perturbed -> {moved}/{N} point features move "
          f"(stage-0 order unchanged: {bool((o1 == o2).all())}, one-block bound {C.patch})")

    # 5. the train_family.py contract, exercised exactly as the harness calls it, at a
    #    point count that is NOT a multiple of patch — the trainer's sub_frac augmentation
    #    produces one, and evaluation then uses the full cloud, so both must run
    meta = dict(nl=NL, contours=CONTOURS, scale=SCALE, npts=N, fold=0, dev="cpu",
                n_train_ears=272, artefacts={})
    fam = MODEL({**MODEL.DEFAULTS, "width": 32, "stages": 2, "use_nrm": C.use_nrm,
                 "use_crv": C.use_crv}, meta)
    import train_family as TF                     # guarded __main__, so importing is safe
    for nsub in (N, int(N * 0.625) - 7):
        batch = {"pc": pc[:, :nsub], "coarse": q0, "ear": torch.arange(B)}
        if "nrm" in MODEL.NEEDS:
            batch["nrm"] = ft[:, :nsub, :3]
        if "crv" in MODEL.NEEDS:
            batch["crv"] = ft[:, :nsub, -C.n_crv:]
        out = fam(batch)
        # the harness's dispatcher must reach THIS family's loss, not default_loss's own
        # geometric weights over the interleaved pairs
        L = TF.default_loss(out, q0, fam, batch)
        assert abs(float(L) - float(fam.loss(out, q0))) < 1e-9, "trainer bypassed loss()"
        L.backward()
        print(f"MODEL contract: N={nsub:5d} (pad {(-nsub) % C.patch:3d}) pred "
              f"{tuple(out['pred'].shape)} aux {len(out['aux'])}x{tuple(out['aux'][0].shape)}"
              f" params {sum(p.numel() for p in fam.parameters()):,}"
              f" NEEDS {MODEL.NEEDS} SAMPLES {MODEL.SAMPLES} loss {float(L):.2f}")
        assert out["pred"].shape == (2, NL, 3), out["pred"].shape
        assert all(a.shape == (2, NL, 3) for a in out["aux"])
    print("OK")
