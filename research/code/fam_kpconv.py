"""
KPCONV BACKBONE for the 85-landmark refinement head — pure torch, genuine
fixed-radius neighbourhoods in MILLIMETRES.

WHY THIS EXISTS.  The shipped 813k-param DGCNN refiner is dominated by ORDERED
CORRESPONDENCE error: along-contour RMSE 1.4456 (77% of the energy), across-contour
0.7233 (20%), surface-normal 0.2822 (2%) — and that decomposition did not move for
any of seven screened variants of the same top-K DGCNN family. One structural
suspect is the backbone's neighbourhood definition: `knn(pos, pos, GK)` is a
top-K graph, so its physical support breathes with the local point density and
with the augmentation's +-10% scale jitter. A landmark descriptor therefore has no
fixed metric meaning, which is exactly the sloppiness that lets phase along a
contour drift. This module makes EVERY neighbourhood a metric ball of a stated mm
radius and reduces resolution by grid subsampling at a stated mm voxel size.
Nothing else changes: the offset/snap head, the per-contour smoothing head, the
landmark embeddings, the feature width handed to the head (FDIM=256) and the deep-
supervision weights are the baseline's, so the ONE thing under test is the
backbone's notion of a neighbourhood.

KPConv, explicitly (rigid, i.e. deformable-free):
    y_i = sum_k W_k^T ( sum_{j in N_i} h(p_j - p_i, x_k) f_j ) / |N_i|
    h(y, x_k) = max(0, 1 - ||y - x_k|| / sigma),      sigma = KP_SIGMA * r
    N_i = { j : ||p_j - p_i|| <= r }                  (a BALL, not a top-K set)
`x_k` are NKP kernel points: one at the origin plus NKP-1 spread over a shell of
radius KP_RHO*r by Fibonacci initialisation followed by tangential repulsion. The
1/|N_i| normalisation makes the response independent of point density; without it
the deeper levels (whose occupancy varies most) would be scaled by their own
sampling.

LAYER SCHEDULE (r_l = R0*2^l, voxel_l = V0*2^l), defaults R0=2.5mm V0=1.0mm STAGES=3:

    level | points come from           | voxel | conv r | strided-in r | ~|N|
    ------+----------------------------+-------+--------+--------------+------
      0   | raw cloud (spacing ~1.0mm) |  --   |  2.5   |      --      |  20
      1   | grid subsample of level 0  |  2.0  |  5.0   |  2.5         |  20
      2   | grid subsample of level 1  |  4.0  | 10.0   |  5.0         |  20
      3   | grid subsample of level 2  |  8.0  | 20.0   | 10.0         |  20

Level 0 is NOT subsampled: the GRID-EQUIVALENT spacing at 2048 points is 0.98mm, which
is already the V0=1.0mm grid, and the snap head must attend to real surface points.
Because r_l/spacing_l is constant, the expected ball occupancy pi*r^2/spacing^2 ~ 20
is the same at every level — that is the point of doubling both together.

V0 IS A PROPERTY OF THE DATA, NOT A FREE KNOB — AND IT IS NOT THE NEAREST-NEIGHBOUR
DISTANCE. V0 is the GRID-equivalent spacing sqrt(surface_area / n_points). For a
random (not gridded) sample of a surface the mean nearest-neighbour distance is only
HALF that — measured on the smoke sheet: 0.505mm mean-NN vs 0.983mm sqrt(A/n) at 2048
points. Only sqrt(A/n) reproduces the occupancy pi*(R0/V0)^2 (19.6 predicted, 20
measured); using the mean-NN value would halve V0, quadruple the auto KMAX to 220 and
quarter the intended stride. So: 2048 pts -> V0=1.0, 4096 pts -> V0=0.7. The smoke test
INVERTS the occupancy to recover the spacing the cloud actually has and asserts it
matches V0, which is the check to trust rather than any measurement of mean-NN.

THE ~20 OCCUPANCY IS THE EVALUATION REGIME. train_family.py trains on sub_frac=0.625
of the cloud with a +-10% scale jitter and evaluates on the full cloud, so at NPTS=2048
level-0 occupancy is 11-16 (median 13) while TRAINING and 20 while EVALUATING, and the
coarse levels shift with it (eval 2048->566->180->64, train 1280->501->174->64). That
gap is inherited from the baseline harness, and a fixed metric radius plus the 1/|N_i|
normalisation is precisely what makes it survivable — a top-K graph would instead have
changed its physical radius by the same 1.26x. But it does mean R0 is effectively
tuned for 13 neighbours, not 20: to hit occupancy 20 during training R0 must be
2.5*sqrt(20/13) = 3.1mm, which is why R0=3.0 is in SEARCH_SPACE.

CHANGING THE POINT COUNT FORCES A CHOICE, and it is the same confound that made the
earlier 8192-point DGCNN test undecidable. Occupancy is pi*(R0/V0)^2, so at 4096 pts
(V0=0.7) keeping R0=2.5mm holds the PHYSICAL radius and doubles the occupancy to a
measured ~40 (the auto cap follows to 113); setting R0=1.75mm instead holds the
occupancy at ~20 and shrinks the physical window. Pick one deliberately and say which.

STAGES IS BOUNDED BY THE ORGAN. At R0=2.5 a fourth stage has a 40mm ball, which
exceeds the pinna's extent: level 4 holds ~30 points and every one of them is a
neighbour of every other, so the layer degenerates into a global pool (visible as
cap == the level's own point count in the report). STAGES=3 is the useful maximum here.

A strided block queries the COARSE level but gathers from the FINE level, so its ball
is the FINE level's radius r_l; the radius doubles AFTER the stride (as in the
reference KPConv). Using r_{l+1} there makes the ball hold ~4x as many fine points as
the cap allows and truncates 82% of the queries at strided_0->1 (measured) — the
neighbour report below is what caught that during development.

NEIGHBOUR TRUNCATION IS REPORTED, NOT HIDDEN.  A ball can hold any number of points,
so the gather is capped at KMAX members (nearest first). The count is taken BEFORE
the cap, and min/median/max/mean plus the fraction of truncated and of empty balls is
reported per layer (`enc.stats`, printed once per run). KMAX defaults to
ceil(2.8 * pi * (R0/V0)^2) — 2.8x the designed occupancy — so it follows R0/V0 and a
radius sweep cannot silently start truncating. If `frac_truncated` is not ~0 the
(R0, V0, KMAX) triple is mis-set and the run is not measuring what it claims to.

    python research/code/fam_kpconv.py                  # CPU smoke test, B=2, <60s
    SMOKE_TRAIN=1 python research/code/fam_kpconv.py     # + train_family.py end-to-end

TWO ENTRY POINTS, one architecture:
  Net(cin=None, cfg=None)            gpu_screen.py idiom, config from the environment
      forward(pc, q0, ft=None) -> ([(q1,q2)] * NPASS, final)
  MODEL = FamKPConv(cfg, meta)       train_family.py FAMILY CONTRACT
      forward(batch) -> {'pred', 'aux', 'pre'};  loss(out, tg);  diag()
The env vars below are the DEFAULTS dict the trainer merges CFG_* over, so the same
knob is reachable as `R0=3.0 python ...` and as `CFG_R0=3.0 ... train_family.py`
(which is how search_driver.py sweeps it).
"""
import os, math
import numpy as np
import torch
import torch.nn as nn

NL, SCALE = 85, 30.0                          # landmarks; mm -> unit, as in gpu_screen.py
CONTOURS = [(0, 24), (25, 54), (55, 74), (75, 84)]


def _mmlist(v):
    """"11,9,7.4" | [11, 9] | "" | None  ->  list[float] | None"""
    if v is None or v == "" or v == []:
        return None
    return [float(x) for x in (v.split(",") if isinstance(v, str) else v)]


ENV_DEFAULTS = dict(
    npts=int(os.environ.get("NPTS", "2048")),        # points the module targets (encoder is
                                                     #   size-agnostic; sets the smoke cloud)
    width=int(os.environ.get("WIDTH", "48")),        # level-0 width; level l is width*(l+1)
    stages=int(os.environ.get("STAGES", "3")),       # strided downsamplings (stages+1 levels)
    r0=float(os.environ.get("R0", "2.5")),           # level-0 conv radius, mm
    v0=float(os.environ.get("V0", "1.0")),           # level-0 point spacing, mm (see docstring)
    npass=int(os.environ.get("NPASS", "4")),         # offset/snap refinement passes
    dropout=float(os.environ.get("DROPOUT", "0.1")),
    use_nrm=int(os.environ.get("USE_NRM", "0")),     # append per-point normals to the input
    fdim=int(os.environ.get("FDIM", "256")),         # width handed to the head (base: 256)
    nkp=int(os.environ.get("NKP", "15")),            # kernel points (1 centre + nkp-1 on a shell)
    kp_rho=float(os.environ.get("KP_RHO", "0.66")),  # shell radius as a fraction of r
    kp_sigma=float(os.environ.get("KP_SIGMA", "0.55")),   # correlation width sigma = kp_sigma*r
    kmax=int(os.environ.get("KMAX", "0")),           # ball gather cap; 0 = auto from r0/v0
    bottle=int(os.environ.get("BOTTLE", "4")),       # residual-block bottleneck ratio
    in_pos=int(os.environ.get("IN_POS", "1")),       # feed pos/SCALE as input channels
    norm=os.environ.get("NORM", "layer"),            # "layer" | "none"
    head_k=int(os.environ.get("HEAD_K", "48")),      # head window size (base: 48 = 7.35mm)
    head_radii=os.environ.get("HEAD_RADII", ""),     # "" = base top-K head; else mm per pass
    head_offs=os.environ.get("HEAD_OFFS", ""),       # "" = unbounded offsets (base); else mm
    untied=int(os.environ.get("UNTIED", "0")),       # untie the per-pass head weights
    nb_chunk=int(os.environ.get("NB_CHUNK", "1024")),     # query chunk for the radius search
    nb_stats=int(os.environ.get("NB_STATS", "2")),   # instrument this many forwards, then stop
                                                     #   (each stat is a device->host sync)
)


def derive(cfg):
    """Normalise a config (env defaults + CFG_* overrides) and add the mm schedule."""
    c = dict(ENV_DEFAULTS, **(cfg or {}))
    for k in ("npts", "width", "stages", "npass", "use_nrm", "fdim", "nkp", "kmax",
              "bottle", "in_pos", "head_k", "untied", "nb_chunk", "nb_stats"):
        c[k] = int(c[k])
    for k in ("r0", "v0", "dropout", "kp_rho", "kp_sigma"):
        c[k] = float(c[k])
    c["head_radii"] = _mmlist(c["head_radii"])
    c["head_offs"] = _mmlist(c["head_offs"])
    c["rad"] = [c["r0"] * 2 ** l for l in range(c["stages"] + 1)]
    c["vox"] = [c["v0"] * 2 ** l for l in range(c["stages"] + 1)]
    c["wid"] = [c["width"] * (l + 1) for l in range(c["stages"] + 1)]
    if c["kmax"] <= 0:                       # 2.8x the designed occupancy pi*(r/spacing)^2,
        c["kmax"] = max(16, int(math.ceil(2.8 * math.pi * (c["r0"] / c["v0"]) ** 2)))
    return c                                 # so a radius sweep cannot silently truncate


# ------------------------------------------------------------------ geometry ops
_KPCACHE = {}


def kernel_points(n, rho, iters=400):
    """1 centre + (n-1) points on a sphere of radius `rho`, Fibonacci init then
    tangential repulsion (a deterministic stand-in for KPConv's optimised sets)."""
    if (n, rho) in _KPCACHE:
        return _KPCACHE[(n, rho)].clone()
    m = n - 1
    i = torch.arange(m, dtype=torch.float64)
    z = 1.0 - 2.0 * (i + 0.5) / m
    r = (1.0 - z * z).clamp(min=0).sqrt()
    phi = i * math.pi * (3.0 - math.sqrt(5.0))
    x = torch.stack([r * phi.cos(), r * phi.sin(), z], -1)
    for _ in range(iters):
        d = x[:, None, :] - x[None, :, :]
        n2 = (d * d).sum(-1).clamp(min=1e-12)
        f = (d / n2[..., None] ** 1.5).sum(1)
        f = f - (x * f).sum(-1, keepdim=True) * x                  # keep the tangential part
        x = x + 0.05 * f / f.norm(dim=-1).amax().clamp(min=1e-12)
        x = x / x.norm(dim=-1, keepdim=True)
    out = torch.cat([torch.zeros(1, 3, dtype=torch.float64), rho * x], 0).float()
    _KPCACHE[(n, rho)] = out
    return out.clone()


def grid_subsample(pts, mask, voxel):
    """Voxel-barycentre subsampling of a PADDED batch.

    pts (B,N,3) mm, mask (B,N) bool. Voxel keys are packed EXACTLY into int64
    (batch, ix, iy, iz -> one integer over the batch's own bounding box), so there
    are no hash collisions and clusters come out grouped by batch element. Returns
    coarse (B,Nmax,3), cmask (B,Nmax), fine->coarse index (B,N), coarse count (B,).
    """
    B, N, _ = pts.shape
    b = torch.arange(B, device=pts.device)[:, None].expand(B, N)[mask]
    p = pts[mask]
    iv = torch.floor(p / voxel).long()
    iv = iv - iv.amin(0, keepdim=True)
    D = iv.amax(0) + 1
    span = int(D[0]) * int(D[1]) * int(D[2])
    assert span * B < 2 ** 62, "voxel grid too fine for exact int64 packing"
    key = b * span + (iv[:, 0] * D[1] + iv[:, 1]) * D[2] + iv[:, 2]
    uk, inv = torch.unique(key, return_inverse=True)               # sorted -> batch-grouped
    C = uk.numel()
    cnt = torch.zeros(C, device=pts.device, dtype=pts.dtype).index_add_(
        0, inv, torch.ones_like(p[:, 0]))
    ctr = torch.zeros(C, 3, device=pts.device, dtype=pts.dtype).index_add_(0, inv, p) / cnt[:, None]
    cb = torch.zeros(C, dtype=torch.long, device=pts.device).scatter_(0, inv, b)
    per = torch.bincount(cb, minlength=B)
    Nmax = int(per.amax())
    pos = torch.arange(C, device=pts.device) - (torch.cumsum(per, 0) - per)[cb]
    coarse = torch.zeros(B, Nmax, 3, device=pts.device, dtype=pts.dtype)
    cmask = torch.zeros(B, Nmax, dtype=torch.bool, device=pts.device)
    coarse[cb, pos] = ctr
    cmask[cb, pos] = True
    fi = torch.zeros(B, N, dtype=torch.long, device=pts.device)
    fi[mask] = pos[inv]
    return coarse, cmask, fi, per


def radius_neighbors(q, qmask, s, smask, radius, kmax, chunk=1024):
    """TRUE metric ball: every support point with ||p_j - q_i|| <= radius, gathered up
    to `kmax` members, nearest first. `ntrue` is the UNCAPPED ball size, so the caller
    can prove the cap is not truncating. Padded queries get an all-False mask and
    padded support points are pushed to infinite distance, so they never contribute."""
    B, Q, _ = q.shape
    K = min(kmax, s.shape[1])
    idx = torch.zeros(B, Q, K, dtype=torch.long, device=q.device)
    nm = torch.zeros(B, Q, K, dtype=torch.bool, device=q.device)
    nt = torch.zeros(B, Q, dtype=torch.long, device=q.device)
    for a in range(0, Q, chunk):
        z = min(a + chunk, Q)
        d = torch.cdist(q[:, a:z], s).masked_fill(~smask[:, None, :], float("inf"))
        nt[:, a:z] = (d <= radius).sum(-1)
        dv, ii = d.topk(K, largest=False, dim=-1)
        idx[:, a:z] = ii
        nm[:, a:z] = (dv <= radius) & qmask[:, a:z, None]
    return idx, nm, nt * qmask.long()


def gather_nb(x, idx):
    B, Q, K = idx.shape
    C = x.shape[-1]
    return torch.gather(x, 1, idx.reshape(B, Q * K, 1).expand(-1, -1, C)).view(B, Q, K, C)


def nb_stats(nt, qmask, K):
    v = nt[qmask].float()
    return dict(min=int(v.amin()), median=int(v.median()), max=int(v.amax()),
                mean=round(float(v.mean()), 2), cap=int(K),
                frac_truncated=round(float((v > K).float().mean()), 4),
                frac_empty=round(float((v == 0).float().mean()), 4))


def stats_table(st):
    """The per-layer point/neighbour report — printed once per run, and by the smoke test."""
    L = ["level  voxel_mm  conv_r_mm  width   points per batch elem   padded"]
    for t, n, pad in zip(st["table"], st["points_per_level"], st["padded_per_level"]):
        vx = "raw" if t["voxel_mm"] is None else f"{t['voxel_mm']:.2f}"
        L.append(f"  {t['level']}    {vx:>6s}     {t['conv_radius_mm']:6.2f}  {t['width']:5d}   "
                 f"{str(n):>21s}  {pad:6d}")
    L.append("")
    L.append(f"{'layer':16s} {'r_mm':>6s} {'min':>4s} {'med':>4s} {'max':>4s} {'mean':>6s} "
             f"{'cap':>4s} {'trunc':>6s} {'empty':>6s}   (ball size BEFORE the cap)")
    for s in st["layers"]:
        L.append(f"{s['kind']:16s} {s['radius_mm']:6.2f} {s['min']:4d} {s['median']:4d} "
                 f"{s['max']:4d} {s['mean']:6.2f} {s['cap']:4d} {s['frac_truncated']:6.3f} "
                 f"{s['frac_empty']:6.3f}")
    return "\n".join(L)


# ------------------------------------------------------------------ KPConv
class KPConv(nn.Module):
    """Rigid KPConv: sum over kernel points of W_k applied to the h-weighted neighbour
    features, normalised by the ball size."""

    def __init__(self, c, cin, cout, radius):
        super().__init__()
        kp = kernel_points(c["nkp"], c["kp_rho"] * radius)
        self.register_buffer("kp", kp)
        self.register_buffer("kp2", (kp * kp).sum(-1))
        self.W = nn.Parameter(torch.randn(c["nkp"], cin, cout) / math.sqrt(cin * c["nkp"]))
        self.sigma = c["kp_sigma"] * radius
        self.nkp, self.cin, self.cout = c["nkp"], cin, cout

    def forward(self, q, sp, sf, idx, nm):
        B, Q, K = idx.shape
        fj = gather_nb(sf, idx)
        # The correlation depends on DATA only (positions are never differentiated in this
        # pipeline), so it is built under no_grad: that keeps the (B,Q,K,NKP) tensor off the
        # autograd tape as a differentiable node. A deformable variant would have to drop this.
        with torch.no_grad():
            rel = gather_nb(sp, idx) - q[:, :, None, :]
            d2 = (rel * rel).sum(-1)[..., None] - 2.0 * (rel @ self.kp.T) + self.kp2
            h = (1.0 - d2.clamp(min=0).sqrt() / self.sigma).clamp(min=0) * nm[..., None]
        s = torch.einsum("bqkn,bqkc->bqnc", h, fj) / nm.sum(-1).clamp(min=1)[:, :, None, None]
        return s.reshape(B, Q, self.nkp * self.cin) @ self.W.reshape(self.nkp * self.cin, self.cout)


class KPBlock(nn.Module):
    """Bottleneck residual block: unary -> KPConv -> unary, with a shortcut that max-pools
    over the ball when the block is strided.

    LayerNorm (per point, so padding-safe) is the one deviation from gpu_screen's norm-free
    MLPs: with 2+2*STAGES stacked convs whose input scale depends on local ball occupancy the
    unnormalised stack does not train. NORM=none turns it off for an ablation, but measure
    before spending a run on it: at init the encoder output std is 2.8e-1 with LayerNorm and
    3.2e-2 without (~9x collapse over 8 blocks), because each KPConv splits its ball across
    NKP kernel points and so shrinks the signal by design.
    """

    def __init__(self, c, cin, cout, radius, strided=False):
        super().__init__()
        nrm = (lambda w: nn.LayerNorm(w)) if c["norm"] == "layer" else (lambda w: nn.Identity())
        cm = max(cout // c["bottle"], 8)
        self.u1 = nn.Sequential(nn.Linear(cin, cm), nrm(cm), nn.ReLU())
        self.kp = KPConv(c, cm, cm, radius)
        self.n2 = nn.Sequential(nrm(cm), nn.ReLU())
        self.u2 = nn.Sequential(nn.Linear(cm, cout), nrm(cout))
        self.sc = nn.Linear(cin, cout) if cin != cout else nn.Identity()
        self.strided = strided
        self.act = nn.ReLU()

    def forward(self, q, sp, sf, idx, nm):
        x = self.u2(self.n2(self.kp(q, sp, self.u1(sf), idx, nm)))
        sh = sf
        if self.strided:                     # max-pool over the ball THEN the unary, as in the
            sh = (gather_nb(sh, idx)         # reference block (also the cheaper order); rows
                  .masked_fill(~nm[..., None], -1e4)     # whose ball is empty (padding) -> 0
                  .amax(2) * nm.any(-1, keepdim=True))
        return self.act(x + self.sc(sh))


class Encoder(nn.Module):
    """stages+1 levels of fixed-radius KPConv, nearest-ancestor upsampled back to the input
    points through the exact pooling tree and fused to fdim (the width the head expects)."""

    def __init__(self, c, cin):
        super().__init__()
        self.c = c
        S, W, R = c["stages"], c["wid"], c["rad"]
        self.stem = KPBlock(c, cin + 1, W[0], R[0])          # +1 = KPConv's constant channel
        self.same = nn.ModuleList([KPBlock(c, W[l], W[l], R[l]) for l in range(S + 1)])
        self.down = nn.ModuleList([KPBlock(c, W[l], W[l + 1], R[l], strided=True)
                                   for l in range(S)])
        self.fuse = nn.Sequential(nn.Linear(sum(W), c["fdim"]), nn.ReLU())
        self.mix = nn.Sequential(nn.Linear(2 * c["fdim"], c["fdim"]), nn.ReLU(),
                                 nn.Dropout(c["dropout"]))
        self.left = c["nb_stats"]            # forwards still to instrument (each stat syncs)
        self.stats = {}

    def table(self):
        c = self.c
        return [dict(level=l, voxel_mm=(None if l == 0 else c["vox"][l]),
                     conv_radius_mm=c["rad"][l], width=c["wid"][l])
                for l in range(c["stages"] + 1)]

    def forward(self, pc, ft=None):
        c = self.c
        S, W, R, KM, CH = c["stages"], c["wid"], c["rad"], c["kmax"], c["nb_chunk"]
        B, N, _ = pc.shape
        pts = [pc]
        msk = [torch.ones(B, N, dtype=torch.bool, device=pc.device)]
        invs = []
        for l in range(1, S + 1):
            cp, cm, iv, _ = grid_subsample(pts[l - 1], msk[l - 1], c["vox"][l])
            pts.append(cp); msk.append(cm); invs.append(iv)

        x = torch.ones(B, N, 1, device=pc.device, dtype=pc.dtype)
        x = torch.cat([x, pc / SCALE if c["in_pos"] else torch.zeros_like(pc)], -1)
        if ft is not None:
            x = torch.cat([x, ft], -1)

        rec = self.left > 0
        stats, feats = [], []
        for l in range(S + 1):
            if l > 0:                        # strided conv onto the coarser level, FINE radius
                idx, nm, nt = radius_neighbors(pts[l], msk[l], pts[l - 1], msk[l - 1],
                                               R[l - 1], KM, CH)
                if rec:
                    stats.append(dict(kind=f"strided_{l-1}->{l}", radius_mm=R[l - 1],
                                      **nb_stats(nt, msk[l], idx.shape[-1])))
                x = self.down[l - 1](pts[l], pts[l - 1], x, idx, nm)
            idx, nm, nt = radius_neighbors(pts[l], msk[l], pts[l], msk[l], R[l], KM, CH)
            if rec:
                stats.append(dict(kind=f"same_{l}", radius_mm=R[l],
                                  **nb_stats(nt, msk[l], idx.shape[-1])))
            if l == 0:
                x = self.stem(pts[0], pts[0], x, idx, nm)
            x = self.same[l](pts[l], pts[l], x, idx, nm)
            feats.append(x)

        chain = torch.arange(N, device=pc.device)[None].expand(B, N)
        up = [feats[0]]
        for l in range(1, S + 1):            # exact pooling tree, so no extra search is needed
            chain = torch.gather(invs[l - 1], 1, chain)
            up.append(torch.gather(feats[l], 1, chain[..., None].expand(-1, -1, W[l])))
        h = self.fuse(torch.cat(up, -1))
        g = h.amax(1, keepdim=True).expand(-1, N, -1)
        if rec:
            first = not self.stats
            self.stats = dict(points_per_level=[[int(v) for v in m.sum(1)] for m in msk],
                              padded_per_level=[int(m.shape[1]) for m in msk],
                              layers=stats, table=self.table(), kmax=KM)
            self.left -= 1
            if first:                        # so an unmodified driver still logs the audit
                print(stats_table(self.stats), flush=True)
        return self.mix(torch.cat([h, g], -1))


# ------------------------------------------------------------------ head (baseline's)
class Head(nn.Module):
    """One refinement pass: bounded-or-free offset, then a surface snap. Structurally
    identical to gpu_screen.py's Head; `radius` is None by default so the one change under
    test stays the backbone."""

    def __init__(self, c, C, max_off=None):
        super().__init__()
        self.emb = nn.Embedding(NL, 32); self.embO = nn.Embedding(NL, 32)
        self.offset = nn.Sequential(nn.Linear(2 * C + 32, 256), nn.ReLU(), nn.Dropout(c["dropout"]),
                                    nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 3))
        self.attn = nn.Sequential(nn.Linear(C + 3 + 32, 128), nn.ReLU(),
                                  nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.C, self.K, self.max_off = C, c["head_k"], max_off

    def gather(self, pc, h, q, radius):
        idx = torch.cdist(q, pc).topk(self.K, largest=False, dim=-1).indices
        fK, pK = gather_nb(h, idx), gather_nb(pc, idx)
        dist = (pK - q[:, :, None, :]).norm(dim=-1)
        mask = None
        if radius is not None:
            keep8 = dist <= dist.topk(8, largest=False, dim=-1).values[..., -1:]
            mask = (dist <= radius) | keep8              # never leave a landmark blind
        return fK, pK, mask, dist

    def forward(self, pc, h, q, radius=None):
        fK, pK, mask, _ = self.gather(pc, h, q, radius)
        if mask is None:
            ctx = torch.cat([fK.mean(2), fK.amax(2)], -1)
        else:
            w = mask.float()[..., None]
            ctx = torch.cat([(fK * w).sum(2) / w.sum(2).clamp(min=1),
                             fK.masked_fill(~mask[..., None], -1e4).amax(2)], -1)
        eo = self.embO(torch.arange(NL, device=pc.device))[None].expand(pc.shape[0], -1, -1)
        off = self.offset(torch.cat([ctx, eo], -1))
        if self.max_off is not None:
            off = self.max_off * torch.tanh(off / max(self.max_off, 1e-6))
        q1 = q + off
        fK2, pK2, mask2, d1 = self.gather(pc, h, q1, radius)
        rel = (pK2 - q1[:, :, None, :]) / SCALE
        e = self.emb(torch.arange(NL, device=pc.device))[None, :, None, :].expand(
            pc.shape[0], NL, self.K, 32)
        logit = self.attn(torch.cat([fK2, rel, e], -1)).squeeze(-1)
        if mask2 is not None:
            logit = logit.masked_fill(~mask2, -1e4)
        w = torch.softmax(logit, -1)
        self.win = d1.amax(-1).mean().detach()           # physical size of the head window, mm
        return q1, (w[..., None] * pK2).sum(2)


class Net(nn.Module):
    """gpu_screen.py-shaped model: KPConv encoder + NPASS offset/snap passes + per-contour
    smoothing. `cfg` overrides the environment defaults entry by entry."""

    def __init__(self, cin=None, cfg=None):
        super().__init__()
        c = self.c = derive(cfg)
        cin = 3 + 3 * c["use_nrm"] if cin is None else cin
        self.enc = Encoder(c, cin)
        F, HO = c["fdim"], c["head_offs"]
        self.heads = nn.ModuleList([Head(c, F, None if HO is None else HO[min(i, len(HO) - 1)])
                                    for i in range(c["npass"] if c["untied"] else 1)])
        self.lmfeat = nn.Sequential(nn.Linear(F, 64), nn.ReLU())
        self.contour_nets = nn.ModuleList([
            nn.Sequential(nn.Conv1d(3 + 64 + 32, 96, 5, padding=2), nn.ReLU(),
                          nn.Conv1d(96, 96, 3, padding=1), nn.ReLU(), nn.Conv1d(96, 3, 1))
            for _ in CONTOURS])
        self.embC = nn.Embedding(NL, 32)

    def contour(self, pc, h, q):
        B = pc.shape[0]
        idx = torch.cdist(q, pc).topk(1, largest=False, dim=-1).indices.squeeze(-1)
        f = self.lmfeat(torch.gather(h, 1, idx[..., None].expand(-1, -1, self.c["fdim"])))
        e = self.embC(torch.arange(NL, device=pc.device))[None].expand(B, -1, -1)
        inp = torch.cat([q / SCALE, f, e], -1)
        out = torch.zeros(B, NL, 3, device=pc.device, dtype=q.dtype)
        for (lo, hi), net in zip(CONTOURS, self.contour_nets):
            out[:, lo:hi + 1] = net(inp[:, lo:hi + 1].transpose(1, 2)).transpose(1, 2)
        return q + out

    def forward(self, pc, q0, ft=None):
        c = self.c
        h = self.enc(pc, ft)
        outs, q = [], q0
        for i in range(c["npass"]):
            hd = self.heads[i] if c["untied"] else self.heads[0]
            r = None if c["head_radii"] is None else c["head_radii"][min(i, len(c["head_radii"]) - 1)]
            q1, q2 = hd(pc, h, q, r)
            outs.append((q1, q2)); q = q2
        return outs, self.contour(pc, h, q)


# ------------------------------------------------------------------ train_family contract
class FamKPConv(nn.Module):
    """train_family.py adapter. `NEEDS` is a CLASS attribute the trainer reads before any
    instance exists, so normals must be requested through the ENVIRONMENT (USE_NRM=1);
    CFG_USE_NRM alone cannot make the trainer load 'nrm' and is rejected in __init__."""

    DEFAULTS = ENV_DEFAULTS
    SEARCH_SPACE = dict(r0=[2.0, 2.5, 3.0, 3.5], stages=[2, 3, 4], width=[32, 48, 64],
                        nkp=[9, 15, 21], kp_rho=[0.55, 0.66, 0.80],
                        kp_sigma=[0.40, 0.55, 0.70], bottle=[2, 4], npass=[3, 4, 5],
                        dropout=[0.0, 0.1, 0.2], untied=[0, 1], head_k=[32, 48, 64],
                        norm=["layer", "none"], lr=[7e-4, 1.5e-3, 3e-3])
    NEEDS = ("nrm",) if ENV_DEFAULTS["use_nrm"] else ()
    ROTATES = ("nrm",)
    SAMPLES = 1

    def __init__(self, cfg, meta):
        super().__init__()
        self.net = Net(cfg=cfg)
        c = self.net.c
        assert not (c["use_nrm"] and "nrm" not in self.NEEDS), \
            "use_nrm is on but NEEDS is empty: set USE_NRM=1 in the ENVIRONMENT (not only " \
            "CFG_USE_NRM) so the trainer loads 'nrm', and point DATA at screen_data_2048nrm.npz"
        assert meta["nl"] == NL and meta["contours"] == CONTOURS, "head geometry contract broken"
        self.sup = np.array([0.5 ** (c["npass"] - 1 - t) for t in range(c["npass"])])
        self.sup /= self.sup.sum()

    def forward(self, b):
        outs, fin = self.net(b["pc"], b["coarse"], b.get("nrm") if self.net.c["use_nrm"] else None)
        return {"pred": fin, "aux": [q2 for _, q2 in outs], "pre": [q1 for q1, _ in outs]}

    def loss(self, out, tg):
        """gpu_screen.py's objective verbatim: geometric deep supervision over the passes,
        the pre-snap position weighted 0.4 against the post-snap one, plus the final."""
        L = 0.0
        for t, (q1, q2) in enumerate(zip(out["pre"], out["aux"])):
            L = L + float(self.sup[t]) * (0.4 * ((q1 - tg) ** 2).sum(-1).mean()
                                          + ((q2 - tg) ** 2).sum(-1).mean())
        return L + ((out["pred"] - tg) ** 2).sum(-1).mean()

    def diag(self):
        return self.net.enc.stats


MODEL = FamKPConv


# ------------------------------------------------------------------ smoke test
def synth(B, npts, seed=0):
    """A pinna-scale curved sheet ~50 x 36 mm, so the mm radii and voxels mean on this
    cloud what they will mean on real ears (measured spacing at 2048 pts: 0.995mm)."""
    g = torch.Generator().manual_seed(seed)
    u = (torch.rand(B, npts, generator=g) - 0.5) * 50.0
    v = (torch.rand(B, npts, generator=g) - 0.5) * 36.0
    z = 0.006 * u ** 2 - 0.010 * v ** 2 + 3.0 * torch.sin(u / 6.0)
    pc = torch.stack([u, v, z], -1) + torch.randn(B, npts, 3, generator=g) * 0.05
    nrm = torch.stack([-(0.012 * u + 0.5 * torch.cos(u / 6.0)), 0.020 * v, torch.ones_like(u)], -1)
    nrm = nrm / nrm.norm(dim=-1, keepdim=True)
    pick = torch.stack([torch.randperm(npts, generator=g)[:NL] for _ in range(B)])
    tgt = torch.gather(pc, 1, pick[..., None].expand(-1, -1, 3))
    return pc, nrm, tgt + torch.randn(B, NL, 3, generator=g) * 0.9, tgt


def trainer_smoke():
    """End-to-end through train_family.py on a synthetic bundle: proves the FAMILY CONTRACT
    (cls(cfg, meta) / forward(batch) / loss / report schema / full inference pipeline)."""
    import tempfile, train_family as TF
    tmp = os.path.join(tempfile.gettempdir(), "fam_kpconv_smoke")
    dp, tp, sp = TF.fake_bundle(tmp, npts=384)[:3]     # [3] is the layout-A mesh, unused here
    env = dict(FAMILY="kpconv", FAMILY_MODULE="fam_kpconv", FAMILY_CLASS="FamKPConv",
               FOLD="0", SEED="0", EPOCHS="2", WORK=tmp, DATA=dp, TRIS=tp, SSM=sp,
               TTA="1", EVAL_EVERY="2", ALIAS="0", TAG="fam_kpconv_smoke",
               CFG_BS="8", CFG_WIDTH="16", CFG_FDIM="64", CFG_STAGES="2",
               CFG_R0="3.0", CFG_V0="1.3", CFG_NPASS="2")
    keep = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    res = TF.main()
    for k, v in keep.items():
        os.environ.pop(k) if v is None else os.environ.__setitem__(k, v)
    assert res["ordered_MLE_full_mm"] is not None, "full pipeline did not run"
    assert len(res["train_val_curve"]) >= 1 and res["params"] > 0
    print(f"  trainer report OK | raw {res['ordered_MLE_mm']:.4f} -> full "
          f"{res['ordered_MLE_full_mm']:.4f} mm | {res['params']:,} params")


if __name__ == "__main__":
    import time
    t0 = time.time()
    torch.manual_seed(0)
    B, NPTS = 2, ENV_DEFAULTS["npts"]
    pc, nrm, q0, tgt = synth(B, NPTS)

    # nb_stats>=1 regardless of NB_STATS: the geometry checks below read the audit, so a run
    # with the syncs switched off must still produce one forward's worth of it.
    fam = FamKPConv(dict(nb_stats=max(1, ENV_DEFAULTS["nb_stats"])),
                    dict(nl=NL, contours=CONTOURS, scale=SCALE, npts=NPTS, fold=0,
                         dev="cpu", n_train_ears=272, artefacts={}))
    c = fam.net.c
    npar = sum(p.numel() for p in fam.parameters())
    print(f"fam_kpconv  NPTS={c['npts']} WIDTH={c['width']} STAGES={c['stages']} R0={c['r0']} "
          f"V0={c['v0']} NPASS={c['npass']} DROPOUT={c['dropout']} USE_NRM={c['use_nrm']} "
          f"NKP={c['nkp']} KMAX={c['kmax']}(auto={ENV_DEFAULTS['kmax'] <= 0})")
    print(f"params: {npar:,}   radii {c['rad']} mm   voxels {[None] + c['vox'][1:]} mm")
    kp = fam.net.enc.stem.kp.kp
    print(f"kernel points {tuple(kp.shape)}: |x_k| = "
          + "/".join(f"{r:.3f}" for r in kp.norm(dim=-1).round(decimals=4).unique().tolist())
          + f" mm, min pairwise on the shell {torch.cdist(kp[1:], kp[1:]).fill_diagonal_(9e9).amin():.3f}"
          + f" mm, sigma(level 0) {fam.net.enc.stem.kp.sigma:.2f} mm\n")

    out = fam({"pc": pc, "coarse": q0, "ear": torch.arange(B),
               "nrm": nrm if c["use_nrm"] else None})
    loss = fam.loss(out, tgt)
    loss.backward()
    gnorm = sum(float(p.grad.norm()) ** 2 for p in fam.parameters() if p.grad is not None) ** 0.5

    st = fam.diag()
    # INDEPENDENT check of the two claims the module makes about its neighbourhoods:
    # (1) `same_0` really is the metric ball, verified against a brute-force numpy count;
    # (2) grid subsampling really is a voxel partition — a point and its cluster barycentre
    #     lie in one cell of side `voxel`, so their Chebyshev distance is < voxel.
    D = np.linalg.norm(pc[0].numpy()[:, None] - pc[0].numpy()[None], axis=-1)
    ref = (D <= c["rad"][0]).sum(1)
    got = [s for s in st["layers"] if s["kind"] == "same_0"][0]
    print(f"brute-force check of same_0 on batch elem 0: min/med/max {ref.min()}/"
          f"{int(np.median(ref))}/{ref.max()} vs reported {got['min']}/{got['median']}/{got['max']}"
          f"  (reported pools both elems, so min<=, max>=)")
    assert got["min"] <= ref.min() and got["max"] >= ref.max() \
        and abs(got["median"] - np.median(ref)) <= 1
    cp, _, fi, per = grid_subsample(pc, torch.ones(B, NPTS, dtype=torch.bool), c["vox"][1])
    cheb = (torch.gather(cp, 1, fi[..., None].expand(-1, -1, 3)) - pc).abs().amax()
    print(f"grid_subsample({c['vox'][1]}mm): {NPTS} -> {per.tolist()} pts, "
          f"max |p - barycentre|_inf = {float(cheb):.3f} mm < voxel {c['vox'][1]}")
    assert float(cheb) < c["vox"][1] and per.tolist() == st["points_per_level"][1]

    # (3) V0 is the GRID-equivalent spacing, so INVERT the measured occupancy to recover the
    # spacing this cloud actually has: |N| = pi*(r/s)^2  =>  s = r / sqrt(|N|/pi). Comparing
    # that with V0 is the executable form of "V0 is a property of the data" — a V0 set from a
    # mean-nearest-neighbour measurement instead is 2x too small and fails here.
    nn = torch.cdist(pc[0], pc[0]).fill_diagonal_(9e9).amin(-1).mean()
    s_occ = c["rad"][0] / math.sqrt(got["median"] / math.pi)
    print(f"V0 check: mean-NN {float(nn):.3f} mm (NOT V0) vs occupancy-implied spacing "
          f"{s_occ:.3f} mm  ->  V0={c['v0']}  ratio {s_occ / c['v0']:.2f}\n")
    assert 0.75 < s_occ / c["v0"] < 1.35, f"V0={c['v0']} disagrees with the measured occupancy"

    print(f"head window (mean max-dist over {c['head_k']} gathered pts): "
          f"{float(fam.net.heads[0].win):.2f} mm   head_radii={c['head_radii']} "
          f"head_offs={c['head_offs']}")
    print(f"passes {len(out['aux'])}  pred {tuple(out['pred'].shape)}  loss {float(loss):.4f}  "
          f"grad-norm {gnorm:.3e}  ({time.time()-t0:.1f}s)")
    assert out["pred"].shape == (2, NL, 3), out["pred"].shape
    assert torch.isfinite(out["pred"]).all()
    nog = [n for n, p in fam.named_parameters() if p.grad is None]
    assert not nog, nog
    print("OK")

    if os.environ.get("SMOKE_TRAIN"):
        print("\n--- train_family.py end-to-end (SMOKE_TRAIN=1) ---")
        trainer_smoke()
        print(f"OK  ({time.time()-t0:.1f}s total)")
