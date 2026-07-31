"""
GPU training — ITERATIVE local soft-argmax landmark net (RTX A6000).

Improvements over the Colab baseline:
  * Iterative refinement (T passes): the local attention window for each of the
    85 landmarks is re-centered on the CURRENT estimate every pass, so a weak
    coarse init (v1 baseline, ~3.7mm) is progressively corrected — coarse-to-fine
    INSIDE the net.  q0 = coarse ; q_{t+1} = softargmax_t.
  * Deep supervision: loss summed over all passes (later passes weighted more).
  * Richer backbone: two rounds of local+global (PointNet-with-context) feature
    mixing, giving each point global shape awareness.
  * Stronger augmentation: 3D rotation (random axis, moderate angle), anisotropic
    scale, jitter, point resampling — critical with only ~280 train ears.
  * Saves best-by-(SSM-projected) weights continuously to gpu_weights.npz.

Surface-constrained (predictions are convex combos of real surface points), so it
ports to a dependency-free numpy forward for the submission.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, time, sys, os

DATA = os.environ.get("DATA", "/home/ubuntu/ear/deep_dataset.npz")
SEED = int(os.environ.get("SEED", "0"))
OUT = os.environ.get("OUT", f"/home/ubuntu/ear/gpu_weights_s{SEED}.npz")
K = 48                 # neighbours per landmark window
T = 4                  # refinement passes
CONTOUR_ITERS = 2      # contour-refinement passes
EPOCHS = 1200
BS = 16
LR = 1.5e-3
SCALE = 30.0
HUBER = float(os.environ.get("HUBER", "0"))   # >0 = Huber delta (mm) on final output; 0 = MSE
METRIC_LOSS = os.environ.get("METRIC_LOSS", "0") == "1"   # loss == official metric (mean Euclid dist)
EQUI_ON = os.environ.get("EQUI", "1") == "1"  # hard equal-arc-length resample layer
SPACING_W = float(os.environ.get("SPACING_W", "0"))  # soft equidistance penalty (alternative)
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED); np.random.seed(SEED)
print(f"device: {dev} | seed {SEED} | K {K} T {T} epochs {EPOCHS} -> {OUT}")

d = np.load(DATA, allow_pickle=True)
clouds = torch.tensor(d["clouds"]).float()      # (N,P,3) canonical mm
coarse = torch.tensor(d["coarse"]).float()      # (N,85,3)
true = torch.tensor(d["true"]).float()          # (N,85,3)
Rmat = d["R"]; c0 = d["c0"]; split = d["split"]
ssm_mean = d["ssm_mean"].astype(np.float64)     # (255,)
ssm_comp = d["ssm_comp"].astype(np.float64)     # (30,255)
feats = torch.tensor(d["feats"]).float() if "feats" in d.files else None   # (N,P,Fdim) curvature
FDIM = feats.shape[-1] if feats is not None else 0
NL = 85; P = clouds.shape[1]
CONTOURS = [tuple(c) for c in d["contours"]]    # [(0,24),(25,54),(55,74),(75,84)] inclusive
tr_idx = np.where(split == "train")[0]; va_idx = np.where(split == "val")[0]

# ---- 5-fold GroupKFold by SUBJECT (both ears of a subject stay together) ----
# Ears are stored as consecutive (left,right) pairs per subject, so subject = i//2.
# FOLD=-1 keeps the original fixed split; FOLD=0..4 cross-validates over ALL ears in
# this file (held-out test subjects are not present in the file at all, so they stay
# untouched). This gives out-of-fold predictions for every ear = a far more reliable
# estimate than a single 60-ear split.
FOLD = int(os.environ.get("FOLD", "-1"))
NFOLD = int(os.environ.get("NFOLD", "5"))
if FOLD >= 0:
    subj = np.arange(len(split)) // 2
    rs = np.random.RandomState(12345)
    order_s = rs.permutation(np.unique(subj))
    val_s = set(np.array_split(order_s, NFOLD)[FOLD].tolist())
    va_idx = np.array([i for i in range(len(split)) if subj[i] in val_s])
    tr_idx = np.array([i for i in range(len(split)) if subj[i] not in val_s])
    assert not (set(subj[tr_idx]) & set(subj[va_idx])), "subject leak across folds"
    print(f"FOLD {FOLD}/{NFOLD}: {len(set(subj[tr_idx]))} train / {len(val_s)} val "
          f"SUBJECTS (disjoint, both ears kept together)")

print(f"input feats: {FDIM} extra per-point channels")
print(f"{len(tr_idx)} train / {len(va_idx)} val ears, {P} pts/ear")


# Contours whose GT landmarks are EXACT equal-arc-length samples (measured:
# poly-resample residual 0.083mm / 0.030mm, gap CV 0.018 / 0.011). Enforcing
# equidistance by construction removes the tangential (sliding) degree of freedom.
# Helix/antihelix are NOT equidistant (residual 2.7/2.1mm) -> left untouched.
EQUI = [i for i, (lo, hi) in enumerate(CONTOURS) if (lo, hi) in ((55, 74), (75, 84))]


def equidistant_resample(P):
    """(B,n,3) -> resampled at equal arc length along the polyline. Differentiable."""
    B, n, _ = P.shape
    seg = torch.norm(P[:, 1:] - P[:, :-1], dim=-1)                    # (B,n-1)
    s = torch.cat([torch.zeros(B, 1, device=P.device), torch.cumsum(seg, 1)], 1)  # (B,n)
    total = s[:, -1:]                                                 # (B,1)
    tgt = torch.linspace(0, 1, n, device=P.device)[None] * total       # (B,n)
    idx = (torch.searchsorted(s.contiguous(), tgt.contiguous(), right=True) - 1).clamp(0, n - 2)
    s0 = torch.gather(s, 1, idx); s1 = torch.gather(s, 1, idx + 1)
    f = ((tgt - s0) / (s1 - s0 + 1e-6)).unsqueeze(-1)                 # (B,n,1)
    P0 = torch.gather(P, 1, idx[..., None].expand(-1, -1, 3))
    P1 = torch.gather(P, 1, (idx + 1)[..., None].expand(-1, -1, 3))
    return P0 + f * (P1 - P0)


def knn(q, pc, k):
    # q (B,L,3), pc (B,P,3) -> idx (B,L,k) nearest pc points to each q
    dist = torch.cdist(q, pc)                    # (B,L,P)
    return dist.topk(k, largest=False, dim=-1).indices


GK = 20   # edge-conv graph neighbours


def edge_conv(idx, feat, mlp):
    # DGCNN EdgeConv on a precomputed STATIC graph idx (B,P,k). feat (B,P,Cin).
    B, Pn, Cin = feat.shape
    k = idx.shape[-1]
    fj = torch.gather(feat, 1, idx.reshape(B, Pn * k, 1).expand(-1, -1, Cin)).view(B, Pn, k, Cin)
    fi = feat[:, :, None, :].expand(-1, -1, k, -1)
    e = mlp(torch.cat([fi, fj - fi], -1))               # (B,P,k,Cout)
    return e.max(2).values                              # (B,P,Cout)


class Net(nn.Module):
    def __init__(self, C=256):
        super().__init__()
        self.ec1 = nn.Sequential(nn.Linear(2 * (3 + FDIM), 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.ec2 = nn.Sequential(nn.Linear(2 * 64, 128), nn.ReLU())
        self.ec3 = nn.Sequential(nn.Linear(2 * 128, 128), nn.ReLU())
        self.fuse = nn.Sequential(nn.Linear(64 + 128 + 128, C), nn.ReLU())
        self.mix = nn.Sequential(nn.Linear(2 * C, C), nn.ReLU())  # + global context
        self.emb = nn.Embedding(NL, 32)                          # for attention
        self.embO = nn.Embedding(NL, 32)                         # for offset head
        # offset head: from pooled window context -> unconstrained displacement (mm)
        self.offset = nn.Sequential(nn.Linear(2 * C + 32, 256), nn.ReLU(), nn.Dropout(0.1),
                                    nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 3))
        # surface snap: attention over K neighbours -> convex combination (on-surface)
        self.attn = nn.Sequential(nn.Linear(C + 3 + 32, 128), nn.ReLU(),
                                  nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        # per-landmark feature compressor for the contour head
        self.lmfeat = nn.Sequential(nn.Linear(C, 64), nn.ReLU())
        # CONTOUR-STRUCTURED REFINEMENT: one small 1-D conv stack PER contour
        # (region-specific). Input/landmark: [pos/scale(3), feat(64), emb(32)] = 99.
        cin = 3 + 64 + 32
        self.contour_nets = nn.ModuleList([
            nn.Sequential(nn.Conv1d(cin, 96, 5, padding=2), nn.ReLU(),
                          nn.Conv1d(96, 96, 3, padding=1), nn.ReLU(),
                          nn.Conv1d(96, 3, 1))
            for _ in CONTOURS])
        self.embC = nn.Embedding(NL, 32)
        self.C = C

    def backbone(self, pc, ft=None):
        pos = pc / SCALE
        gidx = knn(pos, pos, GK)                                  # graph on GEOMETRY only (B,P,GK)
        x = pos if ft is None else torch.cat([pos, ft], -1)      # per-point input = [xyz | curvature]
        h1 = edge_conv(gidx, x, self.ec1)                        # (B,P,64)
        h2 = edge_conv(gidx, h1, self.ec2)                       # (B,P,128)
        h3 = edge_conv(gidx, h2, self.ec3)                       # (B,P,128)
        h = self.fuse(torch.cat([h1, h2, h3], -1))               # (B,P,C)
        g = h.max(1, keepdim=True).values.expand(-1, pc.shape[1], -1)
        h = self.mix(torch.cat([h, g], -1))                      # (B,P,C) local+global
        return h

    def _gather(self, pc, h, q, k):
        B = pc.shape[0]
        idx = knn(q, pc, k).reshape(B, NL * k)                   # (B,L*k)
        featK = torch.gather(h, 1, idx[..., None].expand(-1, -1, self.C)).view(B, NL, k, self.C)
        posK = torch.gather(pc, 1, idx[..., None].expand(-1, -1, 3)).view(B, NL, k, 3)
        return featK, posK

    def step(self, pc, h, q):
        # 1) local context around current query -> unconstrained offset (can reach far)
        featK, posK = self._gather(pc, h, q, K)
        ctx = torch.cat([featK.mean(2), featK.max(2).values], -1)          # (B,L,2C)
        eo = self.embO(torch.arange(NL, device=pc.device))[None].expand(pc.shape[0], -1, -1)
        q1 = q + self.offset(torch.cat([ctx, eo], -1))                     # (B,L,3) mm
        # 2) re-window at q1 and snap to surface via soft-argmax (precision)
        featK2, posK2 = self._gather(pc, h, q1, K)
        rel = (posK2 - q1[:, :, None, :]) / SCALE
        e = self.emb(torch.arange(NL, device=pc.device))[None, :, None, :].expand(pc.shape[0], NL, K, 32)
        logit = self.attn(torch.cat([featK2, rel, e], -1)).squeeze(-1)
        w = F.softmax(logit, dim=-1)
        q2 = (w[..., None] * posK2).sum(2)                                 # (B,L,3)
        return q1, q2

    def contour_refine(self, pc, h, q):
        # region-specific along-contour correction (fixes tangential sliding)
        B = pc.shape[0]
        idx = knn(q, pc, 1).squeeze(-1)                            # (B,85) nearest cloud pt
        f = self.lmfeat(torch.gather(h, 1, idx[..., None].expand(-1, -1, self.C)))  # (B,85,64)
        e = self.embC(torch.arange(NL, device=pc.device))[None].expand(B, -1, -1)
        inp = torch.cat([q / SCALE, f, e], -1)                     # (B,85,99)
        out = torch.zeros(B, NL, 3, device=pc.device)
        for (lo, hi), net in zip(CONTOURS, self.contour_nets):
            seg = inp[:, lo:hi + 1, :].transpose(1, 2)             # (B,99,L)
            out[:, lo:hi + 1] = net(seg).transpose(1, 2)           # (B,L,3) correction
        return q + out

    def forward(self, pc, q0, ft=None):
        h = self.backbone(pc, ft)
        outs = []           # list of (q1_offset, q2_snap) per pass
        q = q0
        for _ in range(T):
            q1, q2 = self.step(pc, h, q)
            outs.append((q1, q2))
            q = q2
        q_ref = self.contour_refine(pc, h, q)                      # final structured refinement
        if EQUI_ON and EQUI:            # enforce GT's equal-arc-length construction
            # build by concatenation (NOT in-place slice assign, which breaks autograd)
            parts, prev = [], 0
            for ci in sorted(EQUI):
                lo, hi = int(CONTOURS[ci][0]), int(CONTOURS[ci][1])
                if lo > prev:
                    parts.append(q_ref[:, prev:lo])
                parts.append(equidistant_resample(q_ref[:, lo:hi + 1]))
                prev = hi + 1
            if prev < NL:
                parts.append(q_ref[:, prev:])
            q_ref = torch.cat(parts, 1)
        outs.append((q, q_ref))
        return outs


# -------- SSM projection (numpy, matches inference) --------
def procrustes(src, tgt):
    ms, mt = src.mean(0), tgt.mean(0)
    A, B = src - ms, tgt - mt
    U, _, Vt = np.linalg.svd(A.T @ B)
    Rr = U @ Vt
    if np.linalg.det(Rr) < 0:
        U[:, -1] *= -1; Rr = U @ Vt
    s = (B * (A @ Rr)).sum() / (A * A).sum()
    return s, Rr, ms, mt


def ssm_project(pts):
    s, Rr, ms, mt = procrustes(pts, ssm_mean.reshape(NL, 3))
    aligned = s * ((pts - ms) @ Rr) + mt
    coeff = (aligned.flatten() - ssm_mean) @ ssm_comp.T
    recon = (ssm_mean + coeff @ ssm_comp).reshape(NL, 3)
    return ((recon - mt) @ Rr.T) / s + ms


cl_d, co_d, tr_d = clouds.to(dev), coarse.to(dev), true.to(dev)
ft_d = feats.to(dev) if feats is not None else None

# fixed TTA rotations (small angles about each axis), applied in canonical frame
def _rot(ax, a):
    x, y, z = ax; c, s, C = np.cos(a), np.sin(a), 1 - np.cos(a)
    return np.array([[c+x*x*C, x*y*C-z*s, x*z*C+y*s],
                     [y*x*C+z*s, c+y*y*C, y*z*C-x*s],
                     [z*x*C-y*s, z*y*C+x*s, c+z*z*C]])
TTA_ROTS = [np.eye(3)] + [_rot(a, ang) for a in ([1,0,0],[0,1,0],[0,0,1])
                          for ang in (0.18, -0.18)]   # 1 + 6 views


def predict_raw(net, i, tta=False):
    """best raw (world-frame) prediction for val ear i, optional TTA-averaged."""
    R, cc = Rmat[i], c0[i]
    rots = TTA_ROTS if tta else [np.eye(3)]
    acc = np.zeros((NL, 3))
    with torch.no_grad():
        for M in rots:
            Mt = torch.tensor(M, dtype=torch.float32, device=dev)
            pc = cl_d[i:i+1] @ Mt.T
            co = co_d[i:i+1] @ Mt.T
            ft = ft_d[i:i+1] if ft_d is not None else None    # curvature is rotation-invariant
            p = net(pc, co, ft)[-1][1][0].cpu().numpy().astype(np.float64)
            acc += p @ M            # un-rotate back to canonical
    pred = acc / len(rots)
    return pred @ R + cc           # world frame


def evaluate_percontour(net):
    """mean error per contour (raw), to verify the equidistance lever targets 55-74/75-84"""
    net.eval()
    acc = np.zeros(NL); n = 0
    for i in va_idx:
        pw = predict_raw(net, i)
        R, cc = Rmat[i], c0[i]
        gtw = true[i].numpy().astype(np.float64) @ R + cc
        acc += np.linalg.norm(pw - gtw, axis=1); n += 1
    per = acc / n
    return [per[lo:hi + 1].mean() for lo, hi in CONTOURS]


def evaluate(net, tta=False):
    net.eval()
    raw, proj, blend = [], [], []
    for i in va_idx:
        pw = predict_raw(net, i, tta=tta)
        R, cc = Rmat[i], c0[i]
        gtw = true[i].numpy().astype(np.float64) @ R + cc
        pj = ssm_project((pw - cc) @ R.T) @ R + cc                 # projected, world
        bl = 0.5 * pw + 0.5 * pj                                   # blend (final recipe)
        raw.append(np.linalg.norm(pw - gtw, axis=1).mean())
        proj.append(np.linalg.norm(pj - gtw, axis=1).mean())
        blend.append(np.linalg.norm(bl - gtw, axis=1).mean())
    return float(np.mean(raw)), float(np.mean(proj)), float(np.mean(blend))


def rand_rot(B):
    # random axis, moderate angle (canonical frame -> keep it modest)
    ax = torch.randn(B, 3, device=dev); ax = ax / ax.norm(dim=1, keepdim=True)
    ang = (torch.rand(B, device=dev) - .5) * (2 * 0.60)          # ±~34deg
    c, s = ang.cos(), ang.sin()
    x, y, z = ax[:, 0], ax[:, 1], ax[:, 2]
    C = 1 - c
    R = torch.stack([
        torch.stack([c + x*x*C, x*y*C - z*s, x*z*C + y*s], -1),
        torch.stack([y*x*C + z*s, c + y*y*C, y*z*C - x*s], -1),
        torch.stack([z*x*C - y*s, z*y*C + x*s, c + z*z*C], -1)], 1)  # (B,3,3)
    return R


NSUB = min(1280, P)   # random point subsample per step (anti-overfit augmentation)


def augment(pc, ft, q, tg):
    B, Pn = pc.shape[0], pc.shape[1]
    # random point subsample (different subset each time) — key regularizer; vectorized
    sub = torch.rand(B, Pn, device=dev).argsort(1)[:, :NSUB]
    pc = torch.gather(pc, 1, sub[..., None].expand(-1, -1, 3))
    if ft is not None:                                            # subsample curvature with SAME idx; no rotation (invariant)
        ft = torch.gather(ft, 1, sub[..., None].expand(-1, -1, ft.shape[-1]))
    R = rand_rot(B)
    s = 1 + (torch.rand(B, 1, 1, device=dev) - .5) * 0.20
    pc = torch.bmm(pc, R.transpose(1, 2)) * s + torch.randn_like(pc) * 0.25
    q = torch.bmm(q, R.transpose(1, 2)) * s + torch.randn_like(q) * 0.9    # jitter coarse init (offset-head robustness)
    tg = torch.bmm(tg, R.transpose(1, 2)) * s
    return pc, ft, q, tg


net = Net().to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=5e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
sup_w = torch.tensor([0.5 ** (T - 1 - t) for t in range(T)], device=dev)  # later passes weighted more
sup_w = sup_w / sup_w.sum()

# Per-landmark loss weights: the ENDPOINTS of the equidistant contours anchor the
# arc-length parametrization, so their error propagates to every point on that
# contour -> weight them higher (END_W=1 disables).
END_W = float(os.environ.get("END_W", "1.0"))
LM_W = torch.ones(NL, device=dev)
if EQUI_ON and END_W != 1.0:
    for ci in EQUI:
        lo, hi = int(CONTOURS[ci][0]), int(CONTOURS[ci][1])
        LM_W[lo] = END_W; LM_W[hi] = END_W
    print(f"endpoint loss weight {END_W} on landmarks "
          f"{[int(CONTOURS[c][j]) for c in EQUI for j in (0, 1)]}")

best = 99.0; t0 = time.time()
for ep in range(EPOCHS):
    net.train()
    perm = np.random.permutation(tr_idx)
    for b in range(0, len(perm), BS):
        bi = perm[b:b+BS]
        ftb = ft_d[bi] if ft_d is not None else None
        pc, ftb, q, tg = augment(cl_d[bi], ftb, co_d[bi], tr_d[bi])
        opt.zero_grad()
        outs = net(pc, q, ftb)
        # deep supervision: offset relocation (q1) + surface snap (q2) per pass,
        # later passes weighted more; then the contour-refined final output (strong).
        loss = 0.0
        for t in range(T):
            q1, q2 = outs[t]
            loss = loss + sup_w[t] * (0.4 * ((q1 - tg) ** 2).sum(-1).mean()
                                      + ((q2 - tg) ** 2).sum(-1).mean())
        q_pre, q_ref = outs[T]                       # contour-refinement stage
        if METRIC_LOSS:
            # loss == the official metric: mean Euclidean distance (not squared).
            # MSE over-weights large errors relative to what is actually scored.
            r2 = ((q_ref - tg) ** 2).sum(-1)
            loss = loss + (torch.sqrt(r2 + 1e-8) * LM_W).mean() / LM_W.mean()
        elif HUBER:                                  # Huber on Euclidean dist (de-weights tail)
            r = torch.norm(q_ref - tg, dim=-1)
            loss = loss + torch.where(r < HUBER, 0.5 * r ** 2, HUBER * (r - 0.5 * HUBER)).mean()
        else:
            se = ((q_ref - tg) ** 2).sum(-1)         # (B,85) per-landmark squared error
            loss = loss + (se * LM_W).mean() / LM_W.mean()
        if SPACING_W > 0:                            # SOFT equidistance: penalize uneven spacing
            for ci in EQUI:                          # (gentler than the hard resample layer)
                lo, hi = int(CONTOURS[ci][0]), int(CONTOURS[ci][1])
                seg = torch.norm(q_ref[:, lo + 1:hi + 1] - q_ref[:, lo:hi], dim=-1)   # (B,n-1)
                loss = loss + SPACING_W * ((seg - seg.mean(1, keepdim=True)) ** 2).mean()
        loss.backward(); opt.step()
    sched.step()
    if (ep + 1) % 25 == 0:
        r, pr, bl = evaluate(net)                    # save/track by BLEND (final recipe)
        flag = "  <== BEATS 1.29" if bl < 1.29 else ""
        tta_s = ""
        if (ep + 1) % 200 == 0:
            tta_s = " | contours " + "/".join(f"{v:.2f}" for v in evaluate_percontour(net))
        print(f"ep {ep+1:4d} | raw {r:.3f} | proj {pr:.3f} | blend {bl:.3f}mm | best {min(best,bl):.3f}{tta_s} | {time.time()-t0:.0f}s{flag}", flush=True)
        if bl < best:
            best = bl
            sd = {k: v.cpu().numpy() for k, v in net.state_dict().items()}
            np.savez(OUT, **sd, best=best, K=K, T=T, SCALE=SCALE, seed=SEED)

# reload best, dump per-ear val predictions (raw world) for offline ensembling
ck = np.load(OUT); net.load_state_dict({k: torch.tensor(ck[k]) for k in ck.files
                                        if k not in ("best","K","T","SCALE","seed")})
net.eval()
val_raw = np.stack([predict_raw(net, i, tta=False) for i in va_idx])       # (Nval,85,3) world
val_tta = np.stack([predict_raw(net, i, tta=True) for i in va_idx])
val_gt = np.stack([true[i].numpy().astype(np.float64) @ Rmat[i] + c0[i] for i in va_idx])
np.savez(OUT.replace(".npz", "_valpred.npz"), raw=val_raw, tta=val_tta, gt=val_gt, va_idx=va_idx)
r0, p0, b0 = evaluate(net); _, _, bt = evaluate(net, tta=True)
print(f"DONE seed {SEED}: best-blend {best:.3f} | reload raw {r0:.3f} proj {p0:.3f} blend {b0:.3f} blend+TTA {bt:.3f}mm -> {OUT}", flush=True)
