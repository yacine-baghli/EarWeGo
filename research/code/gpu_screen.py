"""
BASE-MODEL RETRAIN SCREENING — one representative fold, ONE change per run.

Variants (VARIANT env), each differing from `base` in exactly one respect:
  base     tied 4 passes, K=48, GK=20, 2048 pts (run twice to measure TRAINING NOISE)
  untied4  4 passes with UNTIED head weights            (only weight sharing changes)
  untied6  6 untied coarse-to-fine passes with FIXED-RADIUS neighbourhoods, calibrated
           from the baseline: measured K=48 window = 7.35mm, GK=20 graph = 4.94mm,
           coarse residual p99 = 10.6mm, final residual p90 = 2.5mm
           -> radii 11.0, 9.0, 7.4, 5.5, 4.0, 3.0 mm
           -> max offsets 7.0, 4.5, 3.0, 2.0, 1.2, 0.7 mm
  pts4096  4096 points with K=96 / GK=40 (holds the PHYSICAL radius constant) and
           sampling WITHOUT replacement at train and inference
  normals  XYZ + consistently oriented triangle-derived normals (no curvature channels)
  fusion2  two independent surface samples per step through the shared backbone,
           landmark-level mean/max/variance fusion + weak consistency loss
  chamfer  end-to-end ordered + curve-Chamfer objective on the BASE model

Reports for every run: ordered MLE, per-contour MLE, directional decomposition
(tangent/across/normal), fresh-sample prediction variance, paired per-subject bootstrap
interval vs the reference, train-vs-val curve, parameter count and runtime.

  VARIANT=base SEED=0 FOLD=0 python3 gpu_screen.py
"""
import os, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

WORK = "/home/ubuntu/ear"
VARIANT = os.environ.get("VARIANT", "base")
SEED = int(os.environ.get("SEED", "0"))
FOLD = int(os.environ.get("FOLD", "0"))
EPOCHS = int(os.environ.get("EPOCHS", "1200"))
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED); np.random.seed(SEED)

CFG = dict(base=dict(npass=4, untied=False, K=48, GK=20, npts=2048),
           untied4=dict(npass=4, untied=True, K=48, GK=20, npts=2048),
           untied6=dict(npass=6, untied=True, K=96, GK=20, npts=2048,
                        radii=[11.0, 9.0, 7.4, 5.5, 4.0, 3.0],
                        offs=[7.0, 4.5, 3.0, 2.0, 1.2, 0.7]),
           pts4096=dict(npass=4, untied=False, K=96, GK=40, npts=4096),
           normals=dict(npass=4, untied=False, K=48, GK=20, npts=2048, use_nrm=True),
           fusion2=dict(npass=4, untied=False, K=48, GK=20, npts=2048, fuse=2),
           chamfer=dict(npass=4, untied=False, K=48, GK=20, npts=2048, chamfer=True))[VARIANT]
NPASS, UNTIED = CFG["npass"], CFG["untied"]
K, GK, NPTS = CFG["K"], CFG["GK"], CFG["npts"]
RADII = CFG.get("radii"); OFFS = CFG.get("offs")
USE_NRM = CFG.get("use_nrm", False); FUSE = CFG.get("fuse", 1); CHAMFER = CFG.get("chamfer", False)
SCALE, NL = 30.0, 85
BS, LR = 16, 1.5e-3

DATA = os.environ.get("DATA", f"{WORK}/screen_data_{NPTS}.npz")
d = np.load(DATA, allow_pickle=True)
clouds = torch.tensor(d["clouds"]).float()          # (N,M,NPTS,3) fresh samples
nrm = torch.tensor(d["nrm"]).float() if ("nrm" in d.files and USE_NRM) else None
coarse = torch.tensor(d["coarse"]).float(); true = torch.tensor(d["true"]).float()
Rm, c0 = d["R"], d["c0"]
CONTOURS = [(0, 24), (25, 54), (55, 74), (75, 84)]
NE, M = clouds.shape[0], clouds.shape[1]
subj = np.arange(NE) // 2
rs = np.random.RandomState(12345)
val_s = set(np.array_split(rs.permutation(np.unique(subj)), 5)[FOLD].tolist())
va_idx = np.array([i for i in range(NE) if subj[i] in val_s])
tr_idx = np.array([i for i in range(NE) if subj[i] not in val_s])
print(f"[{VARIANT} seed{SEED} fold{FOLD}] {len(tr_idx)} train / {len(va_idx)} val ears, "
      f"{NPTS} pts, passes={NPASS} untied={UNTIED} K={K} GK={GK} fuse={FUSE}", flush=True)


def knn(q, pc, k):
    return torch.cdist(q, pc).topk(k, largest=False, dim=-1).indices


def edge_conv(gidx, feat, mlp):
    B, P, C = feat.shape; k = gidx.shape[-1]
    fj = torch.gather(feat, 1, gidx.reshape(B, P * k, 1).expand(-1, -1, C)).view(B, P, k, C)
    fi = feat[:, :, None, :].expand(-1, -1, k, -1)
    return mlp(torch.cat([fi, fj - fi], -1)).max(2).values


class Head(nn.Module):
    """one refinement pass: offset (bounded) then surface snap"""
    def __init__(self, C, max_off=None):
        super().__init__()
        self.emb = nn.Embedding(NL, 32); self.embO = nn.Embedding(NL, 32)
        self.offset = nn.Sequential(nn.Linear(2 * C + 32, 256), nn.ReLU(), nn.Dropout(0.1),
                                    nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 3))
        self.attn = nn.Sequential(nn.Linear(C + 3 + 32, 128), nn.ReLU(),
                                  nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.C = C; self.max_off = max_off

    def gather(self, pc, h, q, radius):
        B = pc.shape[0]
        idx = knn(q, pc, K).reshape(B, NL * K)
        fK = torch.gather(h, 1, idx[..., None].expand(-1, -1, self.C)).view(B, NL, K, self.C)
        pK = torch.gather(pc, 1, idx[..., None].expand(-1, -1, 3)).view(B, NL, K, 3)
        dist = (pK - q[:, :, None, :]).norm(dim=-1)
        mask = (dist <= radius) if radius is not None else None
        if mask is not None:                          # keep at least the 8 nearest
            keep8 = dist <= dist.topk(8, largest=False, dim=-1).values[..., -1:]
            mask = mask | keep8
        return fK, pK, mask

    def forward(self, pc, h, q, radius=None):
        fK, pK, mask = self.gather(pc, h, q, radius)
        if mask is None:
            ctx = torch.cat([fK.mean(2), fK.max(2).values], -1)
        else:
            w = mask.float()[..., None]
            ctx = torch.cat([(fK * w).sum(2) / w.sum(2).clamp(min=1),
                             fK.masked_fill(~mask[..., None], -1e4).max(2).values], -1)
        eo = self.embO(torch.arange(NL, device=pc.device))[None].expand(pc.shape[0], -1, -1)
        off = self.offset(torch.cat([ctx, eo], -1))
        if self.max_off is not None:
            off = self.max_off * torch.tanh(off / max(self.max_off, 1e-6))
        q1 = q + off
        fK2, pK2, mask2 = self.gather(pc, h, q1, radius)
        rel = (pK2 - q1[:, :, None, :]) / SCALE
        e = self.emb(torch.arange(NL, device=pc.device))[None, :, None, :].expand(
            pc.shape[0], NL, K, 32)
        logit = self.attn(torch.cat([fK2, rel, e], -1)).squeeze(-1)
        if mask2 is not None:
            logit = logit.masked_fill(~mask2, -1e4)
        w = torch.softmax(logit, -1)
        return q1, (w[..., None] * pK2).sum(2)


class Net(nn.Module):
    def __init__(self, C=256, cin=3):
        super().__init__()
        self.ec1 = nn.Sequential(nn.Linear(2 * cin, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.ec2 = nn.Sequential(nn.Linear(2 * 64, 128), nn.ReLU())
        self.ec3 = nn.Sequential(nn.Linear(2 * 128, 128), nn.ReLU())
        self.fuse = nn.Sequential(nn.Linear(320, C), nn.ReLU())
        self.mix = nn.Sequential(nn.Linear(2 * C, C), nn.ReLU())
        nh = NPASS if UNTIED else 1
        self.heads = nn.ModuleList([Head(C, None if OFFS is None else OFFS[min(i, len(OFFS)-1)])
                                    for i in range(nh)])
        self.lmfeat = nn.Sequential(nn.Linear(C, 64), nn.ReLU())
        self.contour_nets = nn.ModuleList([
            nn.Sequential(nn.Conv1d(3 + 64 + 32, 96, 5, padding=2), nn.ReLU(),
                          nn.Conv1d(96, 96, 3, padding=1), nn.ReLU(), nn.Conv1d(96, 3, 1))
            for _ in CONTOURS])
        self.embC = nn.Embedding(NL, 32)
        self.C = C
        if FUSE > 1:
            self.fus = nn.Sequential(nn.Linear(3 * 3, 32), nn.ReLU(), nn.Linear(32, 3))

    def backbone(self, pc, ft=None):
        pos = pc / SCALE
        gidx = knn(pos, pos, GK)
        x = pos if ft is None else torch.cat([pos, ft], -1)
        h1 = edge_conv(gidx, x, self.ec1); h2 = edge_conv(gidx, h1, self.ec2)
        h3 = edge_conv(gidx, h2, self.ec3)
        h = self.fuse(torch.cat([h1, h2, h3], -1))
        g = h.max(1, keepdim=True).values.expand(-1, pc.shape[1], -1)
        return self.mix(torch.cat([h, g], -1))

    def contour(self, pc, h, q):
        B = pc.shape[0]
        idx = knn(q, pc, 1).squeeze(-1)
        f = self.lmfeat(torch.gather(h, 1, idx[..., None].expand(-1, -1, self.C)))
        e = self.embC(torch.arange(NL, device=pc.device))[None].expand(B, -1, -1)
        inp = torch.cat([q / SCALE, f, e], -1)
        out = torch.zeros(B, NL, 3, device=pc.device)
        for (lo, hi), net in zip(CONTOURS, self.contour_nets):
            out[:, lo:hi + 1] = net(inp[:, lo:hi + 1].transpose(1, 2)).transpose(1, 2)
        return q + out

    def one(self, pc, q0, ft=None):
        h = self.backbone(pc, ft)
        outs = []; q = q0
        for i in range(NPASS):
            hd = self.heads[i] if UNTIED else self.heads[0]
            r = None if RADII is None else RADII[min(i, len(RADII) - 1)]
            q1, q2 = hd(pc, h, q, r)
            outs.append((q1, q2)); q = q2
        return outs, self.contour(pc, h, q)

    def forward(self, pcs, q0, fts=None):
        """pcs: list of FUSE clouds (fresh samples). Returns (outs_last, final, per_sample)"""
        finals = []; outs = None
        for s, pc in enumerate(pcs):
            o, fin = self.one(pc, q0, None if fts is None else fts[s])
            outs = o; finals.append(fin)
        if len(finals) == 1:
            return outs, finals[0], finals
        st = torch.stack(finals, 0)                       # (S,B,NL,3)
        agg = torch.cat([st.mean(0), st.max(0).values, st.var(0, unbiased=False)], -1)
        return outs, st.mean(0) + self.fus(agg), finals


# ----------------------------------------------------------------- training
def rand_rot(B):
    ax = torch.randn(B, 3, device=dev); ax = ax / ax.norm(dim=1, keepdim=True)
    ang = (torch.rand(B, device=dev) - .5) * 1.2
    c, s = ang.cos(), ang.sin(); x, y, z = ax[:, 0], ax[:, 1], ax[:, 2]; C_ = 1 - c
    return torch.stack([
        torch.stack([c + x*x*C_, x*y*C_ - z*s, x*z*C_ + y*s], -1),
        torch.stack([y*x*C_ + z*s, c + y*y*C_, y*z*C_ - x*s], -1),
        torch.stack([z*x*C_ - y*s, z*y*C_ + x*s, c + z*z*C_], -1)], 1)


NSUB = min(int(NPTS * 0.625), NPTS)          # same subsample FRACTION as the baseline


def augment(pc_list, ft_list, q, tg):
    B = q.shape[0]
    R = rand_rot(B); sc = 1 + (torch.rand(B, 1, 1, device=dev) - .5) * 0.20
    out_pc, out_ft = [], []
    for s, pc in enumerate(pc_list):
        sub = torch.rand(B, pc.shape[1], device=dev).argsort(1)[:, :NSUB]
        p = torch.gather(pc, 1, sub[..., None].expand(-1, -1, 3))
        p = torch.bmm(p, R.transpose(1, 2)) * sc + torch.randn_like(p) * 0.25
        out_pc.append(p)
        if ft_list is not None:
            f = torch.gather(ft_list[s], 1, sub[..., None].expand(-1, -1, ft_list[s].shape[-1]))
            out_ft.append(torch.bmm(f, R.transpose(1, 2)))       # normals rotate
    q = torch.bmm(q, R.transpose(1, 2)) * sc + torch.randn_like(q) * 0.9
    tg = torch.bmm(tg, R.transpose(1, 2)) * sc
    return out_pc, (out_ft if ft_list is not None else None), q, tg


def resample_dense(P, m=64):
    s = torch.cat([torch.zeros(P.shape[0], 1, device=P.device), (P[:, 1:] - P[:, :-1]).norm(dim=-1).cumsum(1)], 1)
    Lt = s[:, -1:]; qq = torch.linspace(0, 1, m, device=P.device)[None] * Lt
    j = torch.clamp(torch.searchsorted(s.contiguous(), qq.contiguous()) - 1, 0, P.shape[1] - 2)
    sj = torch.gather(s, 1, j); sj1 = torch.gather(s, 1, j + 1)
    f = ((qq - sj) / (sj1 - sj).clamp(min=1e-9))[..., None]
    Pj = torch.gather(P, 1, j[..., None].expand(-1, -1, 3))
    Pj1 = torch.gather(P, 1, (j + 1)[..., None].expand(-1, -1, 3))
    return Pj + f * (Pj1 - Pj)


cl_d = clouds.to(dev); co_d = coarse.to(dev); tr_d = true.to(dev)
nrm_d = nrm.to(dev) if nrm is not None else None
net = Net(cin=3 + (3 if USE_NRM else 0)).to(dev)
NPARAM = sum(p.numel() for p in net.parameters())
print(f"params: {NPARAM:,}", flush=True)
opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=5e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
sup_w = torch.tensor([0.5 ** (NPASS - 1 - t) for t in range(NPASS)], device=dev); sup_w /= sup_w.sum()
curve = []; t0 = time.time()


def evaluate(idx, nsample=4):
    net.eval(); acc = []
    with torch.no_grad():
        for i in idx:
            ps = [cl_d[i, j % M][None] for j in range(nsample)]
            fs = [nrm_d[i, j % M][None] for j in range(nsample)] if nrm_d is not None else None
            per = []
            for s in range(nsample):
                _, fin, _ = net([ps[s]] * FUSE, co_d[i:i+1], None if fs is None else [fs[s]] * FUSE)
                per.append(fin[0].cpu().numpy().astype(np.float64))
            per = np.stack(per)
            acc.append((per.mean(0), per))
    P = np.stack([a[0] for a in acc]); PS = np.stack([a[1] for a in acc])
    Pw = np.stack([P[k] @ Rm[i] + c0[i] for k, i in enumerate(idx)])
    Gw = np.stack([true[i].numpy().astype(np.float64) @ Rm[i] + c0[i] for i in idx])
    var = float(np.linalg.norm(PS - PS.mean(1, keepdims=True), axis=3).mean())
    return Pw, Gw, var


best = (9e9, None)
for ep in range(EPOCHS):
    net.train(); perm = np.random.permutation(tr_idx)
    for b in range(0, len(perm), BS):
        bi = perm[b:b + BS]
        js = np.random.randint(0, M, FUSE)
        pcs = [cl_d[bi, j] for j in js]
        fts = [nrm_d[bi, j] for j in js] if nrm_d is not None else None
        pcs, fts, q, tg = augment(pcs, fts, co_d[bi], tr_d[bi])
        opt.zero_grad()
        outs, fin, per = net(pcs, q, fts)
        loss = 0.0
        for t in range(NPASS):
            q1, q2 = outs[t]
            loss = loss + sup_w[t] * (0.4 * ((q1 - tg) ** 2).sum(-1).mean() + ((q2 - tg) ** 2).sum(-1).mean())
        loss = loss + ((fin - tg) ** 2).sum(-1).mean()
        if CHAMFER:
            for lo, hi in CONTOURS:
                A = resample_dense(fin[:, lo:hi+1]); B = resample_dense(tg[:, lo:hi+1])
                D = torch.cdist(A, B)
                loss = loss + 0.3 * (D.min(-1).values.mean(-1) + D.min(-2).values.mean(-1)).mean()
        if FUSE > 1:                                   # weak consistency between samples
            st = torch.stack(per, 0)
            loss = loss + 0.05 * (st - st.mean(0, keepdim=True)).norm(dim=-1).mean()
        loss.backward(); opt.step()
    sch.step()
    if (ep + 1) % 100 == 0:
        Pv, Gv, var = evaluate(va_idx, 2)
        Pt, Gt, _ = evaluate(tr_idx[:40], 1)
        vm = float(np.linalg.norm(Pv - Gv, axis=2).mean()); tm = float(np.linalg.norm(Pt - Gt, axis=2).mean())
        curve.append({"epoch": ep + 1, "train_MLE": round(tm, 4), "val_MLE": round(vm, 4)})
        print(f"  ep{ep+1:4d} train {tm:.4f} val {vm:.4f} sampvar {var:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if vm < best[0]:
            best = (vm, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()})

if best[1] is not None:
    net.load_state_dict(best[1])
Pv, Gv, var = evaluate(va_idx, 4)
E = np.linalg.norm(Pv - Gv, axis=2)
res = {"variant": VARIANT, "seed": SEED, "fold": FOLD, "params": int(NPARAM),
       "runtime_s": round(time.time() - t0, 1), "epochs": EPOCHS,
       "config": {k: v for k, v in CFG.items()},
       "ordered_MLE_mm": round(float(E.mean()), 4),
       "median_mm": round(float(np.median(E)), 4),
       "P90_mm": round(float(np.percentile(E, 90)), 4),
       "per_contour_MLE_mm": {f"{lo}-{hi}": round(float(E[:, lo:hi+1].mean()), 4) for lo, hi in CONTOURS},
       "fresh_sample_pred_variance_mm": round(var, 4),
       "train_val_curve": curve,
       "per_ear_MLE": [round(float(x), 5) for x in E.mean(1)],
       "val_ear_index": [int(i) for i in va_idx]}
json.dump(res, open(f"{WORK}/screen_{VARIANT}_s{SEED}_f{FOLD}.json", "w"), indent=1)
np.save(f"{WORK}/screen_{VARIANT}_s{SEED}_f{FOLD}.npy", Pv)
print(f"\n[{VARIANT} s{SEED} f{FOLD}] ordered MLE {E.mean():.4f} | per-contour "
      + "/".join(f"{E[:, lo:hi+1].mean():.3f}" for lo, hi in CONTOURS)
      + f" | sampvar {var:.4f} | {NPARAM/1e3:.0f}k params | {time.time()-t0:.0f}s", flush=True)
