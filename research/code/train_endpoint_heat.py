"""
ENDPOINT LOCALISATION (heatmap) instead of global affine regression.

The affine-regression gate failed by MEMORISATION: it reached the oracle on training
ears (1.567 -> 0.494, oracle 0.496) while gaining nothing held-out. Diagnosis: regressing
an abstract global scalar from mean/max-pooled features has a weak inductive bias.

Phase is really a LOCALISATION question: "where along this curve does the annotation
convention place the endpoint?". So predict, for each of the two contour endpoints, a
distribution over the T ordered locations and take its soft-argmax. The two localised
endpoints then DEFINE the affine reparametrisation, instead of it being regressed.
Localisation shares weights across positions and cannot memorise a per-ear constant,
which is exactly the failure mode we observed.

    CONTOUR=inner python scratch/train_endpoint_heat.py
"""
import os
import numpy as np
import torch
import torch.nn as nn
from scipy import stats

CNAME = os.environ.get("CONTOUR", "inner")
PAD = 3.0
EPOCHS = int(os.environ.get("EPOCHS", "300"))
torch.manual_seed(0); np.random.seed(0)

z = np.load(f"scratch/seq_{CNAME}.npz")
SEQ = torch.tensor(z["seq"]).float()
PRED = torch.tensor(z["pred"]).double()
GT = torch.tensor(z["gt"]).double()
fold, subj = z["fold"], z["subj"]
N, T, C = SEQ.shape
n = PRED.shape[1]
mu, sd = SEQ.mean((0, 1)), SEQ.std((0, 1)) + 1e-6


def arc_np(P):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]


def arc_t(P):
    return torch.cat([torch.zeros(P.shape[0], 1, dtype=P.dtype),
                      torch.cumsum((P[:, 1:] - P[:, :-1]).norm(dim=-1), 1)], 1)


def warp_ab(P, a, b):
    s = arc_t(P); L = s[:, -1:]; s0 = s[:, :1]
    sq = torch.minimum(torch.clamp(s0 + a[:, None] * (s - s0) + b[:, None], min=-PAD), L + PAD)
    j = torch.clamp(torch.searchsorted(s.contiguous(), sq.contiguous()) - 1, 0, n - 2)
    sj = torch.gather(s, 1, j); sj1 = torch.gather(s, 1, j + 1)
    f = (sq - sj) / torch.clamp(sj1 - sj, min=1e-9)
    Pj = torch.gather(P, 1, j[..., None].expand(-1, -1, 3))
    Pj1 = torch.gather(P, 1, (j + 1)[..., None].expand(-1, -1, 3))
    val = Pj + f[..., None] * (Pj1 - Pj)
    d0 = P[:, 1] - P[:, 0]; d0 = d0 / d0.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    d1 = P[:, -1] - P[:, -2]; d1 = d1 / d1.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    val = torch.where((sq <= 0)[..., None], P[:, :1] + sq[..., None] * d0[:, None, :], val)
    val = torch.where((sq >= L)[..., None], P[:, -1:] + (sq - L)[..., None] * d1[:, None, :], val)
    return val


def mle(P, G):
    return (P - G).norm(dim=-1).mean(-1)


# ---- targets: where do the GT endpoints project onto OUR predicted curve? ----
Pn, Gn = PRED.numpy(), GT.numpy()
tt = np.zeros((N, 2))                      # arc-length positions (mm) of the 2 endpoints
Lall = np.zeros(N)
for i in range(N):
    s = arc_np(Pn[i]); L = s[-1]; Lall[i] = L
    grid = np.linspace(-PAD, L + PAD, 1200)
    # evaluate the polyline on the grid (with linear extrapolation)
    j = np.clip(np.searchsorted(s, grid) - 1, 0, n - 2)
    f = (grid - s[j]) / np.maximum(s[j + 1] - s[j], 1e-12)
    Q = Pn[i][j] + f[:, None] * (Pn[i][j + 1] - Pn[i][j])
    d0 = Pn[i][1] - Pn[i][0]; d0 /= max(np.linalg.norm(d0), 1e-9)
    d1 = Pn[i][-1] - Pn[i][-2]; d1 /= max(np.linalg.norm(d1), 1e-9)
    m0 = grid <= 0; Q[m0] = Pn[i][0] + grid[m0][:, None] * d0
    m1 = grid >= L; Q[m1] = Pn[i][-1] + (grid[m1] - L)[:, None] * d1
    for e, gidx in enumerate((0, n - 1)):
        tt[i, e] = grid[np.argmin(np.linalg.norm(Q - Gn[i][gidx], axis=1))]
# normalised targets in [0,1] over the padded grid
tgt = torch.tensor((tt + PAD) / (Lall[:, None] + 2 * PAD)).float()
print(f"contour {CNAME}: endpoint targets (normalised) mean {tgt.mean(0).numpy().round(3)}, "
      f"sd {tgt.std(0).numpy().round(3)}")


class EndpointNet(nn.Module):
    """per-location logits for the 2 endpoints -> soft-argmax position (localisation)"""
    def __init__(self, cin=C, ch=24):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(cin, ch, 5, padding=2), nn.ReLU(),
            nn.Conv1d(ch, ch, 5, padding=4, dilation=2), nn.ReLU(),
            nn.Conv1d(ch, ch, 5, padding=8, dilation=4), nn.ReLU(),
            nn.Conv1d(ch, 2, 1))
        pos = torch.linspace(0, 1, T)
        self.register_buffer("pos", pos)

    def forward(self, x):
        lg = self.body(x.transpose(1, 2))                 # (B,2,T)
        w = torch.softmax(lg, dim=-1)
        return (w * self.pos).sum(-1)                     # (B,2) in [0,1]


print(f"params: {sum(p.numel() for p in EndpointNet().parameters()):,}")
base = mle(PRED, GT).numpy()
pa = np.ones(N); pb = np.zeros(N)
Lt = torch.tensor(Lall)
for f in sorted(set(fold.tolist())):
    te = fold == f; tr = ~te
    Xtr, Xte = (SEQ[tr] - mu) / sd, (SEQ[te] - mu) / sd
    net = EndpointNet()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    ntr = int(tr.sum()); ytr = tgt[tr]
    for ep in range(EPOCHS):
        net.train(); perm = np.random.permutation(ntr)
        for s0 in range(0, ntr, 32):
            bi = perm[s0:s0 + 32]
            p = net(Xtr[bi])
            loss = (p - ytr[bi]).abs().mean()             # localisation loss
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    net.eval()
    with torch.no_grad():
        p = net(Xte).double()
    # endpoints -> affine: s0 -> t0, s_{n-1} -> t1
    Lte = Lt[te]
    t0 = p[:, 0] * (Lte + 2 * PAD) - PAD
    t1 = p[:, 1] * (Lte + 2 * PAD) - PAD
    a = ((t1 - t0) / Lte).clamp(0.82, 1.18)
    b = t0                                              # s0 = 0 maps to t0
    pa[te] = a.numpy(); pb[te] = b.numpy()
    with torch.no_grad():
        e = mle(warp_ab(PRED[te], a, b), GT[te]).numpy()
    print(f"  fold {f}: base {base[te].mean():.4f} -> endpoint-warp {e.mean():.4f} "
          f"({base[te].mean()-e.mean():+.4f})", flush=True)

with torch.no_grad():
    learned = mle(warp_ab(PRED, torch.tensor(pa), torch.tensor(pb)), GT).numpy()
imp = base - learned
print(f"\n=== {CNAME}: endpoint-localisation -> affine (OOF) ===")
print(f"baseline {base.mean():.4f} -> learned {learned.mean():.4f}  gain {imp.mean():+.4f}")
print(f"ears improved {(imp>0).sum()}/{N} ({(imp>0).mean()*100:.0f}%)")
t, p_ = stats.ttest_rel(base, learned); print(f"paired t={t:.2f} p={p_:.2e}")
rs = np.random.RandomState(0); us = np.unique(subj); bs = []
for _ in range(4000):
    s = rs.choice(us, len(us), replace=True)
    bs.append(np.mean([imp[subj == k].mean() for k in s]))
lo, hi = np.percentile(bs, [2.5, 97.5])
print(f"subject bootstrap 95%CI [{lo:+.4f},{hi:+.4f}] -> {'EXCLUDES ZERO' if lo>0 else 'includes zero'}")
np.savez(f"scratch/endpoint_pred_{CNAME}.npz", a=pa, b=pb, learned=learned, base=base)
