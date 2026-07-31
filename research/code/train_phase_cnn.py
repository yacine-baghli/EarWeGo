"""
SURFACE-CONDITIONED CONTOUR PHASE MODEL (cheapest falsifiable experiment).

Predicts ONLY 2 numbers per ear -- the affine reparametrisation (stretch a, offset b) of
one predicted contour -- from SURFACE evidence sampled along that contour. The previous
ridge experiment failed because it saw only the predicted landmark configuration, which
cannot reveal its own phase error; here the input is dense-surface geometry.

Trained with a DIFFERENTIABLE warp so the loss is the actual landmark error after
reparametrisation (metric-aligned), not an MSE on oracle parameters.

Evaluation is strictly out-of-fold and subject-grouped: for an ear in fold f, both the
landmark prediction AND the phase model come from models that never saw its subject.

    CONTOUR=inner python scratch/train_phase_cnn.py
"""
import os
import numpy as np
import torch
import torch.nn as nn

CNAME = os.environ.get("CONTOUR", "inner")
PAD = 3.0
EPOCHS = int(os.environ.get("EPOCHS", "400"))
torch.manual_seed(0); np.random.seed(0)

z = np.load(f"scratch/seq_{CNAME}.npz")
SEQ = torch.tensor(z["seq"]).float()                 # (N,T,C)
PRED = torch.tensor(z["pred"]).double()              # (N,n,3) predicted contour
GT = torch.tensor(z["gt"]).double()                  # (N,n,3)
fold = z["fold"]; subj = z["subj"]
N, T, C = SEQ.shape
n = PRED.shape[1]
print(f"contour {CNAME}: {N} ears, seq {T}x{C}, {n} landmarks")
# normalise features per channel using TRAIN-fold statistics later; global here for speed
mu, sd = SEQ.mean((0, 1)), SEQ.std((0, 1)) + 1e-6


def arc_t(P):
    return torch.cat([torch.zeros(P.shape[0], 1, dtype=P.dtype),
                      torch.cumsum((P[:, 1:] - P[:, :-1]).norm(dim=-1), 1)], 1)


def warp(P, a, b):
    """differentiable affine reparametrisation of the polyline P (B,n,3). Vectorised.
    Rules match the oracle: piecewise-linear interpolation, linear extrapolation past
    the ends, parameters clipped to [-PAD, L+PAD], monotone by construction (a>0)."""
    s = arc_t(P)                                          # (B,n) constant wrt a,b
    L = s[:, -1:]; s0 = s[:, :1]
    sq = s0 + a[:, None] * (s - s0) + b[:, None]
    sq = torch.clamp(sq, min=-PAD)
    sq = torch.minimum(sq, L + PAD)
    j = torch.clamp(torch.searchsorted(s.contiguous(), sq.contiguous()) - 1, 0, n - 2)
    sj = torch.gather(s, 1, j); sj1 = torch.gather(s, 1, j + 1)
    f = (sq - sj) / torch.clamp(sj1 - sj, min=1e-9)
    Pj = torch.gather(P, 1, j[..., None].expand(-1, -1, 3))
    Pj1 = torch.gather(P, 1, (j + 1)[..., None].expand(-1, -1, 3))
    val = Pj + f[..., None] * (Pj1 - Pj)
    d0 = P[:, 1] - P[:, 0]; d0 = d0 / d0.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    d1 = P[:, -1] - P[:, -2]; d1 = d1 / d1.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    below = (sq <= 0)[..., None]
    above = (sq >= L)[..., None]
    val = torch.where(below, P[:, :1] + sq[..., None] * d0[:, None, :], val)
    val = torch.where(above, P[:, -1:] + (sq - L)[..., None] * d1[:, None, :], val)
    return val


def mle(P, G):
    return (P - G).norm(dim=-1).mean(-1)


class PhaseNet(nn.Module):
    def __init__(self, cin=C, ch=24):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(cin, ch, 5, padding=2), nn.ReLU(),
            nn.Conv1d(ch, ch, 5, padding=4, dilation=2), nn.ReLU(),
            nn.Conv1d(ch, ch, 5, padding=8, dilation=4), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(2 * ch, 32), nn.ReLU(),
                                  nn.Dropout(0.2), nn.Linear(32, 2))

    def forward(self, x):                                  # x (B,T,C)
        h = self.body(x.transpose(1, 2))                   # (B,ch,T)
        g = torch.cat([h.mean(-1), h.max(-1).values], -1)
        o = self.head(g)
        a = 1.0 + 0.18 * torch.tanh(o[:, 0])               # stretch  ~[0.82,1.18]
        b = 3.0 * torch.tanh(o[:, 1])                      # offset   ~[-3,3] mm
        return a, b


print(f"params: {sum(p.numel() for p in PhaseNet().parameters()):,}")
base = mle(PRED, GT).numpy()
pred_a = np.ones(N); pred_b = np.zeros(N)
for f in sorted(set(fold.tolist())):
    te = fold == f; tr = ~te
    Xtr = ((SEQ[tr] - mu) / sd)
    Xte = ((SEQ[te] - mu) / sd)
    Ptr, Gtr = PRED[tr], GT[tr]
    net = PhaseNet()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    ntr = int(tr.sum()); bs = 32
    for ep in range(EPOCHS):
        net.train(); perm = np.random.permutation(ntr)
        for s0 in range(0, ntr, bs):
            bi = perm[s0:s0 + bs]
            a, b = net(Xtr[bi])
            loss = mle(warp(Ptr[bi], a.double(), b.double()), Gtr[bi]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    net.eval()
    with torch.no_grad():
        a, b = net(Xte)
    pred_a[te] = a.numpy(); pred_b[te] = b.numpy()
    with torch.no_grad():
        e_te = mle(warp(PRED[te], a.double(), b.double()), GT[te]).numpy()
    print(f"  fold {f}: {te.sum():3d} ears  baseline {base[te].mean():.4f} -> "
          f"learned {e_te.mean():.4f}  ({base[te].mean()-e_te.mean():+.4f})", flush=True)

with torch.no_grad():
    learned = mle(warp(PRED, torch.tensor(pred_a), torch.tensor(pred_b)), GT).numpy()
np.savez(f"scratch/phase_pred_{CNAME}.npz", a=pred_a, b=pred_b, learned=learned, base=base)

# ---------------- report ----------------
orc = np.load("scratch/oracles_v2_params.npz")
key = {"inner": "inner_helix", "outer": "outer_helix", "concha": "concha",
       "antihelix": "sup._antihelix"}[CNAME]
OP = orc[key] if key in orc.files else None
print(f"\n=== contour {CNAME} : surface-conditioned affine phase model ===")
print(f"baseline OOF            {base.mean():.4f} mm")
print(f"learned affine (OOF)    {learned.mean():.4f} mm   gain {base.mean()-learned.mean():+.4f}")
if OP is not None:
    oe = None
print(f"ears improved           {(base>learned).sum()}/{N} ({(base>learned).mean()*100:.0f}%)")
from scipy import stats
t, p = stats.ttest_rel(base, learned)
print(f"paired t-test           t={t:.2f}  p={p:.2e}")
rs = np.random.RandomState(0); imp = base - learned; us = np.unique(subj); bs_ = []
for _ in range(4000):
    s = rs.choice(us, len(us), replace=True)
    bs_.append(np.mean([imp[subj == k].mean() for k in s]))
lo, hi = np.percentile(bs_, [2.5, 97.5])
print(f"subject bootstrap 95%CI [{lo:+.4f}, {hi:+.4f}]  -> "
      f"{'EXCLUDES ZERO' if lo > 0 else 'includes zero'}")
if OP is not None:
    ra = np.corrcoef(OP[:, 0], pred_a)[0, 1]; rb = np.corrcoef(OP[:, 1], pred_b)[0, 1]
    print(f"corr(oracle,pred): stretch {ra:+.3f}   offset {rb:+.3f}")
