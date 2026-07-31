"""
ORTHOGONAL GEOMETRY CORRECTOR (accepts the annotation-phase finding instead of fighting it).

p_corrected = p_base + alpha * b + beta * n          (optionally + gamma * t, |gamma| <= 0.2)

with b = across-contour (in-surface) and n = oriented mesh normal. Because no displacement
along the contour tangent t is permitted, this head CANNOT slide landmarks along their
contour, so it cannot memorise the per-annotation-session phase that defeated the previous
two experiments. It can only improve the anatomical curve geometry -- exactly the 20.2 %
across-contour + 2.1 % normal energy that the valid decomposition identified as the
non-phase part of the error.

Inputs are the FROZEN final-pipeline OOF predictions, so the corrector never sees
unrealistically accurate in-sample base predictions. Trained and evaluated with
subject-grouped folds (the same folds as the base OOF predictions).

Phase-robust objective (phase terms deliberately down-weighted):
   symmetric Chamfer between densely resampled predicted / GT contour polylines
 + robust (Huber) across-contour error
 + robust surface-normal error
 + 0.1 * tangential ordered-landmark term
 + 0.2 * plain ordered-landmark L2

    python scratch/train_ortho.py
"""
import os, sys
import numpy as np
import torch
import torch.nn as nn
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
EPOCHS = int(os.environ.get("EPOCHS", "260"))
ALLOW_T = float(os.environ.get("ALLOW_T", "0.0"))      # 0 = no tangential motion; 0.2 = ablation
LOSS = os.environ.get("LOSS", "proxy")                 # proxy = phase-robust; metric = ordered MLE
CONT = [(0, 24), (25, 54), (55, 74), (75, 84)]
torch.manual_seed(0); np.random.seed(0)

d = np.load("scratch/ortho_feats.npz")
X = torch.tensor(d["feats"]).float()          # (N,85,F) per-landmark patch features
BASE = torch.tensor(d["base"]).double()       # (N,85,3) frozen OOF predictions
GT = torch.tensor(d["gt"]).double()
Tv = torch.tensor(d["t"]).double()            # (N,85,3) contour tangent
Bv = torch.tensor(d["b"]).double()            # across-contour
Nv = torch.tensor(d["n"]).double()            # oriented normal
fold, subj = d["fold"], d["subj"]
N, L, F = X.shape
mu, sd = X.mean((0, 1)), X.std((0, 1)) + 1e-6
print(f"{N} ears, {L} landmarks, {F} features/landmark | ALLOW_T={ALLOW_T}")


def huber(x, delta=0.6):
    a = x.abs()
    return torch.where(a < delta, 0.5 * a ** 2, delta * (a - 0.5 * delta))


def resample_dense(P, m=96):
    """dense points along the contour polyline, for the Chamfer term"""
    s = torch.cat([torch.zeros(P.shape[0], 1, dtype=P.dtype),
                   torch.cumsum((P[:, 1:] - P[:, :-1]).norm(dim=-1), 1)], 1)
    Lt = s[:, -1:]
    q = torch.linspace(0, 1, m, dtype=P.dtype)[None, :] * Lt
    j = torch.clamp(torch.searchsorted(s.contiguous(), q.contiguous()) - 1, 0, P.shape[1] - 2)
    sj = torch.gather(s, 1, j); sj1 = torch.gather(s, 1, j + 1)
    f = ((q - sj) / torch.clamp(sj1 - sj, min=1e-9))[..., None]
    Pj = torch.gather(P, 1, j[..., None].expand(-1, -1, 3))
    Pj1 = torch.gather(P, 1, (j + 1)[..., None].expand(-1, -1, 3))
    return Pj + f * (Pj1 - Pj)


def chamfer(A, B):
    D = torch.cdist(A, B)
    return D.min(-1).values.mean(-1) + D.min(-2).values.mean(-1)


class Ortho(nn.Module):
    """per-landmark MLP on its own patch features + a per-contour 1-D conv for context"""
    def __init__(self, fin=F, ch=48):
        super().__init__()
        self.emb = nn.Embedding(L, 16)
        self.mlp = nn.Sequential(nn.Linear(fin + 16, ch), nn.ReLU(), nn.Linear(ch, ch), nn.ReLU())
        self.ctx = nn.Conv1d(ch, ch, 5, padding=2)
        self.head = nn.Sequential(nn.ReLU(), nn.Dropout(0.1), nn.Linear(ch, 3))

    def forward(self, x):
        B = x.shape[0]
        e = self.emb(torch.arange(L, device=x.device))[None].expand(B, -1, -1)
        h = self.mlp(torch.cat([x, e], -1))                    # (B,L,ch)
        h = h + self.ctx(h.transpose(1, 2)).transpose(1, 2)    # local along-contour context
        o = self.head(h)                                       # (B,L,3)
        alpha = 1.2 * torch.tanh(o[..., 0])                    # across-contour, mm
        beta = 0.6 * torch.tanh(o[..., 1])                     # normal, mm
        gamma = ALLOW_T * torch.tanh(o[..., 2])                # tangential (0 unless ablation)
        return alpha, beta, gamma


def apply_corr(base, a, b_, g):
    return base + a[..., None] * Bv_b + b_[..., None] * Nv_b + g[..., None] * Tv_b


print(f"params: {sum(p.numel() for p in Ortho().parameters()):,}")
base_e = (BASE - GT).norm(dim=-1).mean(-1).numpy()
out = BASE.clone()
for f_ in sorted(set(fold.tolist())):
    te = fold == f_; tr = ~te
    Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
    net = Ortho()
    opt = torch.optim.AdamW(net.parameters(), lr=1.5e-3, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    idx_tr = np.where(tr)[0]
    for ep in range(EPOCHS):
        net.train(); perm = np.random.permutation(len(idx_tr))
        for s0 in range(0, len(idx_tr), 16):
            bi = idx_tr[perm[s0:s0 + 16]]
            Bv_b, Nv_b, Tv_b = Bv[bi], Nv[bi], Tv[bi]
            a, b_, g = net((X[bi] - mu) / sd)
            pc = apply_corr(BASE[bi], a.double(), b_.double(), g.double())
            gt_b = GT[bi]
            err = pc - gt_b
            e_b = (err * Bv_b).sum(-1); e_n = (err * Nv_b).sum(-1); e_t = (err * Tv_b).sum(-1)
            if LOSS == "metric":
                # The orthogonality constraint ALREADY makes phase memorisation
                # impossible (e_t is mathematically unchanged), so the objective can be
                # the real metric -- no need to weaken it with a proxy.
                loss = err.norm(dim=-1).mean()
            else:
                loss = 0.0
                for lo, hi in CONT:                            # phase-robust Chamfer
                    loss = loss + chamfer(resample_dense(pc[:, lo:hi + 1]),
                                          resample_dense(gt_b[:, lo:hi + 1])).mean()
                loss = loss + 3.0 * huber(e_b).mean() + 3.0 * huber(e_n).mean()
                loss = loss + 0.1 * huber(e_t).mean() + 0.2 * err.norm(dim=-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    net.eval()
    with torch.no_grad():
        Bv_b, Nv_b, Tv_b = Bv[te], Nv[te], Tv[te]
        a, b_, g = net(Xte)
        out[te] = apply_corr(BASE[te], a.double(), b_.double(), g.double())
    e = (out[te] - GT[te]).norm(dim=-1).mean(-1).numpy()
    print(f"  fold {f_}: base {base_e[te].mean():.4f} -> ortho {e.mean():.4f} "
          f"({base_e[te].mean()-e.mean():+.4f})", flush=True)

corr_e = (out - GT).norm(dim=-1).mean(-1).numpy()
imp = base_e - corr_e
print(f"\n=== orthogonal geometry corrector (OOF, ALLOW_T={ALLOW_T}) ===")
print(f"base {base_e.mean():.4f} -> corrected {corr_e.mean():.4f}   gain {imp.mean():+.4f}")
print(f"ears improved {(imp>0).sum()}/{N} ({(imp>0).mean()*100:.0f}%)")
folds_pos = sum(1 for f_ in range(5) if imp[fold == f_].mean() > 0)
print(f"folds improved {folds_pos}/5")
t, p = stats.ttest_rel(base_e, corr_e); print(f"paired t={t:.2f} p={p:.2e}")
rs = np.random.RandomState(0); us = np.unique(subj); bs = []
for _ in range(4000):
    s = rs.choice(us, len(us), replace=True)
    bs.append(np.mean([imp[subj == k].mean() for k in s]))
lo, hi = np.percentile(bs, [2.5, 97.5])
print(f"subject bootstrap 95%CI [{lo:+.4f},{hi:+.4f}] -> {'PROMOTE' if lo>0 and folds_pos>=4 else 'REJECT'}")
np.save(f"scratch/ortho_out_{LOSS}.npy", out.numpy())
