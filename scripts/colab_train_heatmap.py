"""
================================================================================
COLAB GPU TRAINING — heatmap / local soft-argmax landmark net for pinna landmarks
================================================================================
Self-contained. Upload `deep_dataset.npz` (from scripts/preprocess_deep.py),
set a GPU runtime, and run:

    python colab_train_heatmap.py            # or paste into a Colab cell

Model: PointNet backbone (per-point + global-context features) + a LOCAL
soft-argmax head — each of the 85 landmarks attends ONLY to the K cloud points
nearest its coarse position, and predicts the expected (surface-constrained)
position. Local attention is the precision fix over global soft-argmax.

Outputs `deep_weights.npz` (numpy arrays) for dependency-free inference back in
the pipeline. Reports val mean-landmark-distance (raw and SSM-projected).

Targets < 1.29mm (classical baseline in this repo: 1.85mm).
"""
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

DATA = "deep_dataset.npz"          # uploaded file
OUT = "deep_weights.npz"
K = 48                              # neighbours per landmark (local window)
EPOCHS = 1500
BS = 24
LR = 1e-3
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev)
torch.manual_seed(0); np.random.seed(0)

# ---------------------------------------------------------------- data
d = np.load(DATA, allow_pickle=True)
clouds = torch.tensor(d["clouds"])           # (N,P,3) canonical mm
coarse = torch.tensor(d["coarse"])           # (N,85,3)
true = torch.tensor(d["true"])               # (N,85,3)
Rmat = d["R"]; c0 = d["c0"]; split = d["split"]
ssm_mean = d["ssm_mean"].astype(np.float64)  # (255,)
ssm_comp = d["ssm_comp"].astype(np.float64)  # (30,255)
NL = 85; P = clouds.shape[1]
tr_idx = np.where(split == "train")[0]; va_idx = np.where(split == "val")[0]
print(f"{len(tr_idx)} train / {len(va_idx)} val ears, {P} pts")

# precompute per-ear local neighbour indices (85,K): nearest cloud pts to coarse.
# rotation/scale-invariant in ordering, so computed once on the un-augmented cloud.
def knn_to_coarse(cl, co):
    dist = torch.cdist(co, cl)                # (85,P)
    return dist.topk(K, largest=False).indices  # (85,K)
NN = torch.stack([knn_to_coarse(clouds[i], coarse[i]) for i in range(len(clouds))])  # (N,85,K)

SCALE = 30.0

# ---------------------------------------------------------------- model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(),
                                 nn.Linear(128, 256), nn.ReLU())
        self.pointfeat = nn.Sequential(nn.Linear(256 + 256, 256), nn.ReLU())
        self.emb = nn.Embedding(NL, 32)
        self.attn = nn.Sequential(nn.Linear(256 + 3 + 32, 128), nn.ReLU(),
                                  nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, pc, nn_idx, co):
        # pc (B,P,3) mm, nn_idx (B,85,K), co (B,85,3)
        B = pc.shape[0]
        h = self.enc(pc / SCALE)                                   # (B,P,256)
        g = h.max(1, keepdim=True).values.expand(-1, P, -1)        # global context
        h = self.pointfeat(torch.cat([h, g], -1))                  # (B,P,256)
        # gather local neighbours per landmark
        idx = nn_idx.reshape(B, NL * K)                            # (B,85K)
        featK = torch.gather(h, 1, idx[..., None].expand(-1, -1, 256)).view(B, NL, K, 256)
        posK = torch.gather(pc, 1, idx[..., None].expand(-1, -1, 3)).view(B, NL, K, 3)
        rel = (posK - co[:, :, None, :]) / SCALE                   # (B,85,K,3)
        e = self.emb(torch.arange(NL, device=pc.device))[None, :, None, :].expand(B, NL, K, 32)
        logit = self.attn(torch.cat([featK, rel, e], -1)).squeeze(-1)  # (B,85,K)
        w = F.softmax(logit, dim=-1)                               # attention over K
        return (w[..., None] * posK).sum(2)                        # (B,85,3) expected pos


# ---------------------------------------------------------------- ssm project (numpy)
def procrustes(src, tgt):
    ms, mt = src.mean(0), tgt.mean(0)
    A, B = src - ms, tgt - mt
    U, _, Vt = np.linalg.svd(A.T @ B)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1; R = U @ Vt
    s = (B * (A @ R)).sum() / (A * A).sum()
    return s, R, ms, mt

def ssm_project(pts):
    s, R, ms, mt = procrustes(pts, ssm_mean.reshape(NL, 3))
    aligned = s * ((pts - ms) @ R) + mt
    coeff = (aligned.flatten() - ssm_mean) @ ssm_comp.T
    recon = (ssm_mean + coeff @ ssm_comp).reshape(NL, 3)
    return ((recon - mt) @ R.T) / s + ms          # inverse transform

def evaluate(net):
    net.eval()
    raw, proj = [], []
    with torch.no_grad():
        for i in va_idx:
            pc = clouds[i:i+1].to(dev); nn_i = NN[i:i+1].to(dev); co = coarse[i:i+1].to(dev)
            pred = net(pc, nn_i, co)[0].cpu().numpy().astype(np.float64)
            R, cc = Rmat[i], c0[i]
            gtw = true[i].numpy().astype(np.float64) @ R + cc
            raw.append(np.linalg.norm(pred @ R + cc - gtw, axis=1).mean())
            proj.append(np.linalg.norm(ssm_project(pred) @ R + cc - gtw, axis=1).mean())
    return np.mean(raw), np.mean(proj)

# ---------------------------------------------------------------- train
net = Net().to(dev)
opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
cl_d, co_d, tr_d, nn_d = clouds.to(dev), coarse.to(dev), true.to(dev), NN.to(dev)

def augment(pc, co, tg):
    B = pc.shape[0]
    th = (torch.rand(B, device=dev) - .5) * 0.7                  # ±20° about z
    ca, sa = th.cos(), th.sin()
    R = torch.zeros(B, 3, 3, device=dev); R[:, 2, 2] = 1
    R[:, 0, 0] = ca; R[:, 0, 1] = -sa; R[:, 1, 0] = sa; R[:, 1, 1] = ca
    s = (1 + (torch.rand(B, 1, 1, device=dev) - .5) * 0.1)
    pc = torch.bmm(pc, R.transpose(1, 2)) * s + torch.randn_like(pc) * 0.3
    co = torch.bmm(co, R.transpose(1, 2)) * s
    tg = torch.bmm(tg, R.transpose(1, 2)) * s
    return pc, co, tg

best = 99
for ep in range(EPOCHS):
    net.train()
    perm = np.random.permutation(tr_idx)
    for b in range(0, len(perm), BS):
        bi = perm[b:b+BS]
        pc, co, tg = augment(cl_d[bi], co_d[bi], tr_d[bi])
        opt.zero_grad()
        pred = net(pc, nn_d[bi], co)
        loss = ((pred - tg) ** 2).sum(-1).mean()
        loss.backward(); opt.step()
    sched.step()
    if (ep + 1) % 50 == 0:
        r, pr = evaluate(net)
        star = "  <-- BEATS 1.29" if pr < 1.29 else ""
        print(f"epoch {ep+1}: raw {r:.3f} | +project {pr:.3f}mm{star}")
        if pr < best:
            best = pr
            sd = {k: v.cpu().numpy() for k, v in net.state_dict().items()}
            np.savez(OUT, **sd, best=best)
print(f"best +project: {best:.3f}mm  ->  {OUT}")
