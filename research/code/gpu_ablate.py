"""
Cheap OOF INFERENCE ablations (no retraining), all on the exact fold models.

  CLOUDS=/path/clouds.npz   which surface sample set (ablation A: sampler)
  GK_OV=12|20|32            EdgeConv graph neighbours at inference   (ablation B)
  K_OV=32|48|64|96          landmark window size                     (ablation B)
  TEMP=0.7|1.0|1.3|1.7      soft-argmax temperature                  (ablation B)

Saves PER-SAMPLE OOF predictions so aggregation rules (ablation C: mean / coordinate
median / geometric median / trimmed mean) can be compared offline without re-running.

Each ear is predicted only by the fold model that never saw its subject.
"""
import os
import numpy as np
import torch

WORK = "/home/ubuntu/ear"
src = open(f"{WORK}/gpu_train.py").read().split("net = Net().to(dev)")[0]
exec(src)                                              # Net, transport, dev, K, GK ...

CLOUDS = os.environ.get("CLOUDS", f"{WORK}/all_multisample.npz")
GK_OV = int(os.environ.get("GK_OV", str(GK)))
K_OV = int(os.environ.get("K_OV", str(K)))
TEMP = float(os.environ.get("TEMP", "1.0"))
TAG = os.environ.get("TAG", "base")

# --- patch inference-time neighbourhood sizes and softmax temperature ---
import types
GK = GK_OV                                             # used inside backbone()
K = K_OV                                               # used inside step()


def patched_step(self, pc, h, q):
    featK, posK = self._gather(pc, h, q, K_OV)
    ctx = torch.cat([featK.mean(2), featK.max(2).values], -1)
    eo = self.embO(torch.arange(NL, device=pc.device))[None].expand(pc.shape[0], -1, -1)
    q1 = q + self.offset(torch.cat([ctx, eo], -1))
    featK2, posK2 = self._gather(pc, h, q1, K_OV)
    rel = (posK2 - q1[:, :, None, :]) / SCALE
    e = self.emb(torch.arange(NL, device=pc.device))[None, :, None, :].expand(
        pc.shape[0], NL, K_OV, 32)
    logit = self.attn(torch.cat([featK2, rel, e], -1)).squeeze(-1) / TEMP
    w = torch.softmax(logit, dim=-1)
    return q1, (w[..., None] * posK2).sum(2)


def patched_backbone(self, pc, ft=None):
    pos = pc / SCALE
    gidx = knn(pos, pos, GK_OV)
    x = pos if ft is None else torch.cat([pos, ft], -1)
    h1 = edge_conv(gidx, x, self.ec1); h2 = edge_conv(gidx, h1, self.ec2)
    h3 = edge_conv(gidx, h2, self.ec3)
    h = self.fuse(torch.cat([h1, h2, h3], -1))
    g = h.max(1, keepdim=True).values.expand(-1, pc.shape[1], -1)
    return self.mix(torch.cat([h, g], -1))


Net.step = patched_step
Net.backbone = patched_backbone

ms = np.load(CLOUDS)
clouds = torch.tensor(ms["clouds"]).float()
co = torch.tensor(ms["coarse"]).float()
Rv, cv, tv = ms["R"], ms["c0"], ms["true"]
NE, M = clouds.shape[0], clouds.shape[1]
subj = np.arange(NE) // 2
rs = np.random.RandomState(12345)
folds = np.array_split(rs.permutation(np.unique(subj)), 5)

pred = np.zeros((NE, M, 85, 3)); fo = -np.ones(NE, int)
for f in range(5):
    ck = np.load(f"{WORK}/gpu_cv_f{f}.npz")
    net = Net().to(dev)
    net.load_state_dict({k: torch.tensor(ck[k]) for k in ck.files
                         if k not in ("best", "K", "T", "SCALE", "seed")})
    net.eval()
    val_s = set(folds[f].tolist())
    idx = [i for i in range(NE) if subj[i] in val_s]
    with torch.no_grad():
        for i in idx:
            out = net(clouds[i].to(dev), co[i:i+1].expand(M, -1, -1).to(dev))[-1][1]
            pred[i] = out.cpu().numpy().astype(np.float64) @ Rv[i] + cv[i]
            fo[i] = f
gt = np.stack([tv[i].astype(np.float64) @ Rv[i] + cv[i] for i in range(NE)])
e = np.linalg.norm(pred.mean(1) - gt, axis=2)
print(f"[{TAG}] clouds={os.path.basename(CLOUDS)} GK={GK_OV} K={K_OV} temp={TEMP} "
      f"M={M} -> OOF mean-agg {e.mean():.4f} mm")
np.savez(f"{WORK}/ab_{TAG}.npz", pred=pred, gt=gt, fold=fo)
