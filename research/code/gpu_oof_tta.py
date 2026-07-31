"""
OOF predictions through the EXACT final pipeline (safeguard 1).

Each CV fold model predicts ONLY its own held-out subjects, averaged over M fresh
surface samples (fresh-sample TTA). Fold assignment is reproduced identically to
training: subject = ear_index//2, RandomState(12345).permutation, array_split into 5.

Output: oof_tta.npz  pred (340,85,3) world, gt (340,85,3), fold (340,)
Surface projection is applied afterwards on the host (needs mesh faces).
The dense-SSM stage is intentionally EXCLUDED here: that model was fitted on the
fixed-split training ears, so applying it to those same ears out-of-fold would leak.
"""
import os
import numpy as np
import torch

WORK = "/home/ubuntu/ear"
src = open(f"{WORK}/gpu_train.py").read().split("net = Net().to(dev)")[0]
exec(src)                                    # Net, transport, dev, ...

ms = np.load(f"{WORK}/all_multisample.npz")
clouds = torch.tensor(ms["clouds"]).float()        # (340,M,2048,3)
co = torch.tensor(ms["coarse"]).float()
Rv, cv, tv = ms["R"], ms["c0"], ms["true"]
NE, M = clouds.shape[0], clouds.shape[1]

subj = np.arange(NE) // 2
rs = np.random.RandomState(12345)
order_s = rs.permutation(np.unique(subj))
folds = np.array_split(order_s, 5)

pred = np.zeros((NE, 85, 3)); fold_of = -np.ones(NE, int)
for f in range(5):
    wpath = f"{WORK}/gpu_cv_f{f}.npz"
    if not os.path.exists(wpath):
        print(f"  MISSING {wpath}"); continue
    ck = np.load(wpath)
    net = Net().to(dev)
    net.load_state_dict({k: torch.tensor(ck[k]) for k in ck.files
                         if k not in ("best", "K", "T", "SCALE", "seed")})
    net.eval()
    val_s = set(folds[f].tolist())
    idx = [i for i in range(NE) if subj[i] in val_s]
    with torch.no_grad():
        for i in idx:
            pc = clouds[i].to(dev)                         # (M,2048,3)
            q = co[i:i+1].expand(M, -1, -1).to(dev)
            out = net(pc, q)[-1][1].cpu().numpy().astype(np.float64)   # (M,85,3)
            pred[i] = (out.mean(0)) @ Rv[i] + cv[i]
            fold_of[i] = f
    print(f"  fold {f}: {len(idx)} ears predicted with {M} fresh samples", flush=True)

gt = np.stack([tv[i].astype(np.float64) @ Rv[i] + cv[i] for i in range(NE)])
assert (fold_of >= 0).all(), "some ears were never predicted"
np.savez(f"{WORK}/oof_tta.npz", pred=pred, gt=gt, fold=fold_of)
e = np.linalg.norm(pred - gt, axis=2)
print(f"\nOOF ({M}-sample TTA, before surface projection): {e.mean():.4f} mm")
for f in range(5):
    m = fold_of == f
    print(f"  fold {f}: {m.sum():3d} ears  {e[m].mean():.4f}")
print("saved oof_tta.npz")
