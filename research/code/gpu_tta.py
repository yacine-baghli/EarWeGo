"""
Fresh-surface-sample TTA evaluation (brief v2 §7.2), GPU.

Runs the 4-seed ensemble over M independent surface samples of the SAME ear and
measures the gain as a function of the number of samples (1,2,4,8). Only the surface
sample changes -- framing, pose and coarse init are identical -- so this isolates the
variance our rotation-only TTA could not average.
"""
import os
import numpy as np
import torch

WORK = "/home/ubuntu/ear"
# reuse the exact model definition from the trainer (everything before training starts)
src = open(f"{WORK}/gpu_train.py").read().split("net = Net().to(dev)")[0]
exec(src)                                                    # defines Net, transport, dev...

ms = np.load(f"{WORK}/val_multisample.npz")
clouds_ms = torch.tensor(ms["clouds"]).float()               # (60,M,2048,3)
co_ms = torch.tensor(ms["coarse"]).float()                   # (60,85,3)
Rv, cv, tv = ms["R"], ms["c0"], ms["true"]
NV, M = clouds_ms.shape[0], clouds_ms.shape[1]
print(f"{NV} val ears x {M} fresh surface samples")

SEEDS = [f"{WORK}/gpu_cont_s{s}.npz" for s in range(4)]
SEEDS = [p for p in SEEDS if os.path.exists(p)]
print(f"seeds: {[os.path.basename(p) for p in SEEDS]}")
nets = []
for p in SEEDS:
    ck = np.load(p)
    net = Net().to(dev)
    net.load_state_dict({k: torch.tensor(ck[k]) for k in ck.files
                         if k not in ("best", "K", "T", "SCALE", "seed")})
    net.eval(); nets.append(net)

# preds[seed, ear, sample] -> (85,3) world
preds = np.zeros((len(nets), NV, M, 85, 3))
with torch.no_grad():
    for si, net in enumerate(nets):
        for k in range(NV):
            pc = clouds_ms[k].to(dev)                        # (M,2048,3)
            co = co_ms[k:k+1].expand(M, -1, -1).to(dev)
            out = net(pc, co)[-1][1].cpu().numpy().astype(np.float64)   # (M,85,3)
            preds[si, k] = out @ Rv[k] + cv[k]
        print(f"  seed {si} done", flush=True)

gtw = np.stack([tv[k].astype(np.float64) @ Rv[k] + cv[k] for k in range(NV)])


def mle(P):
    return np.linalg.norm(P - gtw, axis=2).mean()


print(f"\n{'config':34s} {'MLE':>8s}")
print(f"{'1 seed , 1 sample':34s} {mle(preds[0, :, 0]):8.4f}")
print(f"{'4 seeds, 1 sample  (current)':34s} {mle(preds[:, :, 0].mean(0)):8.4f}")
for m in [2, 4, 8]:
    if m <= M:
        print(f"{f'1 seed , {m} samples':34s} {mle(preds[0, :, :m].mean(1)):8.4f}")
for m in [2, 4, 8]:
    if m <= M:
        print(f"{f'4 seeds, {m} samples':34s} {mle(preds[:, :, :m].mean(axis=(0, 2))):8.4f}")
# how much does the prediction move between samples? (the variance being averaged)
disp = np.linalg.norm(preds[:, :, 0] - preds[:, :, 1], axis=3).mean()
print(f"\nmean prediction displacement between two fresh samples: {disp:.4f} mm")
np.savez(f"{WORK}/val_tta_preds.npz", preds=preds, gt=gtw)
print("saved val_tta_preds.npz")
