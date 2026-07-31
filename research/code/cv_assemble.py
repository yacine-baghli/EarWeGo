"""
Assemble out-of-fold (OOF) predictions from the 5 CV folds and report the reliable
estimate: every ear is predicted by a model that never saw its subject.

Also reports per-fold spread (the brief's §4.5: a model good on average but unstable
across folds may generalise worse to the hidden test).
    python3 cv_assemble.py          (run on the instance)
"""
import glob, os
import numpy as np

CONT = [(0, 24), (25, 54), (55, 74), (75, 84)]
NM = ["Helix", "Antihelix", "Concha/outer", "Lobe"]
WORK = os.environ.get("WORK", "/home/ubuntu/ear")

files = sorted(glob.glob(f"{WORK}/gpu_cv_f*_valpred.npz"))
print(f"folds found: {[os.path.basename(f) for f in files]}")
if not files:
    raise SystemExit("no fold predictions yet")

N = 340
oof = np.full((N, 85, 3), np.nan)
gt_all = np.full((N, 85, 3), np.nan)
per_fold = []
for f in files:
    z = np.load(f)
    idx = z["va_idx"]; raw = z["raw"].astype(np.float64); gt = z["gt"].astype(np.float64)
    oof[idx] = raw; gt_all[idx] = gt
    e = np.linalg.norm(raw - gt, axis=2).mean()
    per_fold.append(e)
    print(f"  {os.path.basename(f):26s} {len(idx):3d} ears  MLE {e:.4f} mm")

have = ~np.isnan(oof[:, 0, 0])
E = np.linalg.norm(oof[have] - gt_all[have], axis=2)
print(f"\nOUT-OF-FOLD over {have.sum()} ears (every ear predicted by a model that "
      f"never saw its subject):")
print(f"  MEAN  {E.mean():.4f} mm      median {np.median(E):.4f}")
print(f"  per-fold: mean {np.mean(per_fold):.4f}  sd {np.std(per_fold):.4f}  "
      f"worst {np.max(per_fold):.4f}  best {np.min(per_fold):.4f}")
print(f"  P90 {np.percentile(E,90):.3f}  P95 {np.percentile(E,95):.3f}")
print(f"  SR@2mm {(E<2).mean()*100:.1f}%  SR@3mm {(E<3).mean()*100:.1f}%")
for (lo, hi), nm in zip(CONT, NM):
    print(f"    {nm:14s} {E[:, lo:hi+1].mean():.4f}")
np.savez(f"{WORK}/cv_oof.npz", oof=oof, gt=gt_all, have=have)
print(f"\nsaved {WORK}/cv_oof.npz")
print("NOTE: single fixed-split val was 1.329 (4-seed ensemble) / ~1.40 (1 seed).")
print("      Each CV model trains on 272 ears (vs 280) and is a SINGLE seed, so")
print("      compare against the ~1.40 single-seed number, not 1.329.")
