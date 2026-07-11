"""
Build a COMPACT dataset for GPU training on Colab (no 1.6GB mesh upload).

Per ear: the ear point cloud in the canonical frame (downsampled to NPTS),
the coarse landmark estimate (v1 baseline), and the ground-truth landmarks,
plus the frame (R, c0) to map back to world and the SSM (mean+components) for
the projection/denoising step. Saved as a single .npz.

Run locally:  python scripts/preprocess_deep.py
Output:       scratch/deep_dataset.npz   (upload this to Colab)
"""
import sys, pickle
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_loader import load_all_landmarks, NUM_LANDMARKS
from src.geometry import StatisticalShapeModel, procrustes_align
from src.splits import get_split
from src.dataset import Dataset

MESH = "2026 Munich Tech Arena - Datas/2026 Munich Tech Arena - Datas/mesh"
LM = "2026 Munich Tech Arena - Datas/2026 Munich Tech Arena - Datas/landmarks"
CACHE = Path(r"C:\Users\Yacine\AppData\Local\Temp\claude\C--Yacine-doc-Polytech-Python-Huawei-tech-arena\178eec39-410e-4f2a-b8b6-f531c1415898\scratchpad\baseline_preds.pkl")
OUT = Path(__file__).resolve().parent.parent / "scratch" / "deep_dataset.npz"
NPTS = 2048
rng = np.random.RandomState(0)


def main():
    allm = load_all_landmarks()
    ds = Dataset(MESH, LM); pid2idx = {p: i for i, p in enumerate(ds.subject_ids)}
    tr = get_split("train", mesh_dir=Path(MESH)); va = get_split("val", mesh_dir=Path(MESH))
    L = np.stack([allm[p]["left"] for p in tr]); Rr = np.stack([allm[p]["right"] for p in tr])
    Rm = Rr.copy(); Rm[:, :, 1] *= -1
    ssm = StatisticalShapeModel(30); ssm.fit(np.concatenate([L, Rm]))
    mean = ssm.get_mean_shape()
    preds = pickle.load(open(CACHE, "rb"))

    clouds, coarses, trues, Rs, c0s, splits = [], [], [], [], [], []
    for split, pids in (("train", tr), ("val", va)):
        for p in pids:
            mv = np.asarray(ds[pid2idx[p]][0].vertices)
            pl, pr = preds[p]
            for verts, coarse, gt in ((mv, pl, allm[p]["left"]),
                                      (mv * [1, -1, 1], pr * [1, -1, 1], allm[p]["right"] * [1, -1, 1])):
                tf = procrustes_align(mean, coarse, allow_scale=True)[1]
                R, c0 = tf["R"], tf["t_tgt"]
                lo, hi = coarse.min(0) - 14, coarse.max(0) + 14
                m = np.all((verts >= lo) & (verts <= hi), axis=1)
                cl = (verts[m] - c0) @ R.T
                idx = rng.randint(0, len(cl), NPTS)
                clouds.append(cl[idx].astype(np.float32))
                coarses.append(((coarse - c0) @ R.T).astype(np.float32))
                trues.append(((gt - c0) @ R.T).astype(np.float32))
                Rs.append(R.astype(np.float32)); c0s.append(c0.astype(np.float32))
                splits.append(split)

    np.savez_compressed(
        OUT,
        clouds=np.array(clouds), coarse=np.array(coarses), true=np.array(trues),
        R=np.array(Rs), c0=np.array(c0s), split=np.array(splits),
        ssm_mean=ssm.mean_shape.astype(np.float32),
        ssm_comp=ssm.components.astype(np.float32),
        contours=np.array([[0, 24], [25, 54], [55, 74], [75, 84]]),
    )
    n = len(clouds)
    mb = OUT.stat().st_size / 1e6
    print(f"Saved {OUT} : {n} ears ({splits.count('train')} train / {splits.count('val')} val), "
          f"{NPTS} pts/ear, {mb:.1f} MB")
    print("Upload this file to Colab and run colab_train_heatmap.py")


if __name__ == "__main__":
    main()
