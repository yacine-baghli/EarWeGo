"""
Measure the real impact of each refinement on your validation split.

Run this on the machine with the dataset AFTER wiring refinement.py into
predictor.predict() (see INTEGRATION.md). It evaluates the SAME fitted model under
several refinement configs and prints MD / worst / P90 / per-region so you can see
exactly which flag helps and by how much — before committing to a v2.

    python ab_test.py --weights runs/<run_id>/weights/predictor.pkl \
                      --split-file data/splits/val_pids.txt

Compares: baseline (legacy snap) vs +clamp vs +resample vs +selective_snap vs all.
Nothing here changes the model; it only toggles post-hoc refinement flags.
"""
from __future__ import annotations
import argparse, pickle, sys
from pathlib import Path
import numpy as np

# import the repo's modules
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.data_loader import load_mesh, load_landmarks, MESH_DIR, LANDMARK_DIR
try:
    from src.ear_detector import EarDetector
except Exception:
    EarDetector = None

CONTOURS = {"outer_helix": (0, 25), "concha": (25, 55),
            "inner_helix": (55, 75), "sup_antihelix": (75, 85)}

CONFIGS = {
    "baseline (legacy snap)": {"legacy_snap": True},
    "+clamp":                 {"clamp": True, "legacy_snap": True},
    "+resample":              {"resample": True, "legacy_snap": True},
    "+selective_snap":        {"selective_snap": True},
    "ALL (clamp+resample+sel)": {"clamp": True, "resample": True, "selective_snap": True},
}


def per_ear(pred, gt):
    return np.linalg.norm(pred - gt, axis=1)


def evaluate(predictor, pairs, refine):
    """pairs: list of (mesh, side, gt(85,3)). Returns dict of aggregate metrics."""
    per_ear_md, region = [], {k: [] for k in CONTOURS}
    detector = EarDetector() if EarDetector is not None else None
    for mesh, side, gt in pairs:
        pred = predictor.predict(mesh, side=side, ear_detector=detector, refine=refine)
        d = per_ear(pred, gt)
        per_ear_md.append(d.mean())
        for name, (lo, hi) in CONTOURS.items():
            region[name].append(d[lo:hi].mean())
    a = np.array(per_ear_md)
    out = {"MD": a.mean(), "median": np.median(a), "worst": a.max(),
           "P90": np.percentile(a, 90), "std": a.std()}
    for name in CONTOURS:
        out[name] = float(np.mean(region[name]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split-file", required=True, help="val_pids.txt")
    ap.add_argument("--mesh-dir", default=str(MESH_DIR))
    ap.add_argument("--landmarks-dir", default=str(LANDMARK_DIR))
    a = ap.parse_args()

    with open(a.weights, "rb") as f:
        predictor = pickle.load(f)

    pids = [p.strip() for p in Path(a.split_file).read_text().split() if p.strip()]
    md, ld = Path(a.mesh_dir), Path(a.landmarks_dir)
    pairs = []
    for pid in pids:
        mesh = load_mesh(md / f"{pid}.ply")
        for side in ("left", "right"):
            gt = load_landmarks(ld / f"{pid}_{side}_ear_landmarks.csv")
            pairs.append((mesh, side, gt))
    print(f"Evaluating {len(pids)} subjects ({len(pairs)} ears)\n")

    rows = {name: evaluate(predictor, pairs, refine) for name, refine in CONFIGS.items()}

    base = rows["baseline (legacy snap)"]["MD"]
    cols = ["MD", "median", "worst", "P90", "inner_helix", "concha", "outer_helix"]
    hdr = f'{"config":26s} ' + " ".join(f"{c:>11s}" for c in cols) + "   ΔMD"
    print(hdr); print("-" * len(hdr))
    for name, m in rows.items():
        delta = m["MD"] - base
        line = f'{name:26s} ' + " ".join(f"{m[c]:11.3f}" for c in cols)
        print(line + f"   {delta:+.3f}")
    print("\nΔMD < 0 means the refinement helped. Pick the winning combination as v2.")


if __name__ == "__main__":
    main()
