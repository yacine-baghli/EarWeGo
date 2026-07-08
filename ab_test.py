"""
Ablation test: evaluate the same v1 weights with different refinement flags.
No retraining needed — refinements are post-prediction.

Configs tested:
  A) baseline         — refine={}  (legacy snap)
  B) +resample        — resample only
  C) +selective_snap  — selective snap only (no legacy snap)
  D) resample+snap    — both on
"""

import json
import numpy as np
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.dataset import Dataset
from src.ear_detector import EarDetector
from src.predictor import LandmarkPredictor
from src.metrics import compute_mean_landmark_distance

# ─── Paths ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "2026 Munich Tech Arena - Datas" / "2026 Munich Tech Arena - Datas"
MESH_DIR = str(DATA_ROOT / "mesh")
LM_DIR = str(DATA_ROOT / "landmarks")
WEIGHTS = ROOT / "models"

# ─── Ablation configs ────────────────────────────────────────────────────────
ABLATIONS = {
    "A_baseline": {},
    "B_resample": {"resample": True, "selective_snap": False, "legacy_snap": True},
    "C_selective_snap": {"resample": False, "selective_snap": True, "legacy_snap": False},
    "D_resample+snap": {"resample": True, "selective_snap": True, "legacy_snap": False},
}

# Per-region index ranges
REGIONS = {
    "Outer_Helix_0-24":    list(range(0, 25)),
    "Concha_25-54":        list(range(25, 55)),
    "Inner_Helix_55-74":   list(range(55, 75)),
    "Sup_Antihelix_75-84": list(range(75, 85)),
}


def per_region_mle(pred, gt, region_indices):
    """Mean landmark error for a subset of indices."""
    return float(np.mean(np.linalg.norm(pred[region_indices] - gt[region_indices], axis=1)))


def run_ablation():
    # Load models once
    print("Loading models...")
    detector = EarDetector()
    detector.load(WEIGHTS / "ear_detector.pkl")
    predictor = LandmarkPredictor()
    predictor.load(WEIGHTS / "landmark_predictor.pkl")

    # Load val dataset
    dataset = Dataset(mesh_dir=MESH_DIR, landmarks_dir=LM_DIR, split="val")
    n = len(dataset)
    print(f"Evaluating on {n} val subjects x 2 ears = {n*2} predictions\n")

    results = {}

    for label, refine in ABLATIONS.items():
        print(f"{'='*60}")
        print(f"  {label}  refine={refine}")
        print(f"{'='*60}")

        subject_dists = []
        region_dists = {r: [] for r in REGIONS}
        t0 = time.time()

        for idx in range(n):
            pid = dataset.get_identifier(idx)
            mesh, gt_left, gt_right = dataset[idx]

            pred_l = predictor.predict(mesh, side="left", ear_detector=detector, refine=refine)
            pred_r = predictor.predict(mesh, side="right", ear_detector=detector, refine=refine)

            d_l = compute_mean_landmark_distance(pred_l, gt_left)
            d_r = compute_mean_landmark_distance(pred_r, gt_right)
            subject_dists.append((d_l + d_r) / 2)

            # Per-region (both ears averaged)
            for rname, ridx in REGIONS.items():
                rl = per_region_mle(pred_l, gt_left, ridx)
                rr = per_region_mle(pred_r, gt_right, ridx)
                region_dists[rname].append((rl + rr) / 2)

        elapsed = time.time() - t0
        md = float(np.mean(subject_dists))
        print(f"  MD = {md:.4f} mm  (median={np.median(subject_dists):.4f}, "
              f"worst={np.max(subject_dists):.4f})  [{elapsed:.1f}s]")
        for rname in REGIONS:
            rmean = float(np.mean(region_dists[rname]))
            print(f"    {rname:25s}: {rmean:.4f} mm")

        results[label] = {
            "MD": md,
            "median": float(np.median(subject_dists)),
            "worst": float(np.max(subject_dists)),
            "regions": {r: float(np.mean(region_dists[r])) for r in REGIONS},
        }

    # ─── Summary table ───────────────────────────────────────────────────
    baseline_md = results["A_baseline"]["MD"]
    print(f"\n{'='*80}")
    print(f"  ABLATION SUMMARY  (baseline MD = {baseline_md:.4f} mm)")
    print(f"{'='*80}")
    header = f"{'Config':<25} {'MD':>8} {'delta':>8} {'Helix':>8} {'Concha':>8} {'Inner':>8} {'Antihlx':>8}"
    print(header)
    print("-" * 80)

    for label, res in results.items():
        delta = res["MD"] - baseline_md
        sign = "+" if delta >= 0 else ""
        rg = res["regions"]
        print(f"{label:<25} {res['MD']:>8.4f} {sign}{delta:>7.4f} "
              f"{rg['Outer_Helix_0-24']:>8.4f} {rg['Concha_25-54']:>8.4f} "
              f"{rg['Inner_Helix_55-74']:>8.4f} {rg['Sup_Antihelix_75-84']:>8.4f}")

    print("=" * 80)

    # Save
    out = ROOT / "output" / "ablation_results.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    run_ablation()
