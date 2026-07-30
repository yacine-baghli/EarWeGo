"""
Reproduce the deep contour-ensemble metrics + results figure (validation, 60 ears).

    python -m deep_model.evaluate_deep

Reads the committed `val_errors.npz` (per-landmark error DISTANCES of the 4-seed
ensemble + left/right side — no landmark coordinates, so no challenge data is
published), writes results/metrics.json, and renders results/deep_results.png.
Needs no PyTorch and no raw data.
"""
import json
from pathlib import Path
import numpy as np

from src.evaluation import LANDMARK_GROUPS, HRTF_CRITICAL_LANDMARKS

HERE = Path(__file__).resolve().parent
PROGRESSION = [("classical (test)", 2.65), ("classical (val)", 1.85),
               ("deep DGCNN", 1.50), ("+contour head", 1.375),
               ("+4-seed ensemble", 1.329), ("+surface proj", 1.309)]
SCALING = [(1, 1.375), (2, 1.350), (4, 1.329), (6, 1.330)]
ASYMPTOTE = 1.33          # MEASURED ensemble saturation (4-seed 1.329 == 6-seed 1.330)
TARGET = 0.5              # organizers' target: <0.5mm "good", <0.2mm "very good"


def _bootstrap_ci(per_ear, n=1000, seed=42):
    rng = np.random.RandomState(seed)
    means = [rng.choice(per_ear, len(per_ear), replace=True).mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compute_metrics(dists):
    per_ear = dists.mean(1)                                   # (N,)
    crit = sorted({i for idx in HRTF_CRITICAL_LANDMARKS.values() for i in idx})
    lo, hi = _bootstrap_ci(per_ear)
    out = {
        "n_ears": int(dists.shape[0]),
        "MLE_mm": round(float(dists.mean()), 3),
        "median_mm": round(float(np.median(dists, axis=1).mean()), 3),
        "RMSE_mm": round(float(np.sqrt((dists ** 2).mean(1)).mean()), 3),
        "worst_ear_mm": round(float(per_ear.max()), 3),
        "best_ear_mm": round(float(per_ear.min()), 3),
        "CI95_mm": [round(lo, 3), round(hi, 3)],
        "SR@2mm": round(float((dists < 2).mean()) * 100, 1),
        "SR@3mm": round(float((dists < 3).mean()) * 100, 1),
        "SR@5mm": round(float((dists < 5).mean()) * 100, 1),
        "HRTF_SR@2mm": round(float((dists[:, crit] < 2).mean()) * 100, 1),
        "HRTF_critical_MLE_mm": round(float(dists[:, crit].mean()), 3),
        "per_region_MLE_mm": {g: round(float(dists[:, idx].mean()), 3)
                              for g, idx in LANDMARK_GROUPS.items()},
    }
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def make_figure(dists):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    per_lm = dists.ravel()
    CONT = [("Helix", 0, 24), ("Antihelix/Concha", 25, 54),
            ("Outer boundary", 55, 74), ("Cross-section", 75, 84)]
    MUTED, GRID, BLUE = "#9aa0a6", "#e6e8eb", "#0072B2"
    OKABE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.2))
    fig.suptitle("Pinna landmark model — deep contour ensemble (validation, 60 ears)",
                 fontsize=13, fontweight="bold", y=0.98)

    def strip(a):
        a.grid(color=GRID, lw=0.8); a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)

    a = ax[0, 0]
    vals = [v for _, v in PROGRESSION]
    cols = [MUTED, MUTED, "#b9c0c7", "#7fb3d5", BLUE]
    a.bar(range(len(vals)), vals, color=cols[:len(vals)], width=0.66, zorder=3)
    a.axhline(TARGET, color="#D55E00", lw=1.4, ls=(0, (4, 3)), zorder=2)
    a.text(len(vals) - 0.5, TARGET + 0.06, "0.5mm target", color="#D55E00",
           va="bottom", ha="right", fontsize=8.5)
    for i, v in enumerate(vals):
        a.text(i, v + 0.05, f"{v:.2f}", ha="center", va="bottom",
               fontweight="bold" if i == len(vals) - 1 else "normal", fontsize=9)
    short = {"classical (test)": "classical\n(test)", "classical (val)": "classical\n(val)",
             "deep DGCNN": "deep\nDGCNN", "+contour head": "+contour\nhead",
             "+4-seed ensemble": "+4-seed\nensemble", "+surface proj": "+surface\nproj"}
    a.set_xticks(range(len(vals)))
    a.set_xticklabels([short.get(n, n) for n, _ in PROGRESSION], fontsize=7.6)
    a.set_ylabel("mean landmark error (mm)"); a.set_ylim(0, 2.9)
    a.set_title("A · Accuracy progression", loc="left", fontweight="bold", fontsize=10.5); strip(a)

    a = ax[0, 1]
    rvals = [dists[:, lo:hi + 1].mean() for _, lo, hi in CONT]
    a.barh(range(4), rvals, color=OKABE, height=0.62, zorder=3)
    for i, v in enumerate(rvals):
        a.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=9)
    a.set_yticks(range(4)); a.set_yticklabels([c[0] for c in CONT], fontsize=9)
    a.invert_yaxis(); a.set_xlabel("mean landmark error (mm)"); a.set_xlim(0, 2.0)
    a.set_title("B · Error by anatomical region", loc="left", fontweight="bold", fontsize=10.5); strip(a)

    a = ax[1, 0]
    xs = np.linspace(0, 6, 200)
    a.plot(xs, [(per_lm < t).mean() * 100 for t in xs], color=BLUE, lw=2, zorder=3)
    for thr in (2, 3, 5):
        sr = (per_lm < thr).mean() * 100
        a.plot([thr, thr], [0, sr], color=MUTED, lw=1, ls=":", zorder=2)
        a.scatter([thr], [sr], color=BLUE, s=28, zorder=4)
        a.text(thr, sr - 6, f"SR@{thr}mm\n{sr:.0f}%", ha="center", va="top", fontsize=8.3)
    a.set_xlabel("error threshold (mm)"); a.set_ylabel("% landmarks within threshold")
    a.set_xlim(0, 6); a.set_ylim(0, 102)
    a.set_title("C · Success-rate curve (all 5,100 landmarks)", loc="left", fontweight="bold", fontsize=10.5); strip(a)

    a = ax[1, 1]
    ns = [n for n, _ in SCALING]; es = [e for _, e in SCALING]
    a.plot(ns, es, "-o", color=BLUE, lw=2, markersize=7, zorder=3)
    for n, e in zip(ns, es):
        a.text(n, e + 0.006, f"{e:.3f}", ha="center", va="bottom", fontsize=9)
    a.axhline(ASYMPTOTE, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=2)
    a.text(6, ASYMPTOTE + 0.003, f"saturates ≈{ASYMPTOTE} (measured)", ha="right",
           va="bottom", color=MUTED, fontsize=8.3)
    a.set_xticks(ns); a.set_xlabel("ensemble size (seeds)"); a.set_ylabel("mean landmark error (mm)")
    a.set_ylim(1.30, 1.39)
    a.set_title("D · Ensemble saturates (more seeds ≠ better)", loc="left",
                fontweight="bold", fontsize=10.5); strip(a)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(HERE / "results" / "deep_results.png", bbox_inches="tight", facecolor="white")
    print("wrote results/deep_results.png")


if __name__ == "__main__":
    z = np.load(HERE / "val_errors.npz", allow_pickle=True)
    dists = z["dists"].astype(np.float64)
    compute_metrics(dists)
    make_figure(dists)
