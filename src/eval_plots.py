"""
Diagnostic visualization suite for the rigorous evaluation framework.
Generates publication-quality figures for each evaluation dimension.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import (
    PinnaLandmarkEvaluator, LANDMARK_GROUPS,
    HRTF_CRITICAL_LANDMARKS, NUM_LANDMARKS,
)

OUTPUT = Path(__file__).resolve().parent.parent / "output"
OUTPUT.mkdir(exist_ok=True)

# Color palette
PALETTE = {
    "primary":   "#2E86AB",
    "secondary": "#F24236",
    "accent1":   "#4ECDC4",
    "accent2":   "#FFA07A",
    "success":   "#27AE60",
    "warning":   "#F39C12",
    "danger":    "#E74C3C",
    "bg":        "#FAFAFA",
    "grid":      "#E0E0E0",
}

GROUP_COLORS = {
    "Helix":             "#FF6B6B",
    "Antihelix/Concha":  "#4ECDC4",
    "Outer boundary":    "#45B7D1",
    "Cross-section":     "#FFA07A",
}


def plot_evaluation_dashboard(evaluator: PinnaLandmarkEvaluator, save_dir: str = None):
    """Generate a comprehensive multi-panel evaluation dashboard."""
    if save_dir is None:
        save_dir = str(OUTPUT)
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    report = evaluator._report or evaluator.generate_report()
    
    # Collect raw data
    all_pred_errors = []
    all_gt = []
    all_pred = []
    sides = []
    
    for entry in evaluator.predictions:
        errors = np.linalg.norm(entry["pred"] - entry["gt"], axis=1)
        all_pred_errors.append(errors)
        all_gt.append(entry["gt"])
        all_pred.append(entry["pred"])
        sides.append(entry["side"])
    
    err_matrix = np.stack(all_pred_errors)  # (N_ears, 85)
    per_ear_means = err_matrix.mean(axis=1)
    
    # ─── Figure 1: Executive Summary Dashboard ────────────────────
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.35)
    fig.suptitle("Pinna Landmark Extraction — Evaluation Dashboard",
                 fontsize=18, fontweight="bold", y=0.98)
    
    # 1a. Error histogram
    ax1 = fig.add_subplot(gs[0, 0:2])
    n_bins = 35
    ax1.hist(per_ear_means, bins=n_bins, color=PALETTE["accent1"],
             edgecolor="white", linewidth=0.5, alpha=0.85, density=True)
    ax1.axvline(per_ear_means.mean(), color=PALETTE["secondary"], linestyle="--",
                linewidth=2, label=f"Mean: {per_ear_means.mean():.2f} mm")
    ax1.axvline(np.median(per_ear_means), color=PALETTE["primary"], linestyle=":",
                linewidth=2, label=f"Median: {np.median(per_ear_means):.2f} mm")
    ax1.set_xlabel("Per-Ear Mean Error (mm)", fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    ax1.set_title("Error Distribution", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)
    
    # 1b. Per-group box plot
    ax2 = fig.add_subplot(gs[0, 2:4])
    group_data = []
    group_labels = []
    group_colors = []
    for gname, indices in LANDMARK_GROUPS.items():
        group_data.append(err_matrix[:, indices].flatten())
        group_labels.append(gname.replace("/", "/\n"))
        group_colors.append(GROUP_COLORS[gname])
    
    bp = ax2.boxplot(group_data, tick_labels=group_labels, patch_artist=True,
                     showfliers=False, widths=0.6,
                     medianprops={"color": "black", "linewidth": 1.5})
    for patch, c in zip(bp["boxes"], group_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.8)
    ax2.set_ylabel("Euclidean Error (mm)", fontsize=11)
    ax2.set_title("Error by Anatomical Region", fontsize=13, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    
    # 1c. Per-landmark bar chart
    ax3 = fig.add_subplot(gs[1, :])
    lm_means = err_matrix.mean(axis=0)
    lm_stds = err_matrix.std(axis=0)
    bar_colors = []
    for i in range(NUM_LANDMARKS):
        found = False
        for gname, indices in LANDMARK_GROUPS.items():
            if i in indices:
                bar_colors.append(GROUP_COLORS[gname])
                found = True
                break
        if not found:
            bar_colors.append("#999")
    
    ax3.bar(range(NUM_LANDMARKS), lm_means, yerr=lm_stds,
            color=bar_colors, alpha=0.85, edgecolor="black", linewidth=0.2,
            capsize=1, error_kw={"linewidth": 0.6, "alpha": 0.6})
    ax3.axhline(lm_means.mean(), color=PALETTE["secondary"], linestyle="--",
                alpha=0.7, linewidth=1.5, label=f"Mean: {lm_means.mean():.2f}mm")
    
    # Highlight HRTF-critical landmarks
    for gname, indices in HRTF_CRITICAL_LANDMARKS.items():
        for idx in indices:
            ax3.axvspan(idx - 0.4, idx + 0.4, alpha=0.08, color="red")
    
    ax3.set_xlabel("Landmark Index", fontsize=11)
    ax3.set_ylabel("Mean Error (mm)", fontsize=11)
    ax3.set_title("Per-Landmark Error (red shading = HRTF-critical)", fontsize=13, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.set_xlim(-1, 85)
    ax3.grid(axis="y", alpha=0.3)
    
    # 1d. Success rate at different thresholds
    ax4 = fig.add_subplot(gs[2, 0])
    thresholds = [1, 2, 3, 4, 5, 7, 10, 15]
    sr = [(err_matrix < t).mean() * 100 for t in thresholds]
    ax4.plot(thresholds, sr, "o-", color=PALETTE["primary"], linewidth=2,
             markersize=6, markerfacecolor=PALETTE["accent1"])
    ax4.set_xlabel("Threshold (mm)", fontsize=11)
    ax4.set_ylabel("Success Rate (%)", fontsize=11)
    ax4.set_title("Cumulative Success Rate", fontsize=13, fontweight="bold")
    ax4.set_ylim(0, 105)
    ax4.grid(alpha=0.3)
    for t, s in zip(thresholds, sr):
        if t in [2, 3, 5]:
            ax4.annotate(f"{s:.0f}%", (t, s), textcoords="offset points",
                        xytext=(5, 8), fontsize=9, fontweight="bold")
    
    # 1e. Left vs Right comparison
    ax5 = fig.add_subplot(gs[2, 1])
    left_mask = [s == "left" for s in sides]
    right_mask = [s == "right" for s in sides]
    left_errs = per_ear_means[left_mask]
    right_errs = per_ear_means[right_mask]
    
    bp2 = ax5.boxplot([left_errs, right_errs],
                      tick_labels=["Left", "Right"],
                      patch_artist=True, showfliers=True,
                      medianprops={"color": "black", "linewidth": 1.5},
                      flierprops={"marker": "o", "markersize": 3, "alpha": 0.4})
    bp2["boxes"][0].set_facecolor("#FF6B6B")
    bp2["boxes"][1].set_facecolor("#45B7D1")
    bp2["boxes"][0].set_alpha(0.7)
    bp2["boxes"][1].set_alpha(0.7)
    ax5.set_ylabel("Mean Error (mm)", fontsize=11)
    ax5.set_title("Left vs Right Ear", fontsize=13, fontweight="bold")
    ax5.grid(axis="y", alpha=0.3)
    
    # 1f. Bias quiver plot (mean error direction per landmark)
    ax6 = fig.add_subplot(gs[2, 2])
    all_signed = np.stack([p["pred"] - p["gt"] for p in evaluator.predictions])
    mean_signed = all_signed.mean(axis=0)  # (85, 3)
    
    # Project to 2D (Y-Z plane = ear frontal view)
    for gname, indices in LANDMARK_GROUPS.items():
        color = GROUP_COLORS[gname]
        for i in indices:
            ax6.arrow(0, 0,
                     mean_signed[i, 1], mean_signed[i, 2],
                     head_width=0.04, head_length=0.02,
                     color=color, alpha=0.4, linewidth=0.8)
    
    ax6.set_xlabel("Y-bias (mm)", fontsize=11)
    ax6.set_ylabel("Z-bias (mm)", fontsize=11)
    ax6.set_title("Directional Bias (Y-Z)", fontsize=13, fontweight="bold")
    ax6.axhline(0, color="gray", linewidth=0.5)
    ax6.axvline(0, color="gray", linewidth=0.5)
    ax6.set_aspect("equal")
    ax6.grid(alpha=0.3)
    
    # 1g. Composite score radar chart
    ax7 = fig.add_subplot(gs[2, 3], polar=True)
    composite = evaluator._compute_composite_score(report)
    categories = list(composite.keys())
    values = list(composite.values())
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values_plot = values + [values[0]]
    angles += [angles[0]]
    
    ax7.fill(angles, values_plot, color=PALETTE["accent1"], alpha=0.25)
    ax7.plot(angles, values_plot, color=PALETTE["primary"], linewidth=2)
    ax7.scatter(angles[:-1], values, color=PALETTE["secondary"], s=50, zorder=5)
    ax7.set_xticks(angles[:-1])
    ax7.set_xticklabels([c.replace(" ", "\n") for c in categories], fontsize=7)
    ax7.set_ylim(0, 100)
    ax7.set_title("Composite\nScore", fontsize=11, fontweight="bold", pad=15)
    
    plt.savefig(save_dir / "evaluation_dashboard.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    print(f"Saved: {save_dir / 'evaluation_dashboard.png'}")
    plt.close(fig)
    
    # ─── Figure 2: Shape Fidelity Deep-Dive ───────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Shape Fidelity Analysis", fontsize=15, fontweight="bold")
    
    # 2a. Procrustes vs Raw error scatter
    ax = axes[0]
    raw_means = [np.linalg.norm(e["pred"] - e["gt"], axis=1).mean() for e in evaluator.predictions]
    proc_means = []
    for e in evaluator.predictions:
        from src.geometry import procrustes_align
        aligned, _ = procrustes_align(e["pred"], e["gt"], allow_scale=True)
        proc_err = np.linalg.norm(aligned - e["gt"], axis=1).mean()
        proc_means.append(proc_err)
    
    ax.scatter(raw_means, proc_means, alpha=0.5, s=20, c=PALETTE["primary"])
    lims = [0, max(max(raw_means), max(proc_means)) * 1.1]
    ax.plot(lims, lims, "--", color="gray", linewidth=1, label="y=x (no improvement)")
    ax.set_xlabel("Raw Mean Error (mm)", fontsize=11)
    ax.set_ylabel("Procrustes-Aligned Error (mm)", fontsize=11)
    ax.set_title("Raw vs Shape Error", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    # 2b. Scale ratio distribution
    ax = axes[1]
    scale_ratios = []
    for e in evaluator.predictions:
        gt, pred = e["gt"], e["pred"]
        gt_s = np.linalg.norm(gt - gt.mean(axis=0))
        pred_s = np.linalg.norm(pred - pred.mean(axis=0))
        scale_ratios.append(pred_s / gt_s)
    
    ax.hist(scale_ratios, bins=25, color=PALETTE["accent2"], edgecolor="white",
            linewidth=0.5, alpha=0.85)
    ax.axvline(1.0, color="red", linestyle="--", linewidth=2, label="Perfect (1.0)")
    ax.set_xlabel("Scale Ratio (Pred/GT)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Scale Accuracy", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    
    # 2c. Centroid error distribution
    ax = axes[2]
    centroid_errs = [np.linalg.norm(e["pred"].mean(0) - e["gt"].mean(0))
                     for e in evaluator.predictions]
    ax.hist(centroid_errs, bins=25, color=PALETTE["accent1"], edgecolor="white",
            linewidth=0.5, alpha=0.85)
    ax.set_xlabel("Centroid Error (mm)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Centroid Accuracy", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(save_dir / "shape_fidelity_analysis.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {save_dir / 'shape_fidelity_analysis.png'}")
    plt.close(fig)
    
    # ─── Figure 3: HRTF Impact Heatmap ───────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("HRTF Acoustic Impact Analysis", fontsize=15, fontweight="bold")
    
    # 3a. HRTF-critical vs non-critical comparison
    critical_idx = []
    for indices in HRTF_CRITICAL_LANDMARKS.values():
        critical_idx.extend(indices)
    non_critical_idx = [i for i in range(NUM_LANDMARKS) if i not in critical_idx]
    
    crit_errs = err_matrix[:, critical_idx].mean(axis=1)
    non_crit_errs = err_matrix[:, non_critical_idx].mean(axis=1)
    
    bp3 = ax1.boxplot([crit_errs, non_crit_errs],
                      tick_labels=["HRTF-Critical\n(31 landmarks)", "Non-Critical\n(54 landmarks)"],
                      patch_artist=True, showfliers=True,
                      medianprops={"color": "black", "linewidth": 1.5},
                      flierprops={"marker": "o", "markersize": 3, "alpha": 0.4})
    bp3["boxes"][0].set_facecolor("#E74C3C")
    bp3["boxes"][1].set_facecolor("#27AE60")
    bp3["boxes"][0].set_alpha(0.7)
    bp3["boxes"][1].set_alpha(0.7)
    ax1.set_ylabel("Mean Error (mm)", fontsize=11)
    ax1.set_title("Critical vs Non-Critical Landmarks", fontsize=13, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)
    
    # 3b. Per HRTF region breakdown
    hrtf_group_means = []
    hrtf_group_names = []
    for gname, indices in HRTF_CRITICAL_LANDMARKS.items():
        hrtf_group_means.append(err_matrix[:, indices].mean())
        hrtf_group_names.append(gname.replace("_", "\n"))
    
    colors_hrtf = ["#E74C3C", "#F39C12", "#2E86AB", "#4ECDC4", "#9B59B6"]
    ax2.barh(hrtf_group_names, hrtf_group_means, color=colors_hrtf[:len(hrtf_group_names)],
             edgecolor="black", linewidth=0.5, alpha=0.85)
    for i, v in enumerate(hrtf_group_means):
        ax2.text(v + 0.05, i, f"{v:.2f}mm", va="center", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Mean Error (mm)", fontsize=11)
    ax2.set_title("Error by HRTF-Critical Region", fontsize=13, fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(save_dir / "hrtf_impact_analysis.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {save_dir / 'hrtf_impact_analysis.png'}")
    plt.close(fig)
    
    print("\nAll diagnostic plots generated!")


if __name__ == "__main__":
    # Quick test — load CSV results if available
    print("Use this module via plot_evaluation_dashboard(evaluator)")
