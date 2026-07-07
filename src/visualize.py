"""
Visualization utilities for ear meshes and landmarks.
Generates matplotlib figures for analysis and presentation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path
from typing import Optional


# ─── Landmark Color Scheme ───────────────────────────────────────────────────

LANDMARK_GROUPS = {
    "Helix (outer rim)": (range(0, 25), "#FF6B6B"),
    "Inner ear contours": (range(25, 55), "#4ECDC4"),
    "Outer boundary": (range(55, 75), "#45B7D1"),
    "Cross-section": (range(75, 85), "#FFA07A"),
}


def get_landmark_colors(n: int = 85) -> np.ndarray:
    """Get color array for landmarks based on anatomical grouping."""
    colors = np.zeros((n, 4))
    for name, (indices, hex_color) in LANDMARK_GROUPS.items():
        r, g, b = int(hex_color[1:3], 16)/255, int(hex_color[3:5], 16)/255, int(hex_color[5:7], 16)/255
        for i in indices:
            if i < n:
                colors[i] = [r, g, b, 1.0]
    return colors


# ─── 3D Plotting ─────────────────────────────────────────────────────────────

def plot_landmarks_3d(
    landmarks: np.ndarray,
    title: str = "Ear Landmarks",
    ax: Optional[plt.Axes] = None,
    show_indices: bool = True,
    figsize: tuple = (12, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot landmarks in 3D with color-coded anatomical groups.
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()
    
    colors = get_landmark_colors(len(landmarks))
    
    # Plot each group
    for name, (indices, hex_color) in LANDMARK_GROUPS.items():
        idx_list = [i for i in indices if i < len(landmarks)]
        pts = landmarks[idx_list]
        ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            c=hex_color, s=50, label=name, alpha=0.9,
            edgecolors="black", linewidth=0.5,
        )
        # Connect sequential landmarks with lines
        if len(idx_list) > 1:
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], c=hex_color, alpha=0.4, linewidth=1)
    
    if show_indices:
        for i, pt in enumerate(landmarks):
            ax.text(pt[0], pt[1], pt[2], str(i), fontsize=5, alpha=0.6)
    
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)
    
    # Equal aspect ratio
    max_range = np.ptp(landmarks, axis=0).max() / 2
    mid = landmarks.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    return fig


def plot_mesh_with_landmarks(
    vertices: np.ndarray,
    faces: np.ndarray,
    landmarks: np.ndarray,
    title: str = "Ear Mesh with Landmarks",
    figsize: tuple = (14, 10),
    max_faces: int = 20000,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot mesh surface with landmarks overlaid.
    Downsamples faces for reasonable rendering.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    
    # Subsample faces if too many
    if len(faces) > max_faces:
        idx = np.random.choice(len(faces), max_faces, replace=False)
        plot_faces = faces[idx]
    else:
        plot_faces = faces
    
    # Create polygon collection
    verts_for_poly = vertices[plot_faces]
    mesh_collection = Poly3DCollection(
        verts_for_poly,
        alpha=0.15,
        facecolor="#E8E8E8",
        edgecolor="#CCCCCC",
        linewidth=0.1,
    )
    ax.add_collection3d(mesh_collection)
    
    # Plot landmarks on top
    for name, (indices, hex_color) in LANDMARK_GROUPS.items():
        idx_list = [i for i in indices if i < len(landmarks)]
        pts = landmarks[idx_list]
        ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            c=hex_color, s=80, label=name, alpha=1.0,
            edgecolors="black", linewidth=0.8, zorder=5,
        )
        if len(idx_list) > 1:
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                    c=hex_color, alpha=0.5, linewidth=1.5, zorder=4)
    
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)
    
    # Set bounds to landmark region with some margin
    margin = 5
    lm_min = landmarks.min(axis=0) - margin
    lm_max = landmarks.max(axis=0) + margin
    ax.set_xlim(lm_min[0], lm_max[0])
    ax.set_ylim(lm_min[1], lm_max[1])
    ax.set_zlim(lm_min[2], lm_max[2])
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    return fig


def plot_prediction_comparison(
    gt_landmarks: np.ndarray,
    pred_landmarks: np.ndarray,
    title: str = "Prediction vs Ground Truth",
    figsize: tuple = (14, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot ground truth and predicted landmarks side by side with error lines."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    
    # Ground truth in green
    ax.scatter(
        gt_landmarks[:, 0], gt_landmarks[:, 1], gt_landmarks[:, 2],
        c="green", s=60, label="Ground Truth", alpha=0.8,
        edgecolors="darkgreen", linewidth=0.5,
    )
    
    # Predictions in red
    ax.scatter(
        pred_landmarks[:, 0], pred_landmarks[:, 1], pred_landmarks[:, 2],
        c="red", s=40, label="Predicted", alpha=0.7,
        marker="^", edgecolors="darkred", linewidth=0.5,
    )
    
    # Error lines connecting GT to predicted
    errors = np.linalg.norm(gt_landmarks - pred_landmarks, axis=1)
    max_err = errors.max()
    for i in range(len(gt_landmarks)):
        color_val = errors[i] / (max_err + 1e-10)
        ax.plot(
            [gt_landmarks[i, 0], pred_landmarks[i, 0]],
            [gt_landmarks[i, 1], pred_landmarks[i, 1]],
            [gt_landmarks[i, 2], pred_landmarks[i, 2]],
            c=cm.hot(color_val), alpha=0.5, linewidth=1,
        )
    
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(f"{title}\nMean error: {errors.mean():.2f} mm, Max: {errors.max():.2f} mm",
                 fontsize=12, fontweight="bold")
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    return fig


def plot_error_analysis(
    errors_per_landmark: dict,
    title: str = "Per-Landmark Error Distribution",
    figsize: tuple = (16, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart of per-landmark mean errors."""
    fig, ax = plt.subplots(figsize=figsize)
    
    indices = sorted(errors_per_landmark.keys())
    means = [np.mean(errors_per_landmark[i]) for i in indices]
    stds = [np.std(errors_per_landmark[i]) for i in indices]
    
    # Color by group
    bar_colors = []
    for i in indices:
        found = False
        for name, (group_idx, hex_color) in LANDMARK_GROUPS.items():
            if i in group_idx:
                bar_colors.append(hex_color)
                found = True
                break
        if not found:
            bar_colors.append("#999999")
    
    ax.bar(indices, means, yerr=stds, color=bar_colors, alpha=0.8,
           edgecolor="black", linewidth=0.3, capsize=2)
    
    ax.set_xlabel("Landmark Index", fontsize=12)
    ax.set_ylabel("Mean Euclidean Error (mm)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axhline(y=np.mean(means), color="red", linestyle="--", alpha=0.5,
               label=f"Overall mean: {np.mean(means):.2f} mm")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    return fig


def plot_ssm_variance(
    explained_variance_ratio: np.ndarray,
    title: str = "SSM Explained Variance",
    figsize: tuple = (10, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot explained variance of SSM components."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    n = len(explained_variance_ratio)
    ax1.bar(range(n), explained_variance_ratio * 100, color="#4ECDC4", 
            edgecolor="black", linewidth=0.3)
    ax1.set_xlabel("Component")
    ax1.set_ylabel("Explained Variance (%)")
    ax1.set_title("Individual Components")
    ax1.grid(axis="y", alpha=0.3)
    
    cumulative = np.cumsum(explained_variance_ratio) * 100
    ax2.plot(range(n), cumulative, "o-", color="#FF6B6B", linewidth=2)
    ax2.axhline(y=95, color="gray", linestyle="--", alpha=0.5, label="95%")
    ax2.axhline(y=99, color="gray", linestyle=":", alpha=0.5, label="99%")
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Variance (%)")
    ax2.set_title("Cumulative Variance")
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    return fig


def plot_multiple_ears(
    landmarks_dict: dict,
    n_show: int = 6,
    figsize: tuple = (18, 12),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot multiple ears in a grid for visual comparison."""
    pids = sorted(landmarks_dict.keys())[:n_show]
    n_cols = 3
    n_rows = (len(pids) + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=figsize)
    
    for i, pid in enumerate(pids):
        ax = fig.add_subplot(n_rows, n_cols, i + 1, projection="3d")
        lm = landmarks_dict[pid]["left"]
        
        for name, (indices, hex_color) in LANDMARK_GROUPS.items():
            idx_list = [j for j in indices if j < len(lm)]
            pts = lm[idx_list]
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                      c=hex_color, s=20, alpha=0.8)
            if len(idx_list) > 1:
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                       c=hex_color, alpha=0.3, linewidth=1)
        
        ax.set_title(f"{pid} (Left)", fontsize=10)
        ax.set_xlabel("X", fontsize=7)
        ax.set_ylabel("Y", fontsize=7)
        ax.set_zlabel("Z", fontsize=7)
        ax.tick_params(labelsize=6)
    
    fig.suptitle("Ear Landmark Variability Across Participants",
                 fontsize=16, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    
    return fig
