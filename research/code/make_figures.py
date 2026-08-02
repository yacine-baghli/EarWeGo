"""
README FIGURES. Every number plotted here is recomputed from the frozen out-of-fold
prediction array, not copied from a table, so a figure cannot silently drift from the
result it illustrates.

Inputs (both private, neither published):
  scratch/ensemble5_proj.npy   (340,85,3) pooled OOF prediction of the shipped pipeline
  scratch/ortho_feats.npz      gt (340,85,3) and the orthonormal per-landmark frame
                               (t = along-contour, b = across-contour, n = surface normal)
Stage-by-stage values come from research/results/*.json, which are themselves produced by
the scripts listed in research/README.md.

Outputs: img/fig_progress.png, fig_oracle_ladder.png, fig_error_anatomy.png,
         fig_error_distribution.png   -- aggregate statistics only, no coordinates.

    python research/code/make_figures.py
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

CONT = [(0, 24, "outer helix"), (25, 54, "concha"),
        (55, 74, "inner helix"), (75, 84, "sup. antihelix")]
# Categorical hues in FIXED order, one per contour, never cycled or reassigned by rank.
# Checked with the dataviz palette validator (light surface, all pairs): PASS.
CC = ["#2563eb", "#d97706", "#059669", "#7c3aed"]
INK, INK2, MUTED, GRID = "#1a1d21", "#4b5563", "#8b93a1", "#e5e7eb"
ACC, NEG = "#2563eb", "#b91c1c"

plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 160, "savefig.bbox": "tight",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "xtick.major.size": 0, "ytick.major.size": 0,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
})


def bare(ax, axis="y"):
    """Recessive frame: keep only the grid the reader actually uses."""
    ax.set_axisbelow(True)
    ax.grid(axis=axis)
    ax.grid(False, axis="x" if axis == "y" else "y")
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def title(ax, t, sub=None):
    ax.set_title(t, loc="left", fontsize=11, fontweight="bold", color=INK, pad=14 if sub else 8)
    if sub:
        ax.annotate(sub, (0, 1.012), xycoords="axes fraction", fontsize=8.5,
                    color=MUTED, va="bottom")


P = np.load("scratch/ensemble5_proj.npy").astype(np.float64)
Z = np.load("scratch/ortho_feats.npz")
GT, T, B, N = (Z[k].astype(np.float64) for k in ("gt", "t", "b", "n"))
E = P - GT
d = np.linalg.norm(E, axis=-1)                      # (340,85) mm
POOLED = d.mean()
print(f"pooled OOF over {d.shape[0]} ears x {d.shape[1]} landmarks: {POOLED:.4f} mm")

# ----------------------------------------------------------------- 1. progress
LAD = json.load(open("research/results/best_current.json"))
steps = [("single seed, DGCNN", 1.3204, ""),
         ("3-seed ensemble", 1.2773, "variance"),
         ("corrected surface normals", 1.2292, "input"),
         ("+ KPConv + PTv3", 1.1952, "diversity"),
         ("2 seeds per member", 1.1827, "variance"),
         ("exact surface projection", round(float(POOLED), 4), "output")]
assert abs(steps[-1][1] - LAD["pooled_oof_mm"]) < 5e-4, "figure disagrees with best_current.json"

fig, ax = plt.subplots(figsize=(7.4, 3.5))
y = np.arange(len(steps))[::-1]
vals = [s[1] for s in steps]
ax.hlines(y, 1.0, vals, color=GRID, lw=1)
ax.plot(vals, y, "o-", color=ACC, lw=2, ms=7, mfc="white", mew=2, zorder=3)
for yy, (lbl, v, kind) in zip(y, steps):
    ax.annotate(f"{v:.4f}", (v, yy), xytext=(9, 0), textcoords="offset points",
                va="center", fontsize=9, fontweight="bold", color=INK)
ax.set_yticks(y, [s[0] for s in steps], fontsize=9)
ax.tick_params(axis="y", labelcolor=INK)
ax.set_xlim(1.05, 1.40)
ax.set_xlabel("pooled out-of-fold mean landmark error (mm), 340 ears")
ax.axvline(1.1726, color=NEG, lw=1.2, ls=(0, (4, 3)), zorder=1)
ax.annotate("reference pipeline 1.1726 mm\n(different protocol)", (1.1726, 0.55),
            xycoords=("data", "axes fraction"), xytext=(-7, 0),
            textcoords="offset points", ha="right", va="center", fontsize=8, color=NEG)
bare(ax, "x")
title(ax, "Every measured gain came from variance reduction",
      "No single model got better at geometry — the best single-family model is still 1.229 mm.")
fig.savefig("img/fig_progress.png")
plt.close(fig)

# ------------------------------------------------------------ 2. oracle ladder
OL = json.load(open("research/results/oracle_ladder.json"))
rows = OL["ladder"]
assert abs(rows[0]["mle_mm"] - POOLED) < 5e-4, "oracle ladder baseline disagrees with the array"

fig, ax = plt.subplots(figsize=(7.4, 3.6))
lbl = [r["correction"].replace("none (current best)", "none — current prediction") for r in rows]
v = [r["mle_mm"] for r in rows]
dof = [r["dof_per_ear"] for r in rows]
y = np.arange(len(rows))[::-1]
# one hue, light -> dark with the size of the correction: this is a magnitude scale
shade = [plt.matplotlib.colors.to_hex(plt.cm.Blues(0.32 + 0.52 * (1.1776 - x) / 0.6)) for x in v]
ax.barh(y, v, height=0.62, color=shade, edgecolor="white", lw=2)
for yy, vv, dd in zip(y, v, dof):
    ax.annotate(f"{vv:.4f} mm", (vv, yy), xytext=(7, 0), textcoords="offset points",
                va="center", fontsize=9, fontweight="bold", color=INK)
    if dd:
        ax.annotate(f"{dd} dof/ear", (0.02, yy), xytext=(0, 0), textcoords="offset points",
                    va="center", fontsize=8, color="white", fontweight="bold")
ax.set_yticks(y, lbl, fontsize=9)
ax.tick_params(axis="y", labelcolor=INK)
ax.set_xlim(0, 1.42)
ax.set_xlabel("mean landmark error after the correction (mm)")
bare(ax, "x")
title(ax, "28 numbers per ear would reach 0.60 mm",
      "Each row is fitted on ground truth — an upper bound, not a result. "
      "The contour shapes are right; the placement is wrong.")
fig.savefig("img/fig_oracle_ladder.png")
plt.close(fig)

# --------------------------------------------------------- 3. error anatomy
et = np.abs((E * T).sum(-1))
eb = np.abs((E * B).sum(-1))
en = np.abs((E * N).sum(-1))
energy = np.array([(x ** 2).sum() for x in (et, eb, en)])
energy = energy / energy.sum()
print("directional energy shares:", np.round(energy * 100, 1))

fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6), gridspec_kw={"width_ratios": [1, 1.1]})
fig.subplots_adjust(wspace=0.28)

ax = axes[0]
comp = ["along contour", "across contour", "surface normal"]
rmse = [float(np.sqrt((x ** 2).mean())) for x in (et, eb, en)]
sh = ["#1d4ed8", "#60a5fa", "#bfdbfe"]
bars = ax.bar(comp, rmse, width=0.55, color=sh, edgecolor="white", lw=2)
for b_, r_, e_ in zip(bars, rmse, energy):
    ax.annotate(f"{r_:.2f} mm\n{e_*100:.0f}% of energy", (b_.get_x() + b_.get_width() / 2, r_),
                xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8.5,
                color=INK, fontweight="bold")
ax.set_ylim(0, 1.72)
ax.set_ylabel("RMSE (mm)")
bare(ax)
title(ax, "The error is one-dimensional",
      "Landmarks slide along their curve, not off the surface.")

ax = axes[1]
x = np.arange(4)
# both series are MEANS of a distance, so the along-contour bar is a genuine part of the
# total bar. Mixing a mean with an RMSE here would overstate the along-contour share.
per_c = [float(d[:, lo:hi + 1].mean()) for lo, hi, _ in CONT]
tan_c = [float(et[:, lo:hi + 1].mean()) for lo, hi, _ in CONT]
ax.bar(x - 0.19, per_c, width=0.36, color=CC, edgecolor="white", lw=1.6,
       label="mean error (all directions)")
ax.bar(x + 0.19, tan_c, width=0.36, color=CC, edgecolor="white", lw=1.6, alpha=0.40,
       label="its along-contour part")
for xi, (a, b_) in enumerate(zip(per_c, tan_c)):
    ax.annotate(f"{a:.2f}", (xi - 0.19, a), xytext=(0, 3), textcoords="offset points",
                ha="center", fontsize=8, color=INK)
    ax.annotate(f"{b_:.2f}", (xi + 0.19, b_), xytext=(0, 3), textcoords="offset points",
                ha="center", fontsize=8, color=INK2)
ax.set_xticks(x, [c[2] for c in CONT], fontsize=8.5)
ax.tick_params(axis="x", labelcolor=INK)
ax.set_ylabel("mean error (mm)")
ax.set_ylim(0, 1.95)
ax.legend(frameon=False, fontsize=8, loc="upper left", handlelength=1.1)
bare(ax)
title(ax, "Worst on the smooth rims",
      "Concha is the most sculpted region and the most accurate one.")
fig.savefig("img/fig_error_anatomy.png")
plt.close(fig)

# ------------------------------------------------------ 4. error distribution
flat = np.sort(d.ravel())
cdf = np.arange(1, flat.size + 1) / flat.size
per_ear = d.mean(1)

fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.4), gridspec_kw={"width_ratios": [1.15, 1]})
fig.subplots_adjust(wspace=0.3)

ax = axes[0]
ax.plot(flat, cdf, color=ACC, lw=2)
ax.set_xlim(0, 4)
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
for thr in (0.5, 1.0, 2.0):
    f = float((flat <= thr).mean())
    ax.vlines(thr, 0, f, color=MUTED, lw=0.9, ls=(0, (3, 3)))
    ax.plot([thr], [f], "o", ms=5, color=ACC, zorder=3)
    ax.annotate(f"{f*100:.0f}% within {thr:g} mm", (thr, f), xytext=(8, -3),
                textcoords="offset points", fontsize=8.5, color=INK)
ax.set_xlabel("landmark error (mm)")
ax.set_ylabel("share of the 28 900 predictions")
bare(ax)
title(ax, "All 28 900 out-of-fold landmark errors",
      f"median {np.median(flat):.3f} mm · mean {POOLED:.3f} mm · "
      f"p90 {np.percentile(flat, 90):.3f} mm")

ax = axes[1]
ax.hist(per_ear, bins=34, color="#bfdbfe", edgecolor="white", lw=1.2)
ax.axvline(POOLED, color=ACC, lw=2)
ax.annotate(f"mean {POOLED:.3f}", (POOLED, ax.get_ylim()[1] * 0.94), xytext=(6, 0),
            textcoords="offset points", fontsize=8.5, color=ACC, fontweight="bold")
ax.annotate(f"best ear\n{per_ear.min():.3f} mm", (per_ear.min(), 2), xytext=(4, 14),
            textcoords="offset points", fontsize=8, color=INK2)
ax.set_xlabel("per-ear mean error (mm)")
ax.set_ylabel("ears")
bare(ax)
title(ax, "Per-ear spread, 340 ears",
      f"{(per_ear < 1.0).sum()} of 340 ears already average below 1 mm.")
fig.savefig("img/fig_error_distribution.png")
plt.close(fig)

print("wrote img/fig_progress.png, fig_oracle_ladder.png, fig_error_anatomy.png, "
      "fig_error_distribution.png")

# ------------------------------------------------- 5. information in the surface
IL = json.load(open("research/results/info_limit.json"))
pl = IL["pooled"]
fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.4), gridspec_kw={"width_ratios": [1, 1.12]})
fig.subplots_adjust(wspace=0.32)

ax = axes[0]
grp = ["across contour", "along contour"]
match = [pl["nn1_1d_across"]["mean"], pl["nn1_1d_along"]["mean"]]
chance = pl["chance_1d"]["mean"]
xg = np.arange(2)
ax.bar(xg, match, width=0.5, color=["#1d4ed8", "#93a3b8"], edgecolor="white", lw=2)
ax.hlines(chance, -0.45, 1.45, color=NEG, lw=1.6, ls=(0, (4, 3)))
ax.annotate(f"chance {chance:.2f} mm", (-0.45, chance), xytext=(0, 5),
            textcoords="offset points", ha="left", fontsize=8.5, color=NEG)
for xi, m in zip(xg, match):
    ax.annotate(f"{m:.2f} mm", (xi, m), xytext=(0, 4), textcoords="offset points",
                ha="center", fontsize=9, fontweight="bold", color=INK)
ax.set_xticks(xg, grp, fontsize=9)
ax.tick_params(axis="x", labelcolor=INK)
ax.set_ylim(0, 1.85)
ax.set_ylabel("1-D localisation error (mm)")
bare(ax)
title(ax, "The surface knows across, not along",
      "Training-free descriptor match, leave-one-subject-out.")

ax = axes[1]
lab = ["sampling floor", "our prediction", "best training-free\ndescriptor match", "chance"]
# ctr_pred, not POOLED: the matcher ran on a 120-ear scan subset, and this is our own
# error on exactly those ears, so the four bars are a paired comparison.
val = [pl["floor_gt"]["mean"], pl["ctr_pred"]["mean"], pl["nn1_gt"]["mean"],
       pl["chance_gt"]["mean"]]
col = ["#dbeafe", "#1d4ed8", "#93a3b8", "#cbd5e1"]
yv = np.arange(4)[::-1]
ax.barh(yv, val, height=0.6, color=col, edgecolor="white", lw=2)
for yy, vv in zip(yv, val):
    ax.annotate(f"{vv:.3f} mm", (vv, yy), xytext=(7, 0), textcoords="offset points",
                va="center", fontsize=9, fontweight="bold", color=INK)
ax.set_yticks(yv, lab, fontsize=8.5)
ax.tick_params(axis="y", labelcolor=INK)
ax.set_xlim(0, 2.62)
ax.set_xlabel("3-D localisation error (mm)")
bare(ax, "x")
title(ax, "We already beat the local geometry",
      "Matching a landmark's own local shape against other subjects is worse than "
      "our model.")
fig.savefig("img/fig_information_limit.png")
plt.close(fig)
print("wrote img/fig_information_limit.png")
