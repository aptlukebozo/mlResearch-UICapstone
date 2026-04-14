"""
Poster-quality visualizations for adversarial defense results.
Generates three figures:
  1. Defense Confidence Comparison (adversarial vs. each defense variant)
  2. Defense Restoration Success Rate (heatmap)
  3. Adversarial Detector Scores (clean vs. adversarial)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

IMAGES = ["car.jpg", "mouse.jpg", "tiger.jpg"]
IMG_LABELS = ["Car", "Mouse", "Tiger"]
PALETTE = {
    "adversarial": "#d62728",
    "original":    "#2ca02c",
    "compression": "#1f77b4",
    "squeezing":   "#ff7f0e",
    "smoothing":   "#9467bd",
}

# ── Load data ──────────────────────────────────────────────────────────────────
comp = pd.read_csv("compression_defense_results.csv")
fs   = pd.read_csv("feature_squeezing_results.csv")
rs   = pd.read_csv("randomized_smoothing_results.csv")
det  = pd.read_csv("detection_results.csv")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 – Confidence comparison across defenses
# ══════════════════════════════════════════════════════════════════════════════
fig1, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
fig1.suptitle("Confidence After Each Defense vs. Adversarial Input", fontsize=17, fontweight="bold", y=1.02)

# defense variants and their short display names
defense_cols = {
    # (df, conf_col, label)
    "Adv":      (comp,  "Adv_Conf",    PALETTE["adversarial"]),
    "JPEG Q25": (comp,  "Q25_Conf",    PALETTE["compression"]),
    "JPEG Q50": (comp,  "Q50_Conf",    PALETTE["compression"]),
    "JPEG Q75": (comp,  "Q75_Conf",    PALETTE["compression"]),
    "FS Smooth":(fs,    "Smooth_Conf", PALETTE["squeezing"]),
    "FS Bit-1": (fs,    "Bit1_Conf",   PALETTE["squeezing"]),
    "FS Bit-2": (fs,    "Bit2_Conf",   PALETTE["squeezing"]),
    "FS Bit-4": (fs,    "Bit4_Conf",   PALETTE["squeezing"]),
    "RS σ=0.10":(rs,    "RS_s0.1_VoteFrac",  PALETTE["smoothing"]),
    "RS σ=0.25":(rs,    "RS_s0.25_VoteFrac", PALETTE["smoothing"]),
    "RS σ=0.50":(rs,    "RS_s0.5_VoteFrac",  PALETTE["smoothing"]),
}

names  = list(defense_cols.keys())
colors = [v[2] for v in defense_cols.values()]
x      = np.arange(len(names))
width  = 0.65

for ax, img, img_label in zip(axes, IMAGES, IMG_LABELS):
    vals = []
    for name, (df, col, _) in defense_cols.items():
        row = df[df["Filename"] == img]
        vals.append(float(row[col].values[0]) if len(row) else 0.0)

    bars = ax.bar(x, vals, width, color=colors, edgecolor="white", linewidth=0.6)

    # horizontal line at adversarial confidence
    adv_conf = float(comp[comp["Filename"] == img]["Adv_Conf"].values[0])
    ax.axhline(adv_conf, color=PALETTE["adversarial"], linewidth=1.8,
               linestyle="--", label=f"Adv. conf = {adv_conf:.2f}")
    # original clean confidence
    orig_conf = float(comp[comp["Filename"] == img]["O_Conf"].values[0])
    ax.axhline(orig_conf, color=PALETTE["original"], linewidth=1.8,
               linestyle=":", label=f"Clean conf = {orig_conf:.2f}")

    ax.set_title(img_label, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Confidence / Vote Fraction" if ax == axes[0] else "")
    ax.legend(fontsize=9, framealpha=0.7)

# Legend for defence groups
legend_patches = [
    mpatches.Patch(color=PALETTE["compression"], label="JPEG Compression"),
    mpatches.Patch(color=PALETTE["squeezing"],   label="Feature Squeezing"),
    mpatches.Patch(color=PALETTE["smoothing"],   label="Randomized Smoothing"),
]
fig1.legend(handles=legend_patches, loc="lower center", ncol=3,
            bbox_to_anchor=(0.5, -0.06), framealpha=0.8, fontsize=11)

fig1.tight_layout()
fig1.savefig("poster_fig1_confidence.png")
print("Saved poster_fig1_confidence.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 – Restoration success heatmap
# ══════════════════════════════════════════════════════════════════════════════
restoration_cols = {
    "JPEG Q25": (comp, "Q25_Restored"),
    "JPEG Q50": (comp, "Q50_Restored"),
    "JPEG Q75": (comp, "Q75_Restored"),
    "FS Smooth":(fs,   "Smooth_Restored"),
    "FS Bit-1": (fs,   "Bit1_Restored"),
    "FS Bit-2": (fs,   "Bit2_Restored"),
    "FS Bit-4": (fs,   "Bit4_Restored"),
    "FS C1":    (fs,   "Combo1_Restored"),
    "FS C2":    (fs,   "Combo2_Restored"),
    "FS C4":    (fs,   "Combo4_Restored"),
    "RS σ=0.10":(rs,   "RS_s0.1_Restored"),
    "RS σ=0.25":(rs,   "RS_s0.25_Restored"),
    "RS σ=0.50":(rs,   "RS_s0.5_Restored"),
}

matrix = np.zeros((len(IMAGES), len(restoration_cols)))
for j, (name, (df, col)) in enumerate(restoration_cols.items()):
    for i, img in enumerate(IMAGES):
        row = df[df["Filename"] == img]
        if len(row):
            val = row[col].values[0]
            matrix[i, j] = 1 if str(val).strip().lower() == "true" else 0

fig2, ax = plt.subplots(figsize=(13, 4))
fig2.suptitle("Defense Restoration Success\n(Green = original label recovered)", fontsize=16, fontweight="bold")

cmap = plt.cm.RdYlGn
im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")

ax.set_xticks(range(len(restoration_cols)))
ax.set_xticklabels(list(restoration_cols.keys()), rotation=40, ha="right", fontsize=11)
ax.set_yticks(range(len(IMAGES)))
ax.set_yticklabels(IMG_LABELS, fontsize=12)

for i in range(len(IMAGES)):
    for j in range(len(restoration_cols)):
        text = "Yes" if matrix[i, j] == 1 else "No"
        color = "black"
        ax.text(j, i, text, ha="center", va="center", fontsize=10, color=color, fontweight="bold")

# vertical separators for defense groups
for xpos in [2.5, 6.5, 9.5]:
    ax.axvline(xpos, color="white", linewidth=2.5)

# group labels above the heatmap
group_info = [("JPEG\nCompression", 1), ("Feature Squeezing", 6), ("Rand.\nSmoothing", 11)]
for label, midcol in group_info:
    ax.annotate(label, xy=(midcol, -0.7), xycoords=("data", "axes fraction"),
                ha="center", va="bottom", fontsize=10, color="#333333",
                annotation_clip=False)

plt.colorbar(im, ax=ax, ticks=[0, 1], label="Restored")
fig2.tight_layout()
fig2.savefig("poster_fig2_restoration.png")
print("Saved poster_fig2_restoration.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 – Detector scores: clean vs. adversarial
# ══════════════════════════════════════════════════════════════════════════════
# Use first occurrence of each (Filename, Type) pair
det_clean = det[det["Type"] == "clean"].drop_duplicates(subset="Filename").set_index("Filename")
det_adv   = det[det["Type"] == "adversarial"].drop_duplicates(subset="Filename").set_index("Filename")

score_cols  = ["Score_Smooth", "Score_Bit-2", "Score_Bit-2+Smooth", "Max_Score"]
score_names = ["Smooth\nResidual", "Bit-2\nResidual", "Bit-2 +\nSmooth", "Max\nScore"]

fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
fig3.suptitle("Adversarial Detector Scores: Clean vs. Adversarial", fontsize=17, fontweight="bold", y=1.02)

x3    = np.arange(len(score_cols))
w3    = 0.35
thresh = 1.0  # detection threshold used in detector.py

for ax, img, img_label in zip(axes3, IMAGES, IMG_LABELS):
    clean_vals = [float(det_clean.loc[img, c]) if img in det_clean.index else 0 for c in score_cols]
    adv_vals   = [float(det_adv.loc[img, c])   if img in det_adv.index   else 0 for c in score_cols]

    bars_c = ax.bar(x3 - w3/2, clean_vals, w3, label="Clean",       color=PALETTE["original"],     edgecolor="white")
    bars_a = ax.bar(x3 + w3/2, adv_vals,   w3, label="Adversarial", color=PALETTE["adversarial"],  edgecolor="white")

    ax.axhline(thresh, color="#444", linewidth=1.6, linestyle="--", label=f"Threshold = {thresh}")
    ax.set_title(img_label, fontweight="bold")
    ax.set_xticks(x3)
    ax.set_xticklabels(score_names, fontsize=10)
    ax.set_ylabel("Detector Score" if ax == axes3[0] else "")
    ax.set_ylim(0, max(max(clean_vals), max(adv_vals)) * 1.18 + 0.05)
    ax.legend(fontsize=9, framealpha=0.7)

    # annotate bars with values
    for bar in list(bars_c) + list(bars_a):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8.5)

fig3.tight_layout()
fig3.savefig("poster_fig3_detector.png")
print("Saved poster_fig3_detector.png")

plt.show()
print("Done.")
