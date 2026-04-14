"""
Poster Figure 4 (non-technical audience version):
Feature Squeezing Adversarial Detector — simplified with plain-English labels.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

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

THRESHOLD  = 1.6
IMAGES     = ["car.jpg", "mouse.jpg", "tiger.jpg"]
IMG_LABELS = ["Car", "Mouse", "Tiger"]
C_CLEAN    = "#2ca02c"   # green
C_ADV      = "#d62728"   # red

# ── Load & deduplicate ──────────────────────────────────────────────────────
det = pd.read_csv("detection_results.csv")
det = det.drop_duplicates(subset=["Filename", "Type"])
clean = det[det["Type"] == "clean"].set_index("Filename")
adv   = det[det["Type"] == "adversarial"].set_index("Filename")

clean_max = [float(clean.loc[img, "Max_Score"]) for img in IMAGES]
adv_max   = [float(adv.loc[img,   "Max_Score"]) for img in IMAGES]

tp = sum(s > THRESHOLD for s in adv_max)
tn = sum(s <= THRESHOLD for s in clean_max)
fp = sum(s > THRESHOLD for s in clean_max)
fn = sum(s <= THRESHOLD for s in adv_max)

# ── Figure layout: 3 bar panels + 1 results summary ────────────────────────
fig = plt.figure(figsize=(17, 7))

# Main title + subtitle
fig.text(0.5, 0.98,
         "Catching Adversarial Images",
         ha="center", va="top", fontsize=20, fontweight="bold")
fig.text(0.5, 0.925,
         "We test each photo by simplifying it in different ways.\n"
         "Adversarial images change a lot more than normal ones.",
         ha="center", va="top", fontsize=13, color="#444444",
         linespacing=1.5)

gs = fig.add_gridspec(1, 4, wspace=0.55, top=0.80, bottom=0.18,
                      left=0.07, right=0.97)
axes_img = [fig.add_subplot(gs[0, i]) for i in range(3)]
ax_sum   = fig.add_subplot(gs[0, 3])

# ── Bar charts (one per image, only Max Score shown) ───────────────────────
x = np.array([0, 1])   # 0 = clean, 1 = adversarial
w = 0.5

for ax, img, img_label, c_score, a_score in zip(
        axes_img, IMAGES, IMG_LABELS, clean_max, adv_max):

    bars_c = ax.bar(0, c_score, w, color=C_CLEAN, edgecolor="white", linewidth=0.5, zorder=3)
    bars_a = ax.bar(1, a_score, w, color=C_ADV,   edgecolor="white", linewidth=0.5, zorder=3)

    ymax = max(c_score, a_score) * 1.35 + 0.15
    ax.set_ylim(0, ymax)

    # Threshold line
    ax.axhline(THRESHOLD, color="#333333", linewidth=2.2, linestyle="--", zorder=4)

    # Shaded "flagged" region
    ax.fill_between([-0.5, 1.5], THRESHOLD, ymax,
                    color="#d62728", alpha=0.08, zorder=2)

    # "FLAGGED" label in shaded region
    ax.text(0.5, (THRESHOLD + ymax) / 2, "FLAGGED\nAS ATTACK",
            ha="center", va="center", fontsize=9, color="#b00000",
            fontweight="bold", alpha=0.65, transform=ax.transData)

    # Bar value labels
    for bar in [bars_c[0], bars_a[0]]:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.04,
                f"{h:.2f}", ha="center", va="bottom", fontsize=11,
                fontweight="bold")

    # Outcome label below each bar
    c_outcome = "Safe" if c_score <= THRESHOLD else "False alarm"
    a_outcome = "Caught!" if a_score > THRESHOLD else "Missed"
    c_color   = C_CLEAN if c_score <= THRESHOLD else "#ff7f0e"
    a_color   = C_ADV   if a_score > THRESHOLD  else "#ff7f0e"

    ax.text(0, -0.22, c_outcome, ha="center", va="top", fontsize=10,
            color=c_color, fontweight="bold", transform=ax.get_xaxis_transform())
    ax.text(1, -0.22, a_outcome, ha="center", va="top", fontsize=10,
            color=a_color, fontweight="bold", transform=ax.get_xaxis_transform())

    ax.set_title(img_label, fontweight="bold", fontsize=15, pad=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Normal\nphoto", "Hacked\nphoto"], fontsize=11)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel("Suspicion Score\n(how much the image changed\nwhen we simplified it)",
                  fontsize=10) if ax == axes_img[0] else ax.set_ylabel("")
    ax.tick_params(left=True)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", bottom=False)

# ── Results summary panel ───────────────────────────────────────────────────
ax_sum.axis("off")

# Big result numbers
ax_sum.text(0.5, 0.97, "Results", ha="center", va="top",
            fontsize=16, fontweight="bold", transform=ax_sum.transAxes)

rows = [
    ("Attacks caught",    f"{tp} / 3", C_ADV),
    ("Normal photos\ncorrectly passed", f"{tn} / 3", C_CLEAN),
    ("False alarms\n(normal flagged)", f"{fp} / 3", "#ff7f0e" if fp > 0 else C_CLEAN),
    ("Attacks missed",    f"{fn} / 3", "#ff7f0e" if fn > 0 else C_CLEAN),
]

y_positions = [0.78, 0.57, 0.36, 0.15]
for (label, value, color), y in zip(rows, y_positions):
    ax_sum.text(0.5, y, value, ha="center", va="center",
                fontsize=28, fontweight="bold", color=color,
                transform=ax_sum.transAxes)
    ax_sum.text(0.5, y - 0.09, label, ha="center", va="center",
                fontsize=10, color="#444", transform=ax_sum.transAxes,
                linespacing=1.3)

# Border around summary panel
for spine in ["top", "bottom", "left", "right"]:
    ax_sum.spines[spine].set_visible(False)
rect = plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="#cccccc",
                      linewidth=1.5, transform=ax_sum.transAxes, clip_on=False)
ax_sum.add_patch(rect)

# ── Shared legend at bottom ─────────────────────────────────────────────────
legend_elements = [
    mpatches.Patch(facecolor=C_CLEAN, label="Normal photo — should score LOW"),
    mpatches.Patch(facecolor=C_ADV,   label="Hacked photo — should score HIGH"),
    Line2D([0], [0], color="#333333", linewidth=2.2, linestyle="--",
           label=f"Alert threshold ({THRESHOLD}) — anything above is flagged"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=3,
           fontsize=11, framealpha=0.9,
           bbox_to_anchor=(0.5, 0.01),
           handlelength=1.8, handleheight=1.0)

fig.savefig("poster_fig4_detector.png")
print("Saved poster_fig4_detector.png")
