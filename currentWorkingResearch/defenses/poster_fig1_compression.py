"""
Poster Figure 1 (non-technical audience version):
JPEG Compression Defense — plain-English labels.
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

IMAGES     = ["car.jpg", "mouse.jpg", "tiger.jpg"]
IMG_LABELS = ["Car", "Mouse", "Tiger"]
C_CLEAN    = "#2ca02c"   # green  (restored)
C_ADV      = "#d62728"   # red    (still fooled)

# ── Load & index ────────────────────────────────────────────────────────────
df = pd.read_csv("compression_defense_results.csv").set_index("Filename")

QUALITIES = [25, 50, 75]
QLABELS   = ["Heavy\ncompression\n(Q25)", "Medium\ncompression\n(Q50)", "Light\ncompression\n(Q75)"]

data = {
    img: {
        "clean_conf": float(df.loc[img, "O_Conf"]),
        "results": [
            (float(df.loc[img, f"Q{q}_Conf"]),
             str(df.loc[img, f"Q{q}_Restored"]).strip().lower() == "true")
            for q in QUALITIES
        ],
    }
    for img in IMAGES
}

total_restored = sum(r for img in IMAGES for _, r in data[img]["results"])
total_cases    = 9

# ── Figure layout ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(17, 7))

fig.text(0.5, 0.98, "Can Compressing the Image Remove the Hidden Attack?",
         ha="center", va="top", fontsize=20, fontweight="bold")
fig.text(0.5, 0.925,
         "After hacking a photo, we re-save it at three quality levels.\n"
         "If the AI correctly names the subject again, the defense worked.",
         ha="center", va="top", fontsize=13, color="#444444", linespacing=1.5)

gs = fig.add_gridspec(1, 4, wspace=0.55, top=0.80, bottom=0.18,
                      left=0.07, right=0.97)
axes_img = [fig.add_subplot(gs[0, i]) for i in range(3)]
ax_sum   = fig.add_subplot(gs[0, 3])

x = np.array([0, 1, 2])
w = 0.5

# ── Bar panels (one per image) ───────────────────────────────────────────────
for ax, img, img_label in zip(axes_img, IMAGES, IMG_LABELS):
    results  = data[img]["results"]
    clean_c  = data[img]["clean_conf"]

    ymax = 1.30
    ax.set_ylim(0, ymax)

    for i, (conf, restored) in enumerate(results):
        color = C_CLEAN if restored else C_ADV
        ax.bar(i, conf, w, color=color, edgecolor="white", linewidth=0.5, zorder=3)

        # Value label on bar
        ax.text(i, conf + 0.03, f"{conf:.0%}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

        # Outcome label below x-axis
        outcome = "Fixed!" if restored else "Still fooled"
        c_out   = C_CLEAN if restored else C_ADV
        ax.text(i, -0.16, outcome,
                ha="center", va="top", fontsize=9.5,
                color=c_out, fontweight="bold",
                transform=ax.get_xaxis_transform())

    # Dashed line: original (pre-attack) confidence
    ax.axhline(clean_c, color="#666666", linewidth=1.8, linestyle="--",
               zorder=2, alpha=0.8)

    ax.set_title(img_label, fontweight="bold", fontsize=15, pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(QLABELS, fontsize=10)
    ax.set_xlim(-0.6, 2.6)
    if ax == axes_img[0]:
        ax.set_ylabel("AI's confidence in its answer\n(1 = fully certain, 0 = unsure)",
                      fontsize=10)
    else:
        ax.set_ylabel("")
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", bottom=False)

# ── Results summary panel ────────────────────────────────────────────────────
ax_sum.axis("off")
ax_sum.text(0.5, 0.97, "Results",
            ha="center", va="top", fontsize=16, fontweight="bold",
            transform=ax_sum.transAxes)

rows = [
    ("Attacks\nneutralized",  f"{total_restored} / {total_cases}", C_CLEAN),
    ("Still\nfooling the AI", f"{total_cases - total_restored} / {total_cases}", C_ADV),
]
y_positions = [0.70, 0.38]
for (label, value, color), y in zip(rows, y_positions):
    ax_sum.text(0.5, y, value,
                ha="center", va="center", fontsize=36, fontweight="bold",
                color=color, transform=ax_sum.transAxes)
    ax_sum.text(0.5, y - 0.13, label,
                ha="center", va="center", fontsize=11, color="#444",
                transform=ax_sum.transAxes, linespacing=1.3)

verdict = "Mostly\nineffective" if total_restored < 5 else "Mostly\neffective"
v_color = C_ADV if total_restored < 5 else C_CLEAN
ax_sum.text(0.5, 0.10, "Verdict:", ha="center", va="center",
            fontsize=11, color="#444", transform=ax_sum.transAxes)
ax_sum.text(0.5, 0.04, verdict, ha="center", va="center",
            fontsize=13, fontweight="bold", color=v_color,
            transform=ax_sum.transAxes, linespacing=1.2)

rect = plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="#cccccc",
                      linewidth=1.5, transform=ax_sum.transAxes, clip_on=False)
ax_sum.add_patch(rect)

# ── Shared legend at bottom ───────────────────────────────────────────────────
legend_elements = [
    mpatches.Patch(facecolor=C_CLEAN,
                   label="Attack neutralized — AI answers correctly again"),
    mpatches.Patch(facecolor=C_ADV,
                   label="Attack survived — AI still gives the wrong answer"),
    Line2D([0], [0], color="#666666", linewidth=1.8, linestyle="--",
           label="AI's original confidence before the attack"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=3,
           fontsize=11, framealpha=0.9,
           bbox_to_anchor=(0.5, 0.01),
           handlelength=1.8, handleheight=1.0)

fig.savefig("poster_fig1_compression.png")
print("Saved poster_fig1_compression.png")
