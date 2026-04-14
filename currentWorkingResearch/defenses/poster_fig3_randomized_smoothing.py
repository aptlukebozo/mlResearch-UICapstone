"""
Poster Figure 3 (non-technical audience version):
Randomized Smoothing Defense — plain-English labels.

The defense makes 100 predictions by adding a different random noise pattern
each time, then takes the majority vote.  Bars show the vote fraction for
whichever class won — green if it's the correct class, red if still wrong.
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
df = pd.read_csv("randomized_smoothing_results.csv").set_index("Filename")

SIGMAS  = [0.1, 0.25, 0.5]
SLABELS = ["Low noise\n(σ = 0.10)", "Medium noise\n(σ = 0.25)", "High noise\n(σ = 0.50)"]

data = {
    img: {
        "clean_conf": float(df.loc[img, "O_Conf"]),
        "results": [
            (float(df.loc[img, f"RS_s{s}_VoteFrac"]),
             str(df.loc[img,  f"RS_s{s}_Restored"]).strip().lower() == "true")
            for s in SIGMAS
        ],
    }
    for img in IMAGES
}

total_restored = sum(r for img in IMAGES for _, r in data[img]["results"])
total_cases    = 9

# ── Figure layout ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(17, 7))

fig.text(0.5, 0.98, "Can Averaging Noisy Predictions Remove the Hidden Attack?",
         ha="center", va="top", fontsize=20, fontweight="bold")
fig.text(0.5, 0.925,
         "We add random noise to the hacked image 100 times and vote on the result.\n"
         "If the majority vote matches the correct answer, the defense worked.",
         ha="center", va="top", fontsize=13, color="#444444", linespacing=1.5)

gs = fig.add_gridspec(1, 4, wspace=0.55, top=0.80, bottom=0.18,
                      left=0.07, right=0.97)
axes_img = [fig.add_subplot(gs[0, i]) for i in range(3)]
ax_sum   = fig.add_subplot(gs[0, 3])

x = np.array([0, 1, 2])
w = 0.5

# ── Bar panels ───────────────────────────────────────────────────────────────
for ax, img, img_label in zip(axes_img, IMAGES, IMG_LABELS):
    results = data[img]["results"]
    clean_c = data[img]["clean_conf"]

    ax.set_ylim(0, 1.30)

    for i, (vote_frac, restored) in enumerate(results):
        color = C_CLEAN if restored else C_ADV
        ax.bar(i, vote_frac, w, color=color, edgecolor="white", linewidth=0.5, zorder=3)

        ax.text(i, vote_frac + 0.03, f"{vote_frac:.0%}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

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
    ax.set_xticklabels(SLABELS, fontsize=10)
    ax.set_xlim(-0.6, 2.6)
    if ax == axes_img[0]:
        ax.set_ylabel("Fraction of votes for predicted answer\n(100 noisy predictions cast)",
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

verdict = "Completely\nineffective" if total_restored == 0 else (
          "Mostly\nineffective"    if total_restored < 5  else
          "Mostly\neffective")
v_color = C_ADV if total_restored < 5 else C_CLEAN
ax_sum.text(0.5, 0.10, "Verdict:", ha="center", va="center",
            fontsize=11, color="#444", transform=ax_sum.transAxes)
ax_sum.text(0.5, 0.04, verdict, ha="center", va="center",
            fontsize=13, fontweight="bold", color=v_color,
            transform=ax_sum.transAxes, linespacing=1.2)

rect = plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="#cccccc",
                      linewidth=1.5, transform=ax_sum.transAxes, clip_on=False)
ax_sum.add_patch(rect)

# ── Shared legend ────────────────────────────────────────────────────────────
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

fig.savefig("poster_fig3_randomized_smoothing.png")
print("Saved poster_fig3_randomized_smoothing.png")
