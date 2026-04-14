"""
5compare_models.py
Runs the feature-squeezing adversarial detector across three model architectures
and compares detection performance with a poster-quality figure.

For each model:
  1. Load pretrained ImageNet weights
  2. Generate white-box PGD adversarial examples (ε=8/255, 20 steps)
  3. Score every clean + adversarial image with the three squeezers
  4. Sweep thresholds to build a ROC curve; mark threshold=1.6

Output
  model_comparison.png  –  ROC curves + metrics bar chart
  model_comparison.csv  –  per-image scores for all three models
"""

import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.models import (
    googlenet,          GoogLeNet_Weights,
    resnet50,           ResNet50_Weights,
    mobilenet_v3_large, MobileNet_V3_Large_Weights,
)
import PIL.Image as Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from glob import glob

# ── Settings ─────────────────────────────────────────────────────────────────
IMAGE_FOLDER = "../pgd/my_images"
CSV_OUT      = "model_comparison.csv"
THRESHOLD    = 1.6
EPS          = 8  / 255
EPS_ITER     = 2  / 255
STEPS        = 20
device       = torch.device("cpu")

MODELS = {
    "GoogLeNet":     (googlenet,          GoogLeNet_Weights.IMAGENET1K_V1),
    "ResNet-50":     (resnet50,           ResNet50_Weights.IMAGENET1K_V2),
    "MobileNet-V3":  (mobilenet_v3_large, MobileNet_V3_Large_Weights.IMAGENET1K_V2),
}

MODEL_COLORS = {
    "GoogLeNet":    "#1f77b4",   # blue
    "ResNet-50":    "#ff7f0e",   # orange
    "MobileNet-V3": "#2ca02c",   # green
}

# ── Model wrapper ─────────────────────────────────────────────────────────────
class NormalizedModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
    def forward(self, x):
        return self.model((x - self.mean) / self.std)

# ── PGD attack ────────────────────────────────────────────────────────────────
def pgd(model_fn, x, eps, eps_iter, nb_iter, y=None):
    if y is None:
        with torch.no_grad():
            y = torch.argmax(model_fn(x), 1)
    eta  = torch.zeros_like(x).uniform_(-eps, eps)
    adv  = torch.clamp(x + eta, 0, 1)
    for _ in range(nb_iter):
        adv = adv.clone().detach().requires_grad_(True)
        loss = nn.CrossEntropyLoss()(model_fn(adv), y)
        loss.backward()
        adv = adv + eps_iter * adv.grad.sign()
        adv = torch.clamp(x + torch.clamp(adv - x, -eps, eps), 0, 1).detach()
    return adv

# ── Squeezers ─────────────────────────────────────────────────────────────────
def bit_depth_reduce(t, bits):
    levels = 2 ** bits - 1
    return torch.round(t * levels) / levels

def gaussian_smooth(t, k=3):
    return TF.gaussian_blur(t.squeeze(0), kernel_size=[k, k], sigma=1.0).unsqueeze(0)

SQUEEZERS = {
    "Smooth":        lambda x: gaussian_smooth(x),
    "Bit-2":         lambda x: bit_depth_reduce(x, 2),
    "Bit-2+Smooth":  lambda x: gaussian_smooth(bit_depth_reduce(x, 2)),
}

def max_squeezing_score(model_fn, x):
    """Max L1 distance between raw and squeezed softmax outputs."""
    with torch.no_grad():
        p_raw = torch.softmax(model_fn(x), dim=1)
        scores = {
            name: torch.norm(p_raw - torch.softmax(model_fn(sq(x)), dim=1), p=1).item()
            for name, sq in SQUEEZERS.items()
        }
    return max(scores.values()), scores

# ── ROC helpers ───────────────────────────────────────────────────────────────
def compute_roc(clean_scores, adv_scores):
    all_scores = clean_scores + adv_scores
    labels     = [0] * len(clean_scores) + [1] * len(adv_scores)
    thresholds = sorted(set(all_scores + [0.0, max(all_scores) + 0.01]))
    fprs, tprs = [], []
    for t in thresholds:
        preds = [1 if s > t else 0 for s in all_scores]
        tp = sum(p == 1 and l == 1 for p, l in zip(preds, labels))
        fp = sum(p == 1 and l == 0 for p, l in zip(preds, labels))
        tn = sum(p == 0 and l == 0 for p, l in zip(preds, labels))
        fn = sum(p == 0 and l == 1 for p, l in zip(preds, labels))
        tprs.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)
    return fprs, tprs

def auc(fprs, tprs):
    """Trapezoidal AUC."""
    pts = sorted(zip(fprs, tprs))
    xs, ys = zip(*pts)
    return float(np.trapezoid(ys, xs))

# ── Main loop ─────────────────────────────────────────────────────────────────
transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])

img_files = sorted(
    f for ext in ("*.png", "*.jpg", "*.jpeg")
    for f in glob(os.path.join(IMAGE_FOLDER, ext))
)
if not img_files:
    raise FileNotFoundError(f"No images found in {IMAGE_FOLDER}")

print(f"Images: {len(img_files)}\n")

all_results  = {}   # model_name -> {clean_scores, adv_scores, tp, tn, fp, fn, fprs, tprs, auc_val}
csv_rows     = []

for model_name, (model_fn, weights) in MODELS.items():
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    raw   = model_fn(weights=weights).to(device).eval()
    model = NormalizedModel(raw)

    clean_scores, adv_scores = [], []

    for f_path in img_files:
        fname = os.path.basename(f_path)
        x     = transform(Image.open(f_path).convert("RGB")).unsqueeze(0)

        # White-box PGD attack
        adv_x = pgd(model, x, EPS, EPS_ITER, STEPS)

        c_max, c_scores = max_squeezing_score(model, x)
        a_max, a_scores = max_squeezing_score(model, adv_x)

        clean_scores.append(c_max)
        adv_scores.append(a_max)

        c_tag = "FP" if c_max > THRESHOLD else "TN"
        a_tag = "TP" if a_max > THRESHOLD else "FN"
        print(f"  {fname:<30}  clean={c_max:.3f} ({c_tag})   adv={a_max:.3f} ({a_tag})")

        csv_rows.append({
            "model":        model_name,
            "filename":     fname,
            "clean_score":  f"{c_max:.4f}",
            "adv_score":    f"{a_max:.4f}",
            **{f"clean_{k}": f"{v:.4f}" for k, v in c_scores.items()},
            **{f"adv_{k}":   f"{v:.4f}" for k, v in a_scores.items()},
        })

    tp = sum(1 for s in adv_scores   if s >  THRESHOLD)
    tn = sum(1 for s in clean_scores if s <= THRESHOLD)
    fp = sum(1 for s in clean_scores if s >  THRESHOLD)
    fn = sum(1 for s in adv_scores   if s <= THRESHOLD)
    n  = len(img_files)

    acc  = (tp + tn) / (2 * n)
    tpr  = tp / n
    fpr  = fp / n

    fprs, tprs = compute_roc(clean_scores, adv_scores)
    auc_val    = auc(fprs, tprs)

    all_results[model_name] = dict(
        clean_scores=clean_scores, adv_scores=adv_scores,
        tp=tp, tn=tn, fp=fp, fn=fn,
        acc=acc, tpr=tpr, fpr=fpr,
        fprs=fprs, tprs=tprs, auc_val=auc_val,
    )

    print(f"\n  TP={tp}  TN={tn}  FP={fp}  FN={fn}  "
          f"Acc={acc:.0%}  TPR={tpr:.0%}  FPR={fpr:.0%}  AUC={auc_val:.3f}\n")

# ── Save CSV ──────────────────────────────────────────────────────────────────
if csv_rows:
    with open(CSV_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved {CSV_OUT}")

# ── Plot ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

fig = plt.figure(figsize=(18, 6))

fig.text(0.5, 0.99,
         "Feature Squeezing Detector — Comparison Across Three Model Architectures",
         ha="center", va="top", fontsize=17, fontweight="bold")
fig.text(0.5, 0.945,
         "Each model is attacked with white-box PGD (ε = 8/255, 20 steps) and then tested with its own detector.",
         ha="center", va="top", fontsize=12, color="#444444")

gs = fig.add_gridspec(1, 3, wspace=0.40, top=0.86, bottom=0.14,
                      left=0.06, right=0.97)
ax_roc   = fig.add_subplot(gs[0, 0])
ax_bar   = fig.add_subplot(gs[0, 1])
ax_score = fig.add_subplot(gs[0, 2])

# ── Left: ROC curves ──────────────────────────────────────────────────────────
ax_roc.plot([0, 1], [0, 1], "k--", linewidth=1.2, alpha=0.5, label="Random guess")

for model_name, res in all_results.items():
    color = MODEL_COLORS[model_name]
    ax_roc.plot(res["fprs"], res["tprs"],
                color=color, linewidth=2.2,
                label=f"{model_name}  (AUC={res['auc_val']:.2f})")
    # Mark chosen threshold operating point
    ax_roc.scatter([res["fpr"]], [res["tpr"]],
                   color=color, s=90, zorder=5, edgecolors="white", linewidths=1.2)

ax_roc.set_xlabel("False Positive Rate\n(clean images wrongly flagged)")
ax_roc.set_ylabel("True Positive Rate\n(attacks correctly caught)")
ax_roc.set_title("ROC Curves", fontweight="bold")
ax_roc.set_xlim(-0.04, 1.04)
ax_roc.set_ylim(-0.04, 1.04)
ax_roc.legend(fontsize=10, framealpha=0.85, loc="lower right")
ax_roc.grid(True, alpha=0.25)
ax_roc.text(0.62, 0.08, f"● = threshold {THRESHOLD}",
            fontsize=9, color="#555", transform=ax_roc.transAxes)

# ── Middle: Accuracy / TPR / FPR bar chart ────────────────────────────────────
model_names = list(all_results.keys())
metrics     = ["acc", "tpr", "fpr"]
metric_labels = ["Accuracy", "Attack\ndetection rate\n(TPR)", "False alarm\nrate\n(FPR)"]
x      = np.arange(len(metrics))
width  = 0.22
colors = [MODEL_COLORS[m] for m in model_names]

for i, (mname, color) in enumerate(zip(model_names, colors)):
    vals = [all_results[mname][m] for m in metrics]
    offset = (i - 1) * width
    bars = ax_bar.bar(x + offset, vals, width,
                      color=color, edgecolor="white", linewidth=0.5,
                      label=mname, zorder=3)
    for bar, val in zip(bars, vals):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.015,
                    f"{val:.0%}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(metric_labels, fontsize=10)
ax_bar.set_ylim(0, 1.20)
ax_bar.set_ylabel("Rate  (higher = better for Acc & TPR,\nlower = better for FPR)")
ax_bar.set_title("Detection Metrics at Threshold 1.6", fontweight="bold")
ax_bar.legend(fontsize=10, framealpha=0.85)
ax_bar.axhline(1.0, color="#aaa", linewidth=0.8, linestyle="--", zorder=1)
ax_bar.grid(True, alpha=0.20, axis="y")
ax_bar.spines["bottom"].set_visible(False)
ax_bar.tick_params(axis="x", bottom=False)

# ── Right: TP / TN / FP / FN grouped bars ────────────────────────────────────
count_metrics = ["tp", "tn", "fp", "fn"]
count_labels  = ["Attacks\ncaught\n(TP)", "Clean\npassed\n(TN)",
                 "False\nalarms\n(FP)", "Missed\nattacks\n(FN)"]
count_colors  = ["#2ca02c", "#2ca02c", "#d62728", "#d62728"]   # good=green, bad=red
x2     = np.arange(len(count_metrics))
n_imgs = len(img_files)

for i, (mname, mcolor) in enumerate(zip(model_names, colors)):
    vals   = [all_results[mname][m] for m in count_metrics]
    offset = (i - 1) * width
    bars   = ax_score.bar(x2 + offset, vals, width,
                          color=mcolor, edgecolor="white", linewidth=0.5,
                          label=mname, zorder=3)
    for bar, val in zip(bars, vals):
        if val > 0:
            ax_score.text(bar.get_x() + bar.get_width() / 2,
                          bar.get_height() + 0.12,
                          str(val), ha="center", va="bottom",
                          fontsize=10, fontweight="bold")

ax_score.set_xticks(x2)
ax_score.set_xticklabels(count_labels, fontsize=10)
ax_score.set_ylim(0, n_imgs * 1.35)
ax_score.set_ylabel(f"Count  (out of {n_imgs} images each)")
ax_score.set_title("Raw Counts at Threshold 1.6", fontweight="bold")
ax_score.legend(fontsize=10, framealpha=0.85)
ax_score.axhline(n_imgs, color="#aaa", linewidth=0.8, linestyle="--", zorder=1,
                 label=f"Max = {n_imgs}")
ax_score.grid(True, alpha=0.20, axis="y")
ax_score.spines["bottom"].set_visible(False)
ax_score.tick_params(axis="x", bottom=False)

fig.savefig("model_comparison.png")
print("Saved model_comparison.png")
plt.show()
