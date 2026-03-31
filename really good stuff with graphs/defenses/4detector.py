import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.models import resnet18, ResNet18_Weights
import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import requests
from glob import glob

# --- 1. SETTINGS ---
device = torch.device("cpu")
IMAGE_FOLDER = "../pgd/my_images"
CSV_FILENAME = "detection_results.csv"
THRESHOLD = 1.6    # L1 distance threshold: above = adversarial

# --- 2. MODEL WRAPPER ---
class NormalizedModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    def forward(self, x):
        return self.model((x - self.mean) / self.std)

# --- 3. PGD ATTACK ---
def fast_gradient_method(model_fn, x, eps, norm, clip_min=0.0, clip_max=1.0, y=None):
    x_adv = x.clone().detach().requires_grad_(True)
    logits = model_fn(x_adv)
    if y is None:
        y = torch.argmax(logits, 1)
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()
    grad = x_adv.grad.data
    if norm == np.inf:
        grad = torch.sign(grad)
    elif norm == 2:
        grad = grad / (torch.norm(grad, p=2, dim=(1,2,3), keepdim=True) + 1e-12)
    return torch.clamp(x_adv + eps * grad, clip_min, clip_max).detach()

def projected_gradient_descent(model_fn, x, eps, eps_iter, nb_iter, norm, clip_min=0.0, clip_max=1.0, y=None, rand_init=True):
    if y is None:
        with torch.no_grad():
            y = torch.argmax(model_fn(x), 1)
    eta = torch.zeros_like(x).uniform_(-eps, eps) if rand_init else torch.zeros_like(x)
    eta = torch.clamp(eta, -eps, eps)
    adv_x = torch.clamp(x + eta, clip_min, clip_max)
    for _ in range(nb_iter):
        adv_x = fast_gradient_method(model_fn, adv_x, eps_iter, norm, clip_min, clip_max, y)
        eta = torch.clamp(adv_x - x, -eps, eps)
        adv_x = torch.clamp(x + eta, clip_min, clip_max)
    return adv_x

# --- 4. SQUEEZERS ---
def bit_depth_reduce(tensor, bits):
    levels = 2 ** bits - 1
    return torch.round(tensor * levels) / levels

def gaussian_smooth(tensor, kernel_size=3):
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return TF.gaussian_blur(tensor.squeeze(0), kernel_size=[k, k], sigma=1.0).unsqueeze(0)

SQUEEZERS = {
    "Smooth":       lambda x: gaussian_smooth(x),
    "Bit-2":        lambda x: bit_depth_reduce(x, 2),
    "Bit-2+Smooth": lambda x: gaussian_smooth(bit_depth_reduce(x, 2)),
}

# --- 5. DETECTION METRIC ---
def squeezing_distance(model_fn, x, squeezer):
    """L1 distance between softmax outputs of raw and squeezed input."""
    with torch.no_grad():
        p_raw      = torch.softmax(model_fn(x),             dim=1)
        p_squeezed = torch.softmax(model_fn(squeezer(x)),   dim=1)
    return torch.norm(p_raw - p_squeezed, p=1).item()

def detect(model_fn, x, threshold=THRESHOLD):
    """Returns (is_adversarial, scores_dict) using max distance across squeezers."""
    scores = {name: squeezing_distance(model_fn, x, sq) for name, sq in SQUEEZERS.items()}
    max_score = max(scores.values())
    return max_score > threshold, scores, max_score

# --- 6. ROC CURVE ---
def compute_roc(clean_scores, adv_scores):
    all_scores = clean_scores + adv_scores
    labels     = [0] * len(clean_scores) + [1] * len(adv_scores)  # 0=clean, 1=adversarial

    thresholds = sorted(set(all_scores + [0.0, 1.0]))
    tprs, fprs = [], []

    for t in thresholds:
        preds = [1 if s > t else 0 for s in all_scores]
        tp = sum(p == 1 and l == 1 for p, l in zip(preds, labels))
        fp = sum(p == 1 and l == 0 for p, l in zip(preds, labels))
        tn = sum(p == 0 and l == 0 for p, l in zip(preds, labels))
        fn = sum(p == 0 and l == 1 for p, l in zip(preds, labels))
        tprs.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0)

    return fprs, tprs, thresholds

# --- 7. LOGGING ---
def log_result(filename, is_clean, label, detected, max_score, scores):
    file_exists = os.path.isfile(CSV_FILENAME)
    with open(CSV_FILENAME, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Filename", "Type", "Label", "Detected", "Max_Score"] +
                            [f"Score_{k}" for k in SQUEEZERS])
        writer.writerow([
            os.path.basename(filename),
            "clean" if is_clean else "adversarial",
            label,
            detected,
            f"{max_score:.4f}",
            *[f"{scores[k]:.4f}" for k in SQUEEZERS]
        ])

# --- 8. MAIN ---
if __name__ == "__main__":
    print("[*] Loading model...")
    raw_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device).eval()
    model = NormalizedModel(raw_model)

    labels = requests.get(
        "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    ).text.splitlines()

    img_files = sorted(
        f for ext in ("*.jpg", "*.jpeg", "*.png")
        for f in glob(os.path.join(IMAGE_FOLDER, ext))
    )

    if not img_files:
        print(f"[!] No images found in {IMAGE_FOLDER}")
        exit()

    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])

    EPS      = 8 / 255
    EPS_ITER = 2 / 255
    STEPS    = 20

    print(f"[*] Found {len(img_files)} image(s). Running detection test...\n")
    print(f"{'Input':<30} {'Type':<12} {'Max Score':>10}  {'Detected?':>10}  Scores per squeezer")
    print("-" * 90)

    clean_scores, adv_scores = [], []
    results = []  # for plotting

    for f_path in img_files:
        x = transform(Image.open(f_path).convert("RGB")).unsqueeze(0).to(device)
        adv_x = projected_gradient_descent(model, x, EPS, EPS_ITER, STEPS, np.inf)

        with torch.no_grad():
            _, o_idx   = torch.max(torch.softmax(model(x),     1), 1)
            _, adv_idx = torch.max(torch.softmax(model(adv_x), 1), 1)

        # Test clean image
        detected_c, scores_c, max_c = detect(model, x)
        clean_scores.append(max_c)
        log_result(f_path, True, labels[o_idx.item()], detected_c, max_c, scores_c)
        score_str = "  ".join(f"{k}={v:.3f}" for k, v in scores_c.items())
        tag = "WRONG (FP)" if detected_c else "correct"
        print(f"{os.path.basename(f_path):<30} {'clean':<12} {max_c:>10.4f}  {tag:>10}  {score_str}")

        # Test adversarial image
        detected_a, scores_a, max_a = detect(model, adv_x)
        adv_scores.append(max_a)
        log_result(f_path, False, labels[adv_idx.item()], detected_a, max_a, scores_a)
        score_str = "  ".join(f"{k}={v:.3f}" for k, v in scores_a.items())
        tag = "correct" if detected_a else "WRONG (FN)"
        print(f"{os.path.basename(f_path):<30} {'adversarial':<12} {max_a:>10.4f}  {tag:>10}  {score_str}")

        results.append((os.path.basename(f_path), x, adv_x, max_c, max_a,
                        labels[o_idx.item()], labels[adv_idx.item()]))
        print()

    # --- Summary ---
    tp = sum(1 for s in adv_scores   if s >  THRESHOLD)
    tn = sum(1 for s in clean_scores if s <= THRESHOLD)
    fp = sum(1 for s in clean_scores if s >  THRESHOLD)
    fn = sum(1 for s in adv_scores   if s <= THRESHOLD)
    print(f"[*] Detection @ threshold={THRESHOLD}:  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"    Accuracy: {(tp+tn)/(tp+tn+fp+fn):.0%}   TPR: {tp/(tp+fn) if (tp+fn)>0 else 0:.0%}   FPR: {fp/(fp+tn) if (fp+tn)>0 else 0:.0%}")

    # --- ROC curve ---
    fprs, tprs, thresholds = compute_roc(clean_scores, adv_scores)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: score distribution
    ax = axes[0]
    names = [r[0] for r in results]
    x_pos = np.arange(len(names))
    ax.bar(x_pos - 0.2, [r[3] for r in results], 0.4, label="Clean",       color="steelblue")
    ax.bar(x_pos + 0.2, [r[4] for r in results], 0.4, label="Adversarial", color="tomato")
    ax.axhline(THRESHOLD, color='black', linestyle='--', linewidth=1.2, label=f"Threshold={THRESHOLD}")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylabel("Max Squeezing Distance (L1)")
    ax.set_title("Detection Scores: Clean vs Adversarial")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: ROC curve
    ax = axes[1]
    ax.plot(fprs, tprs, marker='o', color='darkorange', linewidth=2, label="Detector")
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label="Random")
    # Mark chosen threshold
    chosen_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    chosen_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    ax.scatter([chosen_fpr], [chosen_tpr], color='red', s=100, zorder=5,
               label=f"Threshold={THRESHOLD}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("Feature Squeezing Adversarial Detector", fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_path = "detector_results.png"
    plt.savefig(out_path)
    plt.show()
    print(f"\n[*] Plot saved to {out_path}")
    print(f"[*] Results saved to {CSV_FILENAME}")
