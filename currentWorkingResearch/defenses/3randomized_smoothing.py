import torch
import torch.nn as nn
import torchvision.transforms as T
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
CSV_FILENAME = "randomized_smoothing_results.csv"
SIGMA_VALUES = [0.1, 0.25, 0.5]   # Noise levels to test
N_SAMPLES    = 100                 # Forward passes per prediction

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
def fast_gradient_method(model_fn, x, eps, norm, clip_min=0.0, clip_max=1.0, y=None, targeted=False):
    x_adv = x.clone().detach().requires_grad_(True)
    logits = model_fn(x_adv)
    if y is None:
        y = torch.argmax(logits, 1)
    loss = nn.CrossEntropyLoss()(logits, y)
    if targeted:
        loss = -loss
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

# --- 4. RANDOMIZED SMOOTHING ---
def randomized_smoothing_predict(model_fn, x, sigma, n_samples, n_classes=1000):
    """
    Adds Gaussian noise to x `n_samples` times, runs each through the model,
    and returns the majority-vote class + its vote fraction as confidence.
    """
    vote_counts = torch.zeros(n_classes)

    with torch.no_grad():
        for _ in range(n_samples):
            noise = torch.randn_like(x) * sigma
            noisy = torch.clamp(x + noise, 0.0, 1.0)
            logits = model_fn(noisy)
            pred = torch.argmax(logits, dim=1)
            vote_counts[pred.item()] += 1

    winner = torch.argmax(vote_counts).item()
    confidence = vote_counts[winner].item() / n_samples
    return winner, confidence

# --- 5. LOGGING ---
def log_result(filename, orig_label, adv_label, o_conf, adv_conf, rs_results):
    file_exists = os.path.isfile(CSV_FILENAME)
    with open(CSV_FILENAME, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            header = ["Filename", "Original", "Adversarial", "O_Conf", "Adv_Conf"]
            for s in SIGMA_VALUES:
                header += [f"RS_s{s}_Label", f"RS_s{s}_VoteFrac", f"RS_s{s}_Restored"]
            writer.writerow(header)

        row = [os.path.basename(filename), orig_label, adv_label, f"{o_conf:.4f}", f"{adv_conf:.4f}"]
        for s in SIGMA_VALUES:
            label, vote_frac, restored = rs_results[s]
            row += [label, f"{vote_frac:.4f}", restored]
        writer.writerow(row)

# --- 6. PLOTTING ---
def plot_results(orig_t, adv_t, labels, o_idx, adv_idx, o_conf, adv_conf, rs_results, filename):
    def prep(t):
        return np.clip(t.squeeze(0).permute(1, 2, 0).detach().numpy(), 0, 1)

    n_cols = 2 + len(SIGMA_VALUES)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    axes[0].imshow(prep(orig_t))
    axes[0].set_title(f"Original\n{labels[o_idx][:16]}\n{o_conf:.1%}")
    axes[0].axis('off')

    axes[1].imshow(prep(adv_t))
    axes[1].set_title(f"Adversarial\n{labels[adv_idx][:16]}\n{adv_conf:.1%}")
    axes[1].axis('off')

    # Show a noisy sample at each sigma level alongside the prediction
    for i, s in enumerate(SIGMA_VALUES):
        label_idx, vote_frac, restored = rs_results[s]
        color = 'green' if restored else 'red'
        status = "RESTORED" if restored else "still fooled"

        # Show one noisy sample so the viewer sees what the model sees
        sample_noise = torch.randn_like(adv_t) * s
        noisy_display = torch.clamp(adv_t + sample_noise, 0, 1)

        axes[2 + i].imshow(prep(noisy_display))
        axes[2 + i].set_title(f"RS σ={s}\n{labels[label_idx][:16]}\nvote: {vote_frac:.0%}", color=color)
        axes[2 + i].set_xlabel(status, color=color, fontweight='bold')
        axes[2 + i].axis('off')

    plt.suptitle(f"Randomized Smoothing Defense — {os.path.basename(filename)}", fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_path = f"rs_defense_{os.path.basename(filename)}.png"
    plt.savefig(out_path)
    plt.show()
    print(f"  [>] Saved plot: {out_path}")

# --- 7. MAIN ---
if __name__ == "__main__":
    print("[*] Loading model...")
    raw_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device).eval()
    model = NormalizedModel(raw_model)

    labels = requests.get("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt").text.splitlines()

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

    print(f"[*] Found {len(img_files)} image(s). Running attack + defense...\n")
    print(f"[*] Note: RS uses {N_SAMPLES} samples per prediction — this takes a moment.\n")

    for f_path in img_files:
        try:
            print(f"[*] Processing: {os.path.basename(f_path)}")
            x = transform(Image.open(f_path).convert("RGB")).unsqueeze(0).to(device)
            adv_x = projected_gradient_descent(model, x, EPS, EPS_ITER, STEPS, np.inf)

            with torch.no_grad():
                o_conf,   o_idx   = torch.max(torch.softmax(model(x),     1), 1)
                adv_conf, adv_idx = torch.max(torch.softmax(model(adv_x), 1), 1)

            print(f"  [+] Attack: {labels[o_idx.item()]} -> {labels[adv_idx.item()]} (conf: {adv_conf.item():.1%})")

            rs_results = {}
            for s in SIGMA_VALUES:
                print(f"  [~] Running RS σ={s} ({N_SAMPLES} samples)...", end=" ", flush=True)
                pred_idx, vote_frac = randomized_smoothing_predict(model, adv_x, s, N_SAMPLES)
                restored = (pred_idx == o_idx.item())
                rs_results[s] = (pred_idx, vote_frac, restored)
                status = "RESTORED" if restored else "still fooled"
                print(f"{labels[pred_idx][:20]:<20} vote: {vote_frac:.0%}  [{status}]")

            log_result(f_path, labels[o_idx.item()], labels[adv_idx.item()],
                       o_conf.item(), adv_conf.item(), rs_results)
            plot_results(x, adv_x, labels, o_idx.item(), adv_idx.item(),
                         o_conf.item(), adv_conf.item(), rs_results, f_path)

        except Exception as e:
            print(f"[!] Error on {f_path}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n[*] Done. Results saved to {CSV_FILENAME}")
