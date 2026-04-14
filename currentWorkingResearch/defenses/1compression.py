import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import io
import requests
from glob import glob

# --- 1. SETTINGS ---
device = torch.device("cpu")
IMAGE_FOLDER = "../pgd/my_images"
CSV_FILENAME = "compression_defense_results.csv"
JPEG_QUALITIES = [25, 50, 75]  # Quality levels to test

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
    x_adv = x_adv + eps * grad
    return torch.clamp(x_adv, clip_min, clip_max).detach()

def projected_gradient_descent(model_fn, x, eps, eps_iter, nb_iter, norm, clip_min=0.0, clip_max=1.0, y=None, targeted=False, rand_init=True):
    if y is None:
        with torch.no_grad():
            y = torch.argmax(model_fn(x), 1)

    eta = torch.zeros_like(x).uniform_(-eps, eps) if rand_init else torch.zeros_like(x)
    if norm == np.inf:
        eta = torch.clamp(eta, -eps, eps)
    adv_x = torch.clamp(x + eta, clip_min, clip_max)

    for _ in range(nb_iter):
        adv_x = fast_gradient_method(model_fn, adv_x, eps_iter, norm, clip_min, clip_max, y, targeted)
        eta = torch.clamp(adv_x - x, -eps, eps)
        adv_x = torch.clamp(x + eta, clip_min, clip_max)

    return adv_x

# --- 4. JPEG COMPRESSION DEFENSE ---
def jpeg_compress(tensor, quality):
    """Apply JPEG compression to a [1, C, H, W] tensor and return the same shape."""
    img = tensor.squeeze(0).permute(1, 2, 0).numpy()
    img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)

    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer).convert("RGB")

    compressed_t = torch.from_numpy(np.array(compressed).astype(np.float32) / 255.0)
    return compressed_t.permute(2, 0, 1).unsqueeze(0)

# --- 5. LOGGING ---
def log_result(filename, orig_label, adv_label, results_by_quality, o_conf, adv_conf):
    file_exists = os.path.isfile(CSV_FILENAME)
    with open(CSV_FILENAME, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            header = ["Filename", "Original", "Adv_Label", "O_Conf", "Adv_Conf"]
            for q in JPEG_QUALITIES:
                header += [f"Q{q}_Label", f"Q{q}_Conf", f"Q{q}_Restored"]
            writer.writerow(header)

        row = [os.path.basename(filename), orig_label, adv_label, f"{o_conf:.4f}", f"{adv_conf:.4f}"]
        for q in JPEG_QUALITIES:
            label, conf, restored = results_by_quality[q]
            row += [label, f"{conf:.4f}", restored]
        writer.writerow(row)

# --- 6. PLOTTING ---
def plot_results(orig_t, adv_t, compressed_by_quality, labels, o_idx, adv_idx, o_conf, adv_conf, results_by_quality, filename):
    def prep(t):
        return np.clip(t.squeeze(0).permute(1, 2, 0).numpy(), 0, 1)

    n_cols = 2 + len(JPEG_QUALITIES)
    fig, ax = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    ax[0].imshow(prep(orig_t))
    ax[0].set_title(f"Original\n{labels[o_idx][:18]}\n{o_conf:.1%}")
    ax[0].axis('off')

    ax[1].imshow(prep(adv_t))
    ax[1].set_title(f"Adversarial (PGD)\n{labels[adv_idx][:18]}\n{adv_conf:.1%}")
    ax[1].axis('off')

    for i, q in enumerate(JPEG_QUALITIES):
        label, conf, restored = results_by_quality[q]
        color = 'green' if restored else 'red'
        status = "RESTORED" if restored else "still fooled"
        ax[2 + i].imshow(prep(compressed_by_quality[q]))
        ax[2 + i].set_title(f"JPEG Q={q}\n{labels[int(label)][:18]}\n{conf:.1%}", color=color)
        ax[2 + i].set_xlabel(status, color=color, fontweight='bold')
        ax[2 + i].axis('off')

    plt.suptitle(f"JPEG Compression Defense — {os.path.basename(filename)}", fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_path = f"defense_{os.path.basename(filename)}.png"
    plt.savefig(out_path)
    plt.show()
    print(f"  [>] Saved plot: {out_path}")

# --- 7. MAIN ---
if __name__ == "__main__":
    print("[*] Loading model...")
    raw_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device).eval()
    model = NormalizedModel(raw_model)

    labels = requests.get("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt").text.splitlines()

    img_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        img_files.extend(glob(os.path.join(IMAGE_FOLDER, ext)))
    img_files = sorted(img_files)

    if not img_files:
        print(f"[!] No images found in {IMAGE_FOLDER}")
        exit()

    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])

    # PGD attack params (same as 2pgd.py)
    EPS      = 8 / 255
    EPS_ITER = 2 / 255
    STEPS    = 20

    print(f"[*] Found {len(img_files)} image(s). Running attack + defense...\n")

    for f_path in img_files:
        try:
            print(f"[*] Processing: {os.path.basename(f_path)}")

            img_raw = Image.open(f_path).convert("RGB")
            x = transform(img_raw).unsqueeze(0).to(device)

            # Generate adversarial example
            adv_x = projected_gradient_descent(model, x, EPS, EPS_ITER, STEPS, np.inf)

            with torch.no_grad():
                o_logits  = model(x)
                adv_logits = model(adv_x)
                o_conf,   o_idx   = torch.max(torch.softmax(o_logits,   1), 1)
                adv_conf, adv_idx = torch.max(torch.softmax(adv_logits, 1), 1)

            print(f"  [+] Attack: {labels[o_idx.item()]} -> {labels[adv_idx.item()]} (adv conf: {adv_conf.item():.1%})")

            # Apply JPEG compression at each quality level
            results_by_quality   = {}
            compressed_by_quality = {}

            for q in JPEG_QUALITIES:
                comp = jpeg_compress(adv_x, q)
                with torch.no_grad():
                    c_logits = model(comp)
                    c_conf, c_idx = torch.max(torch.softmax(c_logits, 1), 1)
                restored = (c_idx.item() == o_idx.item())
                results_by_quality[q]    = (c_idx.item(), c_conf.item(), restored)
                compressed_by_quality[q] = comp
                status = "RESTORED" if restored else "still fooled"
                print(f"  [>] Q={q:>3}: {labels[c_idx.item()][:20]:<20} {c_conf.item():.1%}  [{status}]")

            log_result(f_path, labels[o_idx.item()], labels[adv_idx.item()], results_by_quality, o_conf.item(), adv_conf.item())
            plot_results(x, adv_x, compressed_by_quality, labels, o_idx.item(), adv_idx.item(), o_conf.item(), adv_conf.item(), results_by_quality, f_path)

        except Exception as e:
            print(f"[!] Error on {f_path}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n[*] Done. Results saved to {CSV_FILENAME}")
