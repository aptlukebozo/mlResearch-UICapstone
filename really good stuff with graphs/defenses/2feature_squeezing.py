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
CSV_FILENAME = "feature_squeezing_results.csv"
BIT_DEPTHS = [1, 2, 4]       # Bit-depth reduction levels to test
SMOOTH_KERNEL = 3             # Gaussian blur kernel size

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

# --- 4. FEATURE SQUEEZING DEFENSES ---
def bit_depth_reduce(tensor, bits):
    """Reduce pixel precision to `bits` bits per channel."""
    levels = 2 ** bits - 1
    return torch.round(tensor * levels) / levels

def gaussian_smooth(tensor, kernel_size=3):
    """Apply Gaussian blur with the given kernel size."""
    # kernel_size must be odd
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    blurred = TF.gaussian_blur(tensor.squeeze(0), kernel_size=[k, k], sigma=1.0)
    return blurred.unsqueeze(0)

def combined_squeeze(tensor, bits, kernel_size=3):
    """Bit-depth reduction followed by Gaussian smoothing."""
    return gaussian_smooth(bit_depth_reduce(tensor, bits), kernel_size)

# --- 5. LOGGING ---
def log_result(filename, orig_label, adv_label, o_conf, adv_conf, bit_results, smooth_result, combo_results):
    file_exists = os.path.isfile(CSV_FILENAME)
    with open(CSV_FILENAME, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            header = ["Filename", "Original", "Adversarial", "O_Conf", "Adv_Conf", "Smooth_Label", "Smooth_Conf", "Smooth_Restored"]
            for b in BIT_DEPTHS:
                header += [f"Bit{b}_Label", f"Bit{b}_Conf", f"Bit{b}_Restored"]
            for b in BIT_DEPTHS:
                header += [f"Combo{b}_Label", f"Combo{b}_Conf", f"Combo{b}_Restored"]
            writer.writerow(header)

        s_label, s_conf, s_restored = smooth_result
        row = [os.path.basename(filename), orig_label, adv_label, f"{o_conf:.4f}", f"{adv_conf:.4f}",
               s_label, f"{s_conf:.4f}", s_restored]
        for b in BIT_DEPTHS:
            label, conf, restored = bit_results[b]
            row += [label, f"{conf:.4f}", restored]
        for b in BIT_DEPTHS:
            label, conf, restored = combo_results[b]
            row += [label, f"{conf:.4f}", restored]
        writer.writerow(row)

# --- 6. PLOTTING ---
def plot_results(orig_t, adv_t, squeezed_tensors, labels, o_idx, adv_idx, o_conf, adv_conf, all_results, filename):
    def prep(t):
        return np.clip(t.squeeze(0).permute(1, 2, 0).detach().numpy(), 0, 1)

    titles_and_tensors = [
        (f"Original\n{labels[o_idx][:16]}\n{o_conf:.1%}", orig_t, None),
        (f"Adversarial\n{labels[adv_idx][:16]}\n{adv_conf:.1%}", adv_t, None),
    ]
    for key, t in squeezed_tensors.items():
        label_idx, conf, restored = all_results[key]
        color = 'green' if restored else 'red'
        titles_and_tensors.append((f"{key}\n{labels[label_idx][:16]}\n{conf:.1%}", t, color))

    n = len(titles_and_tensors)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))

    for ax, (title, tensor, color) in zip(axes, titles_and_tensors):
        ax.imshow(prep(tensor))
        ax.set_title(title, color=color if color else 'black')
        if color:
            status = "RESTORED" if color == 'green' else "still fooled"
            ax.set_xlabel(status, color=color, fontweight='bold')
        ax.axis('off')

    plt.suptitle(f"Feature Squeezing Defense — {os.path.basename(filename)}", fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_path = f"fs_defense_{os.path.basename(filename)}.png"
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

    for f_path in img_files:
        try:
            print(f"[*] Processing: {os.path.basename(f_path)}")
            x = transform(Image.open(f_path).convert("RGB")).unsqueeze(0).to(device)
            adv_x = projected_gradient_descent(model, x, EPS, EPS_ITER, STEPS, np.inf)

            with torch.no_grad():
                o_conf,   o_idx   = torch.max(torch.softmax(model(x),     1), 1)
                adv_conf, adv_idx = torch.max(torch.softmax(model(adv_x), 1), 1)

            print(f"  [+] Attack: {labels[o_idx.item()]} -> {labels[adv_idx.item()]} (conf: {adv_conf.item():.1%})")

            def evaluate(t):
                with torch.no_grad():
                    conf, idx = torch.max(torch.softmax(model(t), 1), 1)
                restored = (idx.item() == o_idx.item())
                return idx.item(), conf.item(), restored

            # Gaussian smooth only
            smooth_t = gaussian_smooth(adv_x, SMOOTH_KERNEL)
            smooth_result = evaluate(smooth_t)
            status = "RESTORED" if smooth_result[2] else "still fooled"
            print(f"  [>] Smooth only:    {labels[smooth_result[0]][:20]:<20} {smooth_result[1]:.1%}  [{status}]")

            # Bit-depth reduction only
            bit_results = {}
            bit_tensors = {}
            for b in BIT_DEPTHS:
                t = bit_depth_reduce(adv_x, b)
                result = evaluate(t)
                bit_results[b] = result
                bit_tensors[b] = t
                status = "RESTORED" if result[2] else "still fooled"
                print(f"  [>] Bit-depth {b}bit:  {labels[result[0]][:20]:<20} {result[1]:.1%}  [{status}]")

            # Combined: bit-depth + smooth
            combo_results = {}
            combo_tensors = {}
            for b in BIT_DEPTHS:
                t = combined_squeeze(adv_x, b, SMOOTH_KERNEL)
                result = evaluate(t)
                combo_results[b] = result
                combo_tensors[b] = t
                status = "RESTORED" if result[2] else "still fooled"
                print(f"  [>] Combo  {b}bit+sm: {labels[result[0]][:20]:<20} {result[1]:.1%}  [{status}]")

            # Build ordered dict for plotting
            squeezed_tensors = {"Smooth": smooth_t}
            squeezed_tensors.update({f"Bit-{b}": bit_tensors[b] for b in BIT_DEPTHS})
            squeezed_tensors.update({f"Bit-{b}+Smooth": combo_tensors[b] for b in BIT_DEPTHS})

            all_results = {"Smooth": smooth_result}
            all_results.update({f"Bit-{b}": bit_results[b] for b in BIT_DEPTHS})
            all_results.update({f"Bit-{b}+Smooth": combo_results[b] for b in BIT_DEPTHS})

            log_result(f_path, labels[o_idx.item()], labels[adv_idx.item()],
                       o_conf.item(), adv_conf.item(), bit_results, smooth_result, combo_results)
            plot_results(x, adv_x, squeezed_tensors, labels,
                         o_idx.item(), adv_idx.item(), o_conf.item(), adv_conf.item(), all_results, f_path)

        except Exception as e:
            print(f"[!] Error on {f_path}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n[*] Done. Results saved to {CSV_FILENAME}")
