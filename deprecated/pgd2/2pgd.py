import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
import PIL.Image as Image
import requests
import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob

# --- 1. SETUP ---
device = torch.device("cpu")
IMAGE_FOLDER = "./my_images"

# --- 2. PGD UTILITIES ---
def fast_gradient_method(model_fn, x, eps, norm, clip_min=None, clip_max=1.0, y=None, targeted=False):
    x_adv = x.clone().detach().requires_grad_(True)
    logits = model_fn(x_adv)

    if y is None:
        y = torch.argmax(logits, 1)

    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, y)
    if targeted:
        loss = -loss

    loss.backward()
    grad = x_adv.grad.data

    if norm == np.inf:
        grad = torch.sign(grad)
    elif norm == 2:
        grad = grad / (torch.norm(grad, p=2, dim=(1,2,3), keepdim=True) + 1e-12)

    x_adv = x_adv + eps * grad
    if clip_min is not None or clip_max is not None:
        x_adv = torch.clamp(x_adv, clip_min, clip_max)
    return x_adv.detach()

def clip_eta(eta, norm, eps):
    if norm == np.inf:
        return torch.clamp(eta, -eps, eps)
    elif norm == 2:
        avoid_zero = 1e-12
        norm_val = torch.norm(eta, p=2, dim=(1,2,3), keepdim=True)
        return eta * torch.min(torch.ones_like(norm_val), eps / (norm_val + avoid_zero))
    return eta

def projected_gradient_descent(model_fn, x, eps, eps_iter, nb_iter, norm, clip_min=0, clip_max=1, y=None, targeted=False, rand_init=True):
    # Determine the class to track (original prediction)
    if y is None:
        with torch.no_grad():
            logits = model_fn(x)
            y = torch.argmax(logits, 1)

    if rand_init:
        eta = torch.zeros_like(x).uniform_(-eps, eps)
    else:
        eta = torch.zeros_like(x)

    eta = clip_eta(eta, norm, eps)
    adv_x = torch.clamp(x + eta, clip_min, clip_max)

    conf_history = []

    for i in range(nb_iter):
        # Track confidence of the original class before this step
        with torch.no_grad():
            logits = model_fn(adv_x)
            probs = torch.softmax(logits, 1)
            conf_history.append(probs[0, y].item())

        adv_x = fast_gradient_method(model_fn, adv_x, eps_iter, norm, clip_min, clip_max, y, targeted)
        eta = clip_eta(adv_x - x, norm, eps)
        adv_x = torch.clamp(x + eta, clip_min, clip_max)

    # Track final confidence
    with torch.no_grad():
        logits = model_fn(adv_x)
        probs = torch.softmax(logits, 1)
        conf_history.append(probs[0, y].item())

    return adv_x, conf_history

# --- 3. DATA WRAPPER ---
class NormalizedModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    def forward(self, x):
        return self.model((x - self.mean) / self.std)

# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    print("[*] Loading Model...")
    raw_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device).eval()
    model = NormalizedModel(raw_model)

    # Load ImageNet labels
    labels = requests.get("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt").text.splitlines()

    # Find images
    img_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        img_files.extend(glob(os.path.join(IMAGE_FOLDER, ext)))

    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])

    # PGD Params
    EPS = 8/255
    EPS_ITER = 2/255
    STEPS = 20

    print(f"[*] Found {len(img_files)} images. Starting PGD...")

    for f_path in img_files:
        try:
            img_raw = Image.open(f_path).convert("RGB")
            x = transform(img_raw).unsqueeze(0).to(device)

            # Run Attack and get confidence history
            adv_x, conf_history = projected_gradient_descent(model, x, EPS, EPS_ITER, STEPS, np.inf)

            with torch.no_grad():
                o_logits = model(x)
                a_logits = model(adv_x)

                o_conf, o_idx = torch.max(torch.softmax(o_logits, 1), 1)
                a_conf, a_idx = torch.max(torch.softmax(a_logits, 1), 1)
                l2_dist = torch.norm(x - adv_x, p=2).item()

                o_label = labels[o_idx.item()]
                a_label = labels[a_idx.item()]

                print(f"[+] {os.path.basename(f_path)}: {o_label} -> {a_label} (L2: {l2_dist:.2f})")

                # Updated Plotting: 4 subplots
                fig, ax = plt.subplots(1, 4, figsize=(20, 5))
                
                # 1. Original Image
                ax[0].imshow(x.squeeze().permute(1,2,0))
                ax[0].set_title(f"Orig: {o_label[:15]}\nConf: {o_conf.item():.1%}")
                ax[0].axis('off')

                # 2. Adversarial Image
                ax[1].imshow(adv_x.squeeze().permute(1,2,0))
                ax[1].set_title(f"Adv: {a_label[:15]}\nConf: {a_conf.item():.1%}")
                ax[1].axis('off')

                # 3. Perturbation (Amplified)
                diff = (adv_x - x).squeeze().permute(1,2,0).cpu().numpy()
                ax[2].imshow(np.clip((diff * 10) + 0.5, 0, 1))
                ax[2].set_title("Perturbation (10x)")
                ax[2].axis('off')

                # 4. Confidence Curve
                ax[3].plot(conf_history, color='red', marker='o', markersize=4)
                ax[3].set_title("Confidence of Original Class")
                ax[3].set_xlabel("Iteration")
                ax[3].set_ylabel("Confidence")
                ax[3].set_ylim(0, 1.05)
                ax[3].grid(True, linestyle='--', alpha=0.6)

                plt.tight_layout()
                plt.show()

        except Exception as e:
            print(f"[!] Error on {f_path}: {e}")
