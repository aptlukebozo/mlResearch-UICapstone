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
IMAGE_FOLDER = "./my_images"  
CSV_FILENAME = "custom_research_results.csv"

# --- 2. DATA LOADING & WRAPPER ---
def get_local_images(folder):
    extensions = ['*.jpg', '*.jpeg', '*.png']
    files = []
    for ext in extensions:
        files.extend(glob(os.path.join(folder, ext)))
        files.extend(glob(os.path.join(folder, ext.upper())))

    labels = requests.get("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt").text.splitlines()
    return sorted(files), labels

class NormalizedModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    def forward(self, x):
        return self.model((x - self.mean) / self.std)

# --- 3. THE C&W L2 ATTACK (UPDATED TO TRACK CONFIDENCE) ---
def carlini_wagner_l2(model_fn, x, n_classes, y=None, targeted=False, lr=5e-3, confidence=0,
                      clip_min=0, clip_max=1, initial_const=1e-2, binary_search_steps=5, max_iterations=1000):
    
    if y is None: y = torch.argmax(model_fn(x), 1)
    
    # List to store confidence of original class at every single iteration
    conf_history = []
    
    lower_bound, upper_bound = [0.0] * len(x), [1e10] * len(x)
    const = x.new_ones(len(x), 1) * initial_const
    o_bestl2, o_bestscore, o_bestattack = [float("inf")] * len(x), [-1.0] * len(x), x.clone().detach()
    ox = x.clone().detach()
    
    x_tanh = torch.arctanh(((x - clip_min) / (clip_max - clip_min) * 2 - 1) * 0.999999)
    modifier = torch.zeros_like(x_tanh, requires_grad=True)
    optimizer = torch.optim.Adam([modifier], lr=lr)
    y_onehot = torch.nn.functional.one_hot(y, n_classes).to(torch.float).to(device)

    def compare(pred, label):
        return pred == label if targeted else pred != label

    for outer_step in range(binary_search_steps):
        for i in range(max_iterations):
            new_x = (torch.tanh(modifier + x_tanh) + 1) / 2 * (clip_max - clip_min) + clip_min
            logits = model_fn(new_x)
            
            # --- Track Confidence ---
            with torch.no_grad():
                probs = torch.softmax(logits, 1)
                # Track the original class probability for the first image in batch
                conf_history.append(probs[0, y[0]].item())

            real = torch.sum(y_onehot * logits, 1)
            other, _ = torch.max((1 - y_onehot) * logits - y_onehot * 1e4, 1)
            
            optimizer.zero_grad()
            f = torch.max(((other - real) if targeted else (real - other)) + confidence, torch.tensor(0.0).to(device))
            l2 = torch.pow(new_x - ox, 2).sum(list(range(len(x.size())))[1:])
            loss = (const * f + l2).sum()
            loss.backward()
            optimizer.step()

            for n, (l2_n, logits_n, new_x_n) in enumerate(zip(l2, logits, new_x)):
                if l2_n < o_bestl2[n] and compare(torch.argmax(logits_n), y[n]):
                    o_bestl2[n], o_bestscore[n], o_bestattack[n] = l2_n, torch.argmax(logits_n), new_x_n

    return o_bestattack.detach(), conf_history

# --- 4. LOGGING & PLOTTING (UPDATED) ---
def log_and_plot(orig_t, adv_t, labels, o_idx, a_idx, o_conf, a_conf, l2, filename, conf_history):
    # CSV Log
    file_exists = os.path.isfile(CSV_FILENAME)
    with open(CSV_FILENAME, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Filename", "Original", "Adversarial", "O_Conf", "A_Conf", "L2", "Success"])
        writer.writerow([os.path.basename(filename), labels[o_idx], labels[a_idx], f"{o_conf:.4f}", f"{a_conf:.4f}", f"{l2:.4f}", o_idx != a_idx])

    # Visuals
    def prep(t): return t.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    orig, adv = prep(orig_t), prep(adv_t)
    noise = np.clip(((adv - orig) * 50) + 0.5, 0, 1) 

    fig, ax = plt.subplots(1, 4, figsize=(20, 5)) # Changed to 4 columns
    
    ax[0].imshow(orig); ax[0].set_title(f"Original: {labels[o_idx][:15]}\n{o_conf:.2%}"); ax[0].axis('off')
    ax[1].imshow(adv); ax[1].set_title(f"Adversarial: {labels[a_idx][:15]}\n{a_conf:.2%}"); ax[1].axis('off')
    ax[2].imshow(noise); ax[2].set_title("Perturbation (50x)"); ax[2].axis('off')
    
    # 4th Plot: Confidence Curve
    ax[3].plot(conf_history, color='blue', linewidth=1)
    ax[3].set_title("Original Class Confidence")
    ax[3].set_xlabel("Total Iterations")
    ax[3].set_ylabel("Prob")
    ax[3].set_ylim(0, 1.05)
    ax[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"attack_{os.path.basename(filename)}.png")
    plt.show()

# --- 5. EXECUTION ---
if __name__ == "__main__":
    raw_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device).eval()
    model = NormalizedModel(raw_model)
    img_files, labels = get_local_images(IMAGE_FOLDER)

    if not img_files:
        print(f"[!] No images found in {IMAGE_FOLDER}.")
        exit()

    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])

    print(f"[*] Found {len(img_files)} images. Starting attack loop...")
    for f in img_files:
        try:
            raw_img = Image.open(f).convert("RGB")
            img_t = transform(raw_img).unsqueeze(0).to(device)

            # Attack settings
            adv_img, conf_history = carlini_wagner_l2(model, img_t, 1000, max_iterations=100, binary_search_steps=3, lr=0.05, initial_const=1.0)

            with torch.no_grad():
                o_logits, a_logits = model(img_t), model(adv_img)
                o_prob, a_prob = torch.softmax(o_logits, 1), torch.softmax(a_logits, 1)
                o_conf, o_idx = torch.max(o_prob, 1)
                a_conf, a_idx = torch.max(a_prob, 1)
                l2_dist = torch.norm(img_t - adv_img, p=2).item()

                print(f"[+] Done: {os.path.basename(f)} ({labels[o_idx.item()]} -> {labels[a_idx.item()]})")
                log_and_plot(img_t, adv_img, labels, o_idx.item(), a_idx.item(), o_conf.item(), a_conf.item(), l2_dist, f, conf_history)
        except Exception as e:
            print(f"[!] Skipped {f}: {e}")
            import traceback
            traceback.print_exc()
