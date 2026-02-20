import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
import PIL.Image as Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import csv
import os

# --- 1. SETTINGS & HARDWARE ---
device = torch.device("cpu") 
NUM_TEST_IMAGES = 3
CSV_FILENAME = "research_results.csv"

# --- 2. STABLE DATA LOADING ---
def get_research_samples():
    """Downloads 3 stable ImageNet samples with browser-like headers."""
    # These URLs are from official PyTorch/ImageNet mirrors for stability
    urls = [
        "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",      # Golden Retriever
        "https://raw.githubusercontent.com/pytorch/examples/main/imagenet/README.md", # (Dummy check link)
        "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"       # Using Dog twice if others fail
    ]
    # Replaced dead links with reliable PyTorch hub images
    urls = [
        "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg", 
        "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
        "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    ]
    
    headers = {'User-Agent': 'Mozilla/5.0'}
    labels = requests.get("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt").text.splitlines()
    
    images = []
    print("[*] Downloading samples...")
    for url in urls:
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        images.append(Image.open(BytesIO(resp.content)).convert("RGB"))
    return images, labels

class NormalizedModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        return self.model((x - self.mean) / self.std)

# --- 3. THE C&W L2 ATTACK ---
def carlini_wagner_l2(model_fn, x, n_classes, y=None, targeted=False, lr=5e-3, confidence=0, 
                      clip_min=0, clip_max=1, initial_const=1e-2, binary_search_steps=5, max_iterations=1000):
    
    def compare(pred, label, is_logits=False):
        if is_logits:
            pred_copy = pred.clone().detach()
            pred_copy[label] += -confidence if targeted else confidence
            pred = torch.argmax(pred_copy)
        return pred == label if targeted else pred != label

    if y is None:
        y = torch.argmax(model_fn(x), 1)

    lower_bound, upper_bound = [0.0] * len(x), [1e10] * len(x)
    const = x.new_ones(len(x), 1) * initial_const
    o_bestl2, o_bestscore, o_bestattack = [float("inf")] * len(x), [-1.0] * len(x), x.clone().detach()
    ox = x.clone().detach()

    # Tanh-space
    x_tanh = torch.arctanh(((x - clip_min) / (clip_max - clip_min) * 2 - 1) * 0.999999)
    modifier = torch.zeros_like(x_tanh, requires_grad=True)
    optimizer = torch.optim.Adam([modifier], lr=lr)
    y_onehot = torch.nn.functional.one_hot(y, n_classes).to(torch.float).to(device)

    for outer_step in range(binary_search_steps):
        bestl2, bestscore = [float("inf")] * len(x), [-1.0] * len(x)
        for i in range(max_iterations):
            new_x = (torch.tanh(modifier + x_tanh) + 1) / 2 * (clip_max - clip_min) + clip_min
            logits = model_fn(new_x)
            
            real = torch.sum(y_onehot * logits, 1)
            other, _ = torch.max((1 - y_onehot) * logits - y_onehot * 1e4, 1)

            optimizer.zero_grad()
            f = torch.max(((other - real) if targeted else (real - other)) + confidence, torch.tensor(0.0).to(device))
            l2 = torch.pow(new_x - ox, 2).sum(list(range(len(x.size())))[1:])
            loss = (const * f + l2).sum()
            loss.backward()
            optimizer.step()

            for n, (l2_n, logits_n, new_x_n) in enumerate(zip(l2, logits, new_x)):
                if l2_n < o_bestl2[n] and compare(logits_n, y[n], is_logits=True):
                    o_bestl2[n], o_bestscore[n], o_bestattack[n] = l2_n, torch.argmax(logits_n), new_x_n
                if l2_n < bestl2[n] and compare(logits_n, y[n], is_logits=True):
                    bestl2[n], bestscore[n] = l2_n, torch.argmax(logits_n)
    
    return o_bestattack.detach()

# --- 4. VISUALIZATION & LOGGING ---
def log_and_show(orig_t, adv_t, labels, o_idx, a_idx, o_conf, a_conf, l2_dist, i):
    # Log to CSV
    file_exists = os.path.isfile(CSV_FILENAME)
    with open(CSV_FILENAME, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Index", "Original_Label", "Adv_Label", "Orig_Conf", "Adv_Conf", "L2_Dist", "Success"])
        writer.writerow([i, labels[o_idx], labels[a_idx], f"{o_conf:.4f}", f"{a_conf:.4f}", f"{l2_dist:.4f}", o_idx != a_idx])

    # Show Plot
    def prep(t): return t.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    orig, adv = prep(orig_t), prep(adv_t)
    noise_vis = np.clip(((adv - orig) * 50) + 0.5, 0, 1)

    fig, ax = plt.subplots(1, 3, figsize=(15, 6))
    ax[0].imshow(orig); ax[0].set_title(f"Orig: {labels[o_idx]}\nConf: {o_conf:.2%}"); ax[0].axis('off')
    ax[1].imshow(adv); ax[1].set_title(f"Adv: {labels[a_idx]}\nConf: {a_conf:.2%}"); ax[1].axis('off')
    ax[2].imshow(noise_vis); ax[2].set_title("50x Amplified Noise"); ax[2].axis('off')
    plt.tight_layout()
    plt.show()

# --- 5. MAIN ---
if __name__ == "__main__":
    raw_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device).eval()
    model = NormalizedModel(raw_model)
    raw_imgs, labels = get_research_samples()
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])

    print(f"[*] Starting attack on {NUM_TEST_IMAGES} images...")
    for i in range(NUM_TEST_IMAGES):
        img_t = transform(raw_imgs[i]).unsqueeze(0).to(device)
        
        # Attack
        adv_img = carlini_wagner_l2(model, img_t, 1000, max_iterations=80, binary_search_steps=3, lr=0.05, initial_const=1.0)
        
        # Stats
        with torch.no_grad():
            o_prob = torch.softmax(model(img_t), 1)
            a_prob = torch.softmax(model(adv_img), 1)
            o_conf, o_idx = torch.max(o_prob, 1)
            a_conf, a_idx = torch.max(a_prob, 1)
            l2 = torch.norm(img_t - adv_img, p=2).item()

            print(f"[+] Image {i+1}: {labels[o_idx.item()]} -> {labels[a_idx.item()]} (L2: {l2:.2f})")
            log_and_show(img_t, adv_img, labels, o_idx.item(), a_idx.item(), o_conf.item(), a_conf.item(), l2, i+1)
