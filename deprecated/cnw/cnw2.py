import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
import PIL.Image as Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np

# --- 1. SETUP & HARDWARE ---
device = torch.device("cpu") # Optimized for T490 stability

# --- 2. IMAGE UTILITIES ---
def get_imagenet_data():
    """Downloads a sample image and the ImageNet labels."""
    # Standard Golden Retriever sample
    img_url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    response = requests.get(img_url)
    img = Image.open(BytesIO(response.content))
    
    # Class labels
    labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    labels = requests.get(labels_url).text.splitlines()
    return img, labels

class NormalizedModel(nn.Module):
    """Wraps model to handle ImageNet normalization internally."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        return self.model((x - self.mean) / self.std)

# --- 3. THE CARLINI & WAGNER L2 ATTACK ---
def carlini_wagner_l2(model_fn, x, n_classes, y=None, targeted=False, lr=5e-3, confidence=0, 
                      clip_min=0, clip_max=1, initial_const=1e-2, binary_search_steps=5, max_iterations=1000):
    
    def compare(pred, label, is_logits=False):
        if is_logits:
            pred_copy = pred.clone().detach()
            pred_copy[label] += -confidence if targeted else confidence
            pred = torch.argmax(pred_copy)
        return pred == label if targeted else pred != label

    if y is None:
        pred = model_fn(x)
        y = torch.argmax(pred, 1)

    # Optimization Setup
    lower_bound = [0.0] * len(x)
    upper_bound = [1e10] * len(x)
    const = x.new_ones(len(x), 1) * initial_const
    o_bestl2, o_bestscore, o_bestattack = [float("inf")] * len(x), [-1.0] * len(x), x.clone().detach()
    ox = x.clone().detach()

    # Map to tanh-space
    x_tanh = torch.arctanh(((x - clip_min) / (clip_max - clip_min) * 2 - 1) * 0.999999)
    modifier = torch.zeros_like(x_tanh, requires_grad=True)
    optimizer = torch.optim.Adam([modifier], lr=lr)
    y_onehot = torch.nn.functional.one_hot(y, n_classes).to(torch.float).to(x.device)

    for outer_step in range(binary_search_steps):
        bestl2, bestscore = [float("inf")] * len(x), [-1.0] * len(x)
        for i in range(max_iterations):
            new_x = (torch.tanh(modifier + x_tanh) + 1) / 2 * (clip_max - clip_min) + clip_min
            logits = model_fn(new_x)
            
            real = torch.sum(y_onehot * logits, 1)
            other, _ = torch.max((1 - y_onehot) * logits - y_onehot * 1e4, 1)

            optimizer.zero_grad()
            f = torch.max(((other - real) if targeted else (real - other)) + confidence, torch.tensor(0.0).to(x.device))
            l2 = torch.pow(new_x - ox, 2).sum(list(range(len(x.size())))[1:])
            loss = (const * f + l2).sum()
            loss.backward()
            optimizer.step()

            for n, (l2_n, logits_n, new_x_n) in enumerate(zip(l2, logits, new_x)):
                if l2_n < o_bestl2[n] and compare(logits_n, y[n], is_logits=True):
                    o_bestl2[n], o_bestscore[n], o_bestattack[n] = l2_n, torch.argmax(logits_n), new_x_n
                if l2_n < bestl2[n] and compare(logits_n, y[n], is_logits=True):
                    bestl2[n], bestscore[n] = l2_n, torch.argmax(logits_n)

        # Binary search for 'const'
        for n in range(len(x)):
            if compare(bestscore[n], y[n]) and bestscore[n] != -1:
                upper_bound[n] = min(upper_bound[n], const[n])
                if upper_bound[n] < 1e9: const[n] = (lower_bound[n] + upper_bound[n]) / 2
            else:
                lower_bound[n] = max(lower_bound[n], const[n])
                const[n] = (lower_bound[n] + upper_bound[n]) / 2 if upper_bound[n] < 1e9 else const[n] * 10
    return o_bestattack.detach()

# --- 4. VISUALIZATION ---
def show_results(orig_t, adv_t, labels, orig_idx, adv_idx):
    def prep(t): return t.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    
    orig, adv = prep(orig_t), prep(adv_t)
    noise = adv - orig
    # Amplify noise for visibility (50x)
    noise_vis = np.clip((noise * 50) + 0.5, 0, 1)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(orig); ax[0].set_title(f"Original: {labels[orig_idx]}"); ax[0].axis('off')
    ax[1].imshow(adv); ax[1].set_title(f"Adversarial: {labels[adv_idx]}"); ax[1].axis('off')
    ax[2].imshow(noise_vis); ax[2].set_title("Noise (50x Gain)"); ax[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    fig.savefig("research_result.png")

# --- 5. EXECUTION ---
if __name__ == "__main__":
    print("[*] Initializing Laptop-Friendly Research Environment...")
    raw_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device).eval()
    model = NormalizedModel(raw_model)
    img_raw, labels = get_imagenet_data()
    
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    img_t = transform(img_raw).unsqueeze(0).to(device)

    # T490 Settings: High LR and low Iterations for quick testing
    adv_img = carlini_wagner_l2(model, img_t, 1000, max_iterations=100, binary_search_steps=3, lr=0.05, initial_const=1.0)

    with torch.no_grad():
        orig_pred = torch.argmax(model(img_t), 1).item()
        adv_pred = torch.argmax(model(adv_img), 1).item()
        print(f"\n[+] SUCCESS: Label flipped from {labels[orig_pred]} to {labels[adv_pred]}")
        show_results(img_t, adv_img, labels, orig_pred, adv_pred)
