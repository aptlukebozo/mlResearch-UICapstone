import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
import PIL.Image as Image
import requests
from io import BytesIO

# --- 1. SETUP & HARDWARE ---
# T490 Optimization: Use CPU to avoid MX250 memory crashes
device = torch.device("cpu")

# --- 2. THE NORMALIZATION WRAPPER ---
class NormalizedModel(nn.Module):
    """
    Wraps a model to include ImageNet normalization. 
    This allows the C&W attack to work on raw [0, 1] pixels.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        return self.model((x - self.mean) / self.std)

# --- 3. YOUR CARLINI-WAGNER FUNCTION ---
def carlini_wagner_l2(model_fn, x, n_classes, y=None, targeted=False, lr=5e-3, confidence=0, 
                      clip_min=0, clip_max=1, initial_const=1e-2, binary_search_steps=5, max_iterations=1000):
    # (Using the logic from your provided code)
    def compare(pred, label, is_logits=False):
        if is_logits:
            pred_copy = pred.clone().detach()
            pred_copy[label] += -confidence if targeted else confidence
            pred = torch.argmax(pred_copy)
        return pred == label if targeted else pred != label

    if y is None:
        pred = model_fn(x)
        y = torch.argmax(pred, 1)

    lower_bound = [0.0] * len(x)
    upper_bound = [1e10] * len(x)
    const = x.new_ones(len(x), 1) * initial_const
    o_bestl2 = [float("inf")] * len(x)
    o_bestscore = [-1.0] * len(x)
    ox = x.clone().detach()
    o_bestattack = x.clone().detach()

    # Tanh-space mapping
    x = (x - clip_min) / (clip_max - clip_min)
    x = torch.clamp(x, 0, 1) * 2 - 1
    x = torch.arctanh(x * 0.999999)

    modifier = torch.zeros_like(x, requires_grad=True)
    y_onehot = torch.nn.functional.one_hot(y, n_classes).to(torch.float).to(device)
    optimizer = torch.optim.Adam([modifier], lr=lr)

    for outer_step in range(binary_search_steps):
        bestl2 = [float("inf")] * len(x)
        bestscore = [-1.0] * len(x)

        for i in range(max_iterations):
            new_x = (torch.tanh(modifier + x) + 1) / 2
            new_x = new_x * (clip_max - clip_min) + clip_min
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
                succeeded = compare(logits_n, y[n], is_logits=True)
                if l2_n < o_bestl2[n] and succeeded:
                    o_bestl2[n], o_bestscore[n], o_bestattack[n] = l2_n, torch.argmax(logits_n), new_x_n
                if l2_n < bestl2[n] and succeeded:
                    bestl2[n], bestscore[n] = l2_n, torch.argmax(logits_n)

        for n in range(len(x)):
            if compare(bestscore[n], y[n]) and bestscore[n] != -1:
                upper_bound[n] = min(upper_bound[n], const[n])
                if upper_bound[n] < 1e9: const[n] = (lower_bound[n] + upper_bound[n]) / 2
            else:
                lower_bound[n] = max(lower_bound[n], const[n])
                const[n] = (lower_bound[n] + upper_bound[n]) / 2 if upper_bound[n] < 1e9 else const[n] * 10

    return o_bestattack.detach()

# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    print("[*] Loading ResNet-18...")
    raw_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device).eval()
    model = NormalizedModel(raw_model)

    # 1. Download a Sample Image (Golden Retriever)
    url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    img = Image.open(BytesIO(requests.get(url).content))
    
    # 2. Preprocess
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    img_t = transform(img).unsqueeze(0).to(device)

    # 3. Run Attack (LITE PARAMS FOR T490)
    print("[*] Starting Attack. This will take ~30-60 seconds...")
    adv_img = carlini_wagner_l2(
        model, img_t, n_classes=1000, 
        max_iterations=50,       # Keep low for laptop speed
        binary_search_steps=2,   # Keep low for laptop speed
        lr=5e-2,                 # Slightly higher LR helps with fewer steps
        initial_const=1.0        # Forces quicker convergence on laptop
    )

    # 4. Results
    with torch.no_grad():
        orig_pred = torch.argmax(model(img_t), 1).item()
        adv_pred = torch.argmax(model(adv_img), 1).item()
        print(f"\n[+] Results:")
        print(f"    Original ID: {orig_pred}")
        print(f"    Adversarial ID: {adv_pred}")
        print(f"    Status: {'SUCCESS' if orig_pred != adv_pred else 'FAILED (Increase iterations)'}")
