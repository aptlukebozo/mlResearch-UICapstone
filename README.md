# Adversarial Attacks & Defenses on ImageNet

Research project exploring adversarial attacks against a ResNet-18 ImageNet classifier, then measuring the effectiveness of various defenses and a detection method against those attacks.

All runnable scripts are in `really good stuff with graphs/`.

---

## Background

### What is an adversarial example?

A neural network like ResNet-18 classifies images by learning patterns in pixel data. An **adversarial example** is an image that has been slightly modified — often invisibly to the human eye — in a specific way that causes the model to misclassify it with high confidence.

For example, a photo of a car gets classified as "sports car" normally. After an adversarial perturbation is applied, the pixels look nearly identical but the model now says "cab" with 100% confidence.

The modifications are not random noise — they are computed by figuring out exactly which direction to nudge each pixel to push the model's output toward the wrong class.

### What is a perturbation budget (epsilon)?

Attacks are constrained by a maximum perturbation size called **epsilon (ε)**. This limits how much any pixel can change, so the image still looks natural. The scripts use ε = 8/255 under the L∞ norm — meaning no individual pixel channel changes by more than ~3% of its range.

---

## Attacks

### Carlini & Wagner L2 (C&W)
**Scripts:** `cnw/4cnw.py`, `cnw/5cnw.py`

C&W is an optimization-based attack. It directly solves an optimization problem: find the smallest possible change to the image (measured by L2 distance — Euclidean distance across all pixels) that still causes misclassification.

It works by:
1. Mapping the image into "tanh space" so the optimizer can't push pixel values outside [0, 1]
2. Running Adam optimizer over thousands of iterations, minimizing a loss that balances two things: keeping the perturbation small, and making the wrong class more confident than the right class
3. Using binary search to tune the tradeoff constant between those two objectives

C&W tends to produce smaller, cleaner perturbations than PGD. It is slower but more precise.

`5cnw.py` adds **confidence history tracking** — it records the model's confidence in the original class at every iteration and plots it, showing how the attack gradually erodes the model's certainty.

### Projected Gradient Descent (PGD)
**Scripts:** `pgd/1pgd.py`, `pgd/2pgd.py`

PGD is a gradient-based attack. It works by repeatedly taking small steps in the direction that most increases the model's loss (i.e., makes it more wrong), then projecting back into the allowed perturbation budget after each step.

Each step:
1. Compute the gradient of the cross-entropy loss with respect to the input pixels
2. Take a step of size `eps_iter` in the gradient's sign direction (for L∞)
3. Clip the total perturbation back to within ε of the original image
4. Clamp pixel values to [0, 1]

PGD starts from a random point within the perturbation budget (`rand_init=True`) to avoid getting stuck, and runs for a fixed number of iterations (20 steps in these scripts).

`2pgd.py` also tracks confidence history per iteration and plots it alongside the original, adversarial, and perturbation images.

---

## Defenses

All defense scripts are in `defenses/`. Each one generates adversarial examples with PGD first, then applies the defense and checks if the correct classification is restored.

### 1. JPEG Compression — `defenses/1compression.py`

The simplest defense. Before passing an image to the model, re-encode it through JPEG at a reduced quality level (Q=25, 50, 75). JPEG compression removes high-frequency detail by quantizing the image's frequency components, which can destroy adversarial perturbations that live in those frequencies.

**How it works in code:** the adversarial tensor is converted to a PIL image, saved to an in-memory buffer as JPEG at the target quality, then reloaded and converted back to a tensor.

**Results:** largely ineffective against PGD. The adversarial perturbation (spread evenly across all pixels at ε=8/255) survives JPEG compression at all quality levels for most images.

### 2. Feature Squeezing — `defenses/2feature_squeezing.py`

Tests two types of input transformation, individually and combined:

- **Bit-depth reduction:** rounds each pixel value to a lower number of bits (1, 2, or 4 bits per channel instead of 8). For example, 2-bit reduction collapses 256 possible values per channel down to 4. This removes fine-grained pixel differences where adversarial perturbations often hide.
- **Gaussian smoothing:** applies a blur kernel across the image, averaging neighboring pixels. Smooths out sharp local perturbations.
- **Combined:** bit-depth reduction followed by Gaussian smoothing.

**Results:** mostly ineffective, with 2 isolated restorations (car at 2-bit+smooth, tiger at 1-bit+smooth). Notably, 4-bit reduction sometimes *increased* adversarial confidence — the perturbation survived quantization at that precision level almost perfectly. Very aggressive squeezing (1-bit) often just produced random wrong classes rather than restoring the original.

### 3. Randomized Smoothing — `defenses/3randomized_smoothing.py`

Instead of a single forward pass, this defense runs the model 100 times on noisy copies of the input and takes a majority vote. Gaussian noise with standard deviation σ is added each time. The idea is that adversarial perturbations are brittle — they are tuned precisely for the clean input, so adding random noise disrupts them. Clean images, being more robustly classified, should survive the noise better.

The voted class and its vote fraction (how many of the 100 runs agreed) are reported.

**Results:** zero restorations across all images and all σ values. At σ=0.5 (high noise), all images collapse to predicting "bubble" — the noise is so destructive the model loses all semantic signal. The PGD attack is strong enough that the adversarial direction persists even through substantial Gaussian noise without adversarial training of the model itself.

---

## Detection

### Feature Squeezing Detector — `defenses/4detector.py`

Rather than trying to fix adversarial examples, this script tries to **identify** them before classification. The key insight: clean images are stable under input transformations — squeezing a clean image doesn't change the model's output much. Adversarial images are not stable, because squeezing destroys the carefully crafted perturbation and the model's output shifts dramatically.

**How it works:**
1. Run the input through the model to get a softmax probability distribution
2. Apply each squeezer (Gaussian smooth, 2-bit reduction, 2-bit+smooth) and get a new softmax distribution
3. Compute the L1 distance between the original and squeezed distributions for each squeezer
4. Take the maximum distance across all squeezers as the detection score
5. If `score > threshold`, flag the input as adversarial

**Threshold:** set to 1.6 based on the observed score gap — clean images scored 0.33–1.44, adversarial images scored 1.88–2.00. A threshold of 1.6 sits in that gap.

**Output:** a two-panel plot showing the score bar chart (clean vs. adversarial per image) and a ROC curve with the chosen threshold marked, plus a CSV log and console summary of TP/TN/FP/FN.

**Results on 3 images:** 100% detection accuracy (TP=3, TN=3, FP=0, FN=0). Note that the threshold was calibrated on the same 3 images used for evaluation — a larger held-out dataset would be needed to validate generalization.

---

## Running the Scripts

```bash
# Install dependencies
pip install -r requirements.txt

# Attacks (run from their respective folders)
cd "really good stuff with graphs/cnw"
python3 5cnw.py   # C&W attack with confidence tracking

cd "really good stuff with graphs/pgd"
python3 2pgd.py   # PGD attack with confidence tracking

# Defenses (run from the defenses folder)
cd "really good stuff with graphs/defenses"
python3 1compression.py        # JPEG compression defense
python3 2feature_squeezing.py  # Bit-depth + smoothing defense
python3 3randomized_smoothing.py  # Randomized smoothing defense
python3 4detector.py           # Adversarial input detector
```

Each script reads images from `my_images/` in the relevant folder, outputs PNG plots, and logs results to a CSV file.

---

## Model

All scripts use **ResNet-18** pretrained on ImageNet (1000 classes), loaded via `torchvision`. A `NormalizedModel` wrapper applies ImageNet mean/std normalization internally so all inputs can be kept in the [0, 1] range throughout attack and defense computation.

---

## Key Findings

| Defense | Restorations | Notes |
|---|---|---|
| JPEG Compression | 1 / 9 | Perturbation survives all quality levels for 2/3 images |
| Feature Squeezing | 2 / 21 | Inconsistent; 4-bit reduction can increase adversarial confidence |
| Randomized Smoothing | 0 / 9 | High σ destroys image; low σ doesn't break the attack |
| **Detector** | **6 / 6 correct** | Detects adversarial inputs without needing to restore them |

Preprocessing defenses largely fail against strong PGD attacks because they do not change the model's decision boundary — they can only hope to remove enough of the perturbation, which is insufficient when the attack is strong. The detector sidesteps this by exploiting the brittleness of adversarial examples under transformation rather than trying to undo the attack.
