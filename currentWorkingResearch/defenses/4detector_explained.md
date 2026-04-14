# How the Adversarial Image Detector Works

## The Big Picture

This program is a **security guard for an AI image classifier**. Its job is to look at an image and decide: *"Is this a normal photo, or has someone tampered with it to trick the AI?"*

Images that have been tampered with to fool AI systems are called **adversarial images**. They look almost identical to the human eye but contain tiny, invisible changes that cause AI models to misclassify them — for example, making a photo of a tiger get labeled as a toaster.

---

## The Core Idea: "Squeeze and Compare"

The detector uses a technique called **Feature Squeezing**. Here is the intuition:

> If an image is normal, simplifying it slightly should not change how the AI interprets it much. But if an image has been secretly manipulated, simplifying it tends to destroy those hidden changes — and the AI's interpretation shifts dramatically.

Think of it like this: imagine someone whispered a hidden message into a song by adjusting tiny audio details. If you lower the audio quality slightly, the music still sounds the same — but the hidden message gets wiped out. The detector listens for that shift.

---

## Step-by-Step: What the Program Does

### 1. Load the AI Model
The program loads a pre-trained image recognition model (GoogLeNet) that can classify images into 1,000 categories like "cat," "car," or "tiger."

### 2. Load Your Images
It reads all the photos from a folder of test images.

### 3. Create a Tampered Version of Each Image
For every original photo, the program automatically generates an adversarial (tampered) version using a method called **PGD (Projected Gradient Descent)**. This attack makes tiny pixel adjustments — invisible to humans — that cause the AI to misclassify the image.

### 4. Apply "Squeezers" to Each Image
Each image (both original and tampered) is run through three simplification filters called **squeezers**:

| Squeezer | What it does |
|---|---|
| **Smooth** | Applies a slight blur to the image |
| **Bit-2** | Reduces the image to only 4 color levels (very rough) |
| **Bit-2 + Smooth** | Does both of the above |

### 5. Measure the Difference
After squeezing, the program asks the AI to classify both the original and the squeezed version. It then measures **how much the AI's confidence scores changed**.

- **Small change** → the image is probably clean (normal)
- **Large change** → the image is probably adversarial (tampered)

This difference is called the **squeezing distance**.

### 6. Make a Decision
The program compares the largest squeezing distance to a pre-set **threshold** (set at 1.6). If the score is above 1.6, the image is flagged as adversarial.

### 7. Report Results
The program prints a table showing each image, its score, and whether it was correctly identified. It also saves results to a CSV file.

### 8. Generate Charts
Two charts are produced and saved as a PNG file:

- **Left chart:** A bar graph comparing detection scores for clean vs. adversarial images, with a threshold line drawn across it.
- **Right chart:** A **ROC curve** — a standard way to visualize how well a detector performs across all possible thresholds.

---

## Key Terms, Simply Explained

**Adversarial image:** A photo with invisible pixel changes designed to fool an AI.

**Squeezing distance:** How much the AI's answer changes after the image is simplified. High distance = suspicious.

**Threshold:** The cutoff score. Above it = flagged as tampered. Below it = considered clean.

**True Positive (TP):** A tampered image that was correctly flagged.

**True Negative (TN):** A clean image that was correctly cleared.

**False Positive (FP):** A clean image that was wrongly flagged as tampered.

**False Negative (FN):** A tampered image that slipped through undetected.

**ROC Curve:** A graph showing the trade-off between catching real threats (sensitivity) and raising false alarms (specificity). A detector in the top-left corner of the graph is performing well.

---

## In Summary

The detector's logic is simple: **simplify an image, then check if the AI changes its mind**. If the AI's answer barely changes, the image is probably clean. If the AI's answer shifts dramatically, something suspicious was hidden in the image that the simplification destroyed — and the image gets flagged.
