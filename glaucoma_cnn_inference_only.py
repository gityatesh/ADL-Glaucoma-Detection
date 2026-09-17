"""
Glaucoma CNN — Inference Only
==============================
Loads the already-trained best_checkpoint_model.pth (saved by your training
script) and lets you classify new images WITHOUT retraining.

Run this AFTER you've already run the training script once and it has
produced best_checkpoint_model.pth in the same folder.

Must match the exact model architecture used in training (5 conv blocks,
GAP head) — this is copy-pasted from the training script's Section 5 so the
saved weights load correctly.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# ============================================================
# CONFIG — must match training script
# ============================================================
IMG_SIZE = 128
MODEL_PATH = "best_checkpoint_model.pth"   # <-- change path if it's elsewhere

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# CLAHE preprocessing — same as training script
# ============================================================
class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, pil_img):
        img = np.array(pil_img.convert("RGB"))
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l_eq = self.clahe.apply(l)
        lab_eq = cv2.merge((l_eq, a, b))
        rgb_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
        return Image.fromarray(rgb_eq)


clahe_transform = CLAHETransform()

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    clahe_transform,
    transforms.ToTensor(),
])

# ============================================================
# MODEL ARCHITECTURE — must exactly match training script
# ============================================================
class GlaucomaCNN(nn.Module):
    def __init__(self):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.2),
            )

        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 512),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================
# LOAD THE SAVED MODEL — no training happens here
# ============================================================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Could not find '{MODEL_PATH}'. Make sure you've run the training "
        f"script at least once, and that this file is in the same folder "
        f"where the training script saved best_checkpoint_model.pth."
    )

model = GlaucomaCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"Loaded trained model from '{MODEL_PATH}'. Ready for inference.\n")

# ============================================================
# SINGLE-IMAGE INFERENCE
# ============================================================
def predict_image(image_path, threshold=0.5):
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return None

    img = Image.open(image_path).convert("RGB")
    img_tensor = eval_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(img_tensor)
        prob = torch.sigmoid(logit).item()

    predicted_class = "glaucoma" if prob >= threshold else "normal"
    confidence = prob if predicted_class == "glaucoma" else 1 - prob

    print(f"\nImage: {image_path}")
    print(f"Predicted class : {predicted_class}")
    print(f"Confidence      : {confidence:.2%}")
    print(f"Raw probability (glaucoma=1): {prob:.4f}")

    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.title(f"Prediction: {predicted_class} ({confidence:.1%} confidence)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return predicted_class, confidence


print("=" * 60)
print("SINGLE-IMAGE INFERENCE")
print("=" * 60)
print("A file picker window will open — choose a PNG image to classify.")
print("Close the file picker without selecting anything to quit.\n")

# Hide the empty root tkinter window that would otherwise pop up
root = tk.Tk()
root.withdraw()

while True:
    image_path = filedialog.askopenfilename(
        title="Select a fundus image (PNG)",
        filetypes=[("PNG images", "*.png")],
    )
    if not image_path:  # user closed the dialog or clicked Cancel
        print("No file selected. Exiting.")
        break
    predict_image(image_path)

    # Ask (in terminal) whether to pick another image
    again = input("\nClassify another image? (y/n): ").strip().lower()
    if again != "y":
        print("Exiting.")
        break

root.destroy()