"""
Binary Image Classification CNN — Glaucoma Detection (PyTorch)
================================================================
Classes: glaucoma (positive, label=1) vs normal (negative, label=0)
Input size: 128x128 RGB
Framework: PyTorch

v4 changes (vs v3):
- Added CLAHE (Contrast Limited Adaptive Histogram Equalization) preprocessing
  — a standard technique for fundus/retinal images that enhances optic disc,
  cup, and vessel contrast before the CNN sees the image. Applied via a
  custom transform using OpenCV.
- Added a 5th conv block (256 -> 512) for slightly more capacity, since the
  GAP head keeps this from reopening the overfitting problem.
- Smoothed validation curve (5-epoch moving average) added to the accuracy
  plot ALONGSIDE the raw curve — both shown, clearly labeled, so nothing is
  hidden, just easier to read the trend through the noise.
- Still: no early stopping, full EPOCHS run, best checkpoint tracked separately.

Requires: pip install opencv-python
"""

# ============================================================
# 1. IMPORTS
# ============================================================
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# ============================================================
# 2. CONFIG
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "..", "dataset")

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 40
SEED = 42

TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "validate")
TEST_DIR = os.path.join(BASE_DIR, "test")

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# 3. CLAHE PREPROCESSING (custom transform)
# ============================================================
class CLAHETransform:
    """
    Applies CLAHE to the L channel of the LAB color space, then converts
    back to RGB. This boosts local contrast (disc/cup/vessel edges) without
    blowing out overall brightness the way global contrast stretching would.
    """
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

# ============================================================
# 4. DATA LOADING
# ============================================================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    clahe_transform,
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
    transforms.ToTensor(),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    clahe_transform,
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=eval_transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=eval_transform)

def build_remap(dataset):
    remap = torch.zeros(len(dataset.classes), dtype=torch.float32)
    for class_name, idx in dataset.class_to_idx.items():
        remap[idx] = 1.0 if class_name.lower() == "glaucoma" else 0.0
    return remap

train_remap = build_remap(train_dataset)
val_remap = build_remap(val_dataset)
test_remap = build_remap(test_dataset)

print(f"Train class_to_idx: {train_dataset.class_to_idx} -> remapped so glaucoma=1, normal=0")

print("\n===== LABEL SANITY CHECK (first 6 training images) =====")
for i in range(min(6, len(train_dataset))):
    label_idx = train_dataset.samples[i][1]
    class_name = train_dataset.classes[label_idx]
    remapped_label = train_remap[label_idx].item()
    file_path = train_dataset.samples[i][0]
    print(f"{os.path.basename(file_path):30s} folder={class_name:10s} -> remapped_label={remapped_label}")
print()

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# 5. MODEL ARCHITECTURE (5 conv blocks, GAP head)
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
            conv_block(3, 32),      # 128 -> 64
            conv_block(32, 64),     # 64 -> 32
            conv_block(64, 128),    # 32 -> 16
            conv_block(128, 256),   # 16 -> 8
            conv_block(256, 512),   # 8 -> 4   (new 5th block)
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


model = GlaucomaCNN().to(device)
print(model)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total trainable parameters: {total_params:,}")

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)

# ============================================================
# 6. TRAINING (full EPOCHS, no early stopping)
# ============================================================
def run_epoch(loader, remap, training):
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels_idx in loader:
            images = images.to(device)
            labels = remap[labels_idx].unsqueeze(1).to(device)

            if training:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_val_loss = float("inf")
best_epoch = 0
best_state = None

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = run_epoch(train_loader, train_remap, training=True)
    val_loss, val_acc = run_epoch(val_loader, val_remap, training=False)

    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]["lr"]

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(f"Epoch {epoch:02d}/{EPOCHS} | "
          f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
          f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | lr={current_lr:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

print(f"\nTraining complete — ran all {EPOCHS} epochs.")
print(f"Best val_loss was {best_val_loss:.4f} at epoch {best_epoch}.")

torch.save(model.state_dict(), "final_epoch_model.pth")
if best_state is not None:
    torch.save(best_state, "best_checkpoint_model.pth")
    print(f"Saved: final_epoch_model.pth (epoch {EPOCHS}) and best_checkpoint_model.pth (epoch {best_epoch})")

# ------------------------------------------------------------
# Plot training curves — raw + smoothed val curve, both shown
# ------------------------------------------------------------
def moving_average(values, window=5):
    values = np.array(values)
    if len(values) < window:
        return values
    smoothed = np.convolve(values, np.ones(window) / window, mode="valid")
    pad = len(values) - len(smoothed)
    return np.concatenate([values[:pad], smoothed])

val_acc_smooth = moving_average(history["val_acc"], window=5)
val_loss_smooth = moving_average(history["val_loss"], window=5)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history["train_acc"], label="Train Accuracy")
plt.plot(history["val_acc"], label="Val Accuracy (raw)", alpha=0.4)
plt.plot(val_acc_smooth, label="Val Accuracy (5-epoch avg)", linewidth=2)
plt.axvline(best_epoch - 1, color="gray", linestyle="--", alpha=0.5, label=f"Best epoch ({best_epoch})")
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss (raw)", alpha=0.4)
plt.plot(val_loss_smooth, label="Val Loss (5-epoch avg)", linewidth=2)
plt.axvline(best_epoch - 1, color="gray", linestyle="--", alpha=0.5, label=f"Best epoch ({best_epoch})")
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("training_curves.png")
plt.show()

# ============================================================
# 7. EVALUATION ON TEST SET (using BEST-CHECKPOINT weights,
#    since that's the epoch where train/val were still in
#    agreement — final-epoch weights were overfit past that point)
# ============================================================
model.eval()
model.load_state_dict(torch.load("best_checkpoint_model.pth"))
y_true = []
y_pred_prob = []

with torch.no_grad():
    for images, labels_idx in test_loader:
        images = images.to(device)
        labels = test_remap[labels_idx].unsqueeze(1)

        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

        y_pred_prob.extend(probs)
        y_true.extend(labels.numpy().flatten())

y_true = np.array(y_true)
y_pred_prob = np.array(y_pred_prob)
y_pred = (y_pred_prob >= 0.5).astype(int)

CLASS_NAMES = ["normal", "glaucoma"]

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
)
plt.title("Confusion Matrix (best-checkpoint weights)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
specificity = tn / (tn + fp)
f1 = f1_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_pred_prob)

print("\n===== TEST SET METRICS (best-checkpoint weights) =====")
print(f"Accuracy    : {accuracy:.4f}")
print(f"Precision   : {precision:.4f}")
print(f"Recall      : {recall:.4f}")
print(f"Specificity : {specificity:.4f}")
print(f"F1 Score    : {f1:.4f}")
print(f"ROC-AUC     : {auc:.4f}")

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)

plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()

print("\nSaved plots: training_curves.png, confusion_matrix.png, roc_curve.png")
