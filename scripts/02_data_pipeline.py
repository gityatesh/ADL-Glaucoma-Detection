import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# ==============================
# CONFIGURATION
# ==============================

DATASET_DIR = "dataset"

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "validate")
TEST_DIR = os.path.join(DATASET_DIR, "test")

IMAGE_SIZE = 128
BATCH_SIZE = 32

# Automatically select device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")   # For Mac Apple Silicon
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# ==============================
# TRANSFORMS
# ==============================

# Training data gets augmentation
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.RandomResizedCrop(
        size=IMAGE_SIZE,
        scale=(0.9, 1.0)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Validation and test data should NOT be augmented
eval_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==============================
# DATASETS
# ==============================

train_dataset = datasets.ImageFolder(
    root=TRAIN_DIR,
    transform=train_transforms
)

val_dataset = datasets.ImageFolder(
    root=VAL_DIR,
    transform=eval_transforms
)

test_dataset = datasets.ImageFolder(
    root=TEST_DIR,
    transform=eval_transforms
)

# ==============================
# DATA LOADERS
# ==============================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ==============================
# BASIC DATASET INFORMATION
# ==============================

print("\n==============================")
print("DATA LOADING SUMMARY")
print("==============================")

print(f"Classes: {train_dataset.classes}")
print(f"Class to index mapping: {train_dataset.class_to_idx}")

print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")
print(f"Testing images: {len(test_dataset)}")

print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
print(f"Testing batches: {len(test_loader)}")

# Check one batch
images, labels = next(iter(train_loader))

print("\nOne training batch:")
print(f"Image batch shape: {images.shape}")
print(f"Label batch shape: {labels.shape}")

# ==============================
# VISUALIZE SAMPLE IMAGES
# ==============================

def imshow(img):
    """
    Unnormalize and display image.
    """
    img = img.numpy().transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img = std * img + mean
    img = np.clip(img, 0, 1)

    return img


plt.figure(figsize=(10, 6))

for i in range(8):
    plt.subplot(2, 4, i + 1)
    img = imshow(images[i])
    label = labels[i].item()
    class_name = train_dataset.classes[label]

    plt.imshow(img)
    plt.title(class_name)
    plt.axis("off")

plt.tight_layout()
plt.show()

print("\nData pipeline check completed successfully.")