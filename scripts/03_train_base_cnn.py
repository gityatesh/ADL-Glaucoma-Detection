"""Train a custom CNN from scratch: python scripts/03_train_base_cnn.py."""

import csv
import math
import os
import random
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

matplotlib.use("Agg")  # Save plots without requiring a display or blocking training.
import matplotlib.pyplot as plt


# Paths are relative to the project, even when launched from another directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "validate"
BEST_LOSS_PATH = PROJECT_ROOT / "base_cnn_best_loss.pth"
BEST_ACCURACY_PATH = PROJECT_ROOT / "base_cnn_best_accuracy.pth"
HISTORY_PATH = PROJECT_ROOT / "results" / "metrics" / "base_cnn_history.csv"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"

IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.03
SEED = 42
EARLY_STOPPING_PATIENCE = 12
EARLY_STOPPING_MIN_DELTA = 1e-4
LR_PATIENCE = 4
MIN_LR = 1e-6


def seed_everything(seed):
    """Seed sampling, augmentation and initialization on the selected backend."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Unsupported deterministic kernels warn instead of preventing MPS/CUDA runs.
    # Exact reproducibility can still vary across backends and PyTorch versions.
    torch.use_deterministic_algorithms(True, warn_only=True)


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_loaders(device):
    # Mild augmentation limits memorization without aggressively cropping the disc.
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(
            degrees=7,
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.ColorJitter(brightness=0.075, contrast=0.075),
        transforms.ToTensor(),
        normalize,
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transforms)
    if train_dataset.classes != ["glaucoma", "normal"]:
        raise ValueError("Training data must contain glaucoma and normal classes only.")
    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError("Training and validation class mappings must match.")

    # Loading in the main process keeps augmentation seeded and works on Windows.
    # Separate generators prevent validation iteration from changing training order.
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(SEED),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(SEED + 1),
    )
    print("Class mapping:", train_dataset.class_to_idx)
    print(f"Training images: {len(train_dataset)} | Validation images: {len(val_dataset)}")
    return train_loader, val_loader


class BaseCNN(nn.Module):
    """Four convolution blocks with GroupNorm and modest dropout."""

    def __init__(self):
        super().__init__()
        layers = []
        in_channels = 3
        for out_channels, spatial_dropout in (
            (32, 0.0), (64, 0.0), (128, 0.10), (256, 0.15)
        ):
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                # All channel counts divide evenly into eight groups. GroupNorm
                # uses per-image statistics in both training and evaluation.
                nn.GroupNorm(num_groups=8, num_channels=out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ])
            if spatial_dropout > 0:
                layers.append(nn.Dropout2d(spatial_dropout))
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        # Return logits; CrossEntropyLoss applies log-softmax internally.
        return self.classifier(self.features(x))


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            if training:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += batch_size

    return running_loss / total, correct / total


def save_plots(history):
    epochs = [row["epoch"] for row in history]
    for metric, ylabel in [("loss", "Loss"), ("accuracy", "Accuracy")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, [row[f"train_{metric}"] for row in history], label="Training")
        ax.plot(epochs, [row[f"val_{metric}"] for row in history], label="Validation")
        ax.set(xlabel="Epoch", ylabel=ylabel, title=f"Base CNN - {ylabel}")
        if metric == "accuracy":
            ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"base_cnn_{metric}_curve.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step(epochs, [row["learning_rate"] for row in history], where="post")
    ax.set(
        xlabel="Epoch", ylabel="Learning rate", yscale="log",
        title="Base CNN - Learning Rate",
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "base_cnn_learning_rate_curve.png", dpi=150)
    plt.close(fig)


def save_checkpoint(model, path, metrics, description):
    """Save a portable state_dict and report the metrics from that exact epoch."""
    torch.save(
        {name: value.detach().cpu() for name, value in model.state_dict().items()},
        path,
    )
    print(
        f"  Saved {description} checkpoint: {path.name} | Epoch {metrics['epoch']} | "
        f"Train accuracy: {metrics['train_accuracy']:.2%} | "
        f"Validation accuracy: {metrics['val_accuracy']:.2%} | "
        f"Validation loss: {metrics['val_loss']:.4f} | "
        f"LR: {metrics['learning_rate']:.2e}"
    )


def main():
    seed_everything(SEED)
    device = select_device()
    print(f"Using device: {device} | Seed: {SEED}")
    train_loader, val_loader = build_loaders(device)
    model = BaseCNN().to(device)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {parameter_count:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=LR_PATIENCE,
        threshold=EARLY_STOPPING_MIN_DELTA,
        threshold_mode="abs",
        min_lr=MIN_LR,
    )

    for path in (BEST_LOSS_PATH, BEST_ACCURACY_PATH, HISTORY_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    history = []
    best_loss_metrics = None
    best_accuracy_metrics = None
    stopping_best_loss = float("inf")
    epochs_without_improvement = 0
    fieldnames = [
        "epoch", "learning_rate", "train_loss", "train_accuracy",
        "val_loss", "val_accuracy", "generalization_gap",
    ]

    with HISTORY_PATH.open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(1, EPOCHS + 1):
            learning_rate = optimizer.param_groups[0]["lr"]
            train_loss, train_acc = run_epoch(
                model, train_loader, criterion, device, optimizer
            )
            val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
            if not math.isfinite(train_loss) or not math.isfinite(val_loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}; stopping training.")

            # Step once per epoch, after validation, using this epoch's loss.
            scheduler.step(val_loss)
            # Training accuracy includes augmentation and active dropout, so this
            # diagnostic is not a comparison of two evaluation-mode accuracies.
            generalization_gap = train_acc - val_acc
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
                "learning_rate": learning_rate,  # Rate used for this epoch.
                "generalization_gap": generalization_gap,  # Fraction, like accuracy.
            }
            history.append(row)
            writer.writerow(row)
            history_file.flush()
            print(
                f"Epoch {epoch:02d}/{EPOCHS} | LR: {learning_rate:.2e} | "
                f"Train loss: {train_loss:.4f}, accuracy: {train_acc:.2%} | "
                f"Val loss: {val_loss:.4f}, accuracy: {val_acc:.2%} | "
                f"Generalization gap: {100 * generalization_gap:+.2f} pp"
            )

            # Every new loss minimum is saved, even below the early-stopping delta.
            if best_loss_metrics is None or val_loss < best_loss_metrics["val_loss"]:
                best_loss_metrics = row.copy()
                save_checkpoint(
                    model, BEST_LOSS_PATH, best_loss_metrics, "best validation-loss"
                )

            # Independent of loss: a higher-accuracy epoch must also be retained.
            if best_accuracy_metrics is None or val_acc > best_accuracy_metrics["val_accuracy"]:
                best_accuracy_metrics = row.copy()
                save_checkpoint(
                    model, BEST_ACCURACY_PATH, best_accuracy_metrics, "best validation-accuracy"
                )

            # Patience exceeds LR_PATIENCE so a reduced rate has time to help.
            if val_loss < stopping_best_loss - EARLY_STOPPING_MIN_DELTA:
                stopping_best_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if optimizer.param_groups[0]["lr"] < learning_rate:
                print(f"  Learning rate reduced to {optimizer.param_groups[0]['lr']:.2e}.")
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(
                    f"Early stopping after {EARLY_STOPPING_PATIENCE} epochs "
                    "without meaningful loss improvement."
                )
                break

    save_plots(history)
    print("\nTraining completed.")
    print(
        f"Best validation-loss epoch: {best_loss_metrics['epoch']} | "
        f"Validation loss: {best_loss_metrics['val_loss']:.4f} | "
        f"Validation accuracy: {best_loss_metrics['val_accuracy']:.2%}"
    )
    print(
        f"Best validation-accuracy epoch: {best_accuracy_metrics['epoch']} | "
        f"Validation accuracy: {best_accuracy_metrics['val_accuracy']:.2%} | "
        f"Validation loss: {best_accuracy_metrics['val_loss']:.4f}"
    )
    print(f"Final train accuracy: {history[-1]['train_accuracy']:.2%}")
    print(f"Trainable parameters: {parameter_count:,}")
    print(f"Best-loss model: {BEST_LOSS_PATH}")
    print(f"Best-accuracy model: {BEST_ACCURACY_PATH}")
    print(f"History: {HISTORY_PATH}")
    print(f"Plots: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
