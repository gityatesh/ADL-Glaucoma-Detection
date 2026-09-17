"""Run: python efficientnet_scripts/04_train_efficientnet_b0.py.

Only training and validation images are read. Test evaluation belongs in a
separate script after the experiment and model-selection rules are finalized.
"""

import csv
import math
import os
import platform
import random
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = PROJECT_ROOT / "dataset" / "train"
VAL_DIR = PROJECT_ROOT / "dataset" / "validate"
MODELS_DIR = PROJECT_ROOT / "models"
BEST_LOSS_PATH = MODELS_DIR / "efficientnet_b0_best_loss.pth"
BEST_ACCURACY_PATH = MODELS_DIR / "efficientnet_b0_best_accuracy.pth"
HISTORY_PATH = PROJECT_ROOT / "results" / "metrics" / "efficientnet_b0_history.csv"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"

MODEL_NAME = "torchvision.models.efficientnet_b0"
# Torchvision manages the official download in its normal cache outside this project.
# Checkpoints record the resolved weight version for reproducibility.
WEIGHTS = EfficientNet_B0_Weights.DEFAULT
IMAGE_SIZE = 224
BATCH_SIZE = 32  # Lower this on the Mac only if unified memory is insufficient.
NUM_WORKERS = 0  # Portable on Windows and macOS; augmentation runs in the seeded process.
SEED = 42
CLASSIFIER_DROPOUT = 0.30
LABEL_SMOOTHING = 0.02
WEIGHT_DECAY = 1e-4
STAGE1_EPOCHS = 8
STAGE2_EPOCHS = 15
STAGE1_CLASSIFIER_LR = 1e-3
STAGE2_BACKBONE_LR = 1e-5
STAGE2_CLASSIFIER_LR = 1e-4
UNFREEZE_LAST_BLOCKS = 3  # EfficientNetB0 features[6], features[7], features[8].
LR_PATIENCE = 3
MIN_BACKBONE_LR = 1e-7
MIN_CLASSIFIER_LR = 1e-6
EARLY_STOPPING_PATIENCE = 6
MIN_DELTA = 1e-4
POSITIVE_CLASS = "glaucoma"

HISTORY_FIELDS = [
    "stage",
    "stage_epoch",
    "global_epoch",
    "learning_rate_backbone",
    "learning_rate_classifier",
    "train_loss",
    "train_accuracy",
    "val_loss",
    "val_accuracy",
    "generalization_gap",
]


def set_seed(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Some MPS/CUDA kernels lack deterministic implementations. Warn rather than
    # abort, and do not promise identical results across hardware or library versions.
    torch.use_deterministic_algorithms(True, warn_only=True)


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_rgb_image(path):
    """Fully decode the image here so decoding errors identify the source file."""
    try:
        with Image.open(path) as image:
            image.load()
            return image.convert("RGB")
    except (OSError, ValueError, SyntaxError) as error:
        raise RuntimeError(f"Cannot decode fundus image '{path}': {error}") from error


def build_transforms():
    preset = WEIGHTS.transforms()
    normalize = transforms.Normalize(mean=preset.mean, std=preset.std)
    # Keep the entire fundus: use the official normalization/interpolation but
    # resize directly to 224 rather than using the preset's ImageNet center crop.
    resize = transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE), interpolation=preset.interpolation, antialias=True
    )
    train_transform = transforms.Compose(
        [
            resize,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(
                degrees=5, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ColorJitter(brightness=0.05, contrast=0.05),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_transform = transforms.Compose([resize, transforms.ToTensor(), normalize])
    return train_transform, val_transform


def build_dataloaders(device):
    train_transform, val_transform = build_transforms()
    train_dataset = datasets.ImageFolder(
        TRAIN_DIR, transform=train_transform, loader=load_rgb_image
    )
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform, loader=load_rgb_image)
    expected_mapping = {"glaucoma": 0, "normal": 1}
    if train_dataset.class_to_idx != expected_mapping:
        raise ValueError(f"Expected training class mapping {expected_mapping}.")
    if val_dataset.class_to_idx != train_dataset.class_to_idx:
        raise ValueError("Training and validation class mappings must match.")

    # Independent generators prevent validation iteration from changing training order.
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(SEED),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(SEED + 1),
    )
    print("Class mapping:", train_dataset.class_to_idx)
    print(f"Batch size: {BATCH_SIZE} | DataLoader workers: {NUM_WORKERS}")
    print("Positive disease class: glaucoma (index 0)")
    # Future evaluation must use y_true = (labels == glaucoma_idx) and
    # y_score = softmax(logits, dim=1)[:, glaucoma_idx] for glaucoma ROC-AUC/recall.
    # Numeric label 1 means NORMAL here, not the positive disease class.
    print(f"Training images: {len(train_dataset)} | Training batches: {len(train_loader)}")
    print(f"Validation images: {len(val_dataset)} | Validation batches: {len(val_loader)}")
    return train_loader, val_loader


def build_model():
    # ImageNet features provide a starting point for this relatively small dataset.
    # Download failures propagate; never silently substitute randomly initialized weights.
    model = efficientnet_b0(weights=WEIGHTS)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(CLASSIFIER_DROPOUT), nn.Linear(in_features, 2))
    print(f"Pretrained weights loaded: EfficientNet_B0_Weights.{WEIGHTS.name}")
    return model


def set_stage1_trainable(model):
    # Learn the new classifier before changing the pretrained feature extractor.
    model.requires_grad_(False)
    model.classifier.requires_grad_(True)
    model.zero_grad(set_to_none=True)


def set_stage2_trainable(model):
    set_stage1_trainable(model)
    if not 0 < UNFREEZE_LAST_BLOCKS < len(model.features):
        raise ValueError("Fine tuning must unfreeze only part of the backbone.")
    # Early ImageNet filters capture edges, gradients, textures and simple shapes.
    # Adapt only later semantic features to the retina, limiting overfitting and
    # catastrophic forgetting of useful pretrained representations.
    for block in model.features[-UNFREEZE_LAST_BLOCKS:]:
        block.requires_grad_(True)


def verify_trainable_parameters(model, stage):
    """Check the actual parameter flags before constructing each stage's optimizer."""
    expected = {id(parameter) for parameter in model.classifier.parameters()}
    if stage == 2:
        expected.update(
            id(parameter)
            for block in model.features[-UNFREEZE_LAST_BLOCKS:]
            for parameter in block.parameters()
        )
    elif stage != 1:
        raise ValueError(f"Unknown training stage: {stage}")
    actual = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if actual != expected:
        raise RuntimeError(f"Stage {stage} trainable parameters do not match the intended layers.")
    print(
        f"Stage {stage} trainability verified: "
        + ("classifier only." if stage == 1 else "final feature groups and classifier only.")
    )


def set_training_mode(model):
    model.train()
    model.features.eval()
    for block in model.features:
        if any(parameter.requires_grad for parameter in block.parameters()):
            block.train()
    # Preserve ImageNet BatchNorm running statistics in BOTH stages. During fine
    # tuning, affine parameters in unfrozen blocks still learn through autograd.
    # This avoids noisy running-statistic drift on the small fundus dataset.
    for module in model.features.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


@torch.no_grad()
def sanity_check(model, train_loader, device):
    """Check one real batch without changing the subsequent shuffled/augmented batches."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    loader_state = train_loader.generator.get_state()
    was_training = model.training
    try:
        model.eval()  # No dropout, stochastic depth or BatchNorm buffer updates.
        images, labels = next(iter(train_loader))
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, IMAGE_SIZE, IMAGE_SIZE):
            raise RuntimeError(f"Unexpected input shape: {tuple(images.shape)}")
        if labels.shape != (images.size(0),) or labels.dtype != torch.long:
            raise RuntimeError("Expected one integer class label per image.")
        if not ((labels == 0) | (labels == 1)).all().item():
            raise RuntimeError("Expected glaucoma=0 and normal=1 labels.")
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        if tuple(logits.shape) != (images.size(0), 2):
            raise RuntimeError(f"Expected [batch_size, 2] outputs, got {tuple(logits.shape)}")
        if not (logits.device == images.device == labels.device == next(model.parameters()).device):
            raise RuntimeError("Model, inputs, labels and outputs must use the same device.")
        if not torch.isfinite(logits).all().item():
            raise RuntimeError(
                "Pretrained model produced non-finite outputs during the sanity check."
            )
        print(
            f"Sanity check passed | Input: {list(images.shape)} | "
            f"Output: {list(logits.shape)} | Device: {logits.device}"
        )
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        train_loader.generator.set_state(loader_state)
        if was_training:
            set_training_mode(model)
        else:
            model.eval()


def build_optimizer(model, stage):
    if stage == 1:
        groups = [
            {
                "params": model.classifier.parameters(),
                "lr": STAGE1_CLASSIFIER_LR,
                "name": "classifier",
            }
        ]
    else:
        # Small, discriminative rates protect useful pretrained representations.
        groups = [
            {
                "params": [p for p in model.features.parameters() if p.requires_grad],
                "lr": STAGE2_BACKBONE_LR,
                "name": "backbone",
            },
            {
                "params": model.classifier.parameters(),
                "lr": STAGE2_CLASSIFIER_LR,
                "name": "classifier",
            },
        ]
    return optim.AdamW(groups, weight_decay=WEIGHT_DECAY)


def train_one_epoch(model, loader, criterion, optimizer, device):
    set_training_mode(model)
    loss_sum, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise RuntimeError("Non-finite training loss; stopping before the optimizer step.")
        loss.backward()
        optimizer.step()
        loss_sum += loss_value * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        logits = model(images)
        loss_value = criterion(logits, labels).item()
        if not math.isfinite(loss_value):
            raise RuntimeError("Non-finite validation loss; no checkpoint will be selected.")
        loss_sum += loss_value * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return loss_sum / total, correct / total


def create_output_dirs():
    for directory in (MODELS_DIR, HISTORY_PATH.parent, PLOTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def experiment_metadata(train_loader, val_loader):
    preset = WEIGHTS.transforms()
    class_to_idx = dict(train_loader.dataset.class_to_idx)
    return {
        "model_name": MODEL_NAME,
        "pretrained_weights": f"EfficientNet_B0_Weights.{WEIGHTS.name}",
        "class_to_idx": class_to_idx,
        "positive_class": POSITIVE_CLASS,
        "positive_class_idx": class_to_idx[POSITIVE_CLASS],
        "image_size": IMAGE_SIZE,
        "normalization_mean": list(preset.mean),
        "normalization_std": list(preset.std),
        "resize_interpolation": preset.interpolation.name,
        "resize_policy": "resize directly to square; no crop",
        "classifier_dropout": CLASSIFIER_DROPOUT,
        "seed": SEED,
        "generalization_gap_units": "fraction",
        "training_images": len(train_loader.dataset),
        "validation_images": len(val_loader.dataset),
        "training_config": {
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "stage1_epochs": STAGE1_EPOCHS,
            "stage2_epochs": STAGE2_EPOCHS,
            "stage1_classifier_lr": STAGE1_CLASSIFIER_LR,
            "stage2_backbone_lr": STAGE2_BACKBONE_LR,
            "stage2_classifier_lr": STAGE2_CLASSIFIER_LR,
            "weight_decay": WEIGHT_DECAY,
            "label_smoothing": LABEL_SMOOTHING,
            "unfreeze_last_blocks": UNFREEZE_LAST_BLOCKS,
            "batchnorm_running_statistics": "frozen in both stages",
            "lr_factor": 0.5,
            "lr_patience": LR_PATIENCE,
            "min_backbone_lr": MIN_BACKBONE_LR,
            "min_classifier_lr": MIN_CLASSIFIER_LR,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "min_delta": MIN_DELTA,
            "horizontal_flip_probability": 0.5,
            "rotation_degrees": 5,
            "brightness_jitter": 0.05,
            "contrast_jitter": 0.05,
        },
        "software_versions": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__),
        },
    }


def save_checkpoint(model, path, metrics, metadata, selection_metric):
    # CPU tensors make the checkpoint portable between MPS, CUDA and CPU.
    checkpoint = dict(metadata)
    checkpoint.update(metrics)
    checkpoint["epoch"] = metrics["stage_epoch"]
    checkpoint["validation_loss"] = metrics["val_loss"]
    checkpoint["validation_accuracy"] = metrics["val_accuracy"]
    checkpoint["selection_metric"] = selection_metric
    checkpoint["model_state_dict"] = {
        name: value.detach().cpu() for name, value in model.state_dict().items()
    }
    torch.save(checkpoint, path)
    print(
        f"  Saved best {selection_metric} checkpoint: {path.name}\n"
        f"  Stage {metrics['stage']} | Epoch {metrics['stage_epoch']} "
        f"(global {metrics['global_epoch']}) | "
        f"Train accuracy: {metrics['train_accuracy']:.2%} | "
        f"Validation accuracy: {metrics['val_accuracy']:.2%} | "
        f"Validation loss: {metrics['val_loss']:.4f} | "
        f"LR backbone: {metrics['learning_rate_backbone']:.2e} | "
        f"LR classifier: {metrics['learning_rate_classifier']:.2e}"
    )


def plot_history(history):
    epochs = [row["global_epoch"] for row in history]
    transition = next((row["global_epoch"] - 0.5 for row in history if row["stage"] == 2), None)
    for metric, ylabel in [
        ("loss", "Loss"),
        ("accuracy", "Accuracy"),
        ("learning_rate", "Learning rate"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        if metric == "learning_rate":
            ax.step(
                epochs,
                [row["learning_rate_classifier"] for row in history],
                where="post",
                label="Classifier",
            )
            # The backbone is frozen in Stage 1 (CSV stores 0); omit zeros on a log axis.
            backbone_epochs = [row["global_epoch"] for row in history if row["stage"] == 2]
            backbone_rates = [row["learning_rate_backbone"] for row in history if row["stage"] == 2]
            if backbone_epochs:
                ax.step(backbone_epochs, backbone_rates, where="post", label="Unfrozen backbone")
            ax.set_yscale("log")
        else:
            ax.plot(epochs, [row[f"train_{metric}"] for row in history], label="Training")
            ax.plot(epochs, [row[f"val_{metric}"] for row in history], label="Validation")
            if metric == "accuracy":
                ax.set_ylim(0, 1)
        if transition is not None:
            ax.axvline(transition, color="gray", linestyle="--", label="Start fine tuning")
        ax.set(xlabel="Global epoch", ylabel=ylabel, title=f"EfficientNetB0 - {ylabel}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"efficientnet_b0_{metric}_curve.png", dpi=300)
        plt.close(fig)


def print_best_epochs(label, records):
    loss_record, accuracy_record = records["loss"], records["accuracy"]
    print(
        f"{label} best-loss epoch: {loss_record['stage_epoch']} "
        f"(stage {loss_record['stage']}, global {loss_record['global_epoch']}) | "
        f"Validation loss: {loss_record['val_loss']:.4f} | "
        f"Validation accuracy: {loss_record['val_accuracy']:.2%}"
    )
    print(
        f"{label} best-accuracy epoch: {accuracy_record['stage_epoch']} "
        f"(stage {accuracy_record['stage']}, global {accuracy_record['global_epoch']}) | "
        f"Validation accuracy: {accuracy_record['val_accuracy']:.2%} | "
        f"Validation loss: {accuracy_record['val_loss']:.4f}"
    )


def main():
    set_seed(SEED)
    device = select_device()
    print(f"Using device: {device}\nSeed: {SEED}")
    create_output_dirs()
    train_loader, val_loader = build_dataloaders(device)
    metadata = experiment_metadata(train_loader, val_loader)
    metadata["device"] = str(device)
    model = build_model().to(device)
    print(f"Model: EfficientNetB0 ({MODEL_NAME}) | Weights: {WEIGHTS.name}")
    total_parameters = sum(p.numel() for p in model.parameters())
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    history = []
    global_best = {"loss": None, "accuracy": None}
    stage_summaries, trainable_counts = {}, {}

    with HISTORY_PATH.open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        for stage, max_epochs in [(1, STAGE1_EPOCHS), (2, STAGE2_EPOCHS)]:
            if stage == 1:
                set_stage1_trainable(model)
            else:
                # Before Stage 2, this path necessarily contains Stage 1's best-loss model.
                # Validation loss is the predefined initialization rule, not test performance.
                checkpoint = torch.load(BEST_LOSS_PATH, map_location="cpu", weights_only=True)
                model.load_state_dict(checkpoint["model_state_dict"])
                print(f"\nRestored Stage 1 best-loss checkpoint from epoch {checkpoint['epoch']}.")
                del checkpoint
                set_stage2_trainable(model)
                print(
                    f"Fine tuning feature groups: {list(range(len(model.features) - UNFREEZE_LAST_BLOCKS, len(model.features)))}"
                )

            verify_trainable_parameters(model, stage)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            trainable_counts[stage] = trainable
            print(
                f"\nStage {stage} | Total parameters: {total_parameters:,} | "
                f"Trainable: {trainable:,} | Frozen: {total_parameters - trainable:,}"
            )
            if stage == 1:
                sanity_check(model, train_loader, device)
            optimizer = build_optimizer(model, stage)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=LR_PATIENCE,
                threshold=MIN_DELTA,
                threshold_mode="abs",
                min_lr=(
                    [MIN_CLASSIFIER_LR] if stage == 1 else [MIN_BACKBONE_LR, MIN_CLASSIFIER_LR]
                ),
            )
            stage_best = {"loss": None, "accuracy": None}
            stopping_best_loss, epochs_without_improvement = float("inf"), 0
            for epoch in range(1, max_epochs + 1):
                learning_rates = {group["name"]: group["lr"] for group in optimizer.param_groups}
                train_loss, train_acc = train_one_epoch(
                    model, train_loader, criterion, optimizer, device
                )
                val_loss, val_acc = validate(model, val_loader, criterion, device)
                row = {
                    "stage": stage,
                    "stage_epoch": epoch,
                    "global_epoch": len(history) + 1,
                    "learning_rate_backbone": learning_rates.get("backbone", 0.0),
                    "learning_rate_classifier": learning_rates["classifier"],
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    # Online training accuracy includes augmentation and dropout.
                    # The gap is a fraction in both the CSV and console output.
                    "generalization_gap": train_acc - val_acc,
                }
                history.append(row)
                writer.writerow(row)
                history_file.flush()
                print(
                    f"Stage {stage} | Epoch {epoch:02d}/{max_epochs} | "
                    f"LR backbone: {row['learning_rate_backbone']:.2e} | "
                    f"LR classifier: {row['learning_rate_classifier']:.2e}\n"
                    f"  Train loss: {train_loss:.4f} | Train accuracy: {train_acc:.2%} | "
                    f"Val loss: {val_loss:.4f} | Val accuracy: {val_acc:.2%} | "
                    f"Generalization gap (fraction): {row['generalization_gap']:+.4f}"
                )
                if stage_best["loss"] is None or val_loss < stage_best["loss"]["val_loss"]:
                    stage_best["loss"] = row.copy()
                if (
                    stage_best["accuracy"] is None
                    or val_acc > stage_best["accuracy"]["val_accuracy"]
                ):
                    stage_best["accuracy"] = row.copy()

                # Keep both candidates across BOTH stages; fine tuning must earn an improvement.
                # Ties retain the earlier checkpoint; minimum loss is independent of min_delta.
                if global_best["loss"] is None or val_loss < global_best["loss"]["val_loss"]:
                    global_best["loss"] = row.copy()
                    save_checkpoint(model, BEST_LOSS_PATH, row, metadata, "validation-loss")
                if (
                    global_best["accuracy"] is None
                    or val_acc > global_best["accuracy"]["val_accuracy"]
                ):
                    global_best["accuracy"] = row.copy()
                    save_checkpoint(model, BEST_ACCURACY_PATH, row, metadata, "validation-accuracy")

                scheduler.step(val_loss)
                for group in optimizer.param_groups:
                    if group["lr"] < learning_rates[group["name"]]:
                        print(
                            f"  Reduced {group['name']} LR to {group['lr']:.2e} for the next epoch."
                        )
                # Loss reflects confidence as well as correctness. Use it for stopping,
                # while keeping the independent accuracy candidate for later reporting.
                if val_loss < stopping_best_loss - MIN_DELTA:
                    stopping_best_loss, epochs_without_improvement = val_loss, 0
                else:
                    epochs_without_improvement += 1
                if stage == 2 and epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                    print(
                        f"Early stopping Stage 2 after {EARLY_STOPPING_PATIENCE} epochs without meaningful loss improvement."
                    )
                    break
            stage_summaries[stage] = stage_best

    plot_history(history)
    print("\nEfficientNetB0 training completed")
    print_best_epochs("Stage 1", stage_summaries[1])
    print_best_epochs("Stage 2", stage_summaries[2])
    print_best_epochs("Overall", global_best)
    print(f"Final train accuracy: {history[-1]['train_accuracy']:.2%}")
    print(f"Total parameters: {total_parameters:,}")
    print(f"Trainable parameters during Stage 1: {trainable_counts[1]:,}")
    print(f"Trainable parameters during Stage 2: {trainable_counts[2]:,}")
    print(f"Best-loss model: {BEST_LOSS_PATH}\nBest-accuracy model: {BEST_ACCURACY_PATH}")
    print(f"History CSV: {HISTORY_PATH}\nPlot directory: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
