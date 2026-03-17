#!/usr/bin/env python3
"""Train baseline ResNet18 binary classifier on cleaned Poyang dataset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train baseline ResNet18 for binary classification on cleaned Poyang samples.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/processed/dataset_binary"),
        help="Dataset root that contains train/ val/ test folders.",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num-workers", type=int, default=2, help="Dataloader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/best_model.pth"),
        help="Best model output path.",
    )
    parser.add_argument(
        "--log-csv",
        type=Path,
        default=Path("outputs/train_log.csv"),
        help="Training log CSV path.",
    )
    parser.add_argument(
        "--test-metrics-json",
        type=Path,
        default=Path("outputs/test_metrics.json"),
        help="Test metrics JSON path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run one forward pass and data checks without full training.",
    )
    return parser.parse_args()


def build_transforms() -> Dict[str, transforms.Compose]:
    train_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return {"train": train_tf, "val": eval_tf, "test": eval_tf}


def load_datasets(data_root: Path, tfms: Dict[str, transforms.Compose]) -> Dict[str, datasets.ImageFolder]:
    datasets_map: Dict[str, datasets.ImageFolder] = {}
    for split in ["train", "val", "test"]:
        split_dir = data_root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")
        datasets_map[split] = datasets.ImageFolder(split_dir, transform=tfms[split])
    return datasets_map


def build_dataloaders(
    datasets_map: Dict[str, datasets.ImageFolder],
    batch_size: int,
    num_workers: int,
) -> Dict[str, DataLoader]:
    return {
        "train": DataLoader(
            datasets_map["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "val": DataLoader(
            datasets_map["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "test": DataLoader(
            datasets_map["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
    }


def compute_class_weights(train_dataset: datasets.ImageFolder, device: torch.device) -> torch.Tensor:
    num_classes = len(train_dataset.classes)
    counts = torch.bincount(torch.tensor(train_dataset.targets), minlength=num_classes).float()
    total = counts.sum()
    weights = total / (num_classes * counts.clamp_min(1.0))
    return weights.to(device)


def build_model(device: torch.device) -> nn.Module:
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    return model.to(device)


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> Tuple[float, List[int], List[int]]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_samples = 0
    all_true: List[int] = []
    all_pred: List[int] = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        preds = outputs.argmax(dim=1)
        all_true.extend(labels.detach().cpu().tolist())
        all_pred.extend(preds.detach().cpu().tolist())

    avg_loss = total_loss / max(total_samples, 1)
    return avg_loss, all_true, all_pred


def evaluate_test(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Dict[str, object]:
    model.eval()
    all_true: List[int] = []
    all_pred: List[int] = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_pred.extend(preds)
            all_true.extend(labels.tolist())

    metrics = {
        "accuracy": accuracy_score(all_true, all_pred),
        "precision": precision_score(all_true, all_pred, zero_division=0),
        "recall": recall_score(all_true, all_pred, zero_division=0),
        "f1": f1_score(all_true, all_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(all_true, all_pred).tolist(),
    }
    return metrics


def save_train_log(rows: List[Dict[str, float]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_accuracy"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transforms_map = build_transforms()
    datasets_map = load_datasets(args.data_root, transforms_map)

    if len(datasets_map["train"].classes) != 2:
        raise ValueError("Expected exactly 2 classes for binary classification.")

    print(f"Class to index: {datasets_map['train'].class_to_idx}")
    print(
        f"Dataset sizes - train: {len(datasets_map['train'])}, "
        f"val: {len(datasets_map['val'])}, test: {len(datasets_map['test'])}"
    )

    dataloaders = build_dataloaders(datasets_map, args.batch_size, args.num_workers)
    model = build_model(device)

    if args.dry_run:
        images, labels = next(iter(dataloaders["train"]))
        with torch.no_grad():
            outputs = model(images.to(device))
        print(
            "Dry run successful - "
            f"batch shape: {tuple(images.shape)}, labels shape: {tuple(labels.shape)}, "
            f"logits shape: {tuple(outputs.shape)}"
        )
        return

    class_weights = compute_class_weights(datasets_map["train"], device)
    print(f"Class weights: {class_weights.detach().cpu().tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    history: List[Dict[str, float]] = []
    best_val_acc = -1.0

    args.model_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = run_epoch(model, dataloaders["train"], criterion, optimizer, device)
        val_loss, val_true, val_pred = run_epoch(model, dataloaders["val"], criterion, None, device)
        val_acc = accuracy_score(val_true, val_pred)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | val_accuracy: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.model_path)
            print(f"Saved best model to {args.model_path} (val_accuracy={val_acc:.4f})")

    save_train_log(history, args.log_csv)
    print(f"Training log saved to {args.log_csv}")

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    test_metrics = evaluate_test(model, dataloaders["test"], device)

    args.test_metrics_json.parent.mkdir(parents=True, exist_ok=True)
    with args.test_metrics_json.open("w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)

    print("Test metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value}")
    print(f"Test metrics saved to {args.test_metrics_json}")


if __name__ == "__main__":
    main()
