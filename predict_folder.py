#!/usr/bin/env python3
"""Run batch inference on top-level images in a folder."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLASS_NAMES = ["fail", "pass"]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict pass/fail for top-level images in a folder and export CSV.",
    )
    parser.add_argument("input_folder", type=Path, help="Folder with images (.jpg/.tif), top-level only")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/folder_predictions.csv"),
        help="Path to output CSV (default: outputs/folder_predictions.csv)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/best_model.pth"),
        help="Path to trained model checkpoint (default: models/best_model.pth)",
    )
    return parser.parse_args()


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_model(device: torch.device) -> nn.Module:
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(device)


def iter_top_level_images(input_folder: Path) -> Iterable[Path]:
    for path in sorted(input_folder.iterdir()):
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
            yield path


def predict_image(model: nn.Module, image_path: Path, transform: transforms.Compose, device: torch.device) -> tuple[str, float]:
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()

    return CLASS_NAMES[pred_idx], confidence


def main() -> None:
    args = parse_args()

    if not args.input_folder.exists() or not args.input_folder.is_dir():
        raise FileNotFoundError(f"Input folder not found: {args.input_folder}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    transform = build_transform()
    image_paths = list(iter_top_level_images(args.input_folder))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "predicted_label", "confidence"])
        writer.writeheader()

        for image_path in image_paths:
            label, confidence = predict_image(model, image_path, transform, device)
            writer.writerow(
                {
                    "image_path": str(image_path),
                    "predicted_label": label,
                    "confidence": f"{confidence:.6f}",
                }
            )

    print(f"Processed {len(image_paths)} image(s)")
    print(f"Saved predictions to: {args.output_csv}")


if __name__ == "__main__":
    main()
