#!/usr/bin/env python3
"""Run inference for one image using trained ResNet18 checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

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
        description="Predict pass/fail for one image with models/best_model.pth",
    )
    parser.add_argument("image_path", type=Path, help="Path to input image (.jpg/.tif)")
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


def validate_image_path(path: Path) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    if path.suffix.lower() not in VALID_EXTENSIONS:
        raise ValueError(f"Unsupported image format: {path.suffix}. Use .jpg or .tif")


def main() -> None:
    args = parse_args()
    validate_image_path(args.image_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device)

    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    image = Image.open(args.image_path).convert("RGB")
    image_tensor = build_transform()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()

    print(f"image_path: {args.image_path}")
    print(f"predicted_label: {CLASS_NAMES[pred_idx]}")
    print(f"confidence: {confidence:.6f}")


if __name__ == "__main__":
    main()
