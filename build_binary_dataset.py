#!/usr/bin/env python3
"""Build a binary dataset from cleaned Poyang samples."""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import Counter
from pathlib import Path

SCORES = ("40", "60", "80")
SCORE_TO_BINARY = {"40": "fail", "60": "pass", "80": "pass"}
SUPPORTED_EXTENSIONS = {".jpg", ".tif"}
DEFAULT_INPUT_ROOT = Path("data/interim/poyang_renamed")
DEFAULT_OUTPUT_ROOT = Path("data/processed/dataset_binary")
DEFAULT_MANIFEST = Path("data/processed/dataset_binary_manifest.csv")
DEFAULT_STATS = Path("data/processed/dataset_binary_stats.csv")
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build train/val/test binary dataset from Poyang score folders"
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Input root containing score folders 40/60/80",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root for split binary dataset",
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Manifest CSV output path",
    )
    parser.add_argument(
        "--stats-csv",
        type=Path,
        default=DEFAULT_STATS,
        help="Stats CSV output path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for deterministic split",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_score_files(score_dir: Path) -> list[Path]:
    if not score_dir.exists() or not score_dir.is_dir():
        raise SystemExit(f"Score directory missing: {score_dir}")

    files = [
        path
        for path in score_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files, key=lambda p: p.name.lower())


def split_items(items: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    total = len(items)
    train_count = int(total * 0.70)
    val_count = int(total * 0.15)
    test_count = total - train_count - val_count

    return {
        "train": items[:train_count],
        "val": items[train_count : train_count + val_count],
        "test": items[train_count + val_count : train_count + val_count + test_count],
    }


def unique_target_path(base_path: Path) -> Path:
    if not base_path.exists():
        return base_path

    index = 1
    candidate = base_path
    while candidate.exists():
        candidate = base_path.with_name(f"{base_path.stem}_{index}{base_path.suffix}")
        index += 1
    return candidate


def main() -> None:
    args = parse_args()

    input_root: Path = args.input_root
    output_root: Path = args.output_root
    manifest_csv: Path = args.manifest_csv
    stats_csv: Path = args.stats_csv
    seed: int = args.seed

    if not input_root.exists() or not input_root.is_dir():
        raise SystemExit(f"Input root does not exist or is not a directory: {input_root}")

    all_items: list[dict[str, str]] = []
    for score in SCORES:
        source_dir = input_root / score
        files = list_score_files(source_dir)
        binary_label = SCORE_TO_BINARY[score]
        for src in files:
            all_items.append(
                {
                    "source_path": str(src),
                    "source_name": src.name,
                    "score": score,
                    "binary_label": binary_label,
                }
            )

    rng = random.Random(seed)
    split_groups: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}

    for label in ("fail", "pass"):
        label_items = [item for item in all_items if item["binary_label"] == label]
        rng.shuffle(label_items)
        split_result = split_items(label_items)
        for split_name, entries in split_result.items():
            split_groups[split_name].extend(entries)

    for split_name in split_groups:
        split_groups[split_name].sort(key=lambda item: item["source_name"].lower())

    for split_name in ("train", "val", "test"):
        for label in ("fail", "pass"):
            ensure_dir(output_root / split_name / label)

    if manifest_csv.parent != Path(""):
        ensure_dir(manifest_csv.parent)
    if stats_csv.parent != Path(""):
        ensure_dir(stats_csv.parent)

    manifest_rows: list[dict[str, str]] = []
    stats_counter: Counter[tuple[str, str]] = Counter()

    for split_name in ("train", "val", "test"):
        for item in split_groups[split_name]:
            src = Path(item["source_path"])
            label = item["binary_label"]
            dst_dir = output_root / split_name / label
            dst = unique_target_path(dst_dir / src.name)
            shutil.copy2(src, dst)

            manifest_rows.append(
                {
                    "source_path": item["source_path"],
                    "source_name": item["source_name"],
                    "score": item["score"],
                    "binary_label": label,
                    "split": split_name,
                    "target_path": str(dst),
                }
            )
            stats_counter[(split_name, label)] += 1

    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_path",
                "source_name",
                "score",
                "binary_label",
                "split",
                "target_path",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    with stats_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "binary_label", "image_count"])
        for split_name in ("train", "val", "test"):
            for label in ("fail", "pass"):
                writer.writerow([split_name, label, stats_counter[(split_name, label)]])

    print(f"Total images: {len(all_items)}")
    print(f"Output root: {output_root}")
    print(f"Manifest CSV: {manifest_csv}")
    print(f"Stats CSV: {stats_csv}")


if __name__ == "__main__":
    main()
