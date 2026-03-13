#!/usr/bin/env python3
"""按图片分辨率分类并输出统计 CSV。"""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按分辨率分类图片并导出统计 CSV")
    parser.add_argument("input_dir", type=Path, help="输入图片目录")
    parser.add_argument("output_dir", type=Path, help="输出目录（按分辨率分组）")
    parser.add_argument(
        "--stats-csv",
        type=Path,
        default=Path("resolution_stats.csv"),
        help="统计 CSV 输出路径（默认: resolution_stats.csv）",
    )
    return parser.parse_args()


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def safe_copy(src: Path, dst: Path) -> Path:
    """复制文件，若目标文件名冲突则自动加后缀。"""
    candidate = dst
    index = 1
    while candidate.exists():
        candidate = dst.with_name(f"{dst.stem}_{index}{dst.suffix}")
        index += 1
    shutil.copy2(src, candidate)
    return candidate


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    stats_csv: Path = args.stats_csv

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"输入目录不存在或不是目录: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if stats_csv.parent != Path(""):
        stats_csv.parent.mkdir(parents=True, exist_ok=True)

    resolution_groups: dict[str, int] = defaultdict(int)
    scanned = 0
    skipped = 0

    for path in input_dir.rglob("*"):
        if not is_image_file(path):
            continue

        scanned += 1
        try:
            with Image.open(path) as img:
                width, height = img.size
        except (UnidentifiedImageError, OSError):
            skipped += 1
            continue

        resolution = f"{width}x{height}"
        resolution_groups[resolution] += 1

        target_dir = output_dir / resolution
        target_dir.mkdir(parents=True, exist_ok=True)
        safe_copy(path, target_dir / path.name)

    with stats_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["resolution", "width", "height", "image_count"])
        for resolution, count in sorted(
            resolution_groups.items(), key=lambda item: item[1], reverse=True
        ):
            width, height = resolution.split("x")
            writer.writerow([resolution, width, height, count])

    print(f"扫描图片数量: {scanned}")
    print(f"跳过无效图片: {skipped}")
    print(f"输出目录: {output_dir}")
    print(f"统计文件: {stats_csv}")


if __name__ == "__main__":
    main()
