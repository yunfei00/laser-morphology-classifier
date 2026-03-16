#!/usr/bin/env python3
"""Normalize raw Poyang image filenames and export manifest CSV."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

SCORES = ("40", "60", "80")
SUPPORTED_EXTENSIONS = {".jpg", ".tif"}
DEFAULT_INPUT_ROOT = Path("data/raw/poyang")
DEFAULT_OUTPUT_ROOT = Path("data/interim/poyang_renamed")
DEFAULT_MANIFEST = Path("data/interim/poyang_renamed_manifest.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize raw Poyang image filenames and write manifest CSV"
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Raw Poyang root containing score folders 40/60/80",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root for normalized images",
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Manifest CSV output path",
    )
    return parser.parse_args()


def list_score_files(score_dir: Path) -> list[Path]:
    if not score_dir.exists() or not score_dir.is_dir():
        raise SystemExit(f"Score directory missing: {score_dir}")

    files = [
        path
        for path in score_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files, key=lambda p: p.name.lower())


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    input_root: Path = args.input_root
    output_root: Path = args.output_root
    manifest_csv: Path = args.manifest_csv

    if not input_root.exists() or not input_root.is_dir():
        raise SystemExit(f"Input root does not exist or is not a directory: {input_root}")

    ensure_dir(output_root)
    if manifest_csv.parent != Path(""):
        ensure_dir(manifest_csv.parent)

    manifest_rows: list[dict[str, str]] = []

    for score in SCORES:
        source_dir = input_root / score
        target_dir = output_root / score
        ensure_dir(target_dir)

        files = list_score_files(source_dir)
        for index, src in enumerate(files, start=1):
            ext = src.suffix.lower().lstrip(".")
            new_name = f"poyang_{score}_{index:04d}.{ext}"
            dst = target_dir / new_name
            shutil.copy2(src, dst)

            manifest_rows.append(
                {
                    "source_path": str(src),
                    "source_name": src.name,
                    "score": score,
                    "new_name": new_name,
                    "target_path": str(dst),
                }
            )

    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source_path", "source_name", "score", "new_name", "target_path"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Copied files: {len(manifest_rows)}")
    print(f"Output root: {output_root}")
    print(f"Manifest CSV: {manifest_csv}")


if __name__ == "__main__":
    main()
