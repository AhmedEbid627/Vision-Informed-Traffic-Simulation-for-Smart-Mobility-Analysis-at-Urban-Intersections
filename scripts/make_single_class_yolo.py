from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


TARGET_CLASS_ID = 0
TARGET_CLASS_NAME = "vehicle"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a single-class YOLO dataset by remapping all labels to class 0."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Input YOLO dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output YOLO dataset directory.",
    )
    parser.add_argument(
        "--copy-files",
        action="store_true",
        help="Copy files instead of trying hard-links first.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, force_copy: bool) -> None:
    if dst.exists():
        return
    ensure_dir(dst.parent)
    if force_copy:
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def remap_label_file(src: Path, dst: Path) -> int:
    lines_out: list[str] = []
    object_count = 0
    for raw_line in src.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        _, x_center, y_center, width, height = parts
        lines_out.append(f"{TARGET_CLASS_ID} {x_center} {y_center} {width} {height}")
        object_count += 1
    dst.write_text("\n".join(lines_out), encoding="utf-8")
    return object_count


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    total_images = 0
    total_labels = 0
    total_objects = 0

    for split in ["train", "val"]:
        input_image_dir = input_dir / "images" / split
        input_label_dir = input_dir / "labels" / split
        output_image_dir = output_dir / "images" / split
        output_label_dir = output_dir / "labels" / split
        ensure_dir(output_image_dir)
        ensure_dir(output_label_dir)

        image_paths = sorted(path for path in input_image_dir.iterdir() if path.is_file())
        for image_path in image_paths:
            label_path = input_label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(f"Missing label file for {image_path.name}: {label_path}")

            dst_image = output_image_dir / image_path.name
            dst_label = output_label_dir / label_path.name
            link_or_copy(image_path, dst_image, args.copy_files)
            total_images += 1
            total_labels += 1
            total_objects += remap_label_file(label_path, dst_label)

    yaml_lines = [
        f"path: {output_dir.as_posix()}",
        "train: images/train",
        "val: images/val",
        'names: ["vehicle"]',
    ]
    (output_dir / "dataset.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    print(f"Created single-class dataset at: {output_dir}")
    print(f"Images: {total_images}")
    print(f"Label files: {total_labels}")
    print(f"Objects remapped to '{TARGET_CLASS_NAME}': {total_objects}")


if __name__ == "__main__":
    main()
