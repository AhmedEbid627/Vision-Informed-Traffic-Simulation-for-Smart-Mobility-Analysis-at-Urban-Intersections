from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


TARGET_CLASS_ID = 0
TARGET_CLASS_NAME = "vehicle"
SPLIT_ALIASES = {
    "train": "train",
    "val": "val",
    "valid": "val",
    "test": "test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a single-class YOLO dataset copy from an existing YOLO dataset."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--copy-files", action="store_true")
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


def discover_splits(input_dir: Path) -> dict[str, str]:
    discovered: dict[str, str] = {}
    for child in input_dir.iterdir():
        if not child.is_dir():
            continue
        mapped = SPLIT_ALIASES.get(child.name.lower())
        if mapped:
            discovered[child.name] = mapped
    if "train" not in discovered.values() or "val" not in discovered.values():
        raise ValueError("Input dataset must contain train and val/valid splits.")
    return discovered


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    split_map = discover_splits(input_dir)
    split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    total_objects = 0

    for source_split, target_split in split_map.items():
        input_image_dir = input_dir / source_split / "images"
        input_label_dir = input_dir / source_split / "labels"
        output_image_dir = output_dir / "images" / target_split
        output_label_dir = output_dir / "labels" / target_split
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
            total_objects += remap_label_file(label_path, dst_label)
            split_counts[target_split] += 1

    yaml_lines = [
        f"path: {output_dir.as_posix()}",
        "train: images/train",
        "val: images/val",
    ]
    if split_counts["test"]:
        yaml_lines.append("test: images/test")
    yaml_lines.append('names: ["vehicle"]')
    (output_dir / "dataset.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    print(f"Created single-class dataset at: {output_dir}")
    for split in ["train", "val", "test"]:
        if split_counts[split]:
            print(f"{split}: {split_counts[split]} images")
    print(f"Objects remapped to '{TARGET_CLASS_NAME}': {total_objects}")


if __name__ == "__main__":
    main()
