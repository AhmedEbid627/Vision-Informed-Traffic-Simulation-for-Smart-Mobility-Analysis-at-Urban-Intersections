from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge multiple YOLO datasets with the same class mapping into one dataset."
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        required=True,
        help="Input YOLO dataset directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for the merged YOLO dataset.",
    )
    parser.add_argument(
        "--copy-files",
        action="store_true",
        help="Copy files instead of trying hard-links first.",
    )
    return parser.parse_args()


def load_names(dataset_dir: Path) -> list[str]:
    dataset_yaml = dataset_dir / "dataset.yaml"
    for line in dataset_yaml.read_text(encoding="utf-8").splitlines():
        if line.startswith("names:"):
            return json.loads(line.split(":", 1)[1].strip())
    raise ValueError(f"Could not find class names in {dataset_yaml}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, force_copy: bool) -> None:
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")
    ensure_dir(dst.parent)
    if force_copy:
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    input_dirs = [path.resolve() for path in args.inputs]
    output_dir = args.output_dir.resolve()

    if not input_dirs:
        raise ValueError("At least one input dataset is required.")

    class_names = load_names(input_dirs[0])
    for dataset_dir in input_dirs[1:]:
        other_names = load_names(dataset_dir)
        if other_names != class_names:
            raise ValueError(
                f"Class mismatch between {input_dirs[0]} and {dataset_dir}: {class_names} != {other_names}"
            )

    for split in ["train", "val"]:
        ensure_dir(output_dir / "images" / split)
        ensure_dir(output_dir / "labels" / split)

    counts: dict[str, int] = {"train": 0, "val": 0}

    for dataset_dir in input_dirs:
        prefix = dataset_dir.name
        for split in ["train", "val"]:
            image_dir = dataset_dir / "images" / split
            label_dir = dataset_dir / "labels" / split
            image_paths = sorted(path for path in image_dir.iterdir() if path.is_file())
            for image_path in image_paths:
                output_stem = f"{prefix}__{image_path.stem}"
                dst_image = output_dir / "images" / split / f"{output_stem}{image_path.suffix}"
                dst_label = output_dir / "labels" / split / f"{output_stem}.txt"
                src_label = label_dir / f"{image_path.stem}.txt"
                if not src_label.exists():
                    raise FileNotFoundError(f"Missing label for {image_path}: {src_label}")
                link_or_copy(image_path, dst_image, args.copy_files)
                link_or_copy(src_label, dst_label, args.copy_files)
                counts[split] += 1

    yaml_lines = [
        f"path: {output_dir.as_posix()}",
        "train: images/train",
        "val: images/val",
        f"names: {json.dumps(class_names)}",
    ]
    (output_dir / "dataset.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    print(f"Merged datasets: {[p.name for p in input_dirs]}")
    print(f"Train images: {counts['train']}")
    print(f"Val images: {counts['val']}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
