from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a fresh train/val split from an existing YOLO dataset."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
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


def load_class_names(dataset_dir: Path) -> list[str]:
    dataset_yaml = dataset_dir / "dataset.yaml"
    for line in dataset_yaml.read_text(encoding="utf-8").splitlines():
        if line.startswith("names:"):
            return json.loads(line.split(":", 1)[1].strip())
    raise ValueError(f"Could not find class names in {dataset_yaml}")


def source_group(stem: str) -> str:
    return stem.split("__", 1)[0] if "__" in stem else "default"


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    rng = random.Random(args.seed)

    if not (0.0 < args.val_ratio < 1.0):
        raise ValueError("val-ratio must be between 0 and 1.")

    class_names = load_class_names(input_dir)

    all_images: list[Path] = []
    label_lookup: dict[str, Path] = {}
    grouped_stems: dict[str, list[str]] = defaultdict(list)

    for split in ["train", "val"]:
        image_dir = input_dir / "images" / split
        label_dir = input_dir / "labels" / split
        for image_path in sorted(path for path in image_dir.iterdir() if path.is_file()):
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(f"Missing label file for {image_path.name}: {label_path}")
            all_images.append(image_path)
            label_lookup[image_path.stem] = label_path
            grouped_stems[source_group(image_path.stem)].append(image_path.stem)

    split_assignment: dict[str, str] = {}
    for group_name, stems in grouped_stems.items():
        stems = stems[:]
        rng.shuffle(stems)
        val_count = max(1, round(len(stems) * args.val_ratio)) if len(stems) > 1 else 0
        val_count = min(val_count, len(stems) - 1) if len(stems) > 1 else 0
        val_stems = set(stems[:val_count])
        for stem in stems:
            split_assignment[stem] = "val" if stem in val_stems else "train"

    for split in ["train", "val"]:
        ensure_dir(output_dir / "images" / split)
        ensure_dir(output_dir / "labels" / split)

    counts = {"train": 0, "val": 0}
    for image_path in all_images:
        target_split = split_assignment[image_path.stem]
        dst_image = output_dir / "images" / target_split / image_path.name
        dst_label = output_dir / "labels" / target_split / f"{image_path.stem}.txt"
        link_or_copy(image_path, dst_image, args.copy_files)
        link_or_copy(label_lookup[image_path.stem], dst_label, args.copy_files)
        counts[target_split] += 1

    yaml_lines = [
        f"path: {output_dir.as_posix()}",
        "train: images/train",
        "val: images/val",
        f"names: {json.dumps(class_names)}",
    ]
    (output_dir / "dataset.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    print(f"Created resplit dataset at: {output_dir}")
    print(f"train: {counts['train']} images")
    print(f"val: {counts['val']} images")
    print(f"groups: {sorted(grouped_stems)}")


if __name__ == "__main__":
    main()
