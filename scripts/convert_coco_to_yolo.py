from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from collections import defaultdict
from pathlib import Path


DEFAULT_CLASS_NAMES = ["car", "motorbike", "bus", "lorry"]
EPSILON = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a COCO dataset into YOLO detection format."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root folder containing the images referenced by the COCO file.",
    )
    parser.add_argument(
        "--coco-json",
        type=Path,
        required=True,
        help="Path to the COCO annotations JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output folder for the YOLO dataset.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_CLASS_NAMES,
        help="Class names to keep and remap into YOLO class ids.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio applied at the sequence-folder level.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy files instead of trying a hard-link first.",
    )
    return parser.parse_args()


def sanitize_name(name: str) -> str:
    return name.strip().lower()


def clip_bbox_to_image(bbox: list[float], width: int, height: int) -> tuple[float, float, float, float] | None:
    x_min, y_min, box_w, box_h = bbox
    x_max = x_min + box_w
    y_max = y_min + box_h

    x_min = min(max(x_min, 0.0), float(width))
    y_min = min(max(y_min, 0.0), float(height))
    x_max = min(max(x_max, 0.0), float(width))
    y_max = min(max(y_max, 0.0), float(height))

    clipped_w = x_max - x_min
    clipped_h = y_max - y_min
    if clipped_w <= 0 or clipped_h <= 0:
        return None
    return x_min, y_min, clipped_w, clipped_h


def yolo_box_from_coco_bbox(bbox: list[float], width: int, height: int) -> tuple[float, float, float, float] | None:
    clipped = clip_bbox_to_image(bbox, width, height)
    if clipped is None:
        return None

    x_min, y_min, box_w, box_h = clipped
    x_center = (x_min + box_w / 2.0) / width
    y_center = (y_min + box_h / 2.0) / height
    norm_w = box_w / width
    norm_h = box_h / height

    x_center = min(max(x_center, 0.0), 1.0 - EPSILON)
    y_center = min(max(y_center, 0.0), 1.0 - EPSILON)
    norm_w = min(max(norm_w, EPSILON), 1.0)
    norm_h = min(max(norm_h, EPSILON), 1.0)
    return x_center, y_center, norm_w, norm_h


def format_float(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def ensure_clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_output_stem(file_name: str) -> str:
    parts = Path(file_name).parts
    if len(parts) >= 3:
        return f"{parts[1]}__{Path(parts[-1]).stem}"
    return Path(file_name).stem


def link_or_copy(src: Path, dst: Path, force_copy: bool) -> None:
    if dst.exists():
        return
    ensure_clean_dir(dst.parent)
    if force_copy:
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def pick_val_groups(groups: list[str], val_ratio: float) -> set[str]:
    if len(groups) <= 1:
        return set()
    val_count = max(1, math.ceil(len(groups) * val_ratio))
    val_count = min(val_count, len(groups) - 1)
    return set(groups[-val_count:])


def main() -> None:
    args = parse_args()

    data = json.loads(args.coco_json.read_text(encoding="utf-8"))
    desired_classes = [sanitize_name(name) for name in args.classes]

    coco_categories = {
        category["id"]: sanitize_name(category["name"])
        for category in data["categories"]
    }
    kept_category_ids = {
        category_id: desired_classes.index(category_name)
        for category_id, category_name in coco_categories.items()
        if category_name in desired_classes
    }
    if not kept_category_ids:
        raise ValueError("None of the requested classes were found in the COCO file.")

    images_by_id = {image["id"]: image for image in data["images"]}
    labels_by_image_id: dict[int, list[str]] = defaultdict(list)

    for annotation in data["annotations"]:
        category_id = annotation["category_id"]
        if category_id not in kept_category_ids:
            continue

        image = images_by_id[annotation["image_id"]]
        width = image["width"]
        height = image["height"]
        yolo_box = yolo_box_from_coco_bbox(annotation["bbox"], width, height)
        if yolo_box is None:
            continue

        x_center, y_center, norm_w, norm_h = yolo_box
        yolo_class_id = kept_category_ids[category_id]
        labels_by_image_id[annotation["image_id"]].append(
            " ".join(
                [
                    str(yolo_class_id),
                    format_float(x_center),
                    format_float(y_center),
                    format_float(norm_w),
                    format_float(norm_h),
                ]
            )
        )

    grouped_image_ids: dict[str, list[int]] = defaultdict(list)
    for image_id, image in images_by_id.items():
        parts = Path(image["file_name"]).parts
        group_name = parts[1] if len(parts) > 1 else "default"
        grouped_image_ids[group_name].append(image_id)

    sorted_groups = sorted(grouped_image_ids.keys(), key=lambda name: (len(name), name))
    val_groups = pick_val_groups(sorted_groups, args.val_ratio)

    for split in ("train", "val"):
        ensure_clean_dir(args.output_dir / "images" / split)
        ensure_clean_dir(args.output_dir / "labels" / split)

    written_images = 0
    written_labels = 0

    for group_name in sorted_groups:
        split = "val" if group_name in val_groups else "train"
        for image_id in sorted(grouped_image_ids[group_name]):
            image = images_by_id[image_id]
            src_image = args.dataset_root / image["file_name"]
            if not src_image.exists():
                raise FileNotFoundError(f"Missing image referenced by COCO file: {src_image}")

            output_stem = build_output_stem(image["file_name"])
            dst_image = args.output_dir / "images" / split / f"{output_stem}{src_image.suffix}"
            dst_label = args.output_dir / "labels" / split / f"{output_stem}.txt"

            link_or_copy(src_image, dst_image, args.copy_images)
            written_images += 1

            label_lines = labels_by_image_id.get(image_id, [])
            dst_label.write_text("\n".join(label_lines), encoding="utf-8")
            written_labels += 1

    yaml_lines = [
        f"path: {args.output_dir.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        f"names: {json.dumps(desired_classes)}",
    ]
    (args.output_dir / "dataset.yaml").write_text(
        "\n".join(yaml_lines) + "\n", encoding="utf-8"
    )

    print(f"Kept classes: {desired_classes}")
    print(f"Train groups: {[group for group in sorted_groups if group not in val_groups]}")
    print(f"Val groups: {sorted(val_groups, key=lambda name: (len(name), name))}")
    print(f"Wrote {written_images} images and {written_labels} label files to {args.output_dir}")


if __name__ == "__main__":
    main()
