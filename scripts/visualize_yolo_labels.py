from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2


COLORS = [
    (50, 205, 50),
    (0, 215, 255),
    (255, 140, 0),
    (147, 20, 255),
    (255, 99, 71),
    (64, 224, 208),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw YOLO bounding boxes on images for quick inspection."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Root YOLO dataset directory containing dataset.yaml, images/, and labels/.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="train",
        help="Dataset split to visualize.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=12,
        help="Number of images to visualize.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where rendered previews will be saved.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    return parser.parse_args()


def load_class_names(dataset_dir: Path) -> list[str]:
    dataset_yaml = dataset_dir / "dataset.yaml"
    lines = dataset_yaml.read_text(encoding="utf-8").splitlines()
    for line in lines:
        if line.startswith("names:"):
            return json.loads(line.split(":", 1)[1].strip())
    raise ValueError(f"Could not find class names in {dataset_yaml}")


def parse_label_file(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    labels: list[tuple[int, float, float, float, float]] = []
    if not label_path.exists():
        return labels
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        class_id, x_center, y_center, width, height = line.split()
        labels.append(
            (
                int(class_id),
                float(x_center),
                float(y_center),
                float(width),
                float(height),
            )
        )
    return labels


def yolo_to_xyxy(
    x_center: float, y_center: float, width: float, height: float, image_width: int, image_height: int
) -> tuple[int, int, int, int]:
    box_width = width * image_width
    box_height = height * image_height
    center_x = x_center * image_width
    center_y = y_center * image_height

    x1 = int(round(center_x - box_width / 2))
    y1 = int(round(center_y - box_height / 2))
    x2 = int(round(center_x + box_width / 2))
    y2 = int(round(center_y + box_height / 2))
    return x1, y1, x2, y2


def draw_boxes(image_path: Path, label_path: Path, class_names: list[str], output_path: Path) -> int:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image_height, image_width = image.shape[:2]
    labels = parse_label_file(label_path)

    for class_id, x_center, y_center, width, height in labels:
        x1, y1, x2, y2 = yolo_to_xyxy(
            x_center, y_center, width, height, image_width, image_height
        )
        color = COLORS[class_id % len(COLORS)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        class_name = class_names[class_id] if class_id < len(class_names) else str(class_id)
        label = class_name
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_w, text_h = text_size
        box_top = max(0, y1 - text_h - 10)
        cv2.rectangle(image, (x1, box_top), (x1 + text_w + 8, box_top + text_h + 8), color, -1)
        cv2.putText(
            image,
            label,
            (x1 + 4, box_top + text_h + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    return len(labels)


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir or dataset_dir / "previews" / args.split

    class_names = load_class_names(dataset_dir)
    image_dir = dataset_dir / "images" / args.split
    label_dir = dataset_dir / "labels" / args.split

    image_paths = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")

    sample_count = min(args.count, len(image_paths))
    rng = random.Random(args.seed)
    selected_images = rng.sample(image_paths, sample_count)

    print(f"Saving {sample_count} preview images to {output_dir}")
    for image_path in selected_images:
        label_path = label_dir / f"{image_path.stem}.txt"
        output_path = output_dir / image_path.name
        label_count = draw_boxes(image_path, label_path, class_names, output_path)
        print(f"{image_path.name}: {label_count} boxes")


if __name__ == "__main__":
    main()
