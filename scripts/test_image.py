from __future__ import annotations

import os
from pathlib import Path

import cv2
from PIL import Image


project_root = Path(__file__).resolve().parents[1]
config_dir = project_root / ".ultralytics"
model_path = project_root / "runs" / "detect" / "traffic_vehicle" / "weights" / "best.pt"

# Use either a full absolute Windows path or a path relative to project_root.
# Example absolute path:
# image_path_input = r"D:/Ahmed_Data/TTE/Project/photo-1552101066-0d2fcca437d1.png"
# Example relative path:
# image_path_input = r"yolo_datasets/combined_vehicle/images/val/drone__3200__seq3-drone_0003201.jpg"
image_path_input = r"D:\Ahmed_Data\TTE\Project\360_F_233959210_hR5f8Nm7lM5yHgZ3KczKSefD7obzZLs6.jpg"

# Inference settings.
conf_threshold = 0.25
imgsz = 640
device = "0"
save_dir = project_root / "runs" / "predict"
run_name = "traffic_vehicle_test"
temp_input_dir = project_root / ".tmp_predict_inputs"


def resolve_image_path(path_input: str) -> Path:
    candidate = Path(path_input)
    if candidate.is_absolute():
        return candidate
    return project_root / candidate


def prepare_source_image(image_path: Path) -> tuple[Path, bool]:
    """Return a YOLO-readable image path, converting with PIL if OpenCV cannot read it."""
    image_for_cv2 = cv2.imread(str(image_path))
    if image_for_cv2 is not None:
        return image_path, False

    temp_input_dir.mkdir(parents=True, exist_ok=True)
    converted_path = temp_input_dir / f"{image_path.stem}_converted.png"
    with Image.open(image_path) as img:
        img.convert("RGB").save(converted_path, format="PNG")
    return converted_path, True


def main() -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)

    from ultralytics import YOLO

    image_path = resolve_image_path(image_path_input)

    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Test image not found: {image_path}")

    source_path, was_converted = prepare_source_image(image_path)
    if was_converted:
        print(f"Converted unsupported image format to temporary PNG: {source_path}")

    model = YOLO(str(model_path))
    results = model.predict(
        source=str(source_path),
        conf=conf_threshold,
        imgsz=imgsz,
        device=device,
        project=str(save_dir),
        name=run_name,
        exist_ok=True,
        save=True,
        save_txt=True,
        save_conf=True,
        verbose=True,
    )

    if not results:
        raise RuntimeError("No prediction results were returned.")

    output_dir = Path(results[0].save_dir)
    output_image_path = output_dir / source_path.name
    output_label_path = output_dir / 'labels' / f'{source_path.stem}.txt'
    print(f"Prediction image saved to: {output_image_path}")
    print(f"Prediction labels saved to: {output_label_path}")
    if was_converted:
        print(f"Original input image was: {image_path}")


if __name__ == "__main__":
    main()
