from __future__ import annotations

import argparse
import os
from pathlib import Path


project_root = Path(__file__).resolve().parents[1]
config_dir = project_root / ".ultralytics"
default_model_path = project_root / "models" / "traffic_vehicle_best.pt"
default_video_path = project_root / "examples" / "sample_intersection.mp4"

# Inference settings.
conf_threshold = 0.25
imgsz = 640
device = "0"
save_dir = project_root / "runs" / "predict"
run_name = "traffic_vehicle_video"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run vehicle detection on a video.")
    parser.add_argument("--model", type=Path, default=default_model_path, help="Path to YOLO weights.")
    parser.add_argument("--video", type=Path, default=default_video_path, help="Path to input video.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)

    from ultralytics import YOLO

    if not args.model.exists():
        raise FileNotFoundError(f"Model weights not found: {args.model}")
    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")

    model = YOLO(str(args.model))
    results = model.predict(
        source=str(args.video),
        conf=conf_threshold,
        imgsz=imgsz,
        device=device,
        project=str(save_dir),
        name=run_name,
        exist_ok=True,
        save=True,
        verbose=True,
    )

    if not results:
        raise RuntimeError("No prediction results were returned.")

    output_dir = Path(results[0].save_dir)
    print(f"Annotated video saved to: {output_dir / args.video.name}")


if __name__ == "__main__":
    main()
