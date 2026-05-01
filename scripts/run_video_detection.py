from __future__ import annotations

import os
from pathlib import Path


project_root = Path(__file__).resolve().parents[1]
config_dir = project_root / ".ultralytics"
model_path = project_root / "runs" / "detect" / "traffic_vehicle" / "weights" / "best.pt"

# Set this to the video you want to test.
video_path = project_root / "outputs" / "videos" / "infrastructure_3000.mp4"

# Inference settings.
conf_threshold = 0.25
imgsz = 640
device = "0"
save_dir = project_root / "runs" / "predict"
run_name = "traffic_vehicle_video"


def main() -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)

    from ultralytics import YOLO

    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    model = YOLO(str(model_path))
    results = model.predict(
        source=str(video_path),
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
    print(f"Annotated video saved to: {output_dir / video_path.name}")


if __name__ == "__main__":
    main()
