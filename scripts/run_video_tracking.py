from __future__ import annotations

import os
from pathlib import Path


project_root = Path(__file__).resolve().parents[1]
config_dir = project_root / ".ultralytics"
model_path = project_root / "runs" / "detect" / "traffic_vehicle" / "weights" / "best.pt"

# Set this to the video you want to track.
video_path = project_root / "outputs" / "videos" / "infrastructure_3000.mp4"

# Tracking settings.
conf_threshold = 0.25
imgsz = 640
device = "0"
tracker_config = "bytetrack.yaml"
save_dir = project_root / "runs" / "track"
run_name = "traffic_vehicle_track"


def main() -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    from ultralytics import YOLO

    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    model = YOLO(str(model_path))

    output_dir: Path | None = None
    frame_count = 0
    total_tracks = 0

    results = model.track(
        source=str(video_path),
        conf=conf_threshold,
        imgsz=imgsz,
        device=device,
        tracker=tracker_config,
        project=str(save_dir),
        name=run_name,
        exist_ok=True,
        save=True,
        stream=True,
        verbose=True,
    )

    for result in results:
        frame_count += 1
        if output_dir is None:
            output_dir = Path(result.save_dir)
        if result.boxes is not None:
            ids = result.boxes.id
            if ids is not None:
                total_tracks += len(ids)

    if output_dir is None:
        raise RuntimeError("No tracking results were returned.")

    print(f"Tracked video saved to: {output_dir / video_path.name}")
    print(f"Processed {frame_count} frames.")
    print(f"Tracked detections across all frames: {total_tracks}")


if __name__ == "__main__":
    main()
