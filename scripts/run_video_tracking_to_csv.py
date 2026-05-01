from __future__ import annotations

import csv
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
run_name = "traffic_vehicle_track_csv"


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

    rows: list[list[object]] = []
    output_dir: Path | None = None

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

    for frame_index, result in enumerate(results, start=1):
        if output_dir is None:
            output_dir = Path(result.save_dir)

        boxes = result.boxes
        if boxes is None or boxes.xyxy is None or boxes.conf is None:
            continue

        ids = boxes.id
        xyxy = boxes.xyxy.tolist()
        confs = boxes.conf.tolist()

        for det_index, (bbox, conf) in enumerate(zip(xyxy, confs)):
            track_id = int(ids[det_index]) if ids is not None else -1
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            width = x2 - x1
            height = y2 - y1
            rows.append(
                [
                    frame_index,
                    track_id,
                    round(conf, 6),
                    round(x1, 3),
                    round(y1, 3),
                    round(x2, 3),
                    round(y2, 3),
                    round(center_x, 3),
                    round(center_y, 3),
                    round(width, 3),
                    round(height, 3),
                ]
            )

    if output_dir is None:
        raise RuntimeError("No tracking results were returned.")

    csv_path = output_dir / f"{video_path.stem}_tracks.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame",
                "track_id",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
                "center_x",
                "center_y",
                "width",
                "height",
            ]
        )
        writer.writerows(rows)

    print(f"Tracked video saved to: {output_dir / video_path.name}")
    print(f"Tracking CSV saved to: {csv_path}")
    print(f"CSV rows written: {len(rows)}")


if __name__ == "__main__":
    main()
