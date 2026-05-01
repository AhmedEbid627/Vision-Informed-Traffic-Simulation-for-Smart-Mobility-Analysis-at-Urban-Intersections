from __future__ import annotations

import csv
import os
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import subprocess


project_root = Path(__file__).resolve().parents[1]
config_dir = project_root / ".ultralytics"
model_path = project_root / "runs" / "detect" / "traffic_vehicle" / "weights" / "best.pt"
dataset_root = project_root / "Dataset" / "Infrastructure"
video_output_dir = project_root / "outputs" / "videos" / "infrastructure_batch"
track_output_dir = project_root / "runs" / "track" / "infrastructure_batch"
analysis_output_dir = track_output_dir / "analysis"

# Processing settings.
fps = 10.0
conf_threshold = 0.25
imgsz = 640
device = "0"
tracker_config = "bytetrack.yaml"
run_name = "infrastructure_batch_track"

# Metrics settings shared across infrastructure clips.
line_start = (700, 0)
line_end = (700, 639)
polygon_points = [
    (119, 196),
    (628, 636),
    (518, 639),
    (6, 636),
    (1, 268),
]
polygon_array = np.array(polygon_points, dtype=np.int32)

valid_suffixes = {".jpg", ".jpeg", ".png"}


def numeric_sequence_dirs(root: Path) -> list[Path]:
    return sorted(
        [path for path in root.iterdir() if path.is_dir() and path.name.isdigit()],
        key=lambda path: int(path.name),
    )


def build_video_from_frames(sequence_dir: Path, output_path: Path) -> int:
    image_paths = sorted(
        path for path in sequence_dir.iterdir() if path.is_file() and path.suffix.lower() in valid_suffixes
    )
    if not image_paths:
        raise FileNotFoundError(f"No frames found in {sequence_dir}")

    first_frame = cv2.imread(str(image_paths[0]))
    if first_frame is None:
        raise ValueError(f"Could not read first frame: {image_paths[0]}")

    height, width = first_frame.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    written = 0
    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)
        written += 1

    writer.release()
    return written


def track_video_to_rows(video_path: Path) -> tuple[Path, list[list[object]], int]:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    rows: list[list[object]] = []
    output_dir: Path | None = None
    frame_count = 0

    results = model.track(
        source=str(video_path),
        conf=conf_threshold,
        imgsz=imgsz,
        device=device,
        tracker=tracker_config,
        project=str(track_output_dir),
        name=run_name,
        exist_ok=True,
        save=True,
        stream=True,
        verbose=False,
    )

    for frame_index, result in enumerate(results, start=1):
        frame_count += 1
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
        raise RuntimeError(f"No tracking results were returned for {video_path}")

    return output_dir, rows, frame_count


def save_tracking_csv(csv_path: Path, rows: list[list[object]]) -> None:
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


def point_side(x: float, y: float, start: tuple[int, int], end: tuple[int, int]) -> float:
    x1, y1 = start
    x2, y2 = end
    return (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)


def line_label(side_before: float, side_after: float) -> str:
    if side_before < 0 and side_after > 0:
        return "negative_to_positive"
    if side_before > 0 and side_after < 0:
        return "positive_to_negative"
    return "crossing"


def point_in_zone(x: float, y: float) -> bool:
    return cv2.pointPolygonTest(polygon_array, (float(x), float(y)), False) >= 0


def analyze_rows(rows: list[list[object]], video_path: Path, csv_stem: str) -> dict[str, object]:
    analysis_output_dir.mkdir(parents=True, exist_ok=True)

    frame_counts: Counter[int] = Counter()
    track_lengths: Counter[int] = Counter()
    first_frame_by_track: dict[int, int] = {}
    last_frame_by_track: dict[int, int] = {}
    track_points: dict[int, list[tuple[int, float, float]]] = {}
    zone_counts: Counter[int] = Counter()
    zone_tracks_by_frame: dict[int, set[int]] = {}

    for row in rows:
        frame = int(row[0])
        track_id = int(row[1])
        center_x = float(row[7])
        center_y = float(row[8])

        frame_counts[frame] += 1
        track_lengths[track_id] += 1
        track_points.setdefault(track_id, []).append((frame, center_x, center_y))

        if track_id not in first_frame_by_track or frame < first_frame_by_track[track_id]:
            first_frame_by_track[track_id] = frame
        if track_id not in last_frame_by_track or frame > last_frame_by_track[track_id]:
            last_frame_by_track[track_id] = frame

        if point_in_zone(center_x, center_y):
            zone_counts[frame] += 1
            zone_tracks_by_frame.setdefault(frame, set()).add(track_id)

    sorted_frames = sorted(frame_counts)
    counts = [frame_counts[frame] for frame in sorted_frames]

    summary_path = analysis_output_dir / f"{csv_stem}_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["total_rows", len(rows)])
        writer.writerow(["total_frames", len(sorted_frames)])
        writer.writerow(["unique_track_ids", len(track_lengths)])
        writer.writerow(["max_vehicles_in_frame", max(counts)])
        writer.writerow(["min_vehicles_in_frame", min(counts)])
        writer.writerow(["avg_vehicles_per_frame", round(sum(counts) / len(counts), 4)])
        writer.writerow(["longest_track_length_frames", max(track_lengths.values())])

    frame_counts_path = analysis_output_dir / f"{csv_stem}_frame_counts.csv"
    with frame_counts_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "vehicle_count"])
        for frame in sorted_frames:
            writer.writerow([frame, frame_counts[frame]])

    track_summary_path = analysis_output_dir / f"{csv_stem}_track_summary.csv"
    with track_summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["track_id", "first_frame", "last_frame", "length_frames"])
        for track_id in sorted(track_lengths):
            writer.writerow(
                [
                    track_id,
                    first_frame_by_track[track_id],
                    last_frame_by_track[track_id],
                    track_lengths[track_id],
                ]
            )

    plt.figure(figsize=(10, 4))
    plt.plot(sorted_frames, counts, marker="o", linewidth=1.8, markersize=3)
    plt.title("Vehicle Count Per Frame")
    plt.xlabel("Frame")
    plt.ylabel("Detected Vehicles")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = analysis_output_dir / f"{csv_stem}_frame_counts.png"
    plt.savefig(plot_path, dpi=160)
    plt.close()

    crossings: list[list[object]] = []
    unique_crossing_tracks: set[int] = set()
    for track_id, points in track_points.items():
        points = sorted(points, key=lambda item: item[0])
        for previous, current in zip(points, points[1:]):
            prev_side = point_side(previous[1], previous[2], line_start, line_end)
            curr_side = point_side(current[1], current[2], line_start, line_end)
            if prev_side == 0 or curr_side == 0 or (prev_side < 0 < curr_side) or (prev_side > 0 > curr_side):
                crossings.append(
                    [
                        track_id,
                        previous[0],
                        current[0],
                        round(previous[1], 3),
                        round(previous[2], 3),
                        round(current[1], 3),
                        round(current[2], 3),
                        line_label(prev_side, curr_side),
                    ]
                )
                unique_crossing_tracks.add(track_id)
                break

    crossings_path = analysis_output_dir / f"{csv_stem}_line_crossings.csv"
    with crossings_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "track_id",
                "frame_before",
                "frame_after",
                "center_x_before",
                "center_y_before",
                "center_x_after",
                "center_y_after",
                "direction",
            ]
        )
        writer.writerows(crossings)

    crossings_summary_path = analysis_output_dir / f"{csv_stem}_line_crossings_summary.csv"
    with crossings_summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["line_start_x", line_start[0]])
        writer.writerow(["line_start_y", line_start[1]])
        writer.writerow(["line_end_x", line_end[0]])
        writer.writerow(["line_end_y", line_end[1]])
        writer.writerow(["unique_crossing_tracks", len(unique_crossing_tracks)])

    queue_summary_path = analysis_output_dir / f"{csv_stem}_queue_zone_summary.csv"
    queue_counts_path = analysis_output_dir / f"{csv_stem}_queue_zone_counts.csv"
    sorted_zone_frames = sorted(zone_counts)
    zone_values = [zone_counts[frame] for frame in sorted_zone_frames] if sorted_zone_frames else []

    with queue_summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["polygon_points", polygon_points])
        writer.writerow(["frames_with_vehicles_in_zone", len(sorted_zone_frames)])
        writer.writerow(["max_vehicles_in_zone", max(zone_values) if zone_values else 0])
        writer.writerow(["min_vehicles_in_zone", min(zone_values) if zone_values else 0])
        writer.writerow(
            [
                "avg_vehicles_in_zone_when_nonzero",
                round(sum(zone_values) / len(zone_values), 4) if zone_values else 0,
            ]
        )

    with queue_counts_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "vehicles_in_zone", "track_ids"])
        for frame in sorted_zone_frames:
            track_ids = sorted(zone_tracks_by_frame.get(frame, set()))
            writer.writerow([frame, zone_counts[frame], " ".join(str(track_id) for track_id in track_ids)])

    if zone_values:
        plt.figure(figsize=(10, 4))
        plt.plot(sorted_zone_frames, zone_values, marker="o", linewidth=1.8, markersize=3, color="darkorange")
        plt.title("Vehicles In Queue Zone Per Frame")
        plt.xlabel("Frame")
        plt.ylabel("Vehicles In Zone")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        queue_plot_path = analysis_output_dir / f"{csv_stem}_queue_zone_counts.png"
        plt.savefig(queue_plot_path, dpi=160)
        plt.close()

    cap = cv2.VideoCapture(str(video_path))
    ret, first_frame = cap.read()
    if ret:
        line_preview_path = analysis_output_dir / f"{csv_stem}_line_preview.png"
        line_frame = first_frame.copy()
        cv2.line(line_frame, line_start, line_end, (0, 0, 255), 3)
        cv2.putText(
            line_frame,
            f"Crossings: {len(unique_crossing_tracks)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(line_preview_path), line_frame)

        queue_preview_path = analysis_output_dir / f"{csv_stem}_queue_zone_preview.png"
        max_zone_frame = max(sorted_zone_frames, key=lambda frame: zone_counts[frame]) if sorted_zone_frames else 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, max_zone_frame - 1)
        ret_queue, queue_frame = cap.read()
        if not ret_queue:
            queue_frame = first_frame.copy()
        cv2.polylines(queue_frame, [polygon_array], isClosed=True, color=(0, 165, 255), thickness=3)
        cv2.putText(
            queue_frame,
            f"Max in zone: {max(zone_values) if zone_values else 0} at frame {max_zone_frame}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(queue_preview_path), queue_frame)
    cap.release()

    return {
        "unique_track_ids": len(track_lengths),
        "avg_vehicles_per_frame": round(sum(counts) / len(counts), 4),
        "line_crossings": len(unique_crossing_tracks),
        "max_queue_zone": max(zone_values) if zone_values else 0,
    }


def main() -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    video_output_dir.mkdir(parents=True, exist_ok=True)
    track_output_dir.mkdir(parents=True, exist_ok=True)
    analysis_output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Infrastructure dataset root not found: {dataset_root}")

    sequence_dirs = numeric_sequence_dirs(dataset_root)
    if not sequence_dirs:
        raise RuntimeError(f"No numeric sequence folders found in {dataset_root}")

    batch_summary_rows: list[list[object]] = []

    for sequence_dir in sequence_dirs:
        sequence_name = sequence_dir.name
        video_path = video_output_dir / f"infrastructure_{sequence_name}.mp4"
        frame_total = build_video_from_frames(sequence_dir, video_path)

        output_dir, rows, tracked_frames = track_video_to_rows(video_path)
        csv_stem = video_path.stem
        csv_path = output_dir / f"{csv_stem}_tracks.csv"
        save_tracking_csv(csv_path, rows)

        metrics = analyze_rows(rows, video_path, csv_stem)

        # Run Phase 1 Extractors
        print(f"Running Phase 1 extractors for {sequence_name}...")
        
        # 1. Speeds
        subprocess.run([
            "python", str(project_root / "scripts" / "estimate_speeds.py"),
            "--tracking-csv", str(csv_path),
            "--output-dir", str(analysis_output_dir),
            "--fps", str(fps)
        ], check=True)

        # 2. Travel Time
        subprocess.run([
            "python", str(project_root / "scripts" / "estimate_travel_time.py"),
            "--tracking-csv", str(csv_path),
            "--output-dir", str(analysis_output_dir),
            "--fps", str(fps),
            "--entry-x1", "100", "--entry-y1", "0", "--entry-x2", "100", "--entry-y2", "639",
            "--exit-x1", "800", "--exit-y1", "0", "--exit-x2", "800", "--exit-y2", "639"
        ], check=True)

        # 3. Flow Rate
        line_crossing_csv = analysis_output_dir / f"{csv_stem}_line_crossings.csv"
        if line_crossing_csv.exists():
            subprocess.run([
                "python", str(project_root / "scripts" / "compute_flow_rate.py"),
                "--crossings-csv", str(line_crossing_csv),
                "--output-dir", str(analysis_output_dir),
                "--fps", str(fps),
                "--window-frames", "100"
            ], check=True)

        batch_summary_rows.append(
            [
                sequence_name,
                frame_total,
                tracked_frames,
                len(rows),
                metrics["unique_track_ids"],
                metrics["avg_vehicles_per_frame"],
                metrics["line_crossings"],
                metrics["max_queue_zone"],
            ]
        )
        print(
            f"Processed sequence {sequence_name}: frames={frame_total}, "
            f"tracks={metrics['unique_track_ids']}, line_crossings={metrics['line_crossings']}"
        )

    batch_summary_path = analysis_output_dir / "infrastructure_batch_summary.csv"
    with batch_summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sequence",
                "video_frames",
                "tracked_frames",
                "tracking_rows",
                "unique_track_ids",
                "avg_vehicles_per_frame",
                "line_crossings",
                "max_queue_zone",
            ]
        )
        writer.writerows(batch_summary_rows)

    print(f"Batch summary saved to: {batch_summary_path}")
    print(f"Processed {len(batch_summary_rows)} infrastructure sequences.")


if __name__ == "__main__":
    main()
