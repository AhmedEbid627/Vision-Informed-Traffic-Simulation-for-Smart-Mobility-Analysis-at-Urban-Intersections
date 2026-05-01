from __future__ import annotations

import argparse
import csv
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


project_root = Path(__file__).resolve().parents[1]
config_dir = project_root / ".ultralytics"
model_path = project_root / "runs" / "detect" / "traffic_vehicle" / "weights" / "best.pt"

fps = 10.0
conf_threshold = 0.25
imgsz = 640
device = "0"
tracker_config = "bytetrack.yaml"
save_dir = project_root / "runs" / "track" / "drone_support_experiment"
trail_length = 45


VALID_SUFFIXES = {".jpg", ".jpeg", ".png"}
MIN_TRACK_LENGTH_FOR_MOVEMENT = 8
MIN_DISPLACEMENT_PIXELS = 140.0
CORE_BOX = (0.20, 0.18, 0.82, 0.86)  # x_min, y_min, x_max, y_max in normalized coordinates
ARM_ZONES = {
    "northwest": [(0.00, 0.00), (0.63, 0.00), (0.45, 0.18), (0.10, 0.20)],
    "northeast": [(0.78, 0.24), (1.00, 0.18), (1.00, 0.62), (0.76, 0.50)],
    "southeast": [(0.62, 0.70), (1.00, 0.52), (1.00, 1.00), (0.74, 1.00)],
    "southwest": [(0.00, 0.58), (0.28, 0.42), (0.55, 0.78), (0.12, 1.00), (0.00, 1.00)],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a drone support experiment for arm-to-arm movement estimation."
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default="3200",
        help="Drone sequence folder name to process, for example 3200 or 1000.",
    )
    return parser.parse_args()


def build_video_from_images(image_dir: Path, video_path: Path, fps_value: float) -> tuple[int, int]:
    image_paths = sorted(
        p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_SUFFIXES
    )
    if not image_paths:
        raise FileNotFoundError(f"No image frames found in {image_dir}")

    first_frame = cv2.imread(str(image_paths[0]))
    if first_frame is None:
        raise ValueError(f"Could not read first frame: {image_paths[0]}")

    height, width = first_frame.shape[:2]
    video_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_value,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {video_path}")

    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Skipping unreadable frame: {image_path}")
            continue
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)

    writer.release()
    return width, height


def scale_polygon(points: list[tuple[float, float]], width: int, height: int) -> list[tuple[float, float]]:
    return [(x * width, y * height) for x, y in points]


def build_scaled_arm_zones(width: int, height: int) -> dict[str, list[tuple[float, float]]]:
    return {name: scale_polygon(points, width, height) for name, points in ARM_ZONES.items()}


def point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    contour = np.array(polygon, dtype=np.float32)
    return cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0


def detect_arm_membership(
    ordered_points: list[tuple[int, float, float]],
    scaled_arm_zones: dict[str, list[tuple[float, float]]],
) -> tuple[str | None, str | None]:
    visited_arms: list[str] = []
    for _, x, y in ordered_points:
        for arm_name, polygon in scaled_arm_zones.items():
            if point_in_polygon(x, y, polygon):
                if not visited_arms or visited_arms[-1] != arm_name:
                    visited_arms.append(arm_name)
                break

    if not visited_arms:
        return None, None

    return visited_arms[0], visited_arms[-1]


def movement_color(movement: str) -> str:
    palette = {
        "southwest->northeast": "#27C47D",
        "northeast->southwest": "#E86F5B",
        "northwest->southeast": "#D7A83D",
        "southeast->northwest": "#3C7DFF",
        "southwest->northwest": "#7C4DFF",
        "southwest->southeast": "#009688",
        "northeast->northwest": "#FF7043",
        "northeast->southeast": "#8BC34A",
        "northwest->southwest": "#C2185B",
        "northwest->northeast": "#455A64",
        "southeast->southwest": "#FFB300",
        "southeast->northeast": "#5E35B1",
    }
    return palette.get(movement, "#444444")


def save_tracking_csv(rows: list[list[object]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
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


def save_movement_summary(
    track_points: dict[int, list[tuple[int, float, float]]],
    frame_width: int,
    frame_height: int,
    output_dir: Path,
    sequence_name: str,
) -> tuple[Counter[str], list[dict[str, object]]]:
    movement_counts: Counter[str] = Counter()
    movement_rows: list[dict[str, object]] = []
    core_x1 = frame_width * CORE_BOX[0]
    core_y1 = frame_height * CORE_BOX[1]
    core_x2 = frame_width * CORE_BOX[2]
    core_y2 = frame_height * CORE_BOX[3]
    scaled_arm_zones = build_scaled_arm_zones(frame_width, frame_height)

    for track_id, points in sorted(track_points.items()):
        ordered = sorted(points, key=lambda item: item[0])
        if len(ordered) < MIN_TRACK_LENGTH_FOR_MOVEMENT:
            continue

        _, start_x, start_y = ordered[0]
        _, end_x, end_y = ordered[-1]
        displacement = math.hypot(end_x - start_x, end_y - start_y)
        passes_core = any(core_x1 <= x <= core_x2 and core_y1 <= y <= core_y2 for _, x, y in ordered)
        if displacement < MIN_DISPLACEMENT_PIXELS or not passes_core:
            continue

        start_arm, end_arm = detect_arm_membership(ordered, scaled_arm_zones)
        if start_arm is None or end_arm is None or start_arm == end_arm:
            continue

        movement = f"{start_arm}->{end_arm}"
        movement_counts[movement] += 1
        movement_rows.append(
            {
                "track_id": track_id,
                "start_arm": start_arm,
                "end_arm": end_arm,
                "movement": movement,
                "track_length_frames": len(ordered),
                "displacement_pixels": round(displacement, 2),
                "start_x": round(start_x, 2),
                "start_y": round(start_y, 2),
                "end_x": round(end_x, 2),
                "end_y": round(end_y, 2),
            }
        )

    detailed_path = output_dir / "drone_movement_tracks.csv"
    with detailed_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "track_id",
                "start_arm",
                "end_arm",
                "movement",
                "track_length_frames",
                "displacement_pixels",
                "start_x",
                "start_y",
                "end_x",
                "end_y",
            ],
        )
        writer.writeheader()
        writer.writerows(movement_rows)

    summary_path = output_dir / "drone_movement_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["movement", "track_count"])
        for movement, count in movement_counts.most_common():
            writer.writerow([movement, count])

    text_path = output_dir / "drone_experiment_notes.txt"
    lines = [
        f"Drone support experiment for sequence {sequence_name}",
        "",
        "Why this matters:",
        "- The drone view covers the full intersection geometry.",
        "- This makes it easier to infer directional movements from trajectories.",
        "- That is especially useful for replacing assumed SUMO turning distributions later.",
        "",
        "Observed movement counts:",
    ]
    if movement_counts:
        lines.extend([f"- {movement}: {count}" for movement, count in movement_counts.most_common()])
    else:
        lines.append("- No movement counts passed the minimum track-length threshold.")
    lines.extend(
        [
            "",
            "Interpretation:",
            "- This drone view is most useful for estimating arm-to-arm movements and turning proportions.",
            "- It should support the infrastructure workflow rather than replace it.",
        ]
    )
    text_path.write_text("\n".join(lines), encoding="utf-8")

    return movement_counts, movement_rows


def save_trajectory_overlay(
    background_frame: Path,
    track_points: dict[int, list[tuple[int, float, float]]],
    frame_width: int,
    frame_height: int,
    output_dir: Path,
) -> None:
    frame = cv2.imread(str(background_frame))
    if frame is None:
        raise ValueError(f"Could not read background frame: {background_frame}")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 7))
    plt.imshow(rgb, alpha=0.65)
    plt.xlim(0, frame_width)
    plt.ylim(frame_height, 0)
    plt.axis("off")
    plt.title("Drone Trajectories by Arm-to-Arm Movement", fontsize=16)

    movement_handles: dict[str, bool] = {}
    core_x1 = frame_width * CORE_BOX[0]
    core_y1 = frame_height * CORE_BOX[1]
    core_x2 = frame_width * CORE_BOX[2]
    core_y2 = frame_height * CORE_BOX[3]
    scaled_arm_zones = build_scaled_arm_zones(frame_width, frame_height)
    plt.gca().add_patch(
        plt.Rectangle(
            (core_x1, core_y1),
            core_x2 - core_x1,
            core_y2 - core_y1,
            fill=False,
            edgecolor="#222222",
            linewidth=1.2,
            linestyle="--",
            alpha=0.7,
        )
    )
    for arm_name, polygon in scaled_arm_zones.items():
        xs = [p[0] for p in polygon] + [polygon[0][0]]
        ys = [p[1] for p in polygon] + [polygon[0][1]]
        plt.plot(xs, ys, color="#666666", linewidth=1.2, linestyle=":", alpha=0.75)
        centroid_x = sum(p[0] for p in polygon) / len(polygon)
        centroid_y = sum(p[1] for p in polygon) / len(polygon)
        plt.text(
            centroid_x,
            centroid_y,
            arm_name,
            fontsize=8,
            color="#333333",
            ha="center",
            va="center",
            bbox={"facecolor": "#FFFFFFCC", "edgecolor": "none", "pad": 1.5},
        )

    for track_id, points in track_points.items():
        ordered = sorted(points, key=lambda item: item[0])
        if len(ordered) < MIN_TRACK_LENGTH_FOR_MOVEMENT:
            continue

        displacement = math.hypot(ordered[-1][1] - ordered[0][1], ordered[-1][2] - ordered[0][2])
        passes_core = any(core_x1 <= x <= core_x2 and core_y1 <= y <= core_y2 for _, x, y in ordered)
        if displacement < MIN_DISPLACEMENT_PIXELS or not passes_core:
            continue

        start_arm, end_arm = detect_arm_membership(ordered, scaled_arm_zones)
        if start_arm is None or end_arm is None or start_arm == end_arm:
            continue

        movement = f"{start_arm}->{end_arm}"
        xs = [item[1] for item in ordered]
        ys = [item[2] for item in ordered]
        color = movement_color(movement)
        label = movement if movement not in movement_handles else None
        plt.plot(xs, ys, color=color, linewidth=2.0, alpha=0.85, label=label)
        plt.scatter(xs[0], ys[0], color=color, s=10, alpha=0.9)
        plt.scatter(xs[-1], ys[-1], color=color, s=18, alpha=0.9, marker="x")
        movement_handles[movement] = True

    if movement_handles:
        plt.legend(loc="upper right", fontsize=8, framealpha=0.95)

    overlay_path = output_dir / "drone_trajectory_overlay.png"
    plt.tight_layout()
    plt.savefig(overlay_path, dpi=180, bbox_inches="tight")
    plt.close()


def save_movement_bar_chart(movement_counts: Counter[str], output_dir: Path) -> None:
    chart_path = output_dir / "drone_movement_counts.png"
    if not movement_counts:
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, "No movement counts available", ha="center", va="center", fontsize=14)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(chart_path, dpi=180)
        plt.close()
        return

    labels = [movement for movement, _ in movement_counts.most_common()]
    values = [count for _, count in movement_counts.most_common()]
    colors = [movement_color(label) for label in labels]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values, color=colors)
    plt.title("Inferred Drone Movement Counts")
    plt.xlabel("Movement Pattern")
    plt.ylabel("Track Count")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(chart_path, dpi=180)
    plt.close()


def draw_polyline(frame: np.ndarray, points: list[tuple[float, float]], color_bgr: tuple[int, int, int]) -> None:
    if len(points) < 2:
        return
    contour = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [contour], isClosed=False, color=color_bgr, thickness=2, lineType=cv2.LINE_AA)


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    hex_value = hex_color.lstrip("#")
    if len(hex_value) != 6:
        return (0, 255, 255)
    r = int(hex_value[0:2], 16)
    g = int(hex_value[2:4], 16)
    b = int(hex_value[4:6], 16)
    return (b, g, r)


def save_tracking_trail_video(
    source_video: Path,
    output_video: Path,
    tracking_rows: list[list[object]],
    track_points: dict[int, list[tuple[int, float, float]]],
    movement_rows: list[dict[str, object]],
    fps_value: float,
) -> None:
    per_frame_rows: dict[int, list[list[object]]] = defaultdict(list)
    for row in tracking_rows:
        per_frame_rows[int(row[0])].append(row)

    movement_by_track: dict[int, str] = {
        int(row["track_id"]): str(row["movement"]) for row in movement_rows
    }
    ordered_track_points: dict[int, list[tuple[int, float, float]]] = {
        track_id: sorted(points, key=lambda item: item[0]) for track_id, points in track_points.items()
    }

    capture = cv2.VideoCapture(str(source_video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source video for trail rendering: {source_video}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = capture.get(cv2.CAP_PROP_FPS)
    if actual_fps <= 0:
        actual_fps = fps_value

    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        actual_fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open output video writer: {output_video}")

    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_index += 1

        for row in per_frame_rows.get(frame_index, []):
            track_id = int(row[1])
            x1, y1, x2, y2 = [int(round(float(value))) for value in row[3:7]]
            center_x = float(row[7])
            center_y = float(row[8])
            movement = movement_by_track.get(track_id, "unclassified")
            color_bgr = hex_to_bgr(movement_color(movement)) if movement != "unclassified" else (0, 255, 255)

            if track_id in ordered_track_points:
                prior_points = [
                    (x, y)
                    for point_frame, x, y in ordered_track_points[track_id]
                    if point_frame <= frame_index
                ]
                if trail_length > 0:
                    prior_points = prior_points[-trail_length:]
                draw_polyline(frame, prior_points, color_bgr)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)
            cv2.circle(frame, (int(round(center_x)), int(round(center_y))), 3, color_bgr, -1)
            label = f"ID {track_id}"
            if movement != "unclassified":
                label = f"{label} | {movement}"
            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color_bgr,
                2,
                cv2.LINE_AA,
            )

        writer.write(frame)

    capture.release()
    writer.release()


def main() -> None:
    args = parse_args()
    config_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    drone_sequence_name = args.sequence
    drone_input_dir = project_root / "Dataset" / "Drone" / drone_sequence_name

    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    if not drone_input_dir.exists():
        raise FileNotFoundError(f"Drone sequence not found: {drone_input_dir}")

    output_dir = save_dir / f"drone_{drone_sequence_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    video_path = output_dir / f"drone_{drone_sequence_name}.mp4"
    frame_width, frame_height = build_video_from_images(drone_input_dir, video_path, fps)

    from ultralytics import YOLO

    model = YOLO(str(model_path))

    rows: list[list[object]] = []
    track_points: dict[int, list[tuple[int, float, float]]] = defaultdict(list)

    results = model.track(
        source=str(video_path),
        conf=conf_threshold,
        imgsz=imgsz,
        device=device,
        tracker=tracker_config,
        project=str(output_dir),
        name="tracking",
        exist_ok=True,
        save=True,
        stream=True,
        verbose=True,
    )

    tracking_output_dir: Path | None = None
    for frame_index, result in enumerate(results, start=1):
        if tracking_output_dir is None:
            tracking_output_dir = Path(result.save_dir)

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
            if track_id >= 0:
                track_points[track_id].append((frame_index, center_x, center_y))

    if tracking_output_dir is None:
        raise RuntimeError("No tracking results were returned from the drone experiment.")

    csv_path = output_dir / f"drone_{drone_sequence_name}_tracks.csv"
    save_tracking_csv(rows, csv_path)

    movement_counts, movement_rows = save_movement_summary(
        track_points,
        frame_width,
        frame_height,
        output_dir,
        drone_sequence_name,
    )
    trail_video_path = output_dir / f"drone_{drone_sequence_name}_tracking_with_trails.mp4"
    save_tracking_trail_video(
        video_path,
        trail_video_path,
        rows,
        track_points,
        movement_rows,
        fps,
    )
    first_frame_path = sorted(
        p for p in drone_input_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_SUFFIXES
    )[0]
    save_trajectory_overlay(first_frame_path, track_points, frame_width, frame_height, output_dir)
    save_movement_bar_chart(movement_counts, output_dir)

    summary_path = output_dir / "drone_experiment_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["sequence", drone_sequence_name])
        writer.writerow(["frames", len({row[0] for row in rows})])
        writer.writerow(["tracking_rows", len(rows)])
        writer.writerow(["unique_tracks", len(track_points)])
        writer.writerow(["tracks_used_for_movement_counts", len(movement_rows)])
        writer.writerow(["distinct_movement_patterns", len(movement_counts)])
        writer.writerow(["top_movement", movement_counts.most_common(1)[0][0] if movement_counts else "none"])

    print(f"Drone experiment video saved to: {video_path}")
    print(f"Tracking CSV saved to: {csv_path}")
    print(f"Movement summary saved to: {output_dir / 'drone_movement_summary.csv'}")
    print(f"Tracking-with-trails video saved to: {trail_video_path}")
    print(f"Trajectory overlay saved to: {output_dir / 'drone_trajectory_overlay.png'}")
    print(f"Movement chart saved to: {output_dir / 'drone_movement_counts.png'}")
    print(f"Experiment notes saved to: {output_dir / 'drone_experiment_notes.txt'}")


if __name__ == "__main__":
    main()
