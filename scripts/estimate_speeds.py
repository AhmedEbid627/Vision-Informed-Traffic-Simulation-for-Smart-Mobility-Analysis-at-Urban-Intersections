from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_FPS = 10.0
SMOOTHING_WINDOW = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate per-vehicle speeds from a tracking CSV."
    )
    parser.add_argument(
        "--tracking-csv",
        type=Path,
        required=True,
        help="Path to a tracking CSV (frame,track_id,...,center_x,center_y,...).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to a sibling 'analysis' folder.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_FPS,
        help="Frames per second of the source video.",
    )
    parser.add_argument(
        "--pixels-per-meter",
        type=float,
        default=0.0,
        help="Optional calibration factor. If >0, speeds are also reported in m/s.",
    )
    return parser.parse_args()


def load_tracks(csv_path: Path) -> dict[int, list[tuple[int, float, float]]]:
    tracks: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            track_id = int(row["track_id"])
            frame = int(row["frame"])
            cx = float(row["center_x"])
            cy = float(row["center_y"])
            tracks[track_id].append((frame, cx, cy))
    for track_id in tracks:
        tracks[track_id].sort(key=lambda p: p[0])
    return dict(tracks)


def compute_speeds(
    tracks: dict[int, list[tuple[int, float, float]]],
    fps: float,
    pixels_per_meter: float,
) -> tuple[list[list[object]], list[list[object]]]:
    per_frame_rows: list[list[object]] = []
    summary_rows: list[list[object]] = []

    for track_id, points in sorted(tracks.items()):
        if len(points) < 2:
            continue

        speeds_px: list[float] = []
        frame_indices: list[int] = []

        for i in range(1, len(points)):
            f_prev, x_prev, y_prev = points[i - 1]
            f_curr, x_curr, y_curr = points[i]
            dt_frames = f_curr - f_prev
            if dt_frames <= 0:
                continue
            dist_px = math.sqrt((x_curr - x_prev) ** 2 + (y_curr - y_prev) ** 2)
            speed_px_per_frame = dist_px / dt_frames
            speeds_px.append(speed_px_per_frame)
            frame_indices.append(f_curr)

        if not speeds_px:
            continue

        # Simple rolling average smoothing.
        smoothed: list[float] = []
        for i in range(len(speeds_px)):
            window_start = max(0, i - SMOOTHING_WINDOW // 2)
            window_end = min(len(speeds_px), i + SMOOTHING_WINDOW // 2 + 1)
            smoothed.append(sum(speeds_px[window_start:window_end]) / (window_end - window_start))

        for i, (frame, raw, sm) in enumerate(zip(frame_indices, speeds_px, smoothed)):
            speed_px_s = raw * fps
            speed_sm_px_s = sm * fps
            row: list[object] = [
                track_id,
                frame,
                round(raw, 4),
                round(sm, 4),
                round(speed_px_s, 4),
                round(speed_sm_px_s, 4),
            ]
            if pixels_per_meter > 0:
                row.append(round(speed_px_s / pixels_per_meter, 4))
                row.append(round(speed_sm_px_s / pixels_per_meter, 4))
            per_frame_rows.append(row)

        arr = np.array(speeds_px) * fps
        summary_rows.append([
            track_id,
            len(points),
            len(speeds_px),
            round(float(np.mean(arr)), 4),
            round(float(np.median(arr)), 4),
            round(float(np.max(arr)), 4),
            round(float(np.std(arr)), 4),
            round(float(np.percentile(arr, 25)), 4),
            round(float(np.percentile(arr, 75)), 4),
        ])

    return per_frame_rows, summary_rows


def main() -> None:
    args = parse_args()

    if not args.tracking_csv.exists():
        raise FileNotFoundError(f"Tracking CSV not found: {args.tracking_csv}")

    output_dir = args.output_dir or args.tracking_csv.parent.parent / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.tracking_csv.stem

    tracks = load_tracks(args.tracking_csv)
    per_frame_rows, summary_rows = compute_speeds(tracks, args.fps, args.pixels_per_meter)

    if not per_frame_rows:
        raise RuntimeError("No speed data could be computed from the tracking CSV.")

    # Write per-frame speeds CSV.
    speed_header = [
        "track_id", "frame",
        "speed_px_per_frame", "speed_smoothed_px_per_frame",
        "speed_px_per_sec", "speed_smoothed_px_per_sec",
    ]
    if args.pixels_per_meter > 0:
        speed_header.extend(["speed_m_per_sec", "speed_smoothed_m_per_sec"])

    speeds_path = output_dir / f"{stem}_speeds.csv"
    with speeds_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(speed_header)
        writer.writerows(per_frame_rows)

    # Write per-track speed summary.
    summary_path = output_dir / f"{stem}_speed_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "track_id", "total_detections", "speed_samples",
            "avg_speed_px_s", "median_speed_px_s", "max_speed_px_s",
            "std_speed_px_s", "p25_speed_px_s", "p75_speed_px_s",
        ])
        writer.writerows(summary_rows)

    # Speed distribution histogram.
    all_speeds = [float(row[4]) for row in per_frame_rows]  # speed_px_per_sec
    plt.figure(figsize=(10, 5))
    plt.hist(all_speeds, bins=50, color="steelblue", edgecolor="white", alpha=0.85)
    plt.axvline(np.median(all_speeds), color="red", linestyle="--", linewidth=1.5, label=f"Median: {np.median(all_speeds):.1f} px/s")
    plt.axvline(np.mean(all_speeds), color="orange", linestyle="--", linewidth=1.5, label=f"Mean: {np.mean(all_speeds):.1f} px/s")
    plt.title("Vehicle Speed Distribution")
    plt.xlabel("Speed (pixels/second)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    hist_path = output_dir / f"{stem}_speed_distribution.png"
    plt.savefig(hist_path, dpi=160)
    plt.close()

    # Average speed per track bar chart.
    if summary_rows:
        track_ids = [str(row[0]) for row in summary_rows]
        avg_speeds = [float(row[3]) for row in summary_rows]
        plt.figure(figsize=(max(8, len(track_ids) * 0.4), 5))
        plt.bar(track_ids, avg_speeds, color="darkorange", edgecolor="white")
        plt.title("Average Speed Per Vehicle Track")
        plt.xlabel("Track ID")
        plt.ylabel("Avg Speed (px/s)")
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        bar_path = output_dir / f"{stem}_speed_per_track.png"
        plt.savefig(bar_path, dpi=160)
        plt.close()

    print(f"Per-frame speeds CSV: {speeds_path}")
    print(f"Speed summary CSV: {summary_path}")
    print(f"Speed distribution plot: {hist_path}")
    print(f"Tracks analyzed: {len(summary_rows)}")
    print(f"Median speed: {np.median(all_speeds):.2f} px/s")
    print(f"Mean speed: {np.mean(all_speeds):.2f} px/s")


if __name__ == "__main__":
    main()
