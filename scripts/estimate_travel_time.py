from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_FPS = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate vehicle travel times between two reference lines."
    )
    parser.add_argument(
        "--tracking-csv",
        type=Path,
        required=True,
        help="Path to a tracking CSV.",
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
        help="Frames per second.",
    )
    # Entry line (Line A).
    parser.add_argument("--entry-x1", type=int, default=400, help="Entry line start X.")
    parser.add_argument("--entry-y1", type=int, default=0, help="Entry line start Y.")
    parser.add_argument("--entry-x2", type=int, default=400, help="Entry line end X.")
    parser.add_argument("--entry-y2", type=int, default=639, help="Entry line end Y.")
    # Exit line (Line B).
    parser.add_argument("--exit-x1", type=int, default=700, help="Exit line start X.")
    parser.add_argument("--exit-y1", type=int, default=0, help="Exit line start Y.")
    parser.add_argument("--exit-x2", type=int, default=700, help="Exit line end X.")
    parser.add_argument("--exit-y2", type=int, default=639, help="Exit line end Y.")
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Optional source video to save a preview image with entry/exit lines drawn.",
    )
    return parser.parse_args()


def point_side(x: float, y: float, lx1: int, ly1: int, lx2: int, ly2: int) -> float:
    return (x - lx1) * (ly2 - ly1) - (y - ly1) * (lx2 - lx1)


def find_crossing_frame(
    points: list[tuple[int, float, float]],
    lx1: int,
    ly1: int,
    lx2: int,
    ly2: int,
) -> int | None:
    """Return the frame number when the track first crosses the given line."""
    for i in range(1, len(points)):
        f_prev, x_prev, y_prev = points[i - 1]
        f_curr, x_curr, y_curr = points[i]
        s_prev = point_side(x_prev, y_prev, lx1, ly1, lx2, ly2)
        s_curr = point_side(x_curr, y_curr, lx1, ly1, lx2, ly2)
        if s_prev == 0 or s_curr == 0 or (s_prev < 0 < s_curr) or (s_prev > 0 > s_curr):
            return f_curr
    return None


def main() -> None:
    args = parse_args()

    if not args.tracking_csv.exists():
        raise FileNotFoundError(f"Tracking CSV not found: {args.tracking_csv}")

    output_dir = args.output_dir or args.tracking_csv.parent.parent / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.tracking_csv.stem

    # Load tracks.
    tracks: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    with args.tracking_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            track_id = int(row["track_id"])
            frame = int(row["frame"])
            cx = float(row["center_x"])
            cy = float(row["center_y"])
            tracks[track_id].append((frame, cx, cy))
    for tid in tracks:
        tracks[tid].sort(key=lambda p: p[0])

    # Find crossing frames for each track.
    travel_rows: list[list[object]] = []
    travel_times: list[float] = []

    for track_id, points in sorted(tracks.items()):
        entry_frame = find_crossing_frame(
            points, args.entry_x1, args.entry_y1, args.entry_x2, args.entry_y2
        )
        exit_frame = find_crossing_frame(
            points, args.exit_x1, args.exit_y1, args.exit_x2, args.exit_y2
        )
        if entry_frame is not None and exit_frame is not None and exit_frame > entry_frame:
            dt_frames = exit_frame - entry_frame
            dt_seconds = dt_frames / args.fps
            travel_rows.append([
                track_id,
                entry_frame,
                exit_frame,
                dt_frames,
                round(dt_seconds, 4),
            ])
            travel_times.append(dt_seconds)

    # Write travel time CSV.
    tt_path = output_dir / f"{stem}_travel_times.csv"
    with tt_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "track_id", "entry_frame", "exit_frame", "travel_frames", "travel_time_seconds",
        ])
        writer.writerows(travel_rows)

    # Write summary.
    summary_path = output_dir / f"{stem}_travel_time_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["entry_line", f"({args.entry_x1},{args.entry_y1})->({args.entry_x2},{args.entry_y2})"])
        writer.writerow(["exit_line", f"({args.exit_x1},{args.exit_y1})->({args.exit_x2},{args.exit_y2})"])
        writer.writerow(["vehicles_matched", len(travel_rows)])
        if travel_times:
            arr = np.array(travel_times)
            writer.writerow(["avg_travel_time_sec", round(float(np.mean(arr)), 4)])
            writer.writerow(["median_travel_time_sec", round(float(np.median(arr)), 4)])
            writer.writerow(["min_travel_time_sec", round(float(np.min(arr)), 4)])
            writer.writerow(["max_travel_time_sec", round(float(np.max(arr)), 4)])
            writer.writerow(["std_travel_time_sec", round(float(np.std(arr)), 4)])
        else:
            writer.writerow(["avg_travel_time_sec", "N/A"])

    # Travel time distribution plot.
    if travel_times:
        plt.figure(figsize=(10, 5))
        plt.hist(travel_times, bins=max(5, len(travel_times) // 3), color="teal", edgecolor="white", alpha=0.85)
        plt.axvline(np.median(travel_times), color="red", linestyle="--", linewidth=1.5, label=f"Median: {np.median(travel_times):.1f}s")
        plt.axvline(np.mean(travel_times), color="orange", linestyle="--", linewidth=1.5, label=f"Mean: {np.mean(travel_times):.1f}s")
        plt.title("Vehicle Travel Time Distribution")
        plt.xlabel("Travel Time (seconds)")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = output_dir / f"{stem}_travel_time_distribution.png"
        plt.savefig(plot_path, dpi=160)
        plt.close()
        print(f"Travel time plot: {plot_path}")

    if args.video is not None:
        if not args.video.exists():
            raise FileNotFoundError(f"Video not found: {args.video}")
        cap = cv2.VideoCapture(str(args.video))
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            cv2.line(frame, (args.entry_x1, args.entry_y1), (args.entry_x2, args.entry_y2), (0, 255, 0), 3)
            cv2.line(frame, (args.exit_x1, args.exit_y1), (args.exit_x2, args.exit_y2), (0, 165, 255), 3)
            cv2.putText(
                frame,
                "Entry",
                (args.entry_x1 + 10, args.entry_y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "Exit",
                (args.exit_x1 + 10, args.exit_y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )
            preview_path = output_dir / f"{stem}_travel_time_lines_preview.png"
            cv2.imwrite(str(preview_path), frame)
            print(f"Travel time line preview: {preview_path}")

    print(f"Travel times CSV: {tt_path}")
    print(f"Summary CSV: {summary_path}")
    print(f"Vehicles with complete entry->exit: {len(travel_rows)}")
    if travel_times:
        print(f"Avg travel time: {np.mean(travel_times):.2f}s")
        print(f"Median travel time: {np.median(travel_times):.2f}s")
    else:
        print("No vehicles crossed both lines in the expected direction.")


if __name__ == "__main__":
    main()
