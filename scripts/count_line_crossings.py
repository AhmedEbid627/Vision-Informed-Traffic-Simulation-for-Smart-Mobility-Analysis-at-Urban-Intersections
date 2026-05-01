from __future__ import annotations

import csv
from pathlib import Path

import cv2


project_root = Path(__file__).resolve().parents[1]
tracking_csv_path = project_root / "runs" / "track" / "traffic_vehicle_track_csv" / "infrastructure_3000_tracks.csv"
video_path = project_root / "outputs" / "videos" / "infrastructure_3000.mp4"
output_dir = project_root / "runs" / "track" / "traffic_vehicle_track_csv" / "analysis"

# Edit these two points to move the counting line.
line_start = (700, 0)
line_end = (700, 639)


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


def main() -> None:
    if not tracking_csv_path.exists():
        raise FileNotFoundError(f"Tracking CSV not found: {tracking_csv_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    tracks: dict[int, list[dict[str, float | int]]] = {}
    with tracking_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            track_id = int(row["track_id"])
            tracks.setdefault(track_id, []).append(
                {
                    "frame": int(row["frame"]),
                    "center_x": float(row["center_x"]),
                    "center_y": float(row["center_y"]),
                }
            )

    crossings: list[list[object]] = []
    unique_crossing_tracks: set[int] = set()

    for track_id, points in tracks.items():
        points = sorted(points, key=lambda item: int(item["frame"]))
        for previous, current in zip(points, points[1:]):
            prev_side = point_side(
                float(previous["center_x"]),
                float(previous["center_y"]),
                line_start,
                line_end,
            )
            curr_side = point_side(
                float(current["center_x"]),
                float(current["center_y"]),
                line_start,
                line_end,
            )
            if prev_side == 0 or curr_side == 0 or (prev_side < 0 < curr_side) or (prev_side > 0 > curr_side):
                crossings.append(
                    [
                        track_id,
                        int(previous["frame"]),
                        int(current["frame"]),
                        round(float(previous["center_x"]), 3),
                        round(float(previous["center_y"]), 3),
                        round(float(current["center_x"]), 3),
                        round(float(current["center_y"]), 3),
                        line_label(prev_side, curr_side),
                    ]
                )
                unique_crossing_tracks.add(track_id)
                break

    crossings_path = output_dir / f"{tracking_csv_path.stem}_line_crossings.csv"
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

    summary_path = output_dir / f"{tracking_csv_path.stem}_line_crossings_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["line_start_x", line_start[0]])
        writer.writerow(["line_start_y", line_start[1]])
        writer.writerow(["line_end_x", line_end[0]])
        writer.writerow(["line_end_y", line_end[1]])
        writer.writerow(["unique_crossing_tracks", len(unique_crossing_tracks)])

    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if ret:
        cv2.line(frame, line_start, line_end, (0, 0, 255), 3)
        cv2.putText(
            frame,
            f"Crossings: {len(unique_crossing_tracks)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        preview_path = output_dir / f"{tracking_csv_path.stem}_line_preview.png"
        cv2.imwrite(str(preview_path), frame)
        print(f"Line preview saved to: {preview_path}")

    print(f"Line crossings CSV saved to: {crossings_path}")
    print(f"Line crossings summary saved to: {summary_path}")
    print(f"Unique crossing tracks: {len(unique_crossing_tracks)}")


if __name__ == "__main__":
    main()
