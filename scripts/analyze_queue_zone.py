from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


project_root = Path(__file__).resolve().parents[1]
tracking_csv_path = project_root / "runs" / "track" / "infrastructure_batch" / "infrastructure_batch_track" / "infrastructure_1000_tracks.csv"
video_path = project_root / "outputs" / "videos" / "infrastructure_batch" / "infrastructure_1000.mp4"
output_dir = project_root / "runs" / "track" / "infrastructure_batch" / "analysis"

# Edit this polygon to move the queue analysis zone.
polygon_points = [
    (236, 299),
    (627, 637),
    (506, 639),
    (196, 325),
]

polygon_array = np.array(polygon_points, dtype=np.int32)


def point_in_zone(x: float, y: float) -> bool:
    return cv2.pointPolygonTest(polygon_array, (float(x), float(y)), False) >= 0


def main() -> None:
    if not tracking_csv_path.exists():
        raise FileNotFoundError(f"Tracking CSV not found: {tracking_csv_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    zone_counts: Counter[int] = Counter()
    zone_tracks_by_frame: dict[int, set[int]] = {}

    with tracking_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["frame"])
            track_id = int(row["track_id"])
            center_x = float(row["center_x"])
            center_y = float(row["center_y"])

            if point_in_zone(center_x, center_y):
                zone_counts[frame] += 1
                zone_tracks_by_frame.setdefault(frame, set()).add(track_id)

    if not zone_counts:
        raise RuntimeError("No tracked vehicle centers fell inside the current queue zone.")

    sorted_frames = sorted(zone_counts)
    counts = [zone_counts[frame] for frame in sorted_frames]
    max_zone_frame = max(sorted_frames, key=lambda frame: zone_counts[frame])

    summary_path = output_dir / f"{tracking_csv_path.stem}_queue_zone_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["polygon_points", polygon_points])
        writer.writerow(["frames_with_vehicles_in_zone", len(sorted_frames)])
        writer.writerow(["max_vehicles_in_zone", max(counts)])
        writer.writerow(["min_vehicles_in_zone", min(counts)])
        writer.writerow(["avg_vehicles_in_zone_when_nonzero", round(sum(counts) / len(counts), 4)])

    counts_path = output_dir / f"{tracking_csv_path.stem}_queue_zone_counts.csv"
    with counts_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "vehicles_in_zone", "track_ids"])
        for frame in sorted_frames:
            track_ids = sorted(zone_tracks_by_frame.get(frame, set()))
            writer.writerow([frame, zone_counts[frame], " ".join(str(track_id) for track_id in track_ids)])

    plt.figure(figsize=(10, 4))
    plt.plot(sorted_frames, counts, marker="o", linewidth=1.8, markersize=3, color="darkorange")
    plt.title("Vehicles In Queue Zone Per Frame")
    plt.xlabel("Frame")
    plt.ylabel("Vehicles In Zone")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = output_dir / f"{tracking_csv_path.stem}_queue_zone_counts.png"
    plt.savefig(plot_path, dpi=160)
    plt.close()

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max_zone_frame - 1)
    ret, frame = cap.read()
    cap.release()
    if ret:
        cv2.polylines(frame, [polygon_array], isClosed=True, color=(0, 165, 255), thickness=3)
        cv2.putText(
            frame,
            f"Max in zone: {max(counts)} at frame {max_zone_frame}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )
        preview_path = output_dir / f"{tracking_csv_path.stem}_queue_zone_preview.png"
        cv2.imwrite(str(preview_path), frame)
        print(f"Queue zone preview saved to: {preview_path}")

    print(f"Queue zone summary saved to: {summary_path}")
    print(f"Queue zone counts saved to: {counts_path}")
    print(f"Queue zone plot saved to: {plot_path}")
    print(f"Max vehicles in zone: {max(counts)}")


if __name__ == "__main__":
    main()
