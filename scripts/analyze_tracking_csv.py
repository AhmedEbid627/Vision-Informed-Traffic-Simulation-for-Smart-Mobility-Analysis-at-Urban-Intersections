from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


project_root = Path(__file__).resolve().parents[1]
tracking_csv_path = project_root / "runs" / "track" / "traffic_vehicle_track_csv" / "infrastructure_3000_tracks.csv"
output_dir = project_root / "runs" / "track" / "traffic_vehicle_track_csv" / "analysis"


def main() -> None:
    if not tracking_csv_path.exists():
        raise FileNotFoundError(f"Tracking CSV not found: {tracking_csv_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    with tracking_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise RuntimeError("Tracking CSV is empty.")

    frame_counts: Counter[int] = Counter()
    track_lengths: Counter[int] = Counter()
    first_frame_by_track: dict[int, int] = {}
    last_frame_by_track: dict[int, int] = {}

    for row in rows:
        frame = int(row["frame"])
        track_id = int(row["track_id"])
        frame_counts[frame] += 1
        track_lengths[track_id] += 1

        if track_id not in first_frame_by_track or frame < first_frame_by_track[track_id]:
            first_frame_by_track[track_id] = frame
        if track_id not in last_frame_by_track or frame > last_frame_by_track[track_id]:
            last_frame_by_track[track_id] = frame

    sorted_frames = sorted(frame_counts)
    counts = [frame_counts[frame] for frame in sorted_frames]

    summary_path = output_dir / f"{tracking_csv_path.stem}_summary.csv"
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

    frame_counts_path = output_dir / f"{tracking_csv_path.stem}_frame_counts.csv"
    with frame_counts_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "vehicle_count"])
        for frame in sorted_frames:
            writer.writerow([frame, frame_counts[frame]])

    track_summary_path = output_dir / f"{tracking_csv_path.stem}_track_summary.csv"
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
    plot_path = output_dir / f"{tracking_csv_path.stem}_frame_counts.png"
    plt.savefig(plot_path, dpi=160)
    plt.close()

    print(f"Summary CSV saved to: {summary_path}")
    print(f"Frame counts CSV saved to: {frame_counts_path}")
    print(f"Track summary CSV saved to: {track_summary_path}")
    print(f"Count plot saved to: {plot_path}")
    print(f"Unique track IDs: {len(track_lengths)}")
    print(f"Average vehicles per frame: {sum(counts) / len(counts):.3f}")


if __name__ == "__main__":
    main()
