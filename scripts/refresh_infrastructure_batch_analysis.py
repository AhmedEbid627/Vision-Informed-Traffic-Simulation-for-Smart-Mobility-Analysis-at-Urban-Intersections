from __future__ import annotations

import csv
from pathlib import Path

import cv2

import process_infrastructure_batch as pib


project_root = Path(__file__).resolve().parents[1]


def load_tracking_rows(csv_path: Path) -> list[list[object]]:
    rows: list[list[object]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                [
                    int(row["frame"]),
                    int(row["track_id"]),
                    float(row["confidence"]),
                    float(row["x1"]),
                    float(row["y1"]),
                    float(row["x2"]),
                    float(row["y2"]),
                    float(row["center_x"]),
                    float(row["center_y"]),
                    float(row["width"]),
                    float(row["height"]),
                ]
            )
    return rows


def main() -> None:
    pib.analysis_output_dir.mkdir(parents=True, exist_ok=True)

    sequence_dirs = pib.numeric_sequence_dirs(pib.dataset_root)
    if not sequence_dirs:
        raise RuntimeError(f"No numeric sequence folders found in {pib.dataset_root}")

    batch_summary_rows: list[list[object]] = []

    for sequence_dir in sequence_dirs:
        sequence_name = sequence_dir.name
        video_path = pib.video_output_dir / f"infrastructure_{sequence_name}.mp4"
        csv_stem = video_path.stem
        csv_path = pib.track_output_dir / pib.run_name / f"{csv_stem}_tracks.csv"

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        if not csv_path.exists():
            raise FileNotFoundError(f"Tracking CSV not found: {csv_path}")

        rows = load_tracking_rows(csv_path)
        metrics = pib.analyze_rows(rows, video_path, csv_stem)
        tracked_frames = len({int(row[0]) for row in rows})

        cap = cv2.VideoCapture(str(video_path))
        video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        batch_summary_rows.append(
            [
                sequence_name,
                video_frames,
                tracked_frames,
                len(rows),
                metrics["unique_track_ids"],
                metrics["avg_vehicles_per_frame"],
                metrics["line_crossings"],
                metrics["max_queue_zone"],
            ]
        )
        print(
            f"Refreshed sequence {sequence_name}: "
            f"tracks={metrics['unique_track_ids']}, "
            f"line_crossings={metrics['line_crossings']}, "
            f"max_queue_zone={metrics['max_queue_zone']}"
        )

    batch_summary_path = pib.analysis_output_dir / "infrastructure_batch_summary.csv"
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

    print(f"Batch summary refreshed at: {batch_summary_path}")


if __name__ == "__main__":
    main()
