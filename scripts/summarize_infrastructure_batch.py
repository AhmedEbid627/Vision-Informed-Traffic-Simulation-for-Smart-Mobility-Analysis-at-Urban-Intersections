from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


project_root = Path(__file__).resolve().parents[1]
analysis_dir = project_root / "runs" / "track" / "infrastructure_batch" / "analysis"
batch_summary_path = analysis_dir / "infrastructure_batch_summary.csv"
report_dir = analysis_dir / "report"


def load_metric_map(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    return {key: value for key, value in rows[1:]}


def main() -> None:
    if not batch_summary_path.exists():
        raise FileNotFoundError(f"Batch summary not found: {batch_summary_path}")

    report_dir.mkdir(parents=True, exist_ok=True)

    with batch_summary_path.open("r", encoding="utf-8", newline="") as f:
        batch_rows = list(csv.DictReader(f))

    enriched_rows: list[dict[str, object]] = []
    for row in batch_rows:
        sequence = row["sequence"]
        base = f"infrastructure_{sequence}"

        summary_metrics = load_metric_map(analysis_dir / f"{base}_summary.csv")
        line_metrics = load_metric_map(analysis_dir / f"{base}_line_crossings_summary.csv")
        queue_metrics = load_metric_map(analysis_dir / f"{base}_queue_zone_summary.csv")

        enriched_rows.append(
            {
                "sequence": sequence,
                "video_frames": int(row["video_frames"]),
                "tracked_frames": int(row["tracked_frames"]),
                "tracking_rows": int(row["tracking_rows"]),
                "unique_track_ids": int(row["unique_track_ids"]),
                "avg_vehicles_per_frame": float(row["avg_vehicles_per_frame"]),
                "line_crossings": int(row["line_crossings"]),
                "max_queue_zone": int(row["max_queue_zone"]),
                "longest_track_length_frames": int(summary_metrics["longest_track_length_frames"]),
                "frames_with_vehicles_in_zone": int(queue_metrics["frames_with_vehicles_in_zone"]),
                "avg_vehicles_in_zone_when_nonzero": float(queue_metrics["avg_vehicles_in_zone_when_nonzero"]),
                "line_start_x": int(line_metrics["line_start_x"]),
                "line_start_y": int(line_metrics["line_start_y"]),
                "line_end_x": int(line_metrics["line_end_x"]),
                "line_end_y": int(line_metrics["line_end_y"]),
            }
        )

    ranked_by_crossings = sorted(enriched_rows, key=lambda row: int(row["line_crossings"]), reverse=True)
    ranked_by_density = sorted(enriched_rows, key=lambda row: float(row["avg_vehicles_per_frame"]), reverse=True)
    ranked_by_queue = sorted(enriched_rows, key=lambda row: int(row["max_queue_zone"]), reverse=True)

    report_csv_path = report_dir / "infrastructure_report.csv"
    with report_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(enriched_rows[0].keys()))
        writer.writeheader()
        writer.writerows(enriched_rows)

    narrative_path = report_dir / "infrastructure_report_summary.txt"
    with narrative_path.open("w", encoding="utf-8") as f:
        f.write("Infrastructure Batch Summary\n")
        f.write("============================\n\n")
        f.write(f"Sequences processed: {len(enriched_rows)}\n")
        f.write(f"Top sequence by line crossings: {ranked_by_crossings[0]['sequence']} ({ranked_by_crossings[0]['line_crossings']})\n")
        f.write(f"Top sequence by avg vehicles/frame: {ranked_by_density[0]['sequence']} ({ranked_by_density[0]['avg_vehicles_per_frame']:.3f})\n")
        f.write(f"Top sequence by queue-zone max: {ranked_by_queue[0]['sequence']} ({ranked_by_queue[0]['max_queue_zone']})\n\n")
        f.write("Ranking by line crossings:\n")
        for row in ranked_by_crossings:
            f.write(f"- {row['sequence']}: {row['line_crossings']} crossings\n")
        f.write("\nRanking by avg vehicles/frame:\n")
        for row in ranked_by_density:
            f.write(f"- {row['sequence']}: {row['avg_vehicles_per_frame']:.3f}\n")
        f.write("\nRanking by queue-zone max:\n")
        for row in ranked_by_queue:
            f.write(f"- {row['sequence']}: {row['max_queue_zone']}\n")

    sequences = [str(row["sequence"]) for row in enriched_rows]
    crossings = [int(row["line_crossings"]) for row in enriched_rows]
    avg_density = [float(row["avg_vehicles_per_frame"]) for row in enriched_rows]
    queue_max = [int(row["max_queue_zone"]) for row in enriched_rows]

    plt.figure(figsize=(9, 4))
    plt.bar(sequences, crossings, color="firebrick")
    plt.title("Line Crossings Per Sequence")
    plt.xlabel("Sequence")
    plt.ylabel("Unique Crossing Tracks")
    plt.tight_layout()
    crossings_plot_path = report_dir / "line_crossings_by_sequence.png"
    plt.savefig(crossings_plot_path, dpi=160)
    plt.close()

    plt.figure(figsize=(9, 4))
    plt.bar(sequences, avg_density, color="steelblue")
    plt.title("Average Vehicles Per Frame")
    plt.xlabel("Sequence")
    plt.ylabel("Avg Vehicles / Frame")
    plt.tight_layout()
    density_plot_path = report_dir / "avg_vehicles_per_frame_by_sequence.png"
    plt.savefig(density_plot_path, dpi=160)
    plt.close()

    plt.figure(figsize=(9, 4))
    plt.bar(sequences, queue_max, color="darkorange")
    plt.title("Max Queue-Zone Occupancy")
    plt.xlabel("Sequence")
    plt.ylabel("Max Vehicles In Zone")
    plt.tight_layout()
    queue_plot_path = report_dir / "max_queue_zone_by_sequence.png"
    plt.savefig(queue_plot_path, dpi=160)
    plt.close()

    print(f"Report CSV saved to: {report_csv_path}")
    print(f"Summary text saved to: {narrative_path}")
    print(f"Line crossings plot saved to: {crossings_plot_path}")
    print(f"Avg vehicles plot saved to: {density_plot_path}")
    print(f"Queue-zone plot saved to: {queue_plot_path}")
    print(f"Top crossing sequence: {ranked_by_crossings[0]['sequence']}")
    print(f"Top density sequence: {ranked_by_density[0]['sequence']}")


if __name__ == "__main__":
    main()
